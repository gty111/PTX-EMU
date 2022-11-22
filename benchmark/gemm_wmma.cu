// GEMM by tensor core 
#include <mma.h>
#include <vector>
#include <cstdio>
#include <cassert>
#include <functional>
#include <cuda_runtime.h>
using namespace nvcuda; 
#define PAD(X,Y) (X % Y ? (X/Y+1)*Y : X)

#define BLOCK_DIM_DEFAULT 1024
#define WARP_SIZE 32

template <typename TIN,typename TOUT,int M_TILE,int N_TILE,int K_TILE,
      int M_PAD,int N_PAD,int K_PAD>
__global__ void wmma_kernel(TIN *a, TIN *b, TOUT *c) {
   int idx,midx,nidx,ndim,kdim;
   ndim = N_PAD / N_TILE;
   kdim = K_PAD / K_TILE;
   idx = (blockIdx.x*blockDim.x+threadIdx.x)/WARP_SIZE;
   nidx = idx%ndim;
   midx = idx/ndim;
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, M_TILE, N_TILE, K_TILE, TIN, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, M_TILE, N_TILE, K_TILE, TIN, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, M_TILE, N_TILE, K_TILE, TOUT> c_frag;
   
   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);
   
   TOUT *c_unique = c + nidx*N_TILE + midx*M_TILE*ndim*N_TILE;
   
   for(int kidx=0;kidx<kdim;kidx++){

      // Load the inputs
      TIN *a_unique = a + kidx*K_TILE + midx*M_TILE*kdim*K_TILE;
      TIN *b_unique = b + nidx*N_TILE + kidx*K_TILE*ndim*N_TILE;
      
      wmma::load_matrix_sync(a_frag, a_unique, K_PAD);
      wmma::load_matrix_sync(b_frag, b_unique, N_PAD);
      
      // Perform the matrix multiplication
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
   }
   // Store the output
   wmma::store_matrix_sync(c_unique, c_frag, N_PAD, wmma::mem_row_major);
}      

template <typename T> struct cuda_vector {
  T *device_data;
  std::vector<T> host_data;
  cuda_vector(size_t n) : host_data(n,0) {
    cudaMalloc(&device_data, sizeof(T) * n);
  }
  ~cuda_vector() { cudaFree(device_data); }
  void sync_device() {
    cudaMemcpy(device_data, host_data.data(), sizeof(T) * host_data.size(),
               cudaMemcpyHostToDevice);
  }
  void sync_host() {
    cudaMemcpy(host_data.data(), device_data, sizeof(T) * host_data.size(),
               cudaMemcpyDeviceToHost);
  }
  void print(int dim) {
   printf("---------------\n");
   for(int i=0;i<host_data.size()/dim;i++){
      for(int j=0;j<dim;j++){
         std::printf("%3.0f",(float)host_data[i*dim+j]);
         if(j==dim-1) std::printf("\n");
         else std::printf(" ");
      }
   }
  }
};

enum DIR {ARR2VECTOR,VECTOR2ARR};

template <typename TARR,typename TVEC,
         int ARR_M,int ARR_N,int VEC_M,int VEC_N,
         DIR dir>
void gemm_copy(TARR *arr,cuda_vector<TVEC> &vec){
   assert(VEC_M>=ARR_M && VEC_N>=ARR_N);
   if(dir==ARR2VECTOR){
      for(int i=0;i<ARR_M;i++)
      for(int j=0;j<ARR_N;j++){
         vec.host_data[i*VEC_N+j] = arr[i*ARR_N+j];
      }
   }else if(dir==VECTOR2ARR){   
      for(int i=0;i<ARR_M;i++){
         for(int j=0;j<ARR_N;j++){
            arr[i*ARR_N+j] = vec.host_data[i*VEC_N+j];
         }
      }
   }else assert(0);
}

void Timer(const char *tag, const std::function<void()> &kernel,
               int test_time = 1) {
  float min_time = 9e99;
  for (int i = 0; i < test_time; ++i) {
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg);
    kernel();
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, beg, end);
    min_time = std::min(min_time, elapsed_time);
    std::printf("[%s] iter %d: %f ms elapsed, %f ms min.\n", tag, i,
                elapsed_time, min_time);
  }
}

template <typename TIN, typename TOUT,
         int M,int N,int K,
         typename TGEMMIN=half, typename TGEMMOUT=float,
         int M_TILE=16,int N_TILE=16,int K_TILE=16>
void GEMM(TIN *a_in,TIN *b_in,TOUT *c_out){
   assert(M!=0 && N!=0 && K!=0);
   
   const int M_PAD = PAD(M,M_TILE) ;
   const int N_PAD = PAD(N,N_TILE) ;
   const int K_PAD = PAD(K,K_TILE) ;

   cuda_vector<TGEMMIN> a(M_PAD*K_PAD),b(K_PAD*N_PAD);
   cuda_vector<TGEMMOUT> c(M_PAD*N_PAD);

   //init a b
   gemm_copy<TIN,TGEMMIN,M,K,M_PAD,K_PAD,ARR2VECTOR>(a_in,a);
   gemm_copy<TIN,TGEMMIN,K,N,K_PAD,N_PAD,ARR2VECTOR>(b_in,b);

   #ifdef DEBUG
   a.print(K_PAD);
   b.print(N_PAD);
   #endif
   // sync to device
   a.sync_device();
   b.sync_device();

   int GRID_DIM,BLOCK_DIM,nwarp;
   nwarp = (M_PAD/M_TILE) * (N_PAD/N_TILE);
   if(nwarp*WARP_SIZE < BLOCK_DIM_DEFAULT){
      GRID_DIM = 1;
      BLOCK_DIM = nwarp*WARP_SIZE;
   }else{
      GRID_DIM = (nwarp*WARP_SIZE)%BLOCK_DIM_DEFAULT ? 
         nwarp*WARP_SIZE/BLOCK_DIM_DEFAULT+1 : nwarp*WARP_SIZE/BLOCK_DIM_DEFAULT ;
      BLOCK_DIM = BLOCK_DIM_DEFAULT;
   }
   printf("GRID_DIM:%d BLOCK_DIM:%d\n",GRID_DIM,BLOCK_DIM);
   Timer("gemm_wmma", [&]{
   wmma_kernel<TGEMMIN,TGEMMOUT,M_TILE,N_TILE,K_TILE,M_PAD,N_PAD,K_PAD>
      <<<GRID_DIM,BLOCK_DIM>>>(a.device_data,b.device_data,c.device_data);});

   c.sync_host();
   #ifdef DEBUG
   c.print(N_PAD);
   #endif

   gemm_copy<TOUT,TGEMMOUT,M,N,M_PAD,N_PAD,VECTOR2ARR>(c_out,c);
}

template<typename TIN,typename TOUT,
      int M,int N,int K>
bool valid(TIN *a,TIN *b,TOUT *c,double threshold=1e-3){
   for(int i=0;i<M;i++)
   for(int j=0;j<N;j++){
      TOUT val = 0;
      for(int p=0;p<K;p++){
         val += a[i*K+p] * b[p*N+j];
      }
      TOUT abs = val>c[i*N+j] ? val-c[i*N+j] : c[i*N+j] - val;
      if(abs/val>threshold){
         std::printf("at %d %d ",i,j);
         std::printf("expect:%f got:%f relative error:%f%%(>%f%%)\n",
            (float)val,(float)c[i*N+j],(float)abs/val*100,threshold*100);
         return 0;
      }
   }
   return 1;
}

template<typename T,int size>
void initVal(T* arr){
   for(int i=0;i<size;i++){
      arr[i] = 1.0 * (rand()%1024) / (rand()%1024+1);
   }
}

template<typename TIN,typename TOUT,
      int M,int N,int K>
void benchmark(bool ifcheck){
   printf("M:%d N:%d K:%d\n",M,N,K);

   srand(time(NULL));

   TIN *a,*b;
   TOUT *c;

   a = (TIN*) malloc(M*K*sizeof(TIN));
   b = (TIN*) malloc(K*N*sizeof(TIN));
   c = (TOUT*)malloc(M*N*sizeof(TOUT));

   initVal<TIN,M*K>(a);
   initVal<TIN,K*N>(b);

   GEMM<TIN,TOUT,M,N,K,double,double,8,8,4>(a,b,c);

   if(ifcheck && valid<TIN,TOUT,M,N,K>(a,b,c,1e-3)){
      std::printf("check pass\n");
   }else if(ifcheck){
      std::printf("check fail\n");
   }else{
      std::printf("skip check\n");
   }
   
}
int main(){
   benchmark<float,float,128,128,128>(1);
}
