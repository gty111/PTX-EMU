// GEMM by tensor core 
#include <mma.h>
#include <vector>
#include <cstdio>
#include <cassert>
#include <functional>
#include <cuda_runtime.h>
using namespace nvcuda; 
#define PAD(X,Y) (X % Y ? (X/Y+1)*Y : X)

#define BLOCK_DIM_DEFAULT 512
#define WARP_SIZE 32
#define TIMES 5

template <typename TIN,typename TOUT,int M_TILE,int N_TILE,int K_TILE,
      int M_PAD,int N_PAD,int K_PAD>
__global__ void bmma_kernel(TIN *a, TIN *b, TOUT *c) {
   const int nwarp = BLOCK_DIM_DEFAULT/WARP_SIZE;
   const int C_TILE_SIZE = M_TILE * N_TILE;
   __shared__ TOUT shm[M_TILE][nwarp*N_TILE];
   const int ndim = N_PAD / N_TILE;
   const int kdim = K_PAD / K_TILE;
   const int warpidx = threadIdx.x/WARP_SIZE;
   const int nidx = blockIdx.x%ndim;
   const int midx = blockIdx.x/ndim;
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, M_TILE, N_TILE, K_TILE, TIN, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, M_TILE, N_TILE, K_TILE, TIN, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, M_TILE, N_TILE, K_TILE, TOUT> c_frag;
   
   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);
   
   const int base = nidx*N_TILE + midx*ndim*C_TILE_SIZE;
   TOUT *c_unique = c + base;
   
   for(int kidx=0;kidx<kdim;kidx++){
      if(kidx % nwarp != warpidx)continue;
      // Load the inputs
      TIN *a_unique = a + kidx*K_TILE + midx*M_TILE*kdim*K_TILE;
      TIN *b_unique = b + nidx*N_TILE + kidx*K_TILE*ndim*N_TILE;
      
      wmma::load_matrix_sync(a_frag, a_unique, K_PAD);
      wmma::load_matrix_sync(b_frag, b_unique, N_PAD);
      
      // Perform the matrix multiplication
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
   }
   // Store the output
   wmma::store_matrix_sync(&shm[0][warpidx*N_TILE],
       c_frag, nwarp*N_TILE, wmma::mem_row_major);
   __syncthreads();
   for(int i=warpidx;i<C_TILE_SIZE;i+=nwarp){
         c_unique[i/N_TILE*ndim*N_TILE+i%N_TILE] = 0;
      for(int j=0;j<nwarp;j++){
         c_unique[i/N_TILE*ndim*N_TILE+i%N_TILE] += 
            shm[i/N_TILE][i%N_TILE+j*N_TILE];
      }
   }
}      

//unify memory
template <typename T> struct cuda_data {
  T *data;

  cuda_data(size_t n) {
    cudaMallocManaged(&data, sizeof(T) * n);
    //init to zero
    for(long i=0;i<n;i++){
      data[i] = 0;
    }
  }
  ~cuda_data() { cudaFree(data); }
};

enum DIR {ARR2CUARR,CUARR2ARR};

template <typename TARR,typename TCUARR,DIR dir>
void copy(int ARR_M,int ARR_N,TARR *arr,
      int CUARR_M,int CUARR_N,cuda_data<TCUARR> &cuarr){
   assert(CUARR_M>=ARR_M && CUARR_N>=ARR_N);
   if(dir==ARR2CUARR){
      for(int i=0;i<ARR_M;i++)
      for(int j=0;j<ARR_N;j++){
         cuarr.data[i*CUARR_N+j] = arr[i*ARR_N+j];
      }
   }else if(dir==CUARR2ARR){   
      for(int i=0;i<ARR_M;i++){
         for(int j=0;j<ARR_N;j++){
            arr[i*ARR_N+j] = cuarr.data[i*CUARR_N+j];
         }
      }
   }else assert(0);
}

void Timer(const char *tag, const std::function<void()> &kernel,
               int test_time = TIMES) {
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

   cuda_data<TGEMMIN> a(M_PAD*K_PAD),b(K_PAD*N_PAD);
   cuda_data<TGEMMOUT> c(M_PAD*N_PAD);

   //init a b
   copy<TIN,TGEMMIN,ARR2CUARR>(M,K,a_in,M_PAD,K_PAD,a);
   copy<TIN,TGEMMIN,ARR2CUARR>(K,N,b_in,K_PAD,N_PAD,b);

   int GRID_DIM,BLOCK_DIM;
   GRID_DIM = (M_PAD/M_TILE) * (N_PAD/N_TILE);
   BLOCK_DIM = BLOCK_DIM_DEFAULT;
   printf("GRID_DIM:%d BLOCK_DIM:%d\n",GRID_DIM,BLOCK_DIM);

   Timer("gemm_bmma", [&]{
   bmma_kernel<TGEMMIN,TGEMMOUT,M_TILE,N_TILE,K_TILE,M_PAD,N_PAD,K_PAD>
      <<<GRID_DIM,BLOCK_DIM>>>(a.data,b.data,c.data);});
   
   cudaDeviceSynchronize();//sync for unify memory

   copy<TOUT,TGEMMOUT,CUARR2ARR>(M,N,c_out,M_PAD,N_PAD,c);
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

   GEMM<TIN,TOUT,M,N,K,half,float,16,16,16>(a,b,c);

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
