#include <cassert>
#include <cstdio>
#include <functional>
#include <vector>

#define PAD(X,Y) (X % Y ? (X/Y+1)*Y : X)

#define BLOCK_DIM_DEFAULT 32
#define K_DEFAULT 129
#define TEST_TIME 1

template <typename TIN,typename TOUT,
      int M_TILE,int N_TILE>
__global__ void gemm_kernel(TIN *a, TIN *b, TOUT *c,
      int M_PAD,int N_PAD,int K_PAD) {
    
    // K_DEFAULT equals K_PAD
    __shared__ TIN a_tile[M_TILE][K_DEFAULT];
    __shared__ TIN b_tile[K_DEFAULT][N_TILE];

    const int ndim = N_PAD/N_TILE;
    const int midx = blockIdx.x / ndim;
    const int nidx = blockIdx.x % ndim;

    //copy a_tile
    for(int i=threadIdx.x;i<M_TILE*K_PAD;i+=blockDim.x){
        a_tile[i/K_PAD][i%K_PAD] = a[(i/K_PAD+midx*M_TILE)*K_PAD+i%K_PAD];
    }

    //copy b_tile
    for(int i=threadIdx.x;i<K_PAD*N_TILE;i+=blockDim.x){
        b_tile[i/N_TILE][i%N_TILE] = b[i/N_TILE*N_PAD+i%N_TILE+nidx*N_TILE];
    }
    
    __syncthreads();
    //cal
    TOUT acc;
    const int base = midx*N_PAD*M_TILE + nidx*N_TILE;
    for(int i=threadIdx.x;i<M_TILE*N_TILE;i+=blockDim.x){
        int row = i/N_TILE;
        int col = i%N_TILE;
        acc = 0;
        for(int j=0;j<K_PAD;j++){
            acc += a_tile[row][j]*b_tile[j][col];
        }
        c[row*N_PAD+col+base] = acc;
    }

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
         std::printf("%4.0f",(float)host_data[i*dim+j]);
         if(j==dim-1) std::printf("\n");
         else std::printf(" ");
      }
   }
  }
};


enum DIR {ARR2VECTOR,VECTOR2ARR};

template <typename TARR,typename TVEC,DIR dir>
void gemm_copy(int ARR_M,int ARR_N,TARR *arr,
      int VEC_M,int VEC_N,cuda_vector<TVEC> &vec){
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
               int test_time = TEST_TIME) {
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


template <typename TIN, typename TOUT,int M_TILE,int N_TILE>
void simple_GEMM(int M,int N,int K,TIN *a_in,TIN *b_in,TOUT *c_out){
   assert(M!=0 && N!=0 && K!=0);
   
   const int M_PAD = PAD(M,M_TILE) ;
   const int N_PAD = PAD(N,N_TILE) ;
   const int K_PAD = K ;

   assert(K_PAD==K_DEFAULT);

   cuda_vector<TIN> a(M_PAD*K_PAD),b(K_PAD*N_PAD);
   cuda_vector<TOUT> c(M_PAD*N_PAD);

   //init a b
   gemm_copy<TIN,TIN,ARR2VECTOR>(M,K,a_in,M_PAD,K_PAD,a);
   gemm_copy<TIN,TIN,ARR2VECTOR>(K,N,b_in,K_PAD,N_PAD,b);
   
   // sync to device
   a.sync_device();
   b.sync_device();

   int GRID_DIM,BLOCK_DIM;
   BLOCK_DIM = BLOCK_DIM_DEFAULT;
   GRID_DIM = (M_PAD/M_TILE) * (N_PAD/N_TILE);
   printf("GRID_DIM:%d BLOCK_DIM:%d\n",GRID_DIM,BLOCK_DIM);
   Timer("simpleGEMM", [&]{
   gemm_kernel<TIN,TOUT,M_TILE,N_TILE>
      <<<GRID_DIM,BLOCK_DIM>>>(a.device_data,b.device_data,c.device_data,
            M_PAD,N_PAD,K_PAD);});

   c.sync_host();

   gemm_copy<TOUT,TOUT,VECTOR2ARR>(M,N,c_out,M_PAD,N_PAD,c);
}


template<typename TIN,typename TOUT>
bool valid(int M,int N,int K,TIN *a,TIN *b,TOUT *c,double threshold=1e-3){
   for(int i=0;i<M;i++)
   for(int j=0;j<N;j++){
      TOUT val = 0;
      for(int p=0;p<K;p++){
         val += a[i*K+p] * b[p*N+j];
      }
      TOUT abs = val>c[i*N+j] ? val-c[i*N+j] : c[i*N+j] - val;
      if(abs/val>threshold){
         printf("at %d %d ",i,j);
         printf("expect:%f got:%f relative error:%f%%(>%f%%)\n",
            (float)val,(float)c[i*N+j],(float)abs/val*100,threshold*100);
         return 0;
      }
   }
   return 1;
}

template<typename T>
void initVal(int size,T* arr){
   for(int i=0;i<size;i++){
      arr[i] = 1.0 * (rand()%1024) / (rand()%1024+1);
   }
}

template<typename TIN,typename TOUT>
bool benchmark(int M,int N,int K){
   printf("----------------\n");
   printf("M:%d N:%d K:%d\n",M,N,K);

   srand(time(NULL));

   TIN *a,*b;
   TOUT *c;

   a = (TIN*) malloc(M*K*sizeof(TIN));
   b = (TIN*) malloc(K*N*sizeof(TIN));
   c = (TOUT*)malloc(M*N*sizeof(TOUT));

   initVal<TIN>(M*K,a);
   initVal<TIN>(K*N,b);

   simple_GEMM<TIN,TOUT,16,16>(M,N,K,a,b,c);

   return valid<TIN,TOUT>(M,N,K,a,b,c,1e-6);
}

int main(){
    return !benchmark<int,int>(128,128,K_DEFAULT);
}