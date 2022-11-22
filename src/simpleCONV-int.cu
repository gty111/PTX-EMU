#include <cassert>
#include <cstdio>
#include <functional>
#include <vector>

#define BLOCK_DIM_DEFAULT 128
#define TEST_TIME 1

template <typename TIN,typename TOUT,
      int M_TILE,int N_TILE>
__global__ void conv_kernel(TIN *a, TIN *conv, TOUT *c,int M,int N) {
    __shared__ TIN shm[M_TILE*N_TILE];

    for(int i=threadIdx.x;i<M_TILE*N_TILE;i+=blockDim.x){
      shm[i] = conv[i];
    }

    __syncthreads();
    

    const int M_out = M-M_TILE+1;
    const int N_out = N-N_TILE+1;
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int midx = idx/N_out;
    const int nidx = idx%N_out;

    if(idx>=M_out*N_out)return;

    TOUT acc=0;
    for(int i=0;i<M_TILE;i++)
    for(int j=0;j<N_TILE;j++){
      acc += shm[i*N_TILE+j] * a[(midx+i)*N+nidx+j];
    }
    c[idx] = acc;

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
void copy(int ARR_M,int ARR_N,TARR *arr,
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
void simple_CONV(int M,int N,TIN *a_in,TIN *conv_in,TOUT *c_out){
   assert(M!=0 && N!=0 && M_TILE!=0 && N_TILE!=0);
   assert(M_TILE%2!=0 && N_TILE%2!=0);
   assert(M>=M_TILE && N>=N_TILE);
   
   const int M_out = M-M_TILE+1;
   const int N_out = N-N_TILE+1;

   cuda_vector<TIN> a(M*N),conv(M_TILE*N_TILE);
   cuda_vector<TOUT> c(M_out*N_out);

   //init a b
   copy<TIN,TIN,ARR2VECTOR>(M,N,a_in,M,N,a);
   copy<TIN,TIN,ARR2VECTOR>(M_TILE,N_TILE,conv_in,M_TILE,N_TILE,conv);
   
   // sync to device
   a.sync_device();
   conv.sync_device();

   int GRID_DIM,BLOCK_DIM;
   BLOCK_DIM = BLOCK_DIM_DEFAULT;
   GRID_DIM = (M_out*N_out)%BLOCK_DIM ? (M_out*N_out)/BLOCK_DIM+1 :
                                        (M_out*N_out)/BLOCK_DIM ;
   printf("GRID_DIM:%d BLOCK_DIM:%d\n",GRID_DIM,BLOCK_DIM);
   Timer("simpleCONV", [&]{
   conv_kernel<TIN,TOUT,M_TILE,N_TILE>
      <<<GRID_DIM,BLOCK_DIM>>>(a.device_data,conv.device_data,c.device_data,M,N);});

   c.sync_host();

   copy<TOUT,TOUT,VECTOR2ARR>(M_out,N_out,c_out,M_out,N_out,c);
}


template<typename TIN,typename TOUT,int M_TILE,int N_TILE>
bool valid(int M,int N,TIN *a,TIN *conv,TOUT *c,double threshold=1e-3){
   const int conv_m = (M_TILE-1)/2;
   const int conv_n = (N_TILE-1)/2;
   for(int i=conv_m;i<M-conv_m;i++)
   for(int j=conv_n;j<N-conv_n;j++){
      TOUT val = 0;
      for(int ii=i-conv_m;ii<=i+conv_m;ii++)
      for(int jj=j-conv_n;jj<=j+conv_n;jj++){
         val += a[ii*N+jj] * conv[(ii-(i-conv_m))*N_TILE+jj-(j-conv_n)];
      }
      TOUT abs = val - c[(i-conv_m)*(M-M_TILE+1)+j-conv_n];
      abs = abs>0 ? abs : -abs;
      if(abs/val>threshold){
         std::printf("at %d %d ",i,j);
         std::printf("expect:%f got:%f relative error:%f%%(>%f%%)\n",
            (float)val,(float)c[(i-conv_m)*(M-M_TILE+1)+j-conv_n],
            (float)abs/val*100,threshold*100);
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

template<typename TIN,typename TOUT,int M_TILE,int N_TILE>
void benchmark(int M,int N,bool ifcheck=1){
   printf("----------------\n");
   printf("M:%d N:%d M_TILE:%d N_TILE:%d\n",M,N,M_TILE,N_TILE);

   srand(time(NULL));

   TIN *a,*conv;
   TOUT *c;

   a = (TIN*) malloc(M*N*sizeof(TIN));
   conv = (TIN*) malloc(M_TILE*N_TILE*sizeof(TIN));
   c = (TOUT*)malloc((M-M_TILE+1)*(N-N_TILE+1)*sizeof(TOUT));

   initVal<TIN>(M*N,a);
   initVal<TIN>(M_TILE*N_TILE,conv);

   simple_CONV<TIN,TOUT,M_TILE,N_TILE>(M,N,a,conv,c);

   if(ifcheck && valid<TIN,TOUT,M_TILE,N_TILE>(M,N,a,conv,c,1e-6)){
      std::printf("check pass\n");
   }else if(ifcheck){
      std::printf("check fail\n");
   }else{
      std::printf("skip check\n");
   }
   
}

int main(){
    benchmark<int,int,31,31>(128,128,1);
}