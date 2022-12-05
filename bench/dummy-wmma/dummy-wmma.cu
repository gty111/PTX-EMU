#include <cstdio>
#include <mma.h>
using namespace nvcuda;

      
__global__ void wmma_ker(half *a, half *b, float *c) {
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);

   // Perform the matrix multiplication
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // Store the output
   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}  

template<typename T>
void initVal(T *arr,int size){
   for(int i=0;i<size;i++){
      arr[i] = (rand()%1024) / (rand()%1024+1);
   }
}

bool dummy_wmma(){
   srand(8);
   half *a_h,*a_d,*b_h,*b_d;
   float *c_h,*c_d;
   a_h = (half*)malloc(sizeof(half)*16*16);
   b_h = (half*)malloc(sizeof(half)*16*16);
   c_h = (float*)malloc(sizeof(float)*16*16);

   cudaMalloc(&a_d,sizeof(half)*16*16);
   cudaMalloc(&b_d,sizeof(half)*16*16);
   cudaMalloc(&c_d,sizeof(float)*16*16);

   initVal(a_h,16*16);
   initVal(b_h,16*16);

   cudaMemcpy(a_d,a_h,sizeof(half)*16*16,cudaMemcpyHostToDevice);
   cudaMemcpy(b_d,b_h,sizeof(half)*16*16,cudaMemcpyHostToDevice);
   
   wmma_ker<<<1,32>>>(a_d,b_d,c_d);

   cudaMemcpy(c_h,c_d,sizeof(float)*16*16,cudaMemcpyDeviceToHost);

   for(int i=0;i<16;i++){
      for(int j=0;j<16;j++){
         float sum = 0;
         for(int k=0;k<16;k++){
            sum += (float)a_h[i*16+k]*(float)b_h[k*16+j];
         }
         if(sum-c_h[i*16+j]>1e-6){
            printf("at(%d,%d) expect:%f got:%f\n",i,j,sum,c_h[i*16+j]);
            return 1;
         }
      }
   }
   return 0;
}



int main(){
   return dummy_wmma();
}