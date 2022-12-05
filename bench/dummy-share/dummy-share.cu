#include<cstdio>
#define SIZE 1024
#define BLOCK_DIM 128

template<typename T>
__global__ void dummy_d(T *arr){
    __shared__ T idx_arr[BLOCK_DIM];
    idx_arr[threadIdx.x] = threadIdx.x;
    __syncthreads();
    int idx = threadIdx.x == blockDim.x - 1 ? 0 : threadIdx.x+1;
    arr[blockDim.x*blockIdx.x+threadIdx.x] = idx_arr[idx];
}

template<typename T>
bool dummy_h(){
    bool ifPASS = 0;
    T *a_h,*a_d;

    a_h = (T *)malloc(SIZE*sizeof(T));

    cudaMalloc(&a_d,SIZE*sizeof(T));
    
    printf("host ptr:%p device ptr:%p\n",a_h,a_d);

    dummy_d<T> <<<SIZE/BLOCK_DIM,BLOCK_DIM>>>(a_d);

    cudaMemcpy(a_h,a_d,SIZE*sizeof(T),cudaMemcpyDeviceToHost);

    for(int i=0;i<SIZE;i++){
        if(a_h[i]!=(i%BLOCK_DIM+1)%BLOCK_DIM){
            printf("at:%d expect:%d got:%d\n",i,(i%BLOCK_DIM+1)%BLOCK_DIM,a_h[i]);
            printf("FAIL\n");
            ifPASS = 1;
            goto End;
        }
    }
    printf("PASS\n");

    End:
    cudaFree(a_d);
    free(a_h);

    return ifPASS;
}

int main(){
    return dummy_h<int>();
}   