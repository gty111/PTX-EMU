#include<cstdio>
#include<cassert> 

#define SIZE 1024
#define BLOCK_DIM 32

template<typename T>
__global__ void dummy_d(T *arr){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    arr[idx] = idx;
}

template<typename T>
bool dummy_h(){
    bool ifPASS = 0;
    T *a_h,*a_d;

    a_h = (T *)malloc(SIZE*sizeof(T));

    cudaMalloc(&a_d,SIZE*sizeof(T));
    
    printf("host ptr:%p device ptr:%p\n",a_h,a_d);

    assert(SIZE%BLOCK_DIM==0);
    dummy_d<T> <<<SIZE/BLOCK_DIM,BLOCK_DIM>>>(a_d);

    cudaMemcpy(a_h,a_d,SIZE*sizeof(T),cudaMemcpyDeviceToHost);

    for(int i=0;i<SIZE;i++){
        if(a_h[i]!=i){
            printf("at:%p expect:%d got:%d\n",&a_h[i],i,a_h[i]);
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