#include<cstdio>
#define SIZE 1024

__global__ void dummy(int *arr){
    arr[threadIdx.x] = threadIdx.x;
}

int main(){
    int *a_h,*a_d;

    a_h = (int *)malloc(SIZE*sizeof(int));

    cudaMalloc(&a_d,SIZE*sizeof(int));

    dummy<<<1,SIZE>>>(a_d);

    cudaMemcpy(a_h,a_d,SIZE*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0;i<SIZE;i++){
        if(a_h[i]!=i){
            printf("check fail\n");
            goto End;
        }
    }
    printf("check pass\n");

    End:
    cudaFree(a_d);
    free(a_h);
}