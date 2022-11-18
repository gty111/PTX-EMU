#include<cstdio>
#define SIZE 1024

__global__ void dummy(int *arr){
    arr[threadIdx.x] = threadIdx.x;
}

int main(){
    int *a_h,*a_d;

    a_h = (int *)malloc(SIZE*sizeof(int));

    cudaMalloc(&a_d,SIZE*sizeof(int));
    
    printf("host ptr:%p device ptr:%p\n",a_h,a_d);

    dummy<<<1,SIZE>>>(a_d);

    cudaMemcpy(a_h,a_d,SIZE*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0;i<SIZE;i++){
        if(a_h[i]!=i){
            printf("at:%p expect:%d got:%d\n",&a_h[i],i,a_h[i]);
            printf("FAIL\n");
            goto End;
        }
    }
    printf("PASS\n");

    End:
    cudaFree(a_d);
    free(a_h);
}