#include<cstdio>
#define SIZE 1024
#define NUM 127

template<typename T>
__global__ void dummy_d(T *arr){
    arr[threadIdx.x] = threadIdx.x-NUM;
}

template<typename T>
bool dummy_h(){
    bool ifPASS = 0;
    T *a_h,*a_d;

    a_h = (T *)malloc(SIZE*sizeof(T));

    cudaMalloc(&a_d,SIZE*sizeof(T));
    
    printf("host ptr:%p device ptr:%p\n",a_h,a_d);

    dummy_d<T> <<<1,SIZE>>>(a_d);

    cudaMemcpy(a_h,a_d,SIZE*sizeof(T),cudaMemcpyDeviceToHost);

    for(int i=0;i<SIZE;i++){
        if(a_h[i]!=i-NUM){
            printf("at:%p expect:%d got:%d\n",&a_h[i],i-NUM,a_h[i]);
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