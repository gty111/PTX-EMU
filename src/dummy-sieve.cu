#include<cstdio>
#define SIZE 1024

template<typename T>
__global__ void dummy_d(T *arr){
    if(threadIdx.x<2){
        arr[threadIdx.x] = 0;
        return;
    }
    for(int i=2;i*i<=threadIdx.x;i++){
        if(threadIdx.x%i==0){
            arr[threadIdx.x] = 0;
            return;
        }
    }
    arr[threadIdx.x] = 1;
}

bool prime(int num){
    if(num<2)return false;
    for(int i=2;i*i<=num;i++){
        if(num%i==0)return false;
    }
    return true;
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
        if(a_h[i]!=prime(i)){
            printf("at:%p expect:%d got:%d\n",&a_h[i],prime(i),a_h[i]);
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