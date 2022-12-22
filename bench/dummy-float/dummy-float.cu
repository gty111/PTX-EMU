#include<cstdio>
#include<cmath>
#define SIZE (1024)
#define MULNUM (12.21)
#define ADDNUM (321.123)

template<typename T>
__global__ void dummy_d(T *arr){
    arr[threadIdx.x] = threadIdx.x*MULNUM+ADDNUM;
}

template<typename T>
bool dummy_h(){
    bool ifPASS = 0;
    T *a_h,*a_d;

    a_h = (T *)malloc(SIZE*sizeof(T));

    cudaMalloc(&a_d,SIZE*sizeof(T));

    dummy_d<T> <<<1,SIZE>>>(a_d);

    cudaMemcpy(a_h,a_d,SIZE*sizeof(T),cudaMemcpyDeviceToHost);

    for(int i=0;i<SIZE;i++){
        if(fabs(a_h[i]-(i*MULNUM+ADDNUM))>1e-3){
            printf("at:%p expect:%.10lf got:%.10lf\n",&a_h[i],i*MULNUM+ADDNUM,a_h[i]);
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
    return dummy_h<float>();
}   