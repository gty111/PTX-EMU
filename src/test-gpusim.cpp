#include<cstdio>
#include<driver_types.h>

extern "C" {

void** __cudaRegisterFatBinary(
    void *fatCubin
)
{
    printf("call __cudaRegisterFatBinary\n",fatCubin);
    return nullptr;
}

void __cudaRegisterFunction(
        void   **fatCubinHandle,
  const char    *hostFun,
        char    *deviceFun,
  const char    *deviceName,
        int      thread_limit,
        uint3   *tid,
        uint3   *bid,
        dim3    *bDim,
        dim3    *gDim,
        int     *wSize
)
{
    printf("call __cudaRegisterFunction\n");
}

void __cudaRegisterFatBinaryEnd(
    void **fatCubinHandle
)
{
    printf("call __cudaRegisterFatBinaryEnd\n");
}

cudaError_t cudaMalloc(
    void **p, 
    size_t s
)
{
    printf("call cudaMalloc\n");
    return cudaSuccess;
}

cudaError_t cudaMemcpy(
    void               *dst, 
    const void         *src, 
    size_t              count, 
    enum cudaMemcpyKind kind
)
{
    printf("call cudaMemcpy\n");
    return cudaSuccess;
}

cudaError_t cudaEventCreate(
    cudaEvent_t  *event,
    unsigned int  flags
)
{
    printf("call cudaEventCreate\n");
    return cudaSuccess;
}

cudaError_t cudaEventRecord(
    cudaEvent_t  event, 
    cudaStream_t stream
)
{
    printf("call cudaEventRecord\n");
    return cudaSuccess;
}

unsigned __cudaPushCallConfiguration(
    dim3                gridDim,
    dim3                blockDim, 
    size_t              sharedMem = 0, 
    struct CUstream_st *stream = 0
)
{
    printf("call __cudaPushCallConfiguration\n");
    return 0;
}

cudaError_t cudaEventSynchronize(
    cudaEvent_t event
)
{
    printf("call cudaEventSynchronize\n");
    return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(
    float       *ms, 
    cudaEvent_t  start, 
    cudaEvent_t  end
)
{
    printf("call cudaEventElapsedTime\n");
    return cudaSuccess;
}

cudaError_t cudaFree(
    void *devPtr
)
{
    printf("call cudaFree\n");
    return cudaSuccess;
}

void __cudaUnregisterFatBinary(
    void **fatCubinHandle
)
{
    printf("call __cudaUnregisterFatBinary\n");
}

cudaError_t __cudaPopCallConfiguration(
  dim3         *gridDim,
  dim3         *blockDim,
  size_t       *sharedMem,
  void         *stream
)
{
    printf("call __cudaPopCallConfiguration\n");
    return cudaSuccess;
}

cudaError_t cudaLaunchKernel(
    const void  *func, 
    dim3         gridDim, 
    dim3         blockDim, 
    void       **args, 
    size_t       sharedMem, 
    cudaStream_t stream
){
    printf("call cudaLaunchKernel\n");
    return cudaSuccess;
}

}

