/**
 * @author gtyinstinct
 * generate fake libcudart.so to replace origin libcudart.so
*/

#include<cstdio>
#include<cassert>
#include<driver_types.h>
#include<unistd.h>
#include<cstdlib>
#include<iostream>
#include<fstream>
#include<sstream>
#include<map>
#include<cstdint>

#include "antlr4-runtime.h"
#include "ptxLexer.h"
#include "ptxParser.h"
#include "ptxParserBaseListener.h"
#include "ptx-semantic.h"
#include "ptx-interpreter.h"

#define __my_func__ __func__

using namespace ptxparser;
using namespace antlr4;

std::string ptx_buffer;
std::map<uint64_t,std::string>func2name;
dim3 _gridDim,_blockDim;
size_t _sharedMem;
PtxListener ptxListener;
PtxInterpreter ptxInterpreter;

std::map<uint64_t,bool>memAlloc;

extern "C" {

void** __cudaRegisterFatBinary(
    void *fatCubin
)
{
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif

    static bool if_executed = 0;

    if(!if_executed){
        if_executed = 1;
        // get program abspath 
        char self_exe_path[1025] = "";
        long size = readlink("/proc/self/exe",self_exe_path,1024);
        assert(size!=-1);
        self_exe_path[size] = '\0';
        #ifdef LOGEMU
        printf("EMU: self exe links to %s\n",self_exe_path);
        #endif

        // get ptx file name embedded in binary
        char cmd[1024] = "";
        snprintf(cmd,1024,"cuobjdump -lptx %s | cut -d : -f 2 | awk '{$1=$1}1' > %s",
            self_exe_path,"__ptx_list__");
        if(system(cmd)!=0){
            #ifdef LOGEMU
            printf("EMU: fail to execute %s\n",cmd);
            #endif
            exit(0);
        }

        // get ptx embedded in binary
        std::ifstream infile("__ptx_list__");
        std::string ptx_file;
        while(std::getline(infile,ptx_file)){
            #ifdef LOGEMU
            printf("EMU: extract PTX file %s \n",ptx_file.c_str());
            #endif
            snprintf(cmd,1024,"cuobjdump -xptx %s %s >/dev/null",ptx_file.c_str(),
                self_exe_path);
            if(system(cmd)!=0){
                #ifdef LOGEMU
                printf("EMU: fail to execute %s\n",cmd);
                #endif
                exit(0);
            }
            std::ifstream if_ptx(ptx_file);
            std::ostringstream of_ptx;
            char ch;
            while(of_ptx && if_ptx.get(ch)) of_ptx.put(ch);
            ptx_buffer = of_ptx.str();
            break;
        }

        // clean intermediate file
        snprintf(cmd,1024,"rm __ptx_list__ %s",ptx_file.c_str());
        system(cmd);

        // launch antlr4 parse
        ANTLRInputStream input(ptx_buffer);
        ptxLexer lexer(&input);
        CommonTokenStream tokens(&lexer);
        tokens.fill();
        ptxParser parser(&tokens);
        parser.addParseListener(&ptxListener);
        tree::ParseTree *tree = parser.ast();
    }
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
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    printf("EMU: hostFun %p\n",hostFun);
    printf("EMU: deviceFun %p\n",deviceFun);
    printf("EMU: deviceFunName %s\n",deviceName);
    #endif
    func2name[(uint64_t)hostFun] = *(new std::string(deviceName));
}

void __cudaRegisterFatBinaryEnd(
    void **fatCubinHandle
)
{
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
}

cudaError_t cudaMalloc(
    void **p, 
    size_t s
)
{
    *p = malloc(s);
    memAlloc[(uint64_t)p] = 1;
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

cudaError_t cudaMemcpy(
    void               *dst, 
    const void         *src, 
    size_t              count, 
    enum cudaMemcpyKind kind
)
{
    memcpy(dst,src,count);
    #ifdef LOGEMU
    printf("EMU: memcpy dst:%p src:%p\n",dst,src);
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

cudaError_t cudaEventCreate(
    cudaEvent_t  *event,
    unsigned int  flags
)
{
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

cudaError_t cudaEventRecord(
    cudaEvent_t  event, 
    cudaStream_t stream
)
{
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

unsigned __cudaPushCallConfiguration(
    dim3                gridDim,
    dim3                blockDim, 
    size_t              sharedMem = 0, 
    struct CUstream_st *stream = 0 // temporily ignore stream
)
{
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    printf("EMU: gridDim(%d,%d,%d)\n",gridDim.x,gridDim.y,gridDim.z);
    printf("EMU: blockDim(%d,%d,%d)\n",blockDim.x,blockDim.y,blockDim.z);
    #endif
    _gridDim = gridDim;
    _blockDim = blockDim;
    _sharedMem = sharedMem;
    return 0;
}

cudaError_t cudaEventSynchronize(
    cudaEvent_t event
)
{
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(
    float       *ms, 
    cudaEvent_t  start, 
    cudaEvent_t  end
)
{
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

cudaError_t cudaFree(
    void *devPtr
)
{
    // avoid double free
    if(memAlloc[(uint64_t)devPtr]){
        free(devPtr);
        memAlloc[(uint64_t)devPtr] = 0;
    }
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

void __cudaUnregisterFatBinary(
    void **fatCubinHandle
)
{
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
}

cudaError_t __cudaPopCallConfiguration(
  dim3         *gridDim,
  dim3         *blockDim,
  size_t       *sharedMem,
  void         *stream
)
{
    *gridDim = _gridDim;
    *blockDim = _blockDim;
    *sharedMem = _sharedMem;
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

cudaError_t cudaLaunchKernel(
    const void  *func, 
    dim3         gridDim, 
    dim3         blockDim, 
    void       **args, 
    size_t       sharedMem, 
    cudaStream_t stream     // temporily ignore stream
)
{
    // ptxListener.test_semantic();

    ptxInterpreter.launchPtxInterpreter(ptxListener.ptxContext,
        func2name[(uint64_t)func],args,gridDim,blockDim);
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    printf("EMU: deviceFunName %s\n",func2name[(uint64_t)func].c_str());
    printf("EMU: arg %p\n",args);
    #endif

    return cudaSuccess;
}

void __cudaRegisterVar(
    void      **fatCubinHandle,
    char       *hostVar,           
    char       *deviceAddress,     
    const char *deviceName,  
    int         ext, 
    int         size, 
    int         constant, 
    int         global
) 
{
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
}

cudaError_t cudaMallocManaged ( 
    void**        devPtr, 
    size_t        size, 
    unsigned int  flags = cudaMemAttachGlobal 
)
{
    *devPtr = malloc(size);
    memAlloc[(uint64_t)devPtr] = 1;
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize ( 
    void
)
{
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

cudaError_t cudaMemset (
    void* devPtr,
    int  value, 
    size_t count 
)
{
    memset(devPtr,value,count);
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

cudaError_t cudaGetLastError ( 
    void 
)
{
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync ( 
    void*        devPtr, 
    int          value, 
    size_t       count, 
    cudaStream_t stream = 0 
)
{
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

cudaError_t cudaPeekAtLastError ( 
    void 
)
{
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

cudaError_t cudaThreadSynchronize ( 
    void 
)
{
    #ifdef LOGEMU
    printf("EMU: call %s\n",__my_func__);
    #endif
    return cudaSuccess;
}

} // end of extern "C"

