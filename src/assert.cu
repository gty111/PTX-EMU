/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <chrono>

// Tests assert function.
// Thread whose id > N will print assertion failed error message.
__global__ void testKernel(int N)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x ;
  assert(gid < N) ;
}

// Performance impact of assert()
__global__ void perfKernel()
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x ;
  assert(gid <= blockDim.x * gridDim.x) ;
  int s = 0;
  for (int n = 1; n <= gid; n++) {
    s++; assert(s <= gid);
  }
}

__global__ void perfKernel2()
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x ;
  int s = 0;
  for (int n = 1; n <= gid; n++) {
    s++; assert(s <= gid);
  }
}

// Declaration, forward
bool runPerf(int argc, char **argv);
bool runTest(int argc, char **argv);

int main(int argc, char **argv)
{
  // The test expects no assertError
  runPerf(argc, argv);

  // The test expects assertError
  bool testResult = runTest(argc, argv);

  printf("Test assert completed, returned %s\n",
         testResult ? "OK" : "ERROR!");

  if (!testResult) return EXIT_FAILURE;

  exit(EXIT_SUCCESS);
}

bool runTest(int argc, char **argv)
{
  int Nblocks = 2;
  int Nthreads = 32;
  cudaError_t error ;

  // Kernel configuration, where a one-dimensional
  // grid and one-dimensional blocks are configured.
  dim3 dimGrid(Nblocks);
  dim3 dimBlock(Nthreads);

  printf("\nLaunch kernel to generate assertion failures\n");
  testKernel<<<dimGrid, dimBlock>>>(60);

  // Synchronize (flushes assert output).
  printf("\n-- Begin assert output\n\n");
  error = cudaDeviceSynchronize();
  printf("\n-- End assert output\n\n");

  // Check for errors and failed asserts in asynchronous kernel launch.
  if (error == cudaErrorAssert)
  {
    printf("Device assert failed as expected, "
           "CUDA error message is: %s\n\n",
           cudaGetErrorString(error));
  }

  return (error == cudaErrorAssert);
}

bool runPerf(int argc, char **argv)
{
  int Nblocks = 1000;
  int Nthreads = 256;

  dim3 dimGrid(Nblocks);
  dim3 dimBlock(Nthreads);

  printf("\nLaunch kernel to evaluate the impact of assertion on performance \n");

  printf("Each thread in the kernel executes threadID + 1 assertions\n");
  auto start = std::chrono::steady_clock::now();
  perfKernel<<<dimGrid, dimBlock>>>();
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  printf("Kernel time : %f\n", time.count());

  printf("Each thread in the kernel executes threadID assertions\n");
  start = std::chrono::steady_clock::now();
  perfKernel2<<<dimGrid, dimBlock>>>();
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = end - start;
  printf("Kernel time : %f\n", time.count());

  return true;
}