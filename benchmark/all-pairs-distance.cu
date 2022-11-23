/* This code is provided as supplementary material for the book
   chapter "Exploiting graphics processing units for computational
   biology and bioinformatics," by Payne, Sinnott-Armstrong, and
   Moore, to appear in "The Handbook of Research on Computational and
   Systems Biology: Interdisciplinary applications," by IGI Global.

   Please feel free to use, modify, or redistribute this code.

   Make sure you have a CUDA compatible GPU and the nvcc is installed.
   To compile, type make.
   After compilation, type ./chapter to run
   Output written to timing.txt
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <sys/time.h>

#define INSTANCES 64   /* # of instances */
#define ATTRIBUTES 256 /* # of attributes */
#define THREADS 128    /* # of threads per block */

/* CPU implementation */
void CPU(int * data, int * distance) {
  /* compare all pairs of instances, accessing the attributes in
     row-major order */
#pragma omp parallel for collapse(2)
  for (int i = 0; i < INSTANCES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      for (int k = 0; k < ATTRIBUTES; k++) {
        distance[i + INSTANCES * j] += 
          (data[i * ATTRIBUTES + k] != data[j * ATTRIBUTES + k]);
      }
    }
  }
}


/*  coalesced GPU implementation of the all-pairs kernel using
    character data types and registers */
__global__ void GPUregister(const char *data, int *distance) {
  int idx = threadIdx.x;
  int gx = blockIdx.x;
  int gy = blockIdx.y;

  for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
    char4 j = *(char4 *)(data + i + ATTRIBUTES*gx);
    char4 k = *(char4 *)(data + i + ATTRIBUTES*gy);

    /* use a local variable (stored in register) to hold intermediate
       values. This reduces writes to global memory */
    char count = 0;

    if(j.x ^ k.x) 
      count++; 
    if(j.y ^ k.y)
      count++;
    if(j.z ^ k.z)
      count++;
    if(j.w ^ k.w)
      count++;

    /* Only one atomic write to global memory */
    atomicAdd(distance + INSTANCES*gx + gy, count);
  }
}

/*  coalesced GPU implementation of the all-pairs kernel using
    character data types, registers, and shared memory */
__global__ void GPUshared(const char *data, int *distance) {
  int idx = threadIdx.x;
  int gx = blockIdx.x;
  int gy = blockIdx.y;

  /* Shared memory is the other major memory (other than registers and
     global). It is used to store values between multiple threads. In
     particular, the shared memory access is defined by the __shared__
     attribute and it is a special area of memory on the GPU
     itself. Because the memory is on the chip, it is a lot faster
     than global memory. Multiple threads can still access it, though,
     provided they are in the same block.
   */
  __shared__ int dist[THREADS];

  /* each thread initializes its own location of the shared array */ 
  dist[idx] = 0;

  /* At this point, the threads must be synchronized to ensure that
     the shared array is fully initialized. */
  __syncthreads();

  for(int i = idx*4; i < ATTRIBUTES; i+=THREADS*4) {
    char4 j = *(char4 *)(data + i + ATTRIBUTES*gx);
    char4 k = *(char4 *)(data + i + ATTRIBUTES*gy);
    char count = 0;

    if(j.x ^ k.x) 
      count++;
    if(j.y ^ k.y)
      count++;
    if(j.z ^ k.z)
      count++;
    if(j.w ^ k.w)
      count++;

    /* Increment shared array */
    dist[idx] += count;
  }

  /* Synchronize threads to make sure all have completed their updates
     of the shared array. Since the distances for each thread are read
     by thread 0 below, this must be ensured. Above, it was not
     necessary because each thread was accessing its own memory
   */
  __syncthreads();

  /* Reduction: Thread 0 will add the value of all other threads to
     its own */ 
  if(idx == 0) {
    for(int i = 1; i < THREADS; i++) {
      dist[0] += dist[i];
    }

    /* Thread 0 will then write the output to global memory. Note that
       this does not need to be performed atomically, because only one
       thread per block is writing to global memory, and each block
       corresponds to a unique memory address. 
     */
    distance[INSTANCES*gy + gx] = dist[0];
  }
}

int main(int argc, char **argv) {
  /*
  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }
  */
  
  const int iterations = 1;//atoi(argv[1]);

  /* host data */
  int *data; 
  char *data_char;
  int *cpu_distance; 
  int *gpu_distance; 

  /* device data */
  char *data_char_device;
  int *distance_device; 

  /* block and grid dimensions */
  dim3 dimBlock; 
  dim3 dimGrid; 

  /* used to time CPU and GPU implementations */
  double start_cpu, stop_cpu;
  double start_gpu, stop_gpu;
  float elapsedTime; 
  struct timeval tp;
  struct timezone tzp;
  /* verification result */ 
  int status_reg,status_share;

  /* seed RNG */
  srand(2);

  /* allocate host memory */
  data = (int *)malloc(INSTANCES * ATTRIBUTES * sizeof(int));
  data_char = (char *)malloc(INSTANCES * ATTRIBUTES * sizeof(char));
  cpu_distance = (int *)malloc(INSTANCES * INSTANCES * sizeof(int));
  gpu_distance = (int *)malloc(INSTANCES * INSTANCES * sizeof(int));

  /* randomly initialize host data */
#pragma omp parallel for collapse(2)
  for (int i = 0; i < ATTRIBUTES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      data[i + ATTRIBUTES * j] = data_char[i + ATTRIBUTES * j] = random() % 3;
    }
  }

  /* allocate GPU memory */
  cudaMalloc((void **)&data_char_device, 
      INSTANCES * ATTRIBUTES * sizeof(char));

  cudaMalloc((void **)&distance_device, 
      INSTANCES * INSTANCES * sizeof(int));

  cudaMemcpy(data_char_device, data_char,
      INSTANCES * ATTRIBUTES * sizeof(char),
      cudaMemcpyHostToDevice);

  /* specify grid and block dimensions */
  dimBlock.x = THREADS; 
  dimBlock.y = 1; 
  dimGrid.x = INSTANCES;
  dimGrid.y = INSTANCES;


  /* CPU */
  bzero(cpu_distance,INSTANCES*INSTANCES*sizeof(int));
  gettimeofday(&tp, &tzp);
  start_cpu = tp.tv_sec*1000000+tp.tv_usec;
  CPU(data, cpu_distance);
  gettimeofday(&tp, &tzp);
  stop_cpu = tp.tv_sec*1000000+tp.tv_usec;
  elapsedTime = stop_cpu - start_cpu;
  printf("CPU time: %f (us)\n",elapsedTime);


  elapsedTime = 0; 
  for (int n = 0; n < iterations; n++) {
    // register GPU kernel 
    bzero(gpu_distance,INSTANCES*INSTANCES*sizeof(int));
    gettimeofday(&tp, &tzp);
    start_gpu = tp.tv_sec*1000000+tp.tv_usec;
    cudaMemcpy(distance_device, gpu_distance, INSTANCES * INSTANCES * sizeof(int),
        cudaMemcpyHostToDevice);
    GPUregister<<<dimGrid,dimBlock>>>(data_char_device, distance_device);
    cudaMemcpy(gpu_distance, distance_device,
        INSTANCES * INSTANCES * sizeof(int),
        cudaMemcpyDeviceToHost); 
    gettimeofday(&tp, &tzp);
    stop_gpu = tp.tv_sec*1000000+tp.tv_usec;
    elapsedTime += stop_gpu - start_gpu;
  }

  printf("GPU time (w/o shared memory): %f (us)\n", elapsedTime / iterations);
  status_reg = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
  if (status_reg != 0) printf("FAIL\n");
  else printf("PASS\n");


  elapsedTime = 0; 
  for (int n = 0; n < iterations; n++) {
    /* shared memory GPU kernel */
    bzero(gpu_distance,INSTANCES*INSTANCES*sizeof(int));
    gettimeofday(&tp, &tzp);
    start_gpu = tp.tv_sec*1000000+tp.tv_usec;
    cudaMemcpy(distance_device, gpu_distance, INSTANCES * INSTANCES * sizeof(int),
        cudaMemcpyHostToDevice);
    GPUshared<<<dimGrid,dimBlock>>>(data_char_device, distance_device);
    cudaMemcpy(gpu_distance, distance_device,
        INSTANCES * INSTANCES * sizeof(int),
        cudaMemcpyDeviceToHost); 
    gettimeofday(&tp, &tzp);
    stop_gpu = tp.tv_sec*1000000+tp.tv_usec;
    elapsedTime += stop_gpu - start_gpu;
  }

  printf("GPU time (w/ shared memory): %f (us)\n", elapsedTime / iterations);
  status_share = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
  if (status_share != 0) printf("FAIL\n");
  else printf("PASS\n");

  free(cpu_distance);
  free(gpu_distance);
  free(data);
  cudaFree(data_char_device);
  cudaFree(distance_device);

  return status_reg & status_share;
}

