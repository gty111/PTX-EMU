#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

void reference (int Lx, int Ly, int threshold, int maxRad, 
                float *img, int *box, float *norm, float *out)
{
  float q, sum, s;
  int ksum;

  for (int x = 0; x < Lx; x++) {
    for (int y = 0; y < Ly; y++) {
      sum = 0.f;
      s = q = 1;  // box size
      ksum = 0; // kernel sum

      while (sum < threshold && q < maxRad) {
        s = q;
        sum = 0.f;
        ksum = 0;

        for (int i = -s; i < s+1; i++)
          for (int j = -s; j < s+1; j++)
            if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly) {
              sum += img[(x+i)*Ly+y+j];
              ksum++;
            }
        q++;
      }

      box[x*Ly+y] = s;  // save the box size

      for (int i = -s; i < s+1; i++)
        for (int j = -s; j < s+1; j++)
          if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly)
            if (ksum != 0) norm[(x+i)*Ly+y+j] += 1.f / (float)ksum;
    }
  }

  // normalize the image
  for (int x = 0; x < Lx; x++)
    for (int y = 0; y < Ly; y++) 
      if (norm[x*Ly+y] != 0) img[x*Ly+y] /= norm[x*Ly+y];

  // output file
  for (int x = 0; x < Lx; x++) {
    for (int y = 0; y < Ly; y++) {
      s = box[x*Ly+y];
      sum = 0.f;
      ksum = 0;

      // resmooth with normalized image
      for (int i = -s; i < s+1; i++)
        for (int j = -s; j < s+1; j++) {
          if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly) {
            sum += img[(x+i)*Ly+y+j];
            ksum++;
          }
        }
      if (ksum != 0) out[x*Ly+y] = sum / (float)ksum;
    }
  }
}

void verify (
  const int size,
  const int MaxRad,
  const float* norm,
  const float* h_norm,
  const float* out,
  const float* h_out,
  const   int* box,
  const   int* h_box)
{
  bool ok = true;
  int cnt[10] = {0,0,0,0,0,0,0,0,0,0};
  for (int i = 0; i < size; i++) {
    if (fabsf(norm[i] - h_norm[i]) > 1e-3f) {
      printf("norm: %d %f %f\n", i, norm[i], h_norm[i]);
      ok = false;
      break;
    }
    if (fabsf(out[i] - h_out[i]) > 1e-3f) {
      printf("out: %d %f %f\n", i, out[i], h_out[i]);
      ok = false;
      break;
    }
    if (box[i] != h_box[i]) {
      printf("box: %d %d %d\n", i, box[i], h_box[i]);
      ok = false;
      break;
    } else {
      for (int j = 0; j < MaxRad; j++)
        if (box[i] == j) { cnt[j]++; break; }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (ok) {
    printf("Distribution of box sizes:\n");
    for (int j = 1; j < MaxRad; j++)
      printf("size=%d: %f\n", j, (float)cnt[j]/size);
  }
}



__global__ void smoothingFilter(
    int Lx, int Ly, 
    int Threshold, int MaxRad, 
    const float*__restrict__ Img,
            int*__restrict__ Box,
          float*__restrict__ Norm)
{
  int tid = threadIdx.x;
  int tjd = threadIdx.y;
  int i = blockIdx.x * blockDim.x + tid;
  int j = blockIdx.y * blockDim.y + tjd;
  int stid = tjd * blockDim.x + tid;
  int gtid = j * Lx + i;  

  // part of shared memory may be unused
  __shared__ float s_Img[1024];

  if ( i < Lx && j < Ly )
    s_Img[stid] = Img[gtid];

  __syncthreads();

  if ( i < Lx && j < Ly )
  {
    // Smoothing parameters
    float sum = 0.f;
    int q = 1;
    int s = q;
    int ksum = 0;

    // Continue until parameters are met
    while (sum < Threshold && q < MaxRad)
    {
      s = q;
      sum = 0.f;
      ksum = 0;

      // Normal adaptive smoothing
      for (int ii = -s; ii < s+1; ii++)
        for (int jj = -s; jj < s+1; jj++)
          if ( (i-s >= 0) && (i+s < Ly) && (j-s >= 0) && (j+s < Lx) )
          {
            ksum++;
            // Compute within bounds of block dimensions
            if( tid-s >= 0 && tid+s < blockDim.x && tjd-s >= 0 && tjd+s < blockDim.y )
              sum += s_Img[stid + ii*blockDim.x + jj];
            // Compute block borders with global memory
            else
              sum += Img[gtid + ii*Lx + jj];
          }
      q++;
    }
    Box[gtid] = s;

    // Normalization for each box
    for (int ii = -s; ii < s+1; ii++)
      for (int jj = -s; jj < s+1; jj++)
        if (ksum != 0) 
          atomicAdd(&Norm[gtid + ii*Lx + jj], __fdividef(1.f, (float)ksum));
  }
}

__global__ void normalizeFilter(
    int Lx, int Ly, 
          float*__restrict__ Img,
    const float*__restrict__ Norm)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if ( i < Lx && j < Ly ) {
    int gtid = j * Lx + i;  
    const float norm = Norm[gtid];
    if (norm != 0) Img[gtid] = __fdividef(Img[gtid], norm);
  }
}

__global__ void outFilter( 
    int Lx, int Ly,
    const float*__restrict__ Img,
    const   int*__restrict__ Box,
          float*__restrict__ Out )
{
  int tid = threadIdx.x;
  int tjd = threadIdx.y;
  int i = blockIdx.x * blockDim.x + tid;
  int j = blockIdx.y * blockDim.y + tjd;
  int stid = tjd * blockDim.x + tid;
  int gtid = j * Lx + i;  

  // part of shared memory may be unused
  __shared__ float s_Img[1024];

  if ( i < Lx && j < Ly )
    s_Img[stid] = Img[gtid];

  __syncthreads();

  if ( i < Lx && j < Ly )
  {
    const int s = Box[gtid];
    float sum = 0.f;
    int ksum  = 0;

    for (int ii = -s; ii < s+1; ii++)
      for (int jj = -s; jj < s+1; jj++)
        if ( (i-s >= 0) && (i+s < Lx) && (j-s >= 0) && (j+s < Ly) )
        {
          ksum++;
          if( tid-s >= 0 && tid+s < blockDim.x && tjd-s >= 0 && tjd+s < blockDim.y )
            sum += s_Img[stid + ii*blockDim.y + jj];
          else
            sum += Img[gtid + ii*Ly + jj];
        }
    if ( ksum != 0 ) Out[gtid] = __fdividef(sum , (float)ksum);
  }
}

int main(int argc, char* argv[]) {
  /*
  if (argc != 5) {
     printf("./%s <image dimension> <threshold> <max box size> <iterations>\n", argv[0]);
     exit(1);
  }
  */

  // only a square image is supported
  const int Lx = 8192;//atoi(argv[1]);
  const int Ly = Lx;
  const int size = Lx * Ly;

  const int Threshold = 2000;//atoi(argv[2]);
  const int MaxRad = 9;//atoi(argv[3]);
  const int repeat = 1;//atoi(argv[4]);
 
  // input image
  float *img = (float*) malloc (sizeof(float) * size);

  // host and device results
  float *norm = (float*) malloc (sizeof(float) * size);
  float *h_norm = (float*) malloc (sizeof(float) * size);

  int *box = (int*) malloc (sizeof(int) * size);
  int *h_box = (int*) malloc (sizeof(int) * size);

  float *out = (float*) malloc (sizeof(float) * size);
  float *h_out = (float*) malloc (sizeof(float) * size);

  srand(123);
  for (int i = 0; i < size; i++) {
    img[i] = rand() % 256;
    norm[i] = box[i] = out[i] = 0;
  }

  float *d_img;
  cudaMalloc((void**)&d_img, sizeof(float) * size);

  float *d_norm;
  cudaMalloc((void**)&d_norm, sizeof(float) * size);

  int *d_box;
  cudaMalloc((void**)&d_box, sizeof(int) * size);

  float *d_out;
  cudaMalloc((void**)&d_out, sizeof(float) * size);

  dim3 grids ((Lx+15)/16, (Ly+15)/16);
  dim3 blocks (16, 16);

  // reset output
  cudaMemcpy(d_out, out, sizeof(float) * size, cudaMemcpyHostToDevice);

  double time = 0;

  for (int i = 0; i < repeat; i++) {
    // restore input image
    cudaMemcpy(d_img, img, sizeof(float) * size, cudaMemcpyHostToDevice);
    // reset norm
    cudaMemcpy(d_norm, norm, sizeof(float) * size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    // launch three kernels
    smoothingFilter<<<grids, blocks>>>(Lx, Ly, Threshold, MaxRad, d_img, d_box, d_norm);
    normalizeFilter<<<grids, blocks>>>(Lx, Ly, d_img, d_norm);
    outFilter<<<grids, blocks>>>(Lx, Ly, d_img, d_box, d_out);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Average filtering time %lf (s)\n", (time * 1e-9) / repeat);

  cudaMemcpy(out, d_out, sizeof(float) * size, cudaMemcpyDeviceToHost);
  cudaMemcpy(box, d_box, sizeof(int) * size, cudaMemcpyDeviceToHost);
  cudaMemcpy(norm, d_norm, sizeof(float) * size, cudaMemcpyDeviceToHost);

  // verify
  reference (Lx, Ly, Threshold, MaxRad, img, h_box, h_norm, h_out);
  verify(size, MaxRad, norm, h_norm, out, h_out, box, h_box);

  cudaFree(d_img);
  cudaFree(d_norm);
  cudaFree(d_box);
  cudaFree(d_out);
  free(img);
  free(norm);
  free(h_norm);
  free(box);
  free(h_box);
  free(out);
  free(h_out);
  return 0;
}