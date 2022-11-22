#include <cstdio>
#include <cuda_runtime.h>
#include <functional>
#include <vector>
#include <cstdint>
template <int BLOCK_DIM_X = 128, typename TIN = uint8_t, typename TOUT = float,
          typename TCNT = uint8_t, int CNT_DIM = 256, int CONV_DIM = 2>
static void __global__ __launch_bounds__(BLOCK_DIM_X)
    entropy_with_register_spiling(const int width, const int height,
                                  const TIN *input, TOUT *output) {
  const int idy = blockIdx.y * blockDim.y + threadIdx.y,
            idx = blockIdx.x * blockDim.x + threadIdx.x,
            n = (min(idx, 2) + 1 + min(width - idx, 2)) *
                (min(idy, 2) + 1 + min(height - idy, 2));
  TOUT __shared__ plogp[(2 * CONV_DIM + 1) * (2 * CONV_DIM + 1) + 1];
  for (int i = threadIdx.x; i < (2 * CONV_DIM + 1) * (2 * CONV_DIM + 1) + 1;
       i += BLOCK_DIM_X)
    plogp[i] = i ? (TOUT)i * log((TOUT)i) : 0;
  __syncthreads();
  if (idy < height && idx < width) {
    TCNT cnt[CNT_DIM];
    for (int i = 0; i < CNT_DIM; ++i)
      cnt[i] = 0;
    for (int offsety = -CONV_DIM; offsety <= CONV_DIM; ++offsety) {
      const int py = idy + offsety;
      if (0 <= py && py < height)
        for (int offsetx = -CONV_DIM; offsetx <= CONV_DIM; ++offsetx) {
          const int px = idx + offsetx;
          if (0 <= px && px < width)
            ++cnt[input[py * width + px]];
        }
    }
    TOUT n_inv = (TOUT)1 / n, ans = plogp[n] * n_inv;
    for (int i = 1; i < CNT_DIM; ++i) // i=0, plogp[i]==0
    {
      const auto c = cnt[i];
      if (c)
        ans -= plogp[c] * n_inv;
    }
    output[idy * width + idx] = ans;
  }
}

template <int BLOCK_DIM_X = 128, typename TIN = uint8_t, typename TOUT = float,
          typename TCNT = uint8_t, int CNT_DIM = 256, int CONV_DIM = 2>
static void __global__ __launch_bounds__(BLOCK_DIM_X)
    entropy_with_low_occupancy(const int width, const int height,
                               const TIN *input, TOUT *output) {
  const int idy = blockIdx.y * blockDim.y + threadIdx.y,
            idx = blockIdx.x * blockDim.x + threadIdx.x,
            n = (min(idx, 2) + 1 + min(width - idx, 2)) *
                (min(idy, 2) + 1 + min(height - idy, 2));
  TCNT __shared__ cnt[CNT_DIM][BLOCK_DIM_X];
  TOUT __shared__ plogp[(2 * CONV_DIM + 1) * (2 * CONV_DIM + 1) + 1];
  for (int i = threadIdx.x; i < (2 * CONV_DIM + 1) * (2 * CONV_DIM + 1) + 1;
       i += BLOCK_DIM_X)
    plogp[i] = i ? (TOUT)i * log((TOUT)i) : 0;
  __syncthreads();
  if (idy < height && idx < width) {
    for (int i = 0; i < CNT_DIM; ++i)
      cnt[i][threadIdx.x] = 0;
    for (int offsety = -CONV_DIM; offsety <= CONV_DIM; ++offsety) {
      const int py = idy + offsety;
      if (0 <= py && py < height)
        for (int offsetx = -CONV_DIM; offsetx <= CONV_DIM; ++offsetx) {
          const int px = idx + offsetx;
          if (0 <= px && px < width)
            ++cnt[input[py * width + px]][threadIdx.x];
        }
    }
    TOUT n_inv = (TOUT)1 / n, ans = plogp[n] * n_inv;
    for (int i = 1; i < CNT_DIM; ++i) // i=0, plogp[i]==0
    {
      const auto c = cnt[i][threadIdx.x];
      if (c)
        ans -= plogp[c] * n_inv;
    }
    output[idy * width + idx] = ans;
  }
}
void WuK_Timer(const char *tag, const std::function<void()> &kernel,
               int test_time = 1) {
  float min_time = 9e99;
  for (int i = 0; i < test_time; ++i) {
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg);
    kernel();
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, beg, end);
    min_time = std::min(min_time, elapsed_time);
    std::printf("[%s] iter %d: %f ms elapsed, %f ms min.\n", tag, i,
                elapsed_time, min_time);
  }
}
template <typename T> struct wk_vector {
  T *device_data;
  std::vector<T> host_data;
  wk_vector(size_t n) : host_data(n) {
    cudaMalloc(&device_data, sizeof(T) * n);
  }
  ~wk_vector() { cudaFree(device_data); }
  void sync_device() {
    cudaMemcpy(device_data, host_data.data(), sizeof(T) * host_data.size(),
               cudaMemcpyHostToDevice);
  }
  void sync_host() {
    cudaMemcpy(host_data.data(), device_data, sizeof(T) * host_data.size(),
               cudaMemcpyDeviceToHost);
  }
};

template<typename T>
bool check(wk_vector<T> w1,wk_vector<T> w2){
  for(int i=0;i<w1.host_data.size();i++){
    // printf("at %d:%f %f\n",i,w1.host_data[i],w2.host_data[i]);
    if(w1.host_data[i]-w2.host_data[i]>1e-6){
      printf("at %d:%f %f\n",i,w1.host_data[i],w2.host_data[i]);
      printf("FAIL\n");
      return 1;
    }
  }
  printf("PASS\n");
  return 0;
}

int main() {
  srand(8);
  int W = 128, H = 128;
  printf("W:%d N:%d\n",W,H);
  wk_vector<uint8_t> input(W * H);
  for (int i = 0; i < input.host_data.size(); ++i)
    input.host_data[i] = rand() & 255;
  input.sync_device();
  wk_vector<float> output_reg(W * H),output_occ(W * H);
  const int BLOCK_DIM_X = 128;
  
  WuK_Timer("entropy_with_register_spiling", [&] {
    entropy_with_register_spiling<BLOCK_DIM_X>
        <<<dim3((W + BLOCK_DIM_X - 1) / BLOCK_DIM_X, H, 1), BLOCK_DIM_X>>>(
            W, H, input.device_data, output_reg.device_data);
  },1);
  
  
  WuK_Timer("entropy_with_low_occupancy", [&] {
    entropy_with_low_occupancy<BLOCK_DIM_X>
        <<<dim3((W + BLOCK_DIM_X - 1) / BLOCK_DIM_X, H, 1), BLOCK_DIM_X>>>(
            W, H, input.device_data, output_occ.device_data);
  },1);
  
  output_reg.sync_host();
  output_occ.sync_host();

  return check<float>(output_occ,output_reg);
}
