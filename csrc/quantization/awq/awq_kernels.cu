#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>

#define CDIV(A, B) (((A) + (B)-1) / (B))

__device__ void AtomicAdd(half* address, half val) {
  uint32_t* address_as_ui =
      (uint32_t*)((size_t)address - ((size_t)address & 2));
  uint32_t old = *address_as_ui;
  uint32_t assumed;

  do {
    assumed = old;
    __half_raw hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    half temp = __hadd(hsum, val);
    hsum = __half_raw(temp);
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}

template <int TILE_WIDTH>
__global__ void awq_gemm_kernel(half* a, int* q, int* zeros, half* scales,
                                int size_n, int size_k, int size_m,
                                int group_size, int split_k, half* c) {
  static const int kReverseAwqLookup[8] = {0, 4, 1, 5, 2, 6, 3, 7};
  float output = 0.0f;
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  __shared__ half a_tile[TILE_WIDTH][TILE_WIDTH];
  __shared__ half b_tile[TILE_WIDTH][TILE_WIDTH];
  __shared__ int q_values[TILE_WIDTH][TILE_WIDTH / 8];
  __shared__ int z_values[TILE_WIDTH][TILE_WIDTH / 8];

  int tile_start = 0;
  int tile_end = CDIV(size_k, TILE_WIDTH);
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ay = row;
  int bx = col;
  for (int tile = tile_start; tile < tile_end; ++tile) {
    int ax = tile * TILE_WIDTH + tx;
    if (ay < size_n && ax < size_k) {
      a_tile[ty][tx] = a[ay * size_k + ax];
    } else {
      a_tile[ty][tx] = __ushort_as_half(0);
    }

    int by = tile * TILE_WIDTH + ty;

    if (by < size_k && bx < size_m) {
      half scale = scales[(by / group_size) * size_m + bx];
      int q_value = q[by * (size_m / 8) + (bx / 8)];
      int z_value = zeros[(by / group_size) * (size_m / 8) + (bx / 8)];
      int shift = kReverseAwqLookup[bx % 8] * 4;
      int b_int4 = (q_value >> shift) & 0xF;
      int z_int4 = (z_value >> shift) & 0xF;
      b_tile[ty][tx] = __int2half_rn(b_int4 - z_int4) * scale;
    } else {
      b_tile[ty][tx] = __ushort_as_half(0);
    }

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      output += __half2float(a_tile[ty][k]) * __half2float(b_tile[k][tx]);
    }
    __syncthreads();
  }

  if (row < size_n && col < size_m) {
    c[row * size_m + col] = __float2half_rn(output);
  }
}


// in_feats: M, IC [float16]
// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
// scaling_factors: IC // G, OC [float16]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
// assume that batch_size < 16 for now

torch::Tensor awq_gemm_test(torch::Tensor input_tensor,
                            torch::Tensor qweight_tensor,
                            torch::Tensor scales_tensor,
                            torch::Tensor qzeros_tensor, int64_t splitK) {
  int size_n = input_tensor.size(0);
  int size_k = qweight_tensor.size(0);
  int size_m = qweight_tensor.size(1) * 8;
  int group_size = qweight_tensor.size(0) / qzeros_tensor.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input_tensor));

  auto options = torch::TensorOptions()
                     .dtype(input_tensor.dtype())
                     .device(input_tensor.device());
  at::Tensor result_tensor = torch::zeros({size_n, size_m}, options);

  half* input = reinterpret_cast<half*>(input_tensor.data_ptr<at::Half>());
  int* qweights = reinterpret_cast<int*>(qweight_tensor.data_ptr<int>());
  half* scales = reinterpret_cast<half*>(scales_tensor.data_ptr<at::Half>());
  int* qzeros = reinterpret_cast<int*>(qzeros_tensor.data_ptr<int>());
  half* c = reinterpret_cast<half*>(result_tensor.data_ptr<at::Half>());

  constexpr uint32_t kThreadsPerBlockX = 32;
  constexpr uint32_t kNumBlocksY = 1;
  constexpr int kTileWidth = 32;

  uint32_t threads_per_block_y = splitK;
  uint32_t num_tiles =
      std::min(1, CDIV(size_m * size_n, kTileWidth * kTileWidth));
  uint32_t num_blocks_x = std::min(1, CDIV(num_tiles, kThreadsPerBlockX));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // std::cout << "blocks = (" << num_blocks_x << "," << kNumBlocksY << ")\n";
  //  x dimension processes tiles
  //  y dimension does split-k
  // dim3 threads_per_block(kThreadsPerBlockX, threads_per_block_y);
  dim3 threads_per_block(kTileWidth, kTileWidth);
  // dim3 blocks(num_blocks_x, kNumBlocksY);
  dim3 blocks(CDIV(size_m, kTileWidth),
              CDIV(size_n, kTileWidth));  // CDIV(size_m, kTileWidth));
  // std::cout << "threads_per_block.x = " << threads_per_block.x
  //<< ", threads_per_block.y = " << threads_per_block.y << "\n";
  // std::cout << "blocks.x = " << blocks.x
  //<< ", blocks.y = " << blocks.y << "\n";

  awq_gemm_kernel<kTileWidth><<<blocks, threads_per_block, 0, stream>>>(
      input, qweights, qzeros, scales, size_n, size_k, size_m, group_size,
      splitK, c);
  return result_tensor;
}
