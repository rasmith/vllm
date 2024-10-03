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
                                int size_m, int size_k, int size_n,
                                int group_size, int split_k, half* c) {
  static const int kReverseAwqLookup[8] = {0, 4, 1, 5, 2, 6, 3, 7};
  // half output = __ushort_as_half(0);
  float output = 0.0f;
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  //if (row == 0) {
  //printf("row = %d, col = %d, size_m = %d, size_n = %d\n",
      //row, col, size_m, size_n);
  //}

  __shared__ half a_tile[TILE_WIDTH][TILE_WIDTH];
  __shared__ half b_tile[TILE_WIDTH][TILE_WIDTH];

  int tile_start = 0;
  int tile_end = CDIV(size_k, TILE_WIDTH);
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  for (int tile = tile_start; tile < tile_end; ++tile) {
    int ax = tile * TILE_WIDTH + tx;
    int ay = row;
    if (ay < size_m && ax < size_k) {
      a_tile[ty][tx] = a[ay * size_k + ax];
    } else {
      a_tile[ty][tx] = __ushort_as_half(0);
    }

    int bx = col;
    int by = tile * TILE_WIDTH + ty;
    if (by < size_k && bx < size_n) {
      int q_value = q[by * (size_n / 8) + (bx / 8)];
      int z_value = zeros[(by / group_size) * (size_n / 8) + (bx / 8)];
      half scale = scales[(by / group_size) * size_n + bx];
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

  if (row < size_m && col < size_n) {
    c[row * size_n + col] = __float2half_rn(output);
  }
}

// threadDim.x - c-tiles
// threadDim.y - split-k blocks
// template <int TILE_SIZE_M, int TILE_SIZE_K, int TILE_SIZE_N>
//__global__ void awq_gemm_kernel(half* a, int* q, int* zeros, half* scales,
// int size_m, int size_k, int size_n,
// int group_size, int split_k, half* c) {
// int tid_c = threadIdx.x + blockIdx.x * blockDim.x;
// int tid_k = threadIdx.y + blockIdx.y * blockDim.y;
// static const int kReverseAwqLookup[8] = {0, 4, 1, 5, 2, 6, 3, 7};

//// (ii, jj) is the tile index in C
// int num_tiles_per_row =
// CDIV(size_n, TILE_SIZE_N);  // Number of tiles per row.
// int ii = tid_c / num_tiles_per_row;
// int jj = tid_c % num_tiles_per_row;
// int num_k_tiles_per_thread = CDIV(size_k, split_k * TILE_SIZE_K);
// int start_tile = tid_k * num_k_tiles_per_thread;
// int end_tile = start_tile + num_k_tiles_per_thread;
// for (int i = 0; i < TILE_SIZE_M; ++i) {
// int i_a = ii * TILE_SIZE_M + i;
// int i_c = i_a;
// for (int j = 0; j < TILE_SIZE_N; ++j) {
// int j_b = jj * TILE_SIZE_N + j;
// int j_c = j_b;
// half c_ij = __int2half_rn(0);
// for (int kk = start_tile; kk < end_tile; ++kk) {
// half c_ij_kk = __int2half_rn(0);
// for (int k = 0; k < TILE_SIZE_K; ++k) {
// int k_a = kk * TILE_SIZE_K + k;
// int k_b = k_a;
// if (i_a >= 0 && i_a < size_m && k_b >= 0 && k_b < size_k &&
// j_b >= 0 && j_b < size_n) {
// half a_ik = a[i_a * size_k + k_a];
// int q_value = q[k_b * (size_n / 8) + (j_b / 8)];
// int z_value = zeros[(k_b / group_size) * (size_n / 8) + (j_b / 8)];
// half scale = scales[(k_b / group_size) * size_n + j_b];
// int shift = kReverseAwqLookup[j_b % 8] * 4;
// int b_int4 = (q_value >> shift) & 0xF;
// int z_int4 = (z_value >> shift) & 0xF;
// half b_kj = __int2half_rn(b_int4 - z_int4) * scale;
// c_ij += a_ik * b_kj;
//}
//}
//}
// int output_index = tid_k * size_n * size_m + i_c * size_n + j_c;
// if (i_c >= 0 && i_c < size_m && j_c >= 0 && j_c < size_n) {
// c[output_index] = c_ij;
//}
//}
//}
//}

// in_feats: M, IC [float16]
// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
// scaling_factors: IC // G, OC [float16]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
// assume that batch_size < 16 for now

torch::Tensor awq_gemm_test(torch::Tensor input_tensor,
                            torch::Tensor qweight_tensor,
                            torch::Tensor scales_tensor,
                            torch::Tensor qzeros_tensor, int64_t splitK) {
  int size_m = input_tensor.size(0);
  int size_k = qweight_tensor.size(0);
  int size_n = qweight_tensor.size(1) * 8;
  int group_size = qweight_tensor.size(0) / qzeros_tensor.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input_tensor));

  auto options = torch::TensorOptions()
                     .dtype(input_tensor.dtype())
                     .device(input_tensor.device());
  at::Tensor result_tensor = torch::zeros({size_m, size_n}, options);

  half* input = reinterpret_cast<half*>(input_tensor.data_ptr<at::Half>());
  int* qweights = reinterpret_cast<int*>(qweight_tensor.data_ptr<int>());
  half* scales = reinterpret_cast<half*>(scales_tensor.data_ptr<at::Half>());
  int* qzeros = reinterpret_cast<int*>(qzeros_tensor.data_ptr<int>());
  half* c = reinterpret_cast<half*>(result_tensor.data_ptr<at::Half>());

  constexpr int kTileSizeM = 32;
  constexpr int kTileSizeN = 32;
  constexpr int kTileSizeK = 32;
  constexpr uint32_t kTileSize = kTileSizeM * kTileSizeN;
  constexpr uint32_t kThreadsPerBlockX = 32;
  constexpr uint32_t kNumBlocksY = 1;
  constexpr int kTileWidth = 32;

  uint32_t threads_per_block_y = splitK;
  uint32_t num_tiles =
      std::min(1, CDIV(size_n * size_m, kTileSizeN * kTileSizeM));
  uint32_t num_blocks_x = std::min(1, CDIV(num_tiles, kThreadsPerBlockX));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // std::cout << "blocks = (" << num_blocks_x << "," << kNumBlocksY << ")\n";
  //  x dimension processes tiles
  //  y dimension does split-k
  // dim3 threads_per_block(kThreadsPerBlockX, threads_per_block_y);
  dim3 threads_per_block(kTileWidth, kTileWidth);
  // dim3 blocks(num_blocks_x, kNumBlocksY);
  dim3 blocks(CDIV(size_n, kTileWidth), CDIV(size_m, kTileWidth));// CDIV(size_n, kTileWidth));
  //std::cout << "threads_per_block.x = " << threads_per_block.x 
            //<< ", threads_per_block.y = " << threads_per_block.y << "\n";
  //std::cout << "blocks.x = " << blocks.x 
            //<< ", blocks.y = " << blocks.y << "\n";

  // awq_gemm_kernel<kTileSizeM, kTileSizeK, kTileSizeN>
  //<<<blocks, threads_per_block, 0, stream>>>(input, qweights, qzeros,
  // scales, size_m, size_k, size_n,
  // group_size, splitK, c);
  awq_gemm_kernel<kTileWidth><<<blocks, threads_per_block, 0, stream>>>(
      input, qweights, qzeros, scales, size_m, size_k, size_n, group_size,
      splitK, c);
  return result_tensor;
}
