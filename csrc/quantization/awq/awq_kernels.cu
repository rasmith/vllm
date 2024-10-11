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

using float16_t = _Float16;
using float32_t = float;
using float16x4_t = float16_t __attribute__((ext_vector_type(4)));
using float32x4_t = float32_t __attribute__((ext_vector_type(4)));

template <int TILE_WIDTH>
__global__ __launch_bounds__(128) void awq_gemm_mfma_kernel(
    half* a, int* q, int* zeros, half* scales, int size_n, int size_k,
    int size_m, int group_size, int split_k, half* c) {
  __constant__ static const int kReverseAwqLookup[8] = {0, 4, 1, 5, 2, 6, 3, 7};

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int k = blockIdx.z;  // Get k for splitK

  // dim3 = (16, 4) = 64 threads per block, 1 wave

  // NOTE: Even though the block size is (16, 4), each block is responsible
  // for a 16 x 16 tile and has the 16 * 4 threads necessary to do the work.
  // So, below uses TILE_WIDTH.

  // NOTE: The outputs will be (row, col), (row + 1, col), (row + 2, col)
  // and (row + 3, col).

  // The row is the exact row to work on for pulling values for mfma from.
  int row = blockIdx.x * TILE_WIDTH + tx;
  // The output column that will be used.
  int col = blockIdx.y * TILE_WIDTH + tx;

  int num_tiles = CDIV(size_k, TILE_WIDTH);

  __shared__ int z_values[TILE_WIDTH / 8];
  __shared__ half s_values[TILE_WIDTH];
  __shared__ int q_values[TILE_WIDTH][TILE_WIDTH / 8];

  float16x4_t a_frag{0.0, 0.0, 0.0, 0.0};
  float16x4_t b_frag{0.0, 0.0, 0.0, 0.0};
  float32x4_t accumulator{0.0, 0.0, 0.0, 0.0};

  //  Have CDIV(size_k, split_k * TILE_WIDTH) groups of tiles:
  //
  //   --------------------------------------------
  //        group_0      |       group_1     | ...
  //   --------------------------------------------
  //   0 ... split_k - 1 | 0 ... split_k - 1 | ...
  //   --------------------------------------------
  //
  //  and this thread will work on tiles with "tile index"
  //     threadIdx.z + i * split_k
  for (int tile = k; tile < num_tiles; tile += split_k) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // Go down to the current row, and then over to the current k-tile
      // and then get the 4 values starting at:
      //    (row, tile * TILE_WIDTH + 4 * ty)
      // and load them into registers.
      int aj = tile * TILE_WIDTH + 4 * ty + i;
      int ai = row;

      if (ai >= 0 && ai < size_n && aj >= 0 && aj < size_k) {
        a_frag[i] = a[ai * size_k + aj];
      } else {
        a_frag[i] = __ushort_as_half(0);
      }
    }

    if (ty == 0) {
      int zrow = tile * TILE_WIDTH / group_size;
      int zcol = col / 8;
      if (zrow < size_k / group_size && zcol < size_m / 8) {
        z_values[tx / 8] = zeros[zrow * (size_m / 8) + zcol];
      } else {
        z_values[tx / 8] = 0;
      }
    } else if (ty == 1) {
      int srow = tile * TILE_WIDTH / group_size;
      int scol = col;
      if (srow < size_k / group_size && scol < size_m) {
        s_values[tx] = scales[srow * size_m + scol];
      } else {
        s_values[tx] = __ushort_as_half(0);
      }
    } else {
      int qrow = tile * TILE_WIDTH + tx;
      int qcol = blockIdx.y * TILE_WIDTH / 8 + ty % 2;
      if (qrow < size_k && qcol < size_m / 8) {
        q_values[tx][ty % 2] = q[qrow * (size_m / 8) + qcol];
      } else {
        q_values[tx][ty % 2] = 0;
      }
    }

    __syncthreads();

// if (blockIdx.x == 0 && blockIdx.y == 0) {
// printf("(%d, %d) -> a_fragment: [%.3f, %.3f, %.3f, %.3f]\n", threadIdx.x,
// threadIdx.y, __half2float(a_frag[0]), __half2float(a_frag[1]),
//__half2float(a_frag[2]), __half2float(a_frag[3]));
//}
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // Go down to the current k-tile and then get the 4 values starting at:
      //    (tile * TILE_WIDTH + 4 * ty, col)
      // and load them into registers.

      // Although, really we're just getting the values from LDS.

      int bj = col;
      int bi = tile * TILE_WIDTH + 4 * ty + i;
      int q_value = q_values[4 * ty + i][tx / 8];

      int z_value = z_values[tx / 8];
      int shift = kReverseAwqLookup[bj % 8] * 4;
      int b_int4 = (q_value >> shift) & 0xF;
      int z_int4 = (z_value >> shift) & 0xF;
      half scale = s_values[tx];

      // Since AWQ quantized, actually need to dequantize first.
      b_frag[i] = __int2half_rn(b_int4 - z_int4) * scale;
    }
    // if (blockIdx.x == 0 && blockIdx.y == 0) {
    // printf("[%d] (%d, %d) -> b_fragment: [%.3f, %.3f, %.3f, %.3f]\n", col,
    // threadIdx.x, threadIdx.y, __half2float(b_frag[0]),
    //__half2float(b_frag[1]), __half2float(b_frag[2]),
    //__half2float(b_frag[3]));
    //}

    accumulator = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag,
                                                        accumulator, 0, 0, 0);
  }

// printf("[%d, %d] %.3f %.3f %.3f %.3f\n", tx, ty,
// __half2float(accumulator[0]),
//__half2float(accumulator[1]), __half2float(accumulator[2]),
//__half2float(accumulator[3]));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    // Starting row is (row / TILE_WIDTH) * TILE_WIDTH for this c-tile.
    // So go to ((row / TILE_WIDTH) * TILE_WIDTH + 4 * ty, blockIdx.y *
    // TILE_WIDTH + tx) and start writing values down the column.
    int cj = col;
    int ci = (row / TILE_WIDTH) * TILE_WIDTH + ty * 4 + i;
    if (ci >= 0 && ci < size_n && cj >= 0 && cj < size_m) {
      c[k * size_m * size_n + ci * size_m + cj] = accumulator[i];
    }
  }
  __syncthreads();
}

template <int TILE_WIDTH>
__global__ void awq_gemm_kernel(half* a, int* q, int* zeros, half* scales,
                                int size_n, int size_k, int size_m,
                                int group_size, int split_k, half* c) {
  static const int kReverseAwqLookup[8] = {0, 4, 1, 5, 2, 6, 3, 7};
  float output = 0.0f;
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int k = blockIdx.z;

  __shared__ half a_tile[TILE_WIDTH][TILE_WIDTH];
  __shared__ half b_tile[TILE_WIDTH][TILE_WIDTH];

  int tile_start = 0;
  int tile_end = CDIV(size_k, TILE_WIDTH);
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ay = row;
  int bx = col;
  int num_tiles = CDIV(size_k, TILE_WIDTH);
  // for (int tile = tile_start; tile < tile_end; ++tile) {
  for (int tile = k; tile < num_tiles; tile += split_k) {
    int ax = tile * TILE_WIDTH + tx;
    if (ay < size_n && ax < size_k) {
      a_tile[ty][tx] = a[ay * size_k + ax];
    } else {
      a_tile[ty][tx] = __ushort_as_half(0);
    }

    int by = tile * TILE_WIDTH + ty;

    if (by < size_k && bx < size_m) {
      half scale = scales[(by / group_size) * size_m + bx];
      int q_value = q[(by / group_size) * (size_m / 8) + (bx / 8)];
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
    // c[row * size_m + col] = __float2half_rn(output);
    c[k * size_n * size_m + row * size_m + col] = __float2half_rn(output);
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
  at::Tensor result_tensor = torch::zeros({splitK, size_n, size_m}, options);

  half* input = reinterpret_cast<half*>(input_tensor.data_ptr<at::Half>());
  int* qweights = reinterpret_cast<int*>(qweight_tensor.data_ptr<int>());
  half* scales = reinterpret_cast<half*>(scales_tensor.data_ptr<at::Half>());
  int* qzeros = reinterpret_cast<int*>(qzeros_tensor.data_ptr<int>());
  half* c = reinterpret_cast<half*>(result_tensor.data_ptr<at::Half>());

  constexpr uint32_t kNumBlocksY = 1;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr bool kUseMfma = true;
  constexpr int kTileWidth = (kUseMfma ? 16 : 32);
  if (kUseMfma) {
    dim3 threads_per_block(16, 4);
    // dim3 blocks(num_blocks_x, kNumBlocksY);
    dim3 blocks(CDIV(size_n, kTileWidth), CDIV(size_m, kTileWidth),
                splitK);  // CDIV(size_m, kTileWidth));
    // std::cout << "threads_per_block.x = " << threads_per_block.x
    //<< ", threads_per_block.y = " << threads_per_block.y << "\n";
    // std::cout << "blocks.x = " << blocks.x
    //<< ", blocks.y = " << blocks.y << "\n";

    awq_gemm_mfma_kernel<kTileWidth><<<blocks, threads_per_block, 0, stream>>>(
        input, qweights, qzeros, scales, size_n, size_k, size_m, group_size,
        splitK, c);
  } else {
    dim3 threads_per_block(kTileWidth, kTileWidth);
    // dim3 blocks(num_blocks_x, kNumBlocksY);
    dim3 blocks(CDIV(size_m, kTileWidth), CDIV(size_n, kTileWidth),
                splitK);  // CDIV(size_m, kTileWidth));
    // std::cout << "threads_per_block.x = " << threads_per_block.x
    //<< ", threads_per_block.y = " << threads_per_block.y
    //<< ", threads_per_block.z = " << threads_per_block.z << "\n";
    // std::cout << "blocks.x = " << blocks.x << ", blocks.y = " << blocks.y
    //<< "\n";
    // std::cout << "Launching kernel...\n";

    awq_gemm_kernel<kTileWidth><<<blocks, threads_per_block, 0, stream>>>(
        input, qweights, qzeros, scales, size_n, size_k, size_m, group_size,
        splitK, c);
  }
  return result_tensor.sum(0);
}
