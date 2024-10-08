#include <iostream>
#include <cstdlib>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_common.h>
#include <hip/amd_detail/amd_hip_fp16.h>

#define CHECK_HIP(x)                                                           \
  {                                                                            \
    hipError_t error = (x);                                                    \
    if (error != 0) {                                                          \
      std::cerr << __LINE__ << ": HIP call failed:" #x << " error = " << error \
                << "\n";                                                       \
    }                                                                          \
  }

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

template <typename NumberType>
NumberType* AllocateDeviceVector(size_t size) {
  NumberType* d;
  std::cout << "hipMalloc: " << sizeof(NumberType) * size << " bytes.\n";
  hipError_t error = hipMalloc(&d, sizeof(NumberType) * size);
  if (error != 0) {
    std::cerr << "HIP call failed with error:" << error << "\n";
    return nullptr;
  }
  return d;
}

template <typename NumberType>
class HipArray {
 public:
  HipArray(size_t size)
      : data_(AllocateDeviceVector<NumberType>(size)), size_(size) {}
  ~HipArray() {
    if (!data_) {
      return;
    }
    CHECK_HIP(hipFree(data_));
  }
  NumberType* data() { return data_; }

  void CopyToDevice(const std::vector<NumberType>& v) {
    CHECK_HIP(hipMemcpy(data_, v.data(), v.size() * sizeof(NumberType),
                        hipMemcpyHostToDevice));
  }

  void CopyFromDevice(std::vector<NumberType>& v) {
    v.resize(size_);
    CHECK_HIP(hipMemcpy(v.data(), data_, v.size() * sizeof(NumberType),
                        hipMemcpyDeviceToHost));
  }

 private:
  NumberType* data_;
  size_t size_;
};

template <typename NumberType>
NumberType Convert(int value) {
  return value;
}

template <>
half Convert(int value) {
  float float_value = static_cast<float>(value) / static_cast<float>(RAND_MAX);
  half converted = __float2half_rn(float_value);
  return converted;
}

template <typename NumberType>
NumberType RandomNumber() {
  int value = std::rand();
  NumberType r = Convert<NumberType>(value);
  return r;
}

template <typename NumberType>
void GenerateMatrix(int rows, int cols, std::vector<NumberType>& output) {
  output.resize(rows * cols);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      output[y * cols + x] = RandomNumber<NumberType>();
    }
  }
}

template <typename NumberType>
void Zeros(int rows, int cols, std::vector<NumberType>& output) {
  output.resize(rows * cols);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      output[y * cols + x] = NumberType(0);
    }
  }
}

template <typename NumberType>
void PrintValue(NumberType v) {
  std::cout << v;
}

template <>
void PrintValue(half v) {
  std::cout << __half2float(v);
}

template <typename NumberType>
void PrintRow(int row, int row_padding, int rows, int cols,
              const std::vector<NumberType>& m, int left = 5, int right = 5) {
  if (row > 0) {
    std::string padding(row_padding, ' ');
    std::cout << padding;
  }
  std::cout << "[";
  bool has_valid_bounds = left > 0 && left < cols && right > 0 && right < cols;
  if (!has_valid_bounds) {
    left = cols;
    right = -1;
  }
  for (int x = 0; x < std::min(left, cols); ++x) {
    PrintValue(m[cols * row + x]);
    if (x == cols - 1 && row < rows - 1) {
      std::cout << "]\n";
    } else if (x == cols - 1 && row == rows - 1) {
      std::cout << "]]\n";
    } else {
      std::cout << " ";
    }
  }
  if (left > 0 && left < cols && right > 0 && right < cols) {
    std::cout << " ... ";
    for (int x = cols - right; x < cols; ++x) {
      PrintValue(m[cols * row + x]);
      if (x == cols - 1 && row < rows - 1) {
        std::cout << "]\n";
      } else if (x == cols - 1 && row == rows - 1) {
        std::cout << "]]\n";
      } else {
        std::cout << " ";
      }
    }
  }
}

template <typename NumberType>
void PrintMatrix(const std::string& label, int rows, int cols,
                 const std::vector<NumberType>& m, int top_rows = 2,
                 int bottom_rows = 2, int left_rows = 5, int right_rows = 5) {
  std::cout << label << "=";
  std::cout << "[";
  int padding = label.size() + 2;
  if ((top_rows < 0 || top_rows > rows) ||
      (bottom_rows < 0 || bottom_rows > rows)) {
    top_rows = rows;
  }

  for (int y = 0; y < top_rows; ++y) {
    PrintRow(y, padding, rows, cols, m, left_rows, right_rows);
  }

  if (top_rows > 0 && top_rows < rows && bottom_rows > 0 &&
      bottom_rows < rows) {
    std::string padding_str(padding, ' ');
    std::cout << padding_str << ".\n";
    std::cout << padding_str << ".\n";
    std::cout << padding_str << ".\n\n";
    for (int y = 0; y < bottom_rows; ++y) {
      PrintRow(rows - bottom_rows - y - 1, padding, rows, cols, m, left_rows,
               right_rows);
    }
  }
}

void DequantizeMatrix(int* q, int* zeros, half* scales, int size_k, int size_m,
                      int group_size, std::vector<half>& b) {
  static constexpr std::array<int, 8> reverse_awq_lut = {0, 4, 1, 5,
                                                         2, 6, 3, 7};
  b.resize(size_k * size_m);
  for (int y = 0; y < size_k; ++y) {
    for (int x = 0; x < size_m; ++x) {
      int q_value = q[y * (size_m / 8) + (x / 8)];
      int z_value = zeros[(y / group_size) * (size_m / 8) + (x / 8)];
      int shift = reverse_awq_lut[x % 8] * 4;
      int q_int4 = (q_value >> shift) & 0xF;
      int z_int4 = (z_value >> shift) & 0xF;
      half scale = scales[(y / group_size) * size_m + x];
      half b_value = (q_int4 - z_int4) * __half2float(scale);
      b[y * size_m + x] = b_value;
    }
  }
}

// a  - N x K
// b -  K x (M // 8)
// c -  N x M
// zeros - (K // G) x (M // 8)
// scales - (K // G) x M
// G - group size
void awq_gemm_cpu(half* a, int* q, int* zeros, half* scales, int N, int K,
                  int M, int G, half* c) {
  static constexpr std::array<int, 8> reverse_awq_lut = {0, 4, 1, 5,
                                                         2, 6, 3, 7};
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        half a_ik = a[i * K + k];
        int q_value = q[k * (M / 8) + (j / 8)];
        int z_value = zeros[(k / G) * (M / 8) + (j / 8)];
        int shift = reverse_awq_lut[j % 8] * 4;
        int q_int4 = (q_value >> shift) & 0xF;
        int z_int4 = (z_value >> shift) & 0xF;
        half scale = scales[(k / G) * M + j];
        half b_kj = (q_int4 - z_int4) * __half2float(scale);
        float a_ik32 = __half2float(a_ik);
        float b_kj32 = __half2float(b_kj);
        sum += a_ik32 * b_kj32;
      }
      c[i * M + j] = __float2half_rn(sum);
    }
  }
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

using float16_t = _Float16;
using float32_t = float;
using float16x4_t = float16_t __attribute__((ext_vector_type(4)));
using float32x4_t = float32_t __attribute__((ext_vector_type(4)));

#define WAVE_SIZE 64

template <int TILE_WIDTH>
__global__ void awq_gemm_mfma_kernel(half* a, int* q, int* zeros, half* scales,
                                     int size_n, int size_k, int size_m,
                                     int group_size, int split_k, half* c) {
  static const int kReverseAwqLookup[8] = {0, 4, 1, 5, 2, 6, 3, 7};
  float output = 0.0f;
  // dim3 = (16, 4) = 64 threads per block, 1 wave
  int row = blockIdx.x * blockDim.x + threadIdx.x;  // blockDim.x = TILE_WIDTH
  int col = blockIdx.y * blockDim.y;

  int tile_start = 0;
  int tile_end = CDIV(size_k, TILE_WIDTH);
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  float16x4_t a_frag{0.0, 0.0, 0.0, 0.0};
  float16x4_t b_frag{0.0, 0.0, 0.0, 0.0};
  float32x4_t accumulator{0.0, 0.0, 0.0, 0.0};

  for (int tile = tile_start; tile < tile_end; ++tile) {
    for (int i = 0; i < 4; ++i) {
      // Go down to the current row, and then over to the current k-tile
      // and then get the 4 values starting at:
      //    (row, tile * TILE_WIDTH + 4 * ty)
      // and load them into registers.
      int ax = tile * TILE_WIDTH + 4 * ty + i;
      int ay = row;
      a_frag[i] = a[ay * size_k + ax];
    }

    //if (blockIdx.x == 0 && blockIdx.y == 0) {
      //printf("(%d, %d) -> a_fragment: [%f, %f, %f, %f]\n", threadIdx.x,
             //threadIdx.y, __half2float(a_frag[0]), __half2float(a_frag[1]),
             //__half2float(a_frag[2]), __half2float(a_frag[3]));
    //}G
    for (int i = 0; i < 4; ++i) {
      // Go down to the current k-tile and then get the 4 values starting at:
      //    (tile * TILE_WIDTH + 4 * ty, blockIdx.y * TILE_WIDTH + tx)
      // and load them into registers.
      int bj = blockIdx.y * TILE_WIDTH + tx;
      int bi = tile * TILE_WIDTH + 4 * ty + i;

      // Since AWQ quantized, actually need to dequantize first.
      int q_value = q[bi * (size_m / 8) + (bj / 8)];
      int z_value = zeros[(bi / group_size) * (size_m / 8) + (bj / 8)];
      int shift = kReverseAwqLookup[bj % 8] * 4;
      int b_int4 = (q_value >> shift) & 0xF;
      int z_int4 = (z_value >> shift) & 0xF;
      half scale = scales[(bi / group_size) * size_m + bj];

      b_frag[i] = __int2half_rn(b_int4 - z_int4) * scale;
    }
    //if (blockIdx.x == 0 && blockIdx.y == 0) {
      //printf("[%d] (%d, %d) -> b_fragment: [%f, %f, %f, %f]\n", col, threadIdx.x,
             //threadIdx.y, __half2float(b_frag[0]), __half2float(b_frag[1]),
             //__half2float(b_frag[2]), __half2float(b_frag[3]));
    //}

    accumulator = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag,
                                                        accumulator, 0, 0, 0);
  }

  for (int i = 0; i < 4; ++i) {
    // Starting row is (row / TILE_WIDTH) * TILE_WIDTH for this c-tile.
    // So go to ((row / TILE_WIDTH) * TILE_WIDTH + 4 * ty, blockIdx.y * TILE_WIDTH + tx)
    // and start writing values down the column.
    int cj = blockIdx.y * TILE_WIDTH + tx;
    int ci = (row / TILE_WIDTH) * TILE_WIDTH + ty * 4 + i;
    c[ci * size_m + cj] = accumulator[i];
  }
}

int main(int argc, char** argv) {
  constexpr uint32_t kMatrixSizeN = 128;
  constexpr uint32_t kMatrixSizeM = 128;
  constexpr uint32_t kMatrixSizeK = 128;
  constexpr uint32_t kGroupSize = 16;
  constexpr uint32_t kSplitK = 1;
  std::vector<half> a;
  std::vector<int> q;
  std::vector<half> c;
  std::vector<int> zeros;
  std::vector<half> scales;

  GenerateMatrix(kMatrixSizeN, kMatrixSizeK, a);
  GenerateMatrix(kMatrixSizeK, kMatrixSizeM / 8, q);
  Zeros(kMatrixSizeN, kMatrixSizeM, c);
  GenerateMatrix(kMatrixSizeK / kGroupSize, kMatrixSizeM / 8, zeros);
  GenerateMatrix(kMatrixSizeK / kGroupSize, kMatrixSizeM, scales);

  PrintMatrix("a", kMatrixSizeN, kMatrixSizeK, a, -1, -1, -1, -1);

  std::vector<half> b;
  DequantizeMatrix(q.data(), zeros.data(), scales.data(), kMatrixSizeK,
                   kMatrixSizeM, kGroupSize, b);
  PrintMatrix("b", kMatrixSizeK, kMatrixSizeM, b, -1, -1, -1, -1);

  auto start_cpu = std::chrono::system_clock::now();
  awq_gemm_cpu(a.data(), q.data(), zeros.data(), scales.data(), kMatrixSizeN,
               kMatrixSizeK, kMatrixSizeM, kGroupSize, c.data());
  PrintMatrix("c", kMatrixSizeN, kMatrixSizeM, c);
  auto end_cpu = std::chrono::system_clock::now();
  std::cout << "CPU time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu -
                                                                     start_cpu)
                   .count()
            << " ms\n";

  HipArray<half> hip_array_a(a.size());
  HipArray<int> hip_array_q(q.size());
  std::cout << "Make HipArray for c\n";
  HipArray<half> hip_array_c(c.size() +
                             2);  // add 4 bytes to end for AtomicAdd.
  HipArray<int> hip_array_zeros(zeros.size());
  HipArray<half> hip_array_scales(scales.size());

  std::vector<half> c_gpu;
  Zeros(kMatrixSizeM * kMatrixSizeN * kSplitK, 1, c_gpu);
  HipArray<half> hip_array_c_gpu(c_gpu.size());

  hip_array_a.CopyToDevice(a);
  hip_array_q.CopyToDevice(q);
  hip_array_c_gpu.CopyToDevice(c_gpu);
  hip_array_zeros.CopyToDevice(zeros);
  hip_array_scales.CopyToDevice(scales);

  // x dimension processes tiles
  // y dimension does split-k

  // Attempt #2:
  // x dimension - cols
  // y dimension - rows
  bool useMfMa = true;

  if (useMfMa) {
    constexpr uint16_t kTileWidth = 16;
    constexpr uint16_t kTileSizeN = 16;
    constexpr uint16_t kTileSizeM = 16;
    constexpr uint16_t kTileSizeK = 16;
    constexpr uint32_t kTileSize = kTileSizeN * kTileSizeM;
    constexpr uint32_t kThreadsPerBlockX = kTileSizeM;
    constexpr uint32_t kThreadsPerBlockY = kTileSizeN;
    constexpr uint32_t kNumTiles =
        CDIV(kMatrixSizeM * kMatrixSizeN, kTileSizeM * kTileSizeN);

    constexpr uint32_t kNumBlocksX = CDIV(kMatrixSizeM, kTileWidth);
    constexpr uint32_t kNumBlocksY = CDIV(kMatrixSizeN, kTileWidth);
    dim3 threads_per_block(16, 4);
    dim3 blocks(CDIV(kMatrixSizeM, kTileWidth), CDIV(kMatrixSizeN, kTileWidth));

    std::cout << "A = " << kMatrixSizeM << " x " << kMatrixSizeK << "\n";
    std::cout << "Q = " << kMatrixSizeK / 8 << " x " << kMatrixSizeN << "\n";
    std::cout << "C = " << kMatrixSizeM << " x " << kMatrixSizeN << "\n";
    std::cout << "num_tiles = " << CDIV(kMatrixSizeN * kMatrixSizeM, 16 * 16)
              << "\n";
    std::cout << "kTileSizeN = " << kTileSizeN << " kTileSizeK = " << kTileSizeK
              << " kTileSizeM = " << kTileSizeM << "\n";
    std::cout << "threads_per_block = (" << kThreadsPerBlockX << ","
              << kThreadsPerBlockY << ")\n";
    std::cout << "blocks = (" << kNumBlocksX << "," << kNumBlocksY << ")\n";

    auto start = std::chrono::system_clock::now();
    awq_gemm_mfma_kernel<kTileWidth><<<blocks, threads_per_block>>>(
        hip_array_a.data(), hip_array_q.data(), hip_array_zeros.data(),
        hip_array_scales.data(), kMatrixSizeN, kMatrixSizeK, kMatrixSizeM,
        kGroupSize, kSplitK, hip_array_c.data());
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::system_clock::now();
    std::cout << "GPU time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << " ms \n";
  } else {
    constexpr uint32_t kTileWidth = 32;
    constexpr uint32_t kTileSizeN = 32;
    constexpr uint32_t kTileSizeM = 32;
    constexpr uint32_t kTileSizeK = 32;
    constexpr uint32_t kTileSize = kTileSizeN * kTileSizeM;
    // constexpr uint32_t kThreadsPerBlockX = 8;
    // constexpr uint32_t kThreadsPerBlockY = kSplitK;
    constexpr uint32_t kThreadsPerBlockX = kTileSizeM;
    constexpr uint32_t kThreadsPerBlockY = kTileSizeN;
    constexpr uint32_t kNumTiles =
        CDIV(kMatrixSizeM * kMatrixSizeN, kTileSizeM * kTileSizeN);
    // constexpr uint32_t kNumBlocksX = cdiv_const(kNumTiles,
    // kThreadsPerBlockX); constexpr uint32_t kNumBlocksY = 1; constexpr
    // uint32_t kNumBlocksX = cdiv_const(kMatrixSizeM, kTileSizeM); constexpr
    // uint32_t kNumBlocksY = cdiv_const(kMatrixSizeN, kTileSizeN);

    constexpr uint32_t kNumBlocksX = CDIV(kMatrixSizeM, kTileWidth);
    constexpr uint32_t kNumBlocksY = CDIV(kMatrixSizeN, kTileWidth);
    dim3 threads_per_block(kThreadsPerBlockX, kThreadsPerBlockY);
    dim3 blocks(kNumBlocksX, kNumBlocksY);

    std::cout << "A = " << kMatrixSizeM << " x " << kMatrixSizeK << "\n";
    std::cout << "Q = " << kMatrixSizeK / 8 << " x " << kMatrixSizeN << "\n";
    std::cout << "C = " << kMatrixSizeM << " x " << kMatrixSizeN << "\n";
    std::cout << "num_tiles = " << kNumTiles << "\n";
    std::cout << "kTileSizeN = " << kTileSizeN << " kTileSizeK = " << kTileSizeK
              << " kTileSizeM = " << kTileSizeM << "\n";
    std::cout << "threads_per_block = (" << kThreadsPerBlockX << ","
              << kThreadsPerBlockY << ")\n";
    std::cout << "blocks = (" << kNumBlocksX << "," << kNumBlocksY << ")\n";

    auto start = std::chrono::system_clock::now();
    awq_gemm_kernel<kTileWidth><<<blocks, threads_per_block>>>(
        hip_array_a.data(), hip_array_q.data(), hip_array_zeros.data(),
        hip_array_scales.data(), kMatrixSizeN, kMatrixSizeK, kMatrixSizeM,
        kGroupSize, kSplitK, hip_array_c.data());
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::system_clock::now();
    std::cout << "GPU time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << " ms \n";
  }
  hip_array_c.CopyFromDevice(c_gpu);
  // std::vector<half> result;
  // Zeros(kMatrixSizeM, kMatrixSizeN, result);
  // for (int i = 0; i < kMatrixSizeN; ++i) {
  // for (int j = 0; j < kMatrixSizeM; ++j) {
  // for (int k = 0; k < kSplitK; ++k) {
  // result[i * kMatrixSizeM + j] = __float2half_rn(
  //__half2float(result[i * kMatrixSizeM + j]) +
  //__half2float(
  // c_gpu[k * kMatrixSizeM * kMatrixSizeN + i * kMatrixSizeM + j]));
  //}
  //}
  //}

  CHECK_HIP(hipDeviceSynchronize());

  PrintMatrix("c_gpu", kMatrixSizeN, kMatrixSizeM, c_gpu);
  return 0;
}
