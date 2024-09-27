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

__device__ half AtomicAdd(half* address, half value) {
  int* address_as_int = reinterpret_cast<int*>(address);
  //printf("address = %p, address_as_int = %p\n", address, address_as_int);
  int old = *address_as_int;
  half old_half = *address;
  half new_value = 0;
  int assumed = 0;
  int update = 0;
  int lower = 0;
  int upper = 0;
  do {
    assumed = old;
    new_value = value + old_half;
    lower = assumed & 0xFF;
    *(reinterpret_cast<uint16_t*>(&upper) + 1) =
        *reinterpret_cast<uint16_t*>(&new_value);
    update = upper << 16 & lower;
    old = atomicCAS(address_as_int, assumed, update);
    *address_as_int = update;
    old_half = *reinterpret_cast<half*>(&old);
  } while (assumed != old);
  return old_half;
}

template <typename NumberType>
constexpr NumberType cdiv_const(NumberType a, NumberType b) {
  return (a + b - 1) / b;
}

template <typename NumberType>
__host__ __device__ NumberType cdiv(NumberType a, NumberType b) {
  return (a + b - 1) / b;
}

template <typename NumberType>
NumberType* AllocateDeviceVector(size_t size) {
  NumberType* d;
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
      output.push_back(NumberType(0));
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
void PrintMatrix(const std::string& label, int rows, int cols,
                 const std::vector<NumberType>& m) {
  std::cout << label << "=";
  std::cout << "[";
  for (int y = 0; y < rows; ++y) {
    if (y > 0) {
      std::string padding(label.size() + 2, ' ');
      std::cout << padding;
    }
    std::cout << "[";
    for (int x = 0; x < cols; ++x) {
      PrintValue(m[cols * y + x]);
      if (x == cols - 1 && y < rows - 1) {
        std::cout << "]\n";
      } else if (x == cols - 1 && y == rows - 1) {
        std::cout << "]]\n";
      } else {
        std::cout << " ";
      }
    }
  }
}

half dequantize_cpu(int i, int j, int* q, int K, int N, int G, int* zeros,
                    half* scales) {
  static constexpr std::array<int, 8> reverse_awq_lut = {0, 4, 1, 5,
                                                         2, 6, 3, 7};
  int packed_values = q[i * (N / 8) + (j / 8)];
  uint8_t q_value = packed_values >> reverse_awq_lut[N % 8] & 0xF;
  half scale = scales[(i / G) * N + j];
  packed_values = zeros[(i / G) * (N / 8) + (j / 8)];
  uint8_t z_value = packed_values >> reverse_awq_lut[N % 8] & 0xF;
  float float_value = (q_value - z_value) * __half2float(scale);
  return __float2half_rn(float_value);
}

// a  - M x K
// b -  K x (N // 8)
// c -  M x N
// zeros - (K // G) x (N // 8)
// scales - (K // G) x N
// G - group size
void awq_gemm_cpu(half* a, int* q, int* zeros, half* scales, int M, int K,
                  int N, int G, half* c) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        half a_ik = a[i * K + k];
        half b_kj = dequantize_cpu(k, j, q, K, N, G, zeros, scales);
        float a_ik32 = __half2float(a_ik);
        float b_kj32 = __half2float(b_kj);
        sum += a_ik32 * b_kj32;
      }
      c[i * N + j] = __float2half_rn(sum);
    }
  }
}

// threadDim.x - c-tiles
// threadDim.y - split-k blocks
template <int TILE_SIZE_M, int TILE_SIZE_K, int TILE_SIZE_N>
__global__ void awq_gemm_kernel(half* a, int* q, int* zeros, half* scales,
                                int size_m, int size_k, int size_n,
                                int group_size, int split_k, half* c) {
  int tid_c = threadIdx.x + blockIdx.x * blockDim.x;
  int tid_k = threadIdx.y + blockIdx.y * blockDim.y;
  __constant__ static const int kReverseAwqLookup[8] = {0, 4, 1, 5, 2, 6, 3, 7};

  // (ii, jj) is the tile index in C
  int ii = tid_c / TILE_SIZE_N;
  int jj = tid_c % TILE_SIZE_N;
  int num_k_tiles_per_thread = cdiv(size_k, split_k * TILE_SIZE_K);

  //if (tid_c != 31) return;
  //if (tid_k != 0) return;
  //printf("tid_c = %d tid_k = %d\n", tid_c, tid_k);
  //printf("num_k_tiles_per_thread = %d, tid_k = %d\n", num_k_tiles_per_thread,
         //tid_k);
  for (int i = 0; i < TILE_SIZE_M; ++i) {
    for (int j = 0; j < TILE_SIZE_N; ++j) {
      int i_a = ii * TILE_SIZE_M + i;
      int j_b = jj * TILE_SIZE_N + j;
      int i_c = i_a;
      int j_c = j_b;
      half c_ij = 0;
      for (int kk = tid_k; kk < tid_k + num_k_tiles_per_thread; ++kk) {
        for (int k = 0; k < TILE_SIZE_K; ++k) {
          int k_a = kk * TILE_SIZE_K + k;
          int k_b = k_a;
          half a_ik = a[i_a * size_k + k_a];
          int q_value = q[k_b * (size_n / 8) + (j_b / 8)];
          int z_value = zeros[(k_b / group_size) * (size_n / 8) + (j_b / 8)];
          int scale = scales[(k_b / group_size) * size_n + j_b];
          int shift = kReverseAwqLookup[j_b % 8] & 0xF;
          int b_int4 = q_value >> shift;
          int z_int4 = z_value >> shift;
          half b_kj = (b_int4 - z_int4) * scale;
          c_ij += a_ik * b_kj;
        }
      }
      // Implements atomic version of: c[i_c * size_n + j_c] += a_ik * b_kj
      //*(c + i_c * size_n + j_c) = 0;
      AtomicAdd(c + i_c * size_n + j_c, c_ij);
    }
  }
}

int main(int argc, char** argv) {
  constexpr uint32_t kMatrixSizeM = 64;
  constexpr uint32_t kMatrixSizeN = 64;
  constexpr uint32_t kMatrixSizeK = 64;
  constexpr uint32_t kGroupSize = 32;
  std::vector<half> a;
  std::vector<int> q;
  std::vector<half> c;
  std::vector<int> zeros;
  std::vector<half> scales;

  GenerateMatrix(kMatrixSizeM, kMatrixSizeK, a);
  GenerateMatrix(kMatrixSizeK, kMatrixSizeN / 8, q);
  Zeros(kMatrixSizeM, kMatrixSizeN, c);
  GenerateMatrix(kMatrixSizeK / kGroupSize, kMatrixSizeN / 8, zeros);
  GenerateMatrix(kMatrixSizeK / kGroupSize, kMatrixSizeN, scales);
  PrintMatrix("a", kMatrixSizeM, kMatrixSizeK, a);
  PrintMatrix("q", kMatrixSizeK, kMatrixSizeN / 8, q);
  PrintMatrix("c", kMatrixSizeK / kGroupSize, kMatrixSizeN / 8, c);
  PrintMatrix("zeros", kMatrixSizeK / kGroupSize, kMatrixSizeN / 8, zeros);
  PrintMatrix("scales", kMatrixSizeK / kGroupSize, kMatrixSizeN, scales);
  awq_gemm_cpu(a.data(), q.data(), zeros.data(), scales.data(), kMatrixSizeM,
               kMatrixSizeK, kMatrixSizeN, kGroupSize, c.data());
  PrintMatrix("c", kMatrixSizeM, kMatrixSizeN, c);

  HipArray<half> hip_array_a(a.size());
  HipArray<int> hip_array_q(q.size());
  HipArray<half> hip_array_c(c.size() +
                             1);  // add 2 bytes to end for AtomicAdd.
  HipArray<int> hip_array_zeros(zeros.size());
  HipArray<half> hip_array_scales(scales.size());

  constexpr uint32_t kSplitK = 4;
  constexpr uint32_t kTileSizeM = 8;
  constexpr uint32_t kTileSizeN = 8;
  constexpr uint32_t kTileSizeK = 8;
  constexpr uint32_t kTileSize = kTileSizeM * kTileSizeN;
  constexpr uint32_t kThreadsPerBlockX = 8;
  constexpr uint32_t kThreadsPerBlockY = kSplitK;
  constexpr uint32_t kNumTiles =
      cdiv_const(kMatrixSizeN * kMatrixSizeM, kTileSizeN * kTileSizeM);
  constexpr uint32_t kNumBlocksX = cdiv_const(kNumTiles, kThreadsPerBlockX);
  constexpr uint32_t kNumBlocksY = 1;

  // x dimension processes tiles
  // y dimension does split-k
  dim3 threads_per_block(kThreadsPerBlockX, kThreadsPerBlockY);
  dim3 blocks(kNumBlocksX, kNumBlocksY);

  std::cout << "A = " << kMatrixSizeN << " x " << kMatrixSizeK << "\n";
  std::cout << "Q = " << kMatrixSizeK << " x " << kMatrixSizeM << "\n";
  std::cout << "C = " << kMatrixSizeN << " x " << kMatrixSizeM << "\n";
  std::cout << "num_tiles = " << kNumTiles << "\n";
  std::cout << "kTileSizeM = " << kTileSizeM << " kTileSizeK = " << kTileSizeK
            << " kTileSizeN = " << kTileSizeN << "\n";
  std::cout << "threads_per_block = (" << kThreadsPerBlockX << ","
            << kThreadsPerBlockY << ")\n";
  std::cout << "blocks = (" << kNumBlocksX << "," << kNumBlocksY << ")\n";

  awq_gemm_kernel<kTileSizeM, kTileSizeK, kTileSizeM>
      <<<blocks, threads_per_block>>>(
          hip_array_a.data(), hip_array_q.data(), hip_array_zeros.data(),
          hip_array_scales.data(), kMatrixSizeM, kMatrixSizeK, kMatrixSizeN,
          kGroupSize, kSplitK, hip_array_c.data());

  CHECK_HIP(hipDeviceSynchronize());

  std::vector<half> c_gpu;

  hip_array_c.CopyFromDevice(c_gpu);
  CHECK_HIP(hipDeviceSynchronize());

  PrintMatrix("c_gpu", kMatrixSizeM, kMatrixSizeN, c_gpu);
  return 0;
}
