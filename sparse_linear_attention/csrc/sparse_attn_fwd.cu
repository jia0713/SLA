#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace {

using namespace nvcuda;

constexpr int kBlockN = 64;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerCTA = 8;
constexpr int kMmaRowsPerCTA = 32;
constexpr int kSoftmaxRowsPerWarp = kMmaRowsPerCTA / kWarpsPerCTA;

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ half from_float<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

__device__ __forceinline__ float warp_allreduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return __shfl_sync(0xffffffff, value, 0);
}

template <typename scalar_t, int BLOCK_M, int HEAD_DIM>
__global__ void sparse_attn_fwd_scalar_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ lut,
    scalar_t* __restrict__ out,
    int64_t L,
    int64_t M_BLOCKS,
    int64_t topk,
    float scale) {
  constexpr int kValuesPerLane = HEAD_DIM / kWarpSize;
  constexpr int kSubtilesPerBlock = BLOCK_M / kWarpsPerCTA;

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;

  const int64_t bh = blockIdx.y;
  const int64_t m_block = blockIdx.x / kSubtilesPerBlock;
  const int row_subtile = blockIdx.x % kSubtilesPerBlock;
  const int row_in_block = row_subtile * kWarpsPerCTA + warp_id;
  const int64_t m = m_block * BLOCK_M + row_in_block;
  const bool row_valid = row_in_block < BLOCK_M && m < L;

  const int64_t qkv_base = bh * L * HEAD_DIM;
  const int64_t lut_base = (bh * M_BLOCKS + m_block) * topk;

  __shared__ scalar_t k_shared[kBlockN * HEAD_DIM];
  __shared__ scalar_t v_shared[kBlockN * HEAD_DIM];
  __shared__ float scores_shared[kWarpsPerCTA * kBlockN];

  float q_frag[kValuesPerLane];
  float out_frag[kValuesPerLane];
#pragma unroll
  for (int i = 0; i < kValuesPerLane; ++i) {
    const int d = lane + i * kWarpSize;
    q_frag[i] = row_valid ? to_float(q[qkv_base + m * HEAD_DIM + d]) : 0.0f;
    out_frag[i] = 0.0f;
  }

  float row_max = -INFINITY;
  float row_sum = 0.0f;

  for (int64_t block_idx = 0; block_idx < topk; ++block_idx) {
    const int64_t n_block = lut[lut_base + block_idx];
    const int64_t block_start = n_block * kBlockN;
    const int valid_cols =
        block_start >= L ? 0 : static_cast<int>(std::min<int64_t>(kBlockN, L - block_start));

    for (int idx = threadIdx.x; idx < kBlockN * HEAD_DIM; idx += blockDim.x) {
      const int n = idx / HEAD_DIM;
      const int d = idx % HEAD_DIM;
      const int64_t kv_pos = block_start + n;
      if (kv_pos < L) {
        k_shared[idx] = k[qkv_base + kv_pos * HEAD_DIM + d];
        v_shared[idx] = v[qkv_base + kv_pos * HEAD_DIM + d];
      } else {
        k_shared[idx] = from_float<scalar_t>(0.0f);
        v_shared[idx] = from_float<scalar_t>(0.0f);
      }
    }
    __syncthreads();

    if (row_valid) {
      float local_max = -INFINITY;

#pragma unroll
      for (int n = 0; n < kBlockN; ++n) {
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < kValuesPerLane; ++i) {
          dot += q_frag[i] * to_float(k_shared[n * HEAD_DIM + lane + i * kWarpSize]);
        }
        dot = warp_allreduce_sum(dot);
        const float score = n < valid_cols ? dot * scale : -INFINITY;
        if (lane == 0) {
          scores_shared[warp_id * kBlockN + n] = score;
        }
        local_max = fmaxf(local_max, score);
      }
      __syncwarp();

      const float new_max = fmaxf(row_max, local_max);
      const float alpha = __expf(row_max - new_max);
      row_sum *= alpha;
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        out_frag[i] *= alpha;
      }

#pragma unroll
      for (int n = 0; n < kBlockN; ++n) {
        if (n < valid_cols) {
          const float p = __expf(scores_shared[warp_id * kBlockN + n] - new_max);
          row_sum += p;
#pragma unroll
          for (int i = 0; i < kValuesPerLane; ++i) {
            out_frag[i] += p * to_float(v_shared[n * HEAD_DIM + lane + i * kWarpSize]);
          }
        }
      }
      row_max = new_max;
    }

    __syncthreads();
  }

  if (row_valid) {
    const float inv_row_sum = 1.0f / row_sum;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      const int d = lane + i * kWarpSize;
      out[qkv_base + m * HEAD_DIM + d] = from_float<scalar_t>(out_frag[i] * inv_row_sum);
    }
  }
}

template <typename scalar_t, int HEAD_DIM>
__global__ void sparse_attn_fwd_mma64_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ lut,
    scalar_t* __restrict__ out,
    int64_t L,
    int64_t M_BLOCKS,
    int64_t topk,
    float scale) {
  constexpr int kValuesPerLane = HEAD_DIM / kWarpSize;
  constexpr int kTilesPerRow = kBlockN / 16;
  constexpr int kNumColTiles = HEAD_DIM / 16;
  constexpr int kColTilesPerPass = 4;
  constexpr int kNumPasses = kNumColTiles / kColTilesPerPass;

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;

  const int64_t bh = blockIdx.y;
  const int64_t m_block = blockIdx.x / 2;
  const int row_subtile = blockIdx.x % 2;
  const int row_block_start = static_cast<int>(m_block * 64 + row_subtile * kMmaRowsPerCTA);
  const int64_t qkv_base = bh * L * HEAD_DIM;
  const int64_t lut_base = (bh * M_BLOCKS + m_block) * topk;

  __shared__ scalar_t q_shared[kMmaRowsPerCTA * HEAD_DIM];
  __shared__ scalar_t k_shared_col[HEAD_DIM * kBlockN];
  __shared__ scalar_t v_shared[kBlockN * HEAD_DIM];
  __shared__ float scores_shared[kMmaRowsPerCTA * kBlockN];
  scalar_t* p_shared = k_shared_col;

  for (int idx = threadIdx.x; idx < kMmaRowsPerCTA * HEAD_DIM; idx += blockDim.x) {
    const int row = idx / HEAD_DIM;
    const int d = idx % HEAD_DIM;
    const int global_row = row_block_start + row;
    if (global_row < L) {
      q_shared[idx] = q[qkv_base + global_row * HEAD_DIM + d];
    } else {
      q_shared[idx] = from_float<scalar_t>(0.0f);
    }
  }
  __syncthreads();

  float row_max[kSoftmaxRowsPerWarp];
  float row_sum[kSoftmaxRowsPerWarp];
  float out_frag[kSoftmaxRowsPerWarp][kValuesPerLane];
#pragma unroll
  for (int r = 0; r < kSoftmaxRowsPerWarp; ++r) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0f;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      out_frag[r][i] = 0.0f;
    }
  }

  for (int64_t block_idx = 0; block_idx < topk; ++block_idx) {
    const int64_t n_block = lut[lut_base + block_idx];
    const int64_t block_start = n_block * kBlockN;
    const int valid_cols =
        block_start >= L ? 0 : static_cast<int>(std::min<int64_t>(kBlockN, L - block_start));

    for (int idx = threadIdx.x; idx < HEAD_DIM * kBlockN; idx += blockDim.x) {
      const int n = idx / HEAD_DIM;
      const int d = idx % HEAD_DIM;
      const int64_t kv_pos = block_start + n;
      if (kv_pos < L) {
        k_shared_col[n * HEAD_DIM + d] = k[qkv_base + kv_pos * HEAD_DIM + d];
        v_shared[n * HEAD_DIM + d] = v[qkv_base + kv_pos * HEAD_DIM + d];
      } else {
        k_shared_col[n * HEAD_DIM + d] = from_float<scalar_t>(0.0f);
        v_shared[n * HEAD_DIM + d] = from_float<scalar_t>(0.0f);
      }
    }
    __syncthreads();

    {
      const int row_tile = warp_id / kTilesPerRow;
      const int col_tile = warp_id % kTilesPerRow;

      wmma::fragment<wmma::matrix_a, 16, 16, 16, scalar_t, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, scalar_t, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
      wmma::fill_fragment(acc_frag, 0.0f);

#pragma unroll
      for (int k_tile = 0; k_tile < HEAD_DIM; k_tile += 16) {
        const scalar_t* a_ptr = q_shared + row_tile * 16 * HEAD_DIM + k_tile;
        const scalar_t* b_ptr = k_shared_col + col_tile * 16 * HEAD_DIM + k_tile;
        wmma::load_matrix_sync(a_frag, a_ptr, HEAD_DIM);
        wmma::load_matrix_sync(b_frag, b_ptr, HEAD_DIM);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }

      float* score_ptr = scores_shared + row_tile * 16 * kBlockN + col_tile * 16;
      wmma::store_matrix_sync(score_ptr, acc_frag, kBlockN, wmma::mem_row_major);
    }
    __syncthreads();

    const int row_base = warp_id * kSoftmaxRowsPerWarp;
#pragma unroll
    for (int lr = 0; lr < kSoftmaxRowsPerWarp; ++lr) {
      const int local_row = row_base + lr;
      const int global_row = row_block_start + local_row;
      if (global_row >= L) {
        continue;
      }

      float local_max = -INFINITY;
#pragma unroll
      for (int n = 0; n < kBlockN; ++n) {
        if (n < valid_cols) {
          local_max = fmaxf(local_max, scores_shared[local_row * kBlockN + n] * scale);
        }
      }

      const float new_max = fmaxf(row_max[lr], local_max);
      const float alpha = __expf(row_max[lr] - new_max);
      row_sum[lr] *= alpha;
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        out_frag[lr][i] *= alpha;
      }

#pragma unroll
      for (int n = 0; n < kBlockN; ++n) {
        if (n < valid_cols) {
          const float p = __expf(scores_shared[local_row * kBlockN + n] * scale - new_max);
          row_sum[lr] += p;
          p_shared[local_row * kBlockN + n] = from_float<scalar_t>(p);
        } else {
          p_shared[local_row * kBlockN + n] = from_float<scalar_t>(0.0f);
        }
      }
      row_max[lr] = new_max;
    }
    __syncthreads();

    for (int pass = 0; pass < kNumPasses; ++pass) {
      const int row_tile = warp_id / kColTilesPerPass;
      const int local_col_tile = warp_id % kColTilesPerPass;
      const int col_tile = pass * kColTilesPerPass + local_col_tile;

      wmma::fragment<wmma::matrix_a, 16, 16, 16, scalar_t, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, scalar_t, wmma::row_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
      wmma::fill_fragment(acc_frag, 0.0f);

#pragma unroll
      for (int k_tile = 0; k_tile < kBlockN; k_tile += 16) {
        const scalar_t* a_ptr = p_shared + row_tile * 16 * kBlockN + k_tile;
        const scalar_t* b_ptr = v_shared + k_tile * HEAD_DIM + col_tile * 16;
        wmma::load_matrix_sync(a_frag, a_ptr, kBlockN);
        wmma::load_matrix_sync(b_frag, b_ptr, HEAD_DIM);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }

      float* pv_ptr = scores_shared + row_tile * 16 * 64 + local_col_tile * 16;
      wmma::store_matrix_sync(pv_ptr, acc_frag, 64, wmma::mem_row_major);
      __syncthreads();

      const int row_base_pass = warp_id * kSoftmaxRowsPerWarp;
#pragma unroll
      for (int lr = 0; lr < kSoftmaxRowsPerWarp; ++lr) {
        const int local_row = row_base_pass + lr;
        const int global_row = row_block_start + local_row;
        if (global_row >= L) {
          continue;
        }

#pragma unroll
        for (int i = 0; i < kValuesPerLane; ++i) {
          const int d = lane + i * kWarpSize;
          if ((d / 64) == pass) {
            out_frag[lr][i] += scores_shared[local_row * 64 + (d % 64)];
          }
        }
      }
      __syncthreads();
    }
  }

  const int row_base = warp_id * kSoftmaxRowsPerWarp;
#pragma unroll
  for (int lr = 0; lr < kSoftmaxRowsPerWarp; ++lr) {
    const int local_row = row_base + lr;
    const int global_row = row_block_start + local_row;
    if (global_row >= L) {
      continue;
    }

    const float inv_row_sum = 1.0f / row_sum[lr];
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      const int d = lane + i * kWarpSize;
      out[qkv_base + global_row * HEAD_DIM + d] = from_float<scalar_t>(out_frag[lr][i] * inv_row_sum);
    }
  }
}

template <typename scalar_t, int BLOCK_M, int HEAD_DIM>
void launch_sparse_attn_fwd_scalar(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& lut,
    torch::Tensor& out,
    int64_t topk) {
  constexpr int kSubtilesPerBlock = BLOCK_M / kWarpsPerCTA;
  const auto L = q.size(2);
  const auto M_BLOCKS = (L + BLOCK_M - 1) / BLOCK_M;
  const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

  dim3 grid(M_BLOCKS * kSubtilesPerBlock, q.size(0) * q.size(1));
  dim3 block(kWarpsPerCTA * kWarpSize);
  sparse_attn_fwd_scalar_kernel<scalar_t, BLOCK_M, HEAD_DIM>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const scalar_t*>(q.data_ptr()),
          reinterpret_cast<const scalar_t*>(k.data_ptr()),
          reinterpret_cast<const scalar_t*>(v.data_ptr()),
          lut.data_ptr<int64_t>(),
          reinterpret_cast<scalar_t*>(out.data_ptr()),
          L,
          M_BLOCKS,
          topk,
          scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int HEAD_DIM>
void launch_sparse_attn_fwd_mma64(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& lut,
    torch::Tensor& out,
    int64_t topk) {
  const auto L = q.size(2);
  const auto M_BLOCKS = (L + 64 - 1) / 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

  dim3 grid(M_BLOCKS * 2, q.size(0) * q.size(1));
  dim3 block(kWarpsPerCTA * kWarpSize);
  sparse_attn_fwd_mma64_kernel<scalar_t, HEAD_DIM>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const scalar_t*>(q.data_ptr()),
          reinterpret_cast<const scalar_t*>(k.data_ptr()),
          reinterpret_cast<const scalar_t*>(v.data_ptr()),
          lut.data_ptr<int64_t>(),
          reinterpret_cast<scalar_t*>(out.data_ptr()),
          L,
          M_BLOCKS,
          topk,
          scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int BLOCK_M>
void launch_sparse_attn_fwd_dispatch(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& lut,
    torch::Tensor& out,
    int64_t topk) {
  const auto D = q.size(3);
  if (BLOCK_M == 64) {
    if (D == 64) {
      launch_sparse_attn_fwd_mma64<scalar_t, 64>(q, k, v, lut, out, topk);
    } else {
      launch_sparse_attn_fwd_mma64<scalar_t, 128>(q, k, v, lut, out, topk);
    }
  } else {
    if (D == 64) {
      launch_sparse_attn_fwd_scalar<scalar_t, BLOCK_M, 64>(q, k, v, lut, out, topk);
    } else {
      launch_sparse_attn_fwd_scalar<scalar_t, BLOCK_M, 128>(q, k, v, lut, out, topk);
    }
  }
}

}  // namespace

torch::Tensor sparse_attn_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor lut,
    int64_t topk,
    int64_t block_m,
    int64_t block_n) {
  (void)block_n;
  const c10::cuda::CUDAGuard device_guard(q.device());
  auto out = torch::empty_like(q);

  if (q.scalar_type() == at::ScalarType::Half) {
    using scalar_t = half;
    if (block_m == 64) {
      launch_sparse_attn_fwd_dispatch<scalar_t, 64>(q, k, v, lut, out, topk);
    } else {
      launch_sparse_attn_fwd_dispatch<scalar_t, 128>(q, k, v, lut, out, topk);
    }
  } else {
    using scalar_t = __nv_bfloat16;
    if (block_m == 64) {
      launch_sparse_attn_fwd_dispatch<scalar_t, 64>(q, k, v, lut, out, topk);
    } else {
      launch_sparse_attn_fwd_dispatch<scalar_t, 128>(q, k, v, lut, out, topk);
    }
  }

  return out;
}
