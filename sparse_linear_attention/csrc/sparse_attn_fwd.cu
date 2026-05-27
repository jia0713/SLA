#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <type_traits>

namespace {

constexpr int kMaxHeadDim = 128;

#if defined(USE_MACA)
using v4f16 = __NATIVE_VECTOR__(4, _Float16);
using v4f32 = __NATIVE_VECTOR__(4, float);

__device__ __forceinline__ v4f32 mma_qk_tile(
    const _Float16* __restrict__ q,
    const _Float16* __restrict__ k,
    int64_t qkv_base,
    int64_t L,
    int64_t D,
    int64_t m_start,
    int64_t n_start,
    int tid) {
  v4f32 acc = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int k_tile = 0; k_tile < D / 16; ++k_tile) {
    v4f16 q_frag;
    v4f16 k_frag;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int q_row = tid % 16;
      const int q_dim = k_tile * 16 + (tid / 16) * 4 + i;
      const int k_col = tid % 16;
      const int k_dim = k_tile * 16 + (tid / 16) * 4 + i;
      const int64_t m = m_start + q_row;
      const int64_t n = n_start + k_col;
      q_frag[i] = (m < L) ? q[qkv_base + m * D + q_dim] : static_cast<_Float16>(0.0f);
      k_frag[i] = (n < L) ? k[qkv_base + n * D + k_dim] : static_cast<_Float16>(0.0f);
    }

    acc = __builtin_mxc_mma_16x16x16f16(q_frag, k_frag, acc);
  }

  return acc;
}

__device__ __forceinline__ void store_qk_scores(
    v4f32 frag,
    float scores[16][64],
    int n_tile,
    int tid,
    float scale,
    int64_t m_start,
    int64_t n_start,
    int64_t L) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = (tid / 16) * 4 + i;
    const int col = tid % 16;
    const int64_t m = m_start + row;
    const int64_t n = n_start + col;
    scores[row][n_tile * 16 + col] = (m < L && n < L) ? frag[i] * scale : -INFINITY;
  }
}

__global__ void sparse_attn_fwd_maca_mma_kernel_bm64(
    const _Float16* __restrict__ q,
    const _Float16* __restrict__ k,
    const _Float16* __restrict__ v,
    const int64_t* __restrict__ lut,
    _Float16* __restrict__ out,
    int64_t L,
    int64_t D,
    int64_t M_BLOCKS,
    int64_t topk,
    float scale) {
  const int64_t m_block = blockIdx.x / 4;
  const int q_tile = blockIdx.x % 4;
  const int64_t bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t qkv_base = bh * L * D;
  const int64_t lut_base = (bh * M_BLOCKS + m_block) * topk;
  const int64_t m_start = m_block * 64 + q_tile * 16;

  __shared__ float scores[16][64];
  __shared__ float row_max[16];
  __shared__ float row_sum[16];

  if (tid < 16) {
    row_max[tid] = -INFINITY;
    row_sum[tid] = 0.0f;
  }
  __syncthreads();

  for (int64_t block_idx = 0; block_idx < topk; ++block_idx) {
    const int64_t n_block = lut[lut_base + block_idx];

#pragma unroll
    for (int n_tile = 0; n_tile < 4; ++n_tile) {
      const int64_t n_start = n_block * 64 + n_tile * 16;
      const v4f32 qk = mma_qk_tile(q, k, qkv_base, L, D, m_start, n_start, tid);
      store_qk_scores(qk, scores, n_tile, tid, scale, m_start, n_start, L);
    }
    __syncthreads();

    if (tid < 16) {
      float local_max = row_max[tid];
#pragma unroll
      for (int n = 0; n < 64; ++n) {
        local_max = fmaxf(local_max, scores[tid][n]);
      }
      row_max[tid] = local_max;
    }
    __syncthreads();
  }

  for (int64_t block_idx = 0; block_idx < topk; ++block_idx) {
    const int64_t n_block = lut[lut_base + block_idx];

#pragma unroll
    for (int n_tile = 0; n_tile < 4; ++n_tile) {
      const int64_t n_start = n_block * 64 + n_tile * 16;
      const v4f32 qk = mma_qk_tile(q, k, qkv_base, L, D, m_start, n_start, tid);
      store_qk_scores(qk, scores, n_tile, tid, scale, m_start, n_start, L);
    }
    __syncthreads();

    if (tid < 16) {
      float local_sum = row_sum[tid];
#pragma unroll
      for (int n = 0; n < 64; ++n) {
        local_sum += expf(scores[tid][n] - row_max[tid]);
      }
      row_sum[tid] = local_sum;
    }
    __syncthreads();
  }

#pragma unroll
  for (int d_tile = 0; d_tile < D / 16; ++d_tile) {
    v4f32 out_acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int64_t block_idx = 0; block_idx < topk; ++block_idx) {
      const int64_t n_block = lut[lut_base + block_idx];

#pragma unroll
      for (int n_tile = 0; n_tile < 4; ++n_tile) {
        const int64_t n_start = n_block * 64 + n_tile * 16;
        const v4f32 qk = mma_qk_tile(q, k, qkv_base, L, D, m_start, n_start, tid);
        store_qk_scores(qk, scores, n_tile, tid, scale, m_start, n_start, L);
      }
      __syncthreads();

#pragma unroll
      for (int n_tile = 0; n_tile < 4; ++n_tile) {
        v4f16 p_frag;
        v4f16 v_frag;

#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int p_row = tid % 16;
          const int p_col = (tid / 16) * 4 + i;
          const int v_row = (tid / 16) * 4 + i;
          const int v_col = tid % 16;
          const int64_t n = n_block * 64 + n_tile * 16 + v_row;
          const int d = d_tile * 16 + v_col;
          const float p = expf(scores[p_row][n_tile * 16 + p_col] - row_max[p_row]) / row_sum[p_row];
          p_frag[i] = static_cast<_Float16>(p);
          v_frag[i] = (n < L) ? v[qkv_base + n * D + d] : static_cast<_Float16>(0.0f);
        }

        out_acc = __builtin_mxc_mma_16x16x16f16(p_frag, v_frag, out_acc);
      }
      __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = (tid / 16) * 4 + i;
      const int col = tid % 16;
      const int64_t m = m_start + row;
      const int d = d_tile * 16 + col;
      if (m < L) {
        out[qkv_base + m * D + d] = static_cast<_Float16>(out_acc[i]);
      }
    }
  }
}
#endif

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
__global__ void sparse_attn_fwd_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ lut,
    scalar_t* __restrict__ out,
    int64_t B,
    int64_t H,
    int64_t L,
    int64_t D,
    int64_t M_BLOCKS,
    int64_t topk,
    float scale) {
  const int64_t m_block = blockIdx.x;
  const int64_t bh = blockIdx.y;
  const int row_in_block = threadIdx.x;
  const int64_t m = m_block * BLOCK_M + row_in_block;

  if (row_in_block >= BLOCK_M || m >= L) {
    return;
  }

  const int64_t qkv_base = bh * L * D;
  const int64_t lut_base = (bh * M_BLOCKS + m_block) * topk;

  float acc[kMaxHeadDim];
#pragma unroll
  for (int d = 0; d < kMaxHeadDim; ++d) {
    if (d < D) {
      acc[d] = 0.0f;
    }
  }

  float row_max = -INFINITY;
  float row_sum = 0.0f;

  for (int64_t block_idx = 0; block_idx < topk; ++block_idx) {
    const int64_t n_block = lut[lut_base + block_idx];
    float scores[BLOCK_N];
    float local_max = -INFINITY;

#pragma unroll
    for (int n = 0; n < BLOCK_N; ++n) {
      const int64_t kv_pos = n_block * BLOCK_N + n;
      float score = -INFINITY;
      if (kv_pos < L) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
          const float q_val = static_cast<float>(q[qkv_base + m * D + d]);
          const float k_val = static_cast<float>(k[qkv_base + kv_pos * D + d]);
          dot += q_val * k_val;
        }
        score = dot * scale;
        local_max = fmaxf(local_max, score);
      }
      scores[n] = score;
    }

    const float new_max = fmaxf(row_max, local_max);
    const float alpha = expf(row_max - new_max);
    row_sum *= alpha;
    for (int d = 0; d < D; ++d) {
      acc[d] *= alpha;
    }

#pragma unroll
    for (int n = 0; n < BLOCK_N; ++n) {
      const int64_t kv_pos = n_block * BLOCK_N + n;
      if (kv_pos < L) {
        const float p = expf(scores[n] - new_max);
        row_sum += p;
        for (int d = 0; d < D; ++d) {
          acc[d] += p * static_cast<float>(v[qkv_base + kv_pos * D + d]);
        }
      }
    }
    row_max = new_max;
  }

  const float inv_sum = 1.0f / row_sum;
  for (int d = 0; d < D; ++d) {
    out[qkv_base + m * D + d] = static_cast<scalar_t>(acc[d] * inv_sum);
  }
}

template <typename scalar_t, int BLOCK_M>
void launch_sparse_attn_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& lut,
    torch::Tensor& out,
    int64_t topk) {
  const auto B = q.size(0);
  const auto H = q.size(1);
  const auto L = q.size(2);
  const auto D = q.size(3);
  const auto M_BLOCKS = (L + BLOCK_M - 1) / BLOCK_M;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

#if defined(USE_MACA)
  if constexpr (std::is_same_v<scalar_t, at::Half> && BLOCK_M == 64) {
    if (D == 64 || D == 128) {
      dim3 grid(M_BLOCKS * 4, B * H);
      dim3 block(64);
      sparse_attn_fwd_maca_mma_kernel_bm64<<<
          grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const _Float16*>(q.data_ptr<scalar_t>()),
          reinterpret_cast<const _Float16*>(k.data_ptr<scalar_t>()),
          reinterpret_cast<const _Float16*>(v.data_ptr<scalar_t>()),
          lut.data_ptr<int64_t>(),
          reinterpret_cast<_Float16*>(out.data_ptr<scalar_t>()),
          L,
          D,
          M_BLOCKS,
          topk,
          scale);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }
#endif

  dim3 grid(M_BLOCKS, B * H);
  dim3 block(BLOCK_M);
  sparse_attn_fwd_kernel<scalar_t, BLOCK_M, 64><<<
      grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<scalar_t>(),
      k.data_ptr<scalar_t>(),
      v.data_ptr<scalar_t>(),
      lut.data_ptr<int64_t>(),
      out.data_ptr<scalar_t>(),
      B,
      H,
      L,
      D,
      M_BLOCKS,
      topk,
      scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "sparse_attn_forward_cuda",
      [&] {
        if (block_m == 64) {
          launch_sparse_attn_fwd<scalar_t, 64>(q, k, v, lut, out, topk);
        } else {
          launch_sparse_attn_fwd<scalar_t, 128>(q, k, v, lut, out, topk);
        }
      });

  return out;
}
