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
#define MACA_WAVE_SYNC() __syncwave()

__device__ __forceinline__ int float_to_i32(float value) {
  union {
    float f;
    int i;
  } u;
  u.f = value;
  return u.i;
}

__device__ __forceinline__ float i32_to_float(int value) {
  union {
    int i;
    float f;
  } u;
  u.i = value;
  return u.f;
}

#define SLA_ROW_BROADCAST_FLOAT(value, lane) \
  i32_to_float(__builtin_mxc_mov_shfl(float_to_i32(value), 0x150 + lane, 0xf, 0xf, false))

__device__ __forceinline__ float row_reduce_max(float value) {
  float result = -INFINITY;
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 0));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 1));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 2));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 3));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 4));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 5));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 6));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 7));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 8));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 9));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 10));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 11));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 12));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 13));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 14));
  result = fmaxf(result, SLA_ROW_BROADCAST_FLOAT(value, 15));
  return result;
}

__device__ __forceinline__ float row_reduce_sum(float value) {
  float result = 0.0f;
  result += SLA_ROW_BROADCAST_FLOAT(value, 0);
  result += SLA_ROW_BROADCAST_FLOAT(value, 1);
  result += SLA_ROW_BROADCAST_FLOAT(value, 2);
  result += SLA_ROW_BROADCAST_FLOAT(value, 3);
  result += SLA_ROW_BROADCAST_FLOAT(value, 4);
  result += SLA_ROW_BROADCAST_FLOAT(value, 5);
  result += SLA_ROW_BROADCAST_FLOAT(value, 6);
  result += SLA_ROW_BROADCAST_FLOAT(value, 7);
  result += SLA_ROW_BROADCAST_FLOAT(value, 8);
  result += SLA_ROW_BROADCAST_FLOAT(value, 9);
  result += SLA_ROW_BROADCAST_FLOAT(value, 10);
  result += SLA_ROW_BROADCAST_FLOAT(value, 11);
  result += SLA_ROW_BROADCAST_FLOAT(value, 12);
  result += SLA_ROW_BROADCAST_FLOAT(value, 13);
  result += SLA_ROW_BROADCAST_FLOAT(value, 14);
  result += SLA_ROW_BROADCAST_FLOAT(value, 15);
  return result;
}

template <int HEAD_DIM, bool FULL_TILES>
__device__ __forceinline__ v4f32 mma_qk_tile(
    const _Float16* __restrict__ q,
    const _Float16* __restrict__ k,
    int64_t qkv_base,
    int64_t L,
    int64_t m_start,
    int64_t n_start,
    int tid) {
  v4f32 acc = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int k_tile = 0; k_tile < HEAD_DIM / 16; ++k_tile) {
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
      if constexpr (FULL_TILES) {
        q_frag[i] = q[qkv_base + m * HEAD_DIM + q_dim];
        k_frag[i] = k[qkv_base + n * HEAD_DIM + k_dim];
      } else {
        q_frag[i] = (m < L) ? q[qkv_base + m * HEAD_DIM + q_dim] : static_cast<_Float16>(0.0f);
        k_frag[i] = (n < L) ? k[qkv_base + n * HEAD_DIM + k_dim] : static_cast<_Float16>(0.0f);
      }
    }

    acc = __builtin_mxc_mma_16x16x16f16(q_frag, k_frag, acc);
  }

  return acc;
}

template <int HEAD_DIM, bool FULL_TILES>
__device__ __forceinline__ v4f32 mma_qk_tile_smem_k(
    const _Float16* __restrict__ q,
    const _Float16* __restrict__ k_smem,
    int64_t qkv_base,
    int64_t L,
    int64_t m_start,
    int tid) {
  v4f32 acc = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int k_tile = 0; k_tile < HEAD_DIM / 16; ++k_tile) {
    v4f16 q_frag;
    v4f16 k_frag;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int q_row = tid % 16;
      const int q_dim = k_tile * 16 + (tid / 16) * 4 + i;
      const int k_col = tid % 16;
      const int k_dim = k_tile * 16 + (tid / 16) * 4 + i;
      const int64_t m = m_start + q_row;
      if constexpr (FULL_TILES) {
        q_frag[i] = q[qkv_base + m * HEAD_DIM + q_dim];
      } else {
        q_frag[i] = (m < L) ? q[qkv_base + m * HEAD_DIM + q_dim] : static_cast<_Float16>(0.0f);
      }
      k_frag[i] = k_smem[k_col * HEAD_DIM + k_dim];
    }

    acc = __builtin_mxc_mma_16x16x16f16(q_frag, k_frag, acc);
  }

  return acc;
}

template <bool FULL_TILES>
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
    if constexpr (FULL_TILES) {
      scores[row][n_tile * 16 + col] = frag[i] * scale;
    } else {
      scores[row][n_tile * 16 + col] = (m < L && n < L) ? frag[i] * scale : -INFINITY;
    }
  }
}

template <bool FULL_TILES>
__device__ __forceinline__ void store_qk_scores_update_max(
    v4f32 frag,
    float scores[16][64],
    float block_local_max[4],
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
    float score;
    if constexpr (FULL_TILES) {
      score = frag[i] * scale;
    } else {
      score = (m < L && n < L) ? frag[i] * scale : -INFINITY;
    }
    scores[row][n_tile * 16 + col] = score;
    block_local_max[i] = fmaxf(block_local_max[i], score);
  }
}

template <int HEAD_DIM, bool FULL_TILES>
__device__ __forceinline__ void load_kv_block_smem(
    const _Float16* __restrict__ k,
    const _Float16* __restrict__ v,
    _Float16* __restrict__ k_smem,
    _Float16* __restrict__ v_smem,
    int64_t qkv_base,
    int64_t L,
    int64_t n_block,
    int cta_tid) {
  const int64_t n_block_start = n_block * 64;

  v4f16* k_smem_vec = reinterpret_cast<v4f16*>(k_smem);
  v4f16* v_smem_vec = reinterpret_cast<v4f16*>(v_smem);
  const v4f16* k_vec = reinterpret_cast<const v4f16*>(k + qkv_base + n_block_start * HEAD_DIM);
  const v4f16* v_vec = reinterpret_cast<const v4f16*>(v + qkv_base + n_block_start * HEAD_DIM);
#pragma unroll
  for (int vec_idx = cta_tid; vec_idx < (64 * HEAD_DIM) / 4; vec_idx += 256) {
    const int elem_idx = vec_idx * 4;
    const int n = elem_idx / HEAD_DIM;
    const int64_t src_n = n_block_start + n;
    if constexpr (FULL_TILES) {
      k_smem_vec[vec_idx] = k_vec[vec_idx];
      v_smem_vec[vec_idx] = v_vec[vec_idx];
    } else {
      const bool valid_n = src_n < L;
      if (valid_n) {
        k_smem_vec[vec_idx] = k_vec[vec_idx];
        v_smem_vec[vec_idx] = v_vec[vec_idx];
      } else {
        k_smem_vec[vec_idx] = {
            static_cast<_Float16>(0.0f),
            static_cast<_Float16>(0.0f),
            static_cast<_Float16>(0.0f),
            static_cast<_Float16>(0.0f)};
        v_smem_vec[vec_idx] = {
            static_cast<_Float16>(0.0f),
            static_cast<_Float16>(0.0f),
            static_cast<_Float16>(0.0f),
            static_cast<_Float16>(0.0f)};
      }
    }
  }
}

template <int HEAD_DIM, bool FULL_TILES>
__device__ __forceinline__ void compute_topk_block_w4_smem(
    const _Float16* __restrict__ q,
    float scores[16][64],
    const _Float16* __restrict__ k_smem,
    const _Float16* __restrict__ v_smem,
    v4f32* __restrict__ out_acc,
    float row_max[4],
    float row_sum[4],
    float row_alpha[4],
    int64_t qkv_base,
    int64_t L,
    int64_t n_block,
    int64_t m_start,
    int tid,
    float scale) {
  constexpr int kDTiles = HEAD_DIM / 16;
  const int64_t n_block_start = n_block * 64;

  float block_local_max[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    block_local_max[i] = -INFINITY;
  }

#pragma unroll
  for (int n_tile = 0; n_tile < 4; ++n_tile) {
    const int64_t n_start = n_block_start + n_tile * 16;
    const v4f32 qk = mma_qk_tile_smem_k<HEAD_DIM, FULL_TILES>(
        q, k_smem + n_tile * 16 * HEAD_DIM, qkv_base, L, m_start, tid);
    store_qk_scores_update_max<FULL_TILES>(
        qk, scores, block_local_max, n_tile, tid, scale * 1.4426950408889634f, m_start, n_start, L);
  }
  MACA_WAVE_SYNC();

  const int score_group = tid / 16;
  const int score_col = tid % 16;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = score_group * 4 + i;
    const float block_max = row_reduce_max(block_local_max[i]);
    const bool valid_m = (m_start + row) < L;
    const float old_max = row_max[i];
    const float new_max = fmaxf(old_max, block_max);
    row_alpha[i] = valid_m ? exp2f(old_max - new_max) : 0.0f;
    row_max[i] = new_max;
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = score_group * 4 + i;
    float local_sum = 0.0f;
#pragma unroll
    for (int n_tile = 0; n_tile < 4; ++n_tile) {
      const float p = exp2f(scores[row][n_tile * 16 + score_col] - row_max[i]);
      scores[row][n_tile * 16 + score_col] = p;
      local_sum += p;
    }
    local_sum = row_reduce_sum(local_sum);
    row_sum[i] = ((m_start + row) < L) ? row_sum[i] * row_alpha[i] + local_sum : 1.0f;
  }
  MACA_WAVE_SYNC();

  for (int d_tile = 0; d_tile < kDTiles; ++d_tile) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      out_acc[d_tile][i] *= row_alpha[i];
    }

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
        const int d = d_tile * 16 + v_col;
        p_frag[i] = static_cast<_Float16>(scores[p_row][n_tile * 16 + p_col]);
        v_frag[i] = v_smem[(n_tile * 16 + v_row) * HEAD_DIM + d];
      }

      out_acc[d_tile] = __builtin_mxc_mma_16x16x16f16(p_frag, v_frag, out_acc[d_tile]);
    }
  }
}

template <int HEAD_DIM, int TOPK, bool FULL_TILES>
__global__ void sparse_attn_fwd_maca_mma_kernel_bm64_w4_smem(
    const _Float16* __restrict__ q,
    const _Float16* __restrict__ k,
    const _Float16* __restrict__ v,
    const int64_t* __restrict__ lut,
    _Float16* __restrict__ out,
    int64_t L,
    int64_t M_BLOCKS,
    int64_t topk,
    float scale) {
  const int64_t m_block = blockIdx.x;
  const int q_tile = threadIdx.x / 64;
  const int64_t bh = blockIdx.y;
  const int tid = threadIdx.x % 64;
  const int cta_tid = threadIdx.x;
  constexpr int kDTiles = HEAD_DIM / 16;
  const int64_t qkv_base = bh * L * HEAD_DIM;
  const int64_t lut_base = (bh * M_BLOCKS + m_block) * topk;
  const int64_t m_start = m_block * 64 + q_tile * 16;

  __shared__ float scores_storage[4][16][64];
  __shared__ _Float16 k_smem[64 * HEAD_DIM];
  __shared__ _Float16 v_smem[64 * HEAD_DIM];
  float (*scores)[64] = scores_storage[q_tile];
  float row_max[4];
  float row_sum[4];
  float row_alpha[4];

  v4f32 out_acc[kDTiles];
#pragma unroll
  for (int d_tile = 0; d_tile < kDTiles; ++d_tile) {
    out_acc[d_tile] = {0.0f, 0.0f, 0.0f, 0.0f};
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    row_max[i] = -INFINITY;
    row_sum[i] = 0.0f;
    row_alpha[i] = 0.0f;
  }

  if constexpr (TOPK > 0) {
#pragma unroll
    for (int block_idx = 0; block_idx < TOPK; ++block_idx) {
      const int64_t n_block = lut[lut_base + block_idx];
      load_kv_block_smem<HEAD_DIM, FULL_TILES>(
          k,
          v,
          k_smem,
          v_smem,
          qkv_base,
          L,
          n_block,
          cta_tid);
      __syncthreads();
      compute_topk_block_w4_smem<HEAD_DIM, FULL_TILES>(
          q,
          scores,
          k_smem,
          v_smem,
          out_acc,
          row_max,
          row_sum,
          row_alpha,
          qkv_base,
          L,
          n_block,
          m_start,
          tid,
          scale);
      __syncthreads();
    }
  } else {
    for (int block_idx = 0; block_idx < static_cast<int>(topk); ++block_idx) {
      const int64_t n_block = lut[lut_base + block_idx];
      load_kv_block_smem<HEAD_DIM, FULL_TILES>(
          k,
          v,
          k_smem,
          v_smem,
          qkv_base,
          L,
          n_block,
          cta_tid);
      __syncthreads();
      compute_topk_block_w4_smem<HEAD_DIM, FULL_TILES>(
          q,
          scores,
          k_smem,
          v_smem,
          out_acc,
          row_max,
          row_sum,
          row_alpha,
          qkv_base,
          L,
          n_block,
          m_start,
          tid,
          scale);
      __syncthreads();
    }
  }

  for (int d_tile = 0; d_tile < kDTiles; ++d_tile) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = (tid / 16) * 4 + i;
      const int col = tid % 16;
      const int64_t m = m_start + row;
      const int d = d_tile * 16 + col;
      if constexpr (FULL_TILES) {
        out[qkv_base + m * HEAD_DIM + d] = static_cast<_Float16>(out_acc[d_tile][i] / row_sum[i]);
      } else if (m < L) {
        out[qkv_base + m * HEAD_DIM + d] = static_cast<_Float16>(out_acc[d_tile][i] / row_sum[i]);
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
      dim3 grid(M_BLOCKS, B * H);
      dim3 block(256);
      const bool full_tiles = (L % 64) == 0;
#define SLA_LAUNCH_MACA_MMA(HEAD_DIM_VALUE, TOPK_VALUE, FULL_TILES_VALUE) \
  sparse_attn_fwd_maca_mma_kernel_bm64_w4_smem<HEAD_DIM_VALUE, TOPK_VALUE, FULL_TILES_VALUE><<< \
      grid, block, 0, at::cuda::getCurrentCUDAStream()>>>( \
      reinterpret_cast<const _Float16*>(q.data_ptr<scalar_t>()), \
      reinterpret_cast<const _Float16*>(k.data_ptr<scalar_t>()), \
      reinterpret_cast<const _Float16*>(v.data_ptr<scalar_t>()), \
      lut.data_ptr<int64_t>(), \
      reinterpret_cast<_Float16*>(out.data_ptr<scalar_t>()), \
      L, \
      M_BLOCKS, \
      topk, \
      scale)
#define SLA_DISPATCH_MACA_MMA(HEAD_DIM_VALUE, FULL_TILES_VALUE) \
  do { \
    if (topk == 1) { \
      SLA_LAUNCH_MACA_MMA(HEAD_DIM_VALUE, 1, FULL_TILES_VALUE); \
    } else if (topk == 2) { \
      SLA_LAUNCH_MACA_MMA(HEAD_DIM_VALUE, 2, FULL_TILES_VALUE); \
    } else if (topk == 4) { \
      SLA_LAUNCH_MACA_MMA(HEAD_DIM_VALUE, 4, FULL_TILES_VALUE); \
    } else if (topk == 8) { \
      SLA_LAUNCH_MACA_MMA(HEAD_DIM_VALUE, 8, FULL_TILES_VALUE); \
    } else if (topk == 16) { \
      SLA_LAUNCH_MACA_MMA(HEAD_DIM_VALUE, 16, FULL_TILES_VALUE); \
    } else if (topk == 32) { \
      SLA_LAUNCH_MACA_MMA(HEAD_DIM_VALUE, 32, FULL_TILES_VALUE); \
    } else { \
      SLA_LAUNCH_MACA_MMA(HEAD_DIM_VALUE, -1, FULL_TILES_VALUE); \
    } \
  } while (0)
      if (D == 64) {
        if (full_tiles) {
          SLA_DISPATCH_MACA_MMA(64, true);
        } else {
          SLA_DISPATCH_MACA_MMA(64, false);
        }
      } else {
        if (full_tiles) {
          SLA_DISPATCH_MACA_MMA(128, true);
        } else {
          SLA_DISPATCH_MACA_MMA(128, false);
        }
      }
#undef SLA_DISPATCH_MACA_MMA
#undef SLA_LAUNCH_MACA_MMA
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
