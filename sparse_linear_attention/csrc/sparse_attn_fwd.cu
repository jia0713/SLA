#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>

namespace {

constexpr int kMaxHeadDim = 128;

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
