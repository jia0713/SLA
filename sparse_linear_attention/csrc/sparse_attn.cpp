#include <torch/extension.h>

torch::Tensor sparse_attn_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor lut,
    int64_t topk,
    int64_t block_m,
    int64_t block_n);

torch::Tensor sparse_attn_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor lut,
    int64_t topk,
    int64_t block_m,
    int64_t block_n) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(lut.is_cuda(), "lut must be a CUDA tensor");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(lut.is_contiguous(), "lut must be contiguous");
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");
  TORCH_CHECK(q.sizes() == v.sizes(), "q and v must have the same shape");
  TORCH_CHECK(q.dim() == 4, "q must have shape (B, H, L, D)");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16, "CUTE forward currently supports fp16 only");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "q, k, and v must have the same dtype");
  TORCH_CHECK(lut.scalar_type() == torch::kInt64, "lut must be int64");
  TORCH_CHECK(topk > 0, "topk must be positive");
  TORCH_CHECK(block_m == 64 && block_n == 64, "CUTE forward requires block_m=64 and block_n=64");
  TORCH_CHECK(q.size(3) == 64 || q.size(3) == 128, "CUTE forward requires head_dim=64 or 128");
  TORCH_CHECK(q.size(2) % 64 == 0, "CUTE forward requires seqlen to be a multiple of 64");
  return sparse_attn_forward_cuda(q, k, v, lut, topk, block_m, block_n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sparse_attn_forward, "SLA sparse attention forward (MACA CUTE)");
}
