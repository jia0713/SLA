"""Forward-only CUDA backend for the sparse softmax attention branch."""

import torch


def _load_extension():
    try:
        import sparse_attn_cuda
    except ImportError as exc:
        raise RuntimeError(
            "The CUDA sparse attention extension is not built. "
            "Install on a machine with CUDA/NVCC or use sparse_backend='triton'."
        ) from exc
    return sparse_attn_cuda


def sparse_attn_forward(q, k, v, lut, topk, BLOCK_M, BLOCK_N):
    return _sparse_attn_forward_impl("forward", q, k, v, lut, topk, BLOCK_M, BLOCK_N)


def sparse_attn_forward_cute(q, k, v, lut, topk, BLOCK_M, BLOCK_N):
    return _sparse_attn_forward_impl("forward_cute", q, k, v, lut, topk, BLOCK_M, BLOCK_N)


def _sparse_attn_forward_impl(entrypoint, q, k, v, lut, topk, BLOCK_M, BLOCK_N):
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise RuntimeError(
            "The CUDA sparse attention backend is forward-only. "
            "Use sparse_backend='triton' for training."
        )
    if not q.is_cuda:
        raise RuntimeError("The CUDA sparse attention backend requires CUDA tensors.")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("The CUDA sparse attention backend supports fp16 and bf16 inputs.")
    if q.shape != k.shape or q.shape != v.shape:
        raise RuntimeError("q, k, and v must have the same shape.")
    if q.shape[-1] not in (64, 128):
        raise RuntimeError("The CUDA sparse attention backend supports head_dim 64 or 128.")
    if BLOCK_M not in (64, 128) or BLOCK_N != 64:
        raise RuntimeError("The CUDA sparse attention backend supports BLOCK_M in {64, 128} and BLOCK_N=64.")
    if int(topk) <= 0:
        raise RuntimeError("topk must be positive.")

    ext = _load_extension()
    return getattr(ext, entrypoint)(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        lut.contiguous(),
        int(topk),
        int(BLOCK_M),
        int(BLOCK_N),
    )
