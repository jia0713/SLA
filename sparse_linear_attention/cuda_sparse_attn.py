"""Forward-only MACA CUTE backend for the sparse softmax attention branch."""

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
    """Run the CUTE kernel.

    The current kernel is intentionally narrow: fp16, head_dim in {64, 128}, block
    shape 64x64, and full 64-token tiles. Unsupported cases should use the
    Triton backend.
    """
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise RuntimeError(
            "The CUDA sparse attention backend is forward-only. "
            "Use sparse_backend='triton' for training."
        )
    if not q.is_cuda:
        raise RuntimeError("The CUDA sparse attention backend requires CUDA tensors.")
    if q.dtype != torch.float16:
        raise RuntimeError("The CUDA sparse attention backend currently supports fp16 inputs only.")
    if q.shape != k.shape or q.shape != v.shape:
        raise RuntimeError("q, k, and v must have the same shape.")
    if q.shape[-1] not in (64, 128):
        raise RuntimeError("The CUDA sparse attention backend requires head_dim=64 or 128.")
    if q.shape[-2] % 64 != 0:
        raise RuntimeError("The CUDA sparse attention backend requires sequence length to be a multiple of 64.")
    if BLOCK_M != 64 or BLOCK_N != 64:
        raise RuntimeError("The CUDA sparse attention backend requires BLOCK_M=64 and BLOCK_N=64.")
    if int(topk) <= 0:
        raise RuntimeError("topk must be positive.")

    ext = _load_extension()
    return ext.forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        lut.contiguous(),
        int(topk),
        int(BLOCK_M),
        int(BLOCK_N),
    )
