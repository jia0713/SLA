"""
Python bindings and autograd Function for CUDA attention.
Wraps the pure CUDA C++ implementation.
"""

import torch
import torch.nn.functional as F

# Try to load the compiled extension
try:
    from sparse_linear_attention.cuda_attention import cuda_attention_forward as _cuda_fwd
    _EXTENSION_LOADED = True
except ImportError:
    _EXTENSION_LOADED = False


class CUDAAttentionFunction(torch.autograd.Function):
    """Autograd function wrapping CUDA sparse attention forward."""

    @staticmethod
    def forward(ctx, q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
        if qk_scale is None:
            qk_scale = q.size(-1) ** -0.5

        B, H, L, D = q.shape
        M_BLOCKS = (L + BLOCK_M - 1) // BLOCK_M

        # Use CUDA kernel if extension is loaded
        if _EXTENSION_LOADED and q.is_cuda:
            os = _cuda_fwd(q, k, v, lut, BLOCK_M, BLOCK_N, qk_scale)
        else:
            # Fallback to Triton
            from sparse_linear_attention.kernel import _attention
            os = _attention.apply(q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale)

        ctx.save_for_backward(q, k, v, k_block_id, lut)
        ctx.qk_scale = qk_scale
        ctx.topk = topk
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N

        return os

    @staticmethod
    def backward(ctx, do_s):
        # Fallback to Triton for backward (Phase 2 will implement CUDA backward)
        from sparse_linear_attention.kernel import _attention
        q, k, v, k_block_id, lut = ctx.saved_tensors
        dq, dk, dv, _, _, _, _, _, _ = _attention.apply(
            q, k, v, k_block_id, lut, ctx.topk, ctx.BLOCK_M, ctx.BLOCK_N, ctx.qk_scale
        ).backward(do_s)
        return dq, dk, dv, None, None, None, None, None, None
