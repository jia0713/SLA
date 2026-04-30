"""
CUDA attention module - pure CUDA implementation of sparse linear attention.
"""

import os
import sys
import torch

_extension = None
_extension_error = None

# Check if already compiled
try:
    from . import sparse_linear_attention_cuda_ext as _extension
except (ImportError, OSError):
    pass

if _extension is None:
    src_dir = os.path.dirname(__file__)

    # Build for current CUDA architecture only (faster)
    major, minor = torch.cuda.get_device_capability()
    arch = f'{major}{minor}'
    arch_flags = ['-gencode', f'arch=compute_{arch},code=sm_{arch}']

    try:
        from torch.utils.cpp_extension import load as _load
        _extension = _load(
            name='sparse_linear_attention_cuda_ext',
            sources=[
                os.path.join(src_dir, 'attention.cu'),
            ],
            extra_include_paths=[
                '/usr/local/cuda/include',
            ] + torch.utils.cpp_extension.include_paths(),
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++17'] + arch_flags,
            verbose=False,
        )
    except Exception as e:
        _extension_error = str(e)


def cuda_attention_forward(q, k, v, lut, BLOCK_M, BLOCK_N, qk_scale):
    """Call CUDA attention forward, returns (os, lse) tuple."""
    if _extension is None:
        raise RuntimeError(
            f"CUDA extension not available: {_extension_error}\n"
            "Falling back to Triton."
        )
    D = q.size(3)
    L = q.size(2)
    M_BLOCKS = (L + BLOCK_M - 1) // BLOCK_M
    return _extension.cuda_attention_forward(q, k, v, lut, BLOCK_M, BLOCK_N, qk_scale, M_BLOCKS)


class CUDAAttentionFunction(torch.autograd.Function):
    """Autograd Function wrapping CUDA sparse attention."""

    @staticmethod
    def forward(ctx, q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
        if qk_scale is None:
            qk_scale = q.size(-1) ** -0.5

        os = cuda_attention_forward(q, k, v, lut.to(torch.int32), BLOCK_M, BLOCK_N, qk_scale)[0].to(q.dtype)
        os = os.to(q.dtype)

        ctx.save_for_backward(q, k, v, k_block_id, lut)
        ctx.qk_scale = qk_scale
        ctx.topk = topk
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N

        return os

    @staticmethod
    def backward(ctx, do_s):
        from sparse_linear_attention.kernel import _attention
        q, k, v, k_block_id, lut = ctx.saved_tensors
        dq, dk, dv, _, _, _, _, _, _ = _attention.apply(
            q, k, v, k_block_id, lut, ctx.topk, ctx.BLOCK_M, ctx.BLOCK_N, ctx.qk_scale
        ).backward(do_s)
        return dq, dk, dv, None, None, None, None, None, None
