"""
Build script for CUDA attention extension.

Usage:
    python -m sparse_linear_attention.cuda_attention.build
"""

import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDAExtension, load


def build_extension():
    src_dir = Path(__file__).parent
    build_dir = src_dir

    cuda_ext = CUDAExtension(
        name='sparse_linear_attention.cuda_attention.sparse_linear_attention_cuda_ext',
        sources=[
            str(src_dir / 'attention.cpp'),
            str(src_dir / 'attention_kernel.cu'),
        ],
        extra_include_paths=[
            str(torch.utils.cpp_extension.include_path()),
            str(Path(torch.__file__).parent / 'include'),
        ],
        extra_cflags=['-O3'],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
            '-gencode=arch=compute_87,code=sm_87',
            '-gencode=arch=compute_90,code=sm_90',
        ],
        verbose=True,
    )

    ext = load(
        name='sparse_linear_attention_cuda_ext',
        sources=[
            str(src_dir / 'attention.cpp'),
            str(src_dir / 'attention_kernel.cu'),
        ],
        extra_include_paths=[
            str(torch.utils.cpp_extension.include_path()),
        ],
        extra_cflags=['-O3'],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
            '-gencode=arch=compute_87,code=sm_87',
            '-gencode=arch=compute_90,code=sm_90',
        ],
        verbose=True,
    )
    print(f"\nExtension built and loaded: {ext}")

    print(f"\nExtension built successfully in {src_dir}")
    print("Import it with: from sparse_linear_attention.cuda_attention import CUDAAttentionFunction")


if __name__ == '__main__':
    build_extension()
