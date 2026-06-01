""" 
Copyright (c) 2025 by SLA team.

Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention}, 
  author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}
"""

import os

from setuptools import find_packages, setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
except Exception:
    BuildExtension = None
    CUDAExtension = None
    CUDA_HOME = None


def should_build_cuda_extension():
    value = os.environ.get("SLA_BUILD_CUDA")
    if value is not None:
        return value not in ("0", "false", "False", "OFF", "off")
    return CUDAExtension is not None and CUDA_HOME is not None


ext_modules = []
cmdclass = {}
if should_build_cuda_extension():
    if CUDAExtension is None or CUDA_HOME is None:
        raise RuntimeError("SLA_BUILD_CUDA is enabled, but CUDA/NVCC was not found.")
    flashattn_csrc = os.path.abspath(os.path.join("flashattn_cute_example", "flashattn", "csrc"))
    maca_cute_include_dirs = []
    if os.path.isdir(flashattn_csrc):
        maca_cute_include_dirs = [
            os.path.join(flashattn_csrc, "flash_attn", "flash_kernel"),
            os.path.join(flashattn_csrc, "flash_attn", "flash_kernel", "feature"),
            os.path.join(flashattn_csrc, "flash_attn", "flash_kernel", "kernel_traits"),
            os.path.join(flashattn_csrc, "flash_attn", "utils"),
            os.path.join(flashattn_csrc, "mctlass", "include"),
        ]
        nvcc_args_define_maca_cute = ["-DSLA_ENABLE_MACA_CUTE"]
    else:
        nvcc_args_define_maca_cute = []
    nvcc_args = [
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__FAST_HALF_CVT__",
        "-D__MERGE_LDS_B64",
    ] + nvcc_args_define_maca_cute
    ext_modules.append(
        CUDAExtension(
            name="sparse_attn_cuda",
            sources=[
                "sparse_linear_attention/csrc/sparse_attn.cpp",
                "sparse_linear_attention/csrc/sparse_attn_cute_fwd.cu",
            ],
            include_dirs=maca_cute_include_dirs,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": nvcc_args,
            },
        )
    )
    cmdclass["build_ext"] = BuildExtension

setup(
    name='sparse_linear_attention',
    version='0.1.0',
    description='Sparse Linear Attention',
    author='Jintao Zhang, Haoxu Wang',
    author_email='jtzhang6@gmail.com',
    url='https://github.com/thu-ml/SLA',
    packages=find_packages(),
    python_requires='>=3.12',
    install_requires=[
        'torch>=2.7.0',
        'triton>=3.3.0',
    ],
    extras_require={
        'benchmark': ['flash-attn']
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
