#!/usr/bin/env python3
"""
Benchmark CUDA sparse attention kernel vs Triton reference.

Measures sparse attention forward only (no linear attention), comparing
the CUDA kernel from attention.cu against the Triton kernel from kernel.py.

Usage:
    python evaluate/bench_cuda_kernel.py
"""

import sys
import torch

from sparse_linear_attention.utils import get_block_map
from sparse_linear_attention.kernel import _attention


def get_cuda_fn():
    try:
        from sparse_linear_attention.cuda_attention import cuda_attention_forward
        return cuda_attention_forward
    except Exception as e:
        print(f"CUDA extension not available: {e}")
        return None


def bench_kernel(fn, warmup=10, repeat=50):
    """Time a kernel using CUDA events. Returns median ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def bench_config(B, H, L, D, BLOCK_M, BLOCK_N, topk_ratio, cuda_fn, device="cuda"):
    """Benchmark CUDA and Triton on a specific configuration."""
    dtype = torch.bfloat16
    q = torch.randn(B, H, L, D, dtype=dtype, device=device).contiguous()
    k = torch.randn(B, H, L, D, dtype=dtype, device=device).contiguous()
    v = torch.randn(B, H, L, D, dtype=dtype, device=device).contiguous()

    sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio, BLKQ=BLOCK_M, BLKK=BLOCK_N)

    # --- Triton ---
    def triton_fn():
        return _attention.apply(q, k, v, sparse_map, lut, real_topk, BLOCK_M, BLOCK_N)

    ms_triton = bench_kernel(triton_fn)

    # --- CUDA ---
    qk_scale = D ** -0.5
    def cuda_fn_wrapper():
        o, _ = cuda_fn(q, k, v, lut, BLOCK_M, BLOCK_N, qk_scale)
        return o

    ms_cuda = bench_kernel(cuda_fn_wrapper)

    return ms_triton, ms_cuda


def main():
    if not torch.cuda.is_available():
        print("FATAL: CUDA not available")
        sys.exit(1)

    cuda_fn = get_cuda_fn()
    if cuda_fn is None:
        print("FATAL: CUDA extension not available")
        sys.exit(1)

    B, H = 2, 8
    configs = []
    for D in [64, 128]:
        for BM in [64, 128]:
            for L in [512, 1024, 2048, 4096, 8192]:
                for topk in [0.1, 0.2, 0.5]:
                    configs.append((D, BM, 64, L, topk))

    print("=" * 90)
    print(" CUDA vs Triton Sparse Kernel Benchmark (forward only)")
    print(f" B={B}, H={H}")
    print("=" * 90)
    print(f" {'Config':<45s} {'Triton(ms)':>10s} {'CUDA(ms)':>10s} {'Speedup':>8s}")
    print("-" * 90)

    results = []
    for D, BM, BN, L, topk in configs:
        ms_triton, ms_cuda = bench_config(B, H, L, D, BM, BN, topk, cuda_fn)
        speedup = ms_triton / ms_cuda
        results.append((D, BM, L, topk, ms_triton, ms_cuda, speedup))

        desc = f"D={D} BM={BM} L={L:5d} topk={topk:.1f}"
        marker = " ***" if speedup > 1.5 else ""
        print(f" {desc:<45s} {ms_triton:10.4f} {ms_cuda:10.4f} {speedup:7.2f}x{marker}")

    # Summary
    triton_total = sum(r[4] for r in results)
    cuda_total = sum(r[5] for r in results)
    geomean_speedup = triton_total / cuda_total

    print("-" * 90)
    print(f" {'Overall (sum of all configs)':<45s} {triton_total:10.4f} {cuda_total:10.4f} {geomean_speedup:7.2f}x")
    print(f" {'Geomean speedup':>78s}")
    print()

    # Breakdown by D
    for D in [64, 128]:
        subset = [r for r in results if r[0] == D]
        t_sum = sum(r[4] for r in subset)
        c_sum = sum(r[5] for r in subset)
        print(f"  D={D:3d}: total Triton={t_sum:.2f}ms  CUDA={c_sum:.2f}ms  speedup={t_sum/c_sum:.2f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
