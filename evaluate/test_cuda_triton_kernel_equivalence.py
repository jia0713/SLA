#!/usr/bin/env python3
"""
Compare CUDA sparse attention forward kernel vs Triton reference.
Runs both kernels on identical inputs and reports element-wise differences.

Usage:
    python evaluate/test_cuda_triton_kernel_equivalence.py
"""

import sys
import torch

from sparse_linear_attention.utils import get_block_map
from sparse_linear_attention.kernel import _attention


def get_cuda_fn():
    """Try to get the CUDA attention forward function."""
    try:
        from sparse_linear_attention.cuda_attention import cuda_attention_forward
        return cuda_attention_forward
    except Exception as e:
        print(f"CUDA extension not available: {e}")
        return None


def analyze_diff(triton_out, cuda_out, name, threshold=8e-3):
    """Compare two tensors element-wise."""
    diff = (triton_out.float() - cuda_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    exact_match = (triton_out == cuda_out).sum().item()
    total = triton_out.numel()

    passed = max_diff < threshold
    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {status}")
    print(f"    max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  "
          f"exact={exact_match}/{total} ({100*exact_match/total:.1f}%)")
    print(f"    threshold: {threshold}")
    return passed, {"max_diff": max_diff, "mean_diff": mean_diff}


def run_single_test(B, H, L, D, BLOCK_M, BLOCK_N, topk_ratio, cuda_fn, seed=42):
    """Run CUDA and Triton kernels on the same data, compare outputs."""
    torch.manual_seed(seed)
    device = "cuda"

    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=device).contiguous()
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=device).contiguous()
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=device).contiguous()

    # Get LUT (same for both paths)
    sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio, BLKQ=BLOCK_M, BLKK=BLOCK_N)

    # --- Triton path ---
    with torch.no_grad():
        o_triton = _attention.apply(q, k, v, sparse_map, lut, real_topk, BLOCK_M, BLOCK_N)

    # --- CUDA path ---
    qk_scale = D ** -0.5
    try:
        o_cuda_f32, lse_cuda = cuda_fn(q, k, v, lut, BLOCK_M, BLOCK_N, qk_scale)
        o_cuda = o_cuda_f32.to(torch.bfloat16)
    except Exception as e:
        print(f"  CUDA kernel error: {e}")
        return False, {"error": str(e)}

    # Compare outputs
    passed, info = analyze_diff(o_triton, o_cuda, "sparse_output")
    return passed, info


def main():
    if not torch.cuda.is_available():
        print("FATAL: CUDA not available")
        sys.exit(1)

    cuda_fn = get_cuda_fn()
    if cuda_fn is None:
        print("FATAL: CUDA extension not available")
        sys.exit(1)

    print("=" * 70)
    print(" CUDA vs Triton Sparse Kernel Equivalence Test")
    print("=" * 70)

    configs = [
        # Description             B   H   L    D    BLOCK_M  BLOCK_N  topk
        ("D=64  BM=64  base",     2,  8,  512, 64,  64,      64,      0.2),
        ("D=64  BM=128",          2,  8,  512, 64,  128,     64,      0.2),
        ("D=128 BM=64",           2,  8,  512, 128, 64,      64,      0.2),
        ("D=128 BM=128",          2,  8,  512, 128, 128,     64,      0.2),
        # Edge cases
        ("D=64  L=511 (non-div)", 2,  8,  511, 64,  64,      64,      0.2),
        ("D=64  L=1024 topk=0.1", 1,  4,  1024,64,  64,      64,      0.1),
        ("D=64  topk=0.5",        2,  8,  512, 64,  64,      64,      0.5),
        ("D=128 L=500 topk=0.3",  2,  8,  500, 128, 64,      64,      0.3),
        ("D=64  B=1 H=1",         1,  1,  256, 64,  64,      64,      0.5),
    ]

    results = []
    for desc, B, H, L, D, BM, BN, topk in configs:
        print(f"\n{'─' * 70}")
        print(f" Config: {desc}  (B={B}, H={H}, L={L}, D={D}, BM={BM}, topk={topk})")
        print(f"{'─' * 70}")
        passed, info = run_single_test(B, H, L, D, BM, BN, topk, cuda_fn)
        results.append((desc, passed, info))

    print(f"\n{'=' * 70}")
    print(" SUMMARY")
    print(f"{'=' * 70}")
    all_passed = True
    for desc, passed, info in results:
        md = info.get("max_diff", float("nan"))
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {desc}  (max_diff={md:.2e})")
        all_passed = all_passed and passed

    print(f"\n  OVERALL: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
