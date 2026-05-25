import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import torch


def preload_torch_libs():
    torch_root = Path(torch.__file__).resolve().parent
    torch_lib_dir = torch_root / "lib"
    libs = [
        "libc10.so",
        "libtorch_cpu.so",
        "libtorch_cuda.so",
        "libc10_cuda.so",
        "libtorch_python.so",
    ]
    for name in libs:
        path = torch_lib_dir / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def bench_ms(fn, warmup, repeat):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / repeat


def make_inputs(batch, heads, seq_len, head_dim, dtype):
    q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=dtype).contiguous()
    k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=dtype).contiguous()
    v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=dtype).contiguous()
    return q, k, v


def build_attn(head_dim, topk, feature_map, block_m, block_n, dtype, backend):
    from sparse_linear_attention import SparseLinearAttention

    if block_n != 64:
        raise ValueError("Both Triton and CUDA sparse backends currently assume BLOCK_N=64.")
    return SparseLinearAttention(
        head_dim=head_dim,
        topk=topk,
        feature_map=feature_map,
        BLKQ=block_m,
        BLKK=block_n,
        use_bf16=(dtype == torch.bfloat16),
        sparse_backend=backend,
    ).cuda().eval()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SLA forward with Triton vs CUDA sparse attention backends."
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128, choices=[64, 128])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--topk", type=float, default=0.2)
    parser.add_argument("--block-m", type=int, default=64, choices=[64, 128])
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--feature-map", choices=["softmax", "relu", "elu"], default="softmax")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def format_result(result, emit_json):
    if emit_json:
        print(json.dumps(result))
        return

    case = result["case"]
    corr = result["correctness"]
    sparse_ms = result["sparse_ms"]
    full_ms = result["sla_fwd_ms"]
    print(
        "case "
        f"B={case['B']} H={case['H']} L={case['L']} D={case['D']} "
        f"dtype={case['dtype']} topk={case['topk']} real_topk={case['real_topk']}"
    )
    print(
        "  correctness "
        f"max_abs={corr['max_abs']:.8f} mean_abs={corr['mean_abs']:.8f}"
    )
    if sparse_ms is not None:
        print(
            "  sparse_only_ms "
            f"triton={sparse_ms['triton']:.4f} cuda={sparse_ms['cuda']:.4f} "
            f"cuda_speedup={sparse_ms['cuda_speedup_vs_triton']:.3f}x"
        )
    if full_ms is not None:
        print(
            "  sla_fwd_ms "
            f"triton={full_ms['triton']:.4f} cuda={full_ms['cuda']:.4f} "
            f"cuda_speedup={full_ms['cuda_speedup_vs_triton']:.3f}x"
        )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    preload_torch_libs()
    from sparse_linear_attention import SparseLinearAttention
    from sparse_linear_attention.cuda_sparse_attn import sparse_attn_forward
    from sparse_linear_attention.kernel import _attention
    from sparse_linear_attention.utils import get_block_map

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(
        f"device={torch.cuda.get_device_name(0)} "
        f"torch={torch.__version__} "
        f"cuda={torch.version.cuda}"
    )

    for seq_len in args.seq_lens:
        q, k, v = make_inputs(args.batch, args.heads, seq_len, args.head_dim, dtype)
        sparse_map, lut, real_topk = get_block_map(
            q,
            k,
            topk_ratio=args.topk,
            BLKQ=args.block_m,
            BLKK=args.block_n,
        )

        if int(real_topk) <= 0:
            raise RuntimeError(
                f"real_topk became 0 for L={seq_len}. "
                "Increase seq_len or topk."
            )

        attn_triton = build_attn(
            args.head_dim,
            args.topk,
            args.feature_map,
            args.block_m,
            args.block_n,
            dtype,
            "triton",
        )
        attn_cuda = build_attn(
            args.head_dim,
            args.topk,
            args.feature_map,
            args.block_m,
            args.block_n,
            dtype,
            "cuda",
        )
        attn_cuda.load_state_dict(attn_triton.state_dict())

        with torch.no_grad():
            out_triton = attn_triton(q, k, v)
            out_cuda = attn_cuda(q, k, v)
            torch.cuda.synchronize()

        diff = (out_triton - out_cuda).abs()
        result = {
            "case": {
                "B": args.batch,
                "H": args.heads,
                "L": seq_len,
                "D": args.head_dim,
                "dtype": str(dtype),
                "topk": args.topk,
                "real_topk": int(real_topk),
                "feature_map": args.feature_map,
                "block_m": args.block_m,
                "block_n": args.block_n,
            },
            "correctness": {
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.float().mean().item()),
            },
            "sparse_ms": None,
            "sla_fwd_ms": None,
        }

        if not args.check_only:
            sparse_triton = lambda: _attention.apply(
                q, k, v, sparse_map, lut, real_topk, args.block_m, args.block_n
            )
            sparse_cuda = lambda: sparse_attn_forward(
                q, k, v, lut, real_topk, args.block_m, args.block_n
            )
            full_triton = lambda: attn_triton(q, k, v)
            full_cuda = lambda: attn_cuda(q, k, v)

            ms_sparse_triton = bench_ms(sparse_triton, args.warmup, args.repeat)
            ms_sparse_cuda = bench_ms(sparse_cuda, args.warmup, args.repeat)
            ms_full_triton = bench_ms(full_triton, args.warmup, args.repeat)
            ms_full_cuda = bench_ms(full_cuda, args.warmup, args.repeat)

            result["sparse_ms"] = {
                "triton": ms_sparse_triton,
                "cuda": ms_sparse_cuda,
                "cuda_speedup_vs_triton": ms_sparse_triton / ms_sparse_cuda,
            }
            result["sla_fwd_ms"] = {
                "triton": ms_full_triton,
                "cuda": ms_full_cuda,
                "cuda_speedup_vs_triton": ms_full_triton / ms_full_cuda,
            }

        format_result(result, args.json)


if __name__ == "__main__":
    main()
