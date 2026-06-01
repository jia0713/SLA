import argparse
import statistics
import time

import torch

from sparse_linear_attention.cuda_sparse_attn import sparse_attn_forward, sparse_attn_forward_cute
from sparse_linear_attention.kernel import _attention
from sparse_linear_attention.utils import get_block_map


def _sync():
    torch.cuda.synchronize()


def _time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    _sync()

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.mean(samples), statistics.median(samples), min(samples)


def _make_case(batch, heads, seqlen, head_dim, dtype, block_m, block_n, topk_ratio):
    q = torch.randn((batch, heads, seqlen, head_dim), device="cuda", dtype=dtype).contiguous()
    k = torch.randn((batch, heads, seqlen, head_dim), device="cuda", dtype=dtype).contiguous()
    v = torch.randn((batch, heads, seqlen, head_dim), device="cuda", dtype=dtype).contiguous()
    sparse_map, lut, topk = get_block_map(q, k, topk_ratio=topk_ratio, BLKQ=block_m, BLKK=block_n)
    return q, k, v, sparse_map, lut, topk


@torch.no_grad()
def bench_case(batch, heads, seqlen, head_dim, dtype, block_m, block_n, topk_ratio, warmup, iters):
    q, k, v, sparse_map, lut, topk = _make_case(
        batch, heads, seqlen, head_dim, dtype, block_m, block_n, topk_ratio
    )

    expected = _attention.apply(q, k, v, sparse_map, lut, topk, block_m, block_n)
    actual = sparse_attn_forward(q, k, v, lut, topk, block_m, block_n)
    actual_cute = sparse_attn_forward_cute(q, k, v, lut, topk, block_m, block_n)
    _sync()
    max_abs = (actual - expected).abs().max().item()
    max_abs_cute = (actual_cute - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual_cute, expected, atol=1e-3, rtol=1e-3)

    triton_fn = lambda: _attention.apply(q, k, v, sparse_map, lut, topk, block_m, block_n)
    maca_fn = lambda: sparse_attn_forward(q, k, v, lut, topk, block_m, block_n)
    cute_fn = lambda: sparse_attn_forward_cute(q, k, v, lut, topk, block_m, block_n)

    triton_mean, triton_median, triton_min = _time_ms(triton_fn, warmup, iters)
    maca_mean, maca_median, maca_min = _time_ms(maca_fn, warmup, iters)
    cute_mean, cute_median, cute_min = _time_ms(cute_fn, warmup, iters)

    return {
        "B": batch,
        "H": heads,
        "L": seqlen,
        "D": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "topk": int(topk),
        "topk_ratio": topk_ratio,
        "max_abs": max_abs,
        "max_abs_cute": max_abs_cute,
        "triton_mean_ms": triton_mean,
        "triton_median_ms": triton_median,
        "triton_min_ms": triton_min,
        "maca_mean_ms": maca_mean,
        "maca_median_ms": maca_median,
        "maca_min_ms": maca_min,
        "cute_mean_ms": cute_mean,
        "cute_median_ms": cute_median,
        "cute_min_ms": cute_min,
        "speedup_mean": triton_mean / maca_mean,
        "speedup_median": triton_median / maca_median,
        "speedup_min": triton_min / maca_min,
        "cute_speedup_mean": triton_mean / cute_mean,
        "cute_speedup_median": triton_median / cute_median,
        "cute_speedup_min": triton_min / cute_min,
    }


def _print_table(rows):
    headers = [
        "B",
        "H",
        "L",
        "D",
        "dtype",
        "BM",
        "BN",
        "topk",
        "max_abs",
        "cute_abs",
        "triton_ms",
        "maca_ms",
        "cute_ms",
        "speedup",
        "cute_spd",
    ]
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["B"]),
                    str(row["H"]),
                    str(row["L"]),
                    str(row["D"]),
                    row["dtype"],
                    str(row["BLOCK_M"]),
                    str(row["BLOCK_N"]),
                    str(row["topk"]),
                    f"{row['max_abs']:.4g}",
                    f"{row['max_abs_cute']:.4g}",
                    f"{row['triton_median_ms']:.3f}",
                    f"{row['maca_median_ms']:.3f}",
                    f"{row['cute_median_ms']:.3f}",
                    f"{row['speedup_median']:.3f}x",
                    f"{row['cute_speedup_median']:.3f}x",
                ]
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark MACA sparse attention forward against Triton.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--topk-ratio", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/MACA device is required")

    torch.manual_seed(0)
    rows = []
    for seqlen in args.seqlens:
        rows.append(
            bench_case(
                batch=args.batch,
                heads=args.heads,
                seqlen=seqlen,
                head_dim=args.head_dim,
                dtype=torch.float16,
                block_m=args.block_m,
                block_n=args.block_n,
                topk_ratio=args.topk_ratio,
                warmup=args.warmup,
                iters=args.iters,
            )
        )

    _print_table(rows)


if __name__ == "__main__":
    main()
