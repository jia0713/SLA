import argparse
import statistics
import time

import torch

from sparse_linear_attention.cuda_sparse_attn import sparse_attn_forward
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


def _sparse_attn_reference(q, k, v, lut, topk, block_m, block_n):
    B, H, L, D = q.shape
    m_blocks = (L + block_m - 1) // block_m
    scale = D ** -0.5
    out = torch.empty_like(q)
    lut_view = lut.reshape(B * H, m_blocks, topk)

    for b in range(B):
        for h in range(H):
            bh = b * H + h
            for m_block in range(m_blocks):
                m_start = m_block * block_m
                m_end = min(m_start + block_m, L)
                block_ids = lut_view[bh, m_block, :topk].to(torch.long)
                key_chunks = []
                for n_block in block_ids.tolist():
                    n_start = n_block * block_n
                    n_end = min(n_start + block_n, L)
                    if n_start < n_end:
                        key_chunks.append(torch.arange(n_start, n_end, device=q.device))
                key_idx = torch.cat(key_chunks)

                q_block = q[b, h, m_start:m_end].float()
                k_block = k[b, h, key_idx].float()
                v_block = v[b, h, key_idx].float()
                scores = q_block @ k_block.T * scale
                probs = torch.softmax(scores, dim=-1)
                out[b, h, m_start:m_end] = (probs @ v_block).to(q.dtype)

    return out


@torch.no_grad()
def bench_case(batch, heads, seqlen, head_dim, dtype, block_m, block_n, topk_ratio, warmup, iters):
    q, k, v, sparse_map, lut, topk = _make_case(
        batch, heads, seqlen, head_dim, dtype, block_m, block_n, topk_ratio
    )

    if head_dim == 64:
        expected = _attention.apply(q, k, v, sparse_map, lut, topk, block_m, block_n)
    else:
        expected = _sparse_attn_reference(q, k, v, lut, topk, block_m, block_n)
    actual = sparse_attn_forward(q, k, v, lut, topk, block_m, block_n)
    _sync()
    max_abs = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    triton_fn = lambda: _attention.apply(q, k, v, sparse_map, lut, topk, block_m, block_n)
    cute_fn = lambda: sparse_attn_forward(q, k, v, lut, topk, block_m, block_n)

    triton_error = None
    try:
        triton_mean, triton_median, triton_min = _time_ms(triton_fn, warmup, iters)
    except Exception as exc:
        triton_error = type(exc).__name__
        triton_mean = triton_median = triton_min = None
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
        "triton_error": triton_error,
        "triton_mean_ms": triton_mean,
        "triton_median_ms": triton_median,
        "triton_min_ms": triton_min,
        "cute_mean_ms": cute_mean,
        "cute_median_ms": cute_median,
        "cute_min_ms": cute_min,
        "cute_speedup_mean": None if triton_mean is None else triton_mean / cute_mean,
        "cute_speedup_median": None if triton_median is None else triton_median / cute_median,
        "cute_speedup_min": None if triton_min is None else triton_min / cute_min,
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
        "triton_ms",
        "cute_ms",
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
                    (
                        row["triton_error"]
                        if row["triton_median_ms"] is None
                        else f"{row['triton_median_ms']:.3f}"
                    ),
                    f"{row['cute_median_ms']:.3f}",
                    (
                        "n/a"
                        if row["cute_speedup_median"] is None
                        else f"{row['cute_speedup_median']:.3f}x"
                    ),
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
