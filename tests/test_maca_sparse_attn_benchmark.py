import pytest
import torch

from evaluate.bench_maca_sparse_attn import bench_case


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/MACA device is required")


@pytest.mark.parametrize(
    "batch,heads,seqlen,head_dim,block_m,topk_ratio,warmup,iters",
    [
        pytest.param(1, 2, 512, 64, 64, 0.5, 5, 10, id="b1-h2-l512-d64-topk50"),
        pytest.param(1, 2, 1024, 64, 64, 0.5, 5, 10, id="b1-h2-l1024-d64-topk50"),
        pytest.param(1, 2, 2048, 64, 64, 0.5, 5, 10, id="b1-h2-l2048-d64-topk50"),
        pytest.param(1, 2, 512, 128, 64, 0.5, 5, 10, id="b1-h2-l512-d128-topk50"),
        pytest.param(1, 2, 1024, 128, 64, 0.5, 5, 10, id="b1-h2-l1024-d128-topk50"),
        pytest.param(1, 2, 2048, 128, 64, 0.5, 3, 6, id="b1-h2-l2048-d128-topk50"),
        pytest.param(2, 16, 512, 64, 64, 0.25, 5, 10, id="b2-h16-l512-d64-topk25"),
        pytest.param(2, 16, 1024, 64, 64, 0.25, 3, 6, id="large-b2-h16-l1024-d64-topk25"),
        pytest.param(1, 16, 2048, 64, 64, 0.25, 3, 6, id="large-b1-h16-l2048-d64-topk25"),
        pytest.param(1, 8, 4096, 64, 64, 0.25, 2, 4, id="large-b1-h8-l4096-d64-topk25"),
        pytest.param(2, 16, 1024, 128, 64, 0.25, 3, 6, id="large-b2-h16-l1024-d128-topk25"),
        pytest.param(1, 8, 2048, 128, 64, 0.25, 2, 4, id="large-b1-h8-l2048-d128-topk25"),
    ],
)
@torch.no_grad()
def test_maca_sparse_attn_matches_triton_and_reports_perf(
    batch,
    heads,
    seqlen,
    head_dim,
    block_m,
    topk_ratio,
    warmup,
    iters,
):
    torch.manual_seed(0)
    row = bench_case(
        batch=batch,
        heads=heads,
        seqlen=seqlen,
        head_dim=head_dim,
        dtype=torch.float16,
        block_m=block_m,
        block_n=64,
        topk_ratio=topk_ratio,
        warmup=warmup,
        iters=iters,
    )

    triton_ms = (
        row["triton_error"]
        if row["triton_median_ms"] is None
        else f"{row['triton_median_ms']:.3f}ms"
    )
    speedup = (
        "n/a"
        if row["cute_speedup_median"] is None
        else f"{row['cute_speedup_median']:.3f}x"
    )
    print(
        "MACA sparse attention benchmark: "
        f"B={row['B']} H={row['H']} L={row['L']} D={row['D']} "
        f"topk={row['topk']} max_abs={row['max_abs']:.4g} "
        f"triton={triton_ms} "
        f"cute={row['cute_median_ms']:.3f}ms "
        f"cute_speedup={speedup}"
    )
