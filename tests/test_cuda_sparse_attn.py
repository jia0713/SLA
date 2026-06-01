import pytest
import torch

from sparse_linear_attention.cuda_sparse_attn import sparse_attn_forward
from sparse_linear_attention.kernel import _attention
from sparse_linear_attention.utils import get_block_map


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("topk_ratio", [0.25, 0.5])
@torch.no_grad()
def test_cuda_sparse_attn_forward_matches_triton(topk_ratio):
    torch.manual_seed(0)
    B, H, L, D = 1, 2, 512, 64
    block_m = 64
    block_n = 64
    q = torch.randn((B, H, L, D), device="cuda", dtype=torch.float16)
    k = torch.randn((B, H, L, D), device="cuda", dtype=torch.float16)
    v = torch.randn((B, H, L, D), device="cuda", dtype=torch.float16)

    sparse_map, lut, topk = get_block_map(q, k, topk_ratio=topk_ratio, BLKQ=block_m, BLKK=block_n)
    expected = _attention.apply(q, k, v, sparse_map, lut, topk, block_m, block_n)
    actual = sparse_attn_forward(q, k, v, lut, topk, block_m, block_n)

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "shape,dtype,block_m",
    [
        ((1, 2, 512, 128), torch.float16, 64),
        ((1, 2, 512, 64), torch.bfloat16, 64),
        ((1, 2, 193, 64), torch.float16, 64),
        ((1, 2, 512, 64), torch.float16, 128),
    ],
)
@torch.no_grad()
def test_cuda_sparse_attn_rejects_unsupported_shapes(shape, dtype, block_m):
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    block_n = 64
    m_blocks = (shape[2] + block_m - 1) // block_m
    k_blocks = (shape[2] + block_n - 1) // block_n
    topk = max(1, int(0.5 * k_blocks))
    lut = (torch.arange(topk, device="cuda", dtype=torch.int64) % k_blocks).repeat(shape[0], shape[1], m_blocks, 1)

    with pytest.raises(RuntimeError):
        sparse_attn_forward(q, k, v, lut, topk, block_m, block_n)
