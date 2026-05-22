import pytest
import torch

from sparse_linear_attention.cuda_sparse_attn import sparse_attn_forward
from sparse_linear_attention.kernel import _attention
from sparse_linear_attention.utils import get_block_map


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_m", [64, 128])
@torch.no_grad()
def test_cuda_sparse_attn_forward_matches_triton(dtype, block_m):
    torch.manual_seed(0)
    B, H, L, D = 1, 2, 193, 64
    block_n = 64
    q = torch.randn((B, H, L, D), device="cuda", dtype=dtype)
    k = torch.randn((B, H, L, D), device="cuda", dtype=dtype)
    v = torch.randn((B, H, L, D), device="cuda", dtype=dtype)

    sparse_map, lut, topk = get_block_map(q, k, topk_ratio=0.5, BLKQ=block_m, BLKK=block_n)
    expected = _attention.apply(q, k, v, sparse_map, lut, topk, block_m, block_n)
    actual = sparse_attn_forward(q, k, v, lut, topk, block_m, block_n)

    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
