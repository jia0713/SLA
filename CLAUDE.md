# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SLA (Sparse-Linear Attention) is a trainable attention method that fuses sparse and linear attention to accelerate diffusion transformers. Two implementations are provided:

- **SLA** (`sparse_linear_attention/`): Main implementation with custom Triton kernels for forward/backward passes
- **SageSLA** (`SageSLA/`): Faster quantized forward-only implementation based on SageAttention and SpargeAttn

## Installation

```bash
pip install -e .
```

SageSLA requires additional dependencies:
```bash
pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation
```

Requirements: Python >= 3.12, torch >= 2.7.0, triton >= 3.3.0

## Running Benchmarks

```bash
python evaluate/bench_kernel.py
```

## Architecture

### Sparse Attention Path (`sparse_linear_attention/kernel.py`)
- Triton JIT kernels (`_attn_fwd`, `_attn_bwd_dq`, `_attn_bwd_dkdv`) implement the sparse attention forward and backward passes
- Uses block-based sparse attention with a lookup table (LUT) for top-k key blocks
- BLOCK_M/BLOCK_N are typically 64 or 128

### Linear Attention Path (`sparse_linear_attention/core.py`)
- `calc_linear()` computes linear attention via `k^T @ v` and `sum(k)` normalization
- Feature maps (softmax, elu, relu) project Q and K before linear attention
- `proj_l` is a trainable linear layer initialized to zero

### Block Map Calculation (`sparse_linear_attention/utils.py`)
- `get_block_map()`: Computes sparse block selection using mean-pooled Q/K blocks
- Uses smooth-k technique (subtract mean from K) similar to SageAttention
- `mean_pool()`: Triton kernel for block-wise mean pooling

### SageSLA (`SageSLA/core.py`)
- Uses `spas_sage_attn` (SpargeAttn) for quantized sparse attention on sm80/sm86/sm87/sm90
- Architecture detection via `get_cuda_arch()` to select appropriate block sizes (BLKQ=64/BLKK=128 for sm90, BLKQ=128/BLKK=64 for others)
- Forward-only; does not support backprop

## Key Design Patterns

1. **Hybrid attention**: Output = sparse_attention(q,k,v) + linear_attention(q,k,v)
2. **Trainable components**: Only `proj_l` (linear projection) is trainable; sparse selection uses fixed top-k
3. **Block sparsity**: Keys/values are grouped into blocks, and top-k blocks are selected per query block
4. **bfloat16/float16 computation**: Internal computation uses bfloat16 by default; inputs/outputs can be float32
