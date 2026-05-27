# MACA MMA Sparse Attention Test Plan

## Scope

This plan covers migration and validation of the forward-only sparse attention
backend from CUDA to MACA on MetaX C500/XCORE1000. The target MMA instruction is:

```cpp
using v4f16 = __NATIVE_VECTOR__(4, _Float16);
using v4f32 = __NATIVE_VECTOR__(4, float);

v4f32 __builtin_mxc_mma_16x16x16f16(v4f16 a, v4f16 b, v4f32 c);
```

All MMA layout tests must follow the project skill ground truth:

- Warp/wave size: 64 threads.
- Native A layout: `row = tid % 16`, `col = (tid / 16) * 4 + i`.
- Native B/C layout: `row = (tid / 16) * 4 + i`, `col = tid % 16`.
- Do not use NVIDIA PTX or CUDA MMA layout assumptions.

## Confirmed Remote Baseline

Remote host:

```bash
ssh acl_ici@10.0.180.24
```

Confirmed on 2026-05-26:

- Hostname: `sw-R5300-G6`
- GPU: 4 x `MetaX C500`
- MACA version: `3.7.0.36`
- Driver version: `3.8.23`
- ISA: `XCORE1000`
- Wavefront size: `64`
- Compiler: `/opt/maca-3.7.0/mxgpu_llvm/bin/mxcc`
- Basic MACA sample `vectorAdd`: compile and run passed.
- Minimal MMA intrinsic probe: compile and run passed.

Known gap:

- The active test environment is the `sla` conda environment:

  ```bash
  source /home/acl_ici/miniforge3/etc/profile.d/conda.sh
  conda activate sla
  ```

- The SLA repository is synchronized to `/home/acl_ici/workspace/SLA`.
- PyTorch, Triton, pytest, and ninja are installed in the `sla` environment.

## Required Remote Dependencies

These dependencies are required before end-to-end PyTorch extension testing:

1. Python environment
   - Python 3.12, matching this package's `setup.py`.
   - `pip`, `setuptools`, `wheel`, `ninja`, `packaging`.

2. PyTorch for MACA
   - A MACA-compatible PyTorch build, not upstream CUDA-only PyTorch.
   - It must expose tensor device support compatible with MetaX/MACA.
   - It must include C++ extension headers and libraries.
   - It should support fp16 tensors on MetaX C500.

3. Build tooling
   - `gcc`, `g++`, `make`.
   - `/opt/maca` or `/opt/maca-3.7.0` available.
   - `mxcc` available at `/opt/maca-3.7.0/mxgpu_llvm/bin/mxcc`.
   - Runtime libraries resolvable, either via default linker config or:

     ```bash
     export MACA_PATH=/opt/maca
     export PATH=/opt/maca/mxgpu_llvm/bin:$PATH
     export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH
     ```

4. Python package dependencies for SLA
   - `triton>=3.3.0` only if the reference Triton path is expected to run on
     that environment.
   - If Triton is not available on MACA, CPU/PyTorch reference tests should be
     used for correctness and local CUDA/Triton tests should remain separate.
   - Optional benchmark dependency: `flash-attn`, only if benchmark scripts need
     it.

5. Repository transfer
   - Place the repo at:

     ```bash
     /home/acl_ici/workspace/SLA
     ```

## Phase 0: Machine And Toolchain Smoke Tests

Purpose: verify the remote host can compile and run MACA kernels before touching
the SLA extension.

Commands:

```bash
ssh acl_ici@10.0.180.24 mx-smi
ssh acl_ici@10.0.180.24 /opt/maca-3.7.0/bin/macainfo
ssh acl_ici@10.0.180.24 /opt/maca-3.7.0/mxgpu_llvm/bin/mxcc --version
```

Compile/run SDK sample:

```bash
ssh acl_ici@10.0.180.24 'mkdir -p /home/acl_ici/workspace/maca_probe'
ssh acl_ici@10.0.180.24 'cp -r /opt/maca-3.7.0/samples/0_Introduction/vectorAdd /home/acl_ici/workspace/maca_probe/vectorAdd'
ssh acl_ici@10.0.180.24 'make -C /home/acl_ici/workspace/maca_probe/vectorAdd'
ssh acl_ici@10.0.180.24 'make -C /home/acl_ici/workspace/maca_probe/vectorAdd run'
```

Pass criteria:

- `mx-smi` reports available MetaX GPUs.
- `vectorAdd` prints `Test PASSED`.

## Phase 1: Minimal MMA Intrinsic Probe

Purpose: validate the compiler accepts the required native vector types and MMA
builtin, then validate the skill-defined layout with a trivial numerical case.

Probe logic:

- Allocate `A[16,16]`, `B[16,16]`, `C[16,16]`.
- Fill A and B with `1.0`.
- Launch one block with 64 threads.
- Each thread loads:
  - A: `row = tid % 16`, `col = (tid / 16) * 4 + i`.
  - B: `row = (tid / 16) * 4 + i`, `col = tid % 16`.
- Call `__builtin_mxc_mma_16x16x16f16(a, b, c)`.
- Write C using Native C layout:
  - `row = (tid / 16) * 4 + i`, `col = tid % 16`.

Pass criteria:

- Compilation succeeds with:

  ```bash
  /opt/maca-3.7.0/mxgpu_llvm/bin/mxcc -x maca -offload-arch native probe.cpp -o probe --maca-path=/opt/maca
  ```

- Runtime output equals all `16.0` and prints `MMA probe PASSED`.

## Phase 2: Unit Tests For MMA Layout Helpers

Purpose: catch row/column mapping errors before integrating sparse attention.

Test cases:

1. `A = identity`, `B = random`.
   - Expected `C = B`.

2. `A = random`, `B = identity`.
   - Expected `C = A`.

3. `A[row, col] = row * 16 + col`, `B = identity`.
   - Expected C preserves row-major A values.

4. All ones.
   - Expected every output element is `16`.

Pass criteria:

- `max_abs_error <= 1e-3` for fp16 input and fp32 accumulation output.
- Printed mismatch includes row, col, got, expected.

## Phase 3: QK Tile Correctness

Purpose: validate the first sparse attention matmul tile: `scores = Q @ K^T`.

Inputs:

- One `(16, 16)` Q tile.
- One `(16, 16)` K tile in logical row-major `[key, dim]`.
- Load Q as native A.
- Load K as native B with the logical transpose handled by load indexing, not
  by CUDA-style MMA assumptions.

Reference:

```text
scores[m, n] = sum_d Q[m, d] * K[n, d]
```

Pass criteria:

- Deterministic small tensors match CPU reference.
- Random tensors match CPU reference within fp16 accumulation tolerance.
- Output layout is written as row-major scores for inspection.

## Phase 4: PV Tile Correctness

Purpose: validate the second sparse attention matmul tile: `O = P @ V`.

Inputs:

- One `(16, 16)` P tile.
- One `(16, 16)` V tile.

Reference:

```text
out[m, d] = sum_n P[m, n] * V[n, d]
```

Pass criteria:

- Identity P returns V.
- All-one P returns column sums of V.
- Random P/V match CPU reference within tolerance.

## Phase 5: Online Softmax Block Correctness

Purpose: validate the softmax state update independently from full sparse LUT
iteration.

State per query row:

- `row_max`
- `row_sum`
- accumulated output vector

Test cases:

1. Single key block.
2. Two key blocks with increasing score ranges.
3. Two key blocks with decreasing score ranges.
4. Masked tail positions where `kv_pos >= L`.

Reference:

```python
scores = q @ k_selected.T * scale
p = softmax(scores, dim=-1)
out = p @ v_selected
```

Pass criteria:

- Match CPU reference within `atol=5e-2, rtol=5e-2` initially.
- Tighten tolerance after the implementation stabilizes.

## Phase 6: End-To-End Sparse Attention Correctness

Purpose: validate the MACA backend under the same external API shape as the
current CUDA extension:

```python
sparse_attn_forward(q, k, v, lut, topk, BLOCK_M, BLOCK_N)
```

Shapes:

1. Minimal aligned:
   - `B=1, H=1, L=64, D=64, BLOCK_M=64, BLOCK_N=64, topk=1`

2. Non-aligned sequence length:
   - `B=1, H=2, L=193, D=64, BLOCK_M=64, BLOCK_N=64`

3. Larger query block:
   - `B=1, H=2, L=193, D=64, BLOCK_M=128, BLOCK_N=64`

4. Larger head dimension:
   - `B=1, H=2, L=193, D=128, BLOCK_M=64, BLOCK_N=64`

5. Multiple top-k blocks:
   - `topk = 1`, `2`, and a mid-density value from `get_block_map`.

Reference options:

- Preferred: existing Triton `_attention.apply` if Triton works in the test
  environment.
- Fallback: CPU/PyTorch reference implementation for forward-only sparse
  attention.

Initial pass criteria:

```python
torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
```

## Phase 7: Dtype Coverage

Initial target:

- `torch.float16`

Deferred target:

- `torch.bfloat16`

Reason:

- The required skill-defined builtin is explicitly f16:

  ```cpp
  __builtin_mxc_mma_16x16x16f16
  ```

- BF16 support should only be enabled after confirming the correct MACA BF16
  MMA builtin or an accepted conversion strategy.

Pass criteria:

- FP16 passes first.
- BF16 is either implemented and tested, or explicitly rejected with a clear
  runtime error in the MACA backend.

## Phase 8: Performance Smoke Test

Purpose: ensure the MACA path is not only correct but plausibly using MMA.

Run after correctness:

```bash
python -m evaluate.bench_kernel
```

Metrics to collect:

- Runtime per shape.
- `mx-smi` GPU utilization during benchmark.
- Whether kernel launch count is stable.
- Any compiler warnings around spilling or unsupported intrinsic lowering.

Pass criteria:

- No correctness regression.
- Runtime improves over the row-per-thread scalar CUDA-style fallback.
- No obvious uncoalesced epilogue for Native C row-major writes.

## Implementation Checkpoints

Use these checkpoints to avoid debugging the full kernel at once:

1. Standalone `16x16x16` MMA probe passes.
2. QK tile passes.
3. PV tile passes.
4. Online softmax for one block passes.
5. Online softmax for multiple LUT blocks passes.
6. End-to-end FP16 `D=64` passes.
7. End-to-end FP16 `D=128` passes.
8. Optional BF16 decision is documented and tested.
9. Benchmark smoke test runs.

## Notes For Future Work

- Keep the original CUDA extension available until the MACA path is fully
  validated.
- Prefer a separate MACA source file and build flag rather than mixing CUDA and
  MACA code paths in one file.
- If PyTorch MACA extension support differs from `torch.utils.cpp_extension`,
  first build a standalone MACA kernel executable, then integrate with PyTorch.
- The current repository's CUDA kernel is scalar row-per-thread attention. It
  does not contain CUDA MMA, PTX, CuTe, or CUTLASS code to mechanically replace.
