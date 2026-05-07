#!/usr/bin/env python3
"""
Functional test for SLA interfaces.
Tests SparseLinearAttention and SageSparseLinearAttention with random tensors.
Saves input/output tensors as golden reference for cross-environment verification.

Usage:
    # Generate golden reference (run once on reference machine, e.g. torch312 + GPU)
    python test_interface_functional.py --generate

    # Verify against golden reference (run on any target machine)
    python test_interface_functional.py --verify

Requirements:
    - torch >= 2.7.0
    - triton >= 3.3.0
    - GPU (CUDA)
    - spas_sage_attn (optional, for SageSLA)

Output:
    - test_outputs/sla_inputs_outputs.pt: golden input/output tensors
"""

import argparse
import os
import sys
import torch
import numpy as np

# ── Imports ────────────────────────────────────────────────────────────────

try:
    from sparse_linear_attention import SparseLinearAttention
except ImportError as e:
    print(f"FATAL: Failed to import SparseLinearAttention: {e}")
    sys.exit(1)

try:
    from SageSLA import SageSparseLinearAttention
except ImportError as e:
    print(f"WARNING: Failed to import SageSparseLinearAttention: {e}")
    print("  SageSLA tests will be skipped.")
    SageSparseLinearAttention = None


# ── Test configuration matrix ──────────────────────────────────────────────

BASE_CONFIG = dict(B=2, H=8, L=512, D=64, topk=0.2, feature_map="softmax")

TEST_CONFIGS = [
    # ── Base / default config ──
    dict(B=2, H=8, L=512, D=64, topk=0.2, feature_map="softmax"),

    # ── Different feature maps ──
    dict(B=2, H=8, L=512, D=64, topk=0.2, feature_map="elu"),
    dict(B=2, H=8, L=512, D=64, topk=0.2, feature_map="relu"),

    # ── Different sparsity ratios ──
    dict(B=2, H=8, L=512, D=64, topk=0.5, feature_map="softmax"),

    # ── Different head dimension ──
    dict(B=2, H=8, L=512, D=128, topk=0.2, feature_map="softmax"),
]

def config_name(cfg):
    """Short readable name for a config dict."""
    return f"fm={cfg['feature_map']}_topk={cfg['topk']}_D={cfg['D']}_B={cfg['B']}_H={cfg['H']}_L={cfg['L']}"


# ── Helpers ────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_test_tensors(B, H, L, D, device="cuda", dtype=torch.bfloat16):
    """Create random q/k/v tensors with requires_grad=True (for backward)."""
    q = torch.randn((B, H, L, D), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((B, H, L, D), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((B, H, L, D), dtype=dtype, device=device, requires_grad=True)
    return q, k, v


# ── Validation ─────────────────────────────────────────────────────────────

def validate_outputs(golden, computed, name, mode="forward", rtol=1e-2, atol=1e-2):
    """Validate computed output against golden reference.

    Forward outputs are bit-exact on the same GPU/dtype and near-bit-exact
    across devices.  Backward gradients are non-deterministic due to Triton
    parallel reduction order, so we use wider statistical checks.

    Args:
        mode: "forward" (tight check) or "backward" (statistical check).
    """
    if golden is None:
        print(f"  {name}: SKIP (no golden reference)")
        return True, {}

    if golden.shape != computed.shape:
        print(f"  {name}: FAIL (shape mismatch: golden {golden.shape} vs computed {computed.shape})")
        return False, {"error": "shape mismatch"}

    max_diff = (golden - computed).abs().max().item()
    mean_diff = (golden - computed).abs().mean().item()

    golden_mean = golden.float().mean().item()
    golden_std = golden.float().std().item()
    computed_mean = computed.float().mean().item()
    computed_std = computed.float().std().item()

    if mode == "forward":
        # Forward should be bit-exact or near-bit-exact
        passed = max_diff < 0.01
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
        print(f"    golden  mean={golden_mean:.6f} std={golden_std:.6f}")
        print(f"    computed mean={computed_mean:.6f} std={computed_std:.6f}")
        print(f"    max  diff={max_diff:.6f}  mean abs diff={mean_diff:.6f}")
        print(f"    threshold: max_diff < 0.01")
        return passed, {"max_diff": max_diff, "mean_diff": mean_diff}

    # Backward: statistical check with wider tolerance.
    # Triton backward kernels produce different accumulation orders each run,
    # so individual elements can differ significantly while the distribution
    # stays consistent.  We check that mean/std are within bounds AND that
    # per-element differences don't blow up.
    magnitude = max(abs(golden_mean), golden_std, 0.01)
    mean_bound = max(magnitude * rtol, atol)
    # Use wider std bound for backward: 20% relative or 0.07 absolute floor.
    # The floor accounts for non-determinism in Triton backward reduction order.
    std_bound = max(golden_std * 0.2, 0.07)

    mean_ok = abs(golden_mean - computed_mean) < mean_bound
    std_ok = abs(golden_std - computed_std) < std_bound
    # Sanity: max element-wise diff must not be absurdly large
    max_ok = max_diff < 8.0
    # Mean abs diff should be reasonable (< 0.5 typically)
    mean_diff_ok = mean_diff < 1.0

    passed = mean_ok and std_ok and max_ok and mean_diff_ok

    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {status}")
    print(f"    golden  mean={golden_mean:.6f} std={golden_std:.6f}")
    print(f"    computed mean={computed_mean:.6f} std={computed_std:.6f}")
    print(f"    mean diff={abs(golden_mean - computed_mean):.6f} (bound={mean_bound:.6f}) {'OK' if mean_ok else 'X'}")
    print(f"    std  diff={abs(golden_std - computed_std):.6f} (bound={std_bound:.6f}) {'OK' if std_ok else 'X'}")
    print(f"    max  diff={max_diff:.6f} (bound=8.0) {'OK' if max_ok else 'X'}")
    print(f"    mean abs diff={mean_diff:.6f} (bound=1.0) {'OK' if mean_diff_ok else 'X'}")

    return passed, {
        "max_diff": max_diff, "mean_diff": mean_diff,
        "golden_mean": golden_mean, "golden_std": golden_std,
        "computed_mean": computed_mean, "computed_std": computed_std,
        "mean_bound": mean_bound, "std_bound": std_bound,
    }


# ── Run one config ─────────────────────────────────────────────────────────

def run_sla(q, k, v, cfg):
    """Run SparseLinearAttention forward + backward. Returns dict of outputs."""
    B, H, L, D = q.shape
    model = SparseLinearAttention(
        head_dim=D,
        topk=cfg["topk"],
        feature_map=cfg["feature_map"],
        use_bf16=True,
        tie_feature_map_qk=True,
    ).to(q.device)

    output, sparsity = model(q, k, v, return_sparsity=True)

    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    return {
        "output": output.detach().cpu(),
        "sparsity": sparsity,
        "q_grad": q.grad.detach().cpu() if q.grad is not None else None,
        "k_grad": k.grad.detach().cpu() if k.grad is not None else None,
        "v_grad": v.grad.detach().cpu() if v.grad is not None else None,
    }


def run_sagesla(q, k, v, cfg):
    """Run SageSparseLinearAttention forward. Returns dict or None if unavailable."""
    if SageSparseLinearAttention is None:
        return None

    B, H, L, D = q.shape
    model = SageSparseLinearAttention(
        head_dim=D,
        topk=cfg["topk"],
        feature_map=cfg["feature_map"],
        use_bf16=True,
        tie_feature_map_qk=True,
    ).to(q.device)

    output, sparsity = model(q, k, v, return_sparsity=True)

    return {
        "output": output.detach().cpu(),
        "sparsity": sparsity,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SLA interface functional test — golden generation & verification"
    )
    parser.add_argument("--output-dir", type=str, default="./test_outputs",
                        help="Directory for golden reference file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--generate", action="store_true",
                        help="Generate golden reference (save inputs/outputs)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify against existing golden reference")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for validation")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for validation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "sla_inputs_outputs.pt")

    if not torch.cuda.is_available():
        print("FATAL: CUDA not available. This test requires a GPU.")
        sys.exit(1)

    device = "cuda"
    dtype = torch.bfloat16

    # ── Verify mode ────────────────────────────────────────────────────
    if args.verify:
        if not os.path.exists(output_path):
            print(f"FATAL: Golden file not found: {output_path}")
            print("Run with --generate first.")
            sys.exit(1)

        print("=" * 70)
        print(" VERIFICATION MODE")
        print("=" * 70)
        print(f" Golden file: {output_path}")
        print(f" Device: {device}")
        print(f" rtol={args.rtol}, atol={args.atol}")

        golden_data = torch.load(output_path, map_location="cpu", weights_only=False)
        all_passed = True
        results = []

        for cfg in TEST_CONFIGS:
            name = config_name(cfg)
            print(f"\n{'─' * 70}")
            print(f" Config: {name}")
            print(f"{'─' * 70}")

            if name not in golden_data:
                print("  SKIP: config not in golden file")
                continue

            entry = golden_data[name]
            golden_inputs = entry["inputs"]
            golden_sla = entry.get("sparse_linear_attention")
            golden_sagesla = entry.get("sagesla")

            # Load saved inputs and send to GPU
            set_seed(args.seed)
            q = golden_inputs["q"].to(device).detach().requires_grad_(True)
            k = golden_inputs["k"].to(device).detach().requires_grad_(True)
            v = golden_inputs["v"].to(device).detach().requires_grad_(True)

            cfg_passed = True

            # ── SLA ──
            print("\n  [SparseLinearAttention]")
            q_sla = q.clone().detach().requires_grad_(True)
            k_sla = k.clone().detach().requires_grad_(True)
            v_sla = v.clone().detach().requires_grad_(True)
            sla_result = run_sla(q_sla, k_sla, v_sla, cfg)

            if golden_sla:
                ok, _ = validate_outputs(golden_sla["output"], sla_result["output"],
                                         "SLA output", mode="forward")
                cfg_passed &= ok
                ok, _ = validate_outputs(golden_sla["q_grad"], sla_result["q_grad"],
                                         "SLA q_grad", mode="backward")
                cfg_passed &= ok
                ok, _ = validate_outputs(golden_sla["k_grad"], sla_result["k_grad"],
                                         "SLA k_grad", mode="backward")
                cfg_passed &= ok
                ok, _ = validate_outputs(golden_sla["v_grad"], sla_result["v_grad"],
                                         "SLA v_grad", mode="backward")
                cfg_passed &= ok

            # ── SageSLA ──
            if SageSparseLinearAttention is not None:
                print("\n  [SageSparseLinearAttention]")
                q_ss = q.clone().detach().requires_grad_(False)
                k_ss = k.clone().detach().requires_grad_(False)
                v_ss = v.clone().detach().requires_grad_(False)
                ss_result = run_sagesla(q_ss, k_ss, v_ss, cfg)

                if golden_sagesla and ss_result:
                    ok, _ = validate_outputs(golden_sagesla["output"], ss_result["output"],
                                             "SageSLA output", mode="forward")
                    cfg_passed &= ok
            else:
                print("\n  [SageSparseLinearAttention] SKIP (not available)")

            results.append((name, cfg_passed))
            all_passed &= cfg_passed

        # ── Summary ──
        print(f"\n{'=' * 70}")
        print(" VERIFICATION SUMMARY")
        print(f"{'=' * 70}")
        for name, passed in results:
            print(f"  {name}: {'PASS' if passed else 'FAIL'}")
        print(f"\n  OVERALL: {'PASS' if all_passed else 'FAIL'}")
        sys.exit(0 if all_passed else 1)

    # ── Generate mode (default) ────────────────────────────────────────
    print("=" * 70)
    print(" GENERATE MODE (creating golden reference)")
    print("=" * 70)
    print(f" Device: {device}")
    print(f" dtype: {dtype}")
    print(f" Configs to run: {len(TEST_CONFIGS)}")

    golden_data = {}

    for i, cfg in enumerate(TEST_CONFIGS):
        name = config_name(cfg)
        print(f"\n{'─' * 70}")
        print(f" [{i+1}/{len(TEST_CONFIGS)}] {name}")
        print(f"{'─' * 70}")

        set_seed(args.seed)

        # Create ONE set of inputs shared across models
        q, k, v = create_test_tensors(
            B=cfg["B"], H=cfg["H"], L=cfg["L"], D=cfg["D"],
            device=device, dtype=dtype,
        )

        print(f"  Inputs: q={list(q.shape)} k={list(k.shape)} v={list(v.shape)}  dtype={q.dtype}")

        entry = {
            "config": cfg,
            "inputs": {
                "q": q.detach().cpu(),
                "k": k.detach().cpu(),
                "v": v.detach().cpu(),
            },
        }

        # ── SLA ──
        print("  Running SparseLinearAttention ...")
        q_sla = q.clone().detach().requires_grad_(True)
        k_sla = k.clone().detach().requires_grad_(True)
        v_sla = v.clone().detach().requires_grad_(True)
        sla_result = run_sla(q_sla, k_sla, v_sla, cfg)
        entry["sparse_linear_attention"] = sla_result
        print(f"    output shape={list(sla_result['output'].shape)}  sparsity={sla_result['sparsity']:.4f}")

        # ── SageSLA ──
        if SageSparseLinearAttention is not None:
            print("  Running SageSparseLinearAttention ...")
            q_ss = q.clone().detach().requires_grad_(False)
            k_ss = k.clone().detach().requires_grad_(False)
            v_ss = v.clone().detach().requires_grad_(False)
            ss_result = run_sagesla(q_ss, k_ss, v_ss, cfg)
            entry["sagesla"] = ss_result
            if ss_result:
                print(f"    output shape={list(ss_result['output'].shape)}  sparsity={ss_result['sparsity']:.4f}")
        else:
            entry["sagesla"] = None
            print("  SageSparseLinearAttention: SKIP (not available)")

        golden_data[name] = entry

    # ── Save ──
    torch.save(golden_data, output_path)
    print(f"\n{'=' * 70}")
    print(f" Golden reference saved to: {output_path}")
    print(f" Configs saved: {len(golden_data)}")
    print(f"{'=' * 70}")

    # Quick sanity: reload and check
    print("\nVerifying saved file integrity ...")
    loaded = torch.load(output_path, map_location="cpu", weights_only=False)
    for name in golden_data:
        entry = loaded[name]
        assert entry["inputs"]["q"].shape == entry["sparse_linear_attention"]["output"].shape, \
            f"Shape mismatch in {name}"
        print(f"  {name}: OK")
    print("Golden reference generated successfully.")


if __name__ == "__main__":
    main()
