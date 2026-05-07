#!/usr/bin/env python3
"""
Forward-only functional test for SLA interfaces.
Tests only the forward pass (no backward) — forward outputs are bit-exact
and suitable for cross-environment verification.

Usage:
    # Generate golden reference (run once on reference machine, e.g. torch312 + GPU)
    python test_forward_functional.py --generate

    # Verify against golden reference (run on any target machine)
    python test_forward_functional.py --verify

Requirements:
    - torch >= 2.7.0
    - triton >= 3.3.0
    - GPU (CUDA)
    - spas_sage_attn (optional, for SageSLA)

Output:
    - test_outputs/sla_forward_inputs_outputs.pt: golden input/output tensors
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

TEST_CONFIGS = [
    # Base config
    dict(B=2, H=8, L=512, D=64,  topk=0.2, feature_map="softmax"),

    # Feature maps
    dict(B=2, H=8, L=512, D=64,  topk=0.2, feature_map="elu"),
    dict(B=2, H=8, L=512, D=64,  topk=0.2, feature_map="relu"),

    # Sparsity ratios
    dict(B=2, H=8, L=512, D=64,  topk=0.3, feature_map="softmax"),
    dict(B=2, H=8, L=512, D=64,  topk=0.5, feature_map="softmax"),

    # Head dimensions
    dict(B=2, H=8, L=512, D=128, topk=0.2, feature_map="softmax"),

    # Batch / head variations
    dict(B=4, H=4, L=512, D=64,  topk=0.2, feature_map="softmax"),

    # Longer sequence
    dict(B=1, H=4, L=1024, D=64, topk=0.2, feature_map="softmax"),
]

def config_name(cfg):
    return f"fm={cfg['feature_map']}_topk={cfg['topk']}_B={cfg['B']}_H={cfg['H']}_L={cfg['L']}_D={cfg['D']}"


# ── Helpers ────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_inputs(B, H, L, D, device="cuda", dtype=torch.bfloat16):
    """Create random q/k/v tensors (no grad, forward only)."""
    q = torch.randn((B, H, L, D), dtype=dtype, device=device)
    k = torch.randn((B, H, L, D), dtype=dtype, device=device)
    v = torch.randn((B, H, L, D), dtype=dtype, device=device)
    return q, k, v


# ── Validation ─────────────────────────────────────────────────────────────

def validate_forward(golden, computed, name):
    """Validate forward output against golden. Forward is bit-exact."""
    if golden is None:
        print(f"  {name}: SKIP (no golden)")
        return True, {}

    if golden.shape != computed.shape:
        print(f"  {name}: FAIL (shape mismatch: {golden.shape} vs {computed.shape})")
        return False, {"error": "shape mismatch"}

    max_diff = (golden.float() - computed.float()).abs().max().item()
    mean_diff = (golden.float() - computed.float()).abs().mean().item()
    # Count how many elements differ
    num_diff = (golden != computed).sum().item()
    total_elems = golden.numel()

    passed = max_diff < 1e-4

    print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"    golden  mean={golden.float().mean().item():.6f} std={golden.float().std().item():.6f}")
    print(f"    computed mean={computed.float().mean().item():.6f} std={computed.float().std().item():.6f}")
    print(f"    max diff={max_diff:.2e}  mean abs diff={mean_diff:.2e}")
    print(f"    exact match: {total_elems - num_diff}/{total_elems} elements"
          f" ({100*(total_elems-num_diff)/total_elems:.1f}%)")
    print(f"    threshold: max_diff < 1e-4")

    return passed, {"max_diff": max_diff, "mean_diff": mean_diff, "num_diff": num_diff}


# ── Run one config ─────────────────────────────────────────────────────────

def run_sla_forward(q, k, v, cfg):
    """Run SparseLinearAttention forward only."""
    with torch.no_grad():
        model = SparseLinearAttention(
            head_dim=cfg["D"],
            topk=cfg["topk"],
            feature_map=cfg["feature_map"],
            use_bf16=True,
            tie_feature_map_qk=True,
        ).to(q.device)
        output, sparsity = model(q, k, v, return_sparsity=True)
    return {
        "output": output.cpu(),
        "sparsity": sparsity,
    }


def run_sagesla_forward(q, k, v, cfg):
    """Run SageSparseLinearAttention forward only."""
    if SageSparseLinearAttention is None:
        return None
    with torch.no_grad():
        model = SageSparseLinearAttention(
            head_dim=cfg["D"],
            topk=cfg["topk"],
            feature_map=cfg["feature_map"],
            use_bf16=True,
            tie_feature_map_qk=True,
        ).to(q.device)
        output, sparsity = model(q, k, v, return_sparsity=True)
    return {
        "output": output.cpu(),
        "sparsity": sparsity,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SLA forward-only functional test — golden generation & verification"
    )
    parser.add_argument("--output-dir", type=str, default="./test_outputs",
                        help="Directory for golden reference file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--generate", action="store_true",
                        help="Generate golden reference")
    parser.add_argument("--verify", action="store_true",
                        help="Verify against existing golden reference")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "sla_forward_inputs_outputs.pt")

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
        print(" VERIFICATION MODE (forward only)")
        print("=" * 70)
        print(f" Golden file: {output_path}")
        print(f" Device: {device}")

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

            set_seed(args.seed)

            # Use EXACT saved inputs
            q = golden_inputs["q"].to(device)
            k = golden_inputs["k"].to(device)
            v = golden_inputs["v"].to(device)

            cfg_passed = True

            # ── SLA ──
            print("\n  [SparseLinearAttention]")
            sla_result = run_sla_forward(q.clone(), k.clone(), v.clone(), cfg)
            if golden_sla:
                ok, _ = validate_forward(golden_sla["output"], sla_result["output"], "output")
                cfg_passed &= ok

            # ── SageSLA ──
            if SageSparseLinearAttention is not None:
                print("\n  [SageSparseLinearAttention]")
                ss_result = run_sagesla_forward(q.clone(), k.clone(), v.clone(), cfg)
                if golden_sagesla and ss_result:
                    ok, _ = validate_forward(golden_sagesla["output"], ss_result["output"], "output")
                    cfg_passed &= ok
            else:
                print("\n  [SageSparseLinearAttention] SKIP (not available)")

            results.append((name, cfg_passed))
            all_passed &= cfg_passed

        # ── Summary ──
        print(f"\n{'=' * 70}")
        print(" VERIFICATION SUMMARY")
        print(f"{'=' * 70}")
        for n, p in results:
            print(f"  {n}: {'PASS' if p else 'FAIL'}")
        print(f"\n  OVERALL: {'PASS' if all_passed else 'FAIL'}")
        sys.exit(0 if all_passed else 1)

    # ── Generate mode ──────────────────────────────────────────────────
    print("=" * 70)
    print(" GENERATE MODE (forward only)")
    print("=" * 70)
    print(f" Device: {device}")
    print(f" dtype: {dtype}")
    print(f" Configs: {len(TEST_CONFIGS)}")

    golden_data = {}

    for i, cfg in enumerate(TEST_CONFIGS):
        name = config_name(cfg)
        print(f"\n{'─' * 70}")
        print(f" [{i+1}/{len(TEST_CONFIGS)}] {name}")
        print(f"{'─' * 70}")

        set_seed(args.seed)

        q, k, v = create_inputs(
            B=cfg["B"], H=cfg["H"], L=cfg["L"], D=cfg["D"],
            device=device, dtype=dtype,
        )
        print(f"  Inputs: q={list(q.shape)} k={list(k.shape)} v={list(v.shape)}  {q.dtype}")

        entry = {
            "config": cfg,
            "inputs": {
                "q": q.cpu(),
                "k": k.cpu(),
                "v": v.cpu(),
            },
        }

        # ── SLA ──
        print("  SparseLinearAttention ...")
        sla_result = run_sla_forward(q, k, v, cfg)
        entry["sparse_linear_attention"] = sla_result
        print(f"    output: {list(sla_result['output'].shape)}  sparsity={sla_result['sparsity']:.4f}")

        # ── SageSLA ──
        if SageSparseLinearAttention is not None:
            print("  SageSparseLinearAttention ...")
            ss_result = run_sagesla_forward(q, k, v, cfg)
            entry["sagesla"] = ss_result
            if ss_result:
                print(f"    output: {list(ss_result['output'].shape)}  sparsity={ss_result['sparsity']:.4f}")
        else:
            entry["sagesla"] = None
            print("  SageSparseLinearAttention: SKIP")

        golden_data[name] = entry

    # ── Save ──
    torch.save(golden_data, output_path)
    print(f"\n{'=' * 70}")
    print(f" Golden saved to: {output_path}")
    print(f" Configs: {len(golden_data)}")
    print(f"{'=' * 70}")

    # Integrity check
    print("\nVerifying saved file ...")
    loaded = torch.load(output_path, map_location="cpu", weights_only=False)
    for name in golden_data:
        e = loaded[name]
        assert e["inputs"]["q"].shape == e["sparse_linear_attention"]["output"].shape, \
            f"Shape mismatch: {name}"
        print(f"  {name}: OK")
    print("Done.")


if __name__ == "__main__":
    main()
