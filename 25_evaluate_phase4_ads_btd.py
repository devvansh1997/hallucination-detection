"""
25_evaluate_phase4_ads_btd.py — ADS-BTD Section 1
=====================================================
Unembedding SVD Anchor & Robust Pre-Cleaning.
Loaded in 4 incremental sections.
"""

import os, gc
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import torch
import numpy as np
from transformers import AutoModelForCausalLM
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

model_id = cfg["models"][0]["id"]
M_REASON = 256   # trailing singular vectors for reasoning subspace
LAYERS = list(range(15, 24))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# STEP 0: UNEMBEDDING SVD ANCHOR
# ============================================================================
def build_vocab_anchor():
    """Extract unembedding, absorb RMSNorm gamma, SVD in float64, partition V."""
    print("[Step 0] Building vocabulary anchor (float64) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map="cpu",
        trust_remote_code=True)

    W_U = model.lm_head.weight.data.to(dtype=torch.float64)
    D = W_U.shape[1]
    gamma = model.model.norm.weight.data.to(dtype=torch.float64)
    W_tilde = W_U * gamma.unsqueeze(0)

    print(f"  SVD on {W_tilde.shape} (float64) ...")
    U, S, Vh = torch.linalg.svd(W_tilde, full_matrices=False)
    V = Vh.T  # (D, D) in float64

    k = D - M_REASON
    V_S = V[:, :k]
    V_R = V[:, k:]
    P_S = V_S @ V_S.T
    P_R = V_R @ V_R.T

    del model, W_U, W_tilde, U, S
    gc.collect()

    print(f"  V_S: {tuple(V_S.shape)}  V_R: {tuple(V_R.shape)}")
    # Downcast for downstream
    return (V_S.to(torch.bfloat16), V_R.to(torch.bfloat16),
            P_S.to(torch.bfloat16), P_R.to(torch.bfloat16), D)


# ============================================================================
# STEP 1: ROBUST PRE-CLEANING
# ============================================================================
def robust_preclean(X, clip_percentile=99.0):
    """Layer-wise RMS norm + channel-wise percentile clipping.
    X: (N, T, L, D) or batched tensor."""
    # Layer-wise RMS standardization
    rms = X.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6
    X = X / rms

    # Channel-wise percentile clipping across (N, T, L)
    flat = X.flatten(0, 2)                                 # (N*T*L, D)
    threshold = torch.quantile(flat.abs().float(), clip_percentile / 100.0, dim=0)
    X = X.clamp(-threshold, threshold)
    return X


# ============================================================================
# SECTION 1 UNIT TESTS
# ============================================================================
def test_section1(V_S_bf16, V_R_bf16, P_S_bf16, P_R_bf16):
    # Cast back to float64 for exact testing
    V_S = V_S_bf16.to(torch.float64)
    V_R = V_R_bf16.to(torch.float64)
    P_S = P_S_bf16.to(torch.float64)
    P_R = P_R_bf16.to(torch.float64)

    print("\n[Section 1 Tests] (float64)")

    # Test 1A: Orthogonality
    ov = (V_S.T @ V_R).norm(p="fro").item()
    print(f"  Test 1A: ||V_S^T V_R||_F = {ov:.2e}  (target < 1e-10)")
    assert ov < 1e-10, f"Bases not orthogonal: {ov:.2e}"

    # Test 1B: Projection completeness
    h = torch.randn(4096, dtype=torch.float64)
    proj = (P_S + P_R) @ h
    err = (proj - h).norm(p=2).item() / h.norm(p=2).item()
    print(f"  Test 1B: ||(P_S+P_R)h - h|| / ||h|| = {err:.2e}  (target < 1e-10)")
    assert err < 1e-10, f"Projection not complete: {err:.2e}"

    # Test 1C: Outlier suppression
    X = torch.randn(10, 5, 4, 4096)
    X[:, 0, 0, 0] = 5000.0
    Xc = robust_preclean(X)
    assert Xc.max() <= 5000.0 * 0.5, f"Spike not suppressed: {Xc.max():.1f}"
    bulk_ok = (Xc[1:, :, :, :].abs().mean() - X[1:, :, :, :].abs().mean()).abs() < 1.0
    print(f"  Test 1C: max={Xc.max():.1f}  bulk unchanged: {bulk_ok}")
    assert bulk_ok, "Bulk values distorted"

    print("  [PASS] All Section 1 tests\n")


# ============================================================================
# MAIN — SECTION 1
# ============================================================================
if __name__ == "__main__":
    V_S, V_R, P_S, P_R, D = build_vocab_anchor()
    test_section1(V_S, V_R, P_S, P_R)
    print("[PAUSED] Section 1 verified. Waiting for explicit 'go' command "
          "to implement Section 2.")
