"""
Phase II Qwen Pipeline — full HOSVD + classifier evaluation.
Loads phase2_activations_qwen.pt, runs core compression via population
HOSVD, then evaluates LogisticRegression and RandomForest on the
flattened (5, 64) core tensors with a strict 80/20 split.
"""

from data import RawActivations
from utils import (
    pool_activations,
    gram_factor_matrices,
    get_g_pop,
    evaluate_classifiers,
    analyze_core_distributions,
)

R_L = 5
R_D = 64
CKPT = "../phase2_activations_qwen.pt"

# ── 1. FULL DATASET DISTRIBUTION ────────────────────────────────────────────
print("=" * 58)
print("  PHASE II — QWEN 2.5-3B  |  CORE DISTRIBUTION ANALYSIS")
print("=" * 58)
analyze_core_distributions(CKPT, R_L, R_D)

# ── 2. 80/20 SPLIT — SUPERVISED CLASSIFICATION ──────────────────────────────
print("\n" + "=" * 58)
print("  NO-LEAKAGE CLASSIFIER EVALUATION")
print("  (U_L, U_D from training set only)")
print("=" * 58)
results = evaluate_classifiers(CKPT, R_L, R_D, test_size=0.2, random_state=42)
