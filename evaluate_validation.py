"""
OOD Recall Audit — Faizul-Style Enriched Feature Vector
=========================================================
Trains factor matrices and classifiers on Phase I training data, then
strictly evaluates on the holdout validation set where every sample is a
hallucination.  Features are enriched with Frobenius norm and spectral
entropy of each core tensor (5, 64) → 322-dimensional vector.

Reports Recall (True Positive Rate) since AUROC is undefined for a
single-class test set.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import torch
from data import RawActivations
from utils import pool_activations, gram_factor_matrices, get_g_pop

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_PATH = "../res2_train_32_layers_tensor.pt"
VAL_PATH   = "../res2_val32_tensor.pt"
R_L = 5
R_D = 64
RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# FAizul-STYLE FEATURE ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def enrich_features(G_pop: torch.Tensor) -> np.ndarray:
    """Enrich each core tensor G_i ∈ R^(5,64) with two scalar features:

    Frobenius norm:   ||G_i||_F
    Spectral entropy: -Σ ŝ_k log(ŝ_k)   where ŝ_k = σ_k / Σ σ_j
                       (σ_k are singular values of G_i from SVD)

    Returns a (N, 322) NumPy array: 320 flattened entries + norm + entropy.
    """
    N = G_pop.shape[0]
    features = []

    for i in range(N):
        G_i = G_pop[i].float()                    # (5, 64) in float32

        # ── 1. Flattened core entries ──────────────────────────────────
        g_flat = G_i.flatten()                     # (320,)

        # ── 2. Frobenius norm ──────────────────────────────────────────
        core_norm = torch.linalg.norm(G_i, ord="fro")   # scalar

        # ── 3. Spectral entropy via SVD ────────────────────────────────
        S = torch.linalg.svdvals(G_i)              # singular values, shape (5,)
        S_norm = S / S.sum()                       # normalise to probability simplex
        entropy = -torch.sum(
            S_norm * torch.log(S_norm + 1e-9)      # +1e-9 avoids log(0)
        )

        # ── 4. Concatenate: (320) + (1) + (1) → (322,) ────────────────
        x_i = torch.cat([
            g_flat,
            core_norm.unsqueeze(0),
            entropy.unsqueeze(0),
        ])
        features.append(x_i)

    return torch.stack(features).numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  OOD RECALL AUDIT — ENRICHED FEATURES (Faizul)")
print("=" * 60)

print("\nLoading training data ...")
train_ds = RawActivations(TRAIN_PATH)
print(f"  Train samples: {train_ds.N}  |  Layers: {train_ds.L}  |  D: {train_ds.D}")
print(f"  Label balance: {(train_ds.y_train == 0).sum().item()} truthful, "
      f"{(train_ds.y_train == 1).sum().item()} hallucinated")

print("\nLoading validation data ...")
val_ds = RawActivations(VAL_PATH)
print(f"  Val samples: {val_ds.N}  |  Layers: {val_ds.L}  |  D: {val_ds.D}")

# -- Strict data validation ------------------------------------------------
val_raw = torch.load(VAL_PATH, weights_only=False)
assert sorted(val_raw.keys()) == ["all_emb", "all_hallucination_flag"], \
    f"FAIL: Validation keys mismatch — got {sorted(val_raw.keys())}"
print("  [PASS] (a) Validation keys are ['all_emb', 'all_hallucination_flag']")

val_emb = val_raw["all_emb"]
val_layer_keys = sorted(val_emb.keys(), key=lambda k: int(k.split(".")[-1]))
assert len(val_layer_keys) == 32, \
    f"FAIL: Expected 32 layer keys, got {len(val_layer_keys)}"
for i, key in enumerate(val_layer_keys):
    assert key == f"model.layers.{i}", \
        f"FAIL: Expected 'model.layers.{i}', got '{key}'"
print("  [PASS] (b) Validation data has 32 layer keys (model.layers.0 … 31)")

val_flags = val_raw["all_hallucination_flag"]
assert all(val_flags), \
    f"FAIL: Not all validation labels are True — found {sum(1 for f in val_flags if not f)} non-halluc. samples"
print(f"  [PASS] (c) All {len(val_flags)} validation labels are True "
      f"(hallucination-only set)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. THE IRON CURTAIN — ZERO LEAKAGE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("  Computing factor matrices from TRAINING DATA ONLY")
print("─" * 60)

X_train = pool_activations(train_ds.raw_activation_list)
U_L, U_D = gram_factor_matrices(X_train, R_L, R_D)
print(f"  X_train shape:  {tuple(X_train.shape)}")
print(f"  U_L shape:      {tuple(U_L.shape)}      (orthonormal: "
      f"{torch.allclose(U_L.T @ U_L, torch.eye(R_L), atol=1e-4)})")
print(f"  U_D shape:      {tuple(U_D.shape)}     (orthonormal: "
      f"{torch.allclose(U_D.T @ U_D, torch.eye(R_D), atol=1e-4)})")

G_train = get_g_pop(X_train, U_L, U_D)
print(f"  G_train shape:  {tuple(G_train.shape)}")

X_val = pool_activations(val_ds.raw_activation_list)
G_val = get_g_pop(X_val, U_L, U_D)
print(f"  X_val shape:    {tuple(X_val.shape)}")
print(f"  G_val shape:    {tuple(G_val.shape)}")

# ── Enrich: flatten + norm + entropy → 322-dim ────────────────────────
print("\n  Enriching features (flatten + Frobenius norm + spectral entropy) ...")
X_train_enriched = enrich_features(G_train)
X_val_enriched   = enrich_features(G_val)
print(f"  X_train_enriched: {X_train_enriched.shape}  "
      f"({R_L}×{R_D} = {R_L*R_D} flattened + 2 scalars)")
print(f"  X_val_enriched:   {X_val_enriched.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN & EVALUATE RECALL
# ══════════════════════════════════════════════════════════════════════════════

y_train = train_ds.y_train.numpy().astype(np.int64)

print("\n" + "─" * 60)
print("  Training classifiers on 322-dim enriched features …")

# -- Logistic Regression --
lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_SEED,
)
lr.fit(X_train_enriched, y_train)
lr_preds = lr.predict(X_val_enriched)
lr_recall = (lr_preds == 1).sum() / len(lr_preds)
print(f"  LogisticRegression  —  recall: {lr_recall:.4f}  "
      f"({(lr_preds == 1).sum()}/{len(lr_preds)} detected as hallucination)")

# -- Random Forest --
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
rf.fit(X_train_enriched, y_train)
rf_preds = rf.predict(X_val_enriched)
rf_recall = (rf_preds == 1).sum() / len(rf_preds)
print(f"  RandomForest        —  recall: {rf_recall:.4f}  "
      f"({(rf_preds == 1).sum()}/{len(rf_preds)} detected as hallucination)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. REPORT
# ══════════════════════════════════════════════════════════════════════════════

n_val = len(val_ds)
print(f"\n{'=' * 60}")
print(f"  OOD RECALL RESULTS  (322-dim enriched features)")
print(f"{'=' * 60}")
print(f"  Validation samples:            {n_val}")
print(f"  Ground-truth:                  all hallucination (label = True)")
print(f"  Feature vector:                322-dim")
print(f"    └─ flattened core:           320  ({R_L} × {R_D})")
print(f"    └─ Frobenius norm:           1")
print(f"    └─ Spectral entropy:         1")
print(f"")
print(f"  LogisticRegression recall:     {lr_recall:.4f}  "
      f"({lr_recall * n_val:.0f} / {n_val})")
print(f"  RandomForest recall:           {rf_recall:.4f}  "
      f"({rf_recall * n_val:.0f} / {n_val})")
print(f"")
print(f"  Interpretation:")
print(f"    The Frobenius norm captures overall activation magnitude,")
print(f"    spectral entropy captures the spread of singular values")
print(f"    (low entropy = rank-1 dominated, high = uniform spectrum).")
print(f"    Together with the flattened core entries, these provide")
print(f"    the classifier with both structural and scalar signals.")
print(f"{'=' * 60}")
