"""
Fusion Evaluation — HOSVD Cores + Attention Log-Diagonal Features
==================================================================
Augments the 320-dimensional HOSVD core vectors with Faizul's 32-dimensional
attention log-aggregation features to form a 352-dimensional fused vector.

Evaluates OOD Recall on the 420-sample hallucination-only validation set.
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

TRAIN_HIDDEN_PATH   = "../res2_train_32_layers_tensor.pt"
VAL_HIDDEN_PATH     = "../res2_val32_tensor.pt"
TRAIN_ATTN_PATH     = "../res2_attn_train_pad.pt"
VAL_ATTN_PATH       = "../res2_attn_val_pad.pt"

R_L = 5
R_D = 64
RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION LOG-AGGREGATION  (Faizul's formula)
# ══════════════════════════════════════════════════════════════════════════════

def extract_attention_features(attn_ckpt_path: str) -> torch.Tensor:
    """Load a Faizul attention checkpoint and collapse each layer's
    (H=32, T_i) tensor into a single scalar via log-aggregation.

    Returns (N, 32) — one scalar per layer per sample.
    """
    ckpt = torch.load(attn_ckpt_path, weights_only=False)
    all_emb = ckpt["all_emb"]

    # Sort layer keys numerically
    layer_keys = sorted(all_emb.keys(), key=lambda k: int(k.split(".")[-1]))
    num_layers = len(layer_keys)              # 32
    num_samples = len(all_emb[layer_keys[0]]) # N

    features = []

    for i in range(num_samples):
        sample_vec = []
        for key in layer_keys:
            # tensor shape: (H=32, T_i)  — 32 attention heads × variable tokens
            tensor = all_emb[key][i].float()

            # Faizul's log-aggregation:
            #   1. Stabilise:  + 1e-9
            #   2. Log
            #   3. Mean across tokens  (dim=1)
            #   4. Mean across heads   (dim=0)
            t_log = torch.log(tensor + 1e-9)        # (32, T_i)
            t_token_avg = t_log.mean(dim=1)          # (32,)
            scalar = t_token_avg.mean(dim=0)          # scalar
            sample_vec.append(scalar)

        # Stack 32 scalars → (32,) vector for this sample
        features.append(torch.stack(sample_vec))

    return torch.stack(features)                     # (N, 32)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  FUSION EVALUATION — HOSVD + ATTENTION")
print("=" * 60)

# ── Hidden states ──────────────────────────────────────────────────────
print("\nLoading hidden-state data ...")
train_ds = RawActivations(TRAIN_HIDDEN_PATH)
val_ds   = RawActivations(VAL_HIDDEN_PATH)
print(f"  Train: {train_ds.N} samples, {train_ds.L} layers, D={train_ds.D}")
print(f"  Val:   {val_ds.N} samples, {val_ds.L} layers, D={val_ds.D}")

# Validation checks
val_raw = torch.load(VAL_HIDDEN_PATH, weights_only=False)
assert sorted(val_raw.keys()) == ["all_emb", "all_hallucination_flag"]
assert len(val_raw["all_emb"]) == 32
assert all(val_raw["all_hallucination_flag"]), "Not all val labels are True"
print(f"  [PASS] Validation: 32 layers, all {val_ds.N} labels = True")

# ── Attention features ─────────────────────────────────────────────────
print("\nExtracting attention log-aggregation features ...")
X_attn_train = extract_attention_features(TRAIN_ATTN_PATH)
X_attn_val   = extract_attention_features(VAL_ATTN_PATH)
print(f"  X_attn_train:  {tuple(X_attn_train.shape)}")
print(f"  X_attn_val:    {tuple(X_attn_val.shape)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. HOSVD CORE GENERATION — ZERO LEAKAGE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("  Computing HOSVD factor matrices from TRAINING DATA ONLY")
print("─" * 60)

X_train_pooled = pool_activations(train_ds.raw_activation_list)
U_L, U_D = gram_factor_matrices(X_train_pooled, R_L, R_D)
print(f"  U_L: {tuple(U_L.shape)}  (orthonormal: "
      f"{torch.allclose(U_L.T @ U_L, torch.eye(R_L), atol=1e-4)})")
print(f"  U_D: {tuple(U_D.shape)}  (orthonormal: "
      f"{torch.allclose(U_D.T @ U_D, torch.eye(R_D), atol=1e-4)})")

# Project both sets through training-only factor matrices
G_train = get_g_pop(X_train_pooled, U_L, U_D)
X_val_pooled = pool_activations(val_ds.raw_activation_list)
G_val = get_g_pop(X_val_pooled, U_L, U_D)

# Flatten cores
X_core_train = G_train.reshape(G_train.shape[0], -1).float()  # (3979, 320)
X_core_val   = G_val.reshape(G_val.shape[0], -1).float()      # (420, 320)
print(f"  X_core_train:     {tuple(X_core_train.shape)}")
print(f"  X_core_val:       {tuple(X_core_val.shape)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FUSION — CONCATENATE & CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("  Fusing features: 320 HOSVD + 32 Attention → 352")
print("─" * 60)

X_train_fused = torch.cat([X_core_train, X_attn_train], dim=1).numpy()
X_val_fused   = torch.cat([X_core_val,   X_attn_val],   dim=1).numpy()
print(f"  X_train_fused:    {X_train_fused.shape}")
print(f"  X_val_fused:      {X_val_fused.shape}")

y_train = train_ds.y_train.numpy().astype(np.int64)

print("\n" + "─" * 60)
print("  Training classifiers on 352-dim fused features …")

# -- Logistic Regression --
lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_SEED,
)
lr.fit(X_train_fused, y_train)
lr_preds = lr.predict(X_val_fused)
lr_recall = (lr_preds == 1).sum() / len(lr_preds)
print(f"  LogisticRegression  —  recall: {lr_recall:.4f}  "
      f"({(lr_preds == 1).sum()}/{len(lr_preds)} detected)")

# -- Random Forest --
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
rf.fit(X_train_fused, y_train)
rf_preds = rf.predict(X_val_fused)
rf_recall = (rf_preds == 1).sum() / len(rf_preds)
print(f"  RandomForest        —  recall: {rf_recall:.4f}  "
      f"({(rf_preds == 1).sum()}/{len(rf_preds)} detected)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. REPORT
# ══════════════════════════════════════════════════════════════════════════════

n_val = val_ds.N
print(f"\n{'=' * 60}")
print(f"  FUSION OOD RECALL RESULTS  (352-dim)")
print(f"{'=' * 60}")
print(f"  Validation samples:            {n_val}  (all hallucination)")
print(f"  Feature composition:")
print(f"    └─ HOSVD core (flattened):   320  ({R_L} × {R_D})")
print(f"    └─ Attention log-diagonal:   32   (1 per layer)")
print(f"    └─ Total:                    352")
print(f"")
print(f"  LogisticRegression recall:     {lr_recall:.4f}  "
      f"({lr_recall * n_val:.0f} / {n_val})")
print(f"  RandomForest recall:           {rf_recall:.4f}  "
      f"({rf_recall * n_val:.0f} / {n_val})")
print(f"")
print(f"  Comparison with HOSVD-only (320-dim):")
print(f"    LR:  0.714  →  {lr_recall:.3f}   "
      f"({'↑' if lr_recall > 0.714 else '↓'}{abs(lr_recall - 0.714):.3f})")
print(f"    RF:  0.717  →  {rf_recall:.3f}   "
      f"({'↑' if rf_recall > 0.717 else '↓'}{abs(rf_recall - 0.717):.3f})")
print(f"{'=' * 60}")
