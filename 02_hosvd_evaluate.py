"""
02_hosvd_evaluate.py — Population HOSVD AUROC (Zero Leakage)
===============================================================
Loads a pooled debug/full dataset, splits 80/20, computes factor
matrices U_L and U_D from TRAINING data only, projects both
partitions into the (5, 64) core subspace, and reports AUROC.
"""

import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch

# ==============================================================================
# ARGPARSE
# ==============================================================================

parser = argparse.ArgumentParser(
    description="HOSVD subspace projection + RF classification on pooled tensors"
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["truthfulqa", "triviaqa", "tydiqa"],
    help="QA dataset to evaluate",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Load the debug (4-bit, 5% slice) pooled file",
)
args = parser.parse_args()

# ==============================================================================
# CONSTANTS
# ==============================================================================

R_L = 5          # target layer rank
R_D = 64         # target hidden-dim rank
RANDOM_SEED = 42

SUFFIX = "debug" if args.debug else "full"
INPUT_PATH = f"llama3_{args.dataset}_pooled_{SUFFIX}.pt"

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

print("=" * 60)
print(f"  HOSVD AUROC  |  {args.dataset.upper()}  |  {'DEBUG' if args.debug else 'FULL'}")
print("=" * 60)

print(f"\nLoading pooled tensor: {INPUT_PATH}")
data = torch.load(INPUT_PATH, weights_only=False)

# all_emb is a list of (L, D) tensors — stack into (N, L, D)
X = torch.stack(data["all_emb"])                       # (N, L, D)
y = np.array([int(f) for f in data["all_hallucination_flag"]], dtype=np.int64)

N, L, D = X.shape
print(f"  Samples: {N}  |  Layers: {L}  |  Hidden dim: {D}")
print(f"  Truthful: {(y == 0).sum()}  |  Hallucinated: {(y == 1).sum()}")
print(f"  Hallucination rate: {y.sum() / len(y) * 100:.1f}%")

# ==============================================================================
# 2. IRON CURTAIN — 80/20 STRATIFIED SPLIT
# ==============================================================================

# Flatten to 2D for sklearn split (index-level split, then re-index tensors)
indices = np.arange(N)

# If any class has < 2 members, stratified split is impossible — fall back
n_truth = (y == 0).sum()
n_hall  = (y == 1).sum()
use_stratify = (n_truth >= 2 and n_hall >= 2)

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=y if use_stratify else None,
    random_state=RANDOM_SEED,
)

X_train = X[train_idx]               # (N_train, L, D)
X_test  = X[test_idx]                # (N_test,  L, D)
y_train = y[train_idx]
y_test  = y[test_idx]

print(f"\n  Train: {len(train_idx)}  |  Test: {len(test_idx)}"
      f"{' (unstratified — class too small)' if not use_stratify else ''}")
print(f"  Test hallucination rate: {y_test.sum()}/{len(y_test)} "
      f"({y_test.sum() / len(y_test) * 100:.1f}%)")

# ==============================================================================
# 3. POPULATION HOSVD — U_L, U_D from TRAINING DATA ONLY
# ==============================================================================

print(f"\nComputing factor matrices from TRAIN only ...")

# --- Mode-1 unfolding (Layer axis) ---
# Permute (N, L, D) -> (L, N, D) then reshape to (L, N*D)
X_L_unfold = X_train.permute(1, 0, 2).reshape(L, -1).float()   # (L, N_train*D)
U, S, Vt = torch.linalg.svd(X_L_unfold, full_matrices=False)
U_L = U[:, :R_L]                                                # (L, R_L)
print(f"  U_L: {tuple(U_L.shape)}  "
      f"(top-{R_L} explained variance: "
      f"{S[:R_L].pow(2).sum() / S.pow(2).sum() * 100:.1f}%)")

# --- Mode-2 unfolding (Hidden-dim axis) ---
# Permute (N, L, D) -> (D, N, L) then reshape to (D, N*L)
X_D_unfold = X_train.permute(2, 0, 1).reshape(D, -1).float()   # (D, N_train*L)
U_d, S_d, Vt_d = torch.linalg.svd(X_D_unfold, full_matrices=False)
U_D = U_d[:, :R_D]                                              # (D, R_D)
print(f"  U_D: {tuple(U_D.shape)} "
      f"(top-{R_D} explained variance: "
      f"{S_d[:R_D].pow(2).sum() / S_d.pow(2).sum() * 100:.1f}%)")

# ==============================================================================
# 4. SUBSPACE PROJECTION  (both sets through training-only bases)
# ==============================================================================

def project_to_core(X_tensor, U_L, U_D):
    """X_tensor: (N, L, D)  ->  returns (N, R_L, R_D) core tensors."""
    # Batch via:  G = U_L^T @ X @ U_D   — using broadcasting
    # X: (N, L, D),  U_L: (L, R_L),  U_D: (D, R_D)
    # Step 1: X @ U_D  ->  (N, L, R_D)
    # Step 2: U_L^T @ that  ->  (N, R_L, R_D)
    temp = torch.matmul(X_tensor.float(), U_D)          # (N, L, R_D)
    G = torch.matmul(temp.transpose(1, 2), U_L)         # (N, R_D, R_L)
    G = G.transpose(1, 2)                                # (N, R_L, R_D)
    return G

G_train = project_to_core(X_train, U_L, U_D)    # (N_train, R_L, R_D)
G_test  = project_to_core(X_test,  U_L, U_D)    # (N_test,  R_L, R_D)
print(f"\n  G_train shape: {tuple(G_train.shape)}")
print(f"  G_test  shape: {tuple(G_test.shape)}")

# Flatten to 2D feature matrices
X_train_feat = G_train.reshape(G_train.shape[0], -1).numpy()
X_test_feat  = G_test.reshape(G_test.shape[0], -1).numpy()
print(f"  Feature dim: {X_train_feat.shape[1]}  ({R_L} x {R_D})")
print(f"  X_train_feat: {X_train_feat.shape}")
print(f"  X_test_feat:  {X_test_feat.shape}")

# ==============================================================================
# 5. CLASSIFICATION & AUROC
# ==============================================================================

print(f"\nTraining RandomForest (200 trees, balanced) ...")
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_SEED,
    class_weight="balanced",
    n_jobs=-1,
)
rf.fit(X_train_feat, y_train)
y_probs = rf.predict_proba(X_test_feat)[:, 1]
auroc = roc_auc_score(y_test, y_probs)

print(f"  AUROC: {auroc:.4f}")

# ==============================================================================
# REPORT
# ==============================================================================

print(f"\n{'=' * 60}")
print(f"  HOSVD AUROC RESULTS")
print(f"{'=' * 60}")
print(f"  Dataset:      {args.dataset}")
print(f"  Mode:         {'debug' if args.debug else 'full'}")
print(f"  Samples:      {N}  (train {len(train_idx)}, test {len(test_idx)})")
print(f"  Layers:       {L}")
print(f"  Hidden dim:   {D}")
print(f"  Core shape:   ({R_L}, {R_D})")
print(f"  Feature dim:  {R_L * R_D}")
print(f"  Classifier:   RandomForest (200 trees, balanced)")
print(f"  AUROC:        {auroc:.4f}")
print(f"{'=' * 60}")
