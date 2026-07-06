"""
02_hosvd_evaluate.py — Known/Unknown Split HOSVD AUROC
========================================================
HARP-compatible evaluation protocol:
  1. Load pooled tensors (with all_is_known metadata)
  2. Split KNOWN (truthful) samples: 75% train, 25% valid
  3. Add ALL UNKNOWN (hallucinated) samples to valid
  4. Compute U_L, U_D from TRAIN ONLY
  5. Project both sets → 320-dim core features
  6. Train RandomForest, report AUROC on valid

Usage:
  python 02_hosvd_evaluate.py --model_folder llama-3.1-8b-instruct --dataset triviaqa
  python 02_hosvd_evaluate.py  # runs all model/dataset combos found in data/
"""

import argparse
import os

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import torch

# ==============================================================================
# CONFIG
# ==============================================================================

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = cfg["output"]["data_dir"]
R_L = cfg["hosvd"]["layer_rank"]
R_D = cfg["hosvd"]["hidden_rank"]
RANDOM_SEED = 42

# ==============================================================================
# CLI
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, default=None)
parser.add_argument("--dataset", type=str, default=None)
args = parser.parse_args()


# ==============================================================================
# HOSVD CORE
# ==============================================================================

def compute_ul_ud(X_train: torch.Tensor):
    """X_train: (N, L, D). Returns U_L (L, R_L), U_D (D, R_D)."""
    L, D = X_train.shape[1], X_train.shape[2]

    # Mode-1 (layer) unfolding
    X_L = X_train.permute(1, 0, 2).reshape(L, -1).float()
    U_L, _, _ = torch.linalg.svd(X_L, full_matrices=False)
    U_L = U_L[:, :R_L]

    # Mode-2 (hidden) unfolding
    X_D = X_train.permute(2, 0, 1).reshape(D, -1).float()
    U_D, _, _ = torch.linalg.svd(X_D, full_matrices=False)
    U_D = U_D[:, :R_D]

    return U_L, U_D


def project(X: torch.Tensor, U_L: torch.Tensor, U_D: torch.Tensor) -> np.ndarray:
    """X: (N, L, D) → flattened cores (N, R_L*R_D)."""
    temp = torch.matmul(X.float(), U_D)          # (N, L, R_D)
    G = torch.matmul(temp.transpose(1, 2), U_L)   # (N, R_D, R_L)
    G = G.transpose(1, 2)                          # (N, R_L, R_D)
    return G.reshape(G.shape[0], -1).numpy()


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_one(model_folder: str, dataset: str, idx: int = 1, total: int = 1):
    path = os.path.join(DATA_DIR, model_folder, f"{dataset}_pooled.pt")
    if not os.path.exists(path):
        print(f"  [{idx}/{total}] [SKIP] {path} not found")
        return None

    fsize = os.path.getsize(path) / 1e9
    print(f"\n{'=' * 60}")
    print(f"  [{idx}/{total}] {model_folder} / {dataset.upper()}")
    print(f"  File: {fsize:.1f} GB")
    print(f"{'=' * 60}")

    print(f"  [1/5] Loading ...", flush=True, end="")
    data = torch.load(path, weights_only=False)
    X_all = torch.stack(data["all_emb"])
    y_all = np.array([int(f) for f in data["all_hallucination_flag"]])
    is_known = np.array(data["all_is_known"])
    N_beams, L, D = X_all.shape
    n_prompts = len(is_known)
    prompt_idx = np.array(data.get("prompt_indices",
                                   data.get("all_prompt_indices",
                                   list(range(n_prompts)))))
    print(f" done.  {N_beams} beams, {L}x{D}")

    # HARP split: known prompts -> 75/25, all unknown -> valid
    print(f"  [2/5] Splitting (HARP known/unknown) ...", flush=True, end="")

    # HARP split: known prompts -> 75/25, all unknown -> valid
    known_prompt_idx = np.where(is_known)[0]
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(known_prompt_idx)
    split = int(len(known_prompt_idx) * 0.75)
    train_prompts = set(known_prompt_idx[:split])
    valid_prompts = set(known_prompt_idx[split:])

    # Add all unknown prompts to valid
    unknown_prompt_idx = np.where(~is_known)[0]
    valid_prompts.update(unknown_prompt_idx)

    # Map to beam indices
    train_mask = np.array([prompt_idx[i] in train_prompts
                           for i in range(N_beams)])
    valid_mask = np.array([prompt_idx[i] in valid_prompts
                           for i in range(N_beams)])

    X_train = X_all[train_mask]
    X_valid = X_all[valid_mask]
    y_train = y_all[train_mask]
    y_valid = y_all[valid_mask]

    print(f" done.  train={X_train.shape[0]} beams (hall:{y_train.mean():.1%})  "
          f"valid={X_valid.shape[0]} beams (hall:{y_valid.mean():.1%})")

    if X_train.shape[0] < 100 or X_valid.shape[0] < 50:
        print("  [WARN] Too few samples -- skipping")
        return None

    # HOSVD
    print(f"  [3/5] Computing HOSVD (L={L}, D={D}) ...", flush=True, end="")
    U_L, U_D = compute_ul_ud(X_train)
    print(f" done.  U_L: {tuple(U_L.shape)}  U_D: {tuple(U_D.shape)}")

    print(f"  [4/5] Projecting ...", flush=True, end="")
    X_train_feat = project(X_train, U_L, U_D)
    X_valid_feat = project(X_valid, U_L, U_D)
    print(f" done.  {X_train_feat.shape[1]} dim ({R_L} x {R_D})")

    # RandomForest
    print(f"  [5/5] Training RandomForest (200 trees) ...", flush=True, end="")
    rf = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED,
        class_weight="balanced", n_jobs=-1)
    rf.fit(X_train_feat, y_train)
    y_probs = rf.predict_proba(X_valid_feat)[:, 1]
    auroc = roc_auc_score(y_valid, y_probs)
    print(f" done.")

    print(f"\n  >> AUROC = {auroc:.4f} <<")
    return auroc


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    models = cfg["models"]
    if args.model_folder:
        models = [m for m in models if m["folder"] == args.model_folder]
    datasets = cfg["datasets"]
    if args.dataset:
        datasets = [d for d in datasets if d["name"] == args.dataset]

    total = len(models) * len(datasets)
    idx = 0
    results = {}
    for m in models:
        for d in datasets:
            idx += 1
            auroc = evaluate_one(m["folder"], d["name"], idx=idx, total=total)
            if auroc is not None:
                results[(m["folder"], d["name"])] = auroc

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    for (model, ds), auroc in sorted(results.items()):
        print(f"  {model:30s}  {ds:15s}  AUROC = {auroc:.4f}")
