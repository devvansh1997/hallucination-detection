"""
02_hosvd_evaluate.py — Flat 80/20 Split HOSVD AUROC (for existing data)
========================================================================
Quick evaluation for already-generated pooled tensors without prompt-level metadata.
Standard stratified 80/20 split on beams.  No known/unknown split.

Usage:
  python 02_eval_flat.py --model_folder llama-3.1-8b-instruct --dataset triviaqa
  python 02_eval_flat.py  # runs all model/dataset combos found in data/
"""

import argparse
import os

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, default=None)
parser.add_argument("--dataset", type=str, default=None)
args = parser.parse_args()


# ==============================================================================
# HOSVD
# ==============================================================================

def compute_ul_ud(X_train: torch.Tensor):
    L, D = X_train.shape[1], X_train.shape[2]
    X_L = X_train.permute(1, 0, 2).reshape(L, -1).float()
    U_L, _, _ = torch.linalg.svd(X_L, full_matrices=False)
    U_L = U_L[:, :R_L]
    X_D = X_train.permute(2, 0, 1).reshape(D, -1).float()
    U_D, _, _ = torch.linalg.svd(X_D, full_matrices=False)
    U_D = U_D[:, :R_D]
    return U_L, U_D


def project(X: torch.Tensor, U_L: torch.Tensor, U_D: torch.Tensor) -> np.ndarray:
    temp = torch.matmul(X.float(), U_D)
    G = torch.matmul(temp.transpose(1, 2), U_L)
    return G.transpose(1, 2).reshape(G.shape[0], -1).numpy()


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_one(model_folder: str, dataset: str):
    path = os.path.join(DATA_DIR, model_folder, f"{dataset}_pooled.pt")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None

    print(f"\n{'=' * 60}")
    print(f"  {model_folder}  |  {dataset.upper()}")
    print(f"{'=' * 60}")

    data = torch.load(path, weights_only=False)
    X_all = torch.stack(data["all_emb"])
    y_all = np.array([int(f) for f in data["all_hallucination_flag"]])

    N, L, D = X_all.shape
    print(f"  Beams: {N}  |  L={L}  D={D}  |  Hall rate: {y_all.mean():.1%}")

    # Standard 80/20 stratified split
    indices = np.arange(N)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y_all, random_state=RANDOM_SEED)

    X_train = X_all[train_idx]
    X_test  = X_all[test_idx]
    y_train = y_all[train_idx]
    y_test  = y_all[test_idx]

    print(f"  Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    U_L, U_D = compute_ul_ud(X_train)
    X_train_feat = project(X_train, U_L, U_D)
    X_test_feat  = project(X_test,  U_L, U_D)

    rf = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED,
        class_weight="balanced", n_jobs=-1)
    rf.fit(X_train_feat, y_train)
    y_probs = rf.predict_proba(X_test_feat)[:, 1]
    auroc = roc_auc_score(y_test, y_probs)

    print(f"  AUROC: {auroc:.4f}")
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

    results = {}
    for m in models:
        for d in datasets:
            auroc = evaluate_one(m["folder"], d["name"])
            if auroc is not None:
                results[(m["folder"], d["name"])] = auroc

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY  (flat 80/20 split)")
    print(f"{'=' * 60}")
    for (model, ds), auroc in sorted(results.items()):
        print(f"  {model:30s}  {ds:15s}  AUROC = {auroc:.4f}")
