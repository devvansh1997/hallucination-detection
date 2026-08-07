"""
05_evaluate_classifiers.py — HARP-Style Anomaly Detection Split
=================================================================
Replicates HARP's "Train-on-Truth-Only" split:
  - Known (truthful BEAMS) → 75% train / 25% test
  - Unknown (hallucinated BEAMS) → all to test
  - HOSVD factor matrices from TRAIN ONLY (zero leakage)
  - ContrastiveMLP matching HARP's architecture
  - AUROC via scikit-learn

Usage:
  python 05_evaluate_classifiers.py --model llama-3.1-8b-instruct --dataset triviaqa
  python 05_evaluate_classifiers.py  # runs all combos
"""

import argparse
import os

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
parser.add_argument("--model", type=str, default=None,
                    help="e.g. llama-3.1-8b-instruct")
parser.add_argument("--dataset", type=str, default=None,
                    help="e.g. triviaqa")
args = parser.parse_args()

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==============================================================================
# ContrastiveMLP  (matches HARP: Linear → ReLU → Linear → Sigmoid)
# ==============================================================================

class ContrastiveMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def train_mlp(X_train, y_train, input_dim, device, epochs=100):
    """Train ContrastiveMLP with HARP-compatible settings."""
    model = ContrastiveMLP(input_dim, hidden_dim=512).to(device)
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  loss={total_loss/len(loader):.4f}")

    model.eval()
    return model


# ==============================================================================
# HOSVD
# ==============================================================================

def compute_ul_ud(X_train):
    L, D = X_train.shape[1], X_train.shape[2]
    X_L = X_train.permute(1, 0, 2).reshape(L, -1).float()
    U_L, _, _ = torch.linalg.svd(X_L, full_matrices=False)
    U_L = U_L[:, :R_L]
    X_D = X_train.permute(2, 0, 1).reshape(D, -1).float()
    U_D, _, _ = torch.linalg.svd(X_D, full_matrices=False)
    U_D = U_D[:, :R_D]
    return U_L, U_D


def project(X, U_L, U_D):
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

    # HARP anomaly split: known = truthful beams, unknown = hallucinated beams
    known_mask = (y_all == 0)
    unknown_mask = (y_all == 1)

    n_known = known_mask.sum()
    n_unknown = unknown_mask.sum()
    print(f"  Beams: {N}  |  L={L}  D={D}")
    print(f"  Known (truthful):   {n_known}  ({n_known/N*100:.1f}%)")
    print(f"  Unknown (halluc.):  {n_unknown}  ({n_unknown/N*100:.1f}%)")

    if n_known < 10:
        print("  [SKIP] Too few truthful samples for train/test split")
        return None

    # Split KNOWN beams 75/25
    known_beams = X_all[known_mask]
    known_labels = y_all[known_mask]
    known_idx = np.arange(len(known_beams))
    train_idx, test_idx = train_test_split(
        known_idx, test_size=0.25, random_state=RANDOM_SEED)

    X_train = known_beams[train_idx]          # 75% truthful
    X_test_known = known_beams[test_idx]      # 25% truthful
    y_test_known = known_labels[test_idx]

    # Add ALL unknown beams to test
    X_test_unknown = X_all[unknown_mask]
    y_test_unknown = y_all[unknown_mask]

    X_test = torch.cat([X_test_known, X_test_unknown], dim=0)
    y_test = np.concatenate([y_test_known, y_test_unknown])

    print(f"  Train (75% truthful):      {X_train.shape[0]}")
    print(f"  Test (25% truthful):       {X_test_known.shape[0]}")
    print(f"  Test (+ all halluc.):      {X_test.shape[0]}"
          f"  (hall rate: {y_test.mean():.1%})")

    # HOSVD from TRAIN ONLY
    print(f"\n  Computing HOSVD from train ({X_train.shape[0]} truthful beams) ...")
    U_L, U_D = compute_ul_ud(X_train)

    X_train_feat = project(X_train, U_L, U_D)
    X_test_feat  = project(X_test,  U_L, U_D)
    input_dim = X_train_feat.shape[1]
    print(f"  Features: {input_dim} dim  ({R_L} x {R_D})")

    # Train ContrastiveMLP
    print(f"\n  Training ContrastiveMLP ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_mlp(X_train_feat, known_labels[train_idx], input_dim, device)

    # Predict and compute AUROC
    X_test_t = torch.tensor(X_test_feat, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_probs = model(X_test_t).cpu().numpy()
    auroc = roc_auc_score(y_test, y_probs)

    print(f"\n  AUROC: {auroc:.4f}")
    return auroc


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    models = cfg["models"]
    if args.model:
        models = [m for m in models if m["folder"] == args.model]
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
    print(f"  SUMMARY  (HARP split: 75% truthful → train, 25% + all halluc. → test)")
    print(f"{'=' * 60}")
    for (model, ds), auroc in sorted(results.items()):
        print(f"  {model:30s}  {ds:15s}  AUROC = {auroc:.4f}")
