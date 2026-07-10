"""
17_evaluate_dynamic_egte.py — Dynamic Energy-Gated Tensor Extraction
=======================================================================
Automatically locates the reasoning window by computing layer-wise
L2 energy divergence, finding the peak in the middle 70% of layers,
and centering a 9-layer HOSVD window around it.
"""

import argparse
import os

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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

# ==============================================================================
# CLI
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, default=None)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--suffix", type=str, default="",
                    help="File suffix, e.g. _fullbeams for full-beam files")
args = parser.parse_args()


# ==============================================================================
# HOSVD CORE
# ==============================================================================

def compute_ul_ud(X_train: torch.Tensor):
    """X_train: (N, L, D). Returns U_L (L, R_L), U_D (D, R_D).
    Gram matrix trick with chunked matmul -- never materialises
    the full (D, N*L) unfolded matrix at float32."""
    N, L, D = X_train.shape

    # U_L via Gram: (L, N*D) matrix is small because L=28/32
    X_f = X_train.permute(1, 0, 2).reshape(L, -1).float()
    A_L = X_f @ X_f.T
    _, U_L = torch.linalg.eigh(A_L.float())
    U_L = torch.flip(U_L[:, -R_L:], dims=[1])
    del X_f, A_L

    # U_D via chunked Gram: A_D = sum(chunk @ chunk.T) over N*L axis
    X_d = X_train.permute(2, 0, 1).reshape(D, -1)          # stays bf16 (half size)
    cols = N * L
    chunk_size = 50000                                       # ~0.7 GB per chunk
    A_D = torch.zeros(D, D, dtype=torch.float32)
    for start in range(0, cols, chunk_size):
        end = min(start + chunk_size, cols)
        chunk = X_d[:, start:end].float()                   # cast just this slice
        A_D.addmm_(chunk, chunk.T)                           # accumulate Gram
    del X_d

    _, U_D = torch.linalg.eigh(A_D.float())
    U_D = torch.flip(U_D[:, -R_D:], dims=[1])
    del A_D

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
    path = os.path.join(DATA_DIR, model_folder,
                       f"{dataset}_pooled{args.suffix}.pt")
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

    # EGTE: dynamic reasoning window via energy divergence
    total_L = X_all.shape[1]
    X_f = X_all.float()
    energy = X_f.norm(dim=2)                                    # (N, L)
    energy_truth = energy[y_all == 0].mean(dim=0)               # (L,)
    energy_hall  = energy[y_all == 1].mean(dim=0)               # (L,)
    delta = (energy_truth - energy_hall).abs()                   # (L,)

    # Search middle 70% only (exclude first/last 15%)
    lo = int(total_L * 0.15)
    hi = int(total_L * 0.85)
    middle_delta = delta[lo:hi]
    peak_local = int(middle_delta.argmax().item())
    peak_layer = lo + peak_local

    # Center 9-layer window
    start_layer = max(0, peak_layer - 4)
    end_layer   = min(total_L, peak_layer + 5)
    X_all = X_all[:, start_layer:end_layer, :]
    window_L = X_all.shape[1]

    print(f" done.  {total_L}L, peak divergence at layer {peak_layer}")
    print(f"  EGTE window: layers {start_layer}:{end_layer} "
          f"({window_L} layers)")
    N_beams, L, D = X_all.shape
    n_prompts = len(is_known)
    prompt_idx = np.array(data.get("prompt_indices",
                                   data.get("all_prompt_indices",
                                   list(range(n_prompts)))))
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

    # Use indices NOT copies -- avoids duplicating the 27GB tensor in RAM
    train_idx_arr = np.where(train_mask)[0]
    valid_idx_arr = np.where(valid_mask)[0]

    X_train_n = len(train_idx_arr)
    X_valid_n = len(valid_idx_arr)
    y_train = y_all[train_idx_arr]
    y_valid = y_all[valid_idx_arr]

    print(f" done.  train={X_train_n} beams (hall:{y_train.mean():.1%})  "
          f"valid={X_valid_n} beams (hall:{y_valid.mean():.1%})")

    if X_train_n < 100 or X_valid_n < 50:
        print("  [WARN] Too few samples -- skipping")
        return None

    # HOSVD
    print(f"  [3/5] Computing HOSVD (L={L}, D={D}) ...", flush=True, end="")
    U_L, U_D = compute_ul_ud(X_all[train_idx_arr])
    print(f" done.  U_L: {tuple(U_L.shape)}  U_D: {tuple(U_D.shape)}")

    print(f"  [4/5] Projecting ...", flush=True, end="")
    X_train_feat = project(X_all[train_idx_arr], U_L, U_D)
    X_valid_feat = project(X_all[valid_idx_arr], U_L, U_D)
    print(f" done.  {X_train_feat.shape[1]} dim ({R_L} x {R_D})")

    # Free X_all now that we have features
    del X_all
    import gc; gc.collect()

    # Scale full training set, no subsampling
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_valid_feat = scaler.transform(X_valid_feat)
    print(f"  Features scaled (StandardScaler on {X_train_feat.shape[0]} samples)")

    results = {}

    # Random Forest (full train)
    print(f"  [5/5] Training RandomForest ...", flush=True, end="")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train_feat, y_train)
    rf_probs = rf.predict_proba(X_valid_feat)[:, 1]
    results["RandomForest"] = roc_auc_score(y_valid, rf_probs)
    print(f" done.")

    # Logistic Regression (full train)
    print(f"       Training LogisticRegression ...", flush=True, end="")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=RANDOM_SEED)
    lr.fit(X_train_feat, y_train)
    lr_probs = lr.predict_proba(X_valid_feat)[:, 1]
    results["LogisticRegression"] = roc_auc_score(y_valid, lr_probs)
    print(f" done.")

    # MLP with early stopping (full train)
    print(f"       Training MLP (128, early stopping) ...", flush=True, end="")
    mlp = MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                        solver="adam", early_stopping=True,
                        n_iter_no_change=10, max_iter=1000,
                        random_state=RANDOM_SEED)
    mlp.fit(X_train_feat, y_train)
    mlp_probs = mlp.predict_proba(X_valid_feat)[:, 1]
    results["MLP"] = roc_auc_score(y_valid, mlp_probs)
    print(f" done.")

    rf_auroc  = results["RandomForest"]
    lr_auroc  = results["LogisticRegression"]
    mlp_auroc = results["MLP"]
    print(f"\n  >> RandomForest         AUROC = {rf_auroc:.4f}")
    print(f"  >> LogisticRegression    AUROC = {lr_auroc:.4f}")
    print(f"  >> MLP (128)             AUROC = {mlp_auroc:.4f}")
    return results
    return results


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
            res = evaluate_one(m["folder"], d["name"], idx=idx, total=total)
            if res is not None:
                results[(m["folder"], d["name"])] = res

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    for (model, ds), res in sorted(results.items()):
        rf  = res.get("RandomForest", 0)
        lr  = res.get("LogisticRegression", 0)
        mlp = res.get("MLP", 0)
        print(f"  {model:30s}  {ds:15s}  RF={rf:.4f}  LR={lr:.4f}  MLP={mlp:.4f}")
