"""
13_evaluate_grid_search.py — Grid Search over HOSVD Hyperparams
==================================================================
Sweeps R_L, R_D, offset to test if low-variance eigenvectors
carry hallucination signal masked by top-rank semantic noise.

LR-only evaluation for fast iteration.
"""

import argparse
import os

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

def compute_ul_ud(X_train: torch.Tensor, r_l: int, r_d: int, offset: int):
    """Returns U_L (L, r_l), U_D (D, r_d) with eigenvector offset.
    offset=0 -> top r eigenvectors; offset=64 -> eigenvectors 64..64+r."""
    N, L, D = X_train.shape

    # U_L via Gram
    X_f = X_train.permute(1, 0, 2).reshape(L, -1).float()
    A_L = X_f @ X_f.T
    _, U = torch.linalg.eigh(A_L.float())
    U = torch.flip(U, dims=[1])                    # descending
    U_L = U[:, offset:offset + r_l]                # offset slice
    del X_f, A_L, U

    # U_D via chunked Gram
    X_d = X_train.permute(2, 0, 1).reshape(D, -1)
    cols = N * L
    chunk_size = 50000
    A_D = torch.zeros(D, D, dtype=torch.float32)
    for start in range(0, cols, chunk_size):
        end = min(start + chunk_size, cols)
        chunk = X_d[:, start:end].float()
        A_D.addmm_(chunk, chunk.T)
    del X_d

    _, U = torch.linalg.eigh(A_D.float())
    U = torch.flip(U, dims=[1])                    # descending
    U_D = U[:, offset:offset + r_d]                # offset slice
    del A_D, U

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

    # -- Grid search over hyperparams --
    R_L_vals   = [5, 16, 32]
    R_D_vals   = [64, 256, 512]
    OFFSET_vals = [0, 10, 64]

    grid_results = []
    total_combos = len(R_L_vals) * len(R_D_vals) * len(OFFSET_vals)
    combo_idx = 0
    X_train_pool = X_all[train_idx_arr]
    X_valid_pool = X_all[valid_idx_arr]

    for r_l in R_L_vals:
        for r_d in R_D_vals:
            for off in OFFSET_vals:
                combo_idx += 1

                # Clamp: offset + rank must not exceed eigenvectors available
                if off + r_l > L or off + r_d > D:
                    continue

                # Compute factor matrices (no caching per combo)
                U_L, U_D = compute_ul_ud(X_train_pool, r_l, r_d, off)

                # Project
                X_train_feat = project(X_train_pool, U_L, U_D)
                X_valid_feat = project(X_valid_pool, U_L, U_D)

                # Scale + LR
                scaler = StandardScaler()
                X_train_feat = scaler.fit_transform(X_train_feat)
                X_valid_feat = scaler.transform(X_valid_feat)

                lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                                        random_state=RANDOM_SEED)
                lr.fit(X_train_feat, y_train)
                lr_probs = lr.predict_proba(X_valid_feat)[:, 1]
                auroc = roc_auc_score(y_valid, lr_probs)

                grid_results.append((r_l, r_d, off, auroc))
                print(f"    [{combo_idx:2d}/{total_combos}] "
                      f"R_L={r_l:2d} R_D={r_d:3d} off={off:2d}  ->  AUROC={auroc:.4f}",
                      flush=True)

    # Free
    del X_all, X_train_pool, X_valid_pool
    import gc; gc.collect()

    # Print sorted summary
    grid_results.sort(key=lambda x: -x[3])
    print(f"\n  {'='*55}")
    print(f"  GRID SEARCH RESULTS  (best -> worst)")
    print(f"  {'='*55}")
    print(f"  {'R_L':>5s}  {'R_D':>5s}  {'offset':>6s}  {'AUROC':>8s}")
    print(f"  {'-'*30}")
    for r_l, r_d, off, auroc in grid_results:
        print(f"  {r_l:5d}  {r_d:5d}  {off:6d}  {auroc:8.4f}")

    # Save best factor matrices
    best = grid_results[0]
    best_ul, best_ud = compute_ul_ud(X_train_pool, best[0], best[1], best[2])
    best_cache = os.path.join(DATA_DIR, model_folder,
        f"{dataset}_ulud{args.suffix}.pt")
    torch.save({"U_L": best_ul, "U_D": best_ud,
                "R_L": best[0], "R_D": best[1], "offset": best[2],
                "AUROC": best[3]}, best_cache)
    print(f"\n  Best combo saved: R_L={best[0]} R_D={best[1]} "
          f"off={best[2]} AUROC={best[3]:.4f} -> {best_cache}")

    return grid_results
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
    for m in models:
        for d in datasets:
            idx += 1
            evaluate_one(m["folder"], d["name"], idx=idx, total=total)
