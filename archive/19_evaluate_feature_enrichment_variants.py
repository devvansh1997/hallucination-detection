"""
19_evaluate_feature_enrichment_variants.py — Multi-Variant Ablation
====================================================================
Computes baseline HOSVD features (layers 15-24), engineers 3 geometric
metrics, and evaluates 5 feature variants across RF/LR/MLP in one pass.

Variant 1: Baseline (320-dim)
Variant 2: +Energy Magnitude (321-dim)
Variant 3: +SCQ (321-dim)
Variant 4: +Cosine Drift (321-dim)
Variant 5: Full Stack (323-dim)
"""

import argparse
import gc
import os
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = cfg["output"]["data_dir"]
R_L = 5
R_D = 64
RANDOM_SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, default=None)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--suffix", type=str, default="")
args = parser.parse_args()


def compute_ul_ud(X_train):
    N, L, D = X_train.shape
    X_f = X_train.permute(1, 0, 2).reshape(L, -1).float()
    A_L = X_f @ X_f.T
    _, U_L = torch.linalg.eigh(A_L.float())
    U_L = torch.flip(U_L[:, -R_L:], dims=[1])
    del X_f, A_L
    X_d = X_train.permute(2, 0, 1).reshape(D, -1)
    cols = N * L
    chunk_size = 50000
    A_D = torch.zeros(D, D, dtype=torch.float32)
    for start in range(0, cols, chunk_size):
        end = min(start + chunk_size, cols)
        chunk = X_d[:, start:end].float()
        A_D.addmm_(chunk, chunk.T)
    del X_d
    _, U_D = torch.linalg.eigh(A_D.float())
    U_D = torch.flip(U_D[:, -R_D:], dims=[1])
    del A_D
    return U_L, U_D


def project(X, U_L, U_D):
    temp = torch.matmul(X.float(), U_D)
    G = torch.matmul(temp.transpose(1, 2), U_L)
    G = G.transpose(1, 2)
    return G.reshape(G.shape[0], -1)


def evaluate_one(model_folder, dataset, idx=1, total=1):
    path = os.path.join(DATA_DIR, model_folder,
                        f"{dataset}_pooled{args.suffix}.pt")
    if not os.path.exists(path):
        return None

    print(f"\n{'=' * 60}")
    print(f"  [{idx}/{total}] {model_folder} / {dataset.upper()}")
    print(f"{'=' * 60}")

    data = torch.load(path, weights_only=False)
    X_all = torch.stack(data["all_emb"])
    y_all = np.array([int(f) for f in data["all_hallucination_flag"]])
    is_known = np.array(data["all_is_known"])

    # Hardcoded reasoning window
    X_all = X_all[:, 15:24, :]
    N_beams, L, D = X_all.shape
    n_prompts = len(is_known)
    prompt_idx = np.array(data.get("prompt_indices",
                                   data.get("all_prompt_indices",
                                   list(range(n_prompts)))))
    print(f"  Loaded: {N_beams} beams, {L}x{D} (layers 15-23)")

    # HARP split
    known_prompt_idx = np.where(is_known)[0]
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(known_prompt_idx)
    split = int(len(known_prompt_idx) * 0.75)
    train_prompts = set(known_prompt_idx[:split])
    valid_prompts = set(known_prompt_idx[split:])
    unknown_prompt_idx = np.where(~is_known)[0]
    valid_prompts.update(unknown_prompt_idx)
    train_mask = np.array([prompt_idx[i] in train_prompts for i in range(N_beams)])
    valid_mask = np.array([prompt_idx[i] in valid_prompts for i in range(N_beams)])
    train_idx_arr = np.where(train_mask)[0]
    valid_idx_arr = np.where(valid_mask)[0]
    y_train = y_all[train_idx_arr]
    y_valid = y_all[valid_idx_arr]
    print(f"  Split: train={len(train_idx_arr)}, valid={len(valid_idx_arr)}")

    # HOSVD
    U_L, U_D = compute_ul_ud(X_all[train_idx_arr])
    X_train_base = project(X_all[train_idx_arr], U_L, U_D)  # (Nt, 320)
    X_valid_base = project(X_all[valid_idx_arr], U_L, U_D)  # (Nv, 320)

    # --- Geometric metrics ---
    # Metric A: Energy magnitude
    mag_train = X_all[train_idx_arr].float().norm(dim=(1, 2)).unsqueeze(1).numpy()
    mag_valid = X_all[valid_idx_arr].float().norm(dim=(1, 2)).unsqueeze(1).numpy()

    # Metric B: SCQ (spectral concentration quotient from core tensor)
    # Recompute core per sample from HOSVD features
    G_train = project(X_all[train_idx_arr], U_L, U_D).numpy()  # (Nt, 320)
    G_train_img = G_train.reshape(-1, R_L, R_D)                 # (Nt, 5, 64)
    scq_train = []
    for g in G_train_img:
        s = np.linalg.svd(g, compute_uv=False)
        top3 = s[:3].sum()
        rest = s[3:].sum() + 1e-9
        scq_train.append(top3 / rest)
    scq_train = np.array(scq_train).reshape(-1, 1)

    G_valid = project(X_all[valid_idx_arr], U_L, U_D).numpy()
    G_valid_img = G_valid.reshape(-1, R_L, R_D)
    scq_valid = []
    for g in G_valid_img:
        s = np.linalg.svd(g, compute_uv=False)
        top3 = s[:3].sum()
        rest = s[3:].sum() + 1e-9
        scq_valid.append(top3 / rest)
    scq_valid = np.array(scq_valid).reshape(-1, 1)

    # Metric C: Trajectory cosine drift (first vs last layer)
    X_sliced = X_all.float()  # (N, 9, D)
    cos_train = F.cosine_similarity(
        X_sliced[train_idx_arr][:, 0, :],
        X_sliced[train_idx_arr][:, -1, :], dim=1).numpy().reshape(-1, 1)
    cos_valid = F.cosine_similarity(
        X_sliced[valid_idx_arr][:, 0, :],
        X_sliced[valid_idx_arr][:, -1, :], dim=1).numpy().reshape(-1, 1)

    del X_all, X_sliced, G_train_img, G_valid_img
    gc.collect()

    # --- Build 5 feature variants ---
    variants = {
        "V1: Baseline (320)":       (X_train_base, X_valid_base),
        "V2: +Magnitude (321)":     (np.hstack([X_train_base, mag_train]),
                                      np.hstack([X_valid_base, mag_valid])),
        "V3: +SCQ (321)":           (np.hstack([X_train_base, scq_train]),
                                      np.hstack([X_valid_base, scq_valid])),
        "V4: +Cosine Drift (321)":  (np.hstack([X_train_base, cos_train]),
                                      np.hstack([X_valid_base, cos_valid])),
        "V5: Full Stack (323)":     (np.hstack([X_train_base, mag_train, scq_train, cos_train]),
                                      np.hstack([X_valid_base, mag_valid, scq_valid, cos_valid])),
    }

    results = {}
    for name, (X_tr, X_va) in variants.items():
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        res = {}

        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                    random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_tr_s, y_train)
        res["RF"] = roc_auc_score(y_valid, rf.predict_proba(X_va_s)[:, 1])

        lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                                random_state=RANDOM_SEED)
        lr.fit(X_tr_s, y_train)
        res["LR"] = roc_auc_score(y_valid, lr.predict_proba(X_va_s)[:, 1])

        mlp = MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                            solver="adam", early_stopping=True,
                            n_iter_no_change=10, max_iter=1000,
                            random_state=RANDOM_SEED)
        mlp.fit(X_tr_s, y_train)
        res["MLP"] = roc_auc_score(y_valid, mlp.predict_proba(X_va_s)[:, 1])

        results[name] = res
        print(f"  {name:30s}  RF={res['RF']:.4f}  LR={res['LR']:.4f}  MLP={res['MLP']:.4f}")

    return results


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

    print(f"\n{'=' * 70}")
    print(f"  FEATURE ENRICHMENT ABLATION SUMMARY")
    print(f"{'=' * 70}")
    for (model, ds), variants in sorted(results.items()):
        print(f"\n  {model} / {ds}")
        print(f"  {'Variant':30s}  {'RF':>8s}  {'LR':>8s}  {'MLP':>8s}")
        print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
        for name, scores in variants.items():
            print(f"  {name:30s}  {scores['RF']:8.4f}  {scores['LR']:8.4f}  {scores['MLP']:8.4f}")
