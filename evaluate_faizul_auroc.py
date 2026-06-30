"""
Faizul Baseline — In-Distribution AUROC
=========================================
Extracts 8-dim Tucker features from the training set only, then
computes AUROC on a stratified 80/20 split using RandomForest.

Compares against Faizul's reported 0.844 and our HOSVD 0.958.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch.multiprocessing as mp

import torch

from data import RawActivations

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_PATH   = "../res2_train_32_layers_tensor.pt"
RANDOM_SEED  = 42
TUCKER_RANKS = [8, 6, 64]
MAX_WORKERS  = 3

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (same as evaluate_faizul.py — core-unfolding SVD)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_features_from_tensor(tensor, ranks):
    import tensorly as tl
    tl.set_backend("pytorch")
    r_L, r_T, r_D = ranks
    L, T, D = tensor.shape
    if T < r_T:
        r_T = T
    core, factors = tl.decomposition.tucker(tensor, rank=[r_L, r_T, r_D], n_iter_max=15)
    core_norm = float(torch.linalg.norm(core.flatten()))
    core_mode2 = tl.unfold(core, mode=2)
    core_mode2 = torch.as_tensor(core_mode2).float()
    s = torch.linalg.svdvals(core_mode2)
    s_norm = s / (s.sum() + 1e-9)
    s_norm = torch.clamp(s_norm, min=1e-9)
    entropy  = float(-torch.sum(s_norm * torch.log(s_norm)))
    eff_rank = float(torch.exp(torch.tensor(entropy)))
    top_k    = float(s_norm[:3].sum())
    return [entropy, eff_rank, top_k, core_norm]


def process_single_sample(payload):
    idx, H_cpu = payload
    device = torch.device("cuda")
    H = H_cpu.to(device)
    H_delta = H[1:] - H[:-1]
    f_H       = _extract_features_from_tensor(H, TUCKER_RANKS)
    f_H_delta = _extract_features_from_tensor(H_delta, TUCKER_RANKS)
    del H, H_delta
    torch.cuda.empty_cache()
    return idx, f_H + f_H_delta


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mp.freeze_support()

    print("=" * 60)
    print("  FAIZUL BASELINE — IN-DISTRIBUTION AUROC")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────
    print("\nLoading training data ...")
    ds = RawActivations(TRAIN_PATH)
    y = ds.y_train.numpy().astype(np.int64)
    N = ds.N
    print(f"  Samples: {N}  |  Truthful: {(y==0).sum()}  |  Halluc.: {(y==1).sum()}")

    # ── Extract 8-dim features ─────────────────────────────────────────
    print(f"\n{'-' * 60}")
    print(f"  Extracting 8-dim Faizul features "
          f"(Tucker ranks {TUCKER_RANKS}, {MAX_WORKERS} GPU workers)")
    print(f"{'-' * 60}")

    raw_list = ds.raw_activation_list
    X = np.empty((N, 8), dtype=np.float32)
    payloads = [(i, raw_list[i].cpu()) for i in range(N)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=MAX_WORKERS) as pool:
        for idx, feats in tqdm(
            pool.imap_unordered(process_single_sample, payloads, chunksize=1),
            total=N,
            desc="  TRAIN",
        ):
            X[idx] = np.array(feats, dtype=np.float32)

    print(f"  X shape: {X.shape}\n")

    # ── 80/20 split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"  Test hallucination rate: {y_test.sum()}/{len(y_test)} "
          f"({y_test.sum()/len(y_test)*100:.1f}%)")

    # ── Random Forest ──────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_probs = rf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_probs)

    # ── Report ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  IN-DISTRIBUTION AUROC RESULTS")
    print(f"{'=' * 60}")
    print(f"  Faizul Tucker (8-dim):     {auroc:.4f}")
    print(f"")
    print(f"  -- REFERENCE --")
    print(f"  Faizul reported AUROC:     0.844")
    print(f"  Our HOSVD (320-dim):       0.958")
    print(f"{'=' * 60}")
