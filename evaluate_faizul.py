"""
Faizul Baseline -- Per-Sample Tucker Decomposition (GPU MULTIPROCESS)
=====================================================================
Uses torch.multiprocessing with spawn context -- each worker gets its
own CUDA context, avoiding cuSOLVER's thread-safety limitation.

Tucker decomposition runs on GPU.  Limited to 3 concurrent processes
to stay within 12 GB VRAM budget.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from tqdm import tqdm
import torch.multiprocessing as mp

import torch

from data import RawActivations

# ==============================================================================
# CONFIGURATION
# ==============================================================================

TRAIN_PATH   = "../res2_train_32_layers_tensor.pt"
VAL_PATH     = "../res2_val32_tensor.pt"
RANDOM_SEED  = 42
TUCKER_RANKS = [8, 6, 64]
MAX_WORKERS  = 3           # GPU processes (cuSOLVER: 1 context per process)


# ==============================================================================
# GPU WORKER  (runs in child process -- fresh CUDA context)
# ==============================================================================

def _extract_features_from_tensor(tensor, ranks):
    """Tucker-decompose (on GPU) and return [entropy, eff_rank, top_k, core_norm].

    SPECTRAL FEATURES are computed from the CORE TENSOR's mode-2 unfolding
    (embedding axis), NOT the orthogonal factor matrix — whose singular
    values are trivially ~1.
    """
    import tensorly as tl
    tl.set_backend("pytorch")

    r_L, r_T, r_D = ranks
    L, T, D = tensor.shape
    if T < r_T:
        r_T = T

    core, factors = tl.decomposition.tucker(tensor, rank=[r_L, r_T, r_D], n_iter_max=15)

    # 1. Core norm (only feature that varies with the old code)
    core_norm = float(torch.linalg.norm(core.flatten()))

    # 2-4. Spectral features from CORE TENSOR mode-2 unfolding (embedding dim)
    # core shape: (r_L, r_T, r_D)  →  mode-2 unfold: (r_D, r_L * r_T)
    core_mode2 = tl.unfold(core, mode=2)               # tensorly tensor
    core_mode2 = torch.as_tensor(core_mode2).float()    # (r_D, r_L * r_T)

    s = torch.linalg.svdvals(core_mode2)                # singular values
    s_norm = s / (s.sum() + 1e-9)
    s_norm = torch.clamp(s_norm, min=1e-9)              # guard against log(0) NaN

    entropy  = float(-torch.sum(s_norm * torch.log(s_norm)))
    eff_rank = float(torch.exp(torch.tensor(entropy)))
    top_k    = float(s_norm[:3].sum())

    return [entropy, eff_rank, top_k, core_norm]


def process_single_sample(payload):
    """Worker entry point (pickle-safe).  payload = (idx, H_i_cpu).

    Returns (idx, 8-dim list of floats).
    """
    idx, H_cpu = payload
    device = torch.device("cuda")

    H = H_cpu.to(device)                        # (32, T, 4096) on GPU
    H_delta = H[1:] - H[:-1]                    # (31, T, 4096)

    f_H       = _extract_features_from_tensor(H, TUCKER_RANKS)
    f_H_delta = _extract_features_from_tensor(H_delta, TUCKER_RANKS)

    del H, H_delta
    torch.cuda.empty_cache()

    return idx, f_H + f_H_delta                  # 8 scalars


# ==============================================================================
# PARALLEL DISPATCH
# ==============================================================================

def process_dataset_parallel(dataset, label=""):
    """Distribute samples across GPU worker processes via imap_unordered."""
    import sys

    N = dataset.N
    raw_list = dataset.raw_activation_list
    features_flat = np.empty((N, 8), dtype=np.float32)

    payloads = [(i, raw_list[i].cpu()) for i in range(N)]

    print(f"\n  Processing {label} ({N} samples) "
          f"with {MAX_WORKERS} GPU workers ...", flush=True)
    print(f"  Launching worker pool (spawn -- ~10s cold start per process) ...",
          flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=MAX_WORKERS) as pool:
        for idx, feats in tqdm(
            pool.imap_unordered(process_single_sample, payloads, chunksize=1),
            total=N,
            desc=f"  {label}",
            file=sys.stderr,
            mininterval=1.0,
        ):
            features_flat[idx] = np.array(feats, dtype=np.float32)

    print(f"    Done -- X shape: {features_flat.shape}", flush=True)
    return features_flat


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    mp.freeze_support()

    print("=" * 60)
    print("  FAIZUL BASELINE -- GPU MULTIPROCESS TUCKER")
    print("=" * 60)

    print("\nLoading raw activation data ...")
    train_ds = RawActivations(TRAIN_PATH)
    val_ds   = RawActivations(VAL_PATH)
    print(f"  Train: {train_ds.N} samples  |  Val: {val_ds.N} samples")

    val_raw = torch.load(VAL_PATH, weights_only=False)
    assert all(val_raw["all_hallucination_flag"]), "Val labels not all True"
    print(f"  [PASS] All {val_ds.N} validation labels are True")

    print(f"\n{'-' * 60}")
    print(f"  Tucker ranks: {TUCKER_RANKS}  (dynamic: clamped to T if T < rank)")
    print(f"  Features per sample: 8 (4 H + 4 H_delta)")
    print(f"  Workers: {MAX_WORKERS} GPU processes (spawn -- separate CUDA contexts)")

    X_train = process_dataset_parallel(train_ds, label="TRAIN")
    X_val   = process_dataset_parallel(val_ds,   label="VAL")

    print(f"\n{'-' * 60}")
    print(f"  Training RandomForest on 8-dim Faizul features ...")

    y_train = train_ds.y_train.numpy().astype(np.int64)
    y_val   = val_ds.y_train.numpy().astype(np.int64)

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_val)
    rf_recall = recall_score(y_val, rf_preds)

    print(f"  RandomForest recall:  {rf_recall:.4f}  "
          f"({(rf_preds == 1).sum()}/{len(rf_preds)} detected)")

    n_val = val_ds.N
    print(f"\n{'=' * 60}")
    print(f"  FAIZUL BASELINE -- OOD RECALL")
    print(f"{'=' * 60}")
    print(f"  Method:        Per-sample Tucker decomposition (GPU)")
    print(f"  Features:      8-dim  (entropy, eff_rank, top-3, core_norm) x 2")
    print(f"  Tucker ranks:  {TUCKER_RANKS}  (dynamic for T < 6)")
    print(f"  Workers:       {MAX_WORKERS} GPU processes (spawn)")
    print(f"  Validation:    {n_val} samples, all hallucination")
    print(f"")
    print(f"  RandomForest recall:  {rf_recall:.4f}  ({rf_recall*n_val:.0f}/{n_val})")
    print(f"")
    print(f"  -- COMPARISON --")
    print(f"  HOSVD-only (320-dim):         0.717")
    print(f"  HOSVD + attention (352-dim):    0.717")
    print(f"  Faizul Tucker (8-dim):        {rf_recall:.3f}")
    print(f"{'=' * 60}")
