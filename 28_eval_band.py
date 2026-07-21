"""
28_eval_band.py -- Session 02: Reasoning-Band Grouped Evaluation
=====================================================================
CPU-only. Consumes the .npz produced by 27_extract_band.py plus the cached
Phase-1 pooled tensors (same as session01's 26_grouped_baseline.py, which
this script imports read-only -- it is never modified). Mirrors session01's
conventions exactly: seed=0, GroupKFold(5) by prompt, hallucination=positive
class, cluster bootstrap over prompts (1000 resamples) for every CI.

Usage:
  python 28_eval_band.py --self-test
  python 28_eval_band.py --band-npz <path> --model_folder llama-3.1-8b-instruct --dataset truthfulqa
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))

# --- read-only import of session01's script (never modified, no shared data.py exists) ---
_spec = importlib.util.spec_from_file_location("s01", os.path.join(HERE, "26_grouped_baseline.py"))
s01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s01)

SEED = 0
ORIGINAL_SEED = 42
N_SPLITS = 5
N_BOOTSTRAP = 1000
TOKEN_C_VALUES = [1.0, 0.1]   # primary, sensitivity -- never selected between


# ==============================================================================
# B1 -- FOLD-LOCAL ROBUST SCALING
# ==============================================================================

def fit_robust_scale(X_train_tokens):
    """X_train_tokens: (n_tokens, dim). Winsorize [0.5, 99.5] pct per channel,
    robust z-score by median/IQR, params fit train-only."""
    p_lo = np.percentile(X_train_tokens, 0.5, axis=0)
    p_hi = np.percentile(X_train_tokens, 99.5, axis=0)
    Xw = np.clip(X_train_tokens, p_lo, p_hi)
    median = np.median(Xw, axis=0)
    q75, q25 = np.percentile(Xw, 75, axis=0), np.percentile(Xw, 25, axis=0)
    iqr = q75 - q25
    scale = iqr / 1.349 + 1e-8
    return {"p_lo": p_lo, "p_hi": p_hi, "median": median, "scale": scale}


def apply_robust_scale(X, params):
    Xw = np.clip(X, params["p_lo"], params["p_hi"])
    Xz = (Xw - params["median"]) / params["scale"]
    return np.clip(Xz, -6, 6)


# ==============================================================================
# B3 -- PER-BEAM AGGREGATION
# ==============================================================================

EMPTY_AGG = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float64)


def aggregate_beam(token_scores):
    """token_scores: 1D array of P(hallucination) for this beam's tokens (may be empty)."""
    if token_scores.shape[0] == 0:
        return EMPTY_AGG.copy()
    T = token_scores.shape[0]
    return np.array([
        token_scores.max(),
        np.percentile(token_scores, 90),
        token_scores.mean(),
        token_scores[-1],
        (token_scores > 0.5).mean(),
        np.log(T),
    ], dtype=np.float64)


# B4 fusion (residualize the 6 aggregates against the 320 core features, train-fold
# least squares with intercept, then z-score residuals by train stats) is implemented
# inline in run_stream_condition() and run_harp_protocol() below, since both the
# fold-local core features and the fold-local train/val split are already in scope
# there and a standalone helper would need to take nearly every local as a parameter.


# ==============================================================================
# METRIC HELPERS (thin wrappers reusing session01's implementations exactly)
# ==============================================================================

def bootstrap_ci_auroc(scores, labels, prompt_ids, n_boot=N_BOOTSTRAP, seed=SEED):
    return s01.cluster_bootstrap_ci(scores, labels, prompt_ids, n_boot=n_boot, seed=seed)


def within_prompt_with_ci(scores, labels, prompt_ids, n_boot=N_BOOTSTRAP, seed=SEED):
    base = s01.within_prompt_auroc(scores, labels, prompt_ids)
    rng = np.random.default_rng(seed)
    unique_prompts = np.unique(prompt_ids)
    idx_by_prompt = {p: np.where(prompt_ids == p)[0] for p in unique_prompts}
    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(unique_prompts, size=len(unique_prompts), replace=True)
        beam_idx = np.concatenate([idx_by_prompt[p] for p in drawn])
        r = s01.within_prompt_auroc(scores[beam_idx], labels[beam_idx], prompt_ids[beam_idx])
        if r["n_pairs"] > 0 and not np.isnan(r["within_prompt_auroc"]):
            vals.append(r["within_prompt_auroc"])
    ci = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))
    base["ci95"] = ci
    return base


# ==============================================================================
# TOKEN STREAM SLICING (real-run named configs; self-test uses its own)
# ==============================================================================

REAL_BAND_CONDITIONS = {
    "band_primary_-272_-16": (-272, -16),     # r=256, darkest 16 dropped
    "band_ablation_-256_end": (-256, None),
    "band_ablation_-320_-64": (-320, -64),
}


def slice_band(z_band_full, lo, hi):
    """z_band_full: (n_tokens, full_band_dim). lo/hi are indices into the LAST axis,
    Python negative-index slicing semantics (hi=None means to the end)."""
    return z_band_full[:, lo:hi]


# ==============================================================================
# CORE EVALUATION PIPELINE (one run of: token-LR -> aggregate -> band-only / fused)
# ==============================================================================

def slice_tokens_for_beams(z_stream, offsets, beam_ids):
    """Concatenate a stream's tokens for a set of beam ids, and return the
    per-beam (start,end) offsets into the concatenated array."""
    chunks, local_offsets = [], [0]
    for b in beam_ids:
        s, e = offsets[b], offsets[b + 1]
        chunks.append(z_stream[s:e])
        local_offsets.append(local_offsets[-1] + (e - s))
    if chunks:
        return np.concatenate(chunks, axis=0), np.array(local_offsets, dtype=np.int64)
    return np.zeros((0, z_stream.shape[1]), dtype=z_stream.dtype), np.array(local_offsets, dtype=np.int64)


def run_stream_condition(z_stream, offsets, y, prompt_idx, core_by_fold, folds, token_C=1.0, seed=SEED):
    """z_stream: (total_tokens, dim) for ONE named band/rand condition (already sliced).
    offsets: (n_beams+1,) CSR offsets into z_stream.
    core_by_fold: list of (core_all [n_beams, core_dim]) aligned per fold (fold-pure Tucker core).
    folds: list of (train_beam_idx, val_beam_idx) arrays, len N_SPLITS, prompt-disjoint.
    Returns dict with band-only and fused (RF/LR) OOF scores + per-fold aurocs."""
    n_beams = len(offsets) - 1
    oof_band = np.full(n_beams, np.nan)
    oof_fused_rf = np.full(n_beams, np.nan)
    oof_fused_lr = np.full(n_beams, np.nan)
    fold_auroc_band, fold_auroc_fused_rf, fold_auroc_fused_lr = [], [], []
    weight_shares = []

    for fold_i, (tr_beam, va_beam) in enumerate(folds):
        tr_tok, tr_local_off = slice_tokens_for_beams(z_stream, offsets, tr_beam)
        va_tok, va_local_off = slice_tokens_for_beams(z_stream, offsets, va_beam)

        scale_params = fit_robust_scale(tr_tok) if tr_tok.shape[0] > 0 else None
        tr_tok_s = apply_robust_scale(tr_tok, scale_params) if tr_tok.shape[0] > 0 else tr_tok
        va_tok_s = apply_robust_scale(va_tok, scale_params) if va_tok.shape[0] > 0 else va_tok

        # B2: token-level weak-label LR, labels broadcast from beam label
        tr_tok_labels = np.repeat(y[tr_beam], np.diff(tr_local_off))
        if len(np.unique(tr_tok_labels)) < 2 or tr_tok_s.shape[0] == 0:
            tr_scores = np.full(tr_tok_s.shape[0], 0.5)
            va_scores = np.full(va_tok_s.shape[0], 0.5)
            coef_share = 0.0
        else:
            lr = LogisticRegression(class_weight="balanced", max_iter=3000, C=token_C,
                                     random_state=seed + fold_i)
            lr.fit(tr_tok_s, tr_tok_labels)
            tr_scores = lr.predict_proba(tr_tok_s)[:, 1]
            va_scores = lr.predict_proba(va_tok_s)[:, 1] if va_tok_s.shape[0] > 0 else np.zeros(0)
            coef = np.abs(lr.coef_[0])
            coef_share = float(coef[-1] / coef.sum()) if coef.sum() > 0 else 0.0
        weight_shares.append(coef_share)

        # B3: per-beam aggregation for train and val
        agg_train = np.stack([aggregate_beam(tr_scores[tr_local_off[i]:tr_local_off[i + 1]])
                               for i in range(len(tr_beam))])
        agg_val = np.stack([aggregate_beam(va_scores[va_local_off[i]:va_local_off[i + 1]])
                             for i in range(len(va_beam))])

        # band-only: fresh beam-level LR on raw (standardized) aggregates
        scaler = StandardScaler()
        agg_tr_z = scaler.fit_transform(agg_train)
        agg_va_z = scaler.transform(agg_val)
        band_lr = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed + fold_i)
        band_lr.fit(agg_tr_z, y[tr_beam])
        band_scores = band_lr.predict_proba(agg_va_z)[:, 1]
        oof_band[va_beam] = band_scores
        fold_auroc_band.append(float(roc_auc_score(y[va_beam], band_scores)))

        # B4: fused = [core, residualized aggregates]
        core_all_this_fold = core_by_fold[fold_i]
        core_train = core_all_this_fold[tr_beam]
        core_val = core_all_this_fold[va_beam]

        A_train = np.concatenate([core_train, np.ones((core_train.shape[0], 1))], axis=1)
        coef_reg, _, _, _ = np.linalg.lstsq(A_train, agg_train, rcond=None)
        agg_train_hat = A_train @ coef_reg
        resid_train = agg_train - agg_train_hat
        resid_mean, resid_std = resid_train.mean(axis=0), resid_train.std(axis=0) + 1e-8

        A_val = np.concatenate([core_val, np.ones((core_val.shape[0], 1))], axis=1)
        agg_val_hat = A_val @ coef_reg
        resid_val = agg_val - agg_val_hat

        resid_train_z = (resid_train - resid_mean) / resid_std
        resid_val_z = (resid_val - resid_mean) / resid_std

        fused_train = np.concatenate([core_train, resid_train_z], axis=1)
        fused_val = np.concatenate([core_val, resid_val_z], axis=1)

        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=seed + fold_i, n_jobs=-1)
        rf.fit(fused_train, y[tr_beam])
        fused_rf_scores = rf.predict_proba(fused_val)[:, 1]
        oof_fused_rf[va_beam] = fused_rf_scores
        fold_auroc_fused_rf.append(float(roc_auc_score(y[va_beam], fused_rf_scores)))

        fscaler = StandardScaler()
        fused_train_z = fscaler.fit_transform(fused_train)
        fused_val_z = fscaler.transform(fused_val)
        flr = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed + fold_i)
        flr.fit(fused_train_z, y[tr_beam])
        fused_lr_scores = flr.predict_proba(fused_val_z)[:, 1]
        oof_fused_lr[va_beam] = fused_lr_scores
        fold_auroc_fused_lr.append(float(roc_auc_score(y[va_beam], fused_lr_scores)))

    def summarize(oof, foldlist):
        ci = bootstrap_ci_auroc(oof, y, prompt_idx, seed=seed)
        return {"per_fold_auroc": foldlist, "mean_auroc": float(np.mean(foldlist)),
                "std_auroc": float(np.std(foldlist)), "pooled_oof_auroc": float(roc_auc_score(y, oof)),
                "ci95": ci}

    return {
        "band_only": {**summarize(oof_band, fold_auroc_band),
                      "within_prompt": within_prompt_with_ci(oof_band, y, prompt_idx, seed=seed)},
        "fused_RF": {**summarize(oof_fused_rf, fold_auroc_fused_rf),
                     "within_prompt": within_prompt_with_ci(oof_fused_rf, y, prompt_idx, seed=seed)},
        "fused_LR": {**summarize(oof_fused_lr, fold_auroc_fused_lr),
                     "within_prompt": within_prompt_with_ci(oof_fused_lr, y, prompt_idx, seed=seed)},
        # diagnostic: share of |coef| on the stream's LAST channel. On real data this is
        # just "is one projected dimension dominating the token classifier" (no literal
        # spike channel exists outside the self-test's synthetic construction).
        "token_lr_last_channel_weight_share_mean": float(np.mean(weight_shares)),
    }


def run_core_only(X_pooled, y, prompt_idx, folds, r_l, r_d, seed=SEED):
    """Session01-equivalent core-only RF/LR, recomputed under THIS run's own
    fold structure so deltas are internally consistent, plus core_by_fold for reuse."""
    n_beams = X_pooled.shape[0]
    oof_rf = np.full(n_beams, np.nan)
    oof_lr = np.full(n_beams, np.nan)
    fold_rf, fold_lr = [], []
    core_by_fold = []

    for fold_i, (tr_beam, va_beam) in enumerate(folds):
        X_scaled = s01.mad_scale(X_pooled, tr_beam)
        U_L, U_D = s01.compute_ul_ud(X_scaled[tr_beam], r_l, r_d)
        core = s01.project_core(X_scaled, U_L, U_D)
        core_by_fold.append(core)

        rf_scores = s01.fit_eval("RF", core[tr_beam], y[tr_beam], core[va_beam], seed + fold_i)
        oof_rf[va_beam] = rf_scores
        fold_rf.append(float(roc_auc_score(y[va_beam], rf_scores)))

        lr_scores = s01.fit_eval("LR", core[tr_beam], y[tr_beam], core[va_beam], seed + fold_i)
        oof_lr[va_beam] = lr_scores
        fold_lr.append(float(roc_auc_score(y[va_beam], lr_scores)))

    def summarize(oof, foldlist):
        ci = bootstrap_ci_auroc(oof, y, prompt_idx, seed=seed)
        return {"per_fold_auroc": foldlist, "mean_auroc": float(np.mean(foldlist)),
                "std_auroc": float(np.std(foldlist)), "pooled_oof_auroc": float(roc_auc_score(y, oof)),
                "ci95": ci}

    return {
        "RF": {**summarize(oof_rf, fold_rf), "within_prompt": within_prompt_with_ci(oof_rf, y, prompt_idx, seed=seed)},
        "LR": {**summarize(oof_lr, fold_lr), "within_prompt": within_prompt_with_ci(oof_lr, y, prompt_idx, seed=seed)},
    }, core_by_fold


# ==============================================================================
# B6 -- HARP-PROTOCOL READOUT (session01's exact E1 split, seed=42)
# ==============================================================================

def run_harp_protocol(X_pooled, y, prompt_idx, is_known, z_band_primary, offsets, r_l, r_d):
    n_beams = X_pooled.shape[0]
    t_idx, v_idx = s01.original_harp_split(is_known, prompt_idx, n_beams, seed=ORIGINAL_SEED)

    X_scaled = s01.mad_scale(X_pooled, t_idx)
    U_L, U_D = s01.compute_ul_ud(X_scaled[t_idx], r_l, r_d)
    core = s01.project_core(X_scaled, U_L, U_D)

    tr_tok, tr_off = slice_tokens_for_beams(z_band_primary, offsets, t_idx)
    va_tok, va_off = slice_tokens_for_beams(z_band_primary, offsets, v_idx)
    scale_params = fit_robust_scale(tr_tok)
    tr_tok_s = apply_robust_scale(tr_tok, scale_params)
    va_tok_s = apply_robust_scale(va_tok, scale_params)
    tr_tok_labels = np.repeat(y[t_idx], np.diff(tr_off))

    lr = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0, random_state=ORIGINAL_SEED)
    lr.fit(tr_tok_s, tr_tok_labels)
    tr_scores = lr.predict_proba(tr_tok_s)[:, 1]
    va_scores = lr.predict_proba(va_tok_s)[:, 1] if va_tok_s.shape[0] else np.zeros(0)

    agg_train = np.stack([aggregate_beam(tr_scores[tr_off[i]:tr_off[i + 1]]) for i in range(len(t_idx))])
    agg_val = np.stack([aggregate_beam(va_scores[va_off[i]:va_off[i + 1]]) for i in range(len(v_idx))])

    scaler = StandardScaler()
    band_lr = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=ORIGINAL_SEED)
    band_lr.fit(scaler.fit_transform(agg_train), y[t_idx])
    band_only_auroc = float(roc_auc_score(y[v_idx], band_lr.predict_proba(scaler.transform(agg_val))[:, 1]))

    A_train = np.concatenate([core[t_idx], np.ones((len(t_idx), 1))], axis=1)
    coef_reg, _, _, _ = np.linalg.lstsq(A_train, agg_train, rcond=None)
    resid_train = agg_train - A_train @ coef_reg
    resid_mean, resid_std = resid_train.mean(axis=0), resid_train.std(axis=0) + 1e-8
    A_val = np.concatenate([core[v_idx], np.ones((len(v_idx), 1))], axis=1)
    resid_val = (agg_val - A_val @ coef_reg - resid_mean) / resid_std
    resid_train_z = (resid_train - resid_mean) / resid_std

    fused_train = np.concatenate([core[t_idx], resid_train_z], axis=1)
    fused_val = np.concatenate([core[v_idx], resid_val], axis=1)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=ORIGINAL_SEED, n_jobs=-1)
    rf.fit(fused_train, y[t_idx])
    fused_auroc = float(roc_auc_score(y[v_idx], rf.predict_proba(fused_val)[:, 1]))

    return {"band_only_auroc": band_only_auroc, "fused_RF_auroc": fused_auroc,
            "n_train": int(len(t_idx)), "n_valid": int(len(v_idx))}


# ==============================================================================
# SELF-TEST (B8)
# ==============================================================================

def generate_synthetic_tokens(y, prompt_idx, seed=SEED, dim=32, spike_idx=-1):
    """Per-beam token streams: baseline noise; ~1/8 of a hallucinated beam's
    tokens get a planted directional signal; a rare spike channel gets huge
    values on ~1% of ALL tokens regardless of label (tests winsorization)."""
    rng = np.random.default_rng(seed)
    n_beams = len(y)
    v_signal = np.zeros(dim); v_signal[:5] = 1.0; v_signal /= np.linalg.norm(v_signal)

    per_beam_tokens = []
    for i in range(n_beams):
        T_i = int(rng.integers(4, 16))
        tok = rng.normal(0, 0.4, size=(T_i, dim))
        if y[i] == 1:
            n_signal_tok = max(1, T_i // 8)
            idx = rng.choice(T_i, size=n_signal_tok, replace=False)
            tok[idx] += 4.0 * v_signal
        spike_mask = rng.uniform(0, 1, size=T_i) < 0.01
        tok[spike_mask, spike_idx] = rng.choice([-1000.0, 1000.0], size=spike_mask.sum())
        per_beam_tokens.append(tok)

    offsets = np.zeros(n_beams + 1, dtype=np.int64)
    for i, tok in enumerate(per_beam_tokens):
        offsets[i + 1] = offsets[i] + tok.shape[0]
    z = np.concatenate(per_beam_tokens, axis=0).astype(np.float64)
    return z, offsets


def self_test():
    print("=" * 70)
    print("  SELF-TEST (synthetic prompts/beams/tokens, full B1-B6 path)")
    print("=" * 70)
    data = s01.generate_synthetic_data(n_prompts=200, beams_per_prompt=10, L=9, D=64, seed=SEED)
    X, y, prompt_idx = data["X"], data["y"], data["prompt_idx"]
    n_beams = data["n_beams"]
    r_l, r_d = 5, 20

    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = list(gkf.split(X, y, groups=prompt_idx))
    for fold_i, (tr, va) in enumerate(folds):
        assert set(prompt_idx[tr].tolist()).isdisjoint(set(prompt_idx[va].tolist())), \
            f"fold {fold_i}: train/val prompts not disjoint"
    print(f"  [PASS] {N_SPLITS} folds, all prompt-disjoint")

    z_band, offsets = generate_synthetic_tokens(y, prompt_idx, seed=SEED, dim=32, spike_idx=-1)
    z_rand, _ = generate_synthetic_tokens(np.zeros_like(y), prompt_idx, seed=SEED + 1, dim=32, spike_idx=-1)
    # rand control: no planted signal (labels zeroed out before generating -> no signal branch taken)

    core_results, core_by_fold = run_core_only(X, y, prompt_idx, folds, r_l, r_d, seed=SEED)
    print(f"  [INFO] core-only RF pooled AUROC = {core_results['RF']['pooled_oof_auroc']:.4f}")

    band_result = run_stream_condition(z_band, offsets, y, prompt_idx, core_by_fold, folds,
                                        token_C=1.0, seed=SEED)
    rand_result = run_stream_condition(z_rand, offsets, y, prompt_idx, core_by_fold, folds,
                                        token_C=1.0, seed=SEED)

    planted_auroc = band_result["band_only"]["pooled_oof_auroc"]
    print(f"  [INFO] planted-signal band-only AUROC = {planted_auroc:.4f}")
    assert planted_auroc > 0.75, f"planted-signal AUROC not > 0.75: {planted_auroc}"
    print(f"  [PASS] planted-signal band-only AUROC > 0.75")

    spike_share = band_result["token_lr_last_channel_weight_share_mean"]
    print(f"  [INFO] spike-channel learned weight share (mean over folds) = {spike_share:.4f}")
    assert spike_share < 0.3, f"spike channel weight share too large (winsorization not working): {spike_share}"
    print(f"  [PASS] spike-channel weight share is small (winsorization/clipping worked)")

    rand_auroc = rand_result["band_only"]["pooled_oof_auroc"]
    print(f"  [INFO] rand-control (no planted signal) band-only AUROC = {rand_auroc:.4f} "
          f"(expected ~0.5, sanity-check on the control construction)")

    out_path = os.path.join(HERE, "results", "session02_selftest_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"core_only": core_results, "band": band_result, "rand_control": rand_result}, f, indent=2)
    assert os.path.exists(out_path)
    print(f"  [PASS] JSON written to {out_path}")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def load_band_npz(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    shard_dir = os.path.dirname(meta_path)
    shard_paths = [os.path.join(shard_dir, s) for s in meta["shards"]]
    import importlib
    extract_mod_spec = importlib.util.spec_from_file_location(
        "s02_extract", os.path.join(HERE, "27_extract_band.py"))
    extract_mod = importlib.util.module_from_spec(extract_mod_spec)
    extract_mod_spec.loader.exec_module(extract_mod)
    packed = extract_mod.load_packed(shard_paths)
    return packed, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--band-meta", type=str, default=None,
                         help="Path to the *_meta.json written by 27_extract_band.py")
    parser.add_argument("--model_folder", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="truthfulqa")
    parser.add_argument("--pooled-suffix", type=str, default="_maxenergy")
    parser.add_argument("--r_l", type=int, default=5)
    parser.add_argument("--r_d", type=int, default=64)
    parser.add_argument("--output-json", type=str, default="results/session02_metrics.json")
    parser.add_argument("--session01-json", type=str, default="results/session01_metrics.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.band_meta or not args.model_folder:
        print("ERROR: --band-meta and --model_folder are required for a real run.")
        sys.exit(1)

    with open(args.session01_json) as f:
        s01_baseline = json.load(f)
    print(f"Loaded session01 baseline: E1={s01_baseline['E1']['auroc_canonical_orientation']:.4f}  "
          f"E2-RF={s01_baseline['E2']['RF']['pooled_oof_auroc']:.4f}")

    import yaml
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    pooled_path = os.path.join(cfg["output"]["data_dir"], args.model_folder,
                                f"{args.dataset}_pooled{args.pooled_suffix}.pt")
    data = s01.load_real_data(pooled_path)
    X, y, prompt_idx, is_known = data["X"], data["y"], data["prompt_idx"], data["is_known"]
    n_beams = data["n_beams"]

    packed, extract_meta = load_band_npz(args.band_meta)
    if len(packed["label"]) != n_beams:
        raise ValueError(f"band npz beam count ({len(packed['label'])}) != pooled data beam count "
                          f"({n_beams}). Refusing to guess an alignment.")
    if not np.array_equal(packed["label"], y):
        raise ValueError("band npz labels do not match pooled-data labels beam-for-beam. "
                          "Refusing to proceed with a possibly misaligned dataset.")

    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = list(gkf.split(X, y, groups=prompt_idx))
    for fold_i, (tr, va) in enumerate(folds):
        assert set(prompt_idx[tr].tolist()).isdisjoint(set(prompt_idx[va].tolist()))

    t0 = time.time()
    core_results, core_by_fold = run_core_only(X, y, prompt_idx, folds, args.r_l, args.r_d, seed=SEED)
    print(f"[core-only] RF pooled={core_results['RF']['pooled_oof_auroc']:.4f}  "
          f"LR pooled={core_results['LR']['pooled_oof_auroc']:.4f}  [{time.time()-t0:.0f}s]")
    delta_core = abs(core_results["RF"]["pooled_oof_auroc"] - s01_baseline["E2"]["RF"]["pooled_oof_auroc"])
    if delta_core > 0.005:
        print(f"  [WARN] core-only RF differs from session01 JSON by {delta_core:.4f} (> 0.005 threshold)")

    conditions = dict(REAL_BAND_CONDITIONS)
    results_by_condition = {}
    for cname, (lo, hi) in conditions.items():
        z_slice = slice_band(packed["z_band"], lo, hi)
        t0 = time.time()
        res = {}
        for C in TOKEN_C_VALUES:
            res[f"C{C}"] = run_stream_condition(z_slice, packed["offsets"], y, prompt_idx,
                                                 core_by_fold, folds, token_C=C, seed=SEED)
        results_by_condition[cname] = res
        print(f"[{cname}] band-only(C=1.0) pooled={res['C1.0']['band_only']['pooled_oof_auroc']:.4f}  "
              f"fused-RF(C=1.0) pooled={res['C1.0']['fused_RF']['pooled_oof_auroc']:.4f}  "
              f"[{time.time()-t0:.0f}s]")

    t0 = time.time()
    rand_res = {}
    for C in TOKEN_C_VALUES:
        rand_res[f"C{C}"] = run_stream_condition(packed["z_rand"], packed["offsets"], y, prompt_idx,
                                                  core_by_fold, folds, token_C=C, seed=SEED)
    print(f"[rand256] band-only(C=1.0) pooled={rand_res['C1.0']['band_only']['pooled_oof_auroc']:.4f}  "
          f"[{time.time()-t0:.0f}s]")

    primary_slice = REAL_BAND_CONDITIONS["band_primary_-272_-16"]
    z_primary = slice_band(packed["z_band"], *primary_slice)
    harp = run_harp_protocol(X, y, prompt_idx, is_known, z_primary, packed["offsets"], args.r_l, args.r_d)

    empty_share = extract_meta.get("n_empty_completions", 0) / max(extract_meta.get("n_beams", n_beams), 1)

    output = {
        "session01_reference": {
            "E1": s01_baseline["E1"]["auroc_canonical_orientation"],
            "E2_RF": s01_baseline["E2"]["RF"]["pooled_oof_auroc"],
            "E2_RF_ci95": s01_baseline["E2"]["RF"]["ci95"],
            "E3_within_prompt": s01_baseline["E3"]["within_prompt_auroc"],
        },
        "core_only_recomputed": core_results,
        "core_only_delta_from_session01": delta_core,
        "conditions": results_by_condition,
        "rand_control": rand_res,
        "harp_protocol_readout": harp,
        "extraction_metadata": extract_meta,
        "empty_completion_share": empty_share,
        "config": {"seed": SEED, "n_splits": N_SPLITS, "token_C_values": TOKEN_C_VALUES,
                   "r_l": args.r_l, "r_d": args.r_d},
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote: {args.output_json}")

    print(f"\n{'Row':30s}  {'pooled AUROC':>13s}  {'within-prompt':>13s}")
    primary = results_by_condition["band_primary_-272_-16"]["C1.0"]
    print(f"{'core-only (RF)':30s}  {core_results['RF']['pooled_oof_auroc']:13.4f}  "
          f"{core_results['RF']['within_prompt']['within_prompt_auroc']:13.4f}")
    print(f"{'band-only (primary)':30s}  {primary['band_only']['pooled_oof_auroc']:13.4f}  "
          f"{primary['band_only']['within_prompt']['within_prompt_auroc']:13.4f}")
    print(f"{'fused (primary, RF)':30s}  {primary['fused_RF']['pooled_oof_auroc']:13.4f}  "
          f"{primary['fused_RF']['within_prompt']['within_prompt_auroc']:13.4f}")
    r = rand_res["C1.0"]
    print(f"{'rand-control (band-only)':30s}  {r['band_only']['pooled_oof_auroc']:13.4f}  "
          f"{r['band_only']['within_prompt']['within_prompt_auroc']:13.4f}")
    for aname in ("band_ablation_-256_end", "band_ablation_-320_-64"):
        a = results_by_condition[aname]["C1.0"]
        print(f"{aname:30s}  {a['fused_RF']['pooled_oof_auroc']:13.4f}  "
              f"{a['fused_RF']['within_prompt']['within_prompt_auroc']:13.4f}")
    print(f"{'HARP-protocol band-only':30s}  {harp['band_only_auroc']:13.4f}")
    print(f"{'HARP-protocol fused-RF':30s}  {harp['fused_RF_auroc']:13.4f}")
    print(f"{'  (session01 E1 reference)':30s}  {s01_baseline['E1']['auroc_canonical_orientation']:13.4f}")


if __name__ == "__main__":
    main()
