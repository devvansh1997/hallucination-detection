"""
31_eval_session03.py -- Session 03 Parts A+B: Nonlinear Readouts + Within-Prompt Contrast
==============================================================================================
CPU-only. Consumes the manifest-pinned session02 seeded artifacts ONLY -- never loads the LLM,
never regenerates beams, never recomputes labels. Hard-fails via 30_pin_manifest.verify_manifest()
if the on-disk files have drifted from what was pinned.

Part A: paired band-vs-random comparison under two nonlinear token-level readouts (quadratic
structured head, token-level Random Forest), testing whether session02's null result was about
the readout (linear) rather than the subspace.

Part B: within-prompt contrast features (per-beam delta from prompt centroid, PCA of deltas,
cross-beam Gram-matrix spectral statistics) -- a second, independent feature stream that uses
only the beam set already on disk, no new extraction.

Usage:
  python 31_eval_session03.py --self-test
  python 31_eval_session03.py --model_folder llama-3.1-8b-instruct --dataset truthfulqa
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s01 = _load("s01", "26_grouped_baseline.py")
band_mod = _load("s02_extract", "27_extract_band.py")
s02 = _load("s02", "28_eval_band.py")
pin_mod = _load("s03_pin", "30_pin_manifest.py")

SEED = 0
N_SPLITS = 5
N_BOOTSTRAP = 1000
PRIMARY_BAND_SLICE = (-272, -16)   # r=256, same primary config as session02


# ==============================================================================
# SHARED HELPERS
# ==============================================================================

def residualize_fuse_eval(core_train, core_val, feat_train, feat_val, y_tr, y_va, seed):
    """Fit feat ~ core (least squares, intercept) on train; residualize both splits;
    z-score residuals by train stats; concat [core, residual_z]; fit RF + LR."""
    A_train = np.concatenate([core_train, np.ones((core_train.shape[0], 1))], axis=1)
    coef_reg, _, _, _ = np.linalg.lstsq(A_train, feat_train, rcond=None)
    resid_train = feat_train - A_train @ coef_reg
    resid_mean, resid_std = resid_train.mean(axis=0), resid_train.std(axis=0) + 1e-8

    A_val = np.concatenate([core_val, np.ones((core_val.shape[0], 1))], axis=1)
    resid_val = (feat_val - A_val @ coef_reg - resid_mean) / resid_std
    resid_train_z = (resid_train - resid_mean) / resid_std

    fused_train = np.concatenate([core_train, resid_train_z], axis=1)
    fused_val = np.concatenate([core_val, resid_val], axis=1)

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1)
    rf.fit(fused_train, y_tr)
    rf_scores = rf.predict_proba(fused_val)[:, 1]

    scaler = StandardScaler()
    lr = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed)
    lr.fit(scaler.fit_transform(fused_train), y_tr)
    lr_scores = lr.predict_proba(scaler.transform(fused_val))[:, 1]
    return rf_scores, lr_scores


def summarize_oof(oof, y, prompt_idx, foldlist, seed=SEED):
    ci = s02.bootstrap_ci_auroc(oof, y, prompt_idx, seed=seed)
    wp = s02.within_prompt_with_ci(oof, y, prompt_idx, seed=seed)
    return {"per_fold_auroc": foldlist, "mean_auroc": float(np.mean(foldlist)),
            "std_auroc": float(np.std(foldlist)), "pooled_oof_auroc": float(roc_auc_score(y, oof)),
            "ci95": ci, "within_prompt": wp}


def paired_bootstrap_delta(scores_a, scores_b, y, prompt_idx, n_boot=N_BOOTSTRAP, seed=SEED,
                            within_prompt=False):
    """delta = a - b, same resample used for both (paired), CI on the delta."""
    rng = np.random.default_rng(seed)
    unique_prompts = np.unique(prompt_idx)
    idx_by_prompt = {p: np.where(prompt_idx == p)[0] for p in unique_prompts}
    deltas = []
    for _ in range(n_boot):
        drawn = rng.choice(unique_prompts, size=len(unique_prompts), replace=True)
        beam_idx = np.concatenate([idx_by_prompt[p] for p in drawn])
        y_r = y[beam_idx]
        if within_prompt:
            r_a = s01.within_prompt_auroc(scores_a[beam_idx], y_r, prompt_idx[beam_idx])
            r_b = s01.within_prompt_auroc(scores_b[beam_idx], y_r, prompt_idx[beam_idx])
            if r_a["n_pairs"] == 0 or r_b["n_pairs"] == 0:
                continue
            deltas.append(r_a["within_prompt_auroc"] - r_b["within_prompt_auroc"])
        else:
            if len(np.unique(y_r)) < 2:
                continue
            deltas.append(float(roc_auc_score(y_r, scores_a[beam_idx]) - roc_auc_score(y_r, scores_b[beam_idx])))
    if not deltas:
        return {"mean_delta": float("nan"), "ci95": (float("nan"), float("nan")), "excludes_zero": False}
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return {"mean_delta": float(np.mean(deltas)), "ci95": (lo, hi), "excludes_zero": bool(lo > 0 or hi < 0)}


def paired_fold_deltas(fold_a, fold_b):
    return [float(a - b) for a, b in zip(fold_a, fold_b)]


# ==============================================================================
# PART A -- TOKEN SCORERS (pluggable)
# ==============================================================================

def degree2_expand(Z):
    """Z: (n, k) -> [Z, upper_triangle(outer(z,z)) incl diagonal], (n, k + k*(k+1)/2)."""
    n, k = Z.shape
    iu = np.triu_indices(k)
    outer = np.einsum("ni,nj->nij", Z, Z)
    upper = outer[:, iu[0], iu[1]]
    return np.concatenate([Z, upper], axis=1)


def quad_token_scorer(tr_tok_s, tr_tok_labels, va_tok_s, seed, tr_beam_offsets=None):
    if tr_tok_s.shape[0] == 0 or len(np.unique(tr_tok_labels)) < 2:
        return (np.full(tr_tok_s.shape[0], 0.5), np.full(va_tok_s.shape[0], 0.5), {})
    k = min(32, tr_tok_s.shape[1], max(1, tr_tok_s.shape[0] - 1))
    pca = PCA(n_components=k, random_state=seed)
    tr_pca = pca.fit_transform(tr_tok_s)
    va_pca = pca.transform(va_tok_s) if va_tok_s.shape[0] else np.zeros((0, k))
    tr_deg2 = degree2_expand(tr_pca)
    va_deg2 = degree2_expand(va_pca)
    lr = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0, random_state=seed)
    lr.fit(tr_deg2, tr_tok_labels)
    tr_scores = lr.predict_proba(tr_deg2)[:, 1]
    va_scores = lr.predict_proba(va_deg2)[:, 1] if va_deg2.shape[0] else np.zeros(0)
    return tr_scores, va_scores, {"pca_k": k}


def rf_token_scorer(tr_tok_s, tr_tok_labels, va_tok_s, seed, tr_beam_offsets=None,
                     max_train_tokens=60000, time_budget_s=900):
    if tr_tok_s.shape[0] == 0 or len(np.unique(tr_tok_labels)) < 2:
        return (np.full(tr_tok_s.shape[0], 0.5), np.full(va_tok_s.shape[0], 0.5), {"subsampled": False})

    fit_tok, fit_labels = tr_tok_s, tr_tok_labels
    subsampled = False
    if tr_tok_s.shape[0] > max_train_tokens and tr_beam_offsets is not None:
        # stratified by beam: keep whole beams, drop others, until under the token budget
        rng = np.random.default_rng(seed)
        n_beams_fold = len(tr_beam_offsets) - 1
        order = rng.permutation(n_beams_fold)
        keep_mask = np.zeros(tr_tok_s.shape[0], dtype=bool)
        running = 0
        for b in order:
            s, e = tr_beam_offsets[b], tr_beam_offsets[b + 1]
            if running + (e - s) > max_train_tokens and running > 0:
                continue
            keep_mask[s:e] = True
            running += (e - s)
            if running >= max_train_tokens:
                break
        fit_tok, fit_labels = tr_tok_s[keep_mask], tr_tok_labels[keep_mask]
        subsampled = True

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                 class_weight="balanced", random_state=seed, n_jobs=-1)
    rf.fit(fit_tok, fit_labels)
    fit_time = time.time() - t0
    if fit_time > time_budget_s:
        print(f"    [WARN] token-RF fold fit took {fit_time:.0f}s (> {time_budget_s}s budget)")

    tr_scores = rf.predict_proba(tr_tok_s)[:, 1]
    va_scores = rf.predict_proba(va_tok_s)[:, 1] if va_tok_s.shape[0] else np.zeros(0)
    return tr_scores, va_scores, {"subsampled": subsampled, "fit_time_s": fit_time,
                                   "n_fit_tokens": int(fit_tok.shape[0])}


def linear_token_scorer(tr_tok_s, tr_tok_labels, va_tok_s, seed, tr_beam_offsets=None):
    """session02's original readout, kept for continuity in the paired comparison."""
    if tr_tok_s.shape[0] == 0 or len(np.unique(tr_tok_labels)) < 2:
        return (np.full(tr_tok_s.shape[0], 0.5), np.full(va_tok_s.shape[0], 0.5), {})
    lr = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0, random_state=seed)
    lr.fit(tr_tok_s, tr_tok_labels)
    tr_scores = lr.predict_proba(tr_tok_s)[:, 1]
    va_scores = lr.predict_proba(va_tok_s)[:, 1] if va_tok_s.shape[0] else np.zeros(0)
    return tr_scores, va_scores, {}


# ==============================================================================
# PART A -- CONDITION RUNNER
# ==============================================================================

def run_readout_condition(scorer_fn, z_stream, offsets, y, prompt_idx, core_by_fold, folds, seed=SEED):
    n_beams = len(offsets) - 1
    oof_band = np.full(n_beams, np.nan)
    oof_fused_rf = np.full(n_beams, np.nan)
    oof_fused_lr = np.full(n_beams, np.nan)
    fold_band, fold_fused_rf, fold_fused_lr = [], [], []
    per_token_aurocs, fold_diag = [], []

    for fold_i, (tr_beam, va_beam) in enumerate(folds):
        tr_tok, tr_off = s02.slice_tokens_for_beams(z_stream, offsets, tr_beam)
        va_tok, va_off = s02.slice_tokens_for_beams(z_stream, offsets, va_beam)

        scale_params = s02.fit_robust_scale(tr_tok) if tr_tok.shape[0] > 0 else None
        tr_tok_s = s02.apply_robust_scale(tr_tok, scale_params) if tr_tok.shape[0] > 0 else tr_tok
        va_tok_s = s02.apply_robust_scale(va_tok, scale_params) if va_tok.shape[0] > 0 else va_tok

        tr_tok_labels = np.repeat(y[tr_beam], np.diff(tr_off))
        va_tok_labels = np.repeat(y[va_beam], np.diff(va_off))

        t0 = time.time()
        tr_scores, va_scores, diag = scorer_fn(tr_tok_s, tr_tok_labels, va_tok_s, seed + fold_i, tr_off)
        diag = dict(diag); diag["total_fold_time_s"] = time.time() - t0
        fold_diag.append(diag)

        if va_scores.shape[0] and len(np.unique(va_tok_labels)) > 1:
            per_token_aurocs.append(float(roc_auc_score(va_tok_labels, va_scores)))
        else:
            per_token_aurocs.append(float("nan"))

        agg_train = np.stack([s02.aggregate_beam(tr_scores[tr_off[i]:tr_off[i + 1]])
                               for i in range(len(tr_beam))])
        agg_val = np.stack([s02.aggregate_beam(va_scores[va_off[i]:va_off[i + 1]])
                             for i in range(len(va_beam))])

        scaler = StandardScaler()
        band_lr = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed + fold_i)
        band_lr.fit(scaler.fit_transform(agg_train), y[tr_beam])
        band_scores = band_lr.predict_proba(scaler.transform(agg_val))[:, 1]
        oof_band[va_beam] = band_scores
        fold_band.append(float(roc_auc_score(y[va_beam], band_scores)))

        core_this = core_by_fold[fold_i]
        rf_scores, lr_scores = residualize_fuse_eval(core_this[tr_beam], core_this[va_beam],
                                                       agg_train, agg_val, y[tr_beam], y[va_beam], seed + fold_i)
        oof_fused_rf[va_beam] = rf_scores; fold_fused_rf.append(float(roc_auc_score(y[va_beam], rf_scores)))
        oof_fused_lr[va_beam] = lr_scores; fold_fused_lr.append(float(roc_auc_score(y[va_beam], lr_scores)))

    return {
        "band_only": summarize_oof(oof_band, y, prompt_idx, fold_band, seed),
        "fused_RF": summarize_oof(oof_fused_rf, y, prompt_idx, fold_fused_rf, seed),
        "fused_LR": summarize_oof(oof_fused_lr, y, prompt_idx, fold_fused_lr, seed),
        "per_token_auroc_by_fold": per_token_aurocs,
        "fold_diagnostics": fold_diag,
        "_oof": {"band": oof_band, "fused_RF": oof_fused_rf, "fused_LR": oof_fused_lr},
    }


# ==============================================================================
# PART B -- WITHIN-PROMPT CONTRAST + CROSS-BEAM GRAM
# ==============================================================================

def compute_contrast_block(core_fold, prompt_idx, tr_beam, seed):
    """core_fold: (n_beams, core_dim) fold-pure core features for THIS fold.
    Returns (block [n_beams, 39], trace_by_prompt dict)."""
    n_beams = core_fold.shape[0]
    unique_prompts = np.unique(prompt_idx)
    delta = np.zeros_like(core_fold)
    for p in unique_prompts:
        idx = np.where(prompt_idx == p)[0]
        mu = core_fold[idx].mean(axis=0)
        delta[idx] = core_fold[idx] - mu

    pca = PCA(n_components=32, random_state=seed)
    pca.fit(delta[tr_beam])
    delta_pca_all = pca.transform(delta)   # (n_beams, 32), fold-local unsupervised

    delta_norm = np.linalg.norm(delta, axis=1)
    leverage = np.zeros(n_beams)
    for p in unique_prompts:
        idx = np.where(prompt_idx == p)[0]
        total_sq = (delta_norm[idx] ** 2).sum()
        leverage[idx] = (delta_norm[idx] ** 2) / (total_sq + 1e-12)

    log_lambda1 = np.zeros(n_beams)
    spec_entropy = np.zeros(n_beams)
    participation_ratio = np.zeros(n_beams)
    log_trace = np.zeros(n_beams)
    abs_proj_v1 = np.zeros(n_beams)
    trace_by_prompt = {}

    for p in unique_prompts:
        idx = np.where(prompt_idx == p)[0]
        D = delta[idx]                     # (m, core_dim), already centered by construction
        G = D @ D.T                        # (m, m), rank <= m-1
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals = np.clip(np.flip(eigvals), 0, None)
        eigvecs = np.flip(eigvecs, axis=1)
        lam1 = eigvals[0]
        trace = float(eigvals.sum())
        trace_by_prompt[int(p)] = trace

        s = eigvals.sum()
        p_i = eigvals / (s + 1e-12)
        p_nz = p_i[p_i > 1e-12]
        entropy = float(-np.sum(p_nz * np.log(p_nz))) if p_nz.size else 0.0
        pr = float((s ** 2) / ((eigvals ** 2).sum() + 1e-12))

        if lam1 > 1e-10:
            u1 = eigvecs[:, 0]
            v1 = D.T @ u1 / np.sqrt(lam1)          # (core_dim,), unit norm (SVD identity)
            proj = np.abs(D @ v1)
        else:
            proj = np.zeros(D.shape[0])

        log_lambda1[idx] = np.log(lam1 + 1e-8)
        spec_entropy[idx] = entropy
        participation_ratio[idx] = pr
        log_trace[idx] = np.log(trace + 1e-8)
        abs_proj_v1[idx] = proj

    block = np.concatenate([
        delta_pca_all, delta_norm[:, None], leverage[:, None],
        log_lambda1[:, None], spec_entropy[:, None], participation_ratio[:, None],
        log_trace[:, None], abs_proj_v1[:, None],
    ], axis=1)
    return block, trace_by_prompt


def run_contrast_condition(core_by_fold, y, prompt_idx, folds, seed=SEED):
    n_beams = len(y)
    oof_block_rf = np.full(n_beams, np.nan); oof_block_lr = np.full(n_beams, np.nan)
    oof_fused_rf = np.full(n_beams, np.nan); oof_fused_lr = np.full(n_beams, np.nan)
    fold_block_rf, fold_block_lr, fold_fused_rf, fold_fused_lr = [], [], [], []
    trace_by_prompt_oof = {}

    for fold_i, (tr_beam, va_beam) in enumerate(folds):
        core_this = core_by_fold[fold_i]
        block, trace_by_prompt = compute_contrast_block(core_this, prompt_idx, tr_beam, seed + fold_i)
        for p in np.unique(prompt_idx[va_beam]):
            trace_by_prompt_oof[int(p)] = trace_by_prompt[int(p)]

        block_train, block_val = block[tr_beam], block[va_beam]

        rf_b = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=seed + fold_i, n_jobs=-1)
        rf_b.fit(block_train, y[tr_beam])
        rfb_scores = rf_b.predict_proba(block_val)[:, 1]
        oof_block_rf[va_beam] = rfb_scores; fold_block_rf.append(float(roc_auc_score(y[va_beam], rfb_scores)))

        scaler = StandardScaler()
        lr_b = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed + fold_i)
        lr_b.fit(scaler.fit_transform(block_train), y[tr_beam])
        lrb_scores = lr_b.predict_proba(scaler.transform(block_val))[:, 1]
        oof_block_lr[va_beam] = lrb_scores; fold_block_lr.append(float(roc_auc_score(y[va_beam], lrb_scores)))

        core_train, core_val = core_this[tr_beam], core_this[va_beam]
        rf_f_scores, lr_f_scores = residualize_fuse_eval(core_train, core_val, block_train, block_val,
                                                           y[tr_beam], y[va_beam], seed + fold_i)
        oof_fused_rf[va_beam] = rf_f_scores; fold_fused_rf.append(float(roc_auc_score(y[va_beam], rf_f_scores)))
        oof_fused_lr[va_beam] = lr_f_scores; fold_fused_lr.append(float(roc_auc_score(y[va_beam], lr_f_scores)))

    return {
        "block_only_RF": summarize_oof(oof_block_rf, y, prompt_idx, fold_block_rf, seed),
        "block_only_LR": summarize_oof(oof_block_lr, y, prompt_idx, fold_block_lr, seed),
        "fused_RF": summarize_oof(oof_fused_rf, y, prompt_idx, fold_fused_rf, seed),
        "fused_LR": summarize_oof(oof_fused_lr, y, prompt_idx, fold_fused_lr, seed),
        "trace_Gp_by_prompt": trace_by_prompt_oof,
        "_oof": {"block_RF": oof_block_rf, "fused_RF": oof_fused_rf},
    }


def summarize_trace_distribution(trace_by_prompt):
    vals = np.array(list(trace_by_prompt.values()))
    pcts = {f"p{p}": float(np.percentile(vals, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
    thresholds = {}
    for thr in (1e-2, 1e-4, 1e-6):
        thresholds[f"n_below_{thr}"] = int((vals < thr).sum())
    return {"n_prompts": len(vals), "min": float(vals.min()), "max": float(vals.max()),
            "mean": float(vals.mean()), "percentiles": pcts, "near_zero_counts": thresholds}


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: Parts A + B end-to-end on synthetic data")
    print("=" * 70)
    data = s01.generate_synthetic_data(n_prompts=200, beams_per_prompt=10, L=9, D=64, seed=SEED)
    X, y, prompt_idx = data["X"], data["y"], data["prompt_idx"]
    r_l, r_d = 5, 20

    folds = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups=prompt_idx))
    for fold_i, (tr, va) in enumerate(folds):
        assert set(prompt_idx[tr].tolist()).isdisjoint(set(prompt_idx[va].tolist())), \
            f"fold {fold_i} not prompt-disjoint"
    print(f"  [PASS] {N_SPLITS} folds, all prompt-disjoint")

    core_results, core_by_fold = s02.run_core_only(X, y, prompt_idx, folds, r_l, r_d, seed=SEED)
    print(f"  [INFO] core-only RF pooled = {core_results['RF']['pooled_oof_auroc']:.4f}")

    # -- Part A: reuse session02's synthetic token generator (planted signal + spike channel) --
    z_band, offsets = s02.generate_synthetic_tokens(y, prompt_idx, seed=SEED, dim=32, spike_idx=-1)
    z_rand, _ = s02.generate_synthetic_tokens(np.zeros_like(y), prompt_idx, seed=SEED + 1, dim=32, spike_idx=-1)

    t0 = time.time()
    quad_band = run_readout_condition(quad_token_scorer, z_band, offsets, y, prompt_idx, core_by_fold, folds)
    quad_rand = run_readout_condition(quad_token_scorer, z_rand, offsets, y, prompt_idx, core_by_fold, folds)
    print(f"  [PASS] quadratic readout ran on band+rand [{time.time()-t0:.0f}s]  "
          f"band band-only={quad_band['band_only']['pooled_oof_auroc']:.4f}  "
          f"rand band-only={quad_rand['band_only']['pooled_oof_auroc']:.4f}")
    assert quad_band["band_only"]["pooled_oof_auroc"] > 0.6, "quadratic readout found no planted signal"

    t0 = time.time()
    rf_band = run_readout_condition(rf_token_scorer, z_band, offsets, y, prompt_idx, core_by_fold, folds)
    rf_rand = run_readout_condition(rf_token_scorer, z_rand, offsets, y, prompt_idx, core_by_fold, folds)
    print(f"  [PASS] token-RF readout ran on band+rand [{time.time()-t0:.0f}s]  "
          f"band band-only={rf_band['band_only']['pooled_oof_auroc']:.4f}  "
          f"rand band-only={rf_rand['band_only']['pooled_oof_auroc']:.4f}")
    assert rf_band["band_only"]["pooled_oof_auroc"] > 0.6, "token-RF readout found no planted signal"

    delta = paired_bootstrap_delta(quad_band["_oof"]["band"], quad_rand["_oof"]["band"], y, prompt_idx,
                                    n_boot=200, seed=SEED, within_prompt=False)
    assert delta["excludes_zero"], "paired band-vs-rand delta should exclude zero given a real planted signal"
    print(f"  [PASS] paired bootstrap delta (band-only, quad): {delta['mean_delta']:.4f} "
          f"CI={delta['ci95']}, excludes_zero={delta['excludes_zero']}")

    # -- Part B: contrast block, needs a genuine per-beam delta signal --
    # generate_synthetic_data already plants a per-beam "quality" component independent of
    # prompt-difficulty (the same one session01 used to validate E4b) -- delta_n should recover it.
    t0 = time.time()
    contrast = run_contrast_condition(core_by_fold, y, prompt_idx, folds, seed=SEED)
    print(f"  [PASS] contrast block ran [{time.time()-t0:.0f}s]  "
          f"block-only RF={contrast['block_only_RF']['pooled_oof_auroc']:.4f}  "
          f"fused RF={contrast['fused_RF']['pooled_oof_auroc']:.4f}")
    assert contrast["block_only_RF"]["pooled_oof_auroc"] > 0.55, \
        "contrast block did not recover the planted within-prompt delta signal"
    print("  [PASS] contrast block recovers planted within-prompt delta signal (> 0.55 AUROC)")

    trace_summary = summarize_trace_distribution(contrast["trace_Gp_by_prompt"])
    assert trace_summary["n_prompts"] == 200
    print(f"  [PASS] trace(G_p) distribution summarized over {trace_summary['n_prompts']} prompts "
          f"(median={trace_summary['percentiles']['p50']:.4f})")

    core_oof = np.full(len(y), np.nan)
    for fold_i, (tr, va) in enumerate(folds):
        core_oof[va] = s01.fit_eval("RF", core_by_fold[fold_i][tr], y[tr], core_by_fold[fold_i][va], SEED + fold_i)
    delta_vs_core = paired_bootstrap_delta(contrast["_oof"]["fused_RF"], core_oof, y, prompt_idx,
                                            n_boot=200, seed=SEED)
    print(f"  [PASS] paired delta (contrast-fused vs core-only): {delta_vs_core['mean_delta']:.4f} "
          f"CI={delta_vs_core['ci95']}, excludes_zero={delta_vs_core['excludes_zero']}")

    out_path = os.path.join(HERE, "results", "session03_selftest_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "core_only": {k: v for k, v in core_results.items()},
            "quad_band": {k: v for k, v in quad_band.items() if k != "_oof"},
            "quad_rand": {k: v for k, v in quad_rand.items() if k != "_oof"},
            "rf_band": {k: v for k, v in rf_band.items() if k != "_oof"},
            "rf_rand": {k: v for k, v in rf_rand.items() if k != "_oof"},
            "contrast": {k: v for k, v in contrast.items() if k != "_oof"},
            "trace_summary": trace_summary,
            "paired_delta_band_vs_rand_quad": delta,
            "paired_delta_contrast_fused_vs_core": delta_vs_core,
        }, f, indent=2)
    assert os.path.exists(out_path)
    print(f"  [PASS] JSON written to {out_path}")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="truthfulqa")
    parser.add_argument("--pooled-suffix", type=str, default="_maxenergy_seeded")
    parser.add_argument("--manifest", type=str, default="data/manifest_seeded_v1.json")
    parser.add_argument("--r_l", type=int, default=5)
    parser.add_argument("--r_d", type=int, default=64)
    parser.add_argument("--output-json", type=str, default="results/session03_metrics.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.model_folder:
        print("ERROR: --model_folder required."); sys.exit(1)

    manifest_path = os.path.join(HERE, args.manifest) if not os.path.isabs(args.manifest) else args.manifest
    print(f"Verifying manifest: {manifest_path}")
    manifest = pin_mod.verify_manifest(manifest_path)
    print(f"  [OK] manifest verified, hashes match. Extraction commit: "
          f"{manifest['extraction_git_commit']}")
    print(f"  Counts: {manifest['counts']}")

    packed, band_meta = s02.load_band_npz(manifest["band_meta_path"])
    import torch
    pooled = torch.load(manifest["pooled_pt_path"], weights_only=False)
    X = torch.stack(pooled["all_emb"]).float().numpy()
    y = np.array([int(f) for f in pooled["all_hallucination_flag"]], dtype=np.int64)
    prompt_idx = np.array(pooled["prompt_indices"], dtype=np.int64)

    if not np.array_equal(packed["label"], y):
        raise ValueError("Band npz labels != pooled labels beam-for-beam despite manifest match "
                          "-- refusing to proceed.")

    folds = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups=prompt_idx))
    for fold_i, (tr, va) in enumerate(folds):
        assert set(prompt_idx[tr].tolist()).isdisjoint(set(prompt_idx[va].tolist()))

    t0 = time.time()
    core_results = manifest["references"]["core_only_grouped"]
    _, core_by_fold = s02.run_core_only(X, y, prompt_idx, folds, args.r_l, args.r_d, seed=SEED)
    print(f"[core-only] (from manifest) RF pooled={core_results['RF']['pooled_oof_auroc']:.4f}  "
          f"within-prompt={core_results['RF']['within_prompt']['within_prompt_auroc']:.4f}  "
          f"[{time.time()-t0:.0f}s]")
    core_oof_rf = np.full(len(y), np.nan)
    for fold_i, (tr, va) in enumerate(folds):
        core_oof_rf[va] = s01.fit_eval("RF", core_by_fold[fold_i][tr], y[tr], core_by_fold[fold_i][va], SEED + fold_i)
    fresh_core_pooled = float(roc_auc_score(y, core_oof_rf))
    delta_vs_manifest = abs(fresh_core_pooled - core_results["RF"]["pooled_oof_auroc"])
    if delta_vs_manifest > 0.005:
        print(f"  [WARN] this run's freshly-refit core-only RF ({fresh_core_pooled:.4f}) differs from "
              f"the manifest-pinned reference ({core_results['RF']['pooled_oof_auroc']:.4f}) by "
              f"{delta_vs_manifest:.4f} (> 0.005) -- known sklearn/threading nondeterminism; the "
              f"paired deltas below use this run's own core_oof_rf, so pairing stays internally valid.")

    z_band_primary = s02.slice_band(packed["z_band"], *PRIMARY_BAND_SLICE)
    offsets = packed["offsets"]
    z_rand = packed["z_rand"]

    readouts = {"linear": linear_token_scorer, "quad": quad_token_scorer, "tokenRF": rf_token_scorer}
    part_a_results = {}
    for rname, scorer in readouts.items():
        t0 = time.time()
        res_band = run_readout_condition(scorer, z_band_primary, offsets, y, prompt_idx, core_by_fold, folds)
        res_rand = run_readout_condition(scorer, z_rand, offsets, y, prompt_idx, core_by_fold, folds)
        print(f"[{rname}] band band-only={res_band['band_only']['pooled_oof_auroc']:.4f}  "
              f"rand band-only={res_rand['band_only']['pooled_oof_auroc']:.4f}  [{time.time()-t0:.0f}s]")

        delta_pooled = paired_bootstrap_delta(res_band["_oof"]["band"], res_rand["_oof"]["band"], y, prompt_idx)
        delta_wp = paired_bootstrap_delta(res_band["_oof"]["band"], res_rand["_oof"]["band"], y, prompt_idx,
                                           within_prompt=True)
        delta_fused_vs_core = paired_bootstrap_delta(res_band["_oof"]["fused_RF"], core_oof_rf, y, prompt_idx)
        delta_fused_vs_core_wp = paired_bootstrap_delta(res_band["_oof"]["fused_RF"], core_oof_rf, y, prompt_idx,
                                                          within_prompt=True)
        part_a_results[rname] = {
            "band": {k: v for k, v in res_band.items() if k != "_oof"},
            "rand": {k: v for k, v in res_rand.items() if k != "_oof"},
            "paired_delta_band_vs_rand_pooled": delta_pooled,
            "paired_delta_band_vs_rand_within_prompt": delta_wp,
            "paired_fold_deltas_pooled": paired_fold_deltas(res_band["band_only"]["per_fold_auroc"],
                                                              res_rand["band_only"]["per_fold_auroc"]),
            "paired_delta_fused_vs_core_pooled": delta_fused_vs_core,
            "paired_delta_fused_vs_core_within_prompt": delta_fused_vs_core_wp,
        }

    t0 = time.time()
    contrast = run_contrast_condition(core_by_fold, y, prompt_idx, folds, seed=SEED)
    trace_summary = summarize_trace_distribution(contrast["trace_Gp_by_prompt"])
    delta_contrast_pooled = paired_bootstrap_delta(contrast["_oof"]["fused_RF"], core_oof_rf, y, prompt_idx)
    delta_contrast_wp = paired_bootstrap_delta(contrast["_oof"]["fused_RF"], core_oof_rf, y, prompt_idx,
                                                within_prompt=True)
    print(f"[contrast] block-only RF={contrast['block_only_RF']['pooled_oof_auroc']:.4f}  "
          f"fused RF={contrast['fused_RF']['pooled_oof_auroc']:.4f}  [{time.time()-t0:.0f}s]")

    harp_core = manifest["references"]["harp_protocol_core_only"]

    output = {
        "manifest_hash_summary": {"pooled_pt_sha256": manifest["pooled_pt_sha256"],
                                   "band_npz_sha256": manifest["band_npz_sha256"]},
        "part0_references": {"core_only_grouped": core_results, "harp_protocol_core_only": harp_core},
        "part_a": {k: v for k, v in part_a_results.items()},
        "part_b_contrast": {k: v for k, v in contrast.items() if k != "_oof"},
        "part_b_trace_distribution": trace_summary,
        "part_b_paired_delta_vs_core_pooled": delta_contrast_pooled,
        "part_b_paired_delta_vs_core_within_prompt": delta_contrast_wp,
        "config": {"seed": SEED, "n_splits": N_SPLITS, "r_l": args.r_l, "r_d": args.r_d,
                   "primary_band_slice": PRIMARY_BAND_SLICE},
    }
    out_path = args.output_json
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote: {out_path}")

    # Spec's Part A readout list is {quad, tokenRF} only; 'linear' is session02's original
    # readout, kept and reported for continuity but excluded from the primary table below
    # per the row list in the prompt -- see "Deviations" in the final summary.
    primary_readouts = ["quad", "tokenRF"]

    # "best-fused per basis": across primary readouts x {RF, LR}, the single best fused AUROC
    def best_fused_for(basis_key):
        best = None
        for rname in primary_readouts:
            for clf in ("fused_RF", "fused_LR"):
                d = part_a_results[rname][basis_key][clf]
                if best is None or d["pooled_oof_auroc"] > best[1]["pooled_oof_auroc"]:
                    best = (f"{rname}/{clf}", d)
        return best

    print(f"\n{'Row':32s} {'pooled':>8s} {'pooled CI':>16s} {'within-p':>9s} "
          f"{'wp CI':>16s} {'pairedDelta(wp)':>12s} {'Delta CI':>16s} {'excl0':>6s}")

    def row(name, d, delta=None):
        ci = d["ci95"]; wci = d["within_prompt"]["ci95"]
        if delta is not None:
            dci = delta["ci95"]
            dstr = f"{delta['mean_delta']:12.4f}"
            dcistr = f"[{dci[0]:.3f},{dci[1]:.3f}]"
            excl = "yes" if delta["excludes_zero"] else "no"
        else:
            dstr, dcistr, excl = f"{'--':>12s}", f"{'--':>16s}", "--"
        print(f"{name:32s} {d['pooled_oof_auroc']:8.4f} [{ci[0]:.3f},{ci[1]:.3f}]".ljust(58) +
              f" {d['within_prompt']['within_prompt_auroc']:9.4f} [{wci[0]:.3f},{wci[1]:.3f}]".ljust(28) +
              f" {dstr} {dcistr:>16s} {excl:>6s}")

    print("  (paired delta column: for *-only rows it is band-vs-rand; for fused rows it is fused-vs-core-only)")
    row("core-only (RF)", core_results["RF"])
    for rname in primary_readouts:
        d_bvr = part_a_results[rname]["paired_delta_band_vs_rand_within_prompt"]
        row(f"band band-only ({rname})", part_a_results[rname]["band"]["band_only"], d_bvr)
        row(f"rand band-only ({rname})", part_a_results[rname]["rand"]["band_only"], d_bvr)

    for basis_key, label in (("band", "band"), ("rand", "rand")):
        best_label, best_d = best_fused_for(basis_key)
        d_fvc = part_a_results[best_label.split("/")[0]]["paired_delta_fused_vs_core_within_prompt"]
        row(f"best-fused, {label} basis ({best_label})", best_d, d_fvc)

    row("contrast block-only (RF)", contrast["block_only_RF"])
    row("contrast fused (RF)", contrast["fused_RF"], delta_contrast_wp)

    print(f"\nHARP-protocol core-only (Part 0, new): RF={harp_core['auroc_canonical_orientation']:.4f}")
    s02_json_path = os.path.join(HERE, "results", "session02_metrics.json")
    if os.path.exists(s02_json_path):
        with open(s02_json_path) as f:
            s02_prev = json.load(f)
        prev_fused = s02_prev.get("harp_protocol_readout", {}).get("fused_RF_auroc")
        if prev_fused is not None:
            print(f"HARP-protocol best condition (session02, linear readout, fused-RF, for context "
                  f"only -- not recomputed with nonlinear readouts this session): {prev_fused:.4f}")
    else:
        print("HARP-protocol best-condition context number not found (results/session02_metrics.json "
              "missing) -- skipped.")

    print("\n[Decision-rule summary]")
    for rname in primary_readouts:
        d = part_a_results[rname]["paired_delta_band_vs_rand_within_prompt"]
        verdict = "REAL SIGNAL (band > rand)" if (d["excludes_zero"] and d["mean_delta"] > 0) else \
                  "band ~= rand or band < rand" if not d["excludes_zero"] or d["mean_delta"] <= 0 else "?"
        print(f"  {rname}: band-vs-rand within-prompt delta = {d['mean_delta']:.4f} "
              f"CI={d['ci95']} excludes_zero={d['excludes_zero']}  -> {verdict}")
    d = delta_contrast_wp
    verdict_b = "Proposal B becomes a permanent stream" if (d["excludes_zero"] and d["mean_delta"] > 0) \
        else "contrast-fused does not beat core-only"
    print(f"  contrast-fused vs core-only within-prompt delta = {d['mean_delta']:.4f} "
          f"CI={d['ci95']} excludes_zero={d['excludes_zero']}  -> {verdict_b}")
    print(f"HARP-protocol core-only: band_only=n/a  fused=n/a  "
          f"RF_auroc={harp_core['auroc_canonical_orientation']:.4f}")


if __name__ == "__main__":
    main()
