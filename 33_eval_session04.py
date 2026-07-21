"""
33_eval_session04.py -- Session 04 Parts B+C: Velocity/Kinematic Streams, Hygiene Ladder,
Pooling Ablation, and the Session-03 Stacking Fix
================================================================================================
CPU-only. B1/B2/B4 consume 32_extract_velocity.py's output, which is BLOCKED this session (see
that file's docstring and this session's final handoff) -- their real-data paths hard-fail
cleanly with an explanation. B3 (hygiene ladder) and Part C (stacking fix) use only the ALREADY
pinned session02/03 artifacts and are fully runnable now.

Usage:
  python 33_eval_session04.py --self-test
  python 33_eval_session04.py --model_folder llama-3.1-8b-instruct --dataset truthfulqa
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
s03 = _load("s03", "31_eval_session03.py")
vel_mod = _load("s04_extract", "32_extract_velocity.py")

SEED = 0
N_SPLITS = 5
N_INNER_SPLITS = 3

# re-exported for readability
residualize_fuse_eval = s03.residualize_fuse_eval
summarize_oof = s03.summarize_oof
paired_bootstrap_delta = s03.paired_bootstrap_delta


# ==============================================================================
# B3 -- HYGIENE LADDER (runnable now: operates on the existing static pooled tensor)
# ==============================================================================

def robust_scale_3d(X, train_idx):
    """(N, L, D) -> same shape, winsorize+IQR robust-scaled per (layer, channel), train-fold fit.
    Treats each (layer, dim) pair as its own channel, matching mad_scale's convention."""
    N, L, D = X.shape
    X_flat = X.reshape(N, L * D)
    params = s02.fit_robust_scale(X_flat[train_idx])
    X_scaled = s02.apply_robust_scale(X_flat, params)
    return X_scaled.reshape(N, L, D)


def remove_common_pcs(X_scaled, train_idx, k, seed=SEED):
    """Removes the top-k principal components PER LAYER, fit on the train fold only."""
    N, L, D = X_scaled.shape
    X_clean = X_scaled.copy()
    for l in range(L):
        Xl_train = X_scaled[train_idx, l, :]
        pca = PCA(n_components=k, random_state=seed)
        pca.fit(Xl_train)
        V = pca.components_                        # (k, D)
        Xl_all = X_scaled[:, l, :]
        proj = (Xl_all @ V.T) @ V
        X_clean[:, l, :] = Xl_all - proj
    return X_clean


PREPROCESS_VARIANTS = {
    "vanilla_mad": lambda X, tr: s01.mad_scale(X, tr),
    "robust_scale": lambda X, tr: robust_scale_3d(X, tr),
    "robust_scale_pc1": lambda X, tr: remove_common_pcs(robust_scale_3d(X, tr), tr, k=1),
    "robust_scale_pc3": lambda X, tr: remove_common_pcs(robust_scale_3d(X, tr), tr, k=3),
}


def run_core_variant(X_raw, y, prompt_idx, folds, preprocess_fn, r_l, r_d, seed=SEED):
    n_beams = X_raw.shape[0]
    oof_rf = np.full(n_beams, np.nan); oof_lr = np.full(n_beams, np.nan)
    fold_rf, fold_lr = [], []
    core_by_fold = []
    for fold_i, (tr, va) in enumerate(folds):
        X_scaled = preprocess_fn(X_raw, tr)
        U_L, U_D = s01.compute_ul_ud(X_scaled[tr], r_l, r_d)
        core = s01.project_core(X_scaled, U_L, U_D)
        core_by_fold.append(core)

        rf_scores = s01.fit_eval("RF", core[tr], y[tr], core[va], seed + fold_i)
        oof_rf[va] = rf_scores; fold_rf.append(float(roc_auc_score(y[va], rf_scores)))
        lr_scores = s01.fit_eval("LR", core[tr], y[tr], core[va], seed + fold_i)
        oof_lr[va] = lr_scores; fold_lr.append(float(roc_auc_score(y[va], lr_scores)))

    return ({"RF": summarize_oof(oof_rf, y, prompt_idx, fold_rf, seed),
              "LR": summarize_oof(oof_lr, y, prompt_idx, fold_lr, seed)},
            {"RF": oof_rf, "LR": oof_lr}, core_by_fold)


def run_hygiene_ladder(X, y, prompt_idx, folds, r_l, r_d, seed=SEED):
    results, oofs = {}, {}
    for vname, fn in PREPROCESS_VARIANTS.items():
        summary, oof, _ = run_core_variant(X, y, prompt_idx, folds, fn, r_l, r_d, seed)
        results[vname] = summary
        oofs[vname] = oof

    vanilla_oof = oofs["vanilla_mad"]["RF"]
    deltas = {}
    for vname in PREPROCESS_VARIANTS:
        if vname == "vanilla_mad":
            continue
        d_pooled = paired_bootstrap_delta(oofs[vname]["RF"], vanilla_oof, y, prompt_idx, seed=seed)
        d_wp = paired_bootstrap_delta(oofs[vname]["RF"], vanilla_oof, y, prompt_idx, seed=seed,
                                       within_prompt=True)
        deltas[vname] = {"pooled": d_pooled, "within_prompt": d_wp}
    return results, deltas, oofs


# ==============================================================================
# B1 -- VELOCITY-CORE (blocked on 32_extract_velocity.py's real output)
# ==============================================================================

def run_velocity_condition(V_concat, y, prompt_idx, core_by_fold, folds, r_l=4, r_d=64, seed=SEED):
    """V_concat: (n_beams, 8, 8192) = concat(V95, V05) along the channel axis."""
    n_beams = V_concat.shape[0]
    oof_vonly_rf = np.full(n_beams, np.nan); oof_vonly_lr = np.full(n_beams, np.nan)
    oof_fused_rf = np.full(n_beams, np.nan); oof_fused_lr = np.full(n_beams, np.nan)
    fold_vrf, fold_vlr, fold_frf, fold_flr = [], [], [], []

    for fold_i, (tr, va) in enumerate(folds):
        Vs = s01.mad_scale(V_concat, tr)
        U_L, U_D = s01.compute_ul_ud(Vs[tr], r_l, r_d)
        vcore = s01.project_core(Vs, U_L, U_D)      # (n_beams, r_l*r_d) = 256-dim

        rf_v = s01.fit_eval("RF", vcore[tr], y[tr], vcore[va], seed + fold_i)
        oof_vonly_rf[va] = rf_v; fold_vrf.append(float(roc_auc_score(y[va], rf_v)))
        lr_v = s01.fit_eval("LR", vcore[tr], y[tr], vcore[va], seed + fold_i)
        oof_vonly_lr[va] = lr_v; fold_vlr.append(float(roc_auc_score(y[va], lr_v)))

        core_this = core_by_fold[fold_i]
        rf_f, lr_f = residualize_fuse_eval(core_this[tr], core_this[va], vcore[tr], vcore[va],
                                            y[tr], y[va], seed + fold_i)
        oof_fused_rf[va] = rf_f; fold_frf.append(float(roc_auc_score(y[va], rf_f)))
        oof_fused_lr[va] = lr_f; fold_flr.append(float(roc_auc_score(y[va], lr_f)))

    return {
        "velocity_only": {"RF": summarize_oof(oof_vonly_rf, y, prompt_idx, fold_vrf, seed),
                           "LR": summarize_oof(oof_vonly_lr, y, prompt_idx, fold_vlr, seed)},
        "fused": {"RF": summarize_oof(oof_fused_rf, y, prompt_idx, fold_frf, seed),
                  "LR": summarize_oof(oof_fused_lr, y, prompt_idx, fold_flr, seed)},
        "_oof": {"velocity_only_RF": oof_vonly_rf, "fused_RF": oof_fused_rf},
    }


# ==============================================================================
# B2 -- KINEMATIC SCALARS (blocked on 32_extract_velocity.py's real output)
# ==============================================================================

def run_kinematic_condition(kin, y, prompt_idx, core_by_fold, folds, seed=SEED):
    """kin: (n_beams, 30) raw kinematic scalar block."""
    n_beams = kin.shape[0]
    oof_konly_rf = np.full(n_beams, np.nan); oof_konly_lr = np.full(n_beams, np.nan)
    oof_fused_rf = np.full(n_beams, np.nan); oof_fused_lr = np.full(n_beams, np.nan)
    fold_krf, fold_klr, fold_frf, fold_flr = [], [], [], []

    for fold_i, (tr, va) in enumerate(folds):
        rf_k = s01.fit_eval("RF", kin[tr], y[tr], kin[va], seed + fold_i)
        oof_konly_rf[va] = rf_k; fold_krf.append(float(roc_auc_score(y[va], rf_k)))

        scaler = StandardScaler()
        lr = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed + fold_i)
        lr.fit(scaler.fit_transform(kin[tr]), y[tr])
        lr_k = lr.predict_proba(scaler.transform(kin[va]))[:, 1]
        oof_konly_lr[va] = lr_k; fold_klr.append(float(roc_auc_score(y[va], lr_k)))

        core_this = core_by_fold[fold_i]
        rf_f, lr_f = residualize_fuse_eval(core_this[tr], core_this[va], kin[tr], kin[va],
                                            y[tr], y[va], seed + fold_i)
        oof_fused_rf[va] = rf_f; fold_frf.append(float(roc_auc_score(y[va], rf_f)))
        oof_fused_lr[va] = lr_f; fold_flr.append(float(roc_auc_score(y[va], lr_f)))

    return {
        "kinematic_only": {"RF": summarize_oof(oof_konly_rf, y, prompt_idx, fold_krf, seed),
                            "LR": summarize_oof(oof_konly_lr, y, prompt_idx, fold_klr, seed)},
        "fused": {"RF": summarize_oof(oof_fused_rf, y, prompt_idx, fold_frf, seed),
                  "LR": summarize_oof(oof_fused_lr, y, prompt_idx, fold_flr, seed)},
        "_oof": {"kinematic_only_RF": oof_konly_rf, "fused_RF": oof_fused_rf},
    }


# ==============================================================================
# B4 -- STATIC RE-POOLING ABLATION (blocked on 32_extract_velocity.py's real output)
# ==============================================================================

def run_repooling_ablation(S_concat, y, prompt_idx, vanilla_core_oof_rf, folds, r_l=5, r_d=64, seed=SEED):
    """S_concat: (n_beams, 9, 8192) = concat(S95, S05). Standalone comparison against the
    vanilla positive-max core (paired, same folds)."""
    summary, oof, core_by_fold = run_core_variant(S_concat, y, prompt_idx, folds,
                                                    lambda X, tr: robust_scale_3d(X, tr), r_l, r_d, seed)
    d_pooled = paired_bootstrap_delta(oof["RF"], vanilla_core_oof_rf, y, prompt_idx, seed=seed)
    d_wp = paired_bootstrap_delta(oof["RF"], vanilla_core_oof_rf, y, prompt_idx, seed=seed,
                                   within_prompt=True)
    return summary, {"pooled": d_pooled, "within_prompt": d_wp}, core_by_fold


# ==============================================================================
# B5 -- COMBINATIONS (only components individually non-negative on within-prompt paired delta;
# no post-hoc subset search)
# ==============================================================================

def select_nonnegative_components(component_deltas):
    """component_deltas: {name: {"within_prompt": {"mean_delta": ...}}}. Returns names with
    mean_delta >= 0 (a looser bar than CI-excludes-zero -- 'individually non-negative')."""
    return [name for name, d in component_deltas.items() if d["within_prompt"]["mean_delta"] >= 0]


def run_combination(core_by_fold, extra_blocks_by_fold, y, prompt_idx, folds, seed=SEED):
    """extra_blocks_by_fold: list (len n_folds) of (n_beams, d_extra) arrays already selected
    by select_nonnegative_components -- concatenated as ONE combined residualized block, not
    searched over subsets."""
    n_beams = len(y)
    oof_rf = np.full(n_beams, np.nan); oof_lr = np.full(n_beams, np.nan)
    fold_rf, fold_lr = [], []
    for fold_i, (tr, va) in enumerate(folds):
        core_this = core_by_fold[fold_i]
        block = extra_blocks_by_fold[fold_i]
        rf_scores, lr_scores = residualize_fuse_eval(core_this[tr], core_this[va], block[tr], block[va],
                                                       y[tr], y[va], seed + fold_i)
        oof_rf[va] = rf_scores; fold_rf.append(float(roc_auc_score(y[va], rf_scores)))
        oof_lr[va] = lr_scores; fold_lr.append(float(roc_auc_score(y[va], lr_scores)))
    return {"RF": summarize_oof(oof_rf, y, prompt_idx, fold_rf, seed),
            "LR": summarize_oof(oof_lr, y, prompt_idx, fold_lr, seed)}


# ==============================================================================
# PART C -- SESSION-03 STACKING FIX (runnable now: uses the existing band npz)
# ==============================================================================

def run_readout_condition_stacking_fixed(scorer_fn, z_stream, offsets, y, prompt_idx, core_by_fold,
                                          folds, seed=SEED, n_inner_splits=N_INNER_SPLITS):
    """Identical to 31_eval_session03.run_readout_condition's fused step, EXCEPT the training-fold
    beam aggregates are built from INNER GroupKFold(n_inner_splits) out-of-fold token scores,
    not in-sample scores from the model that was fit on those exact tokens. Validation path
    (score outer-val tokens with the model fit on the full outer-train fold) is unchanged."""
    n_beams = len(offsets) - 1
    oof_fused_rf = np.full(n_beams, np.nan)
    oof_fused_lr = np.full(n_beams, np.nan)
    fold_fused_rf, fold_fused_lr = [], []

    for fold_i, (tr_beam, va_beam) in enumerate(folds):
        tr_tok, tr_off = s02.slice_tokens_for_beams(z_stream, offsets, tr_beam)
        va_tok, va_off = s02.slice_tokens_for_beams(z_stream, offsets, va_beam)
        scale_params = s02.fit_robust_scale(tr_tok) if tr_tok.shape[0] > 0 else None
        tr_tok_s = s02.apply_robust_scale(tr_tok, scale_params) if tr_tok.shape[0] > 0 else tr_tok
        va_tok_s = s02.apply_robust_scale(va_tok, scale_params) if va_tok.shape[0] > 0 else va_tok
        tr_tok_labels = np.repeat(y[tr_beam], np.diff(tr_off))

        # validation path -- unchanged from session03
        _, va_scores, _ = scorer_fn(tr_tok_s, tr_tok_labels, va_tok_s, seed + fold_i, tr_off)
        agg_val = np.stack([s02.aggregate_beam(va_scores[va_off[i]:va_off[i + 1]])
                             for i in range(len(va_beam))])

        # train path -- inner GroupKFold OOF scoring, fixing the stacking leak
        tr_prompt_idx = prompt_idx[tr_beam]
        inner_gkf = GroupKFold(n_splits=n_inner_splits)
        tr_scores_oof = np.full(tr_tok.shape[0], np.nan)
        for inner_i, (itr_local, iva_local) in enumerate(
                inner_gkf.split(tr_beam, y[tr_beam], groups=tr_prompt_idx)):
            itr_beam, iva_beam = tr_beam[itr_local], tr_beam[iva_local]
            i_tr_tok, i_tr_off = s02.slice_tokens_for_beams(z_stream, offsets, itr_beam)
            i_va_tok, i_va_off = s02.slice_tokens_for_beams(z_stream, offsets, iva_beam)
            # reuse the OUTER fold's train-fit scaling params -- only the aggregate SOURCE
            # changes here, not the scaling, to avoid adding a second leakage axis
            i_tr_tok_s = s02.apply_robust_scale(i_tr_tok, scale_params) if i_tr_tok.shape[0] > 0 else i_tr_tok
            i_va_tok_s = s02.apply_robust_scale(i_va_tok, scale_params) if i_va_tok.shape[0] > 0 else i_va_tok
            i_tr_labels = np.repeat(y[itr_beam], np.diff(i_tr_off))
            _, i_va_scores, _ = scorer_fn(i_tr_tok_s, i_tr_labels, i_va_tok_s,
                                           seed + fold_i * 10 + inner_i, i_tr_off)
            for local_j, beam_local_idx in enumerate(iva_local):
                s_, e_ = tr_off[beam_local_idx], tr_off[beam_local_idx + 1]
                s2, e2 = i_va_off[local_j], i_va_off[local_j + 1]
                tr_scores_oof[s_:e_] = i_va_scores[s2:e2]

        assert not np.isnan(tr_scores_oof).any(), \
            f"fold {fold_i}: some training tokens never received an inner-OOF score"
        agg_train = np.stack([s02.aggregate_beam(tr_scores_oof[tr_off[i]:tr_off[i + 1]])
                               for i in range(len(tr_beam))])

        core_this = core_by_fold[fold_i]
        rf_scores, lr_scores = residualize_fuse_eval(core_this[tr_beam], core_this[va_beam],
                                                       agg_train, agg_val, y[tr_beam], y[va_beam], seed + fold_i)
        oof_fused_rf[va_beam] = rf_scores; fold_fused_rf.append(float(roc_auc_score(y[va_beam], rf_scores)))
        oof_fused_lr[va_beam] = lr_scores; fold_fused_lr.append(float(roc_auc_score(y[va_beam], lr_scores)))

    return {
        "fused_RF": summarize_oof(oof_fused_rf, y, prompt_idx, fold_fused_rf, seed),
        "fused_LR": summarize_oof(oof_fused_lr, y, prompt_idx, fold_fused_lr, seed),
        "_oof": {"fused_RF": oof_fused_rf, "fused_LR": oof_fused_lr},
    }


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: B1-B5 + Part C end-to-end on synthetic data")
    print("=" * 70)
    data = s01.generate_synthetic_data(n_prompts=200, beams_per_prompt=10, L=9, D=64, seed=SEED)
    X, y, prompt_idx = data["X"], data["y"], data["prompt_idx"]
    n_beams = data["n_beams"]
    r_l, r_d = 5, 20

    folds = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups=prompt_idx))
    for fold_i, (tr, va) in enumerate(folds):
        assert set(prompt_idx[tr].tolist()).isdisjoint(set(prompt_idx[va].tolist()))
    print(f"  [PASS] {N_SPLITS} folds, all prompt-disjoint")

    core_results, oofs, core_by_fold = run_core_variant(X, y, prompt_idx, folds,
                                                          lambda Xr, tr: s01.mad_scale(Xr, tr), r_l, r_d, SEED)
    print(f"  [INFO] vanilla core-only RF pooled = {core_results['RF']['pooled_oof_auroc']:.4f}")

    # -- B3: hygiene ladder --
    t0 = time.time()
    hygiene_results, hygiene_deltas, _ = run_hygiene_ladder(X, y, prompt_idx, folds, r_l, r_d, SEED)
    for vname, d in hygiene_deltas.items():
        print(f"  [PASS] B3 {vname}: within-prompt paired delta = {d['within_prompt']['mean_delta']:.4f} "
              f"CI={d['within_prompt']['ci95']}")
    print(f"  B3 ran in {time.time()-t0:.0f}s")

    # -- B1: velocity-core (synthetic V95/V05 with a planted signal so the assertion is meaningful) --
    rng = np.random.default_rng(0)
    v_signal = rng.normal(0, 1, size=64); v_signal /= np.linalg.norm(v_signal)
    V95 = rng.normal(0, 0.3, size=(n_beams, 8, 64))
    V05 = rng.normal(0, 0.3, size=(n_beams, 8, 64))
    for i in range(n_beams):
        if y[i] == 1:
            V95[i] += 1.5 * v_signal
    V_concat = np.concatenate([V95, V05], axis=2)   # (n_beams, 8, 128) -- small D stand-in for 8192
    t0 = time.time()
    vel_result = run_velocity_condition(V_concat, y, prompt_idx, core_by_fold, folds, r_l=4, r_d=10, seed=SEED)
    print(f"  [PASS] B1 velocity-only RF = {vel_result['velocity_only']['RF']['pooled_oof_auroc']:.4f}  "
          f"fused RF = {vel_result['fused']['RF']['pooled_oof_auroc']:.4f}  [{time.time()-t0:.0f}s]")
    assert vel_result["velocity_only"]["RF"]["pooled_oof_auroc"] > 0.6, \
        "B1 velocity-only did not recover the planted signal"

    # -- B2: kinematic scalars (planted signal in a 30-dim block) --
    kin = rng.normal(0, 0.5, size=(n_beams, 30))
    kin_signal = rng.normal(0, 1, size=30); kin_signal /= np.linalg.norm(kin_signal)
    for i in range(n_beams):
        if y[i] == 1:
            kin[i] += 1.2 * kin_signal
    t0 = time.time()
    kin_result = run_kinematic_condition(kin, y, prompt_idx, core_by_fold, folds, seed=SEED)
    print(f"  [PASS] B2 kinematic-only RF = {kin_result['kinematic_only']['RF']['pooled_oof_auroc']:.4f}  "
          f"fused RF = {kin_result['fused']['RF']['pooled_oof_auroc']:.4f}  [{time.time()-t0:.0f}s]")
    assert kin_result["kinematic_only"]["RF"]["pooled_oof_auroc"] > 0.6, \
        "B2 kinematic-only did not recover the planted signal"

    # -- B4: static re-pooling ablation --
    S95 = rng.normal(0, 1, size=(n_beams, 9, 64)); S05 = rng.normal(0, 1, size=(n_beams, 9, 64))
    S_concat = np.concatenate([S95, S05], axis=2)
    t0 = time.time()
    repool_summary, repool_delta, _ = run_repooling_ablation(S_concat, y, prompt_idx, oofs["RF"],
                                                                folds, r_l=5, r_d=10, seed=SEED)
    print(f"  [PASS] B4 re-pooled-core RF = {repool_summary['RF']['pooled_oof_auroc']:.4f}  "
          f"paired delta vs vanilla core = {repool_delta['within_prompt']['mean_delta']:.4f}  "
          f"[{time.time()-t0:.0f}s]")

    # -- B5: combination of individually-non-negative components --
    component_deltas = {"velocity": {"within_prompt": {"mean_delta":
                         vel_result["fused"]["RF"]["within_prompt"]["within_prompt_auroc"]
                         - core_results["RF"]["within_prompt"]["within_prompt_auroc"]}},
                         "kinematic": {"within_prompt": {"mean_delta":
                         kin_result["fused"]["RF"]["within_prompt"]["within_prompt_auroc"]
                         - core_results["RF"]["within_prompt"]["within_prompt_auroc"]}}}
    selected = select_nonnegative_components(component_deltas)
    print(f"  [INFO] B5 components passing the non-negative filter: {selected}")
    if selected:
        combined_blocks_by_fold = []
        for fold_i in range(N_SPLITS):
            parts = []
            if "velocity" in selected:
                Vs = s01.mad_scale(V_concat, folds[fold_i][0])
                U_L, U_D = s01.compute_ul_ud(Vs[folds[fold_i][0]], 4, 10)
                parts.append(s01.project_core(Vs, U_L, U_D))
            if "kinematic" in selected:
                parts.append(kin)
            combined_blocks_by_fold.append(np.concatenate(parts, axis=1))
        combo_result = run_combination(core_by_fold, combined_blocks_by_fold, y, prompt_idx, folds, SEED)
        print(f"  [PASS] B5 combination RF pooled = {combo_result['RF']['pooled_oof_auroc']:.4f}")
    else:
        print("  [INFO] B5: no components passed the filter, nothing to combine (valid outcome)")

    # -- Part C: stacking fix --
    z_band, offsets = s02.generate_synthetic_tokens(y, prompt_idx, seed=SEED, dim=32, spike_idx=-1)
    t0 = time.time()
    fixed_linear = run_readout_condition_stacking_fixed(s03.linear_token_scorer, z_band, offsets,
                                                          y, prompt_idx, core_by_fold, folds, seed=SEED)
    print(f"  [PASS] Part C stacking-fixed linear fused-RF = "
          f"{fixed_linear['fused_RF']['pooled_oof_auroc']:.4f}  [{time.time()-t0:.0f}s]")
    delta_fixed = paired_bootstrap_delta(fixed_linear["_oof"]["fused_RF"], oofs["RF"], y, prompt_idx,
                                          n_boot=100, seed=SEED, within_prompt=True)
    print(f"  [PASS] Part C paired delta (fixed linear fused vs core-only, within-prompt): "
          f"{delta_fixed['mean_delta']:.4f} CI={delta_fixed['ci95']}")

    out_path = os.path.join(HERE, "results", "session04_selftest_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "core_only": core_results, "hygiene_ladder": hygiene_results,
            "hygiene_deltas": hygiene_deltas,
            "velocity": {k: v for k, v in vel_result.items() if k != "_oof"},
            "kinematic": {k: v for k, v in kin_result.items() if k != "_oof"},
            "repooling": repool_summary, "repooling_delta": repool_delta,
            "b5_selected": selected,
            "part_c_stacking_fixed_linear": {k: v for k, v in fixed_linear.items() if k != "_oof"},
            "part_c_delta_vs_core": delta_fixed,
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
    parser.add_argument("--velocity-meta", type=str, default=None,
                         help="Path to 32_extract_velocity.py's *_meta.json, if it exists. "
                              "B1/B2/B4 are skipped (not hard-failed) if omitted.")
    parser.add_argument("--r_l", type=int, default=5)
    parser.add_argument("--r_d", type=int, default=64)
    parser.add_argument("--output-json", type=str, default="results/session04_metrics.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.model_folder:
        print("ERROR: --model_folder required."); sys.exit(1)

    manifest_path = os.path.join(HERE, args.manifest) if not os.path.isabs(args.manifest) else args.manifest
    manifest = pin_mod.verify_manifest(manifest_path)
    print(f"Manifest verified. Counts: {manifest['counts']}")

    import torch
    pooled = torch.load(manifest["pooled_pt_path"], weights_only=False)
    X = torch.stack(pooled["all_emb"]).float().numpy()
    y = np.array([int(f) for f in pooled["all_hallucination_flag"]], dtype=np.int64)
    prompt_idx = np.array(pooled["prompt_indices"], dtype=np.int64)

    folds = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups=prompt_idx))
    for fold_i, (tr, va) in enumerate(folds):
        assert set(prompt_idx[tr].tolist()).isdisjoint(set(prompt_idx[va].tolist()))

    print("\n[B3] Hygiene ladder (robust scaling +/- common-PC removal) ...")
    t0 = time.time()
    hygiene_results, hygiene_deltas, hygiene_oofs = run_hygiene_ladder(X, y, prompt_idx, folds,
                                                                        args.r_l, args.r_d, SEED)
    for vname, d in hygiene_deltas.items():
        print(f"  {vname}: within-prompt paired delta = {d['within_prompt']['mean_delta']:.4f} "
              f"CI={d['within_prompt']['ci95']} excludes_zero={d['within_prompt']['excludes_zero']}")
    print(f"  [{time.time()-t0:.0f}s]")

    core_results = hygiene_results["vanilla_mad"]
    _, _, core_by_fold = run_core_variant(X, y, prompt_idx, folds,
                                           lambda Xr, tr: s01.mad_scale(Xr, tr), args.r_l, args.r_d, SEED)

    b1_result, b1_delta = None, None
    b2_result, b2_delta = None, None
    b4_result, b4_delta = None, None
    if args.velocity_meta and os.path.exists(args.velocity_meta):
        print("\n[B1/B2/B4] Loading velocity artifacts ...")
        with open(args.velocity_meta) as f:
            vmeta = json.load(f)
        vel_npz_path = os.path.splitext(args.velocity_meta)[0].replace("_meta", "") + ".npz"
        vel_data = dict(np.load(vel_npz_path))
        V_concat = np.concatenate([vel_data["V95"], vel_data["V05"]], axis=2)
        b1_result = run_velocity_condition(V_concat, y, prompt_idx, core_by_fold, folds, seed=SEED)
        b1_delta = paired_bootstrap_delta(b1_result["_oof"]["fused_RF"], hygiene_oofs["vanilla_mad"]["RF"],
                                           y, prompt_idx, within_prompt=True)

        b2_result = run_kinematic_condition(vel_data["kinematic"], y, prompt_idx, core_by_fold, folds, seed=SEED)
        b2_delta = paired_bootstrap_delta(b2_result["_oof"]["fused_RF"], hygiene_oofs["vanilla_mad"]["RF"],
                                           y, prompt_idx, within_prompt=True)

        S_concat = np.concatenate([vel_data["S95"], vel_data["S05"]], axis=2)
        b4_result, b4_delta, _ = run_repooling_ablation(S_concat, y, prompt_idx,
                                                          hygiene_oofs["vanilla_mad"]["RF"], folds, seed=SEED)
    else:
        print("\n[B1/B2/B4] SKIPPED: no --velocity-meta provided (32_extract_velocity.py blocked "
              "this session -- see Deviations in the handoff). Not hard-failing the whole run for "
              "this since B3/Part C don't depend on it.")

    print("\n[Part C] Stacking fix: re-running band linear-fused and band tokenRF-fused with "
          "inner GroupKFold(3) out-of-fold train aggregates ...")
    band_meta_path = os.path.join(os.path.dirname(manifest["pooled_pt_path"]), f"{args.dataset}_band_meta.json")
    packed, _ = s02.load_band_npz(band_meta_path)
    if not np.array_equal(packed["label"], y):
        raise ValueError("Band npz labels != pooled labels -- refusing to proceed.")
    z_band_primary = s02.slice_band(packed["z_band"], *s03.PRIMARY_BAND_SLICE)
    offsets = packed["offsets"]

    part_c = {}
    for rname, scorer in (("linear", s03.linear_token_scorer), ("tokenRF", s03.rf_token_scorer)):
        t0 = time.time()
        fixed = run_readout_condition_stacking_fixed(scorer, z_band_primary, offsets, y, prompt_idx,
                                                       core_by_fold, folds, seed=SEED)
        d_wp = paired_bootstrap_delta(fixed["_oof"]["fused_RF"], hygiene_oofs["vanilla_mad"]["RF"],
                                       y, prompt_idx, within_prompt=True)
        d_pooled = paired_bootstrap_delta(fixed["_oof"]["fused_RF"], hygiene_oofs["vanilla_mad"]["RF"],
                                           y, prompt_idx)
        part_c[rname] = {"fused_RF": fixed["fused_RF"], "fused_LR": fixed["fused_LR"],
                          "paired_delta_vs_core_within_prompt": d_wp,
                          "paired_delta_vs_core_pooled": d_pooled}
        print(f"  [{rname}] stacking-fixed fused-RF pooled={fixed['fused_RF']['pooled_oof_auroc']:.4f}  "
              f"within-prompt paired delta vs core-only={d_wp['mean_delta']:.4f} CI={d_wp['ci95']} "
              f"excludes_zero={d_wp['excludes_zero']}  [{time.time()-t0:.0f}s]")

    output = {
        "part0_reference": manifest["references"]["core_only_grouped"],
        "b3_hygiene_ladder": hygiene_results, "b3_paired_deltas": hygiene_deltas,
        "b1_velocity": None if b1_result is None else {k: v for k, v in b1_result.items() if k != "_oof"},
        "b1_paired_delta": b1_delta,
        "b2_kinematic": None if b2_result is None else {k: v for k, v in b2_result.items() if k != "_oof"},
        "b2_paired_delta": b2_delta,
        "b4_repooling": b4_result, "b4_paired_delta": b4_delta,
        "part_c_stacking_fix": part_c,
        "config": {"seed": SEED, "n_splits": N_SPLITS, "r_l": args.r_l, "r_d": args.r_d},
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote: {args.output_json}")

    print(f"\n{'Row':38s} {'pooled':>8s} {'pooled CI':>16s} {'within-p':>9s} {'wp CI':>16s} "
          f"{'pairedΔ(wp)':>12s} {'excl0':>6s}")

    def row(name, d, delta=None):
        ci = d["ci95"]; wci = d["within_prompt"]["ci95"]
        dstr = f"{delta['mean_delta']:.4f}" if delta else "--"
        excl = ("yes" if delta["excludes_zero"] else "no") if delta else "--"
        print(f"{name:38s} {d['pooled_oof_auroc']:8.4f} [{ci[0]:.3f},{ci[1]:.3f}]".ljust(64) +
              f" {d['within_prompt']['within_prompt_auroc']:9.4f} [{wci[0]:.3f},{wci[1]:.3f}]".ljust(28) +
              f" {dstr:>12s} {excl:>6s}")

    row("core-only vanilla (RF)", core_results["RF"])
    for vname in ("robust_scale", "robust_scale_pc1", "robust_scale_pc3"):
        row(f"B3 {vname} (RF)", hygiene_results[vname]["RF"], hygiene_deltas[vname]["within_prompt"])
    if b1_result: row("B1 velocity fused (RF)", b1_result["fused"]["RF"], b1_delta)
    if b2_result: row("B2 kinematic fused (RF)", b2_result["fused"]["RF"], b2_delta)
    if b4_result: row("B4 re-pooled core (RF)", b4_result["RF"], b4_delta)
    for rname in ("linear", "tokenRF"):
        row(f"C {rname} fused-RF (stacking-fixed)", part_c[rname]["fused_RF"],
            part_c[rname]["paired_delta_vs_core_within_prompt"])


if __name__ == "__main__":
    main()
