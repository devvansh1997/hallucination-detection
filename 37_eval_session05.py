"""
37_eval_session05.py -- Session 05 Parts B+C: Replacement-Lens Evaluation + Powered Finals
================================================================================================
CPU-only. Consumes whichever pooled tensor 36_extraction_forensics.py's Part A determines is
canonical (--core-pooled-pt) plus session04's derived velocity/kinematic/re-pooling streams.
Every condition uses the SAME scaling (robust winsorize+IQR, per session04's hygiene-ladder
finding that 'vanilla_mad' is not actually unscaled -- see A3) so the five conditions in B1 are
compared on equal footing, not mixing scaling conventions the way session04 did across its
separate B1/B4 rows.

Usage:
  python 37_eval_session05.py --self-test
  python 37_eval_session05.py --core-pooled-pt <path> --velocity-meta <path> --band-meta <path> \
      --is-known-source <path-to-a-file-with-all_is_known>
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s01 = _load("s01", "26_grouped_baseline.py")
s02 = _load("s02", "28_eval_band.py")
s03 = _load("s03", "31_eval_session03.py")
s04 = _load("s04", "33_eval_session04.py")

SEED = 0
N_SPLITS = 5
N_REPEATS = 5
N_BOOTSTRAP = 1000

residualize_fuse_eval = s03.residualize_fuse_eval
summarize_oof = s03.summarize_oof
paired_bootstrap_delta = s03.paired_bootstrap_delta
robust_scale_3d = s04.robust_scale_3d


# ==============================================================================
# FOLD-LOCAL TUCKER CORE (uniform scaling across every condition)
# ==============================================================================

def fold_pure_core(X_raw, tr_idx, all_idx_mask_full, r_l, r_d, seed):
    """X_raw: (N, L, D). Robust-scales on tr_idx, fold-pure Tucker fit, projects all beams."""
    X_scaled = robust_scale_3d(X_raw, tr_idx)
    U_L, U_D = s01.compute_ul_ud(X_scaled[tr_idx], r_l, r_d)
    return s01.project_core(X_scaled, U_L, U_D)


def run_pooled_condition(X_raw, y, prompt_idx, folds, r_l, r_d, seed=SEED, label="condition"):
    """Standalone RF/LR on a fold-pure Tucker core of X_raw -- the 'replacement lens' (no fusion
    with anything else)."""
    n_beams = X_raw.shape[0]
    oof_rf = np.full(n_beams, np.nan); oof_lr = np.full(n_beams, np.nan)
    fold_rf, fold_lr = [], []
    core_by_fold = []
    bar = tqdm(list(enumerate(folds)), desc=f"[{label}] folds", unit="fold", leave=False)
    for fold_i, (tr, va) in bar:
        core = fold_pure_core(X_raw, tr, None, r_l, r_d, seed + fold_i)
        core_by_fold.append(core)
        rf_scores = s01.fit_eval("RF", core[tr], y[tr], core[va], seed + fold_i)
        oof_rf[va] = rf_scores; fold_rf.append(float(roc_auc_score(y[va], rf_scores)))
        lr_scores = s01.fit_eval("LR", core[tr], y[tr], core[va], seed + fold_i)
        oof_lr[va] = lr_scores; fold_lr.append(float(roc_auc_score(y[va], lr_scores)))
        bar.set_postfix_str(f"fold {fold_i+1}/{len(folds)} RF={fold_rf[-1]:.3f}")
    bar.close()
    return ({"RF": summarize_oof(oof_rf, y, prompt_idx, fold_rf, seed),
              "LR": summarize_oof(oof_lr, y, prompt_idx, fold_lr, seed)},
            {"RF": oof_rf, "LR": oof_lr}, core_by_fold)


def run_joint_condition(S_concat, V_concat, y, prompt_idx, folds, r_l=8, r_d=64, seed=SEED):
    """Stack q-static (9 layer-slices) and q-velocity (8 layer-slices) along the layer mode ->
    [N, 17, 8192], ONE fold-local Tucker on the combined tensor."""
    joint = np.concatenate([S_concat, V_concat], axis=1)   # (N, 17, 8192)
    return run_pooled_condition(joint, y, prompt_idx, folds, r_l, r_d, seed, label="joint")


def run_concat_condition(S_concat, V_concat, y, prompt_idx, folds, seed=SEED,
                          r_l_static=5, r_d_static=64, r_l_vel=4, r_d_vel=64):
    """Fold-pure Tucker fit SEPARATELY for q-static and q-velocity, then concatenate the two
    resulting core vectors with NO residualization -- literal feature union, not a fusion."""
    n_beams = S_concat.shape[0]
    oof_rf = np.full(n_beams, np.nan); oof_lr = np.full(n_beams, np.nan)
    fold_rf, fold_lr = [], []
    bar = tqdm(list(enumerate(folds)), desc="[core-concat] folds", unit="fold", leave=False)
    for fold_i, (tr, va) in bar:
        static_core = fold_pure_core(S_concat, tr, None, r_l_static, r_d_static, seed + fold_i)
        vel_core = fold_pure_core(V_concat, tr, None, r_l_vel, r_d_vel, seed + fold_i)
        union = np.concatenate([static_core, vel_core], axis=1)

        rf_scores = s01.fit_eval("RF", union[tr], y[tr], union[va], seed + fold_i)
        oof_rf[va] = rf_scores; fold_rf.append(float(roc_auc_score(y[va], rf_scores)))
        lr_scores = s01.fit_eval("LR", union[tr], y[tr], union[va], seed + fold_i)
        oof_lr[va] = lr_scores; fold_lr.append(float(roc_auc_score(y[va], lr_scores)))
        bar.set_postfix_str(f"fold {fold_i+1}/{len(folds)} RF={fold_rf[-1]:.3f}")
    bar.close()
    return {"RF": summarize_oof(oof_rf, y, prompt_idx, fold_rf, seed),
            "LR": summarize_oof(oof_lr, y, prompt_idx, fold_lr, seed)}, {"RF": oof_rf, "LR": oof_lr}


def run_kinematic_standalone(kin, y, prompt_idx, folds, seed=SEED):
    """B3: report for the record, no fusion, no comparison."""
    n_beams = kin.shape[0]
    oof_rf = np.full(n_beams, np.nan); oof_lr = np.full(n_beams, np.nan)
    fold_rf, fold_lr = [], []
    for fold_i, (tr, va) in enumerate(folds):
        rf_scores = s01.fit_eval("RF", kin[tr], y[tr], kin[va], seed + fold_i)
        oof_rf[va] = rf_scores; fold_rf.append(float(roc_auc_score(y[va], rf_scores)))
        scaler = StandardScaler()
        lr = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed + fold_i)
        lr.fit(scaler.fit_transform(kin[tr]), y[tr])
        lr_scores = lr.predict_proba(scaler.transform(kin[va]))[:, 1]
        oof_lr[va] = lr_scores; fold_lr.append(float(roc_auc_score(y[va], lr_scores)))
    return {"RF": summarize_oof(oof_rf, y, prompt_idx, fold_rf, seed),
            "LR": summarize_oof(oof_lr, y, prompt_idx, fold_lr, seed)}


# ==============================================================================
# C1 -- REPEATED GROUPED CV (sklearn's GroupKFold has no native shuffle/seed; repeats are
# obtained by shuffling beam order before grouping, which changes fold composition while still
# respecting group/prompt integrity)
# ==============================================================================

def make_shuffled_grouped_folds(n_beams, prompt_idx, seed, n_splits=N_SPLITS):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_beams)
    shuffled_groups = prompt_idx[perm]
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for tr_s, va_s in gkf.split(np.zeros(n_beams), groups=shuffled_groups):
        splits.append((perm[tr_s], perm[va_s]))
    return splits


def repeated_cv_condition(condition_fn, y, prompt_idx, n_repeats=N_REPEATS, seed=SEED):
    """condition_fn(folds, repeat_seed) -> (summary_dict_with_RF/LR, oof_dict_with_RF/LR).
    Returns per-repeat results plus pooled-across-repeats arrays for the aggregated bootstrap."""
    n_beams = len(y)
    all_fold_aurocs_rf = []
    per_repeat_pooled_rf = []
    oof_by_repeat_rf = []   # list of (n_beams,) arrays, one per repeat
    for r in range(n_repeats):
        repeat_seed = seed + r
        folds = make_shuffled_grouped_folds(n_beams, prompt_idx, repeat_seed)
        summary, oof = condition_fn(folds, repeat_seed)
        all_fold_aurocs_rf.extend(summary["RF"]["per_fold_auroc"])
        per_repeat_pooled_rf.append(summary["RF"]["pooled_oof_auroc"])
        oof_by_repeat_rf.append(oof["RF"])
    return {
        "per_repeat_pooled_auroc_RF": per_repeat_pooled_rf,
        "mean_pooled_auroc_RF": float(np.mean(per_repeat_pooled_rf)),
        "std_pooled_auroc_RF": float(np.std(per_repeat_pooled_rf)),
        "all_25_fold_aurocs_RF": all_fold_aurocs_rf,
        "mean_across_25_folds_RF": float(np.mean(all_fold_aurocs_rf)),
        "std_across_25_folds_RF": float(np.std(all_fold_aurocs_rf)),
        "_oof_by_repeat_RF": oof_by_repeat_rf,
    }


def aggregated_paired_delta(oof_by_repeat_a, oof_by_repeat_b, y, prompt_idx, n_boot=N_BOOTSTRAP,
                             seed=SEED, within_prompt=False):
    """Pools all repeats' (beam, score) pairs together (each beam appears once per repeat, with
    that repeat's own OOF score), then does ONE cluster bootstrap over prompts, resampling
    across the pooled repeat data -- 'aggregated across repeats' per the spec."""
    n_repeats = len(oof_by_repeat_a)
    n_beams = len(y)
    # (n_beams, n_repeats) -> flatten in beam-major order: each beam's n_repeats scores together
    scores_a = np.stack(oof_by_repeat_a, axis=1).flatten()
    scores_b = np.stack(oof_by_repeat_b, axis=1).flatten()
    beam_idx_full = np.repeat(np.arange(n_beams), n_repeats)
    y_full = y[beam_idx_full]
    prompt_full = prompt_idx[beam_idx_full]

    return paired_bootstrap_delta(scores_a, scores_b, y_full, prompt_full, n_boot=n_boot,
                                   seed=seed, within_prompt=within_prompt)


# ==============================================================================
# C2 -- HARP-PROTOCOL ROWS
# ==============================================================================

def run_harp_single(X_raw, y, prompt_idx, is_known, r_l, r_d, seed=42):
    n_beams = X_raw.shape[0]
    t_idx, v_idx = s01.original_harp_split(is_known, prompt_idx, n_beams, seed=seed)
    core = fold_pure_core(X_raw, t_idx, None, r_l, r_d, seed)
    rf_scores = s01.fit_eval("RF", core[t_idx], y[t_idx], core[v_idx], seed)
    auroc = float(roc_auc_score(y[v_idx], rf_scores))
    return {"auroc": auroc, "n_train": int(len(t_idx)), "n_valid": int(len(v_idx))}


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: B1-B3 conditions + C1 repeated CV + C2 HARP rows (synthetic data)")
    print("=" * 70)
    data = s01.generate_synthetic_data(n_prompts=150, beams_per_prompt=10, L=9, D=64, seed=SEED)
    core_raw, y, prompt_idx, is_known = data["X"], data["y"], data["prompt_idx"], data["is_known"]
    n_beams = data["n_beams"]
    rng = np.random.default_rng(0)

    # planted signal: velocity carries it, static doesn't (tests the replacement-lens machinery)
    v_signal = rng.normal(0, 1, size=64); v_signal /= np.linalg.norm(v_signal)
    V95 = rng.normal(0, 0.3, size=(n_beams, 8, 64)); V05 = rng.normal(0, 0.3, size=(n_beams, 8, 64))
    for i in range(n_beams):
        if y[i] == 1:
            V95[i] += 1.5 * v_signal
    V_concat = np.concatenate([V95, V05], axis=2)
    S95 = rng.normal(0, 1, size=(n_beams, 9, 64)); S05 = rng.normal(0, 1, size=(n_beams, 9, 64))
    S_concat = np.concatenate([S95, S05], axis=2)
    kin = rng.normal(0, 1, size=(n_beams, 30))

    folds = list(GroupKFold(n_splits=N_SPLITS).split(core_raw, y, groups=prompt_idx))
    for fold_i, (tr, va) in enumerate(folds):
        assert set(prompt_idx[tr].tolist()).isdisjoint(set(prompt_idx[va].tolist()))
    print(f"  [PASS] {N_SPLITS} folds, prompt-disjoint")

    r_l, r_d = 5, 10
    core_summary, core_oof, _ = run_pooled_condition(core_raw, y, prompt_idx, folds, r_l, r_d, SEED, "core-max")
    vel_summary, vel_oof, _ = run_pooled_condition(V_concat, y, prompt_idx, folds, 4, 10, SEED, "q-velocity")
    static_summary, static_oof, _ = run_pooled_condition(S_concat, y, prompt_idx, folds, 5, 10, SEED, "q-static")
    print(f"  [PASS] core-max RF pooled={core_summary['RF']['pooled_oof_auroc']:.4f}")
    print(f"  [PASS] q-velocity RF pooled={vel_summary['RF']['pooled_oof_auroc']:.4f} "
          f"(planted signal should make this beat core-max)")
    assert vel_summary["RF"]["pooled_oof_auroc"] > core_summary["RF"]["pooled_oof_auroc"], \
        "planted-signal velocity should beat core-max"
    print(f"  [PASS] q-static RF pooled={static_summary['RF']['pooled_oof_auroc']:.4f}")

    joint_summary, joint_oof, _ = run_joint_condition(S_concat, V_concat, y, prompt_idx, folds,
                                                        r_l=8, r_d=10, seed=SEED)
    print(f"  [PASS] joint RF pooled={joint_summary['RF']['pooled_oof_auroc']:.4f}")

    concat_summary, concat_oof = run_concat_condition(S_concat, V_concat, y, prompt_idx, folds, SEED,
                                                        r_l_static=5, r_d_static=10, r_l_vel=4, r_d_vel=10)
    print(f"  [PASS] core-concat RF pooled={concat_summary['RF']['pooled_oof_auroc']:.4f}")

    kin_summary = run_kinematic_standalone(kin, y, prompt_idx, folds, SEED)
    print(f"  [PASS] kinematic-standalone RF pooled={kin_summary['RF']['pooled_oof_auroc']:.4f}")

    d_b2a = paired_bootstrap_delta(vel_oof["RF"], core_oof["RF"], y, prompt_idx, n_boot=100, seed=SEED,
                                    within_prompt=True)
    print(f"  [PASS] B2(a) q-velocity vs core-max within-prompt delta = {d_b2a['mean_delta']:.4f} "
          f"CI={d_b2a['ci95']} excludes_zero={d_b2a['excludes_zero']}")
    assert d_b2a["excludes_zero"] and d_b2a["mean_delta"] > 0, \
        "planted velocity signal should give a significant positive delta vs core-max"

    # -- C1: repeated CV --
    def vel_condition_fn(folds_r, seed_r):
        return run_pooled_condition(V_concat, y, prompt_idx, folds_r, 4, 10, seed_r, "vel-repeat")[:2]

    def core_condition_fn(folds_r, seed_r):
        return run_pooled_condition(core_raw, y, prompt_idx, folds_r, r_l, r_d, seed_r, "core-repeat")[:2]

    t0 = time.time()
    vel_repeated = repeated_cv_condition(vel_condition_fn, y, prompt_idx, n_repeats=3, seed=SEED)
    core_repeated = repeated_cv_condition(core_condition_fn, y, prompt_idx, n_repeats=3, seed=SEED)
    print(f"  [PASS] C1 repeated CV (3 repeats): vel mean={vel_repeated['mean_pooled_auroc_RF']:.4f} "
          f"+/- {vel_repeated['std_pooled_auroc_RF']:.4f}  [{time.time()-t0:.0f}s]")
    assert len(vel_repeated["all_25_fold_aurocs_RF"]) == 15   # 3 repeats x 5 folds

    agg_delta = aggregated_paired_delta(vel_repeated["_oof_by_repeat_RF"], core_repeated["_oof_by_repeat_RF"],
                                         y, prompt_idx, n_boot=100, seed=SEED, within_prompt=True)
    print(f"  [PASS] C1 aggregated paired delta (vel vs core, within-prompt): "
          f"{agg_delta['mean_delta']:.4f} CI={agg_delta['ci95']}")

    # -- C2: HARP protocol --
    harp_core = run_harp_single(core_raw, y, prompt_idx, is_known, r_l, r_d, seed=42)
    harp_vel = run_harp_single(V_concat, y, prompt_idx, is_known, 4, 10, seed=42)
    print(f"  [PASS] C2 HARP-protocol: core={harp_core['auroc']:.4f}  velocity={harp_vel['auroc']:.4f}")

    out_path = os.path.join(HERE, "results", "session05_selftest_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "core_max": {"RF": core_summary["RF"]}, "q_velocity": {"RF": vel_summary["RF"]},
            "q_static": {"RF": static_summary["RF"]}, "joint": {"RF": joint_summary["RF"]},
            "core_concat": {"RF": concat_summary["RF"]}, "kinematic_standalone": {"RF": kin_summary["RF"]},
            "b2a_delta": d_b2a,
            "c1_vel_repeated": {k: v for k, v in vel_repeated.items() if not k.startswith("_")},
            "c1_aggregated_delta": agg_delta,
            "c2_harp": {"core": harp_core, "velocity": harp_vel},
        }, f, indent=2)
    print(f"  [PASS] JSON written to {out_path}")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-pooled-pt", type=str, default=None,
                         help="Whichever pooled tensor 36_extraction_forensics.py's Part A "
                              "determined is canonical.")
    parser.add_argument("--velocity-meta", type=str, default=None)
    parser.add_argument("--r_l", type=int, default=5)
    parser.add_argument("--r_d", type=int, default=64)
    parser.add_argument("--output-json", type=str, default="results/session05_metrics.json")
    parser.add_argument("--leaderboard", type=str, default="results/leaderboard_v1.json")
    parser.add_argument("--dataset-version", type=str, default="v2-or-v3-per-part-A")
    parser.add_argument("--pipeline-version", type=str, default="TBD-per-part-A")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.core_pooled_pt or not args.velocity_meta:
        print("ERROR: --core-pooled-pt and --velocity-meta required."); sys.exit(1)

    import torch
    pooled = torch.load(args.core_pooled_pt, weights_only=False)
    core_raw = torch.stack(pooled["all_emb"]).float().numpy()
    y = np.array([int(f) for f in pooled["all_hallucination_flag"]], dtype=np.int64)
    prompt_idx = np.array(pooled["prompt_indices"], dtype=np.int64)
    is_known = np.array(pooled["all_is_known"], dtype=bool)

    vel_npz_path = os.path.splitext(args.velocity_meta)[0].replace("_meta", "") + ".npz"
    vel_data = dict(np.load(vel_npz_path))
    S_concat = np.concatenate([vel_data["S95"], vel_data["S05"]], axis=2)
    V_concat = np.concatenate([vel_data["V95"], vel_data["V05"]], axis=2)
    kin = vel_data["kinematic"]

    folds = list(GroupKFold(n_splits=N_SPLITS).split(core_raw, y, groups=prompt_idx))
    for fold_i, (tr, va) in enumerate(folds):
        assert set(prompt_idx[tr].tolist()).isdisjoint(set(prompt_idx[va].tolist()))

    print("\n[B1] Five replacement-lens conditions ...")
    core_summary, core_oof, _ = run_pooled_condition(core_raw, y, prompt_idx, folds, args.r_l, args.r_d,
                                                       SEED, "core-max")
    static_summary, static_oof, _ = run_pooled_condition(S_concat, y, prompt_idx, folds, 5, 64, SEED, "q-static")
    vel_summary, vel_oof, _ = run_pooled_condition(V_concat, y, prompt_idx, folds, 4, 64, SEED, "q-velocity")
    joint_summary, joint_oof, _ = run_joint_condition(S_concat, V_concat, y, prompt_idx, folds, 8, 64, SEED)
    concat_summary, concat_oof = run_concat_condition(S_concat, V_concat, y, prompt_idx, folds, SEED)
    kin_summary = run_kinematic_standalone(kin, y, prompt_idx, folds, SEED)

    conditions = {"core_max": (core_summary, core_oof), "q_static": (static_summary, static_oof),
                  "q_velocity": (vel_summary, vel_oof), "joint": (joint_summary, joint_oof),
                  "core_concat": (concat_summary, concat_oof)}
    for name, (summary, _) in conditions.items():
        print(f"  {name}: RF pooled={summary['RF']['pooled_oof_auroc']:.4f}  "
              f"within-prompt={summary['RF']['within_prompt']['within_prompt_auroc']:.4f}")

    print("\n[B2] Pre-registered primary comparisons ...")
    b2a = paired_bootstrap_delta(vel_oof["RF"], core_oof["RF"], y, prompt_idx, within_prompt=True)
    print(f"  (a) q-velocity vs core-max, within-prompt: {b2a['mean_delta']:.4f} CI={b2a['ci95']} "
          f"excludes_zero={b2a['excludes_zero']}")
    b2b_pooled = paired_bootstrap_delta(vel_oof["RF"], static_oof["RF"], y, prompt_idx)
    b2b_wp = paired_bootstrap_delta(vel_oof["RF"], static_oof["RF"], y, prompt_idx, within_prompt=True)
    print(f"  (b) q-velocity vs q-static: pooled {b2b_pooled['mean_delta']:.4f} "
          f"excludes_zero={b2b_pooled['excludes_zero']}, within-prompt {b2b_wp['mean_delta']:.4f} "
          f"excludes_zero={b2b_wp['excludes_zero']}")

    single_best_name = max(("core_max", "q_static", "q_velocity"),
                            key=lambda n: conditions[n][0]["RF"]["within_prompt"]["within_prompt_auroc"])
    combo_best_name = max(("joint", "core_concat"),
                           key=lambda n: conditions[n][0]["RF"]["within_prompt"]["within_prompt_auroc"])
    b2c = paired_bootstrap_delta(conditions[combo_best_name][1]["RF"], conditions[single_best_name][1]["RF"],
                                  y, prompt_idx, within_prompt=True)
    print(f"  (c) best combo ({combo_best_name}) vs best single ({single_best_name}), within-prompt: "
          f"{b2c['mean_delta']:.4f} CI={b2c['ci95']} excludes_zero={b2c['excludes_zero']}")

    print(f"\n[B3] Kinematic-30 standalone (for the record): RF pooled="
          f"{kin_summary['RF']['pooled_oof_auroc']:.4f}  within-prompt="
          f"{kin_summary['RF']['within_prompt']['within_prompt_auroc']:.4f}")

    print("\n[C1] Repeated grouped CV (5x) for the top-2 conditions ...")
    ranked = sorted(conditions.items(), key=lambda kv: kv[1][0]["RF"]["within_prompt"]["within_prompt_auroc"],
                     reverse=True)
    top2_names = [ranked[0][0], ranked[1][0]]
    print(f"  Top-2 by within-prompt AUROC: {top2_names}")

    condition_builders = {
        "core_max": lambda folds_r, seed_r: run_pooled_condition(core_raw, y, prompt_idx, folds_r,
                                                                    args.r_l, args.r_d, seed_r, "core-repeat")[:2],
        "q_static": lambda folds_r, seed_r: run_pooled_condition(S_concat, y, prompt_idx, folds_r, 5, 64,
                                                                    seed_r, "static-repeat")[:2],
        "q_velocity": lambda folds_r, seed_r: run_pooled_condition(V_concat, y, prompt_idx, folds_r, 4, 64,
                                                                      seed_r, "vel-repeat")[:2],
        "joint": lambda folds_r, seed_r: run_joint_condition(S_concat, V_concat, y, prompt_idx, folds_r,
                                                                8, 64, seed_r)[:2],
        "core_concat": lambda folds_r, seed_r: run_concat_condition(S_concat, V_concat, y, prompt_idx,
                                                                       folds_r, seed_r),
    }
    core_repeated = repeated_cv_condition(condition_builders["core_max"], y, prompt_idx, seed=SEED)
    repeated_results = {"core_max": core_repeated}
    for name in top2_names:
        if name == "core_max":
            continue
        t0 = time.time()
        repeated_results[name] = repeated_cv_condition(condition_builders[name], y, prompt_idx, seed=SEED)
        print(f"  {name}: mean_pooled_25folds={repeated_results[name]['mean_across_25_folds_RF']:.4f} "
              f"+/- {repeated_results[name]['std_across_25_folds_RF']:.4f}  [{time.time()-t0:.0f}s]")

    c1_deltas = {}
    for name in top2_names:
        if name == "core_max":
            continue
        d = aggregated_paired_delta(repeated_results[name]["_oof_by_repeat_RF"],
                                     repeated_results["core_max"]["_oof_by_repeat_RF"],
                                     y, prompt_idx, within_prompt=True)
        c1_deltas[name] = d
        print(f"  {name} vs core-max, aggregated within-prompt delta: {d['mean_delta']:.4f} "
              f"CI={d['ci95']} excludes_zero={d['excludes_zero']}")

    print("\n[C2] HARP-protocol rows (core-max + top-2) ...")
    harp_rows = {"core_max": run_harp_single(core_raw, y, prompt_idx, is_known, args.r_l, args.r_d)}
    harp_feature_map = {"q_static": (S_concat, 5, 64), "q_velocity": (V_concat, 4, 64)}
    for name in top2_names:
        if name in harp_feature_map:
            X_raw, rl, rd = harp_feature_map[name]
            harp_rows[name] = run_harp_single(X_raw, y, prompt_idx, is_known, rl, rd)
    for name, r in harp_rows.items():
        print(f"  {name}: HARP-protocol AUROC={r['auroc']:.4f}  (n_train={r['n_train']}, n_valid={r['n_valid']})")

    output = {
        "b1_conditions": {name: {k: v for k, v in summary.items()} for name, (summary, _) in conditions.items()},
        "b2a_velocity_vs_core": b2a, "b2b_velocity_vs_static_pooled": b2b_pooled,
        "b2b_velocity_vs_static_within_prompt": b2b_wp,
        "b2c_best_combo_vs_best_single": {"combo": combo_best_name, "single": single_best_name, "delta": b2c},
        "b3_kinematic_standalone": kin_summary,
        "c1_top2": top2_names,
        "c1_repeated": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                        for k, v in repeated_results.items()},
        "c1_aggregated_deltas": c1_deltas,
        "c2_harp_protocol": harp_rows,
        "config": {"seed": SEED, "n_splits": N_SPLITS, "n_repeats": N_REPEATS, "r_l": args.r_l, "r_d": args.r_d},
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote: {args.output_json}")

    print("\n[C3] Freezing leaderboard ...")
    leaderboard = {
        "dataset_version": args.dataset_version, "pipeline_version": args.pipeline_version,
        "entries": [
            {"name": "core-max (grouped CV)", "protocol": "GroupKFold(5)",
             "pooled_auroc": core_summary["RF"]["pooled_oof_auroc"],
             "within_prompt_auroc": core_summary["RF"]["within_prompt"]["within_prompt_auroc"],
             "ci95_pooled": core_summary["RF"]["ci95"]},
        ] + [
            {"name": f"{name} (grouped CV)", "protocol": "GroupKFold(5)",
             "pooled_auroc": conditions[name][0]["RF"]["pooled_oof_auroc"],
             "within_prompt_auroc": conditions[name][0]["RF"]["within_prompt"]["within_prompt_auroc"],
             "ci95_pooled": conditions[name][0]["RF"]["ci95"]}
            for name in conditions if name != "core_max"
        ] + [
            {"name": f"{name} (HARP-protocol)", "protocol": "HARP 75/25 known split",
             "auroc": r["auroc"]} for name, r in harp_rows.items()
        ],
    }
    os.makedirs(os.path.dirname(args.leaderboard) or ".", exist_ok=True)
    with open(args.leaderboard, "w") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"Wrote: {args.leaderboard}")

    print(f"\n{'Row':30s} {'pooled':>8s} {'within-p':>9s} {'pairedΔ(wp) vs core':>20s} {'excl0':>6s}")
    for name in ("core_max", "q_static", "q_velocity", "joint", "core_concat"):
        summary = conditions[name][0]
        d = c1_deltas.get(name)
        dstr = f"{d['mean_delta']:.4f}" if d else "--"
        excl = ("yes" if d["excludes_zero"] else "no") if d else "--"
        print(f"{name:30s} {summary['RF']['pooled_oof_auroc']:8.4f} "
              f"{summary['RF']['within_prompt']['within_prompt_auroc']:9.4f} {dstr:>20s} {excl:>6s}")
    print(f"{'kinematic_standalone (no cmp)':30s} {kin_summary['RF']['pooled_oof_auroc']:8.4f} "
          f"{kin_summary['RF']['within_prompt']['within_prompt_auroc']:9.4f} {'--':>20s} {'--':>6s}")


if __name__ == "__main__":
    main()
