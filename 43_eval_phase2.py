"""
43_eval_phase2.py -- Session 06 Phase 2 Part B+C: Canonical Evaluation (CPU)
=================================================================================
Identical evaluation harness for all four datasets (TruthfulQA recomputed from its
existing canonical artifacts through this same code path -- not reusing old session04/05
numbers, which used a different exact-eigh Tucker fit; this makes all four rows genuinely
apples-to-apples). Two conditions only (core-max, q-velocity) -- q-static, combinations,
taxonomy, and transfer are all explicitly out of scope this phase.

Tucker fold-local fitting uses a randomized/truncated SVD for the channel-mode basis
instead of 26_grouped_baseline.py's exact-eigh Gram-trick. Required at TriviaQA's N=99,600
scale: compute_ul_ud()'s channel-mode step builds X.transpose(2,0,1).reshape(D,-1), which
forces a full memory COPY of a non-contiguous (D, N*L) array -- at D=8192 (q-velocity's
concatenated channel count) that's tens of GB. This script's compute_ul_ud_randomized()
instead merges axes 0,1 of the natural (N,L,D) layout into (N*L, D) -- a FREE reshape (no
copy, since those axes are already C-contiguous-adjacent) -- and runs
sklearn.utils.extmath.randomized_svd directly on that, never materializing a D x D Gram
matrix or the transposed copy at all. Applied uniformly to all four datasets (not just
TriviaQA) and to both conditions, for one consistent code path across the whole table.

Usage:
  python 43_eval_phase2.py --self-test
  python 43_eval_phase2.py --dataset triviaqa --model_folder llama-3.1-8b-instruct
  python 43_eval_phase2.py --dataset truthfulqa \
      --core-pooled-pt <v3 pooled.pt> --velocity-meta <velocity meta.json>
  python 43_eval_phase2.py --combine
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.extmath import randomized_svd

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s01 = _load("s01", "26_grouped_baseline.py")
s03 = _load("s03", "31_eval_session03.py")
s04 = _load("s04", "33_eval_session04.py")

robust_scale_3d = s04.robust_scale_3d
summarize_oof = s03.summarize_oof
paired_bootstrap_delta = s03.paired_bootstrap_delta

SEED = 0
N_SPLITS = 5
N_BOOTSTRAP = 1000
HARP_SEEDS = [42, 0, 1, 2, 3]
CONDITION_DIMS = {"core_max": (5, 64), "q_velocity": (4, 64)}
DATASETS = ["truthfulqa", "triviaqa", "nq_open", "tydiqa_gp"]

# baseline_report.tex Results table, "HARP" row, LLaMA-3.1-8B block (NQ-Open/TruthfulQA/
# TriviaQA/TyDiQA columns) -- the published reference this phase's master table compares
# against. Quoted, not re-derived.
HARP_PUBLISHED_LLAMA31_8B = {"nq_open": 89.4, "truthfulqa": 88.5, "triviaqa": 92.9, "tydiqa_gp": 86.6}


# ==============================================================================
# TUCKER CORE -- randomized/truncated SVD (channel-mode), required at TriviaQA scale
# ==============================================================================

def compute_ul_ud_randomized(X_train, r_l, r_d, seed=SEED):
    N, L, D = X_train.shape
    X_f = X_train.transpose(1, 0, 2).reshape(L, -1).astype(np.float64)   # L is tiny (8 or 9);
    A_L = X_f @ X_f.T                                                    # exact eigh stays cheap
    _, U_L = np.linalg.eigh(A_L)                                         # regardless of N.
    U_L = np.flip(U_L[:, -r_l:], axis=1).copy()

    X_merged = X_train.reshape(N * L, D).astype(np.float32)   # merge axes 0,1 -- FREE reshape,
    _, _, Vt = randomized_svd(X_merged, n_components=r_d, random_state=seed)   # no transpose-copy
    U_D = Vt.T
    return U_L.astype(np.float32), U_D.astype(np.float32)


def fold_pure_core_randomized(X_raw, tr_idx, r_l, r_d, seed):
    X_scaled = robust_scale_3d(X_raw, tr_idx)
    U_L, U_D = compute_ul_ud_randomized(X_scaled[tr_idx], r_l, r_d, seed)
    return s01.project_core(X_scaled, U_L, U_D)


# ==============================================================================
# SMALL PURE HELPERS
# ==============================================================================

def derive_is_known(y, prompt_idx):
    """Per-prompt boolean array: True if the prompt has >=1 truthful (y==0) beam."""
    unique_prompts = sorted(set(prompt_idx.tolist()))
    is_known = np.zeros(len(unique_prompts), dtype=bool)
    for i, p in enumerate(unique_prompts):
        is_known[i] = (y[prompt_idx == p] == 0).any()
    return is_known, np.asarray(unique_prompts)


def composition_line(dataset_name, y, prompt_idx):
    n_beams, n_prompts = len(y), len(np.unique(prompt_idx))
    halluc_rate = float(y.mean() * 100)
    n_mixed = n_all_t = n_all_h = 0
    for p in np.unique(prompt_idx):
        yp = y[prompt_idx == p]
        if (yp == 1).any() and (yp == 0).any():
            n_mixed += 1
        elif (yp == 1).all():
            n_all_h += 1
        else:
            n_all_t += 1
    print(f"[{dataset_name}] prompts={n_prompts}  beams={n_beams}  "
          f"hallucination_rate={halluc_rate:.1f}%  mixed_prompts={n_mixed}  "
          f"all_truthful={n_all_t}  all_hallucinated={n_all_h}")
    return {"n_prompts": n_prompts, "n_beams": n_beams, "hallucination_rate_pct": halluc_rate,
            "n_mixed_prompts": n_mixed, "n_all_truthful_prompts": n_all_t,
            "n_all_hallucinated_prompts": n_all_h}


# ==============================================================================
# PART B(a) -- GROUPED 5-FOLD CV
# ==============================================================================

def run_grouped_condition(X_raw, y, prompt_idx, folds, r_l, r_d, seed=SEED, label="condition"):
    n_beams = X_raw.shape[0]
    oof_rf = np.full(n_beams, np.nan); oof_lr = np.full(n_beams, np.nan)
    fold_rf, fold_lr = [], []
    for fold_i, (tr, va) in enumerate(folds):
        t0 = time.time()
        core = fold_pure_core_randomized(X_raw, tr, r_l, r_d, seed + fold_i)
        rf_scores = s01.fit_eval("RF", core[tr], y[tr], core[va], seed + fold_i)
        oof_rf[va] = rf_scores; fold_rf.append(float(roc_auc_score(y[va], rf_scores)))
        lr_scores = s01.fit_eval("LR", core[tr], y[tr], core[va], seed + fold_i)
        oof_lr[va] = lr_scores; fold_lr.append(float(roc_auc_score(y[va], lr_scores)))
        elapsed = time.time() - t0
        if fold_i == 0:
            print(f"  [{label}] fold 0: {elapsed:.1f}s -- extrapolated total for {len(folds)} "
                  f"folds: ~{elapsed*len(folds):.0f}s")
    return ({"RF": summarize_oof(oof_rf, y, prompt_idx, fold_rf, seed),
             "LR": summarize_oof(oof_lr, y, prompt_idx, fold_lr, seed)},
            {"RF": oof_rf, "LR": oof_lr})


# ==============================================================================
# PART B(b) -- HARP PROTOCOL, 5 KNOWN-PARTITION SEEDS, PAIRED CONDITIONS
# ==============================================================================

def run_harp_multiseed(conditions, y, prompt_idx, is_known, seeds=HARP_SEEDS, label_prefix=""):
    """conditions: dict name -> (X_raw, r_l, r_d). Both conditions evaluated on the IDENTICAL
    split within each seed (paired), per spec."""
    n_beams = len(y)
    per_seed = {name: [] for name in conditions}
    for seed in seeds:
        t_idx, v_idx = s01.original_harp_split(is_known, prompt_idx, n_beams, seed=seed)
        assert set(prompt_idx[t_idx].tolist()).isdisjoint(set(prompt_idx[v_idx].tolist())), \
            f"HARP split seed={seed}: train/valid prompt sets not disjoint"
        print(f"  [{label_prefix}] HARP seed={seed}: n_train={len(t_idx)}  n_valid={len(v_idx)}")
        for name, (X_raw, r_l, r_d) in conditions.items():
            core = fold_pure_core_randomized(X_raw, t_idx, r_l, r_d, seed)
            row = {"seed": seed, "n_train": int(len(t_idx)), "n_valid": int(len(v_idx))}
            for clf in ("RF", "LR"):
                scores = s01.fit_eval(clf, core[t_idx], y[t_idx], core[v_idx], seed)
                row[clf] = float(roc_auc_score(y[v_idx], scores))
            per_seed[name].append(row)

    summary = {}
    for name, rows in per_seed.items():
        summary[name] = {
            "per_seed": rows,
            "RF_mean": float(np.mean([r["RF"] for r in rows])),
            "RF_std": float(np.std([r["RF"] for r in rows])),
            "LR_mean": float(np.mean([r["LR"] for r in rows])),
            "LR_std": float(np.std([r["LR"] for r in rows])),
        }
    return summary


# ==============================================================================
# DATASET LOADING
# ==============================================================================

def load_new_dataset_features(dataset, data_dir, model_folder):
    """triviaqa/nq_open/tydiqa_gp: Part A's {dataset}_phase2_features.npz + manifest v2."""
    out_dir = os.path.join(data_dir, model_folder)
    npz_path = os.path.join(out_dir, f"{dataset}_phase2_features.npz")
    manifest_path = os.path.join(out_dir, f"manifest_{dataset}_v2.json")
    d = dict(np.load(npz_path))
    core_raw = d["static_max"].astype(np.float32)
    V_concat = np.concatenate([d["velocity_q95"], d["velocity_q05"]], axis=2).astype(np.float32)
    y = d["label"].astype(np.int64)
    prompt_idx = d["prompt_id"].astype(np.int64)
    is_known, unique_prompts = derive_is_known(y, prompt_idx)
    with open(manifest_path) as f:
        manifest = json.load(f)
    return core_raw, V_concat, y, prompt_idx, is_known, manifest.get("decoding_config", {})


def load_truthfulqa_features(core_pooled_pt, velocity_meta):
    import torch
    pooled = torch.load(core_pooled_pt, weights_only=False)
    core_raw = torch.stack(pooled["all_emb"]).float().numpy()
    y = np.array([int(f) for f in pooled["all_hallucination_flag"]], dtype=np.int64)
    prompt_idx = np.array(pooled["prompt_indices"], dtype=np.int64)

    vel_npz_path = os.path.splitext(velocity_meta)[0].replace("_meta", "") + ".npz"
    vel_data = dict(np.load(vel_npz_path))
    V_concat = np.concatenate([vel_data["V95"], vel_data["V05"]], axis=2).astype(np.float32)

    is_known, unique_prompts = derive_is_known(y, prompt_idx)
    with open(velocity_meta) as f:
        vel_meta = json.load(f)
    return core_raw, V_concat, y, prompt_idx, is_known, vel_meta.get("decoding_config", {})


# ==============================================================================
# PER-DATASET ORCHESTRATION
# ==============================================================================

def run_dataset_eval(dataset_name, core_raw, V_concat, y, prompt_idx, is_known, decoding_config):
    comp = composition_line(dataset_name, y, prompt_idx)
    n_beams = len(y)

    folds = list(GroupKFold(n_splits=N_SPLITS).split(core_raw, y, groups=prompt_idx))
    for tr, va in folds:
        assert set(prompt_idx[tr].tolist()).isdisjoint(set(prompt_idx[va].tolist()))

    print(f"\n[{dataset_name}] Part B(a): grouped 5-fold CV ...")
    r_l_c, r_d_c = CONDITION_DIMS["core_max"]
    core_summary, core_oof = run_grouped_condition(core_raw, y, prompt_idx, folds, r_l_c, r_d_c,
                                                     SEED, f"{dataset_name}/core-max")
    r_l_v, r_d_v = CONDITION_DIMS["q_velocity"]
    vel_summary, vel_oof = run_grouped_condition(V_concat, y, prompt_idx, folds, r_l_v, r_d_v,
                                                   SEED, f"{dataset_name}/q-velocity")

    delta_pooled = paired_bootstrap_delta(vel_oof["RF"], core_oof["RF"], y, prompt_idx,
                                           n_boot=N_BOOTSTRAP, seed=SEED, within_prompt=False)
    delta_wp = paired_bootstrap_delta(vel_oof["RF"], core_oof["RF"], y, prompt_idx,
                                       n_boot=N_BOOTSTRAP, seed=SEED, within_prompt=True)
    print(f"  [{dataset_name}] paired delta (q-velocity - core-max) RF: pooled="
          f"{delta_pooled['mean_delta']:.4f} excl0={delta_pooled['excludes_zero']}  "
          f"within-prompt={delta_wp['mean_delta']:.4f} excl0={delta_wp['excludes_zero']}")

    n_known = int(is_known.sum())
    print(f"\n[{dataset_name}] Part B(b): HARP protocol, {len(HARP_SEEDS)} seeds "
          f"(n_known={n_known}/{len(is_known)} prompts) ...")
    harp = run_harp_multiseed(
        {"core_max": (core_raw, r_l_c, r_d_c), "q_velocity": (V_concat, r_l_v, r_d_v)},
        y, prompt_idx, is_known, seeds=HARP_SEEDS, label_prefix=dataset_name)

    return {
        "dataset": dataset_name, "composition": comp, "decoding_config": decoding_config,
        "grouped": {"core_max": {k: v for k, v in core_summary.items()},
                    "q_velocity": {k: v for k, v in vel_summary.items()}},
        "grouped_paired_delta_RF": {"pooled": delta_pooled, "within_prompt": delta_wp},
        "harp": harp, "n_known": n_known, "n_beams": n_beams,
    }


# ==============================================================================
# PART C -- MASTER TABLE, CONFIG COMPARABILITY, LEADERBOARD
# ==============================================================================

def print_master_table(all_results):
    print(f"\n{'='*100}")
    print("MASTER TABLE")
    print(f"{'='*100}")
    header = (f"{'Dataset':12s} {'Condition':10s} {'Grouped pooled':>16s} {'Grouped w/in-p':>16s} "
              f"{'HARP mean+-std':>16s} {'HARP-published':>15s}")
    print(header)
    for ds in DATASETS:
        r = all_results.get(ds)
        if r is None:
            print(f"{ds:12s}  (missing -- not yet run)")
            continue
        published = HARP_PUBLISHED_LLAMA31_8B.get(ds)
        for cond_key, cond_label in (("core_max", "core-max"), ("q_velocity", "q-velocity")):
            g = r["grouped"][cond_key]["RF"]
            pooled_str = f"{g['pooled_oof_auroc']:.4f} [{g['ci95'][0]:.3f},{g['ci95'][1]:.3f}]"
            wp = g["within_prompt"]["within_prompt_auroc"]
            wp_ci = g["within_prompt"]["ci95"]
            wp_str = f"{wp:.4f} [{wp_ci[0]:.3f},{wp_ci[1]:.3f}]"
            h = r["harp"][cond_key]
            harp_str = f"{h['RF_mean']:.4f} +- {h['RF_std']:.4f}"
            pub_str = f"{published:.1f}" if (cond_key == "core_max" and published is not None) else ""
            print(f"{ds:12s} {cond_label:10s} {pooled_str:>16s} {wp_str:>16s} {harp_str:>16s} {pub_str:>15s}")
        d = r["grouped_paired_delta_RF"]
        print(f"  -> paired delta (q-velocity - core-max), RF: pooled={d['pooled']['mean_delta']:.4f} "
              f"excl0={d['pooled']['excludes_zero']}  within-prompt={d['within_prompt']['mean_delta']:.4f} "
              f"excl0={d['within_prompt']['excludes_zero']}")


def build_config_comparability_record(all_results):
    if "truthfulqa" not in all_results:
        return {"note": "truthfulqa not yet run -- comparability record incomplete"}
    ref = all_results["truthfulqa"]["decoding_config"]
    record = {"truthfulqa_v3_decoding_config": ref, "comparisons": {}}
    for ds in ("triviaqa", "nq_open", "tydiqa_gp"):
        if ds not in all_results:
            continue
        cfg = all_results[ds]["decoding_config"]
        diffs = {}
        all_keys = set(ref.keys()) | set(cfg.keys())
        for k in sorted(all_keys):
            if ref.get(k) != cfg.get(k):
                diffs[k] = {"truthfulqa_v3": ref.get(k), ds: cfg.get(k)}
        record["comparisons"][ds] = {"config": cfg, "differing_fields": diffs}
    return record


def update_leaderboard(all_results, leaderboard_path):
    entries = []
    for ds, r in all_results.items():
        for cond_key, cond_label in (("core_max", "core-max"), ("q_velocity", "q-velocity")):
            g = r["grouped"][cond_key]["RF"]
            h = r["harp"][cond_key]
            entries.append({
                "dataset": ds, "condition": cond_label, "pipeline_version": "session06-phase2",
                "window": "canonical (Phase 1 Assert B)", "protocol": "GroupKFold(5)",
                "pooled_auroc": g["pooled_oof_auroc"], "pooled_ci95": g["ci95"],
                "within_prompt_auroc": g["within_prompt"]["within_prompt_auroc"],
                "within_prompt_ci95": g["within_prompt"]["ci95"],
            })
            entries.append({
                "dataset": ds, "condition": cond_label, "pipeline_version": "session06-phase2",
                "window": "canonical (Phase 1 Assert B)", "protocol": "HARP 5-seed",
                "harp_rf_mean": h["RF_mean"], "harp_rf_std": h["RF_std"],
                "harp_lr_mean": h["LR_mean"], "harp_lr_std": h["LR_std"],
            })
    leaderboard = {"entries": entries}
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path) as f:
            existing = json.load(f)
        existing.setdefault("entries", [])
        existing["entries"] = [e for e in existing["entries"]
                                if e.get("pipeline_version") != "session06-phase2"] + entries
        leaderboard = existing
    os.makedirs(os.path.dirname(leaderboard_path) or ".", exist_ok=True)
    with open(leaderboard_path, "w") as f:
        json.dump(leaderboard, f, indent=2)
    return leaderboard_path


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: randomized-SVD Tucker, grouped/HARP eval, table/comparability (synthetic)")
    print("=" * 70)

    data = s01.generate_synthetic_data(n_prompts=150, beams_per_prompt=10, L=9, D=64, seed=0)
    core_raw, y, prompt_idx, is_known = data["X"], data["y"], data["prompt_idx"], data["is_known"]
    n_beams = data["n_beams"]

    is_known2, unique_prompts = derive_is_known(y, prompt_idx)
    assert np.array_equal(is_known2, is_known), "derive_is_known must match generate_synthetic_data's own is_known"
    print("  [PASS] derive_is_known matches the generator's ground-truth is_known array")

    comp = composition_line("selftest", y, prompt_idx)
    assert comp["n_beams"] == n_beams and comp["n_prompts"] == 150
    print(f"  [PASS] composition_line: {comp}")

    # -- randomized-SVD Tucker recovers a planted signal comparably to exact eigh --
    tr_idx = np.arange(n_beams)
    core_exact = (lambda: (
        lambda Xs: s01.project_core(Xs, *s01.compute_ul_ud(Xs[tr_idx], 5, 10))
    )(s01.mad_scale(core_raw, tr_idx)))()
    core_rand = fold_pure_core_randomized(core_raw, tr_idx, 5, 10, seed=0)
    auroc_exact = roc_auc_score(y, s01.fit_eval("RF", core_exact[tr_idx], y[tr_idx], core_exact, 0))
    auroc_rand = roc_auc_score(y, s01.fit_eval("RF", core_rand[tr_idx], y[tr_idx], core_rand, 0))
    assert auroc_exact > 0.65 and auroc_rand > 0.65, \
        f"both Tucker variants should recover the planted signal: exact={auroc_exact:.3f} rand={auroc_rand:.3f}"
    assert abs(auroc_exact - auroc_rand) < 0.1, \
        f"randomized-SVD Tucker should give comparable AUROC to exact eigh: {auroc_exact:.3f} vs {auroc_rand:.3f}"
    print(f"  [PASS] compute_ul_ud_randomized recovers planted signal comparably to exact eigh "
          f"(exact={auroc_exact:.4f}, randomized={auroc_rand:.4f})")

    folds = list(GroupKFold(n_splits=N_SPLITS).split(core_raw, y, groups=prompt_idx))
    core_summary, core_oof = run_grouped_condition(core_raw, y, prompt_idx, folds, 5, 10, SEED, "core-max")
    assert 0.0 <= core_summary["RF"]["pooled_oof_auroc"] <= 1.0
    assert core_summary["RF"]["pooled_oof_auroc"] > 0.6
    print(f"  [PASS] run_grouped_condition: RF pooled={core_summary['RF']['pooled_oof_auroc']:.4f}")

    harp = run_harp_multiseed({"core_max": (core_raw, 5, 10)}, y, prompt_idx, is_known,
                               seeds=[42, 0], label_prefix="selftest")
    assert len(harp["core_max"]["per_seed"]) == 2
    assert harp["core_max"]["RF_mean"] > 0.5
    print(f"  [PASS] run_harp_multiseed: 2 seeds, RF_mean={harp['core_max']['RF_mean']:.4f} "
          f"+/- {harp['core_max']['RF_std']:.4f}")

    # -- paired-split guarantee: both conditions must see the IDENTICAL split within a seed --
    harp2 = run_harp_multiseed({"a": (core_raw, 5, 10), "b": (core_raw, 5, 10)}, y, prompt_idx,
                                is_known, seeds=[42], label_prefix="selftest")
    assert harp2["a"]["per_seed"][0]["n_train"] == harp2["b"]["per_seed"][0]["n_train"]
    assert harp2["a"]["per_seed"][0]["n_valid"] == harp2["b"]["per_seed"][0]["n_valid"]
    print("  [PASS] run_harp_multiseed: paired conditions see identical n_train/n_valid within a seed")

    fake_results = {
        "truthfulqa": {"decoding_config": {"do_sample": True, "top_p": 0.99, "extra": 1}},
        "triviaqa": {"decoding_config": {"do_sample": True, "top_p": 0.95, "extra": 1}},
    }
    record = build_config_comparability_record(fake_results)
    assert "top_p" in record["comparisons"]["triviaqa"]["differing_fields"]
    assert "do_sample" not in record["comparisons"]["triviaqa"]["differing_fields"]
    print(f"  [PASS] build_config_comparability_record: detects the differing field (top_p), "
          f"not the matching one (do_sample)")

    tmp_dir = os.path.join(HERE, "results", "_selftest_eval_phase2")
    os.makedirs(tmp_dir, exist_ok=True)
    fake_all_results = {
        "truthfulqa": {"dataset": "truthfulqa", "composition": comp, "decoding_config": {},
                       "grouped": {"core_max": core_summary, "q_velocity": core_summary},
                       "grouped_paired_delta_RF": {"pooled": {"mean_delta": 0.0, "ci95": (0, 0), "excludes_zero": False},
                                                    "within_prompt": {"mean_delta": 0.0, "ci95": (0, 0), "excludes_zero": False}},
                       "harp": {"core_max": harp["core_max"], "q_velocity": harp["core_max"]},
                       "n_known": int(is_known.sum()), "n_beams": n_beams},
    }
    lb_path = os.path.join(tmp_dir, "leaderboard_selftest.json")
    update_leaderboard(fake_all_results, lb_path)
    with open(lb_path) as f:
        lb = json.load(f)
    assert len(lb["entries"]) == 4   # 2 conditions x 2 protocols
    print(f"  [PASS] update_leaderboard: wrote {len(lb['entries'])} entries")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=DATASETS, default=None)
    parser.add_argument("--model_folder", type=str, default="llama-3.1-8b-instruct")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--core-pooled-pt", type=str, default=None, help="truthfulqa only")
    parser.add_argument("--velocity-meta", type=str, default=None, help="truthfulqa only")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--leaderboard", type=str, default="results/leaderboard_v1.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.combine:
        all_results = {}
        for ds in DATASETS:
            p = os.path.join(args.results_dir, f"session06_phase2_{ds}.json")
            if os.path.exists(p):
                with open(p) as f:
                    all_results[ds] = json.load(f)
        if not all_results:
            print("ERROR: no per-dataset results found to combine."); sys.exit(1)
        print_master_table(all_results)
        record = build_config_comparability_record(all_results)
        print(f"\n[Config comparability] {json.dumps(record, indent=2)[:2000]}")
        out_path = "results/session06_phase2_metrics.json"
        with open(out_path, "w") as f:
            json.dump({"results": all_results, "config_comparability": record}, f, indent=2, default=str)
        print(f"\nWrote: {out_path}")
        lb_path = update_leaderboard(all_results, args.leaderboard)
        print(f"Wrote: {lb_path}")
        return

    if not args.dataset:
        print("ERROR: --dataset required (or --combine)."); sys.exit(1)

    if args.dataset == "truthfulqa":
        if not args.core_pooled_pt or not args.velocity_meta:
            print("ERROR: --core-pooled-pt and --velocity-meta required for truthfulqa."); sys.exit(1)
        core_raw, V_concat, y, prompt_idx, is_known, decoding_config = load_truthfulqa_features(
            args.core_pooled_pt, args.velocity_meta)
    else:
        if args.data_dir:
            data_dir = args.data_dir
        else:
            import yaml
            with open(os.path.join(HERE, "config.yaml")) as f:
                cfg = yaml.safe_load(f)
            data_dir = cfg["output"]["data_dir"]
        core_raw, V_concat, y, prompt_idx, is_known, decoding_config = load_new_dataset_features(
            args.dataset, data_dir, args.model_folder)

    result = run_dataset_eval(args.dataset, core_raw, V_concat, y, prompt_idx, is_known, decoding_config)

    out_json = args.output_json or f"results/session06_phase2_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote: {out_json}")
    print(f"Next: after all 4 datasets are run, call `python 43_eval_phase2.py --combine`")


if __name__ == "__main__":
    main()
