"""
41_harp_protocol_canonical.py -- Micro-session: HARP-Protocol Rows on the Canonical TruthfulQA
Pipeline
=================================================================================================
CPU-only, existing canonical TruthfulQA artifacts only -- NO generation, no extraction, no new
datasets, no interference with any running Phase-1 (session06) jobs.

Reuses, not reimplements: original_harp_split/fit_eval/cluster_bootstrap_ci from
26_grouped_baseline.py (s01), fold_pure_core from 37_eval_session05.py (s05) -- the same
robust-scale + fold-pure-Tucker machinery session04/05 already established, applied here under
the SINGLE 75/25-known HARP split instead of GroupKFold(5). session05's own run_harp_single()
(RF-only, no CI) is the direct precedent for this; this script's only real addition is doing both
RF and LR, and a 1000-rep cluster-bootstrap CI on the held-out split, per this micro-session's ask.

Where "the old 0.796 reference row" traces to: baseline_report.tex's Results table, "HOSVD (ours)"
row, LLaMA-3.1-8B / TruthfulQA column = 79.6%. That table's caption states the recipe explicitly:
"known/unknown split, zero leakage ... 320-dimensional compressed core features and Random
Forest" -- i.e. literally this same protocol, run earlier in the project before the v2/v3
canonical pipeline existed. Cross-checked (not silently trusted) against session01's independent
empirical re-derivation of the identical recipe on session01-era (v2) data: E1 gave AUROC=0.7952
(results/session01_metrics.json), consistent with 0.796 to within the sklearn-version/pipeline-
iteration drift already documented in reports/session01_repo_audit.md's 0.8094-origin analysis
for this exact recipe. Neither number is fabricated to force an exact match -- both are quoted as
found, per this project's standing practice (see that same 0.8094 precedent).

Usage:
  python 41_harp_protocol_canonical.py --self-test
  python 41_harp_protocol_canonical.py --core-pooled-pt <path> --velocity-meta <path>
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s01 = _load("s01", "26_grouped_baseline.py")
s05 = _load("s05", "37_eval_session05.py")

ORIGINAL_SEED = s01.ORIGINAL_SEED   # 42, matching prior E1 runs
N_BOOTSTRAP = 1000

# feature name -> (r_l, r_d), matching 37_eval_session05.py's harp_feature_map exactly, so these
# numbers are genuinely comparable to that session's (never-run-for-real) B1/C2 rows.
FEATURE_DIMS = {"core_max": (5, 64), "q_static": (5, 64), "q_velocity": (4, 64)}

OLD_REFERENCE = {
    "name": "core-max (old, pre-canonical)",
    "auroc": 0.796,
    "source": "baseline_report.tex Results table, 'HOSVD (ours)' row, LLaMA-3.1-8B / TruthfulQA "
              "column = 79.6%, captioned 'known/unknown split, zero leakage ... 320-dimensional "
              "compressed core features and Random Forest' -- the identical recipe, run before "
              "the v2/v3 canonical pipeline existed.",
    "cross_check_session01_E1": 0.7951670341801554,
}


# ==============================================================================
# STEP 1 -- HARP SPLIT ON CANONICAL LABELS
# ==============================================================================

def run_harp_split(is_known, prompt_idx, n_beams, seed=ORIGINAL_SEED):
    t_idx, v_idx = s01.original_harp_split(is_known, prompt_idx, n_beams, seed=seed)
    train_prompts = set(prompt_idx[t_idx].tolist())
    valid_prompts = set(prompt_idx[v_idx].tolist())
    disjoint = train_prompts.isdisjoint(valid_prompts)
    print(f"[Step 1] HARP split (seed={seed}): n_train={len(t_idx)}  n_valid={len(v_idx)}  "
          f"train_prompts={len(train_prompts)}  valid_prompts={len(valid_prompts)}")
    assert disjoint, "HARP split: train/valid prompt sets are NOT disjoint"
    print("  [PASS] prompt-level disjointness")
    print("  [CAVEAT] this reproduces HARP's split RECIPE with our own random partition, "
          "not their literal split file.")
    return t_idx, v_idx


# ==============================================================================
# STEP 2 -- EVALUATE RF + LR UNDER THE SPLIT, WITH CLUSTER-BOOTSTRAP CI
# ==============================================================================

def run_condition(X_raw, y, prompt_idx, t_idx, v_idx, r_l, r_d, seed=ORIGINAL_SEED,
                   n_boot=N_BOOTSTRAP, label="condition"):
    core = s05.fold_pure_core(X_raw, t_idx, None, r_l, r_d, seed)
    result = {"n_train": int(len(t_idx)), "n_valid": int(len(v_idx))}
    for clf in ("RF", "LR"):
        scores = s01.fit_eval(clf, core[t_idx], y[t_idx], core[v_idx], seed)
        auroc = float(roc_auc_score(y[v_idx], scores))
        ci = s01.cluster_bootstrap_ci(scores, y[v_idx], prompt_idx[v_idx], n_boot=n_boot, seed=seed)
        result[clf] = {"auroc": auroc, "ci95": list(ci)}
        print(f"  [{label}] {clf}: AUROC={auroc:.4f}  CI95=[{ci[0]:.4f}, {ci[1]:.4f}]")
    return result


# ==============================================================================
# TABLE
# ==============================================================================

def print_table(results):
    print(f"\n{'Row':32s} {'RF AUROC':>10s} {'RF CI95':>18s} {'LR AUROC':>10s} {'LR CI95':>18s}")
    r = OLD_REFERENCE
    print(f"{r['name']:32s} {r['auroc']:>10.4f} {'(none pinned)':>18s} {'n/a':>10s} {'n/a':>18s}")
    for name in ("core_max", "q_static", "q_velocity"):
        row = results[name]
        rf, lr = row["RF"], row["LR"]
        rf_ci = f"[{rf['ci95'][0]:.4f},{rf['ci95'][1]:.4f}]"
        lr_ci = f"[{lr['ci95'][0]:.4f},{lr['ci95'][1]:.4f}]"
        print(f"{name:32s} {rf['auroc']:>10.4f} {rf_ci:>18s} {lr['auroc']:>10.4f} {lr_ci:>18s}")


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: HARP split + RF/LR + bootstrap CI on synthetic data (CPU only)")
    print("=" * 70)

    data = s01.generate_synthetic_data(n_prompts=200, beams_per_prompt=10, L=9, D=64, seed=0)
    core_raw, y, prompt_idx, is_known = data["X"], data["y"], data["prompt_idx"], data["is_known"]
    n_beams = data["n_beams"]

    rng = np.random.default_rng(1)
    S_concat = rng.normal(0, 0.3, size=(n_beams, 9, 64)).astype(np.float32)
    for l in range(9):
        S_concat[:, l, 0:5] += core_raw[:, 0, 0:5]   # carry the same planted difficulty signal
    V_concat = rng.normal(0, 0.3, size=(n_beams, 8, 64)).astype(np.float32)
    for l in range(8):
        V_concat[:, l, 5:10] += core_raw[:, 0, 5:10]  # carry the same planted quality signal

    t_idx, v_idx = run_harp_split(is_known, prompt_idx, n_beams, seed=ORIGINAL_SEED)
    assert len(t_idx) + len(v_idx) == n_beams, "split must partition all beams"
    assert set(t_idx.tolist()).isdisjoint(set(v_idx.tolist())), "beam index sets must be disjoint"

    results = {}
    for name, X_raw in (("core_max", core_raw), ("q_static", S_concat), ("q_velocity", V_concat)):
        r_l, r_d = FEATURE_DIMS[name]
        results[name] = run_condition(X_raw, y, prompt_idx, t_idx, v_idx, r_l, r_d,
                                       seed=ORIGINAL_SEED, n_boot=200, label=name)
        for clf in ("RF", "LR"):
            auroc = results[name][clf]["auroc"]
            ci = results[name][clf]["ci95"]
            assert 0.0 <= auroc <= 1.0, f"{name}/{clf}: AUROC out of [0,1]: {auroc}"
            assert ci[0] <= ci[1], f"{name}/{clf}: CI lower > upper: {ci}"
    print("  [PASS] all AUROCs in [0,1], all CIs well-ordered")

    # a genuine planted-signal condition should clearly beat random (0.5)
    assert results["core_max"]["RF"]["auroc"] > 0.6, \
        f"planted-signal core_max should clearly beat chance: {results['core_max']['RF']['auroc']}"
    print(f"  [PASS] core_max RF AUROC={results['core_max']['RF']['auroc']:.4f} > 0.6 (planted signal recovered)")

    print_table(results)

    out_dir = os.path.join(HERE, "results", "_selftest_harp_protocol")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "harp_protocol_canonical.json")
    with open(out_path, "w") as f:
        json.dump({"old_reference": OLD_REFERENCE, "results": results,
                   "config": {"seed": ORIGINAL_SEED, "n_boot": 200}}, f, indent=2)
    with open(out_path) as f:
        reloaded = json.load(f)
    assert reloaded["results"]["core_max"]["RF"]["auroc"] == results["core_max"]["RF"]["auroc"]
    print(f"  [PASS] JSON write/reload round-trip: {out_path}")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-pooled-pt", type=str, default=None,
                         help="Canonical v3 pooled tensor (same file used throughout sessions 02-05).")
    parser.add_argument("--velocity-meta", type=str, default=None,
                         help="session04's velocity dataset meta path; sibling .npz has S95/S05/V95/V05.")
    parser.add_argument("--seed", type=int, default=ORIGINAL_SEED)
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--output-json", type=str, default="results/harp_protocol_canonical.json")
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
    n_beams = core_raw.shape[0]

    vel_npz_path = os.path.splitext(args.velocity_meta)[0].replace("_meta", "") + ".npz"
    vel_data = dict(np.load(vel_npz_path))
    S_concat = np.concatenate([vel_data["S95"], vel_data["S05"]], axis=2)
    V_concat = np.concatenate([vel_data["V95"], vel_data["V05"]], axis=2)

    t_idx, v_idx = run_harp_split(is_known, prompt_idx, n_beams, seed=args.seed)

    print("\n[Step 2] Evaluating RF + LR under the HARP split ...")
    results = {}
    for name, X_raw in (("core_max", core_raw), ("q_static", S_concat), ("q_velocity", V_concat)):
        r_l, r_d = FEATURE_DIMS[name]
        results[name] = run_condition(X_raw, y, prompt_idx, t_idx, v_idx, r_l, r_d,
                                       seed=args.seed, n_boot=args.n_boot, label=name)

    print_table(results)

    output = {"old_reference": OLD_REFERENCE, "results": results,
              "config": {"seed": args.seed, "n_boot": args.n_boot, "feature_dims": FEATURE_DIMS,
                         "core_pooled_pt": args.core_pooled_pt, "velocity_meta": args.velocity_meta}}
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote: {args.output_json}")


if __name__ == "__main__":
    main()
