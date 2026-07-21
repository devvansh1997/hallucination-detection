"""
26_grouped_baseline.py -- Session 01: Prompt-Grouped Re-Baselining
=====================================================================
Tests whether the Phase-1 HOSVD baseline (~0.80-0.81 AUROC on the pure
320-dim Tucker core; see reports/session01_repo_audit.md A5 for why the
literal "0.8094" figure actually traces to a 346-dim geometry-augmented
variant, not the pure core) survives prompt-grouped evaluation, and how
much of its signal is question-identity/difficulty vs. per-beam response
quality.

CPU-only. Consumes the already-cached pre-Tucker pooled tensors
(`{dataset}_pooled_maxenergy.pt`, produced by 21_generate_maxpool_datasets.py)
-- no GPU, no model weights, no re-extraction. Tucker/HOSVD projection is
cheap enough to refit per cross-validation fold on CPU.

Usage:
  python 26_grouped_baseline.py --model_folder llama-3.1-8b-instruct --dataset truthfulqa
  python 26_grouped_baseline.py --self-test
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

SEED = 0                 # standing instruction: seed=0 everywhere for NEW evaluations (E2-E4)
ORIGINAL_SEED = 42        # the seed the 0.8094-adjacent baseline actually used (A5) -- E1 only
R_L_DEFAULT = 5
R_D_DEFAULT = 64
N_SPLITS = 5
N_BOOTSTRAP = 1000


# ==============================================================================
# DATA ASSEMBLY (B1)
# ==============================================================================

def load_real_data(path):
    """Load the cached pre-Tucker pooled tensor file. Returns dict with
    X (N, L, D) float32, y (N,) int, prompt_idx (N,) int, is_known (n_prompts,) bool."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cached pooled tensor not found at {path}. This script consumes "
            f"activations already extracted by 21_generate_maxpool_datasets.py "
            f"(GPU stage) -- it does not run the model itself. Run extraction "
            f"on the cluster first, or point --data-path at an existing file.")
    data = torch.load(path, weights_only=False)
    for k in ("all_emb", "all_hallucination_flag", "prompt_indices"):
        if k not in data:
            raise ValueError(f"Cached file is missing required key '{k}' -- "
                              f"cannot proceed without guessing. Found keys: {list(data.keys())}")

    X = torch.stack(data["all_emb"]).float().numpy()          # (N, L, D)
    y = np.array([int(f) for f in data["all_hallucination_flag"]])
    prompt_idx = np.array(data["prompt_indices"], dtype=np.int64)
    is_known = np.array(data.get("all_is_known", []), dtype=bool)

    N = X.shape[0]
    if len(prompt_idx) != N:
        raise ValueError(f"Alignment failure: len(prompt_indices)={len(prompt_idx)} "
                          f"!= n_beams={N}. Refusing to guess an alignment.")
    n_prompts = len(np.unique(prompt_idx))
    if is_known.size and len(is_known) != n_prompts:
        raise ValueError(f"Alignment failure: len(all_is_known)={len(is_known)} "
                          f"!= n_unique_prompts={n_prompts}. Refusing to guess an alignment.")
    # spot-check: every prompt's beam indices must be contiguous under this ordering
    # (required by the original HARP split code we replicate in original_harp_split)
    counts = np.bincount(prompt_idx)
    if not np.all(counts == counts[0]):
        print(f"  [WARN] beams-per-prompt is not uniform (min={counts.min()}, "
              f"max={counts.max()}) -- proceeding, but downstream centroid/delta "
              f"features will average over however many beams each prompt actually has.")

    return {"X": X, "y": y, "prompt_idx": prompt_idx, "is_known": is_known,
            "n_prompts": n_prompts, "n_beams": N}


def generate_synthetic_data(n_prompts=200, beams_per_prompt=10, L=9, D=64, seed=0):
    """Plant two independent signal components in a synthetic (N, L, D) tensor:
    a per-PROMPT difficulty component (shared by all beams of a prompt, dims 0:5)
    and a per-BEAM quality component (independent per beam, dims 5:10). The label
    is a logistic function of both, so centroid-only (difficulty) and delta-only
    (quality) each retain genuine, partial signal after decomposition."""
    rng = np.random.default_rng(seed)
    n_beams = n_prompts * beams_per_prompt

    difficulty = rng.normal(0, 1, size=n_prompts)              # per-prompt
    quality = rng.normal(0, 1, size=n_beams)                   # per-beam

    prompt_idx = np.repeat(np.arange(n_prompts), beams_per_prompt)

    v_d = rng.normal(0, 1, size=5); v_d /= np.linalg.norm(v_d)
    v_q = rng.normal(0, 1, size=5); v_q /= np.linalg.norm(v_q)

    X = rng.normal(0, 0.3, size=(n_beams, L, D)).astype(np.float32)  # background noise
    diff_component = difficulty[prompt_idx][:, None] * v_d[None, :] \
        + rng.normal(0, 0.1, size=(n_beams, 5))                 # small per-beam noise on top
    qual_component = quality[:, None] * v_q[None, :]
    for l in range(L):
        X[:, l, 0:5] += diff_component
        X[:, l, 5:10] += qual_component

    logit = 0.9 * difficulty[prompt_idx] + 0.9 * quality
    p_halluc = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(0, 1, size=n_beams) < p_halluc).astype(np.int64)

    is_known = np.zeros(n_prompts, dtype=bool)
    for p in range(n_prompts):
        beams = np.where(prompt_idx == p)[0]
        is_known[p] = (y[beams] == 0).any()   # "known" = at least one truthful (correct) beam

    return {"X": X, "y": y, "prompt_idx": prompt_idx, "is_known": is_known,
            "n_prompts": n_prompts, "n_beams": n_beams}


# ==============================================================================
# ORIGINAL SPLIT + HOSVD (replicated from 22_evaluate_phase1_kinematics_and_q.py, A4/A5)
# ==============================================================================

def original_harp_split(is_known, prompt_idx, N, seed=ORIGINAL_SEED):
    """Exact replication of the split in 22_evaluate_phase1_kinematics_and_q.py:153-163.
    Prompt-grouped: 75% of "known" prompts -> train, 25% of known + all unknown -> valid."""
    known_idx = np.where(is_known)[0]
    rng_state = np.random.get_state()
    np.random.seed(seed)
    known_idx = known_idx.copy()
    np.random.shuffle(known_idx)
    np.random.set_state(rng_state)

    s = int(len(known_idx) * 0.75)
    tp = set(known_idx[:s].tolist())
    vp = set(known_idx[s:].tolist())
    vp.update(np.where(~is_known)[0].tolist())

    t_mask = np.array([prompt_idx[i] in tp for i in range(N)])
    v_mask = np.array([prompt_idx[i] in vp for i in range(N)])
    return np.where(t_mask)[0], np.where(v_mask)[0]


def mad_scale(X, train_idx):
    """Train-only median/MAD scaling -- robust to LLaMA structural outlier channels
    (standing constraint), replicated from 22_evaluate_phase1_kinematics_and_q.py:165-169."""
    X_t = X[train_idx]
    med = np.median(X_t, axis=0)
    mad = np.median(np.abs(X_t - med), axis=0) + 1e-6
    return (X - med) / mad


def compute_ul_ud(X_train, r_l, r_d):
    """Gram-trick eigendecomposition, train-only. Replicated from
    22_evaluate_phase1_kinematics_and_q.py:98-111 (chunked A_D accumulation)."""
    N, L, D = X_train.shape
    X_f = X_train.transpose(1, 0, 2).reshape(L, -1).astype(np.float64)
    A_L = X_f @ X_f.T
    _, U_L = np.linalg.eigh(A_L)
    U_L = np.flip(U_L[:, -r_l:], axis=1).copy()

    X_d = X_train.transpose(2, 0, 1).reshape(D, -1).astype(np.float64)
    A_D = np.zeros((D, D), dtype=np.float64)
    chunk = 50000
    for start in range(0, N * L, chunk):
        end = min(start + chunk, N * L)
        blk = X_d[:, start:end]
        A_D += blk @ blk.T
    _, U_D = np.linalg.eigh(A_D)
    U_D = np.flip(U_D[:, -r_d:], axis=1).copy()
    return U_L.astype(np.float32), U_D.astype(np.float32)


def project_core(X, U_L, U_D):
    """(N, L, D) -> (N, r_l * r_d) core features, matching
    22_evaluate_phase1_kinematics_and_q.py:180-183 ('X.float() @ U_D' then '.T @ U_L')."""
    temp = X.astype(np.float32) @ U_D           # (N, L, r_d)
    G = np.einsum('nld,lr->nrd', temp, U_L)     # (N, r_l, r_d)  [temp.transpose(1,2) @ U_L then transpose back]
    return G.reshape(G.shape[0], -1)            # (N, r_l * r_d)


# ==============================================================================
# METRICS HELPERS
# ==============================================================================

def cluster_bootstrap_ci(scores, labels, beam_prompt_ids, n_boot=N_BOOTSTRAP, seed=SEED):
    """95% CI via resampling PROMPTS (not beams) with replacement."""
    rng = np.random.default_rng(seed)
    unique_prompts = np.unique(beam_prompt_ids)
    idx_by_prompt = {p: np.where(beam_prompt_ids == p)[0] for p in unique_prompts}

    aurocs = []
    for _ in range(n_boot):
        drawn = rng.choice(unique_prompts, size=len(unique_prompts), replace=True)
        beam_idx = np.concatenate([idx_by_prompt[p] for p in drawn])
        y_r, s_r = labels[beam_idx], scores[beam_idx]
        if len(np.unique(y_r)) < 2:
            continue
        aurocs.append(roc_auc_score(y_r, s_r))
    if not aurocs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5)))


def within_prompt_auroc(scores, labels, beam_prompt_ids):
    """P(score_halluc > score_truthful | same prompt), ties=0.5, pooled over all
    same-prompt opposite-label pairs. Also returns mixed/all-truthful/all-halluc counts."""
    n_mixed = n_all_truthful = n_all_halluc = 0
    concordant = 0.0
    total_pairs = 0
    for p in np.unique(beam_prompt_ids):
        idx = np.where(beam_prompt_ids == p)[0]
        y_p, s_p = labels[idx], scores[idx]
        halluc_scores = s_p[y_p == 1]
        truthful_scores = s_p[y_p == 0]
        if len(halluc_scores) == 0:
            n_all_truthful += 1
            continue
        if len(truthful_scores) == 0:
            n_all_halluc += 1
            continue
        n_mixed += 1
        diffs = halluc_scores[:, None] - truthful_scores[None, :]
        concordant += (diffs > 0).sum() + 0.5 * (diffs == 0).sum()
        total_pairs += diffs.size

    auroc = concordant / total_pairs if total_pairs > 0 else float("nan")
    return {"within_prompt_auroc": float(auroc), "n_mixed_prompts": n_mixed,
            "n_all_truthful_prompts": n_all_truthful, "n_all_hallucinated_prompts": n_all_halluc,
            "n_pairs": int(total_pairs)}


def fit_eval(clf_name, X_train, y_train, X_val, seed):
    if clf_name == "RF":
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      random_state=seed, n_jobs=-1)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_val)[:, 1]
    elif clf_name == "LR":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xva = scaler.transform(X_val)
        clf = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed)
        clf.fit(Xtr, y_train)
        return clf.predict_proba(Xva)[:, 1]
    raise ValueError(clf_name)


# ==============================================================================
# E1 -- REPLICATION ANCHOR
# ==============================================================================

def eval_E1(data, r_l, r_d):
    X, y, prompt_idx, is_known = data["X"], data["y"], data["prompt_idx"], data["is_known"]
    N = data["n_beams"]

    t_idx, v_idx = original_harp_split(is_known, prompt_idx, N, seed=ORIGINAL_SEED)
    assert set(prompt_idx[t_idx].tolist()).isdisjoint(set(prompt_idx[v_idx].tolist())), \
        "E1: train/valid prompt sets are not disjoint -- split replication is broken"

    X_scaled = mad_scale(X, t_idx)
    U_L, U_D = compute_ul_ud(X_scaled[t_idx], r_l, r_d)
    core = project_core(X_scaled, U_L, U_D)     # (N, r_l*r_d) -- the pure "320-dim" feature

    scores = fit_eval("RF", core[t_idx], y[t_idx], core[v_idx], seed=ORIGINAL_SEED)
    y_val = y[v_idx]
    # y already has hallucination=1 (original orientation); canonical (halluc-positive)
    # orientation is IDENTICAL here -- see reports/session01_repo_audit.md A5.
    auroc_original = roc_auc_score(y_val, scores)
    auroc_canonical = auroc_original

    ci = cluster_bootstrap_ci(scores, y_val, prompt_idx[v_idx])

    return {
        "auroc_original_orientation": float(auroc_original),
        "auroc_canonical_orientation": float(auroc_canonical),
        "orientation_note": "identical: all_hallucination_flag is already coded 1=hallucination",
        "ci95": ci,
        "n_train": int(len(t_idx)), "n_valid": int(len(v_idx)),
        "feature_dim": int(core.shape[1]),
        "target_reference_auroc": 0.8094,
        "note": ("Target 0.8094 traces to a 346-dim variant (core + raw geometric "
                 "features), not the pure r_l*r_d core reproduced here -- see A5. "
                 "sklearn-version-dependent by ~0.005-0.01 AUROC even on the exact "
                 "same feature set/seed/split."),
    }


# ==============================================================================
# E2/E3/E4 -- GROUPED CV + SIGNAL DECOMPOSITION
# ==============================================================================

def eval_grouped(data, r_l, r_d, n_splits=N_SPLITS, seed=SEED):
    X, y, prompt_idx = data["X"], data["y"], data["prompt_idx"]
    N = data["n_beams"]
    unique_prompts = np.unique(prompt_idx)
    if len(unique_prompts) < n_splits:
        n_splits = max(2, len(unique_prompts))

    gkf = GroupKFold(n_splits=n_splits)

    per_fold_auroc = {("core", "RF"): [], ("core", "LR"): [],
                       ("centroid", "RF"): [], ("centroid", "LR"): [],
                       ("delta", "RF"): [], ("delta", "LR"): []}
    oof_scores = {k: np.full(N, np.nan) for k in per_fold_auroc}

    for fold_i, (tr_beam, va_beam) in enumerate(gkf.split(X, y, groups=prompt_idx)):
        tr_prompts = set(prompt_idx[tr_beam].tolist())
        va_prompts = set(prompt_idx[va_beam].tolist())
        assert tr_prompts.isdisjoint(va_prompts), \
            f"fold {fold_i}: train/val prompt sets are NOT disjoint"

        X_scaled = mad_scale(X, tr_beam)               # fold-pure: train-only median/MAD
        U_L, U_D = compute_ul_ud(X_scaled[tr_beam], r_l, r_d)   # fold-pure Tucker refit (B2b)
        core = project_core(X_scaled, U_L, U_D)

        # prompt centroids computed within this fold's own core features (no label use)
        centroid = np.zeros_like(core)
        for p in np.unique(prompt_idx):
            idx = np.where(prompt_idx == p)[0]
            centroid[idx] = core[idx].mean(axis=0, keepdims=True)
        delta = core - centroid

        for feat_name, feats in (("core", core), ("centroid", centroid), ("delta", delta)):
            for clf_name in ("RF", "LR"):
                fold_seed = seed + fold_i
                scores = fit_eval(clf_name, feats[tr_beam], y[tr_beam], feats[va_beam], fold_seed)
                oof_scores[(feat_name, clf_name)][va_beam] = scores
                per_fold_auroc[(feat_name, clf_name)].append(
                    float(roc_auc_score(y[va_beam], scores)))

    results = {}
    for key, foldlist in per_fold_auroc.items():
        feat_name, clf_name = key
        scores = oof_scores[key]
        ci = cluster_bootstrap_ci(scores, y, prompt_idx, seed=seed)
        results[f"{feat_name}_{clf_name}"] = {
            "per_fold_auroc": foldlist,
            "mean_auroc": float(np.mean(foldlist)),
            "std_auroc": float(np.std(foldlist)),
            "pooled_oof_auroc": float(roc_auc_score(y, scores)),
            "ci95": ci,
        }

    e3 = within_prompt_auroc(oof_scores[("core", "RF")], y, prompt_idx)
    e3_lr = within_prompt_auroc(oof_scores[("core", "LR")], y, prompt_idx)
    e4b_within = within_prompt_auroc(oof_scores[("delta", "RF")], y, prompt_idx)

    return results, e3, e3_lr, e4b_within, n_splits


# ==============================================================================
# MAIN / REPORTING
# ==============================================================================

def print_summary_table(e1, grouped):
    rows = [
        ("E1", e1["auroc_canonical_orientation"], None, None),
        ("E2-RF", grouped["core_RF"]["mean_auroc"], grouped["core_RF"]["std_auroc"],
         grouped["core_RF"]["ci95"]),
        ("E2-LR", grouped["core_LR"]["mean_auroc"], grouped["core_LR"]["std_auroc"],
         grouped["core_LR"]["ci95"]),
        ("E4a", grouped["centroid_RF"]["mean_auroc"], grouped["centroid_RF"]["std_auroc"],
         grouped["centroid_RF"]["ci95"]),
        ("E4b", grouped["delta_RF"]["mean_auroc"], grouped["delta_RF"]["std_auroc"],
         grouped["delta_RF"]["ci95"]),
    ]
    print(f"\n  {'Row':10s}  {'AUROC':>8s}  {'std':>7s}  {'95% CI':>18s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*18}")
    for name, mean, std, ci in rows:
        std_s = f"{std:.4f}" if std is not None else "   n/a"
        ci_s = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci is not None else "n/a"
        print(f"  {name:10s}  {mean:8.4f}  {std_s:>7s}  {ci_s:>18s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="_maxenergy")
    parser.add_argument("--data-path", type=str, default=None,
                         help="Direct path to a cached *_pooled*.pt file, overrides "
                              "--model_folder/--dataset/--suffix")
    parser.add_argument("--r_l", type=int, default=R_L_DEFAULT)
    parser.add_argument("--r_d", type=int, default=R_D_DEFAULT)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--output-json", type=str, default="results/session01_metrics.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print("=" * 70)
        print("  SELF-TEST (synthetic data, no cluster/GPU/cached artifacts needed)")
        print("=" * 70)
        data = generate_synthetic_data(n_prompts=200, beams_per_prompt=10, L=9, D=64, seed=SEED)
        r_l, r_d = 5, 20   # smaller r_d than production default -- D=64 here, not 4096
        out_path = os.path.join(os.path.dirname(__file__) or ".",
                                 "results", "session01_selftest_metrics.json")
    else:
        if args.data_path:
            path = args.data_path
        else:
            if not (args.model_folder and args.dataset):
                print("ERROR: provide --data-path, or both --model_folder and --dataset.")
                sys.exit(1)
            with open(os.path.join(os.path.dirname(__file__) or ".", "config.yaml")) as f:
                cfg = yaml.safe_load(f)
            data_dir = cfg["output"]["data_dir"]
            path = os.path.join(data_dir, args.model_folder,
                                 f"{args.dataset}_pooled{args.suffix}.pt")
        print(f"Loading: {path}")
        data = load_real_data(path)
        r_l, r_d = args.r_l, args.r_d
        out_path = args.output_json

    print(f"Prompts: {data['n_prompts']}  Beams: {data['n_beams']}  "
          f"Halluc rate: {data['y'].mean()*100:.2f}%")

    t0 = time.time()
    print("\n[E1] Replication anchor (original HARP split, seed=42) ...")
    e1 = eval_E1(data, r_l, r_d)
    print(f"  AUROC = {e1['auroc_canonical_orientation']:.4f}  "
          f"(target ~0.8094 -- see note in JSON)  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    print(f"\n[E2-E4] GroupKFold(n_splits={args.n_splits}) grouped by prompt, seed={SEED} ...")
    grouped, e3, e3_lr, e4b_within, n_splits_used = eval_grouped(
        data, r_l, r_d, n_splits=args.n_splits, seed=SEED)
    print(f"  done. [{time.time()-t0:.1f}s]")

    print_summary_table(e1, grouped)
    print(f"\n  E3  within-prompt AUROC (core, RF): {e3['within_prompt_auroc']:.4f}  "
          f"(mixed={e3['n_mixed_prompts']}, all-truthful={e3['n_all_truthful_prompts']}, "
          f"all-halluc={e3['n_all_hallucinated_prompts']})")
    print(f"  E4b within-prompt AUROC (delta, RF): {e4b_within['within_prompt_auroc']:.4f}")

    audit_flags = {
        "split_type_found": "prompt-grouped (HARP known/unknown, 75/25), NOT beam-level leakage",
        "tucker_factors_fit_on": "train-only, consistently across the whole repo (no fit-on-all leak found)",
        "prompt_tokens_in_pooling_window": False,
        "class_balance_in_repo_brief_vs_actual": "brief said ~62pct truthful; cached data shows 61.18pct hallucinated",
        "0.8094_origin": "346-dim core+raw-geometric variant (V2/V3), not pure r_l*r_d core; sklearn-version sensitive",
    }

    output = {
        "counts": {"n_prompts": data["n_prompts"], "n_beams": data["n_beams"],
                   "class_balance_hallucinated_pct": float(data["y"].mean() * 100),
                   "n_mixed_prompts": e3["n_mixed_prompts"],
                   "n_all_truthful_prompts": e3["n_all_truthful_prompts"],
                   "n_all_hallucinated_prompts": e3["n_all_hallucinated_prompts"]},
        "audit_flags": audit_flags,
        "E1": e1,
        "E2": {"RF": grouped["core_RF"], "LR": grouped["core_LR"], "n_splits": n_splits_used},
        "E3": e3, "E3_LR": e3_lr,
        "E4a_centroid_only": {"RF": grouped["centroid_RF"], "LR": grouped["centroid_LR"]},
        "E4b_delta_only": {"RF": grouped["delta_RF"], "LR": grouped["delta_LR"],
                            "within_prompt_auroc": e4b_within},
        "config": {"r_l": r_l, "r_d": r_d, "seed_new_evals": SEED,
                   "seed_original_split": ORIGINAL_SEED, "self_test": args.self_test},
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote: {out_path}")

    if args.self_test:
        print("\n[SELF-TEST ASSERTIONS]")
        assert grouped["centroid_RF"]["pooled_oof_auroc"] > 0.5, \
            f"E4a (centroid-only) AUROC not > 0.5: {grouped['centroid_RF']['pooled_oof_auroc']}"
        print(f"  [PASS] E4a (centroid-only) AUROC = "
              f"{grouped['centroid_RF']['pooled_oof_auroc']:.4f} > 0.5")
        assert grouped["delta_RF"]["pooled_oof_auroc"] > 0.5, \
            f"E4b (delta-only) AUROC not > 0.5: {grouped['delta_RF']['pooled_oof_auroc']}"
        print(f"  [PASS] E4b (delta-only) AUROC = "
              f"{grouped['delta_RF']['pooled_oof_auroc']:.4f} > 0.5")
        assert os.path.exists(out_path), "JSON output was not written"
        print(f"  [PASS] JSON written to {out_path}")
        print("\n[PASS] All self-test assertions passed.")


if __name__ == "__main__":
    main()
