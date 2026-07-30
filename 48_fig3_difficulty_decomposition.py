"""
48_fig3_difficulty_decomposition.py -- Session 08 FIG 3: what the pooled metric actually measures
=====================================================================================================
CPU-only, read-only. Writes exclusively to analysis_out/{model_folder}/ (gitignored).

Pooled AUROC ranks every beam against every other beam, including beams answering DIFFERENT
questions. A detector can therefore score well by recognising "this is an easy question" rather
than "this particular answer is a lie". This figure separates the two.

Three feature conditions on the same fold-local core_max features:
  (i)   full        -- the baseline, exactly what Phase 3 reports
  (ii)  centroid    -- every beam replaced by its prompt's 10-beam mean. All within-prompt
                       variation is destroyed, so whatever AUROC survives is PURELY question-level
                       difficulty. Within-prompt AUROC is undefined here by construction (all beams
                       of a prompt are identical and cannot be ranked against each other), which is
                       why the spec asks for it only on (i) and (iii).
  (iii) delta       -- every beam minus its prompt centroid. Question-level information is removed,
                       leaving only per-response signal.

Read together: if (ii) alone nearly matches (i), the pooled metric is largely a difficulty detector.

The centroid/delta split is applied to the CORE features (post-Tucker), not to the raw tensors, so
all three conditions share one fold-local basis and differ only in the decomposition -- otherwise
the Tucker basis itself would change between conditions and confound the comparison. Recorded as a
Deviation since the prompt does not specify which side of the Tucker fit the split happens on.

Usage:
  python 48_fig3_difficulty_decomposition.py --self-test
  python 48_fig3_difficulty_decomposition.py --dataset all
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import importlib.util as _ilu

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = _ilu.spec_from_file_location(name, os.path.join(HERE, filename))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


AL = _load("analysis_loader", "45_analysis_loader.py")
s02_eval = _load("s02_eval", "43_eval_phase2.py")
s01 = _load("s01", "26_grouped_baseline.py")

R_L, R_D = 5, 64
CONDITIONS = ["full", "centroid", "delta"]
CONDITION_LABEL = {"full": "(i) full", "centroid": "(ii) centroid-only", "delta": "(iii) delta-only"}
SESSION01_TOLERANCE = 0.01


def prompt_centroids(core, prompt_idx):
    """Per-prompt mean, broadcast back to every beam of that prompt. Uses an index-sorted
    reduction rather than a Python loop over prompts -- at 99,600 beams the loop is the difference
    between seconds and minutes."""
    uniq, inv = np.unique(prompt_idx, return_inverse=True)
    sums = np.zeros((len(uniq), core.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, core.astype(np.float64))
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)[:, None]
    return (sums / counts)[inv].astype(np.float32)


def decompose(core, prompt_idx, condition):
    if condition == "full":
        return core
    cent = prompt_centroids(core, prompt_idx)
    return cent if condition == "centroid" else (core - cent)


def run_dataset(ds, data_dir, model_folder, n_boot):
    info = AL.load_arrays(ds, data_dir, model_folder, need=("core",))
    core_raw, y, pid = info["arrays"]["core"], info["y"], info["prompt_idx"]
    folds = AL.make_folds(y, pid)

    oof = {c: {"RF": np.full(len(y), np.nan), "LR": np.full(len(y), np.nan)} for c in CONDITIONS}
    t0 = time.time()
    for fi, (tr, va) in enumerate(folds):
        # ONE fold-local Tucker basis, shared by all three conditions
        core = s02_eval.fold_pure_core_randomized(core_raw, tr, R_L, R_D, AL.SEED + fi)
        for cond in CONDITIONS:
            X = decompose(core, pid, cond)
            for clf in ("RF", "LR"):
                oof[cond][clf][va] = s01.fit_eval(clf, X[tr], y[tr], X[va], AL.SEED + fi)
        print(f"    [{ds}] fold {fi+1}/{len(folds)} done ({time.time()-t0:.0f}s cumulative)", flush=True)

    res = {"dataset": ds, "n_beams": info["n_beams"], "n_prompts": info["n_prompts"],
           "alignment_verified": info["alignment_verified"], "provenance": info["provenance"],
           "conditions": {}}
    for cond in CONDITIONS:
        entry = {}
        for clf in ("RF", "LR"):
            s = oof[cond][clf]
            pooled = float(roc_auc_score(y, s))
            pooled_ci = AL.cluster_bootstrap_ci(
                lambda idx, s=s: roc_auc_score(y[idx], s[idx]) if len(np.unique(y[idx])) > 1 else None,
                pid, n_boot=n_boot)
            e = {"pooled_auroc": pooled, "pooled_ci95": pooled_ci["ci95"]}
            if cond != "centroid":   # undefined for centroid: all beams of a prompt are identical
                wp = s01.within_prompt_auroc(s, y, pid)
                wp_ci = AL.cluster_bootstrap_ci(
                    lambda idx, s=s: (lambda r: r["within_prompt_auroc"] if r["n_pairs"] else None)(
                        s01.within_prompt_auroc(s[idx], y[idx], pid[idx])),
                    pid, n_boot=n_boot)
                e["within_prompt_auroc"] = float(wp["within_prompt_auroc"])
                e["within_prompt_ci95"] = wp_ci["ci95"]
                e["n_pairs"] = int(wp["n_pairs"])
                e["pooled_minus_within"] = float(pooled - wp["within_prompt_auroc"])
            entry[clf] = e
        res["conditions"][cond] = entry

    rf = res["conditions"]
    res["gap_rf"] = rf["full"]["RF"]["pooled_minus_within"]
    res["centroid_share_rf"] = float((rf["centroid"]["RF"]["pooled_auroc"] - 0.5) /
                                      max(rf["full"]["RF"]["pooled_auroc"] - 0.5, 1e-9))
    print(f"  [{ds}] RF pooled full={rf['full']['RF']['pooled_auroc']:.4f} "
          f"centroid={rf['centroid']['RF']['pooled_auroc']:.4f} "
          f"delta={rf['delta']['RF']['pooled_auroc']:.4f} | "
          f"pooled-minus-within={res['gap_rf']:+.4f} | "
          f"centroid share of lift={res['centroid_share_rf']*100:.0f}%", flush=True)
    return res


def compare_session01(results, path):
    """FIG 3 replicates a session-01 TruthfulQA analysis under the current canonical pipeline.
    Report both numbers and flag any discrepancy > 0.01 rather than quietly superseding."""
    if "truthfulqa" not in results or not os.path.exists(path):
        return {"status": "unavailable",
                "note": f"session-01 reference not compared ({'no truthfulqa in run' if 'truthfulqa' not in results else path + ' absent'})"}
    with open(path) as f:
        s01m = json.load(f)
    flat, stack = {}, [("", s01m)]
    while stack:
        pre, node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                stack.append((f"{pre}.{k}" if pre else k, v))
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            flat[pre] = float(node)
    ours = results["truthfulqa"]["conditions"]["full"]["RF"]
    cands = {k: v for k, v in flat.items()
             if any(t in k.lower() for t in ("auroc", "within")) and 0.4 < v < 1.0}
    out = {"status": "compared", "session01_file": path,
           "ours_pooled_rf": ours["pooled_auroc"], "ours_within_rf": ours.get("within_prompt_auroc"),
           "session01_candidate_metrics": cands, "flags": []}
    for k, v in cands.items():
        for label, mine in (("pooled", ours["pooled_auroc"]), ("within_prompt", ours.get("within_prompt_auroc"))):
            if mine is None:
                continue
            if abs(v - mine) <= SESSION01_TOLERANCE:
                out["flags"].append(f"MATCH within {SESSION01_TOLERANCE}: session01 '{k}'={v:.4f} vs ours {label}={mine:.4f}")
    if not out["flags"]:
        out["flags"].append(
            f"NO session-01 metric matched either of our TruthfulQA RF values within "
            f"{SESSION01_TOLERANCE} (pooled={ours['pooled_auroc']:.4f}, "
            f"within={ours.get('within_prompt_auroc')}). Discrepancy > tolerance -- the current "
            f"canonical pipeline does not reproduce the session-01 numbers exactly; inspect "
            f"'session01_candidate_metrics' and reconcile before citing either.")
    return out


def build_caption(results, s01_cmp):
    gaps = {AL.DATASET_STYLE[d]["label"]: r["gap_rf"] for d, r in results.items()}
    pos = [k for k, v in gaps.items() if v > 0]
    neg = [k for k, v in gaps.items() if v < 0]
    shares = {AL.DATASET_STYLE[d]["label"]: r["centroid_share_rf"] for d, r in results.items()}
    hi = max(shares, key=shares.get)
    sign = ""
    if pos and neg:
        sign = (f" The sign of the gap differs by dataset -- positive for {', '.join(pos)} "
                f"(pooled flatters the detector, consistent with question-difficulty "
                f"confounding) and negative for {', '.join(neg)} (pooled understates it, "
                f"consistent with cross-question variance adding noise the within-prompt "
                f"comparison never sees). That sign is itself diagnostic and is why we report "
                f"both metrics rather than either alone.")
    elif pos:
        sign = f" The gap is positive for every dataset, largest for {max(gaps, key=gaps.get)}."
    flag = ""
    if s01_cmp.get("flags") and s01_cmp["flags"][0].startswith("NO session-01"):
        flag = (" NOTE: our TruthfulQA numbers do not reproduce the session-01 values within "
                "0.01; see the JSON's session01_comparison block.")
    return (
        "Difficulty decomposition of the core_max features, LLaMA-3.1-8B, grouped 5-fold by prompt, "
        "Random Forest. (i) full features; (ii) centroid-only, every beam replaced by its prompt's "
        "10-beam mean, which retains only question-level difficulty; (iii) delta-only, every beam "
        "minus its prompt centroid, which retains only per-response signal. Error bars are 95% "
        "cluster-bootstrap intervals resampling prompts. Centroid-only has no within-prompt AUROC "
        "by construction: all beams of a prompt are identical and cannot be ranked against one "
        f"another. Question-level difficulty alone accounts for the largest share of the pooled "
        f"lift on {hi} ({shares[hi]*100:.0f}%). Right panel: pooled minus within-prompt AUROC per "
        f"dataset." + sign + flag +
        " TruthfulQA's tensors originate from the older Route-N path while the other three come "
        "from script 42; assumed equivalent by construction, not verified.")


def plot(results, stem):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.1, 1.0]})
    ax = axes[0]
    ds_list = list(results)
    n_c = len(CONDITIONS)
    width = 0.8 / n_c
    hatches = ["", "///", "..."]   # grayscale-safe condition encoding
    for ci, cond in enumerate(CONDITIONS):
        xs = np.arange(len(ds_list)) + (ci - (n_c - 1) / 2) * width
        vals = [results[d]["conditions"][cond]["RF"]["pooled_auroc"] for d in ds_list]
        los = [v - results[d]["conditions"][cond]["RF"]["pooled_ci95"][0] for v, d in zip(vals, ds_list)]
        his = [results[d]["conditions"][cond]["RF"]["pooled_ci95"][1] - v for v, d in zip(vals, ds_list)]
        ax.bar(xs, vals, width * 0.92, yerr=[los, his], capsize=3, hatch=hatches[ci],
               color=[AL.DATASET_STYLE[d]["color"] for d in ds_list],
               edgecolor="k", lw=0.7, alpha=0.55 + 0.15 * ci, label=CONDITION_LABEL[cond])
    ax.set_xticks(range(len(ds_list)))
    ax.set_xticklabels([AL.DATASET_STYLE[d]["label"] for d in ds_list])
    ax.axhline(0.5, color="k", lw=0.9, ls=(0, (1, 2)))
    ax.set_ylabel("pooled AUROC (RF)")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("(A) pooled AUROC by feature decomposition", fontsize=11)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=8, loc="lower right")

    ax2 = axes[1]
    gaps = [results[d]["gap_rf"] for d in ds_list]
    ax2.bar(range(len(ds_list)), gaps,
            color=[AL.DATASET_STYLE[d]["color"] for d in ds_list], edgecolor="k", lw=0.7)
    ax2.axhline(0.0, color="k", lw=1.0)
    ax2.set_xticks(range(len(ds_list)))
    ax2.set_xticklabels([AL.DATASET_STYLE[d]["label"] for d in ds_list], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("pooled $-$ within-prompt AUROC")
    ax2.set_title("(B) signed gap (full features)", fontsize=10)
    ax2.grid(axis="y", alpha=0.25, ls=":")
    fig.suptitle("FIG 3 -- what the pooled metric measures: difficulty vs per-response detection",
                 fontsize=12)
    fig.tight_layout()
    return AL.save_figure(fig, stem)


def self_test():
    print("=" * 70)
    print("  SELF-TEST: FIG 3 (synthetic, no cluster files)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    n_prompts, per_prompt, D = 40, 10, 32
    n = n_prompts * per_prompt
    pid = np.repeat(np.arange(n_prompts), per_prompt)

    core = rng.standard_normal((n, D)).astype(np.float32)
    cent = prompt_centroids(core, pid)
    assert cent.shape == core.shape
    for p in np.unique(pid)[:5]:
        m = core[pid == p].mean(axis=0)
        assert np.allclose(cent[pid == p][0], m, atol=1e-4), "centroid must equal the prompt mean"
    assert np.allclose(cent[pid == pid[0]] - cent[pid == pid[0]][0], 0, atol=1e-6), \
        "all beams of a prompt must share one centroid"
    delta = decompose(core, pid, "delta")
    for p in np.unique(pid)[:5]:
        assert np.allclose(delta[pid == p].mean(axis=0), 0, atol=1e-4), "delta must be prompt-centered"
    assert np.allclose(decompose(core, pid, "centroid") + delta, core, atol=1e-4), \
        "centroid + delta must reconstruct the original features exactly"
    print("  [PASS] decompose: centroid==prompt mean, delta prompt-centered, centroid+delta==full")

    # PURE difficulty: label is constant within a prompt -> centroid keeps everything, delta nothing
    y_diff = np.repeat((rng.random(n_prompts) < 0.5).astype(np.int64), per_prompt)
    sig = np.repeat(rng.standard_normal(n_prompts), per_prompt)
    core_d = (rng.standard_normal((n, D)) * 0.3).astype(np.float32)
    core_d[:, 0] += 3.0 * y_diff + 0.2 * sig
    a_full = roc_auc_score(y_diff, decompose(core_d, pid, "full")[:, 0])
    a_cent = roc_auc_score(y_diff, decompose(core_d, pid, "centroid")[:, 0])
    a_delt = roc_auc_score(y_diff, decompose(core_d, pid, "delta")[:, 0])
    assert a_cent > 0.95 and a_full > 0.95, f"difficulty-only signal must survive centroid: {a_cent}"
    assert abs(a_delt - 0.5) < 0.12, f"delta must destroy a purely question-level signal: {a_delt}"
    print(f"  [PASS] pure question-difficulty signal: full={a_full:.3f} centroid={a_cent:.3f} "
          f"delta={a_delt:.3f} (delta correctly ~chance)")

    # PURE per-response: labels balanced within every prompt -> centroid nothing, delta everything
    y_resp = np.tile(np.array([0, 1] * (per_prompt // 2)), n_prompts)
    core_r = (rng.standard_normal((n, D)) * 0.3).astype(np.float32)
    core_r[:, 0] += 3.0 * y_resp
    b_cent = roc_auc_score(y_resp, decompose(core_r, pid, "centroid")[:, 0])
    b_delt = roc_auc_score(y_resp, decompose(core_r, pid, "delta")[:, 0])
    assert abs(b_cent - 0.5) < 0.08, f"centroid must destroy a purely per-response signal: {b_cent}"
    assert b_delt > 0.95, f"delta must retain a purely per-response signal: {b_delt}"
    print(f"  [PASS] pure per-response signal: centroid={b_cent:.3f} (~chance) delta={b_delt:.3f}")

    fake = {ds: {"gap_rf": g, "centroid_share_rf": 0.6,
                 "conditions": {c: {"RF": {"pooled_auroc": 0.8, "pooled_ci95": [0.78, 0.82],
                                           **({} if c == "centroid" else
                                              {"within_prompt_auroc": 0.75,
                                               "within_prompt_ci95": [0.72, 0.78],
                                               "pooled_minus_within": g, "n_pairs": 10})}}
                                for c in CONDITIONS}}
            for ds, g in (("truthfulqa", 0.05), ("tydiqa_gp", -0.02))}
    cap = build_caption(fake, {"flags": ["MATCH within 0.01: ..."]})
    assert "sign of the gap differs by dataset" in cap, "caption must surface the mixed-sign finding"
    print("  [PASS] caption reports the mixed-sign gap when signs actually differ")

    miss = compare_session01(fake, os.path.join(HERE, "results", "_definitely_absent.json"))
    assert miss["status"] == "unavailable"
    print("  [PASS] compare_session01 degrades cleanly when the reference file is absent")

    out = AL.output_dir("_selftest")
    paths = plot(fake, os.path.join(out, "fig3_selftest"))
    assert all(os.path.exists(p) for p in paths)
    import shutil; shutil.rmtree(out, ignore_errors=True)
    print(f"  [PASS] plot rendered {len(paths)} files")
    print("\n[PASS] All self-test assertions passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=AL.DATASETS + ["all"])
    ap.add_argument("--model_folder", default="llama-3.1-8b-instruct")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-model-name", default="llama-3.1-8b")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--session01", default=os.path.join(HERE, "results", "session01_metrics.json"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return

    datasets = AL.DATASETS if a.dataset == "all" else [a.dataset]
    out = AL.output_dir(a.out_model_name)
    stem = os.path.join(out, f"fig3_difficulty_decomposition_{a.out_model_name}_{a.dataset}")
    AL.announce_outputs([f"{stem}.json", f"{stem}.pdf", f"{stem}.png"])

    data_dir = AL.get_data_dir(a.data_dir)
    results = {}
    t0 = time.time()
    for ds in datasets:
        print(f"\n[{ds}] decomposing ...", flush=True)
        results[ds] = run_dataset(ds, data_dir, a.model_folder, a.n_boot)

    s01_cmp = compare_session01(results, a.session01)
    for f in s01_cmp.get("flags", []):
        print(f"  [session-01] {f}", flush=True)

    payload = {"figure": "FIG3_difficulty_decomposition",
               "caption": build_caption(results, s01_cmp),
               "conditions_explained": {
                   "full": "baseline core_max features",
                   "centroid": "each beam replaced by its prompt's 10-beam mean (question-level only)",
                   "delta": "each beam minus its prompt centroid (per-response only)"},
               "decomposition_applied_to": "fold-local Tucker core (post-fit), shared basis across conditions",
               "session01_comparison": s01_cmp,
               "n_bootstrap": a.n_boot, "runtime_seconds": round(time.time() - t0, 1),
               "results": results,
               "provenance_block": AL.base_provenance(
                   {d: {"provenance": results[d]["provenance"],
                        "alignment_verified": results[d]["alignment_verified"],
                        "n_beams": results[d]["n_beams"], "n_prompts": results[d]["n_prompts"]}
                    for d in results})}
    AL.save_json(f"{stem}.json", payload)
    plot(results, stem)
    print(f"\nFIG 3 complete in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
