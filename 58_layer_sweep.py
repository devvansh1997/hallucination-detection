"""
58_layer_sweep.py -- single-layer AUROC at every depth, from 57's all-layer features.
=====================================================================================================
THE QUESTION. The pipeline fixes the layer window at {15..23}. Is that the right band, and does the
answer transfer across task shape? This produces the curve that answers it, or shows there is
nothing to answer.

METHOD. For each layer independently: fold-local feature compression to r_D, then logistic
regression, five-fold grouped on question id. One layer means the layer mode is degenerate, so
r_L = 1 and Equation (4) reduces to the feature projection alone -- the same fitting code the main
results use, via 43_eval_phase2.fold_pure_core_randomized, not a reimplementation. Bases are fitted
on training rows only, so no fold sees its own validation data during compression.

The readout is LR rather than RF. A forest on 4096 raw features would spend most of its capacity on
feature selection and would make 29 layers x 5 folds x 2 streams slow enough to matter, and the
question here is where signal lives, not how much a strong classifier can extract.

WHAT THE SHAPE WOULD MEAN.
  peaked inside {15..23}      the window is justified; report the curve and move on
  peaked elsewhere            the window is wrong, and that is worth knowing before the deadline
  flat                        depth does not matter, which is a cleaner finding than a curve and
                              retires the ablation entirely

TWO DATASETS, CHOSEN FOR CONTRAST. TyDiQA-GP is extractive -- the answer sits in the passage, so
producing it is closer to copying. TruthfulQA is closed-book and adversarial, so producing it means
retrieving from weights against a tempting falsehood. If those resolve at different depths the
curves peak in different places; if they peak together, the window transfers across task shape and
one figure justifies it for the whole paper. NQ-Open is deliberately excluded: at 96.9%
hallucinated only ~3% of rows are positives and its seed variance is +/-2.24, which would swamp the
effect being measured.

LAYER INDEXING. Index 0 is the embedding output, before any block. Index l>0 is block l's output.
The last index is the final block WITHOUT the final norm, so it is not the pinned pipeline's
final_norm slice -- see 57_extract_all_layers.py.

Usage:
  python 58_layer_sweep.py --self-test
  python 58_layer_sweep.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.abspath(os.path.join(HERE, "..", "data-alllayers"))
N_SPLITS = 5
SEED = 0
R_D = 64                 # matches CONDITION_SPECS' feature rank in 44_eval_phase3.py
WINDOW = list(range(15, 24))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_mods = {}


def mods():
    if not _mods:
        _mods["s02"] = _load("s02", "43_eval_phase2.py")     # fold_pure_core_randomized
        _mods["s01"] = _load("s01", "26_grouped_baseline.py")  # fit_eval
        _mods["al"] = _load("al", "45_analysis_loader.py")   # cluster_bootstrap_ci
    return _mods


def sweep(X, y, prompt_id, layer_index, folds, r_d=R_D, seed=SEED, n_boot=1000, label=""):
    """X is (N, L, F). One AUROC per layer, with a question-clustered bootstrap interval."""
    m = mods()
    out = []
    for li in range(X.shape[1]):
        t0 = time.time()
        Xi = np.ascontiguousarray(X[:, li:li + 1, :], dtype=np.float32)
        oof = np.full(len(y), np.nan)
        for fi, (tr, va) in enumerate(folds):
            core = m["s02"].fold_pure_core_randomized(Xi, tr, 1, r_d, seed + fi)
            oof[va] = m["s01"].fit_eval("LR", core[tr], y[tr], core[va], seed + fi)
        point = float(roc_auc_score(y, oof))
        ci = m["al"].cluster_bootstrap_ci(
            lambda idx: roc_auc_score(y[idx], oof[idx]) if len(np.unique(y[idx])) > 1 else None,
            prompt_id, n_boot=n_boot, seed=seed)
        out.append({"layer": int(layer_index[li]), "auroc": point,
                    "boot_mean": ci["mean"], "ci95": ci["ci95"]})
        print("    %s layer %2d: AUROC=%.4f  CI=[%.4f,%.4f]  (%.0fs)"
              % (label, layer_index[li], point, ci["ci95"][0], ci["ci95"][1], time.time() - t0),
              flush=True)
    return out


def describe(points, window=WINDOW):
    """Summarise a curve so the claim is checkable rather than eyeballed off a plot."""
    v = np.array([p["auroc"] for p in points], dtype=float)
    layers = np.array([p["layer"] for p in points], dtype=int)
    if not np.all(np.isfinite(v)) or v.size == 0:
        return {"shape": "undetermined"}
    rng = float(v.max() - v.min())
    best = int(layers[int(np.argmax(v))])
    inw = np.isin(layers, window)
    # "Flat" is judged against the bootstrap width, not an arbitrary constant: if the spread across
    # depths is smaller than the uncertainty at a single depth, there is no curve to interpret.
    mean_ci_w = float(np.mean([p["ci95"][1] - p["ci95"][0] for p in points]))
    return {
        "best_layer": best,
        "best_auroc": float(v.max()),
        "range": rng,
        "mean_ci_width": mean_ci_w,
        "shape": "flat" if rng < mean_ci_w else ("peaked_in_window" if best in window
                                                 else "peaked_outside_window"),
        "window_mean": float(v[inw].mean()) if inw.any() else None,
        "outside_mean": float(v[~inw].mean()) if (~inw).any() else None,
        "window_advantage": (float(v[inw].mean() - v[~inw].mean())
                             if inw.any() and (~inw).any() else None),
    }


def run(dataset, model_folder, in_dir, out_dir, streams, r_d, n_boot, limit_layers):
    path = os.path.join(in_dir, model_folder, "%s_alllayers.npz" % dataset)
    if not os.path.exists(path):
        raise SystemExit("%s not found -- run 57_extract_all_layers.py first." % path)
    z = np.load(path)
    y = np.asarray(z["label"], dtype=int)
    pid = np.asarray(z["prompt_id"])
    layers = np.asarray(z["layer_index"], dtype=int)
    print("  %s: %d answers, %d questions, %d layers, %.1f%% hallucinated"
          % (dataset, len(y), len(np.unique(pid)), len(layers), 100.0 * y.mean()), flush=True)

    built = {}
    if "core" in streams:
        built["core"] = z["core"]
    if "static" in streams:
        # Two-sided, concatenated on the feature axis -- the same construction as the pinned
        # static stream, so a layer's number here is comparable to the main results.
        built["static"] = np.concatenate([z["q95"], z["q05"]], axis=2)

    if limit_layers:
        keep = np.linspace(0, len(layers) - 1, limit_layers).astype(int)
        layers = layers[keep]
        built = {k: v[:, keep, :] for k, v in built.items()}
        print("  [--limit-layers] sweeping %d of the available depths: %s"
              % (len(layers), list(layers)), flush=True)

    folds = list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(len(y)), y, groups=pid))
    res = {"dataset": dataset, "model_folder": model_folder, "n_beams": int(len(y)),
           "n_questions": int(len(np.unique(pid))),
           "hallucination_rate_pct": round(100.0 * float(y.mean()), 3),
           "r_d": r_d, "n_splits": N_SPLITS, "window": WINDOW, "curves": {}}
    for name, X in built.items():
        print("\n  -- stream: %s  (F=%d)" % (name, X.shape[2]), flush=True)
        pts = sweep(X, y, pid, layers, folds, r_d=r_d, n_boot=n_boot, label=name)
        res["curves"][name] = {"points": pts, "summary": describe(pts)}
        s = res["curves"][name]["summary"]
        print("    -> %s | best layer %d (%.4f) | window mean %.4f vs outside %.4f"
              % (s["shape"], s["best_layer"], s["best_auroc"],
                 s["window_mean"] or float("nan"), s["outside_mean"] or float("nan")))

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "layer_sweep_%s_%s.json" % (model_folder, dataset))
    with open(p, "w") as f:
        json.dump(res, f, indent=2)
    print("\nWrote: %s" % p)
    return res


def self_test():
    print("=" * 74)
    print("  SELF-TEST: 58_layer_sweep")
    print("=" * 74)

    # describe(): a planted peak inside the window must be found, and reported as such.
    pts = [{"layer": l, "auroc": 0.60 + (0.15 if l in (18, 19) else 0.0),
            "ci95": [0.55, 0.65]} for l in range(29)]
    d = describe(pts)
    assert d["best_layer"] in (18, 19) and d["shape"] == "peaked_in_window", d
    assert d["window_advantage"] > 0
    print("  [PASS] describe: peak at layer %d inside the window, advantage %+.4f"
          % (d["best_layer"], d["window_advantage"]))

    # A peak OUTSIDE the window must not be quietly reported as vindication.
    pts2 = [{"layer": l, "auroc": 0.60 + (0.15 if l == 4 else 0.0), "ci95": [0.55, 0.65]}
            for l in range(29)]
    assert describe(pts2)["shape"] == "peaked_outside_window"
    print("  [PASS] describe: a peak at layer 4 is reported as OUTSIDE the window")

    # Flatness is judged against the bootstrap width, not a magic constant. Here the spread across
    # depths (0.01) is smaller than the interval at any single depth (0.10), so there is no curve.
    pts3 = [{"layer": l, "auroc": 0.70 + 0.0003 * l, "ci95": [0.65, 0.75]} for l in range(29)]
    d3 = describe(pts3)
    assert d3["shape"] == "flat", d3
    print("  [PASS] describe: spread %.4f < mean CI width %.4f -> flat, not a spurious peak"
          % (d3["range"], d3["mean_ci_width"]))

    # The same spread with TIGHT intervals is a real curve, not flat. Without this the flatness
    # rule could declare everything flat and the ablation would silently answer itself.
    pts4 = [{"layer": l, "auroc": 0.70 + 0.0003 * l, "ci95": [0.699, 0.701]} for l in range(29)]
    assert describe(pts4)["shape"] != "flat"
    print("  [PASS] describe: the same spread with tight CIs is NOT flat")

    # Fold construction must never split a question across folds -- the whole protocol argument
    # rests on this, so it is asserted rather than trusted to GroupKFold's docstring.
    pid = np.repeat(np.arange(40), 10)
    y = np.random.default_rng(0).integers(0, 2, size=400)
    folds = list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(400), y, groups=pid))
    for tr, va in folds:
        assert not (set(pid[tr]) & set(pid[va])), "a question straddled a fold"
    assert sum(len(va) for _, va in folds) == 400
    print("  [PASS] GroupKFold: no question straddles a fold, every row evaluated once")

    print("\n[PASS] All self-test assertions passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tydiqa_gp",
                    choices=["truthfulqa", "triviaqa", "nq_open", "tydiqa_gp"])
    ap.add_argument("--model_folder", default="qwen-2.5-7b-instruct")
    ap.add_argument("--in-dir", default=DEFAULT_IN)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "layer_sweep"))
    ap.add_argument("--streams", default="core,static")
    ap.add_argument("--r-d", type=int, default=R_D)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--limit-layers", type=int, default=None,
                    help="sweep an evenly spaced subset first, to time the full run")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return

    print("=" * 78)
    print("  LAYER SWEEP -- %s / %s" % (a.model_folder, a.dataset))
    print("  single-layer probe at every depth; window under test is %d..%d"
          % (WINDOW[0], WINDOW[-1]))
    print("=" * 78, flush=True)
    run(a.dataset, a.model_folder, a.in_dir, a.out_dir,
        [s.strip() for s in a.streams.split(",") if s.strip()], a.r_d, a.n_boot, a.limit_layers)


if __name__ == "__main__":
    main()
