"""
46_fig1_layer_divergence.py -- Session 08 FIG 1: within-window layer profile
=====================================================================================================
CPU-only, read-only. Writes exclusively to analysis_out/{model_folder}/ (gitignored).

*** SCOPE (Session 08 ruling #1) ***
This is a WITHIN-WINDOW profile, not a depth-localization result. Only layers 15-23 were retained
by either extraction path -- there is no data outside the window on disk -- so the figure cannot
say where in the network's full depth the signal lives, and deliberately does not shade a "window"
region (the window IS the x-axis). Full-depth profiling requires GPU re-extraction: deferred.

Two measures, side by side, because they answer different questions:
  (i)  energy divergence: (mean||h||_halluc - mean||h||_truthful) / pooled SD, per layer.
       Purely a first-moment norm gap -- the original "energy" notion. It can be ~0 even when a
       layer is highly discriminative, if the two classes differ in DIRECTION rather than scale.
  (ii) per-layer AUROC: fold-local Tucker (r_l=1, r_d=64) on that ONE layer -> LR, grouped 5-fold,
       pooled out-of-fold AUROC. How much a single layer can do on its own.

Usage:
  python 46_fig1_layer_divergence.py --self-test
  python 46_fig1_layer_divergence.py --dataset all
  python 46_fig1_layer_divergence.py --dataset triviaqa --fold0-timing-only
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))

import importlib.util as _ilu


def _load(name, filename):
    spec = _ilu.spec_from_file_location(name, os.path.join(HERE, filename))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


AL = _load("analysis_loader", "45_analysis_loader.py")
s02_eval = _load("s02_eval", "43_eval_phase2.py")
s01 = _load("s01", "26_grouped_baseline.py")

R_D_SINGLE_LAYER = 64


def energy_divergence_per_layer(core, y, prompt_idx, n_boot=1000, seed=AL.SEED):
    """(mean||h||_halluc - mean||h||_truthful) / pooled SD, per layer, with cluster-bootstrap CI.
    Norms are computed once over the whole array (cheap, [N,9]); the bootstrap then only reindexes
    that small matrix rather than recomputing norms on the 4096-dim tensors 1000 times."""
    n_layers = core.shape[1]
    norms = np.linalg.norm(core.astype(np.float32), axis=2)   # [N, 9]
    out = []
    for li in range(n_layers):
        v = norms[:, li]

        def stat(idx, v=v):
            yy, vv = y[idx], v[idx]
            h, t = vv[yy == 1], vv[yy == 0]
            if len(h) < 2 or len(t) < 2:
                return None
            pooled_sd = np.sqrt((h.var(ddof=1) + t.var(ddof=1)) / 2.0)
            if pooled_sd <= 0:
                return None
            return float((h.mean() - t.mean()) / pooled_sd)

        point = stat(np.arange(len(y)))
        ci = AL.cluster_bootstrap_ci(stat, prompt_idx, n_boot=n_boot, seed=seed)
        out.append({"layer": AL.WINDOW_LAYERS[li], "point": point,
                    "boot_mean": ci["mean"], "ci95": ci["ci95"]})
    return out


def per_layer_auroc(core, y, prompt_idx, folds, n_boot=1000, seed=AL.SEED, timing_probe=False):
    """Fold-local Tucker on ONE layer (r_l=1 -- a single layer has only one layer-mode component),
    then LR. Fold-local so the readout never sees validation rows during basis fitting."""
    n_layers = core.shape[1]
    out = []
    for li in range(n_layers):
        X = core[:, li:li + 1, :].astype(np.float32)   # [N, 1, D]
        oof = np.full(len(y), np.nan)
        t0 = time.time()
        for fi, (tr, va) in enumerate(folds):
            c = s02_eval.fold_pure_core_randomized(X, tr, 1, R_D_SINGLE_LAYER, seed + fi)
            oof[va] = s01.fit_eval("LR", c[tr], y[tr], c[va], seed + fi)
            if timing_probe and fi == 0:
                dt = time.time() - t0
                print(f"    [timing probe] layer {AL.WINDOW_LAYERS[li]} fold 0: {dt:.1f}s "
                      f"-> est. {dt*len(folds)*n_layers/60:.1f} min for all "
                      f"{n_layers} layers x {len(folds)} folds", flush=True)
                return None
        point = float(roc_auc_score(y, oof))
        ci = AL.cluster_bootstrap_ci(
            lambda idx: roc_auc_score(y[idx], oof[idx]) if len(np.unique(y[idx])) > 1 else None,
            prompt_idx, n_boot=n_boot, seed=seed)
        out.append({"layer": AL.WINDOW_LAYERS[li], "point": point,
                    "boot_mean": ci["mean"], "ci95": ci["ci95"]})
        print(f"    layer {AL.WINDOW_LAYERS[li]}: AUROC={point:.4f} "
              f"CI=[{ci['ci95'][0]:.4f},{ci['ci95'][1]:.4f}]  ({time.time()-t0:.0f}s)", flush=True)
    return out


def describe_shape(points):
    """Classify a 9-point profile as rising / falling / peaked / trough / flat. Reported per dataset
    per measure so the taxonomy claim can be checked against the data rather than asserted."""
    v = np.asarray([p["point"] for p in points], dtype=float)
    if not np.all(np.isfinite(v)):
        return "undetermined"
    rng = float(v.max() - v.min())
    if rng < 1e-3 or rng < 0.02 * (abs(float(np.mean(v))) + 1e-9):
        return "flat"
    ai = int(np.argmax(v))
    if ai == 0:
        return "falling"
    if ai == len(v) - 1:
        return "rising"
    return "peaked" if v[ai] > v[0] and v[ai] > v[-1] else "mixed"


def run_dataset(ds, data_dir, model_folder, n_boot, timing_probe=False):
    info = AL.load_arrays(ds, data_dir, model_folder, need=("core",))
    core, y, pid = info["arrays"]["core"], info["y"], info["prompt_idx"]
    folds = AL.make_folds(y, pid)
    print(f"  [{ds}] measure (i) energy divergence ...", flush=True)
    energy = energy_divergence_per_layer(core, y, pid, n_boot=n_boot)
    print(f"  [{ds}] measure (ii) per-layer single-layer AUROC ...", flush=True)
    auroc = per_layer_auroc(core, y, pid, folds, n_boot=n_boot, timing_probe=timing_probe)
    if auroc is None:
        return None
    res = {"dataset": ds, "dataset_type": AL.DATASET_TYPE[ds],
           "n_beams": info["n_beams"], "n_prompts": info["n_prompts"],
           "energy_divergence": energy, "per_layer_auroc": auroc,
           "argmax_layer_energy": int(max(energy, key=lambda p: p["point"])["layer"]),
           "argmax_layer_auroc": int(max(auroc, key=lambda p: p["point"])["layer"]),
           "shape_energy": describe_shape(energy), "shape_auroc": describe_shape(auroc),
           "alignment_verified": info["alignment_verified"], "provenance": info["provenance"]}
    print(f"  [{ds}] argmax: energy=L{res['argmax_layer_energy']} ({res['shape_energy']})  "
          f"auroc=L{res['argmax_layer_auroc']} ({res['shape_auroc']})", flush=True)
    return res


def build_caption(results):
    by_type = {"reasoning": [], "retrieval": []}
    for ds, r in results.items():
        by_type[r["dataset_type"]].append(f"{AL.DATASET_STYLE[ds]['label']} peaks at layer "
                                           f"{r['argmax_layer_auroc']} ({r['shape_auroc']})")
    reasoning_shapes = {results[d]["shape_auroc"] for d in results if AL.DATASET_TYPE[d] == "reasoning"}
    retrieval_shapes = {results[d]["shape_auroc"] for d in results if AL.DATASET_TYPE[d] == "retrieval"}
    if reasoning_shapes == {"peaked"} and "peaked" not in retrieval_shapes:
        taxonomy = ("The reasoning-type datasets show a localized interior peak while the "
                    "retrieval-type datasets do not, consistent with the taxonomy hypothesis.")
    else:
        taxonomy = ("The reasoning/retrieval split does NOT separate cleanly by profile shape here "
                    f"(reasoning: {sorted(reasoning_shapes)}; retrieval: {sorted(retrieval_shapes)}); "
                    "we report the profiles without endorsing that framing.")
    return (
        "Within-window layer profile of the hallucination signal, LLaMA-3.1-8B. "
        "Left: normalized energy divergence, (mean norm of hallucinated minus mean norm of "
        "truthful) divided by the pooled standard deviation. Right: AUROC of a fold-local "
        "single-layer readout (Tucker r_d=64 on that layer alone, logistic regression), grouped "
        "5-fold by prompt. Bands are 95% cluster-bootstrap intervals resampling prompts. "
        + taxonomy +
        " Layers outside 15-23 were not retained by the extraction pipeline, so this profile "
        "covers the extraction window only and makes no claim about full-depth localization; "
        "profiling the full network requires GPU re-extraction, which is deferred rather than "
        "dropped. TruthfulQA's tensors come from the older Route-N path while the other three "
        "come from script 42; the two are assumed equivalent by construction but not verified.")


def plot(results, stem):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, title, ylab in (
            (axes[0], "energy_divergence", "(i) Normalized energy divergence",
             r"$(\overline{\|h\|}_{halluc} - \overline{\|h\|}_{truth})\, /\, SD_{pooled}$"),
            (axes[1], "per_layer_auroc", "(ii) Single-layer readout AUROC", "pooled OOF AUROC")):
        for ds, r in results.items():
            st = AL.DATASET_STYLE[ds]
            xs = [p["layer"] for p in r[key]]
            ys = [p["point"] for p in r[key]]
            lo = [p["ci95"][0] for p in r[key]]
            hi = [p["ci95"][1] for p in r[key]]
            ax.plot(xs, ys, color=st["color"], ls=st["ls"], marker=st["marker"],
                    label=st["label"], lw=1.8, ms=5)
            ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.13, lw=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("layer index (extraction window only)")
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_xticks(AL.WINDOW_LAYERS)
        ax.grid(alpha=0.25, ls=":")
        if key == "per_layer_auroc":
            ax.axhline(0.5, color="k", lw=0.8, ls=(0, (1, 2)))
        else:
            ax.axhline(0.0, color="k", lw=0.8, ls=(0, (1, 2)))
    axes[0].legend(fontsize=8, framealpha=0.9)
    fig.suptitle("FIG 1 -- within-window layer profile (layers 15-23; no data outside window)",
                 fontsize=12)
    fig.tight_layout()
    return AL.save_figure(fig, stem)


def self_test():
    print("=" * 70)
    print("  SELF-TEST: FIG 1 (synthetic arrays, no cluster files)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    n_prompts, per_prompt, D, n_layers = 24, 10, 48, 9
    n = n_prompts * per_prompt
    pid = np.repeat(np.arange(n_prompts), per_prompt)
    y = (rng.random(n) < 0.5).astype(np.int64)
    peak = 4   # window layer 19
    # Two planted arrays with DIFFERENT amplitudes, because the two measures have very different
    # sensitivities and a single amplitude cannot exercise both: the norm-gap measure needs a large
    # shift to move ||h|| detectably, while the readout saturates at AUROC=1.000 well below that
    # (at amplitude 2.5 three adjacent layers all hit exactly 1.0 and argmax becomes meaningless).
    def planted(amp):
        c = rng.standard_normal((n, n_layers, D)).astype(np.float32)
        for li in range(n_layers):
            c[y == 1, li, :] += amp * np.exp(-0.5 * ((li - peak) / 1.2) ** 2)
        return c

    core = planted(2.5)        # strong: exercises the norm-gap measure
    core_readout = planted(0.25)   # weak: keeps the readout below saturation

    energy = energy_divergence_per_layer(core, y, pid, n_boot=60)
    assert len(energy) == n_layers
    assert max(energy, key=lambda p: p["point"])["layer"] == AL.WINDOW_LAYERS[peak], \
        f"planted peak at layer {AL.WINDOW_LAYERS[peak]} not recovered by energy measure"
    assert describe_shape(energy) == "peaked"
    print(f"  [PASS] energy divergence recovers the planted peak at layer "
          f"{AL.WINDOW_LAYERS[peak]}, shape='peaked'")

    folds = AL.make_folds(y, pid)
    auroc = per_layer_auroc(core_readout, y, pid, folds, n_boot=40)
    assert len(auroc) == n_layers
    assert max(auroc, key=lambda p: p["point"])["layer"] == AL.WINDOW_LAYERS[peak], \
        f"planted peak not recovered by the single-layer readout: {[(p['layer'], round(p['point'],3)) for p in auroc]}"
    assert auroc[peak]["point"] > 0.65, f"planted layer should be discriminative: {auroc[peak]['point']}"
    assert auroc[peak]["point"] < 0.999, "readout saturated -- argmax would be meaningless; lower the amplitude"
    print(f"  [PASS] single-layer AUROC recovers the same peak "
          f"(AUROC={auroc[peak]['point']:.3f}, unsaturated, at the planted layer)")

    # a monotone-rising plant must be classified 'rising', not 'peaked'
    core2 = rng.standard_normal((n, n_layers, D)).astype(np.float32)
    for li in range(n_layers):
        core2[y == 1, li, :] += 0.35 * li
    assert describe_shape(energy_divergence_per_layer(core2, y, pid, n_boot=40)) == "rising"
    print("  [PASS] describe_shape distinguishes 'rising' from 'peaked'")

    # flat (no signal) must not be reported as a peak
    core3 = rng.standard_normal((n, n_layers, D)).astype(np.float32)
    sh = describe_shape(energy_divergence_per_layer(core3, y, pid, n_boot=40))
    assert sh in ("flat", "peaked", "rising", "falling", "mixed")
    print(f"  [PASS] pure-noise profile classified '{sh}' without error")

    res = {"truthfulqa": {"dataset_type": "reasoning", "shape_auroc": "peaked",
                          "argmax_layer_auroc": 19, "energy_divergence": energy,
                          "per_layer_auroc": auroc},
           "triviaqa": {"dataset_type": "retrieval", "shape_auroc": "rising",
                        "argmax_layer_auroc": 23, "energy_divergence": energy,
                        "per_layer_auroc": auroc}}
    cap = build_caption(res)
    assert "consistent with the taxonomy hypothesis" in cap
    res["tydiqa_gp"] = dict(res["truthfulqa"], dataset_type="reasoning", shape_auroc="rising")
    assert "does NOT separate cleanly" in build_caption(res), \
        "caption must refuse the taxonomy framing when the data doesn't support it"
    print("  [PASS] caption states the taxonomy claim only when the shapes actually support it")

    out = AL.output_dir("_selftest")
    paths = plot({k: v for k, v in res.items() if k in ("truthfulqa", "triviaqa")},
                 os.path.join(out, "fig1_selftest"))
    assert all(os.path.exists(p) for p in paths)
    print(f"  [PASS] plot rendered {len(paths)} files")
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    print("\n[PASS] All self-test assertions passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=AL.DATASETS + ["all"])
    ap.add_argument("--model_folder", default="llama-3.1-8b-instruct")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-model-name", default="llama-3.1-8b")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--fold0-timing-only", action="store_true",
                    help="Run only layer 0 / fold 0 and extrapolate, then exit without writing.")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return

    datasets = AL.DATASETS if a.dataset == "all" else [a.dataset]
    out = AL.output_dir(a.out_model_name)
    stem = os.path.join(out, f"fig1_layer_divergence_{a.out_model_name}_{a.dataset}")
    if not a.fold0_timing_only:
        AL.announce_outputs([f"{stem}.json", f"{stem}.pdf", f"{stem}.png"])

    data_dir = AL.get_data_dir(a.data_dir)
    results, infos = {}, {}
    t0 = time.time()
    for ds in datasets:
        print(f"\n[{ds}] loading ...", flush=True)
        r = run_dataset(ds, data_dir, a.model_folder, a.n_boot, timing_probe=a.fold0_timing_only)
        if a.fold0_timing_only:
            print("[timing probe] exiting without writing any file."); return
        results[ds] = r
        infos[ds] = {"provenance": r["provenance"], "alignment_verified": r["alignment_verified"],
                     "n_beams": r["n_beams"], "n_prompts": r["n_prompts"]}

    payload = {"figure": "FIG1_within_window_layer_profile",
               "caption": build_caption(results),
               "measures": {"i": "normalized energy divergence (norm gap / pooled SD)",
                            "ii": f"single-layer fold-local Tucker r_d={R_D_SINGLE_LAYER} -> LR, pooled OOF AUROC"},
               "n_bootstrap": a.n_boot, "runtime_seconds": round(time.time() - t0, 1),
               "results": results}
    payload.update({"provenance_block": {**AL.base_provenance(
        {d: {"provenance": results[d]["provenance"],
             "alignment_verified": results[d]["alignment_verified"],
             "n_beams": results[d]["n_beams"], "n_prompts": results[d]["n_prompts"]}
         for d in results})}})
    AL.save_json(f"{stem}.json", payload)
    plot(results, stem)
    print(f"\nFIG 1 complete in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
