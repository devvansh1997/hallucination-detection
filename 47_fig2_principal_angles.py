"""
47_fig2_principal_angles.py -- Session 08 FIG 2: static vs velocity subspace geometry
=====================================================================================================
CPU-only, read-only. Writes exclusively to analysis_out/{model_folder}/ (gitignored).

Question: WHY does combining the static and velocity streams help? If their channel-mode subspaces
were near-identical, concatenating them would be redundant; if they are complementary, the union
spans more of the representation than either alone.

Per Session 08 ruling #2, subspaces are built exactly as the Phase 3 pipeline builds the streams:
q95 and q05 concatenated along the CHANNEL axis -> 8192 channels, fold-local Tucker factor
8192 x 64. Comparing 4096-channel halves instead would measure an object the pipeline never uses.

THREE curves, because "static and velocity barely overlap" is uninterpretable alone:
  - static vs velocity      : the quantity of interest
  - q95 vs q05 WITHIN a stream (4096 x 64 each, compared in their own R^4096): distinguishes
    "these two STREAMS are complementary" from "any two quantile halves are complementary"
  - random null: two independent random orthonormal 64-dim subspaces of R^8192, seed 0. In high
    dimension random subspaces are nearly orthogonal by default, so without this line a low
    overlap looks like a finding when it may just be dimensionality.

Memory: streams are fit ONE AT A TIME and only the 8192x64 factor is retained, so peak RSS is one
stream (~30GB at TriviaQA scale) rather than both (~56GB). --max-beams subsamples beams for the
subspace estimate only; any use is recorded in the JSON and must be reported as a Deviation.

Usage:
  python 47_fig2_principal_angles.py --self-test
  python 47_fig2_principal_angles.py --dataset all
  python 47_fig2_principal_angles.py --dataset triviaqa --fold0-timing-only
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util as _ilu

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import resource
except ImportError:
    resource = None


def _peak_rss_mb():
    if resource is None:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _load(name, filename):
    spec = _ilu.spec_from_file_location(name, os.path.join(HERE, filename))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


AL = _load("analysis_loader", "45_analysis_loader.py")
s02_eval = _load("s02_eval", "43_eval_phase2.py")

R_L_STATIC, R_L_VELOCITY, R_D = 5, 4, 64
COS_THRESHOLD = 0.5


def channel_factor(X_raw, tr_idx, r_l, r_d, seed):
    """Fold-local channel-mode Tucker factor U_D (D x r_d). Reuses the pipeline's own
    scale-then-randomized-SVD path so the subspace is the SAME object the Phase 3 conditions fit,
    not a re-derivation that might differ subtly."""
    X_scaled = s02_eval.robust_scale_3d(X_raw, tr_idx)
    _, U_D = s02_eval.compute_ul_ud_randomized(X_scaled[tr_idx], r_l, r_d, seed)
    return np.asarray(U_D, dtype=np.float64)


def _subsample(tr, max_beams, seed):
    if max_beams is None or len(tr) <= max_beams:
        return tr, False
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(tr, size=max_beams, replace=False)), True


def run_dataset(ds, data_dir, model_folder, max_beams=None, timing_probe=False):
    """Returns per-fold factors + angle curves. Loads one stream at a time to halve peak memory."""
    t_all = time.time()
    factors = {"static": [], "velocity": [], "static_q95": [], "static_q05": []}
    y = pid = None
    info_meta = None
    subsampled = False

    for stream, r_l in (("static", R_L_STATIC), ("velocity", R_L_VELOCITY)):
        info = AL.load_arrays(ds, data_dir, model_folder,
                              need=(f"{stream}_q95", f"{stream}_q05"), verbose=(stream == "static"))
        if y is None:
            y, pid = info["y"], info["prompt_idx"]
            folds = AL.make_folds(y, pid)
            info_meta = {k: info[k] for k in ("provenance", "alignment_verified", "n_beams", "n_prompts")}
        X = AL.concat_stream(info, stream)          # [N, L, 8192] float32
        # Fit the concatenated stream first and RELEASE it before touching the quantile halves.
        # Holding X (29GB at TriviaQA scale) alongside both halves (29GB) plus robust_scale_3d's
        # copy would put peak RSS near 140GB; sequencing them caps it at roughly one array plus
        # its scaled copy.
        arrays_ref = info["arrays"] if stream == "static" else None
        del info

        for fi, (tr, va) in enumerate(folds):
            tr_s, was_sub = _subsample(tr, max_beams, AL.SEED + fi)
            subsampled = subsampled or was_sub
            t0 = time.time()
            factors[stream].append(channel_factor(X, tr_s, r_l, R_D, AL.SEED + fi))
            dt = time.time() - t0
            rss = _peak_rss_mb()
            print(f"    [{ds}/{stream}] fold {fi+1}/{len(folds)} factor in {dt:.1f}s"
                  + (f"  peakRSS={rss:.0f}MB" if rss else ""), flush=True)
            if timing_probe and fi == 0:
                n_units = len(folds) * 2
                print(f"    [timing probe] {ds}: fold-0 {stream} took {dt:.1f}s -> est. "
                      f"~{dt*n_units/60:.1f} min for both streams x {len(folds)} folds "
                      f"(plus the within-stream reference, roughly +50%)", flush=True)
                return None
        del X

        if arrays_ref is not None:
            # within-stream reference, one half at a time in its own R^4096
            for half in ("static_q95", "static_q05"):
                H = arrays_ref[half].astype(np.float32)
                for fi, (tr, va) in enumerate(folds):
                    tr_s, _ = _subsample(tr, max_beams, AL.SEED + fi)
                    factors[half].append(channel_factor(H, tr_s, r_l, R_D, AL.SEED + fi))
                del H
            del arrays_ref

    n_ch = factors["static"][0].shape[0]
    sv = [AL.principal_angles(s, v) for s, v in zip(factors["static"], factors["velocity"])]
    qq = [AL.principal_angles(a, b) for a, b in zip(factors["static_q95"], factors["static_q05"])]
    null = AL.random_subspace_pair_angles(n_ch, R_D, seed=AL.SEED)

    def summarize(curves, name):
        arr = np.vstack(curves)
        return {"name": name, "mean": arr.mean(axis=0).tolist(),
                "min": arr.min(axis=0).tolist(), "max": arr.max(axis=0).tolist(),
                "shared_gt_0.5_mean_over_folds": float(np.mean([AL.shared_direction_count(c, COS_THRESHOLD) for c in curves])),
                "shared_gt_0.5_per_fold": [AL.shared_direction_count(c, COS_THRESHOLD) for c in curves]}

    res = {"dataset": ds, "channel_dim": int(n_ch), "r_d": R_D,
           "static_vs_velocity": summarize(sv, "static vs velocity"),
           "q95_vs_q05_within_static": summarize(qq, "q95 vs q05 (within static stream)"),
           "random_null": {"name": f"random 64-dim subspaces in R^{n_ch}",
                           "mean": null.tolist(),
                           "shared_gt_0.5": AL.shared_direction_count(null, COS_THRESHOLD)},
           "subsampled_for_subspace_estimate": bool(subsampled),
           "max_beams": max_beams, "runtime_seconds": round(time.time() - t_all, 1),
           **info_meta}
    print(f"  [{ds}] shared directions (cos>{COS_THRESHOLD}): "
          f"static-vs-velocity={res['static_vs_velocity']['shared_gt_0.5_mean_over_folds']:.1f}, "
          f"q95-vs-q05={res['q95_vs_q05_within_static']['shared_gt_0.5_mean_over_folds']:.1f}, "
          f"random-null={res['random_null']['shared_gt_0.5']}", flush=True)
    return res, factors["static"]


def stability_matrix(static_factors_by_ds):
    """Plot B. Diagonal = within-dataset fold-1 vs fold-2 static subspace (how reproducible is the
    basis across resamples of the same data). Off-diagonal = dataset-A fold-0 vs dataset-B fold-0
    (how much of the basis is shared across datasets). Value = mean cos over the first 10
    components, which is where any real shared structure concentrates."""
    names = list(static_factors_by_ds)
    M = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                fa = static_factors_by_ds[a]
                if len(fa) >= 2:
                    M[i, j] = float(AL.principal_angles(fa[0], fa[1])[:10].mean())
            else:
                M[i, j] = float(AL.principal_angles(static_factors_by_ds[a][0],
                                                     static_factors_by_ds[b][0])[:10].mean())
    return names, M


def build_caption(results, stability_names, stability_M):
    ex = next(iter(results.values()))
    nch, rd = ex["channel_dim"], ex["r_d"]
    sv = np.mean([r["static_vs_velocity"]["shared_gt_0.5_mean_over_folds"] for r in results.values()])
    qq = np.mean([r["q95_vs_q05_within_static"]["shared_gt_0.5_mean_over_folds"] for r in results.values()])
    nl = ex["random_null"]["shared_gt_0.5"]
    diag = np.nanmean(np.diag(stability_M))
    off = np.nanmean(stability_M[~np.eye(len(stability_names), dtype=bool)])
    if sv > nl + 1 and sv < qq:
        verdict = (f"The static and velocity subspaces share more directions than chance ({sv:.1f} "
                   f"vs {nl} for the random null) but markedly fewer than two quantile halves of "
                   f"the same stream ({qq:.1f}), i.e. they are genuinely complementary rather than "
                   f"redundant -- which is what makes concatenating them informative.")
    elif sv <= nl + 1:
        verdict = (f"The static and velocity subspaces are no more aligned than two random 64-dim "
                   f"subspaces ({sv:.1f} vs {nl}), i.e. essentially orthogonal: the two streams "
                   f"carry almost entirely distinct channel directions.")
    else:
        verdict = (f"The static and velocity subspaces share {sv:.1f} directions on average, "
                   f"comparable to the {qq:.1f} shared by two quantile halves of the same stream, "
                   f"so the two streams are more redundant than the combination results suggest.")
    return (
        f"Principal angles between fold-local Tucker channel-mode subspaces ({nch} channels, "
        f"rank {rd}), LLaMA-3.1-8B. Left: cos(theta) against component index for static vs "
        f"velocity (one line per dataset, band = min/max across the 5 folds), with two references "
        f"-- two quantile halves of the same stream, and two independent random 64-dim subspaces "
        f"of R^{nch}. The random null is essential context: high-dimensional random subspaces are "
        f"nearly orthogonal by construction. " + verdict +
        f" Right: stability of the static subspace, mean cos(theta) over the first 10 components; "
        f"diagonal compares fold 1 against fold 2 within a dataset (mean {diag:.2f}), off-diagonal "
        f"compares datasets (mean {off:.2f}). Streams are concatenated q95|q05 exactly as the "
        f"Phase 3 pipeline builds them. TruthfulQA's tensors originate from the older Route-N path "
        f"while the other three come from script 42; assumed equivalent, not verified.")


def plot(results, stability_names, stability_M, stem):
    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    # rank comes from the data, not the module constant -- the self-test exercises a smaller rank
    rd = len(next(iter(results.values()))["static_vs_velocity"]["mean"])
    idx = np.arange(1, rd + 1)
    for ds, r in results.items():
        st = AL.DATASET_STYLE[ds]
        m = np.asarray(r["static_vs_velocity"]["mean"])
        ax.plot(idx, m, color=st["color"], ls=st["ls"], marker=st["marker"], ms=3,
                markevery=4, lw=1.7, label=f"{st['label']} (static vs velocity)")
        ax.fill_between(idx, r["static_vs_velocity"]["min"], r["static_vs_velocity"]["max"],
                        color=st["color"], alpha=0.12, lw=0)
    ref = next(iter(results.values()))
    qs = AL.REFERENCE_STYLE["q95_vs_q05"]
    ax.plot(idx, np.mean([r["q95_vs_q05_within_static"]["mean"] for r in results.values()], axis=0),
            color=qs["color"], ls=qs["ls"], lw=2.0, label=qs["label"] + " [mean of datasets]")
    ns = AL.REFERENCE_STYLE["random_null"]
    ax.plot(idx, ref["random_null"]["mean"], color=ns["color"], ls=ns["ls"], lw=2.0,
            label=f"random {rd}-dim null in R^{ref['channel_dim']}")
    ax.axhline(COS_THRESHOLD, color="k", lw=0.8, ls=(0, (1, 3)))
    ax.text(rd * 0.62, COS_THRESHOLD + 0.02, f"shared-direction threshold {COS_THRESHOLD}", fontsize=7)
    ax.set_xlabel("principal component index $i$")
    ax.set_ylabel(r"$\cos(\theta_i)$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(A) static vs velocity subspace overlap", fontsize=11)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=7, framealpha=0.9, loc="upper right")

    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(stability_M, vmin=0, vmax=1, cmap="viridis")
    labels = [AL.DATASET_STYLE[n]["label"] for n in stability_names]
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isfinite(stability_M[i, j]):
                ax2.text(j, i, f"{stability_M[i,j]:.2f}", ha="center", va="center", fontsize=8,
                         color="w" if stability_M[i, j] < 0.6 else "k")
    ax2.set_title("(B) static-subspace stability\n(diag: fold1 vs fold2; off: dataset vs dataset)",
                  fontsize=10)
    fig.colorbar(im, ax=ax2, fraction=0.046, label=r"mean $\cos(\theta)$, first 10 comps")
    fig.suptitle("FIG 2 -- why combining static and velocity helps: subspace geometry", fontsize=12)
    fig.tight_layout()
    return AL.save_figure(fig, stem)


def self_test():
    print("=" * 70)
    print("  SELF-TEST: FIG 2 (synthetic, no cluster files)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    n_prompts, per_prompt, D, r_d = 16, 10, 96, 8
    n = n_prompts * per_prompt
    pid = np.repeat(np.arange(n_prompts), per_prompt)
    y = (rng.random(n) < 0.5).astype(np.int64)
    folds = AL.make_folds(y, pid)

    X = rng.standard_normal((n, 5, D)).astype(np.float32)
    U = channel_factor(X, folds[0][0], 3, r_d, 0)
    assert U.shape == (D, r_d)
    assert np.allclose(U.T @ U, np.eye(r_d), atol=1e-6), "channel factor must be orthonormal"
    print(f"  [PASS] channel_factor: shape {U.shape}, orthonormal")

    U2 = channel_factor(X, folds[0][0], 3, r_d, 0)
    assert np.allclose(AL.principal_angles(U, U2), 1.0, atol=1e-8), "same seed+data must be identical"
    print("  [PASS] channel_factor deterministic under a fixed seed (cos=1 with itself)")

    # a stream sharing a planted low-rank block with another must show MORE overlap than the null
    B = np.linalg.qr(rng.standard_normal((D, D)))[0]
    shared, priv_a, priv_b = B[:, :4], B[:, 4:12], B[:, 12:20]
    Xa = (rng.standard_normal((n, 5, 12)) @ np.concatenate([shared, priv_a], axis=1).T).astype(np.float32)
    Xb = (rng.standard_normal((n, 5, 12)) @ np.concatenate([shared, priv_b], axis=1).T).astype(np.float32)
    Ua = channel_factor(Xa, folds[0][0], 3, r_d, 0)
    Ub = channel_factor(Xb, folds[0][0], 3, r_d, 0)
    cos_ab = AL.principal_angles(Ua, Ub)
    null = AL.random_subspace_pair_angles(D, r_d, seed=0)
    n_shared = AL.shared_direction_count(cos_ab, COS_THRESHOLD)
    assert n_shared >= 4, f"4 planted shared directions expected, got {n_shared}: {np.round(cos_ab,3)}"
    assert n_shared > AL.shared_direction_count(null, COS_THRESHOLD), "must exceed the random null"
    print(f"  [PASS] planted overlap detected: {n_shared} shared directions vs "
          f"{AL.shared_direction_count(null, COS_THRESHOLD)} for the null")

    fake = {ds: {"dataset": ds, "channel_dim": D, "r_d": r_d,
                 "static_vs_velocity": {"mean": cos_ab.tolist(), "min": cos_ab.tolist(),
                                        "max": cos_ab.tolist(),
                                        "shared_gt_0.5_mean_over_folds": float(n_shared),
                                        "shared_gt_0.5_per_fold": [n_shared]},
                 "q95_vs_q05_within_static": {"mean": np.linspace(1, 0.6, r_d).tolist(),
                                              "shared_gt_0.5_mean_over_folds": float(r_d)},
                 "random_null": {"mean": null.tolist(),
                                 "shared_gt_0.5": AL.shared_direction_count(null, COS_THRESHOLD)}}
            for ds in ("truthfulqa", "triviaqa")}
    names, M = stability_matrix({"truthfulqa": [Ua, Ub], "triviaqa": [Ub, Ua]})
    assert M.shape == (2, 2) and np.all(np.isfinite(M))
    assert np.allclose(M, M.T, atol=1e-8), "stability matrix must be symmetric"
    print(f"  [PASS] stability_matrix: 2x2, finite, symmetric (diag={np.diag(M).round(3).tolist()})")

    cap = build_caption(fake, names, M)
    assert "random null" in cap and "Route-N" in cap
    assert "complementary" in cap or "orthogonal" in cap or "redundant" in cap
    print("  [PASS] caption includes the null reference, the provenance caveat, and a verdict")

    out = AL.output_dir("_selftest")
    paths = plot(fake, names, M, os.path.join(out, "fig2_selftest"))
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
    ap.add_argument("--max-beams", type=int, default=None,
                    help="Subsample beams for the subspace estimate only. Any use is recorded in "
                         "the JSON and MUST be reported as a Deviation.")
    ap.add_argument("--fold0-timing-only", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return

    datasets = AL.DATASETS if a.dataset == "all" else [a.dataset]
    out = AL.output_dir(a.out_model_name)
    stem = os.path.join(out, f"fig2_principal_angles_{a.out_model_name}_{a.dataset}")
    if not a.fold0_timing_only:
        AL.announce_outputs([f"{stem}.json", f"{stem}.pdf", f"{stem}.png"])

    data_dir = AL.get_data_dir(a.data_dir)
    results, static_factors = {}, {}
    t0 = time.time()
    for ds in datasets:
        print(f"\n[{ds}] fitting fold-local subspaces ...", flush=True)
        r = run_dataset(ds, data_dir, a.model_folder, a.max_beams, timing_probe=a.fold0_timing_only)
        if a.fold0_timing_only:
            print("[timing probe] exiting without writing any file."); return
        results[ds], static_factors[ds] = r

    names, M = stability_matrix(static_factors)
    payload = {"figure": "FIG2_principal_angles",
               "caption": build_caption(results, names, M),
               "cos_threshold": COS_THRESHOLD,
               "stability": {"datasets": names, "mean_cos_first10": M.tolist(),
                             "diagonal_meaning": "fold 1 vs fold 2, same dataset",
                             "offdiagonal_meaning": "dataset A fold 0 vs dataset B fold 0"},
               "runtime_seconds": round(time.time() - t0, 1), "results": results,
               "provenance_block": AL.base_provenance(
                   {d: {"provenance": results[d]["provenance"],
                        "alignment_verified": results[d]["alignment_verified"],
                        "n_beams": results[d]["n_beams"], "n_prompts": results[d]["n_prompts"]}
                    for d in results})}
    AL.save_json(f"{stem}.json", payload)
    plot(results, names, M, stem)
    print(f"\nFIG 2 complete in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
