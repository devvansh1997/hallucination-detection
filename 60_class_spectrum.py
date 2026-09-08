"""
60_class_spectrum.py -- does low-rank structure differ between hallucinated and truthful answers?
=====================================================================================================
THE QUESTION, asked by the PI on 2026-09-08: our numbers are good, but WHY? What is different about
a hallucination? This is the analysis figure that answers it, or shows there is nothing to answer.

THE HYPOTHESIS THE METHOD IMPLIES. We compress the feature mode with a basis fitted to the second
moment of observed states, and keep r_F = 64 of D directions. That is the right inductive bias only
if the two classes actually differ in how their variance is distributed across directions. So:
fit the spectrum separately over hallucinated and over truthful answers and compare. If truthful
answers concentrate in fewer directions, the whole method has a mechanism rather than an AUROC.

WHAT IS MEASURED, per layer and per class:
  effective rank   exp(H(lambda / sum lambda)), the exponential of the spectral entropy. Equals k
                   for k equal eigenvalues and 1 for a rank-one cloud, so it reads as "how many
                   directions does this class really use".
  participation    (sum lambda)^2 / sum lambda^2. A second, differently-shaped summary; if the two
                   disagree the effect is a property of the summary, not of the data.
  energy@64        fraction of variance inside the leading 64 directions -- literally what our
                   compression keeps, so it converts the figure into a statement about the method.

THE TRAP THIS FILE EXISTS TO AVOID. Sample covariance spectra are biased by sample size: with n
rows in D dimensions the eigenvalues spread further as n shrinks, so a class with fewer answers
looks lower-rank FOR FREE. Our classes are never balanced -- TyDiQA is 59.1% hallucinated,
TruthfulQA 43.3% -- so comparing raw per-class spectra would manufacture exactly the finding we are
looking for. Every comparison here therefore SUBSAMPLES BOTH CLASSES TO THE SAME n, repeated
`--n-boot` times at 80% of the matched size -- a full draw of the minority class would
carry no variability at all -- and reports the spread. The self-test includes a null: two classes drawn from one
distribution with deliberately unequal counts must come out indistinguishable.

SCALING. Channels are centred by the median and scaled by IQR/1.349 over ALL rows, both classes
together, matching 43_eval_phase2's robust scaler. Fitting the scaler on all rows rather than
per-class is deliberate: a per-class scaler would absorb the very difference being measured.

EIGENVALUES VIA THE GRAM TRICK. n < D here, so the non-zero spectrum of X^T X equals that of
X X^T, which is n x n rather than 3584 x 3584. Same numbers, seconds instead of minutes.

Usage:
  python 60_class_spectrum.py --self-test
  python 60_class_spectrum.py --dataset tydiqa_gp  --model_folder qwen-2.5-7b-instruct
  python 60_class_spectrum.py --dataset truthfulqa --model_folder qwen-2.5-7b-instruct --stream q95
  python 60_class_spectrum.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct --resume

Results are written after EVERY layer, so a run that dies keeps what it had; --resume
picks up from the layers already on disk.
"""

import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.abspath(os.path.join(HERE, "..", "data-alllayers"))
DEFAULT_OUT = os.path.join(HERE, "results", "class_spectrum")
R_F = 64                 # the feature rank the pipeline actually keeps
WINDOW = list(range(15, 24))


def robust_scale(X, eps=1e-8):
    """Median/IQR scaling, fitted on ALL rows. Matches 43_eval_phase2's scaler."""
    X = np.asarray(X, dtype=np.float64)
    med = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    scale = (q75 - q25) / 1.349
    scale[scale < eps] = 1.0
    return (X - med) / scale


def spectrum(X):
    """Non-zero eigenvalues of the covariance of X (n, D), descending.

    Gram trick: for n < D the non-zero spectrum of X^T X / n equals that of X X^T / n. Computed
    through singular values, which is the numerically stable route to the same quantity."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return np.zeros(0)
    s = np.linalg.svd(X - X.mean(axis=0, keepdims=True), compute_uv=False)
    lam = (s ** 2) / max(n - 1, 1)
    return lam[lam > 0]


def effective_rank(lam):
    """exp of the spectral entropy. k for k equal eigenvalues, 1 for rank one."""
    lam = np.asarray(lam, dtype=np.float64)
    tot = lam.sum()
    if tot <= 0 or lam.size == 0:
        return float("nan")
    p = lam / tot
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(lam):
    """(sum l)^2 / sum l^2. A differently-shaped summary of the same spectrum."""
    lam = np.asarray(lam, dtype=np.float64)
    d = (lam ** 2).sum()
    return float(lam.sum() ** 2 / d) if d > 0 else float("nan")


def energy_at(lam, k=R_F):
    """Fraction of variance in the leading k directions -- what our compression keeps."""
    lam = np.sort(np.asarray(lam, dtype=np.float64))[::-1]
    tot = lam.sum()
    return float(lam[:k].sum() / tot) if tot > 0 else float("nan")


def compare_classes(X, y, n_boot=20, seed=0, k=R_F, frac=0.8):
    """Per-class spectral summaries at MATCHED sample size.

    Returns {stat: {"0": (mean, lo, hi), "1": (mean, lo, hi)}, "n_draw": n}. Both classes are
    subsampled to the SAME n on every draw, which is the whole point: an unmatched comparison
    reports sample size, not structure.

    WHY frac < 1. Drawing exactly min(n_0, n_1) rows means the minority class is sampled in full
    every time, so its interval collapses to a single point and the majority class's band almost
    never contains it -- separation is then reported for two identical distributions. The first
    version of this function did that and the null test caught it. Taking a fraction of the matched
    size gives BOTH classes real sampling variability, without the row duplication that sampling
    with replacement would introduce."""
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    n = int(frac * min(len(i0), len(i1)))
    out = {s: {"0": [], "1": []} for s in ("erank", "participation", "energy")}
    if n < 8:
        return {s: {c: (float("nan"),) * 3 for c in ("0", "1")} for s in out} | {"n_draw": n}

    for b in range(n_boot):
        for cls, idx in (("0", i0), ("1", i1)):
            sub = rng.choice(idx, size=n, replace=False)
            lam = spectrum(X[sub])
            out["erank"][cls].append(effective_rank(lam))
            out["participation"][cls].append(participation_ratio(lam))
            out["energy"][cls].append(energy_at(lam, k))

    def band(v):
        v = np.asarray(v, dtype=np.float64)
        return (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    res = {s: {c: band(out[s][c]) for c in ("0", "1")} for s in out}
    res["n_draw"] = n
    return res


def separated(a, b):
    """True when the two 95% bands do not overlap -- the only claim this figure may make."""
    return a[2] < b[1] or b[2] < a[1]


def _verdict(rows):
    n_sep = sum(1 for r in rows if separated(r["erank"]["0"], r["erank"]["1"]))
    signs = [np.sign(r["erank"]["1"][0] - r["erank"]["0"][0]) for r in rows
             if separated(r["erank"]["0"], r["erank"]["1"])]
    if n_sep == 0:
        v = ("no separation at any depth -- the classes do not differ in effective rank, and this "
             "figure should not go in the paper")
    elif all(s > 0 for s in signs):
        v = "hallucinated answers use MORE directions at %d/%d depths" % (n_sep, len(rows))
    elif all(s < 0 for s in signs):
        v = "truthful answers use MORE directions at %d/%d depths" % (n_sep, len(rows))
    else:
        v = ("separated at %d/%d depths but the SIGN FLIPS across depth -- not a clean story"
             % (n_sep, len(rows)))
    return n_sep, v


def _save(dst, payload):
    """Write atomically: a run killed mid-write must not leave a truncated JSON behind."""
    tmp = dst + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, dst)


def run(dataset, model_folder, in_dir, out_dir, stream, n_boot, seed, layers=None, resume=False):
    path = os.path.join(in_dir, model_folder, "%s_alllayers.npz" % dataset)
    if not os.path.exists(path):
        raise SystemExit(
            "%s not found. This reads 57_extract_all_layers.py's output:\n"
            "    python 57_extract_all_layers.py --dataset %s --model_folder %s"
            % (path, dataset, model_folder))
    z = np.load(path)
    if stream not in z:
        raise SystemExit("stream %r not in %s; available: %s"
                         % (stream, path, [k for k in z.files if k in ("core", "q95", "q05")]))
    A, y = z[stream], np.asarray(z["label"], dtype=int)
    n_beams, n_layers, D = A.shape
    todo = list(range(n_layers)) if layers is None else list(layers)

    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "spectrum_%s_%s_%s.json" % (model_folder, dataset, stream))

    # RESUME. Each layer costs minutes, so a run that dies at layer 27 must not throw away 27
    # layers of work. Results are written after EVERY layer, and --resume skips what is already
    # on disk. Today four jobs died for unrelated reasons; losing an hour of CPU to the fifth
    # would be self-inflicted.
    rows = []
    if resume and os.path.exists(dst):
        with open(dst) as f:
            prev = json.load(f)
        if prev.get("stream") == stream and prev.get("n_boot") == n_boot:
            rows = prev.get("layers", [])
            done = {r["layer"] for r in rows}
            todo = [li for li in todo if li not in done]
            print("  resuming: %d layers already in %s, %d to go"
                  % (len(done), os.path.basename(dst), len(todo)), flush=True)
        else:
            print("  NOT resuming: %s was written with different settings (stream/n_boot)"
                  % os.path.basename(dst), flush=True)

    print("  [%s/%s] stream=%s  %d answers (%.1f%% hallucinated), %d layers, D=%d"
          % (model_folder, dataset, stream, n_beams, 100.0 * y.mean(), n_layers, D), flush=True)

    t0 = time.time()
    for k, li in enumerate(todo):
        X = robust_scale(np.asarray(A[:, li, :], dtype=np.float32))
        r = compare_classes(X, y, n_boot=n_boot, seed=seed + li)
        r["layer"] = int(li)
        rows.append(r)
        e0, e1 = r["erank"]["0"], r["erank"]["1"]
        el = time.time() - t0
        print("    layer %2d  erank truthful %7.2f [%.2f, %.2f]   hallucinated %7.2f [%.2f, %.2f]"
              "   %-11s  (%.0fs, eta %.0fs)"
              % (li, e0[0], e0[1], e0[2], e1[0], e1[1], e1[2],
                 "SEPARATED" if separated(e0, e1) else "overlapping",
                 el, el / (k + 1) * (len(todo) - k - 1)), flush=True)

        rows.sort(key=lambda r_: r_["layer"])
        n_sep, verdict = _verdict(rows)
        _save(dst, {"dataset": dataset, "model_folder": model_folder, "stream": stream,
                    "source": os.path.abspath(path), "n_boot": n_boot, "seed": seed,
                    "r_f": R_F, "window": WINDOW, "hallucination_rate": float(y.mean()),
                    "n_layers_done": len(rows), "n_layers_total": int(n_layers),
                    "complete": len(rows) == n_layers,
                    "n_layers_separated": n_sep, "verdict": verdict,
                    "note": ("classes are subsampled to equal n on every draw; an unmatched "
                             "comparison would report sample size, not structure"),
                    "layers": rows, "elapsed_seconds": round(time.time() - t0, 1)})

    if rows:
        _, verdict = _verdict(rows)
        print("\n  VERDICT (%d/%d layers): %s" % (len(rows), n_layers, verdict))
    print("  wrote %s" % dst)
    return rows


def self_test():
    print("=" * 78)
    print("  SELF-TEST: 60_class_spectrum")
    print("=" * 78)
    rng = np.random.default_rng(0)

    # effective_rank must read as "how many directions", on spectra whose answer is known.
    assert abs(effective_rank(np.ones(10)) - 10.0) < 1e-9
    assert abs(effective_rank(np.array([1.0])) - 1.0) < 1e-9
    assert abs(effective_rank(np.array([1.0, 0.0, 0.0])) - 1.0) < 1e-9
    assert effective_rank(np.array([10.0, 1.0, 1.0])) < 3.0
    print("  [PASS] effective_rank: 10 equal -> 10.0, rank one -> 1.0, skewed -> < 3")

    assert abs(participation_ratio(np.ones(8)) - 8.0) < 1e-9
    assert abs(energy_at(np.array([9.0, 1.0]), k=1) - 0.9) < 1e-12
    print("  [PASS] participation_ratio on equal spectrum = 8.0; energy@1 of (9,1) = 0.90")

    # The Gram trick must agree with the full covariance eigendecomposition it stands in for.
    X = rng.standard_normal((60, 200))
    lam_gram = np.sort(spectrum(X))[::-1]
    Xc = X - X.mean(axis=0, keepdims=True)
    lam_full = np.sort(np.linalg.eigvalsh(Xc.T @ Xc / (X.shape[0] - 1)))[::-1][:lam_gram.size]
    assert np.allclose(lam_gram, lam_full, atol=1e-8), np.abs(lam_gram - lam_full).max()
    print("  [PASS] Gram-trick spectrum matches the full D x D eigendecomposition to 1e-8")

    # A REAL difference must be detected, in the right direction. Class 1 lives in a 5-dimensional
    # subspace; class 0 is isotropic. Class 1 must come out LOWER rank.
    n, D = 400, 120
    X0 = rng.standard_normal((n, D))
    X1 = rng.standard_normal((n, 5)) @ rng.standard_normal((5, D))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    r = compare_classes(X, y, n_boot=8, seed=1)
    assert r["erank"]["1"][0] < r["erank"]["0"][0], r["erank"]
    assert separated(r["erank"]["0"], r["erank"]["1"])
    print("  [PASS] a genuine rank difference is found, correct direction and bands disjoint "
          "(%.1f vs %.1f)" % (r["erank"]["0"][0], r["erank"]["1"][0]))

    # THE NULL, and the assertion this file is built around. Both classes from ONE distribution but
    # with very unequal counts. Without subsampling the smaller class looks lower-rank purely from
    # its n, which would manufacture the finding. Matched, they must be indistinguishable.
    big, small = 900, 150
    X = rng.standard_normal((big + small, 120))
    y = np.r_[np.zeros(big, int), np.ones(small, int)]
    r = compare_classes(X, y, n_boot=12, seed=2)
    assert not separated(r["erank"]["0"], r["erank"]["1"]), (
        "NULL FAILED: identical distributions were reported as different (%s). The matched "
        "subsampling is not working, and every result from this script would be an artifact."
        % (r["erank"],))
    assert r["n_draw"] < small, (r["n_draw"], small)
    print("  [PASS] null: 900 vs 150 rows from ONE distribution stay indistinguishable "
          "(%.1f vs %.1f, both drawn at n=%d)"
          % (r["erank"]["0"][0], r["erank"]["1"][0], r["n_draw"]))

    # NEITHER band may be degenerate. A zero-width interval on the minority class is what made the
    # first version report separation between identical distributions, and it would not show up in
    # the null assertion alone -- a point estimate can land inside the other band by luck.
    for cls in ("0", "1"):
        lo, hi = r["erank"][cls][1], r["erank"][cls][2]
        assert hi - lo > 1e-6, (
            "class %s has a zero-width interval: it is being sampled in full on every draw, so it "
            "carries no sampling variability and any comparison against it is meaningless" % cls)
    print("  [PASS] both classes have non-degenerate bands (widths %.2f and %.2f)"
          % (r["erank"]["0"][2] - r["erank"]["0"][1], r["erank"]["1"][2] - r["erank"]["1"][1]))

    # And the same data WITHOUT matching must look different -- otherwise the previous assertion
    # passes for the wrong reason and proves nothing about the subsampling.
    unmatched_big = effective_rank(spectrum(X[y == 0]))
    unmatched_small = effective_rank(spectrum(X[y == 1]))
    assert unmatched_small < unmatched_big * 0.9, (unmatched_small, unmatched_big)
    print("  [PASS] the same data UNMATCHED shows a spurious gap (%.1f vs %.1f) -- so the null "
          "above is a real check, not a vacuous one" % (unmatched_big, unmatched_small))

    # Scaling must not itself encode the labels: fitted on all rows, it is one affine map.
    Xs = robust_scale(np.vstack([rng.standard_normal((50, 6)), 100 + rng.standard_normal((50, 6))]))
    assert np.isfinite(Xs).all() and Xs.shape == (100, 6)
    print("  [PASS] robust_scale is finite on a bimodal column and fitted over all rows")

    print("\n  ALL PASS")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dataset")
    p.add_argument("--model_folder")
    p.add_argument("--in-dir", default=DEFAULT_IN)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--stream", default="core", choices=["core", "q95", "q05"])
    p.add_argument("--n-boot", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--resume", action="store_true",
                   help="skip layers already present in the output JSON")
    a = p.parse_args()

    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not a.dataset or not a.model_folder:
        raise SystemExit("--dataset and --model_folder are required (or use --self-test)")
    run(a.dataset, a.model_folder, a.in_dir, a.out_dir, a.stream, a.n_boot, a.seed,
        a.layers, a.resume)


if __name__ == "__main__":
    main()
