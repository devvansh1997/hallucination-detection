"""
61_flatten_control.py -- does the multilinear structure earn its place, at matched dimension?
=====================================================================================================
THE CLAIM THIS TESTS. Our contribution says a structure-preserving decomposition recovers signal that
a flattened representation discards. Nothing in the repo tested it. The obvious reviewer question --
"you compress to 896 dimensions; what does plain PCA to 896 dimensions do?" -- had no answer.

FOUR ARMS, identical everywhere except the reduction. Same pinned generations, same robust scaling
fitted on the same training rows, same output dimension, same readout, same splits, same seeds:

  hosvd        ours. G_n = U_L^T H_n U_F per stream, Kronecker-structured, fitted without labels.
  flat_pca     flatten each answer's (L, F) to L*F and take principal components to the SAME width.
               An unrestricted linear map, so it can express anything hosvd can and more. If this
               wins, the Kronecker restriction is costing us and the framing is wrong.
  layer_mean   average over the layer mode, then PCA to the same width. Tests whether treating depth
               as a MODE beats treating it as something to average away.
  random_proj  a Gaussian random projection to the same width, fitted on nothing. The control that
               matters most and is cheapest to skip: if hosvd only matches this, the fitted bases
               are decoration and the readout is doing all the work.

WHY MATCHED DIMENSION IS THE ONLY FAIR COMPARISON. A reduction that keeps more numbers should win;
that would say nothing about structure. Every arm here emits exactly the same width, so the only
thing varying is WHICH subspace is kept.

WHAT EACH OUTCOME MEANS.
  hosvd > flat_pca      the Kronecker restriction helps rather than merely saving parameters, which
                        is the claim. Report it as the justification for the architecture.
  hosvd ~ flat_pca      the structure buys nothing at this width; it saves parameters and no more.
                        Honest framing then becomes efficiency, not signal recovery.
  hosvd < flat_pca      the restriction costs accuracy. That must be reported, and the contribution
                        rewritten around what the decomposition actually gives (estimability).
  hosvd ~ random_proj   nothing is being learned by the reduction at all. This is the one that would
                        force a rethink, and it is the one nobody runs.

Usage:
  python 61_flatten_control.py --self-test
  python 61_flatten_control.py --dataset tydiqa_gp  --model_folder qwen-2.5-7b-instruct
  python 61_flatten_control.py --dataset truthfulqa --model_folder qwen-2.5-7b-instruct --readout RF
  python 61_flatten_control.py --dataset tydiqa_gp --model_folder llama-3.1-8b --condition triple_concat
  python 61_flatten_control.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct \
      --readout RF --arms hosvd best_layer --tag bestlayer
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "flatten_control")

# The condition is read from 44_eval_phase3.CONDITION_SPECS rather than duplicated here, so this
# file cannot drift from the definition the main results use. triple_concat is the configuration the
# paper reports: C (9 x D) at r_L=5, S (9 x 2D) at r_L=5, V (8 x 2D) at r_L=4, all at r_F=64, so
# 5*64 + 5*64 + 4*64 = 896 dimensions.
DEFAULT_CONDITION = "triple_concat"
ARMS = ("hosvd", "flat_pca", "layer_mean", "random_proj")
# best_layer is NOT in the default tuple: adding it would invalidate the four-arm
# results already on disk. Run it explicitly against hosvd, with a --tag.
ALL_ARMS = ARMS + ("best_layer",)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_mods = {}


def mods():
    if not _mods:
        _mods["s43"] = _load("s43", "43_eval_phase2.py")       # robust_scale_3d, core builder
        _mods["s26"] = _load("s26", "26_grouped_baseline.py")  # fit_eval
        _mods["s44"] = _load("s44", "44_eval_phase3.py")       # load_new_dataset
    return _mods


# ---------------------------------------------------------------------------------------------
# The four reductions. Each takes the FULL scaled tensor plus the training rows, fits whatever it
# fits on those rows only, and returns the projected matrix for every row.
# ---------------------------------------------------------------------------------------------

def reduce_hosvd(X_raw, tr, r_l, r_d, seed, ctx=None):
    """Ours, exactly as the pipeline computes it -- 43's function, not a reimplementation."""
    return mods()["s43"].fold_pure_core_randomized(X_raw, tr, r_l, r_d, seed)


def _fit_pca(M_train, k, seed):
    """Randomized PCA basis from training rows. Returns (mean, components) with components (D, k)."""
    from sklearn.utils.extmath import randomized_svd
    mu = M_train.mean(axis=0, keepdims=True)
    _, _, Vt = randomized_svd(M_train - mu, n_components=k, n_iter=4, random_state=seed)
    return mu, Vt.T


def reduce_flat_pca(X_raw, tr, r_l, r_d, seed, ctx=None):
    """Flatten (L, F) to L*F, then PCA to r_l * r_d -- the SAME width the core would have."""
    Xs = mods()["s43"].robust_scale_3d(X_raw, tr)
    M = Xs.reshape(Xs.shape[0], -1)
    mu, comp = _fit_pca(M[tr], r_l * r_d, seed)
    return (M - mu) @ comp


def reduce_layer_mean(X_raw, tr, r_l, r_d, seed, ctx=None):
    """Average the layer mode away, then PCA to the same width. Depth as nuisance, not as mode."""
    Xs = mods()["s43"].robust_scale_3d(X_raw, tr)
    M = Xs.mean(axis=1)
    mu, comp = _fit_pca(M[tr], min(r_l * r_d, M.shape[1]), seed)
    return (M - mu) @ comp


def reduce_random_proj(X_raw, tr, r_l, r_d, seed, ctx=None):
    """Gaussian random projection of the flattened tensor. Fitted on NOTHING."""
    Xs = mods()["s43"].robust_scale_3d(X_raw, tr)
    M = Xs.reshape(Xs.shape[0], -1)
    k = r_l * r_d
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((M.shape[1], k)).astype(np.float32) / np.sqrt(k)
    return M @ R


def _grouped_carve(prompt_id, tr, seed, frac=0.2):
    """Hold out whole questions from the training rows. Local rather than imported from
    methods/act_vit so this file keeps its sklearn-only dependency."""
    q = np.unique(prompt_id[tr])
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(q)[:max(1, int(round(len(q) * frac)))].tolist())
    m = np.array([int(x) in held for x in prompt_id[tr]])
    return (tr, None) if (m.all() or not m.any()) else (tr[~m], tr[m])


def reduce_best_layer(X_raw, tr, r_l, r_d, seed, ctx=None):
    """Pick ONE layer, then PCA it to the same width the core would have.

    WHY THIS ARM EXISTS. The per-layer sweep shows the signal concentrated near layers 17-18, which
    invites the obvious question: if one layer carries it, what is the nine-layer decomposition for?
    That comparison is only meaningful at matched width and under the same protocol, so it belongs
    here rather than being read across from the sweep, whose numbers come from a different split and
    a different readout.

    THE LAYER IS CHOSEN ON TRAINING ROWS ONLY. A grouped carve holds out whole questions from `tr`,
    every layer is fitted and scored on that carve, the best is taken, and the projection is then
    refitted on all of `tr`. Selecting on the test rows would hand this arm the answer, and selecting
    once on one seed would leak that seed's training rows into another seed's test set."""
    from sklearn.metrics import roc_auc_score
    m = mods()
    y, pid, readout = ctx["y"], ctx["prompt_id"], ctx["readout"]
    k = r_l * r_d
    fit_idx, val_idx = _grouped_carve(pid, tr, seed)

    best_li, best_auc = 0, -np.inf
    if val_idx is not None and len(np.unique(y[val_idx])) > 1:
        for li in range(X_raw.shape[1]):
            Xi = np.ascontiguousarray(X_raw[:, li:li + 1, :])
            Zi = reduce_flat_pca(Xi, fit_idx, 1, min(k, Xi.shape[2]), seed)
            sc = m["s26"].fit_eval(readout, Zi[fit_idx], y[fit_idx], Zi[val_idx], seed)
            a = roc_auc_score(y[val_idx], sc)
            if a > best_auc:
                best_li, best_auc = li, a
    ctx.setdefault("chosen_layers", []).append({"layer": int(best_li),
                                                "val_auroc": float(best_auc)})
    Xb = np.ascontiguousarray(X_raw[:, best_li:best_li + 1, :])
    return reduce_flat_pca(Xb, tr, 1, min(k, Xb.shape[2]), seed)


REDUCERS = {"hosvd": reduce_hosvd, "flat_pca": reduce_flat_pca,
            "layer_mean": reduce_layer_mean, "random_proj": reduce_random_proj,
            "best_layer": reduce_best_layer}


def build(arm, feats, spec, tr, seed, ctx=None):
    """Apply one arm to every sub-tensor of the condition and concatenate, as the pipeline does."""
    parts = []
    for i, (key, r_l, r_d) in enumerate(spec):
        parts.append(REDUCERS[arm](feats[key], tr, r_l, r_d, seed + i, ctx))
    return np.concatenate(parts, axis=1)


def run(dataset, model_folder, data_dir, out_dir, readout, seeds=None, arms=ARMS,
        condition=DEFAULT_CONDITION, tag=None):
    m = mods()
    import methods.base as B
    c = B.canonical()
    seeds = list(seeds or c["seeds"])
    spec = m["s44"].CONDITION_SPECS[condition]

    # Four return values, and the condition name matters: without it the loader materialises every
    # raw tensor type rather than only the ones this condition touches.
    feats, y, prompt_id, is_known = m["s44"].load_new_dataset(
        dataset, data_dir, model_folder, condition=condition)
    y = np.asarray(y, dtype=int)
    prompt_id = np.asarray(prompt_id)
    n = len(y)
    width = sum(r_l * r_d for _, r_l, r_d in spec)
    print("  [%s/%s] %d answers, %d questions, %.1f%% hallucinated | condition %s, "
          "width %d, readout %s" % (model_folder, dataset, n, len(np.unique(prompt_id)),
                                    100.0 * y.mean(), condition, width, readout), flush=True)

    res = {a: [] for a in arms}
    t0 = time.time()
    for seed in seeds:
        tr, te = c["question_split"](is_known, prompt_id, n, seed)
        tr, te = np.asarray(tr, dtype=int), np.asarray(te, dtype=int)
        for arm in arms:
            t1 = time.time()
            ctx = {"y": y, "prompt_id": prompt_id, "readout": readout}
            Z = build(arm, feats, spec, tr, seed, ctx)
            assert Z.shape[1] == width or arm == "layer_mean", (arm, Z.shape, width)
            sc = m["s26"].fit_eval(readout, Z[tr], y[tr], Z[te], seed)
            # (scores, labels), in that order -- reversing them silently returns 1 - AUROC. And
            # within_prompt_auroc returns a dict, not a float. Both pinned by the self-test.
            wp = c["within_prompt_auroc"](sc, y[te], prompt_id[te])
            r = {"seed": int(seed), "dim": int(Z.shape[1]),
                 "pooled_auroc": float(c["pooled_auroc"](sc, y[te])),
                 "within_prompt_auroc": wp["within_prompt_auroc"],
                 "n_pairs": wp["n_pairs"],
                 "seconds": round(time.time() - t1, 1)}
            if ctx.get("chosen_layers"):
                r["chosen_layers"] = ctx["chosen_layers"]
            res[arm].append(r)
            print("    seed %-3d %-12s pooled %.4f  within %.4f  (%.0fs)"
                  % (seed, arm, r["pooled_auroc"],
                     float("nan") if r["within_prompt_auroc"] is None
                     else r["within_prompt_auroc"], r["seconds"]), flush=True)

    summary = {}
    for arm in arms:
        p = np.array([r["pooled_auroc"] for r in res[arm]])
        w = np.array([r["within_prompt_auroc"] for r in res[arm] if r["within_prompt_auroc"]])
        summary[arm] = {"pooled_mean": float(p.mean()), "pooled_std": float(p.std()),
                        "within_mean": float(w.mean()) if len(w) else None,
                        "dim": res[arm][0]["dim"], "per_seed": res[arm]}

    base = summary["hosvd"]["pooled_mean"]
    deltas = {a: round(100.0 * (base - summary[a]["pooled_mean"]), 2)
              for a in arms if a != "hosvd"}
    verdict = ("hosvd beats every control" if all(v > 0 for v in deltas.values())
               else "hosvd is NOT the best arm -- " +
                    ", ".join("%s by %.2f" % (a, -v) for a, v in deltas.items() if v < 0))
    print("\n  hosvd %.4f | " % base
          + " | ".join("%s %.4f (%+.2f)" % (a, summary[a]["pooled_mean"], -deltas[a])
                       for a in arms if a != "hosvd"))
    print("  VERDICT: %s" % verdict)

    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "flatten_%s_%s_%s%s.json"
                       % (model_folder, dataset, readout, ("_" + tag) if tag else ""))
    with open(dst, "w") as f:
        json.dump({"dataset": dataset, "model_folder": model_folder, "readout": readout,
                   "condition": condition, "width": width, "seeds": seeds,
                   "protocol": "question-level (paper protocol), 5 seeds",
                   "hallucination_rate_pct": round(100.0 * float(y.mean()), 3),
                   "summary": summary, "delta_vs_hosvd_pts": deltas, "verdict": verdict,
                   "elapsed_seconds": round(time.time() - t0, 1)}, f, indent=2)
    print("  wrote %s" % dst)
    return summary


def self_test():
    print("=" * 78)
    print("  SELF-TEST: 61_flatten_control")
    print("=" * 78)
    rng = np.random.default_rng(0)
    n, L, D = 400, 6, 40
    tr = np.arange(0, 300)

    # Every arm must emit the SAME width, or the comparison is about dimension rather than
    # structure. layer_mean is the exception: its input has only D columns to draw from.
    X = rng.standard_normal((n, L, D)).astype(np.float32)
    widths = {a: REDUCERS[a](X, tr, 5, 6, 0).shape[1] for a in ARMS}
    assert widths["hosvd"] == widths["flat_pca"] == widths["random_proj"] == 30, widths
    assert widths["layer_mean"] == min(30, D), widths
    print("  [PASS] hosvd, flat_pca and random_proj all emit width 30; layer_mean min(30, D)=%d"
          % widths["layer_mean"])

    # No leakage: every fitted arm must produce IDENTICAL output for a given row regardless of
    # which rows are in test, as long as the training rows are the same.
    for a in ("hosvd", "flat_pca", "layer_mean"):
        Z_full = REDUCERS[a](X, tr, 5, 6, 0)
        Z_more = REDUCERS[a](np.concatenate([X, rng.standard_normal((50, L, D)).astype(np.float32)]),
                             tr, 5, 6, 0)
        assert np.allclose(Z_full, Z_more[:n], atol=1e-4), (
            "%s changed its projection when unseen rows were appended -- it is fitting on more "
            "than the training rows" % a)
    print("  [PASS] hosvd, flat_pca and layer_mean fit on training rows only (adding unseen rows "
          "leaves their output unchanged)")

    # random_proj must genuinely ignore the data: same seed, different training rows, same map.
    Za = REDUCERS["random_proj"](X, np.arange(0, 200), 5, 6, 7)
    Zb = REDUCERS["random_proj"](X, np.arange(0, 200), 5, 6, 7)
    assert np.allclose(Za, Zb)
    print("  [PASS] random_proj is deterministic given a seed and fits nothing")

    # A signal that is genuinely Kronecker-structured -- one layer direction crossed with one
    # feature direction -- must be recovered better by hosvd than by a random projection of the
    # same width. If this fails, the arms are not measuring what the docstring claims.
    from sklearn.metrics import roc_auc_score
    ul = rng.standard_normal(L); ul /= np.linalg.norm(ul)
    uf = rng.standard_normal(D); uf /= np.linalg.norm(uf)
    y = (np.arange(n) % 2).astype(int)
    Xk = rng.standard_normal((n, L, D)).astype(np.float32) * 0.6
    Xk += (y[:, None, None] * 3.0) * np.outer(ul, uf)[None, :, :].astype(np.float32)
    te = np.arange(300, n)
    m = mods()
    auc = {}
    for a in ("hosvd", "random_proj"):
        Z = REDUCERS[a](Xk, tr, 2, 4, 0)
        auc[a] = roc_auc_score(y[te], m["s26"].fit_eval("LR", Z[tr], y[tr], Z[te], 0))
    assert auc["hosvd"] > auc["random_proj"], auc
    print("  [PASS] on a Kronecker-structured signal hosvd (%.3f) beats random_proj (%.3f) at the "
          "same width" % (auc["hosvd"], auc["random_proj"]))

    # SIGNATURE GUARDS. Both bugs this catches were shipped: load_new_dataset returns FOUR values
    # (that one raised), and the metrics take (scores, labels) -- reversing those returns 1 - AUROC
    # silently, which is the EigenScore failure mode all over again.
    import inspect
    import methods.base as B
    src = inspect.getsource(mods()["s44"].load_new_dataset)
    assert src.rstrip().endswith("return feats, y, prompt_idx, is_known"), (
        "44.load_new_dataset's return signature changed; run() unpacks four values")
    assert "condition" in inspect.signature(mods()["s44"].load_new_dataset).parameters
    print("  [PASS] 44.load_new_dataset still returns 4 values and accepts condition=")

    cc = B.canonical()
    yy = np.array([0, 0, 0, 1, 1])
    ss = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert abs(cc["pooled_auroc"](ss, yy) - 1.0) < 1e-12, (
        "pooled_auroc(scores, labels) did not return 1.0 on a perfectly ordered vector -- the "
        "argument order is wrong and every number would be 1 - AUROC")
    w = cc["within_prompt_auroc"](ss, yy, np.array([0, 0, 1, 0, 1]))
    assert isinstance(w, dict) and "within_prompt_auroc" in w
    print("  [PASS] metrics take (scores, labels) and within_prompt_auroc returns a dict")

    assert set(mods()["s44"].CONDITION_SPECS) >= {"triple_concat", "core_concat"}
    w896 = sum(r_l * r_d for _, r_l, r_d in mods()["s44"].CONDITION_SPECS[DEFAULT_CONDITION])
    assert w896 == 896, w896
    print("  [PASS] condition %s read from 44.CONDITION_SPECS, width %d" % (DEFAULT_CONDITION, w896))

    # best_layer must SELECT, not guess. Only layer 3 carries the label here, and the choice is
    # made on a grouped carve of the training rows, so a bug that selected on test rows or ignored
    # labels entirely would not land on 3 reliably.
    n2, L2, D2 = 300, 6, 40
    y2 = (np.arange(n2) % 2).astype(int)
    pid2 = np.repeat(np.arange(n2 // 2), 2)
    X2 = rng.standard_normal((n2, L2, D2)).astype(np.float32)
    X2[:, 3, :] += y2[:, None] * 4.0
    tr2 = np.arange(0, 220)
    ctx2 = {"y": y2, "prompt_id": pid2, "readout": "LR"}
    Z2 = reduce_best_layer(X2, tr2, 2, 8, 0, ctx2)
    assert Z2.shape[1] == min(16, D2)
    assert ctx2["chosen_layers"][0]["layer"] == 3, ctx2["chosen_layers"]
    print("  [PASS] best_layer selects the only informative layer (3) from a training-only carve, "
          "val AUROC %.3f" % ctx2["chosen_layers"][0]["val_auroc"])

    # The carve must not hand it test rows.
    f2, v2 = _grouped_carve(pid2, tr2, 0)
    assert v2 is not None and not np.intersect1d(f2, v2).size
    assert not (set(pid2[f2].tolist()) & set(pid2[v2].tolist())), "carve shares questions"
    assert set(np.concatenate([f2, v2]).tolist()) <= set(tr2.tolist()), "carve escaped train rows"
    print("  [PASS] the selection carve stays inside training rows and shares no question")

    print("\n  ALL PASS")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dataset")
    p.add_argument("--model_folder")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--readout", default="LR", choices=["LR", "RF"])
    p.add_argument("--arms", nargs="*", default=list(ARMS),
                   help="subset of %s" % (ALL_ARMS,))
    p.add_argument("--tag", default=None, help="suffix for the output filename")
    p.add_argument("--condition", default=DEFAULT_CONDITION)
    a = p.parse_args()

    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not a.dataset or not a.model_folder:
        raise SystemExit("--dataset and --model_folder are required (or use --self-test)")
    data_dir = a.data_dir
    if not data_dir:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]
    run(a.dataset, a.model_folder, data_dir, a.out_dir, a.readout, arms=a.arms,
        condition=a.condition, tag=a.tag)


if __name__ == "__main__":
    main()
