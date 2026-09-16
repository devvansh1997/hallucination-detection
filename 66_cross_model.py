"""
66_cross_model.py -- train the detector on one model, apply it to another (T-013).
==================================================================================================

SETTING. A SOURCE model S and a TARGET model T answered the same questions. The detector (triple_concat,
ranks (5,64)/(5,64)/(4,64), random forest -- exactly as reported) is trained on S's answers with S's
labels and scored on T's test answers against T's labels. NO label of T is used anywhere except for
scoring.

WHY AN ALIGNMENT. The core coordinates of the two models are not comparable: each model's Tucker bases
are fitted in its own hidden space (3584-d vs 4096-d) and are defined only up to sign, order and
rotation. So T's answers are first projected with T's OWN label-free bases (fitted on T's training
answers, as T's in-model detector does), and a linear map from T's core space into S's is fitted on
paired data that needs no labels either: T's training answers as T sees them, and the same answers as
S reads them (65_extract_cross_read.py). Ridge regression, penalty chosen by question-grouped 5-fold
cross-validation of the reconstruction R^2, fitted on T's training questions only.

ROWS REPORTED, per seed, all on T's test answers:
  in_model   T's own detector (T labels) -- the reference; must reproduce results/flatten_control
  aligned    S's detector on T's core mapped into S's space            <- the transfer result
  naive      S's detector on T's core with no map (coordinates unaligned; a floor)
  proxy      S's detector on S's hidden states while S reads T's answers (S used as a reader)

QUESTIONS NEVER CROSS. Our own leakage argument applies across models too: a question in T's test set
could be in S's training set, and S's detector could then recognise the question rather than judge the
answer. Every question in T's test split is removed from S's training split before anything is fitted.

  python 66_cross_model.py --self-test
  python 66_cross_model.py --source qwen-2.5-7b-instruct --target llama-3.1-8b --dataset truthfulqa
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CROSS = os.path.abspath(os.path.join(HERE, "..", "data-crossread"))
DEFAULT_OUT = os.path.join(HERE, "results", "cross_model")
CONDITION = "triple_concat"
ROWS = ("in_model", "aligned", "naive", "proxy")
ALPHA_GRID = np.logspace(-4, 1, 11)          # multiplied by the number of training pairs


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_mods = {}


def mods():
    if not _mods:
        _mods["s61"] = _load("s61", "61_flatten_control.py")
        _mods.update(_mods["s61"].mods())            # s43, s26, s44
    return _mods


# ---------------------------------------------------------------------------------------------
# Ridge map, closed form
# ---------------------------------------------------------------------------------------------

def ridge_fit(X, Y, alpha):
    """Standardise X on its own rows, centre Y, solve by SVD. Returns a dict usable by ridge_predict."""
    mx, sx = X.mean(axis=0), X.std(axis=0) + 1e-8
    my = Y.mean(axis=0)
    Xs = (X - mx) / sx
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    W = (Vt.T * (s / (s ** 2 + alpha))) @ (U.T @ (Y - my))
    return {"mx": mx, "sx": sx, "my": my, "W": W, "alpha": float(alpha)}


def ridge_predict(m, X):
    return ((X - m["mx"]) / m["sx"]) @ m["W"] + m["my"]


def r2_vw(Y, P):
    """Variance-weighted R^2 over outputs: 1 - total residual SS / total SS."""
    return float(1.0 - ((Y - P) ** 2).sum() / (((Y - Y.mean(axis=0)) ** 2).sum() + 1e-12))


def fit_alignment(X, Y, groups, seed, n_splits=5):
    """Ridge penalty by question-grouped CV of R^2, then refit on all pairs. Label-free."""
    from sklearn.model_selection import GroupKFold
    alphas = ALPHA_GRID * len(X)
    folds = list(GroupKFold(n_splits=n_splits).split(X, groups=groups))
    cv = []
    for a in alphas:
        scores = [r2_vw(Y[va], ridge_predict(ridge_fit(X[tr], Y[tr], a), X[va])) for tr, va in folds]
        cv.append(float(np.mean(scores)))
    best = int(np.argmax(cv))
    m = ridge_fit(X, Y, alphas[best])
    m.update({"cv_r2": cv[best], "cv_curve": [[float(a), c] for a, c in zip(alphas, cv)]})
    return m


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

def load_cross(cross_dir, source, target, dataset, y_t, pid_t):
    """S's features while S reads T's answers, in T's row order, as the three triple_concat tensors."""
    path = os.path.join(cross_dir, "%s_reads_%s" % (source, target), "%s_window.npz" % dataset)
    if not os.path.exists(path):
        raise SystemExit("%s not found -- run 65_extract_cross_read.py --reader %s --generator %s"
                         % (path, source, target))
    z = np.load(path)
    rows = np.asarray(z["generator_row"])
    if not np.array_equal(rows, np.arange(len(y_t))):
        raise SystemExit("cross file does not cover every target answer in order (%d rows vs %d)"
                         % (len(rows), len(y_t)))
    if not (np.array_equal(np.asarray(z["label"]).astype(int), y_t) and
            np.array_equal(np.asarray(z["prompt_id"]).astype(np.int64), pid_t.astype(np.int64))):
        raise SystemExit("labels or question ids in the cross file do not match the target's pinned "
                         "features row for row")
    feats = {"core": z["static_max"].astype(np.float32),
             "static": np.concatenate([z["static_q95"], z["static_q05"]], axis=2).astype(np.float32),
             "velocity": np.concatenate([z["velocity_q95"], z["velocity_q05"]], axis=2).astype(np.float32)}
    bad = {k: int((~np.isfinite(v)).sum()) for k, v in feats.items()}
    if any(bad.values()):
        raise SystemExit("non-finite entries in the cross features %s (an answer with no tokens after "
                         "re-encoding?) -- decide how to handle them before scoring" % bad)
    return feats


def exclude_questions(tr_idx, pid, drop):
    keep = ~np.isin(pid[tr_idx], np.asarray(sorted(drop)))
    return tr_idx[keep]


# ---------------------------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------------------------

def score_seed(m, c, spec, S, T, X, seed, readout="RF"):
    """S, T: dicts with feats, y, pid, known. X: S's features on T's answers (T's row order)."""
    s61, s26 = m["s61"], m["s26"]
    nS = len(S["y"])
    tr_t, te_t = (np.asarray(v, dtype=int) for v in c["question_split"](T["known"], T["pid"], len(T["y"]), seed))
    tr_s, _ = (np.asarray(v, dtype=int) for v in c["question_split"](S["known"], S["pid"], nS, seed))
    test_q = set(T["pid"][te_t].tolist())
    n_before = len(tr_s)
    tr_s = exclude_questions(tr_s, S["pid"], test_q)
    assert not (set(S["pid"][tr_s].tolist()) & test_q)
    assert not (set(T["pid"][tr_t].tolist()) & test_q)

    # S's label-free projection, fitted on S's training answers only, applied to S's own rows and to
    # S's reading of T's answers in one pass.
    both = {k: np.concatenate([S["feats"][k], X[k]], axis=0) for k in S["feats"]}
    Zboth = s61.build("hosvd", both, spec, tr_s, seed)
    del both
    Zs, Zx = Zboth[:nS], Zboth[nS:]
    # T's label-free projection, exactly as T's in-model detector computes it.
    Zt = s61.build("hosvd", T["feats"], spec, tr_t, seed)

    align = fit_alignment(Zt[tr_t], Zx[tr_t], T["pid"][tr_t], seed)
    Zt_mapped = ridge_predict(align, Zt[te_t])

    k = len(te_t)
    sc = s26.fit_eval(readout, Zs[tr_s], S["y"][tr_s], np.vstack([Zt_mapped, Zt[te_t], Zx[te_t]]), seed)
    scores = {"aligned": sc[:k], "naive": sc[k:2 * k], "proxy": sc[2 * k:]}
    scores["in_model"] = s26.fit_eval(readout, Zt[tr_t], T["y"][tr_t], Zt[te_t], seed)

    y, pid = T["y"][te_t], T["pid"][te_t]
    out = {"seed": int(seed), "n_test_answers": int(k),
           "source_train_answers": int(len(tr_s)), "source_train_removed": int(n_before - len(tr_s)),
           "alignment": {"alpha": align["alpha"], "cv_r2": align["cv_r2"],
                         "test_r2": r2_vw(Zx[te_t], Zt_mapped)}}
    for row in ROWS:
        wp = c["within_prompt_auroc"](scores[row], y, pid)
        out[row] = {"pooled_auroc": float(c["pooled_auroc"](scores[row], y)),
                    "within_prompt_auroc": wp["within_prompt_auroc"]}
    return out


def summarize(per_seed):
    s = {}
    for row in ROWS:
        p = np.array([r[row]["pooled_auroc"] for r in per_seed])
        w = [r[row]["within_prompt_auroc"] for r in per_seed if r[row]["within_prompt_auroc"] is not None]
        s[row] = {"pooled_mean": float(p.mean()), "pooled_std": float(p.std()),
                  "within_mean": float(np.mean(w)) if w else None}
    s["alignment_cv_r2_mean"] = float(np.mean([r["alignment"]["cv_r2"] for r in per_seed]))
    s["alignment_test_r2_mean"] = float(np.mean([r["alignment"]["test_r2"] for r in per_seed]))
    return s


def reference_check(per_seed, target, dataset, readout):
    """in_model must be the reported detector: the flatten control's hosvd arm, seed for seed."""
    path = os.path.join(HERE, "results", "flatten_control", "flatten_%s_%s_%s.json" % (target, dataset, readout))
    if not os.path.exists(path):
        return {"status": "not checked (no %s)" % path}
    with open(path) as f:
        ref = {r["seed"]: r["pooled_auroc"] for r in json.load(f)["summary"]["hosvd"]["per_seed"]}
    d = [abs(r["in_model"]["pooled_auroc"] - ref[r["seed"]]) for r in per_seed if r["seed"] in ref]
    if not d:
        return {"status": "not checked (no overlapping seeds)"}
    return {"status": "EXACT" if max(d) == 0.0 else "DIFFERS", "max_abs_diff": float(max(d))}


# ---------------------------------------------------------------------------------------------

def run(source, target, dataset, data_dir, cross_dir, out_dir, readout, seeds=None):
    m = mods()
    import methods.base as B
    c = B.canonical()
    s61 = m["s61"]
    seeds = [int(x) for x in (seeds or c["seeds"])]
    spec = m["s44"].CONDITION_SPECS[CONDITION]
    dst = os.path.join(out_dir, "cross_%s_to_%s_%s_%s.json" % (source, target, dataset, readout))
    s61._retry_os(lambda: os.makedirs(out_dir, exist_ok=True), "creating %s" % out_dir)

    def load(folder):
        f, y, pid, known = m["s44"].load_new_dataset(dataset, data_dir, folder, condition=CONDITION)
        return {"feats": f, "y": np.asarray(y, dtype=int), "pid": np.asarray(pid), "known": known}

    S, T = load(source), load(target)
    X = load_cross(cross_dir, source, target, dataset, T["y"], T["pid"])
    print("  [%s -> %s / %s] source %d answers, target %d answers | %s | readout %s | seeds %s"
          % (source, target, dataset, len(S["y"]), len(T["y"]), CONDITION, readout, seeds), flush=True)

    per_seed, t0 = [], time.time()
    for seed in seeds:
        t1 = time.time()
        r = score_seed(m, c, spec, S, T, X, seed, readout)
        r["seconds"] = round(time.time() - t1, 1)
        per_seed.append(r)
        print("    seed %-3d in-model %.4f | aligned %.4f | naive %.4f | proxy %.4f | map R2 cv %.3f test %.3f"
              " | source train -%d answers (%.0fs)"
              % (seed, r["in_model"]["pooled_auroc"], r["aligned"]["pooled_auroc"], r["naive"]["pooled_auroc"],
                 r["proxy"]["pooled_auroc"], r["alignment"]["cv_r2"], r["alignment"]["test_r2"],
                 r["source_train_removed"], r["seconds"]), flush=True)
        s61._write_result(dst, {"source": source, "target": target, "dataset": dataset, "readout": readout,
                                "condition": CONDITION, "seeds": seeds, "per_seed": per_seed,
                                "complete": False})

    summ = summarize(per_seed)
    ref = reference_check(per_seed, target, dataset, readout)
    print("\n  pooled AUROC on %s's test answers (mean +- std over %d seeds):" % (target, len(seeds)))
    for row in ROWS:
        print("    %-9s %.4f +- %.4f   within-question %s"
              % (row, summ[row]["pooled_mean"], summ[row]["pooled_std"],
                 "%.4f" % summ[row]["within_mean"] if summ[row]["within_mean"] is not None else "n/a"))
    print("  alignment R2: cv %.3f, test %.3f" % (summ["alignment_cv_r2_mean"], summ["alignment_test_r2_mean"]))
    print("  in-model reproduces the flatten control: %s" % json.dumps(ref))
    s61._write_result(dst, {"source": source, "target": target, "dataset": dataset, "readout": readout,
                            "condition": CONDITION, "seeds": seeds, "per_seed": per_seed, "summary": summ,
                            "reference_check": ref, "complete": True,
                            "elapsed_seconds": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------------------------------

def self_test():
    print("=" * 78)
    print("  SELF-TEST: 66_cross_model")
    print("=" * 78)
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(0)

    X = rng.standard_normal((300, 12))
    Y = X @ rng.standard_normal((12, 7)) + 0.1 * rng.standard_normal((300, 7))
    mr = ridge_fit(X, Y, 3.0)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    sk = Ridge(alpha=3.0).fit(Xs, Y)
    assert np.allclose(ridge_predict(mr, X), sk.predict(Xs), atol=1e-8)
    fa = fit_alignment(X, Y, np.repeat(np.arange(60), 5), 0)
    assert fa["cv_r2"] > 0.95, fa["cv_r2"]
    print("  [PASS] closed-form ridge equals sklearn's; CV picks a penalty with R2 %.3f on a planted map"
          % fa["cv_r2"])

    tr = np.arange(20)
    pid = np.repeat(np.arange(5), 4)
    assert list(exclude_questions(tr, pid, {1, 3})) == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]
    print("  [PASS] target test questions are removed from the source training rows")

    # End to end. One latent signal per answer; each model sees it through its own random embedding of
    # a different width, and the source's reading of a target answer embeds the SAME latent the target
    # holds. Aligned transfer must recover the signal; naive transfer must not.
    m = mods()
    import methods.base as B
    c = B.canonical()
    spec = [(key, r_l, 4) for key, r_l, _ in m["s44"].CONDITION_SPECS[CONDITION]]
    nq, per_q, k = 60, 5, 6

    def world(width, emb_seed):
        g = np.random.default_rng(emb_seed)
        return {key: (g.standard_normal((L, 1)), g.standard_normal((k, F)))
                for key, L, F in (("core", 9, width), ("static", 9, 2 * width), ("velocity", 8, 2 * width))}

    def render(latent, emb, noise_seed):
        g = np.random.default_rng(noise_seed)
        return {key: (np.einsum("nk,kf->nf", latent, B_)[:, None, :] * A[None, :, :]
                      + 0.5 * g.standard_normal((len(latent), A.shape[0], B_.shape[1]))).astype(np.float32)
                for key, (A, B_) in emb.items()}

    def model(width, emb_seed, noise_seed):
        pid = np.repeat(np.arange(nq), per_q)
        y = rng.integers(0, 2, size=nq * per_q)
        latent = rng.standard_normal((nq * per_q, k))
        latent[:, 0] += 2.5 * y
        known, _ = m["s44"].derive_is_known(y, pid)
        return latent, {"feats": render(latent, world(width, emb_seed), noise_seed), "y": y, "pid": pid,
                        "known": known}

    _, S = model(10, 1, 2)
    lat_t, T = model(8, 3, 4)
    X = render(lat_t, world(10, 1), 5)            # the source's embedding, the target's answers
    r = score_seed(m, c, spec, S, T, X, 42)
    assert r["in_model"]["pooled_auroc"] > 0.85, r
    assert r["aligned"]["pooled_auroc"] > 0.8 and r["proxy"]["pooled_auroc"] > 0.8, r
    assert r["aligned"]["pooled_auroc"] - r["naive"]["pooled_auroc"] > 0.15, r
    json.dumps(r)
    print("  [PASS] planted cross-model signal: in-model %.3f, aligned %.3f, proxy %.3f, naive %.3f "
          "(map R2 %.2f)" % (r["in_model"]["pooled_auroc"], r["aligned"]["pooled_auroc"],
                             r["proxy"]["pooled_auroc"], r["naive"]["pooled_auroc"], r["alignment"]["cv_r2"]))

    s = summarize([r, r])
    assert set(ROWS) <= set(s) and reference_check([r], "no-model", "no-ds", "RF")["status"].startswith("not")
    print("  [PASS] summary and reference check")
    print("\n  ALL PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--source")
    ap.add_argument("--target")
    ap.add_argument("--dataset", choices=["truthfulqa", "tydiqa_gp", "nq_open", "triviaqa"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--cross-dir", default=DEFAULT_CROSS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--readout", default="RF", choices=["RF", "LR"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not (a.source and a.target and a.dataset) or a.source == a.target:
        raise SystemExit("--source, --target (different models) and --dataset are required (or --self-test)")
    data_dir = a.data_dir
    if not data_dir:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]
    run(a.source, a.target, a.dataset, data_dir, a.cross_dir, a.out_dir, a.readout, a.seeds)


if __name__ == "__main__":
    main()
