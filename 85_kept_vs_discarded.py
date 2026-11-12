"""
85_kept_vs_discarded.py -- does the Tucker-2 projection keep the hallucination signal, or throw it away?
=====================================================================================================
THE QUESTION (ICLR 2027 reviews). zjPm: "Can the authors compare detection using the retained subspace
with detection using the discarded residual, to establish whether the low-rank projection preserves the
relevant discriminative information?" pjMe asks the same thing: "provide evidence that the learned
low-rank representations preserve hallucination-discriminative information".

THE TEST. For each summary (peak, quantile, cross-layer difference) the detector keeps a 5 x 64 window of
the scaled summary: G = U_L^T X U_F. Everything else is thrown away:

    R = X - (U_L U_L^T) X (U_F U_F^T)          the discarded residual

Four arms, identical in everything except which part of X the classifier sees:

  kept                 the reported detector: the Tucker-2 core of X. Bit-identical to 61's hosvd arm.
  discarded            the Tucker-2 core of R, at the SAME ranks, so the same 896 features -- the
                       best same-size description of what the projection threw away.
  discarded_random     a Gaussian random projection of R to the same width. A second, assumption-free
                       look at the residual.
  kept_plus_discarded  both cores side by side (1,792 features): does the residual add anything on top?

WHAT EACH OUTCOME MEANS.
  kept >> discarded, and kept_plus_discarded ~ kept
        the projection keeps the signal and the residual adds nothing: direct evidence for the claim.
  kept ~ discarded
        the signal is spread across the representation; any reasonable slice carries it. Report it, and
        soften the claim that the low-rank directions are the discriminative ones.
  kept_plus_discarded >> kept
        the projection discards useful signal. Report it.

Everything else is the paper's protocol: triple_concat, ranks (5,64)/(5,64)/(4,64), robust scaling and
bases fitted on training rows only (no labels), random forest, question-level split, seeds {42,0,1,2,3}.

  python 85_kept_vs_discarded.py --self-test
  python 85_kept_vs_discarded.py --dataset truthfulqa --model_folder qwen-2.5-7b-instruct
"""

import argparse
import importlib.util
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "rebuttal")
CONDITION = "triple_concat"
ARMS = ("kept", "discarded", "discarded_random", "kept_plus_discarded")
CHUNK = 2048            # answers per chunk when forming the residual


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


def residual(Xs, U_L, U_D, chunk=CHUNK):
    """R = X - P_L X P_F per answer, with P_L = U_L U_L^T and P_F = U_D U_D^T. Chunked so that no
    (N, L, F) temporary beyond R itself is ever allocated."""
    P_L = (U_L @ U_L.T).astype(np.float32)
    U_D = U_D.astype(np.float32)
    R = np.empty(Xs.shape, dtype=np.float32)
    for a in range(0, len(Xs), chunk):
        x = np.asarray(Xs[a:a + chunk], dtype=np.float32)
        R[a:a + chunk] = x - np.einsum("lm,nmd->nld", P_L, (x @ U_D) @ U_D.T)
    return R


def split_block(X_raw, tr, r_l, r_d, seed):
    """One summary tensor -> {arm: (N, width)} for the three single-part arms.

    The kept core is computed with exactly the calls the pipeline makes (43.fold_pure_core_randomized:
    robust_scale_3d, compute_ul_ud_randomized on the training rows, project_core), so it is the
    reported feature, not a re-implementation of it."""
    m = mods()
    Xs = m["s43"].robust_scale_3d(X_raw, tr)
    U_L, U_D = m["s43"].compute_ul_ud_randomized(Xs[tr], r_l, r_d, seed)
    kept = m["s26"].project_core(Xs, U_L, U_D)

    R = residual(Xs, U_L, U_D)
    del Xs
    U_L2, U_D2 = m["s43"].compute_ul_ud_randomized(R[tr], r_l, r_d, seed)
    disc = m["s26"].project_core(R, U_L2, U_D2)

    k = r_l * r_d
    rng = np.random.default_rng(seed)
    Rf = R.reshape(len(R), -1)
    G = rng.standard_normal((Rf.shape[1], k)).astype(np.float32) / np.sqrt(k)
    disc_rand = Rf @ G
    return {"kept": kept.astype(np.float32), "discarded": disc.astype(np.float32),
            "discarded_random": disc_rand.astype(np.float32)}


def build_all(feats, spec, tr, seed):
    """All four arms for one split. Block i uses seed + i, as 61.build does, so `kept` reproduces it."""
    parts = {a: [] for a in ("kept", "discarded", "discarded_random")}
    for i, (key, r_l, r_d) in enumerate(spec):
        b = split_block(feats[key], tr, r_l, r_d, seed + i)
        for a in parts:
            parts[a].append(b[a])
    Z = {a: np.concatenate(v, axis=1) for a, v in parts.items()}
    Z["kept_plus_discarded"] = np.concatenate([Z["kept"], Z["discarded"]], axis=1)
    return Z


def score_arm(Z, tr, te, y, pid, readout, seed, c):
    m = mods()
    t = time.time()
    s = m["s26"].fit_eval(readout, Z[tr], y[tr], Z[te], seed)
    wp = c["within_prompt_auroc"](s, y[te], pid[te])
    return {"seed": int(seed), "dim": int(Z.shape[1]),
            "pooled_auroc": float(c["pooled_auroc"](s, y[te])),
            "within_prompt_auroc": wp["within_prompt_auroc"], "n_pairs": wp["n_pairs"],
            "seconds": round(time.time() - t, 1)}


def summarize(res):
    out = {}
    for arm, rows in res.items():
        if not rows:
            continue
        p = np.array([r["pooled_auroc"] for r in rows])
        w = [r["within_prompt_auroc"] for r in rows if r["within_prompt_auroc"] is not None]
        out[arm] = {"pooled_mean": float(p.mean()), "pooled_std": float(p.std()),
                    "within_mean": float(np.mean(w)) if w else None, "dim": rows[0]["dim"],
                    "n_seeds": len(rows), "per_seed": rows}
    return out


def verdict(summ):
    if not all(a in summ for a in ("kept", "discarded", "kept_plus_discarded")):
        return "incomplete"
    k = summ["kept"]["pooled_mean"]
    d = summ["discarded"]["pooled_mean"]
    kd = summ["kept_plus_discarded"]["pooled_mean"]
    return ("kept %.2f vs discarded %.2f (kept ahead by %+.2f points); adding the discarded part "
            "changes kept by %+.2f points" % (100 * k, 100 * d, 100 * (k - d), 100 * (kd - k)))


def run(dataset, model_folder, data_dir, out_dir, readout, seeds=None):
    m = mods()
    s61 = m["s61"]
    import methods.base as B
    c = B.canonical()
    seeds = [int(s) for s in (seeds or c["seeds"])]
    spec = m["s44"].CONDITION_SPECS[CONDITION]
    dst = os.path.join(out_dir, "kept_vs_discarded_%s_%s_%s.json" % (model_folder, dataset, readout))
    s61._retry_os(lambda: os.makedirs(out_dir, exist_ok=True), "creating %s" % out_dir)

    feats, y, pid, is_known = m["s44"].load_new_dataset(dataset, data_dir, model_folder,
                                                         condition=CONDITION)
    y, pid = np.asarray(y, dtype=int), np.asarray(pid)
    n = len(y)
    print("  [%s/%s] %d answers, %d questions, %.1f%% hallucinated | %s | readout %s | seeds %s"
          % (model_folder, dataset, n, len(np.unique(pid)), 100.0 * y.mean(), CONDITION, readout,
             seeds), flush=True)

    res = {a: [] for a in ARMS}
    t0 = time.time()
    for seed in seeds:
        tr, te = (np.asarray(v, dtype=int) for v in c["question_split"](is_known, pid, n, seed))
        t1 = time.time()
        Z = build_all(feats, spec, tr, seed)
        t_build = time.time() - t1
        for arm in ARMS:
            r = score_arm(Z[arm], tr, te, y, pid, readout, seed, c)
            res[arm].append(r)
            print("    seed %-3d %-20s dim %4d  pooled %.4f  within %s  (%.0fs)"
                  % (seed, arm, r["dim"], r["pooled_auroc"],
                     "n/a" if r["within_prompt_auroc"] is None else "%.4f" % r["within_prompt_auroc"],
                     r["seconds"]), flush=True)
        print("    seed %-3d projections built in %.0fs" % (seed, t_build), flush=True)
        del Z
        summ = summarize(res)
        s61._write_result(dst, {"dataset": dataset, "model_folder": model_folder, "readout": readout,
                                "condition": CONDITION, "seeds": seeds, "arms": list(ARMS),
                                "protocol": "question-level split, paper protocol",
                                "summary": summ, "verdict": verdict(summ),
                                "complete": len(res["kept"]) == len(seeds),
                                "elapsed_seconds": round(time.time() - t0, 1)})

    summ = summarize(res)
    print("\n  pooled AUROC (mean +- std over %d seeds), within-question in brackets:" % len(seeds))
    for arm in ARMS:
        s = summ[arm]
        print("    %-20s %.4f +- %.4f  [%s]  dim %d"
              % (arm, s["pooled_mean"], s["pooled_std"],
                 "n/a" if s["within_mean"] is None else "%.4f" % s["within_mean"], s["dim"]))
    print("  VERDICT: %s" % verdict(summ))

    # The kept arm IS the reported detector. If a flatten-control file exists for this cell, its hosvd
    # arm must match seed for seed; say so, so a reader can trust the other arms' baseline.
    ref = os.path.join(HERE, "results", "flatten_control",
                       "flatten_%s_%s_%s.json" % (model_folder, dataset, readout))
    if os.path.exists(ref):
        import json
        with open(ref) as f:
            old = {r["seed"]: r["pooled_auroc"] for r in json.load(f)["summary"]["hosvd"]["per_seed"]}
        d = [abs(r["pooled_auroc"] - old[r["seed"]]) for r in res["kept"] if r["seed"] in old]
        print("  kept reproduces the flatten control: %s" % (
            "EXACT" if d and max(d) == 0.0 else "max diff %.2e" % max(d) if d else "no overlap"))
    return summ


# ---------------------------------------------------------------------------------------------

def self_test():
    print("=" * 78)
    print("  SELF-TEST: 85_kept_vs_discarded")
    print("=" * 78)
    from sklearn.metrics import roc_auc_score
    m = mods()
    rng = np.random.default_rng(0)
    n, L, D = 400, 6, 40
    tr, te = np.arange(0, 300), np.arange(300, n)

    # 1. The kept arm is the reported detector, bit for bit.
    X = rng.standard_normal((n, L, D)).astype(np.float32)
    b = split_block(X, tr, 3, 5, 7)
    ref = m["s61"].reduce_hosvd(X, tr, 3, 5, 7)
    assert np.array_equal(b["kept"], ref.astype(np.float32)), "kept differs from 61's hosvd arm"
    print("  [PASS] kept is bit-identical to 61's hosvd arm (the reported detector)")

    # 2. Widths: every single-part arm emits r_l * r_d features.
    assert b["kept"].shape[1] == b["discarded"].shape[1] == b["discarded_random"].shape[1] == 15
    print("  [PASS] kept, discarded and discarded_random all emit width r_l * r_d = 15")

    # 3. The residual has nothing left in the kept subspace.
    Xs = m["s43"].robust_scale_3d(X, tr)
    U_L, U_D = m["s43"].compute_ul_ud_randomized(Xs[tr], 3, 5, 7)
    R = residual(Xs, U_L, U_D, chunk=37)          # an odd chunk size exercises the chunk boundary
    left = m["s26"].project_core(R, U_L, U_D)
    scale = np.abs(m["s26"].project_core(Xs, U_L, U_D)).max()
    assert np.abs(left).max() < 1e-4 * scale, np.abs(left).max()
    assert np.allclose(R + np.einsum("lm,nmd->nld", U_L @ U_L.T, (Xs @ U_D) @ U_D.T), Xs, atol=1e-4)
    print("  [PASS] the residual is orthogonal to the kept subspace and kept + residual = X")

    # 4. No leakage: appending unseen rows must not change any arm on the original rows.
    X2 = np.concatenate([X, rng.standard_normal((50, L, D)).astype(np.float32)])
    b2 = split_block(X2, tr, 3, 5, 7)
    for a in b:
        assert np.allclose(b[a], b2[a][:n], atol=1e-4), "%s fits on more than the training rows" % a
    print("  [PASS] every arm fits on training rows only (unseen rows leave them unchanged)")

    def aucs(Xk, y):
        out = {}
        bb = split_block(Xk, tr, 2, 4, 0)
        for a in ("kept", "discarded"):
            s = m["s26"].fit_eval("LR", bb[a][tr], y[tr], bb[a][te], 0)
            out[a] = roc_auc_score(y[te], s)
        return out

    y = (np.arange(n) % 2).astype(int)

    # 5. Signal in the dominant structure: kept must beat discarded.
    ul = rng.standard_normal(L); ul /= np.linalg.norm(ul)
    uf = rng.standard_normal(D); uf /= np.linalg.norm(uf)
    Xa = rng.standard_normal((n, L, D)).astype(np.float32) * 0.5
    Xa += (y[:, None, None] * 4.0 - 2.0) * np.outer(ul, uf)[None].astype(np.float32)
    a = aucs(Xa, y)
    assert a["kept"] > a["discarded"] + 0.1, a
    print("  [PASS] signal in the dominant directions: kept %.3f > discarded %.3f"
          % (a["kept"], a["discarded"]))

    # 6. Signal outside the kept subspace: discarded must beat kept. Strong unlabelled patterns fill
    # the 2 x 4 window exactly (two layer directions x four feature directions), so the kept bases
    # land on them; a weak labelled pattern sits in a feature direction orthogonal to all four, so
    # the projection removes it and only the residual carries it. If this failed, the test could not
    # tell a projection that throws the signal away from one that keeps it.
    Q, _ = np.linalg.qr(rng.standard_normal((D, 5)))
    nuis_f, sig_f = Q[:, :4], Q[:, 4]
    Ql, _ = np.linalg.qr(rng.standard_normal((L, L)))
    Xb = rng.standard_normal((n, L, D)).astype(np.float32) * 0.05
    for j in range(4):
        coef = rng.standard_normal(n) * 3.0
        Xb += (coef[:, None, None] * np.outer(Ql[:, j % 2], nuis_f[:, j])[None]).astype(np.float32)
    Xb += ((y[:, None, None] * 1.0 - 0.5) * np.outer(Ql[:, 0], sig_f)[None]).astype(np.float32)
    a = aucs(Xb, y)
    assert a["discarded"] > a["kept"] + 0.1, a
    print("  [PASS] signal outside the kept window: discarded %.3f > kept %.3f -- the test can see "
          "a projection that throws the signal away" % (a["discarded"], a["kept"]))

    # 7. The four-arm builder concatenates blocks the way 61.build does.
    feats = {"core": X[:, :4, :], "static": X[:, :, :20]}
    spec = [("core", 2, 3), ("static", 2, 4)]
    Z = build_all(feats, spec, tr, 5)
    assert Z["kept"].shape[1] == 6 + 8 and Z["kept_plus_discarded"].shape[1] == 2 * 14
    assert np.array_equal(Z["kept"], m["s61"].build("hosvd", feats, spec, tr, 5).astype(np.float32))
    print("  [PASS] build_all matches 61.build for the kept arm across blocks (seed + i per block)")

    v = verdict({"kept": {"pooled_mean": 0.88}, "discarded": {"pooled_mean": 0.80},
                 "kept_plus_discarded": {"pooled_mean": 0.881}})
    assert "+8.00" in v and "+0.10" in v, v
    print("  [PASS] verdict reports kept-minus-discarded and the residual's added value")

    print("\n  ALL PASS")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dataset")
    p.add_argument("--model_folder")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--readout", default="RF", choices=["RF", "LR"])
    p.add_argument("--seeds", nargs="+", type=int, default=None)
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
    run(a.dataset, a.model_folder, data_dir, a.out_dir, a.readout, a.seeds)


if __name__ == "__main__":
    main()
