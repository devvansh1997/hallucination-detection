"""
70_tensor_tucker.py -- a true 3-mode Tucker decomposition over layers x token bins x features (T-015).
==================================================================================================

THE CONCERN. The reported detector collapses the token axis with order statistics (peak, range, update)
before any decomposition, so what it decomposes is a layers x features MATRIX per answer: a multilinear
projection with two modes, not a tensor decomposition of the answer's representation. This ablation keeps
a token axis:

  1. POOL THE TOKEN AXIS TO A FIXED SIZE. An answer's completion tokens (T varies per answer) are split
     into T' contiguous bins of near-equal length (numpy.array_split), and each bin is averaged (--pool
     mean, default) or max-pooled (--pool max). An answer shorter than T' tokens is resampled by repeating
     tokens (bin k takes token floor(k*T/T')), so every answer yields the same size.
         state   X in R^{9 x T' x D}       hidden states, indices 16..24 (blocks 15..23)
         update  dX in R^{8 x T' x D}      h^{l+1} - h^{l} per token, then pooled the same way
  2. SCALE per (layer, bin, feature) entry, fitted on training answers: 33_eval_session04.robust_scale_3d
     applied to the tensor reshaped to (N, 9*T', D) -- the same winsorise / median / IQR / clip-6 rule.
  3. DECOMPOSE, label-free, on training answers only: truncated HOSVD in all three modes.
         U_L  (9 x R_L)   top eigenvectors of the layer-mode Gram       sum_n X_(1) X_(1)^T
         U_T  (T' x R_T)  top eigenvectors of the token-mode Gram       sum_n X_(2) X_(2)^T
         U_F  (D x R_F)   top right singular vectors of the feature-mode unfolding (randomized SVD)
     This is 43_eval_phase2.compute_ul_ud_randomized with the token mode added; with T' = 1 it reduces
     to the reported Tucker-2 step.
  4. CORE  G = X x_1 U_L^T x_2 U_T^T x_3 U_F^T in R^{R_L x R_T x R_F}, flattened.
  5. CLASSIFY with the reported random forest, question-level split, five draws.

SETTINGS, fixed before any result exists (TICKETS T-015): T' in {1, 2, 4, 8}; at each T' > 1 the token rank
R_T is both T'/2 (the token mode compressed) and T' (kept whole); R_L = 5 (4 for update), R_F = 64, the
reported ranks. Feature sets: state alone, and state + update. Full token rank is in the grid because the
label-free basis keeps the highest-ENERGY token bins, which need not be the informative ones: in the
self-test a signal planted in the second of two bins is invisible at R_T = 1 (the first bin has more
energy after scaling) and found at R_T = 2.

One job per T' (--settings), so the four pool sizes run in parallel; each writes its own file, named by
the T' values it ran. The bases U_L and U_F do not depend on R_T, so they are fitted once per draw and
every R_T at that T' reuses them.

  python 70_tensor_tucker.py --self-test
  python 70_tensor_tucker.py --dataset truthfulqa --model_folder llama-3.1-8b --settings 8:4,8:8
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.abspath(os.path.join(HERE, "..", "data-tokenstates"))
DEFAULT_OUT = os.path.join(HERE, "results", "tensor_tucker")
SETTINGS = [(1, 1), (2, 1), (2, 2), (4, 2), (4, 4), (8, 4), (8, 8)]      # (T', R_T)
R_L = {"state": 5, "update": 4}
R_F = 64
FEATURE_SETS = [("state",), ("state", "update")]
GRAM_CHUNK = 512


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
        _mods["s33"] = _load("s33", "33_eval_session04.py")
    return _mods


# ---------------------------------------------------------------------------------------------
# 1. token pooling
# ---------------------------------------------------------------------------------------------

def pool_tokens(x, n_bins, op="mean"):
    """x: (T, ...) one answer's tokens. Returns (n_bins, ...)."""
    T = x.shape[0]
    if T == 0:
        raise ValueError("an answer with no tokens cannot be pooled")
    if T < n_bins:
        return x[(np.arange(n_bins) * T) // n_bins]
    sizes = np.full(n_bins, T // n_bins)
    sizes[:T % n_bins] += 1                                  # numpy.array_split's bin sizes
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    if op == "mean":
        return np.add.reduceat(x, starts, axis=0) / sizes.reshape((-1,) + (1,) * (x.ndim - 1))
    if op == "max":
        return np.maximum.reduceat(x, starts, axis=0)
    raise ValueError(op)


def build_tensors(tokens, offsets, n_bins, op, streams):
    """-> {'state': (N, 9, T', D), 'update': (N, 8, T', D)} float32."""
    N = len(offsets) - 1
    _, L, D = tokens.shape
    out = {}
    if "state" in streams:
        out["state"] = np.empty((N, L, n_bins, D), dtype=np.float32)
    if "update" in streams:
        out["update"] = np.empty((N, L - 1, n_bins, D), dtype=np.float32)
    for n in range(N):
        h = np.asarray(tokens[offsets[n]:offsets[n + 1]], dtype=np.float32)          # (T, L, D)
        if "state" in out:
            out["state"][n] = pool_tokens(h, n_bins, op).transpose(1, 0, 2)
        if "update" in out:
            out["update"][n] = pool_tokens(h[:, 1:] - h[:, :-1], n_bins, op).transpose(1, 0, 2)
    return out


# ---------------------------------------------------------------------------------------------
# 2-4. scale, decompose, project
# ---------------------------------------------------------------------------------------------

def scale(m, X, tr):
    N, L, T, D = X.shape
    return m["s33"].robust_scale_3d(X.reshape(N, L * T, D), tr).reshape(N, L, T, D)


def mode_gram(X, mode):
    """Sum over answers of the mode-`mode` unfolding times its transpose (mode 1 = layers, 2 = token bins)."""
    k = X.shape[mode]
    G = np.zeros((k, k), dtype=np.float64)
    perm = (mode,) + tuple(i for i in range(4) if i != mode)
    for a in range(0, X.shape[0], GRAM_CHUNK):
        U = X[a:a + GRAM_CHUNK].astype(np.float64).transpose(perm).reshape(k, -1)
        G += U @ U.T
    return G


def top_eigvecs(G, r):
    _, V = np.linalg.eigh(G)
    return np.flip(V[:, -r:], axis=1).astype(np.float32).copy()


def hosvd3(X_train, r_l, r_t, r_f, seed):
    """Truncated HOSVD. U_T is returned with ALL T' columns ordered by energy; take U_T[:, :r_t]. Each mode is
    fitted separately, so truncating U_T does not change U_L or U_F."""
    from sklearn.utils.extmath import randomized_svd
    U_L = top_eigvecs(mode_gram(X_train, 1), r_l)
    U_T = top_eigvecs(mode_gram(X_train, 2), X_train.shape[2])
    N, L, T, D = X_train.shape
    _, _, Vt = randomized_svd(X_train.reshape(N * L * T, D), n_components=r_f, random_state=seed)
    return U_L, U_T[:, :r_t], Vt.T.astype(np.float32)


def project3(X, U_L, U_T, U_F):
    """(N, L, T', D) -> (N, R_L * R_T * R_F)."""
    t = X @ U_F                                             # (N, L, T', R_F)
    G = np.einsum("nltf,la,tb->nabf", t, U_L, U_T, optimize=True)
    return G.reshape(G.shape[0], -1)


def cores(m, tensors, tr, r_ts, seed):
    """-> {r_t: {stream: (N, R_L * r_t * R_F)}}. Scaling and U_L, U_F once per stream; one projection per r_t."""
    out = {r: {} for r in r_ts}
    for i, (key, X) in enumerate(tensors.items()):
        Xs = scale(m, X, tr)
        U_L, U_T, U_F = hosvd3(Xs[tr], R_L[key], X.shape[2], R_F, seed + i)
        for r in r_ts:
            out[r][key] = project3(Xs, U_L, U_T[:, :min(r, X.shape[2])], U_F)
        del Xs
    return out


# ---------------------------------------------------------------------------------------------
# data and driver
# ---------------------------------------------------------------------------------------------

def load_tokens(in_dir, model_folder, dataset):
    d = os.path.join(in_dir, model_folder, dataset)
    if not os.path.exists(os.path.join(d, "meta.json")):
        raise SystemExit("%s is missing or incomplete -- run 69_extract_token_states.py first" % d)
    meta = json.load(open(os.path.join(d, "meta.json")))
    if meta["nonfinite_entries"] or meta["empty_answers"]:
        raise SystemExit("token store has %d non-finite entries and %d empty answers -- handle before scoring"
                         % (meta["nonfinite_entries"], meta["empty_answers"]))
    return (np.load(os.path.join(d, "tokens.npy"), mmap_mode="r"), np.load(os.path.join(d, "offsets.npy")),
            np.load(os.path.join(d, "label.npy")), np.load(os.path.join(d, "prompt_id.npy")), meta)


def run(dataset, model_folder, data_dir, in_dir, out_dir, readout, op, settings, seeds=None):
    m = mods()
    import methods.base as B
    c = B.canonical()
    s61, s26 = m["s61"], m["s26"]
    seeds = [int(x) for x in (seeds or c["seeds"])]
    preregistered = set(settings) <= set(SETTINGS)
    dst = os.path.join(out_dir, output_name(model_folder, dataset, op, readout, settings))
    s61._retry_os(lambda: os.makedirs(out_dir, exist_ok=True), "creating %s" % out_dir)

    tokens, offsets, y, pid, meta = load_tokens(in_dir, model_folder, dataset)
    pinned = np.load(os.path.join(data_dir, model_folder, "%s_phase2_features.npz" % dataset))
    if not (np.array_equal(np.asarray(pinned["label"]).astype(int), y.astype(int))
            and np.array_equal(np.asarray(pinned["prompt_id"]).astype(np.int64), pid.astype(np.int64))):
        raise SystemExit("token store and pinned features disagree on labels or question ids")
    y, pid = y.astype(int), pid.astype(np.int64)
    is_known, _ = m["s44"].derive_is_known(y, pid)
    ref_path = os.path.join(HERE, "results", "flatten_control", "flatten_%s_%s_%s.json" % (model_folder, dataset, readout))
    reference = json.load(open(ref_path))["summary"]["hosvd"]["pooled_mean"] if os.path.exists(ref_path) else None
    print("  [%s/%s] %d answers, %d tokens | pool %s | settings %s%s | reported detector %s"
          % (model_folder, dataset, len(y), meta["total_tokens"], op, settings,
             "" if preregistered else " (NOT the pre-registered grid)",
             "%.4f" % reference if reference is not None else "n/a"), flush=True)

    res = {}
    if os.path.exists(dst):
        res = json.load(open(dst)).get("results", {})

    def payload(complete, more=None):
        d = {"dataset": dataset, "model_folder": model_folder, "readout": readout, "pool": op,
             "settings": [list(s) for s in settings], "preregistered_grid": preregistered, "R_L": R_L, "R_F": R_F,
             "seeds": seeds, "protocol": "question-level (paper protocol)", "reported_detector_pooled_mean": reference,
             "results": res, "complete": complete}
        d.update(more or {})
        return d

    t_start = time.time()
    for n_bins in sorted({t for t, _ in settings}):
        r_ts = sorted({r for t, r in settings if t == n_bins})
        keys = {(r, fs): setting_key(n_bins, r, fs) for r in r_ts for fs in FEATURE_SETS}
        if all(k in res for k in keys.values()):
            continue
        t0 = time.time()
        tensors = build_tensors(tokens, offsets, n_bins, op, ("state", "update"))
        print("    T'=%d (R_T %s): tensors built (%.0fs)" % (n_bins, r_ts, time.time() - t0), flush=True)
        rows = {k: [] for k in keys.values()}
        for seed in seeds:
            t1 = time.time()
            tr, te = (np.asarray(v, dtype=int) for v in c["question_split"](is_known, pid, len(y), seed))
            Z = cores(m, tensors, tr, r_ts, seed)
            for (r, fs), k in keys.items():
                F = np.concatenate([Z[r][s] for s in fs], axis=1)
                sc = s26.fit_eval(readout, F[tr], y[tr], F[te], seed)
                wp = c["within_prompt_auroc"](sc, y[te], pid[te])
                rows[k].append({"seed": seed, "pooled_auroc": float(c["pooled_auroc"](sc, y[te])),
                                "within_prompt_auroc": wp["within_prompt_auroc"], "width": int(F.shape[1])})
            print("      seed %-3d %s (%.0fs)" % (seed, " | ".join("%s %.4f" % (k, rows[k][-1]["pooled_auroc"])
                                                                   for k in keys.values()), time.time() - t1), flush=True)
        for (r, fs), k in keys.items():
            p = np.array([row["pooled_auroc"] for row in rows[k]])
            w = [row["within_prompt_auroc"] for row in rows[k] if row["within_prompt_auroc"] is not None]
            res[k] = {"T_bins": n_bins, "R_T": r, "features": "+".join(fs), "width": rows[k][0]["width"],
                      "pooled_mean": float(p.mean()), "pooled_std": float(p.std()),
                      "within_mean": float(np.mean(w)) if w else None, "per_seed": rows[k]}
            print("    %-28s width %-5d AUROC %.4f +- %.4f%s" % (k, res[k]["width"], p.mean(), p.std(),
                  "" if reference is None else "   (reported detector %.4f, %+.2f pts)" % (reference, 100 * (p.mean() - reference))),
                  flush=True)
        del tensors
        s61._write_result(dst, payload(False))
    s61._write_result(dst, payload(True, {"elapsed_seconds": round(time.time() - t_start, 1)}))


def setting_key(n_bins, r_t, feature_set):
    return "T%d_RT%d_%s" % (n_bins, r_t, "+".join(feature_set))


def output_name(model_folder, dataset, op, readout, settings):
    """One file per job, named by the T' values it ran, so parallel jobs never write the same file."""
    ts = "-".join(str(t) for t in sorted({t for t, _ in settings}))
    return "tucker3_%s_%s_%s_%s_T%s.json" % (model_folder, dataset, op, readout, ts)


# ---------------------------------------------------------------------------------------------

def self_test():
    print("=" * 78)
    print("  SELF-TEST: 70_tensor_tucker")
    print("=" * 78)
    x = np.arange(5, dtype=np.float32).reshape(5, 1)            # tokens 0..4
    assert pool_tokens(x, 2, "mean").ravel().tolist() == [1.0, 3.5]           # bins [0,1,2] and [3,4]
    assert pool_tokens(x, 2, "max").ravel().tolist() == [2.0, 4.0]
    assert pool_tokens(x, 1, "mean").ravel().tolist() == [2.0]
    assert pool_tokens(x, 5, "mean").ravel().tolist() == [0, 1, 2, 3, 4]
    assert pool_tokens(x[:3], 8, "mean").ravel().tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    print("  [PASS] pooling: array_split bins, mean and max, T'=1 is the token mean, short answers repeat tokens")

    # build_tensors: layer axis and update order. h[t, l, d] = 10*l + t, so the update is 10 everywhere
    # and the state's bin mean at layer l is 10*l + mean token index of the bin.
    T, L, D = 6, 9, 3
    h = (10.0 * np.arange(L)[None, :, None] + np.arange(T)[:, None, None]) * np.ones((1, 1, D))
    tokens = np.concatenate([h, h[:4]]).astype(np.float16)
    offs = np.array([0, 6, 10])
    ten = build_tensors(tokens, offs, 2, "mean", ("state", "update"))
    assert ten["state"].shape == (2, 9, 2, 3) and ten["update"].shape == (2, 8, 2, 3)
    assert np.allclose(ten["state"][0, :, 0, 0], 10 * np.arange(9) + 1.0)          # bin tokens 0,1,2
    assert np.allclose(ten["state"][1, :, 1, 0], 10 * np.arange(9) + 2.5)          # answer 2 bin tokens 2,3
    assert np.allclose(ten["update"], 10.0)
    print("  [PASS] tensors: (N, 9, T', D) state and (N, 8, T', D) update, per answer, layers in order")

    # HOSVD on a planted rank-(1,1,1) tensor recovers the three factors; T'=1 matches the Tucker-2 code.
    rng = np.random.default_rng(0)
    a, b, f = rng.standard_normal(9), rng.standard_normal(4), rng.standard_normal(20)
    a, b, f = a / np.linalg.norm(a), b / np.linalg.norm(b), f / np.linalg.norm(f)
    s = rng.standard_normal(200)[:, None, None, None] + 3.0
    X = (s * np.einsum("l,t,f->ltf", a, b, f)[None] + 0.01 * rng.standard_normal((200, 9, 4, 20))).astype(np.float32)
    U_L, U_T, U_F = hosvd3(X, 1, 1, 1, 0)
    assert abs(abs(U_L[:, 0] @ a) - 1) < 1e-3 and abs(abs(U_T[:, 0] @ b) - 1) < 1e-3 and abs(abs(U_F[:, 0] @ f) - 1) < 1e-3
    assert project3(X, U_L, U_T, U_F).shape == (200, 1)
    m = mods()
    X1 = X[:, :, :1, :]
    U_L1, U_T1, U_F1 = hosvd3(X1, 3, 1, 5, 0)
    UL2, UF2 = m["s43"].compute_ul_ud_randomized(X1[:, :, 0, :], 3, 5, seed=0)
    G3 = np.abs(project3(X1, U_L1, U_T1, U_F1))
    G2 = np.abs(m["s26"].project_core(X1[:, :, 0, :], UL2, UF2))
    assert np.allclose(G3, G2, atol=1e-3), np.abs(G3 - G2).max()
    print("  [PASS] 3-mode HOSVD recovers planted factors; with T'=1 the core equals the Tucker-2 core up to sign")

    # End to end through the reported forest on a planted signal that lives in the SECOND token bin only:
    # T'=2 must find it.
    import methods.base as B
    c = B.canonical()
    nq, per_q, D = 60, 5, 12
    n = nq * per_q
    pid = np.repeat(np.arange(nq), per_q)
    y = rng.integers(0, 2, size=n)
    lens = rng.integers(4, 9, size=n)
    offs = np.concatenate([[0], np.cumsum(lens)])
    toks = rng.standard_normal((offs[-1], 9, D)).astype(np.float32)
    u = rng.standard_normal(D)
    for i in range(n):
        T = lens[i]
        toks[offs[i] + (T + 1) // 2:offs[i + 1]] += (2.0 * y[i]) * u
    known, _ = m["s44"].derive_is_known(y, pid)
    tr, te = (np.asarray(v, dtype=int) for v in c["question_split"](known, pid, n, 42))
    ten = build_tensors(toks.astype(np.float16), offs, 2, "mean", ("state", "update"))
    global R_F
    saved, R_F = R_F, 4
    try:
        Z = cores(m, ten, tr, [1, 2], 42)
    finally:
        R_F = saved
    assert Z[1]["state"].shape == (n, 5 * 1 * 4) and Z[2]["state"].shape == (n, 5 * 2 * 4)
    assert Z[2]["update"].shape == (n, 4 * 2 * 4)
    aucs = {}
    for r in (1, 2):
        sc = m["s26"].fit_eval("RF", Z[r]["state"][tr], y[tr], Z[r]["state"][te], 42)
        aucs[r] = c["pooled_auroc"](sc, y[te])
    assert aucs[2] > 0.85, aucs
    print("  [PASS] end to end: planted second-half signal found at R_T = 2 through scaling, 3-mode HOSVD and the "
          "forest (AUROC %.3f); at R_T = 1 the energy-ranked token basis keeps the other bin (AUROC %.3f)"
          % (aucs[2], aucs[1]))

    assert setting_key(8, 4, ("state", "update")) == "T8_RT4_state+update"
    assert output_name("m", "d", "mean", "RF", [(8, 4), (8, 8)]) == "tucker3_m_d_mean_RF_T8.json"
    assert output_name("m", "d", "mean", "RF", SETTINGS) == "tucker3_m_d_mean_RF_T1-2-4-8.json"
    assert parse_settings("8:4,8:8") == [(8, 4), (8, 8)]
    print("  [PASS] per-job settings and file names (parallel jobs never share a file)")
    print("\n  ALL PASS")
    return True


def parse_settings(s):
    out = []
    for part in s.split(","):
        t, r = part.split(":")
        out.append((int(t), int(r)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dataset", choices=["truthfulqa", "tydiqa_gp", "nq_open", "triviaqa"])
    ap.add_argument("--model_folder")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--in-dir", default=DEFAULT_IN)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--readout", default="RF", choices=["RF", "LR"])
    ap.add_argument("--pool", default="mean", choices=["mean", "max"])
    ap.add_argument("--settings", default=None, help="T':R_T pairs, e.g. 8:8,8:2 (default: the pre-registered grid)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not (a.dataset and a.model_folder):
        raise SystemExit("--dataset and --model_folder are required (or --self-test)")
    data_dir = a.data_dir
    if not data_dir:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]
    settings = parse_settings(a.settings) if a.settings else list(SETTINGS)
    run(a.dataset, a.model_folder, data_dir, a.in_dir, a.out_dir, a.readout, a.pool, settings, a.seeds)


if __name__ == "__main__":
    main()
