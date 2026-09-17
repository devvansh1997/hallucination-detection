"""
70_tensor_tucker.py -- a true 3-mode Tucker decomposition over layers x token bins x features (T-015).
==================================================================================================

THE CONCERN. The reported detector collapses the token axis with order statistics (peak, range, update over
all of an answer's tokens) before any decomposition, so what it decomposes is a layers x features MATRIX per
answer: a two-mode multilinear projection, not a decomposition of a tensor. This ablation keeps a token axis.

PIPELINE, one answer (D = hidden size):
  1. BIN THE TOKEN AXIS. The answer's T completion tokens are split into T' contiguous bins of near-equal
     length (numpy.array_split sizes). An answer shorter than T' repeats tokens (bin k takes token
     floor(k*T/T')), so every answer yields the same T'.
  2. SUMMARISE EACH BIN. Two families (--family), one per job:
       orderstats  the reported summaries, computed within each bin instead of over the whole answer:
                     peak    (9,  T', D)   max over the bin's tokens
                     range   (9,  T', 2D)  q95 and q05 over the bin's tokens
                     update  (8,  T', 2D)  q95 and q05 over the bin's tokens of h^{l+1} - h^{l}
                   With T' = 1 this is EXACTLY the reported triple_concat input, so the T' = 1 row must
                   reproduce the reported detector (results/flatten_control) up to extraction noise; the
                   job reports the difference as its anchor check. T' > 1 then changes one thing only:
                   the token axis is kept and decomposed.
       mean        fixed-window average pooling, the literal reading of the proposal:
                     state   (9, T', D)    mean over the bin's tokens
                     update  (8, T', D)    mean over the bin's tokens of h^{l+1} - h^{l}
  3. SCALE per (layer, bin, feature) entry, fitted on training answers: 28_eval_band.fit_robust_scale /
     apply_robust_scale, the rule inside 33_eval_session04.robust_scale_3d, applied in chunks.
  4. DECOMPOSE, label-free, training answers only: truncated HOSVD in all three modes -- U_L (9 x R_L) and
     U_T (T' x R_T) top eigenvectors of the layer- and token-mode Grams, U_F (D x R_F) randomized SVD of the
     feature-mode unfolding. 43_eval_phase2.compute_ul_ud_randomized with the token mode added; seeds per
     stream follow 61_flatten_control.build (seed + stream index).
  5. CORE G = X x_1 U_L^T x_2 U_T^T x_3 U_F^T in R^{R_L x R_T x R_F}, flattened per stream, concatenated.
  6. CLASSIFY with the reported random forest; question-level split; five draws.

GRID, fixed before any result exists (TICKETS T-015): T' in {1, 2, 4, 8}; R_T in {T'/2, T'} (T' = 1: 1);
R_L = 5 (4 for update); R_F = 64. Full token rank is in the grid because the energy-ranked token basis can
drop the informative bin (self-test: a signal planted in the second of two bins scores 0.53 at R_T = 1 and
1.00 at R_T = 2). One job per (family, T'); --settings runs that T'.

  python 70_tensor_tucker.py --self-test
  python 70_tensor_tucker.py --dataset truthfulqa --model_folder llama-3.1-8b --family orderstats --settings 8:4,8:8
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
FAMILIES = {"orderstats": ("peak", "range", "update"), "mean": ("state", "update")}
R_L = {"peak": 5, "range": 5, "state": 5, "update": 4}
R_F = 64
GRAM_CHUNK = 512
SCALE_CHUNK = 1024
ANCHOR_TOL_PTS = 0.5


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
        _mods["s28"] = _load("s28", "28_eval_band.py")
    return _mods


# ---------------------------------------------------------------------------------------------
# 1-2. bins and summaries
# ---------------------------------------------------------------------------------------------

def bin_bounds(T, n_bins):
    """[(start, end)] per bin, numpy.array_split sizes; for T < n_bins, one repeated token per bin."""
    if T == 0:
        raise ValueError("an answer with no tokens cannot be binned")
    if T < n_bins:
        idx = (np.arange(n_bins) * T) // n_bins
        return [(int(i), int(i) + 1) for i in idx]
    sizes = np.full(n_bins, T // n_bins)
    sizes[:T % n_bins] += 1
    ends = np.cumsum(sizes)
    return [(int(e - s), int(e)) for s, e in zip(sizes, ends)]


def pool_tokens(x, n_bins, op):
    """x: (T, ...) one answer's tokens -> (n_bins, ...). op in mean, max, q95, q05 (linear interpolation, as
    torch.quantile in 32_extract_velocity)."""
    parts = []
    for s, e in bin_bounds(x.shape[0], n_bins):
        seg = x[s:e]
        if op == "mean":
            parts.append(seg.mean(axis=0))
        elif op == "max":
            parts.append(seg.max(axis=0))
        elif op == "q95":
            parts.append(np.quantile(seg, 0.95, axis=0))
        elif op == "q05":
            parts.append(np.quantile(seg, 0.05, axis=0))
        else:
            raise ValueError(op)
    return np.stack(parts)


def summarise(h, n_bins, family):
    """h: (T, 9, D) one answer -> {stream: (L, T', F)} for the family."""
    d = h[:, 1:] - h[:, :-1]                                              # (T, 8, D)
    t = lambda a: a.transpose(1, 0, 2)                                    # noqa: E731  (T', L, F) -> (L, T', F)
    if family == "orderstats":
        return {"peak": t(pool_tokens(h, n_bins, "max")),
                "range": t(np.concatenate([pool_tokens(h, n_bins, "q95"), pool_tokens(h, n_bins, "q05")], axis=2)),
                "update": t(np.concatenate([pool_tokens(d, n_bins, "q95"), pool_tokens(d, n_bins, "q05")], axis=2))}
    if family == "mean":
        return {"state": t(pool_tokens(h, n_bins, "mean")), "update": t(pool_tokens(d, n_bins, "mean"))}
    raise ValueError(family)


def build_tensors(tokens, offsets, n_bins, family):
    """-> {stream: (N, L, T', F) float16}, streams in FAMILIES order."""
    N = len(offsets) - 1
    _, L, D = tokens.shape
    widths = {"peak": D, "state": D, "range": 2 * D, "update": (2 * D if family == "orderstats" else D)}
    out = {s: np.empty((N, 8 if s == "update" else L, n_bins, widths[s]), dtype=np.float16) for s in FAMILIES[family]}
    for n in range(N):
        h = np.asarray(tokens[offsets[n]:offsets[n + 1]], dtype=np.float32)
        for s, v in summarise(h, n_bins, family).items():
            out[s][n] = v
    return out


# ---------------------------------------------------------------------------------------------
# 3-5. scale, decompose, project
# ---------------------------------------------------------------------------------------------

def scale(m, X, tr):
    """robust_scale_3d's rule on (N, L*T', F), fitted on training rows, applied in chunks to bound memory."""
    N, L, T, F = X.shape
    flat = X.reshape(N, L * T, F)
    params = m["s28"].fit_robust_scale(flat[tr].reshape(len(tr), -1).astype(np.float32))
    out = np.empty((N, L, T, F), dtype=np.float32)
    for a in range(0, N, SCALE_CHUNK):
        chunk = flat[a:a + SCALE_CHUNK].reshape(-1, L * T * F).astype(np.float32)
        out[a:a + SCALE_CHUNK] = m["s28"].apply_robust_scale(chunk, params).reshape(-1, L, T, F)
    return out


def mode_gram(X, mode):
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
    """Truncated HOSVD; each mode fitted separately, so truncating U_T does not change U_L or U_F."""
    from sklearn.utils.extmath import randomized_svd
    U_L = top_eigvecs(mode_gram(X_train, 1), r_l)
    U_T = top_eigvecs(mode_gram(X_train, 2), X_train.shape[2])
    N, L, T, F = X_train.shape
    _, _, Vt = randomized_svd(X_train.reshape(N * L * T, F), n_components=r_f, random_state=seed)
    return U_L, U_T[:, :r_t], Vt.T.astype(np.float32)


def project3(X, U_L, U_T, U_F):
    """(N, L, T', F) -> (N, R_L * R_T * R_F)."""
    t = X @ U_F
    G = np.einsum("nltf,la,tb->nabf", t, U_L, U_T, optimize=True)
    return G.reshape(G.shape[0], -1)


def cores(m, tensors, tr, r_ts, seed):
    """-> {r_t: {stream: core}}. Scaling, U_L and U_F once per stream; one projection per r_t."""
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


def setting_key(n_bins, r_t, family):
    return "T%d_RT%d_%s" % (n_bins, r_t, family)


def output_name(model_folder, dataset, family, readout, settings):
    ts = "-".join(str(t) for t in sorted({t for t, _ in settings}))
    return "tucker3_%s_%s_%s_%s_T%s.json" % (model_folder, dataset, family, readout, ts)


def anchor(res, family, reference):
    """orderstats at T'=1, R_T=1 is the reported detector's input and decomposition."""
    k = setting_key(1, 1, "orderstats")
    if family != "orderstats" or k not in res or reference is None:
        return None
    diff = 100.0 * (res[k]["pooled_mean"] - reference)
    return {"diff_pts": round(diff, 3), "reference": reference, "rebuilt": res[k]["pooled_mean"],
            "within_tolerance": bool(abs(diff) <= ANCHOR_TOL_PTS), "tolerance_pts": ANCHOR_TOL_PTS}


def run(dataset, model_folder, data_dir, in_dir, out_dir, readout, family, settings, seeds=None):
    m = mods()
    import methods.base as B
    c = B.canonical()
    s61, s26 = m["s61"], m["s26"]
    seeds = [int(x) for x in (seeds or c["seeds"])]
    preregistered = set(settings) <= set(SETTINGS)
    dst = os.path.join(out_dir, output_name(model_folder, dataset, family, readout, settings))
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
    print("  [%s/%s] %d answers, %d tokens | family %s %s | settings %s%s | reported detector %s"
          % (model_folder, dataset, len(y), meta["total_tokens"], family, FAMILIES[family], settings,
             "" if preregistered else " (NOT the pre-registered grid)",
             "%.4f" % reference if reference is not None else "n/a"), flush=True)

    res = json.load(open(dst)).get("results", {}) if os.path.exists(dst) else {}

    def payload(complete, more=None):
        d = {"dataset": dataset, "model_folder": model_folder, "readout": readout, "family": family,
             "streams": list(FAMILIES[family]), "settings": [list(s) for s in settings],
             "preregistered_grid": preregistered, "R_L": R_L, "R_F": R_F, "seeds": seeds,
             "protocol": "question-level (paper protocol)", "reported_detector_pooled_mean": reference,
             "anchor_T1": anchor(res, family, reference), "results": res, "complete": complete}
        d.update(more or {})
        return d

    t_start = time.time()
    for n_bins in sorted({t for t, _ in settings}):
        r_ts = sorted({r for t, r in settings if t == n_bins})
        keys = {r: setting_key(n_bins, r, family) for r in r_ts}
        if all(k in res for k in keys.values()):
            continue
        t0 = time.time()
        tensors = build_tensors(tokens, offsets, n_bins, family)
        print("    T'=%d (R_T %s): %s built (%.0fs)" % (n_bins, r_ts, {s: X.shape for s, X in tensors.items()},
                                                          time.time() - t0), flush=True)
        rows = {k: [] for k in keys.values()}
        for seed in seeds:
            t1 = time.time()
            tr, te = (np.asarray(v, dtype=int) for v in c["question_split"](is_known, pid, len(y), seed))
            Z = cores(m, tensors, tr, r_ts, seed)
            for r, k in keys.items():
                F = np.concatenate([Z[r][s] for s in FAMILIES[family]], axis=1)
                sc = s26.fit_eval(readout, F[tr], y[tr], F[te], seed)
                wp = c["within_prompt_auroc"](sc, y[te], pid[te])
                rows[k].append({"seed": seed, "pooled_auroc": float(c["pooled_auroc"](sc, y[te])),
                                "within_prompt_auroc": wp["within_prompt_auroc"], "width": int(F.shape[1])})
            print("      seed %-3d %s (%.0fs)" % (seed, " | ".join("%s %.4f" % (k, rows[k][-1]["pooled_auroc"])
                                                                   for k in keys.values()), time.time() - t1), flush=True)
        for r, k in keys.items():
            p = np.array([row["pooled_auroc"] for row in rows[k]])
            w = [row["within_prompt_auroc"] for row in rows[k] if row["within_prompt_auroc"] is not None]
            res[k] = {"T_bins": n_bins, "R_T": r, "family": family, "width": rows[k][0]["width"],
                      "pooled_mean": float(p.mean()), "pooled_std": float(p.std()),
                      "within_mean": float(np.mean(w)) if w else None, "per_seed": rows[k]}
            print("    %-24s width %-5d AUROC %.4f +- %.4f%s" % (k, res[k]["width"], p.mean(), p.std(),
                  "" if reference is None else "   (reported detector %.4f, %+.2f pts)" % (reference, 100 * (p.mean() - reference))),
                  flush=True)
        del tensors
        a = anchor(res, family, reference)
        if a is not None and n_bins == 1:
            print("    ANCHOR (orderstats, T'=1 is the reported detector): %s" % json.dumps(a), flush=True)
        s61._write_result(dst, payload(False))
    s61._write_result(dst, payload(True, {"elapsed_seconds": round(time.time() - t_start, 1)}))


# ---------------------------------------------------------------------------------------------

def self_test():
    print("=" * 78)
    print("  SELF-TEST: 70_tensor_tucker")
    print("=" * 78)
    assert bin_bounds(5, 2) == [(0, 3), (3, 5)] and bin_bounds(3, 8) == [(0, 1)] * 3 + [(1, 2)] * 3 + [(2, 3)] * 2
    x = np.arange(5, dtype=np.float32).reshape(5, 1)
    assert pool_tokens(x, 2, "mean").ravel().tolist() == [1.0, 3.5]
    assert pool_tokens(x, 2, "max").ravel().tolist() == [2.0, 4.0]
    assert np.allclose(pool_tokens(x, 1, "q95").ravel(), np.quantile(x, 0.95))
    assert np.allclose(pool_tokens(x, 2, "q05").ravel(), [0.1, 3.05])
    assert pool_tokens(x[:3], 8, "max").ravel().tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    print("  [PASS] bins (array_split sizes; short answers repeat tokens) and mean/max/q95/q05 within bins")

    # orderstats at T'=1 are the reported summaries over the whole answer (35_derive_streams conventions).
    rng = np.random.default_rng(0)
    h = rng.standard_normal((7, 9, 5)).astype(np.float32)
    s = summarise(h, 1, "orderstats")
    d = h[:, 1:] - h[:, :-1]
    assert np.allclose(s["peak"][:, 0], h.max(axis=0))
    assert np.allclose(s["range"][:, 0], np.concatenate([np.quantile(h, 0.95, axis=0), np.quantile(h, 0.05, axis=0)], axis=1))
    assert np.allclose(s["update"][:, 0], np.concatenate([np.quantile(d, 0.95, axis=0), np.quantile(d, 0.05, axis=0)], axis=1))
    assert s["peak"].shape == (9, 1, 5) and s["range"].shape == (9, 1, 10) and s["update"].shape == (8, 1, 10)
    sm = summarise(h, 4, "mean")
    assert sm["state"].shape == (9, 4, 5) and sm["update"].shape == (8, 4, 5)
    toks = np.concatenate([h, h[:3]]).astype(np.float16)
    ten = build_tensors(toks, np.array([0, 7, 10]), 1, "orderstats")
    assert list(ten) == ["peak", "range", "update"] and ten["range"].shape == (2, 9, 1, 10)
    assert np.allclose(ten["peak"][0, :, 0], h.astype(np.float16).astype(np.float32).max(axis=0), atol=1e-3)
    print("  [PASS] orderstats at T'=1 = peak (max), range (q95|q05), update (q95|q05 of layer differences) "
          "over the whole answer; tensors (N, L, T', F) in FAMILIES order")

    m = mods()
    Xs = rng.standard_normal((40, 9, 3, 6)).astype(np.float32) * 3 + 1
    tr = np.arange(30)
    ref = m["s43"].robust_scale_3d(Xs.reshape(40, 27, 6), tr).reshape(40, 9, 3, 6)
    global SCALE_CHUNK
    saved_chunk, SCALE_CHUNK = SCALE_CHUNK, 7
    try:
        assert np.allclose(scale(m, Xs.astype(np.float16), tr), m["s43"].robust_scale_3d(
            Xs.astype(np.float16).astype(np.float32).reshape(40, 27, 6), tr).reshape(40, 9, 3, 6), atol=1e-5)
    finally:
        SCALE_CHUNK = saved_chunk
    assert ref.shape == Xs.shape
    print("  [PASS] chunked scaling equals robust_scale_3d on the (N, L*T', F) reshape")

    a, b, f = rng.standard_normal(9), rng.standard_normal(4), rng.standard_normal(20)
    a, b, f = a / np.linalg.norm(a), b / np.linalg.norm(b), f / np.linalg.norm(f)
    X = (rng.standard_normal(200)[:, None, None, None] + 3.0) * np.einsum("l,t,f->ltf", a, b, f)[None]
    X = (X + 0.01 * rng.standard_normal((200, 9, 4, 20))).astype(np.float32)
    U_L, U_T, U_F = hosvd3(X, 1, 1, 1, 0)
    assert abs(abs(U_L[:, 0] @ a) - 1) < 1e-3 and abs(abs(U_T[:, 0] @ b) - 1) < 1e-3 and abs(abs(U_F[:, 0] @ f) - 1) < 1e-3
    X1 = X[:, :, :1, :]
    U_L1, U_T1, U_F1 = hosvd3(X1, 3, 1, 5, 0)
    UL2, UF2 = m["s43"].compute_ul_ud_randomized(X1[:, :, 0, :], 3, 5, seed=0)
    assert np.allclose(np.abs(project3(X1, U_L1, U_T1, U_F1)), np.abs(m["s26"].project_core(X1[:, :, 0, :], UL2, UF2)), atol=1e-3)
    print("  [PASS] 3-mode HOSVD recovers planted factors; with T'=1 the core equals the Tucker-2 core up to sign")

    # At T'=1, the whole orderstats pipeline must give the reported detector's features: cores() on
    # T'=1 tensors equals 61_flatten_control.build('hosvd') on the matching (N, L, F) tensors, up to sign.
    N, L, D = 60, 9, 6
    tok = rng.standard_normal((N * 5, L, D)).astype(np.float32)
    offs = np.arange(0, N * 5 + 1, 5)
    ten = build_tensors(tok.astype(np.float16), offs, 1, "orderstats")
    feats3 = {"core": ten["peak"][:, :, 0].astype(np.float32), "static": ten["range"][:, :, 0].astype(np.float32),
              "velocity": ten["update"][:, :, 0].astype(np.float32)}
    tr = np.arange(40)
    global R_F
    saved, R_F = R_F, 3
    try:
        Z = cores(m, ten, tr, [1], 11)[1]
    finally:
        R_F = saved
    ref = m["s61"].build("hosvd", feats3, [("core", 5, 3), ("static", 5, 3), ("velocity", 4, 3)], tr, 11)
    mine = np.concatenate([Z["peak"], Z["range"], Z["update"]], axis=1)
    assert mine.shape == ref.shape and np.allclose(np.abs(mine), np.abs(ref), atol=1e-3), np.abs(np.abs(mine) - np.abs(ref)).max()
    print("  [PASS] orderstats T'=1 through scaling, 3-mode HOSVD and projection = 61.build('hosvd') on the "
          "reported triple (up to sign)")

    # End to end through the forest on a signal planted in the second token bin only.
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
        toks[offs[i] + (lens[i] + 1) // 2:offs[i + 1]] += (2.0 * y[i]) * u
    known, _ = m["s44"].derive_is_known(y, pid)
    tr, te = (np.asarray(v, dtype=int) for v in c["question_split"](known, pid, n, 42))
    ten = build_tensors(toks.astype(np.float16), offs, 2, "mean")
    saved, R_F = R_F, 4
    try:
        Z = cores(m, ten, tr, [1, 2], 42)
    finally:
        R_F = saved
    assert Z[1]["state"].shape == (n, 5 * 1 * 4) and Z[2]["state"].shape == (n, 5 * 2 * 4) and Z[2]["update"].shape == (n, 4 * 2 * 4)
    aucs = {r: c["pooled_auroc"](m["s26"].fit_eval("RF", Z[r]["state"][tr], y[tr], Z[r]["state"][te], 42), y[te]) for r in (1, 2)}
    assert aucs[2] > 0.85, aucs
    print("  [PASS] end to end: planted second-bin signal found at R_T = 2 (AUROC %.3f); at R_T = 1 the "
          "energy-ranked token basis keeps the other bin (AUROC %.3f)" % (aucs[2], aucs[1]))

    assert setting_key(8, 4, "orderstats") == "T8_RT4_orderstats"
    assert output_name("m", "d", "mean", "RF", [(8, 4), (8, 8)]) == "tucker3_m_d_mean_RF_T8.json"
    assert parse_settings("8:4,8:8") == [(8, 4), (8, 8)]
    fake = {"T1_RT1_orderstats": {"pooled_mean": 0.8750}}
    an = anchor(fake, "orderstats", 0.8782)
    assert an["within_tolerance"] and abs(an["diff_pts"] + 0.32) < 1e-9 and anchor(fake, "mean", 0.8782) is None
    print("  [PASS] settings, file names and the T'=1 anchor check")
    print("\n  ALL PASS")
    return True


def parse_settings(s):
    return [(int(t), int(r)) for t, r in (p.split(":") for p in s.split(","))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dataset", choices=["truthfulqa", "tydiqa_gp", "nq_open", "triviaqa"])
    ap.add_argument("--model_folder")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--in-dir", default=DEFAULT_IN)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--readout", default="RF", choices=["RF", "LR"])
    ap.add_argument("--family", default="orderstats", choices=sorted(FAMILIES))
    ap.add_argument("--settings", default=None, help="T':R_T pairs, e.g. 8:4,8:8 (default: the whole grid)")
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
    run(a.dataset, a.model_folder, data_dir, a.in_dir, a.out_dir, a.readout, a.family, settings, a.seeds)


if __name__ == "__main__":
    main()
