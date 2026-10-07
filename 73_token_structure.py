"""
73_token_structure.py -- does the token mode of short QA answers carry structure a functional model could use?
==================================================================================================

WHY. The next project models the token mode as a smooth curve over normalized position (functional CP for
unaligned observations: Han, Shi and Zhang, arXiv:2108.04201; Tang, Kolda and Zhang, arXiv:2410.14046).
That only pays off if (1) hidden states change smoothly from token to token and (2) the hallucination
signal depends on WHERE in the answer it sits, not only on its level. The reported detector already
captures level (peak, quantiles), and its 3-mode ablation (70) found at most 0.37 AUROC points from
contiguous token bins. This script measures (1) and (2) directly, before anyone builds a model.

NO NEW EXTRACTION. Reads ../data-tokenstates/<model>/<dataset>/, which 69 wrote for the 3-mode ablation:
blocks 15..23, completion tokens only, float16. Writes only results/token_structure/. CPU only.

WHAT IS MEASURED, per model and dataset, on the paper's question-level split (five draws):

  M0  space.      Per layer: 28's token scaler and a 64-direction PCA basis, both fitted on a subsample of
                  TRAINING-question tokens, refitted for every draw (no basis sees a test question). Reports
                  the share of variance the 64 directions keep -- the per-token analogue of the 64.7-66.4%
                  the paper reports for the peak summary. M1 and M2 run in this projected space, which is the
                  space a functional model would decompose.

  M1  smoothness. Mean cosine similarity between projected states k tokens apart (k = 1..8), averaged
                  within each answer and then over answers, against a NULL in which each answer's tokens are
                  randomly reordered. Tokens of one answer share their prompt, so they are similar regardless
                  of order; only the excess over the null is order structure:
                      index(k) = (sim(k) - null(k)) / (1 - null(k))
                  1 = adjacent tokens agree fully beyond the shared offset; 0 = order carries nothing.
                  Per layer and per class, on the first draw's space (it is label-free and descriptive).
                  The null reorders WITHIN an answer, so in short answers its random pairs are themselves
                  close in time: the index is conservative there, and sim(k) is reported alongside it.

  M2  position.   A logistic-regression probe on single tokens (all nine layers, 9 x 64 = 576 features;
                  each token carries its answer's label; weights 1/T so every answer counts once, classes
                  balanced), trained on training questions. Its token scores are aggregated per test answer
                  in ways that differ ONLY in where they look:
                      mean, max, first token, last token, first half, second half
                  and each aggregate's pooled and within-question AUROC is reported, mean and std over draws.
                  Also the mean token score by class in 10 bins of normalized position (t + 0.5) / T.
                  The mean-aggregate row doubles as a plain token-probe baseline under the paper protocol.

HOW TO READ IT.
  M1 clearly above zero and decaying over several tokens, AND M2 halves/ends separated by more than their
      spread over draws  -> the token mode has smoothness and positional signal; a functional model has
      something to fit.
  M1 near zero  -> token order carries nothing beyond the shared prompt in these answers; a smooth-curve
      model has no basis here, whatever M2 says.
  M2 aggregates within each other's spread and class curves parallel  -> the signal is a level shift spread
      evenly along the answer, which order statistics already capture (as 70 found). The idea then needs
      longer generations, such as reasoning traces, rather than a new model.

  python 73_token_structure.py --self-test
  python 73_token_structure.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.abspath(os.path.join(HERE, "..", "data-tokenstates"))
DEFAULT_OUT = os.path.join(HERE, "results", "token_structure")
FIRST_BLOCK = 15          # tokens.npy layer j holds block 15 + j (69's WINDOW, hidden-state indices 16..24)
R = 64                    # the feature rank the reported detector keeps
MAX_LAG = 8
N_BINS = 10
N_FIT = 40000             # training tokens per draw used to fit the scaler and the basis
CHUNK = 4096
AGGREGATES = ("mean", "max", "first", "last", "first_half", "second_half")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------------------------
# M0: the projected space
# ---------------------------------------------------------------------------------------------

def token_rows(offsets, idx):
    """Token rows of answers idx, concatenated in idx order."""
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(offsets[n], offsets[n + 1]) for n in idx])


def fit_space(tokens, offsets, train_idx, s28, n_fit, r, seed):
    """Per layer: 28's robust token scaler, the post-scaling mean, and an r-direction PCA basis, all from at
    most n_fit tokens of the training answers. Returns (space, top-r variance share per layer, tokens used)."""
    from sklearn.utils.extmath import randomized_svd
    rows = token_rows(offsets, train_idx)
    if len(rows) > n_fit:
        rows = np.sort(np.random.default_rng(seed).choice(rows, size=n_fit, replace=False))
    X = np.asarray(tokens[rows], dtype=np.float16)
    space, share = [], []
    for j in range(X.shape[1]):
        Xl = X[:, j, :].astype(np.float32)
        params = s28.fit_robust_scale(Xl)
        Z = s28.apply_robust_scale(Xl, params).astype(np.float32)
        mu = Z.mean(axis=0)
        Z -= mu
        total = float(np.sum(Z * Z, dtype=np.float64))
        _, S, Vt = randomized_svd(Z, n_components=r, random_state=seed + j)
        share.append(float(np.sum(S.astype(np.float64) ** 2) / total))
        space.append((params, mu.astype(np.float32), Vt.T.astype(np.float32)))
    return space, share, int(len(rows))


def project_all(tokens, space, s28, chunk=CHUNK):
    """(total_T, P, r) float32: every token scaled and projected in its layer's space, streamed in chunks."""
    n_tok, n_layers = tokens.shape[0], tokens.shape[1]
    out = np.empty((n_tok, n_layers, space[0][2].shape[1]), dtype=np.float32)
    for a in range(0, n_tok, chunk):
        X = np.asarray(tokens[a:a + chunk], dtype=np.float32)
        for j, (params, mu, V) in enumerate(space):
            out[a:a + chunk, j] = (s28.apply_robust_scale(X[:, j, :], params) - mu) @ V
    return out


# ---------------------------------------------------------------------------------------------
# M1: smoothness across tokens
# ---------------------------------------------------------------------------------------------

def lag_sums(Z, offsets, idx, max_lag, rng=None):
    """Sum over answers of the within-answer mean cosine similarity between states k tokens apart, and the
    number of answers contributing, for k = 1..max_lag. Z is (total_T, r) for one layer. With rng, each
    answer's tokens are first put in random order (the null)."""
    sums = np.zeros(max_lag)
    cnt = np.zeros(max_lag, dtype=np.int64)
    for n in idx:
        h = Z[offsets[n]:offsets[n + 1]]
        T = len(h)
        if T < 2:
            continue
        if rng is not None:
            h = h[rng.permutation(T)]
        u = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
        for k in range(1, min(max_lag, T - 1) + 1):
            sums[k - 1] += float(np.mean(np.sum(u[:-k] * u[k:], axis=1)))
            cnt[k - 1] += 1
    return sums, cnt


def smoothness(Z, offsets, idx, max_lag, seed):
    """sim(k), null(k), index(k) and the number of answers behind each lag."""
    s, c = lag_sums(Z, offsets, idx, max_lag)
    sn, cn = lag_sums(Z, offsets, idx, max_lag, rng=np.random.default_rng(seed))
    sim = s / np.maximum(c, 1)
    null = sn / np.maximum(cn, 1)
    index = (sim - null) / np.maximum(1.0 - null, 1e-12)
    return {"sim": sim.tolist(), "null": null.tolist(), "index": index.tolist(), "answers": c.tolist()}


# ---------------------------------------------------------------------------------------------
# M2: where in the answer the signal sits
# ---------------------------------------------------------------------------------------------

def token_weights(offsets, idx, y):
    """1/T per token, then rescaled so both classes carry equal total weight and the mean weight per answer
    is 1 (keeps the probe's regularization on the scale of an answer-level fit)."""
    lengths = np.diff(offsets)[idx]
    w = np.repeat(1.0 / lengths, lengths)
    yt = np.repeat(y[idx], lengths)
    for c in (0, 1):
        m = yt == c
        if m.any():
            w[m] *= 0.5 / w[m].sum()
    return w * (len(idx) / w.sum()), yt


def fit_probe(X, yt, w, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000, random_state=seed)
    clf.fit(sc.transform(X), yt, sample_weight=w)
    return sc, clf


def answer_aggregates(s, offsets, idx):
    """Per answer in idx, aggregates of its token scores s (indexed by global token row)."""
    out = {a: np.empty(len(idx)) for a in AGGREGATES}
    for i, n in enumerate(idx):
        v = s[offsets[n]:offsets[n + 1]]
        T = len(v)
        out["mean"][i] = v.mean()
        out["max"][i] = v.max()
        out["first"][i] = v[0]
        out["last"][i] = v[-1]
        out["first_half"][i] = v[:(T + 1) // 2].mean()
        out["second_half"][i] = v[T // 2:].mean()
    return out


def position_curves(s, offsets, idx, y, n_bins):
    """Mean token score by class in n_bins bins of normalized position (t + 0.5) / T. Each answer averages
    its own tokens per bin first, so long answers do not dominate; bins it does not reach are skipped."""
    sums = np.zeros((2, n_bins))
    cnt = np.zeros((2, n_bins), dtype=np.int64)
    for n in idx:
        v = s[offsets[n]:offsets[n + 1]]
        T = len(v)
        b = np.minimum(((np.arange(T) + 0.5) / T * n_bins).astype(int), n_bins - 1)
        tot = np.bincount(b, weights=v, minlength=n_bins)
        k = np.bincount(b, minlength=n_bins)
        hit = k > 0
        c = int(y[n])
        sums[c, hit] += tot[hit] / k[hit]
        cnt[c, hit] += 1
    return sums / np.maximum(cnt, 1), cnt


# ---------------------------------------------------------------------------------------------

def mean_std(v):
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (None, None)


def run(dataset, model_folder, data_dir, in_dir, out_dir, seeds=None, n_fit=N_FIT):
    import methods.base as B
    c = B.canonical()
    s28 = _load("s28", "28_eval_band.py")
    s70 = _load("s70", "70_tensor_tucker.py")
    seeds = [int(x) for x in (seeds or c["seeds"])]
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "tokstruct_%s_%s.json" % (model_folder, dataset))

    tokens, offsets, y, pid, meta = s70.load_tokens(in_dir, model_folder, dataset)
    pinned = np.load(os.path.join(data_dir, model_folder, "%s_phase2_features.npz" % dataset))
    if not (np.array_equal(np.asarray(pinned["label"]).astype(int), y.astype(int))
            and np.array_equal(np.asarray(pinned["prompt_id"]).astype(np.int64), pid.astype(np.int64))):
        raise SystemExit("token store and pinned features disagree on labels or question ids")
    y, pid = y.astype(int), pid.astype(np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    is_known = c["derive_is_known"](y, pid)
    lengths = np.diff(offsets)
    n_layers = tokens.shape[1]
    blocks = [FIRST_BLOCK + j for j in range(n_layers)]
    ref_path = os.path.join(HERE, "results", "flatten_control", "flatten_%s_%s_RF.json" % (model_folder, dataset))
    reference = json.load(open(ref_path))["summary"]["hosvd"]["pooled_mean"] if os.path.exists(ref_path) else None

    length_stats = {"all": {"mean": float(lengths.mean()), "median": float(np.median(lengths)),
                            "max": int(lengths.max())}}
    for cl, name in ((0, "correct"), (1, "hallucinated")):
        L = lengths[y == cl]
        length_stats[name] = {"mean": float(L.mean()), "median": float(np.median(L)), "n": int(len(L))}
    print("  [%s/%s] %d answers, %d tokens | length mean %.1f median %.0f max %d | correct %.1f, hallucinated %.1f"
          % (model_folder, dataset, len(y), int(offsets[-1]), lengths.mean(), np.median(lengths), lengths.max(),
             length_stats["correct"]["mean"], length_stats["hallucinated"]["mean"]), flush=True)

    out = {"dataset": dataset, "model_folder": model_folder, "blocks": blocks, "r": R, "n_fit": n_fit,
           "seeds": seeds, "protocol": "question-level (paper protocol)", "total_tokens": int(offsets[-1]),
           "n_answers": int(len(y)), "length_stats": length_stats, "reference_lotus_pooled": reference,
           "complete": False}
    share_rows, agg_rows, curve_rows = [], {a: [] for a in AGGREGATES}, []
    t_start = time.time()
    for di, seed in enumerate(seeds):
        t0 = time.time()
        tr, te = (np.asarray(v, dtype=np.int64) for v in c["question_split"](is_known, pid, len(y), seed))
        space, share, used = fit_space(tokens, offsets, tr, s28, n_fit, R, seed)
        Z = project_all(tokens, space, s28)
        share_rows.append(share)
        t1 = time.time()

        if di == 0:
            m1 = {}
            all_idx = np.arange(len(y))
            for j, blk in enumerate(blocks):
                Zl = np.ascontiguousarray(Z[:, j, :])
                m1[str(blk)] = {"all": smoothness(Zl, offsets, all_idx, MAX_LAG, seed),
                                "correct": smoothness(Zl, offsets, all_idx[y == 0], MAX_LAG, seed),
                                "hallucinated": smoothness(Zl, offsets, all_idx[y == 1], MAX_LAG, seed)}
            out["M1_smoothness"] = m1
            print("    M1 index(k=1) by block: %s  (%.0fs)" % (
                "  ".join("%d:%.3f" % (b, m1[str(b)]["all"]["index"][0]) for b in blocks), time.time() - t1), flush=True)

        rows_tr = token_rows(offsets, tr)
        w, yt = token_weights(offsets, tr, y)
        sc, clf = fit_probe(Z[rows_tr].reshape(len(rows_tr), -1), yt, w, seed)
        rows_te = token_rows(offsets, te)
        s = np.full(len(Z), np.nan, dtype=np.float64)
        s[rows_te] = clf.decision_function(sc.transform(Z[rows_te].reshape(len(rows_te), -1)))
        agg = answer_aggregates(s, offsets, te)
        line = []
        for a in AGGREGATES:
            wp = c["within_prompt_auroc"](agg[a], y[te], pid[te])
            agg_rows[a].append({"seed": seed, "pooled_auroc": float(c["pooled_auroc"](agg[a], y[te])),
                                "within_prompt_auroc": wp["within_prompt_auroc"]})
            line.append("%s %.4f" % (a, agg_rows[a][-1]["pooled_auroc"]))
        curves, cnt = position_curves(s, offsets, te, y, N_BINS)
        curve_rows.append({"correct": curves[0].tolist(), "hallucinated": curves[1].tolist(),
                           "answers": cnt.tolist()})
        print("    seed %-3d share64 %.3f-%.3f | %s | fit %d tok (%.0fs)" % (
            seed, min(share), max(share), " | ".join(line), used, time.time() - t0), flush=True)

        out["M0_share64"] = {str(b): mean_std([r_[j] for r_ in share_rows]) for j, b in enumerate(blocks)}
        out["M2_aggregates"] = {a: {"pooled": mean_std([r_["pooled_auroc"] for r_ in agg_rows[a]]),
                                    "within": mean_std([r_["within_prompt_auroc"] for r_ in agg_rows[a]]),
                                    "per_seed": agg_rows[a]} for a in AGGREGATES}
        cs = np.array([r_["correct"] for r_ in curve_rows])
        hs = np.array([r_["hallucinated"] for r_ in curve_rows])
        out["M2_position_curves"] = {"bins": N_BINS, "correct": cs.mean(0).tolist(), "hallucinated": hs.mean(0).tolist(),
                                     "gap": (hs - cs).mean(0).tolist(), "gap_std": (hs - cs).std(0).tolist(),
                                     "per_seed": curve_rows}
        out["elapsed_seconds"] = round(time.time() - t_start, 1)
        json.dump(out, open(dst, "w"), indent=1)

    out["complete"] = True
    json.dump(out, open(dst, "w"), indent=1)

    print("\n  SUMMARY %s / %s" % (model_folder, dataset))
    print("    M0 share of variance in 64 directions, by block: %s" % "  ".join(
        "%s:%.3f" % (b, v[0]) for b, v in out["M0_share64"].items()))
    mid = str(blocks[len(blocks) // 2])
    print("    M1 index(k) at block %s, k=1..%d: %s" % (mid, MAX_LAG, " ".join(
        "%.3f" % v for v in out["M1_smoothness"][mid]["all"]["index"])))
    for a in AGGREGATES:
        p, w_ = out["M2_aggregates"][a]["pooled"], out["M2_aggregates"][a]["within"]
        print("    M2 %-12s pooled %.4f +- %.4f   within %s" % (
            a, p[0], p[1], "%.4f +- %.4f" % w_ if w_[0] is not None else "n/a"))
    print("    M2 gap (hallucinated - correct) by position bin: %s" % " ".join(
        "%.2f" % v for v in out["M2_position_curves"]["gap"]))
    if reference is not None:
        print("    reported detector (flatten_control, Tucker-2 core, RF): %.4f" % reference)
    print("  wrote %s" % dst)


# ---------------------------------------------------------------------------------------------

def self_test():
    """Synthetic answers with planted properties; checks every measurement recovers what was planted."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        print("    [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
        ok = ok and bool(cond)

    # ragged store: 600 answers of length 2..30, r = 16
    lengths = rng.integers(2, 31, size=600)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    r = 16

    # M1: AR(1) trajectories around an answer-specific offset are smooth; i.i.d. ones are not.
    def store(rho):
        Z = np.empty((offsets[-1], r), dtype=np.float32)
        for n in range(len(lengths)):
            base = rng.normal(size=r) * 2.0
            e = rng.normal(size=r)
            for t in range(lengths[n]):
                e = rho * e + np.sqrt(1 - rho ** 2) * rng.normal(size=r)
                Z[offsets[n] + t] = base + e
        return Z
    idx = np.arange(len(lengths))
    sm = smoothness(store(0.9), offsets, idx, 4, seed=1)
    iid = smoothness(store(0.0), offsets, idx, 4, seed=1)
    check("smooth trajectories score high at lag 1", sm["index"][0] > 0.5, "index %.3f" % sm["index"][0])
    check("smooth index decays with lag", sm["index"][0] > sm["index"][3], "k=1 %.3f, k=4 %.3f" % (sm["index"][0], sm["index"][3]))
    check("i.i.d. tokens score near zero", abs(iid["index"][0]) < 0.06, "index %.3f" % iid["index"][0])
    check("shared offset alone does not count", iid["sim"][0] > 0.3 and abs(iid["index"][0]) < 0.06,
          "raw sim %.3f vs index %.3f" % (iid["sim"][0], iid["index"][0]))

    # M2: hallucinated answers differ only in their second half.
    y = rng.integers(0, 2, size=len(lengths))
    s = rng.normal(size=offsets[-1])
    for n in range(len(lengths)):
        if y[n] == 1:
            T = lengths[n]
            s[offsets[n] + T // 2:offsets[n + 1]] += 1.5
    agg = answer_aggregates(s, offsets, idx)
    a1, a2 = roc_auc_score(y, agg["first_half"]), roc_auc_score(y, agg["second_half"])
    check("second-half signal: second half beats first half", a2 > a1 + 0.15, "first %.3f, second %.3f" % (a1, a2))
    curves, cnt = position_curves(s, offsets, idx, y, 10)
    gap = curves[1] - curves[0]
    check("position curve gap sits in the late bins", gap[6:].mean() > 1.0 and abs(gap[:4].mean()) < 0.4,
          "early %.2f, late %.2f" % (gap[:4].mean(), gap[6:].mean()))

    # token weights: every answer counts once, classes balanced.
    w, yt = token_weights(offsets, idx, y)
    per_answer = np.add.reduceat(w, offsets[:-1])
    check("token weights balance classes", abs(w[yt == 0].sum() - w[yt == 1].sum()) < 1e-6 * w.sum(),
          "%.3f vs %.3f" % (w[yt == 0].sum(), w[yt == 1].sum()))
    check("token weights are equal within a class", np.allclose(per_answer[y == 0], per_answer[y == 0][0]),
          "spread %.2e" % np.ptp(per_answer[y == 0]))

    # M0: scaler + basis + projection on a fake (total_T, P, D) store, through 28's real scaler.
    s28 = _load("s28", "28_eval_band.py")
    D, P = 48, 3
    U = np.linalg.qr(rng.normal(size=(D, 4)))[0]
    raw = (rng.normal(size=(offsets[-1], P, 4)) @ U.T * 3.0 + 0.1 * rng.normal(size=(offsets[-1], P, D)))
    raw = raw.astype(np.float16)
    space, share, used = fit_space(raw, offsets, idx[:400], s28, 5000, 4, seed=0)
    check("four planted directions carry most of the variance", min(share) > 0.8, "share %s" % ["%.3f" % v for v in share])
    Zp = project_all(raw, space, s28, chunk=1000)
    check("projection shape", Zp.shape == (offsets[-1], P, 4), str(Zp.shape))
    check("projection is finite", bool(np.isfinite(Zp).all()), "")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dataset", choices=["truthfulqa", "tydiqa_gp", "nq_open", "triviaqa"])
    ap.add_argument("--model_folder")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--in-dir", default=DEFAULT_IN)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--n-fit", type=int, default=N_FIT)
    a = ap.parse_args()
    if a.self_test:
        print("  self-test")
        raise SystemExit(0 if self_test() else 1)
    if not (a.dataset and a.model_folder):
        raise SystemExit("--dataset and --model_folder are required (or --self-test)")
    data_dir = a.data_dir
    if not data_dir:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]
    run(a.dataset, a.model_folder, data_dir, a.in_dir, a.out_dir, a.seeds, a.n_fit)


if __name__ == "__main__":
    main()
