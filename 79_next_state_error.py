"""
79_next_state_error.py -- does a linear model of token-to-token dynamics separate right from wrong math answers?

THE CHECK. One linear model predicts each token's hidden state from the previous token's state:
    next state = W * current state + b
fitted by least squares on RIGHT answers of training questions only. On held-out questions, each answer is
scored by its average prediction error (squared distance between predicted and actual next state) over the
first half, the second half, and the whole answer. Score = same-question AUROC, as in the position check.

ONE COMPARISON ROW. The same error when the previous token is ignored (W = 0: the prediction is always the
average right-answer state). If the linear model separates no better than this, wrong answers sit in a
different place but do not move differently.

Reads the store 77 wrote (9 blocks x 64 directions per token) and labels from the sequences file. The 150
basis questions are left out; folds of whole questions are 78's. CPU only, no new data.

  python 79_next_state_error.py --self-test
  python 79_next_state_error.py --model_folder qwen-2.5-7b-instruct
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "token_structure")
CHUNK = 200000
PARTS = ("first_half", "second_half", "whole_answer")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pair_starts(offsets, idx):
    """Rows t such that t and t + 1 belong to the same answer, for the answers in idx."""
    rows = [np.arange(offsets[n], offsets[n + 1] - 1) for n in idx]
    return np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)


def fit_next_state(X, rows, ridge=1e-6):
    """Least squares for X[t + 1] = W X[t] + b over the given rows. Returns (W, b, mean of X[t + 1])."""
    F = X.shape[1]
    G = np.zeros((F + 1, F + 1))
    H = np.zeros((F + 1, F))
    for a in range(0, len(rows), CHUNK):
        r = rows[a:a + CHUNK]
        A = np.hstack([np.asarray(X[r], dtype=np.float64), np.ones((len(r), 1))])
        G += A.T @ A
        H += A.T @ np.asarray(X[r + 1], dtype=np.float64)
    G[np.diag_indices(F)] += ridge * np.trace(G[:F, :F]) / F
    coef = np.linalg.solve(G, H)
    return coef[:F], coef[F], H[F] / G[F, F]


def errors(X, rows, W, b, mean):
    """Squared prediction error of X[t + 1] for each row t: from the previous token, and from the average."""
    e_prev, e_avg = np.empty(len(rows)), np.empty(len(rows))
    for a in range(0, len(rows), CHUNK):
        r = rows[a:a + CHUNK]
        Y = np.asarray(X[r + 1], dtype=np.float64)
        e_prev[a:a + CHUNK] = ((Y - (np.asarray(X[r], dtype=np.float64) @ W + b)) ** 2).sum(1)
        e_avg[a:a + CHUNK] = ((Y - mean) ** 2).sum(1)
    return e_prev, e_avg


def halves(e, offsets, idx):
    """(answers, 3): mean error in the first half, second half and whole answer. e is indexed by token row and
    is NaN at each answer's first token (nothing precedes it). Halves split as 73 splits them."""
    out = np.full((len(idx), 3), np.nan)
    for i, n in enumerate(idx):
        v = e[offsets[n]:offsets[n + 1]]
        T = len(v)
        for j, part in enumerate((v[:(T + 1) // 2], v[T // 2:], v)):
            part = part[np.isfinite(part)]
            if len(part):
                out[i, j] = part.mean()
    return out


def score(X, offsets, y, pid, keep, fold, log=print):
    """Held-out errors for every token of the evaluated answers. Returns (from previous token, from average)."""
    e_prev = np.full(len(X), np.nan)
    e_avg = np.full(len(X), np.nan)
    for f in range(int(fold.max()) + 1):
        t0 = time.time()
        train_right = keep[(fold != f) & (y[keep] == 0)]
        test = keep[fold == f]
        W, b, mean = fit_next_state(X, pair_starts(offsets, train_right))
        rows = pair_starts(offsets, test)
        e_prev[rows + 1], e_avg[rows + 1] = errors(X, rows, W, b, mean)
        log("    fold %d: fitted on %d right answers, scored %d answers (%.0fs)" % (f, len(train_right), len(test), time.time() - t0))
    return e_prev, e_avg


def run(model_folder, dataset, out_dir, seed, n_boot):
    import torch
    import yaml
    s78 = _load("s78", "78_math_same_question.py")
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    root = os.path.normpath(os.path.join(HERE, cfg["math"]["out_dir"], model_folder))
    d = os.path.join(root, "%s_states" % dataset)
    if not os.path.exists(os.path.join(d, "meta.json")):
        raise SystemExit("%s is missing or incomplete -- run 77_extract_math_states.py first" % d)
    seq = torch.load(os.path.join(root, "%s_sequences_v1.pt" % dataset), weights_only=False)
    y = np.asarray(seq["all_hallucination_flag"]).astype(int)            # 1 = wrong final number
    pid = np.asarray(seq["prompt_id"], dtype=np.int64)
    offsets = np.load(os.path.join(d, "offsets.npy")).astype(np.int64)
    if not np.array_equal(pid, np.load(os.path.join(d, "prompt_id.npy")).astype(np.int64)):
        raise SystemExit("the state store and the sequences file disagree on question ids")
    keep = np.flatnonzero(~np.load(os.path.join(d, "basis_question.npy")))
    tokens = np.load(os.path.join(d, "tokens.npy"))
    X = tokens.reshape(len(tokens), -1)
    fold, _ = s78.question_folds(pid[keep], y[keep], s78.N_FOLDS, seed)
    print("  [%s/%s] %d answers evaluated, %d tokens x %d coordinates" % (model_folder, dataset, len(keep), X.shape[0], X.shape[1]), flush=True)

    e_prev, e_avg = score(X, offsets, y, pid, keep, fold, lambda m: print(m, flush=True))
    out = {"model_folder": model_folder, "dataset": dataset, "seed": seed, "n_boot": n_boot, "columns": list(PARTS),
           "same_question_auroc": {}}
    for name, e in (("from_previous_token", e_prev), ("from_average_state", e_avg)):
        s = s78.summarize(s78.same_question_stats(halves(e, offsets, keep), y[keep], pid[keep]), n_boot, seed)
        out["same_question_auroc"][name] = {"auroc": s["same_question_auroc"], "interval": s["auroc_interval"],
                                            "pairs": s["pairs"], "questions": s["questions"]}
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "next_state_%s_%s.json" % (model_folder, dataset))
    json.dump(s78.clean(out), open(dst, "w"), indent=1)

    r = out["same_question_auroc"]["from_previous_token"]
    print("\n  SUMMARY: same-question AUROC [95%%], %d questions, %d pairs" % (r["questions"][2], r["pairs"][2]))
    print("    %-38s %-20s %-20s %s" % ("", "first half", "second half", "whole answer"))
    for name, label in (("from_previous_token", "prediction error, from previous token"),
                        ("from_average_state", "prediction error, from average state")):
        v = out["same_question_auroc"][name]
        print("    %-38s %s" % (label, "  ".join("%.3f [%.3f, %.3f]" % (v["auroc"][j], v["interval"][0][j], v["interval"][1][j])
                                                  for j in range(3))))
    ref = os.path.join(out_dir, "math_%s_%s.json" % (model_folder, dataset))
    if os.path.exists(ref):
        t = json.load(open(ref))["token_scorers"]["pooled_probe"]["by_tokens_used"]
        a = dict(zip(t["columns"], t["same_question_auroc"]))
        print("    %-38s %-20s %-20s %.3f" % ("token probe (position check)", "%.3f" % a["first_half"], "%.3f" % a["second_half"], a["mean"]))
    print("  wrote %s" % dst)


# ---------------------------------------------------------------------------------------------

def self_test():
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print("    [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
        ok = ok and bool(cond)

    offs = np.array([0, 3, 4, 7])
    check("pairs never cross answers", pair_starts(offs, [0, 1, 2]).tolist() == [0, 1, 4, 5])

    X = rng.normal(size=(500, 6))
    rows = np.arange(499)
    W, b, mean = fit_next_state(X, rows, ridge=0.0)
    A = np.hstack([X[:-1], np.ones((499, 1))])
    ref = np.linalg.lstsq(A, X[1:], rcond=None)[0]
    check("fit equals least squares", np.allclose(W, ref[:6]) and np.allclose(b, ref[6]) and np.allclose(mean, X[1:].mean(0)))

    # Right answers follow one rotation, wrong answers a different rotation in their second half. Both keep
    # the same spread of states, so only the dynamics differ: the previous-token error must separate the
    # second half, and the average-state error must not.
    d, a = 8, 0.9
    R1 = np.linalg.qr(rng.normal(size=(d, d)))[0]
    R2 = np.linalg.qr(rng.normal(size=(d, d)))[0]
    Q = 200
    pid = np.repeat(np.arange(Q), 10)
    y = np.zeros(Q * 10, dtype=int)
    for q in range(Q):
        y[q * 10 + rng.permutation(10)[:int(rng.integers(0, 6))]] = 1
    lengths = rng.integers(30, 61, size=Q * 10)
    offsets = np.zeros(Q * 10 + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    X = np.empty((offsets[-1], d))
    for n in range(Q * 10):
        x = rng.normal(size=d)
        for t in range(lengths[n]):
            R = R2 if (y[n] and t >= (lengths[n] + 1) // 2) else R1     # the halves share the middle token of an odd answer
            x = a * (R @ x) + np.sqrt(1 - a * a) * rng.normal(size=d)
            X[offsets[n] + t] = x
    X = X.astype(np.float16)
    s78 = _load("s78", "78_math_same_question.py")
    keep = np.arange(Q * 10)
    fold, _ = s78.question_folds(pid, y, 5, 0)
    e_prev, e_avg = score(X, offsets, y, pid, keep, fold, lambda m: None)
    check("first token of every answer has no error", np.isnan(e_prev[offsets[:-1]]).all())
    res = {}
    for name, e in (("previous", e_prev), ("average", e_avg)):
        res[name] = s78.summarize(s78.same_question_stats(halves(e, offsets, keep), y, pid), 200, 0)["same_question_auroc"]
    p, m = res["previous"], res["average"]
    check("changed dynamics: previous-token error separates the second half, not the first",
          p[1] > 0.9 and abs(p[0] - 0.5) < 0.08, "first %.3f, second %.3f" % (p[0], p[1]))
    check("same spread of states: average-state error separates neither half",
          abs(m[0] - 0.5) < 0.08 and abs(m[1] - 0.5) < 0.08, "first %.3f, second %.3f" % (m[0], m[1]))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model_folder")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    if a.self_test:
        print("  self-test")
        raise SystemExit(0 if self_test() else 1)
    if not a.model_folder:
        raise SystemExit("--model_folder is required (or --self-test)")
    run(a.model_folder, a.dataset, a.out_dir, a.seed, a.n_boot)


if __name__ == "__main__":
    main()
