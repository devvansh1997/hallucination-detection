"""
78_math_same_question.py -- can a right and a wrong answer to the SAME question be told apart, and where? (T-025)
==================================================================================================

WHY. On short QA and on RAGTruth the gap between hallucinated and correct answers can come from the prompt:
a hard prompt looks different inside the model AND gets more wrong answers. Here every question has ten
sampled answers graded by their final number (76), so a wrong answer is compared with a right answer to the
same question. Whatever separates such a pair cannot be the prompt.

READS (written by 76 and 77; nothing is extracted here, CPU only)
    ../data-math/<model>/<dataset>_sequences_v1.pt    token ids and the right/wrong label of every answer
    ../data-math/<model>/<dataset>_states/            per-token states (9 blocks x 64 directions), LOTUS summaries
WRITES  results/token_structure/math_<model>_<dataset>.json

WHO IS EVALUATED. The questions 77 fitted its scaler and basis on are left out entirely. Every other question
is tested exactly once: whole questions are dealt into five folds, and each fold is scored by detectors
fitted on the other four. No detector ever sees an answer to a question it is tested on.

THE NUMBER REPORTED. Same-question AUROC: over all pairs (wrong answer, right answer) of one question, the
share in which the wrong answer gets the higher score. Ties count half; 0.5 means no separation. Only
questions with both outcomes have such pairs. It is the paper's within-question AUROC (checked against the
canonical function at run time). Intervals are 95%, from resampling those questions. The pooled AUROC, which
also compares answers to different questions, is reported beside it as mean and std over the five folds.

ANSWER-LEVEL DETECTORS, one score per answer.
    LOTUS                     the reported method through the pinned code: 61.build("hosvd") on
                              44.CONDITION_SPECS["triple_concat"] (896 features), 26.fit_eval random forest.
    pooled token probe        73's probe, as on short QA and RAGTruth: logistic regression on single tokens,
                              every token carrying its answer's label, fitted on all training answers.
                              Answer score = mean over its tokens.
    same-question direction   one direction, nothing fitted beyond averages. For each training question with
                              both outcomes: mean state of its wrong answers minus mean state of its right
                              answers. The direction is the average of those differences over questions.
                              Token score = the token's coordinate along it; answer score = mean over tokens.
    answer length, answer NLL controls that use no hidden state.
  Both token scorers work on the 9 x 64 = 576 stored coordinates of a token, each put on a common scale
  (mean 0, std 1 over training tokens). To keep the probe fit small, at most 64 randomly placed tokens of
  each TRAINING answer are used; every test token is scored.

TOKEN SCORES ARE IN STANDARD DEVIATIONS. Within a fold, token scores are shifted and scaled so that 0 is the
average token of a right answer and 1 is one standard deviation of right-answer tokens (held-out answers,
every answer counting once). A change of units only: no AUROC depends on it.

WHERE IN THE ANSWER. The same two numbers -- same-question AUROC, and the same-question gap (wrong minus
right, in those standard deviations, averaged over questions) -- for token scores averaged over:
    each tenth of the answer;  16-token windows counted from the start;  16-token windows counted back from
    the end;  only the first 8, 16, 32, 64, 128 tokens;  and 73's six ways of using the tokens.

HOW TO READ IT.
  Same-question AUROC near 0.5 for every detector -> what earlier results separated was the prompt, not the answer.
  Clearly above 0.5, flat across the tenths       -> a wrong answer differs from its first tokens on; a level
                                                     shift again, with the prompt ruled out as its cause.
  Near 0.5 early and rising along the answer      -> the answers start alike and part ways at some point;
                                                     that point is what a model of the token axis would locate.

  python 78_math_same_question.py --self-test
  python 78_math_same_question.py --model_folder qwen-2.5-7b-instruct
"""

import argparse
import importlib.util
import json
import os
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "token_structure")
N_FOLDS = 5
N_BOOT = 2000
TOKENS_PER_TRAINING_ANSWER = 64
N_TENTHS = 10
WINDOW = 16
WINDOWS_FROM_START = 16
WINDOWS_BEFORE_END = 8
FIRST_TOKENS = (8, 16, 32, 64, 128)
CHUNK = 65536
SCORERS = ("same_question_direction", "pooled_probe")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------------------------

def question_folds(pid, y, n_folds, seed):
    """Fold of every answer. A question's answers always share a fold. Questions are dealt out by kind
    (every answer right, both outcomes, every answer wrong), so each fold tests its share of each kind."""
    pid, y = np.asarray(pid), np.asarray(y)
    q, inv = np.unique(pid, return_inverse=True)
    n = np.bincount(inv)
    wrong = np.bincount(inv, weights=y)
    kind = np.where(wrong == 0, 0, np.where(wrong == n, 2, 1))
    rng = np.random.default_rng(seed)
    fold_of_q = np.empty(len(q), dtype=np.int64)
    start = 0
    for k in (1, 0, 2):
        members = rng.permutation(np.flatnonzero(kind == k))
        fold_of_q[members] = (np.arange(len(members)) + start) % n_folds
        start += len(members)
    return fold_of_q[inv], kind[inv]


# ---------------------------------------------------------------------------------------------
# same-question comparison
# ---------------------------------------------------------------------------------------------

def mean_ignoring_missing(a, axis):
    ok = np.isfinite(a)
    n = ok.sum(axis)
    return np.where(n > 0, np.where(ok, a, 0.0).sum(axis) / np.maximum(n, 1), np.nan)


def same_question_stats(S, y, pid):
    """S: (N,) or (N, B) scores; NaN = this answer has no score in that column. For every question with both
    a wrong and a right answer, per column: the pairs in which the wrong answer scores higher (ties count
    half), the pairs compared, and mean wrong score minus mean right score."""
    S = np.asarray(S, dtype=np.float64)
    if S.ndim == 1:
        S = S[:, None]
    y, pid = np.asarray(y), np.asarray(pid)
    order = np.argsort(pid, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(pid[order])) + 1)
    ids, higher, pairs, gap = [], [], [], []
    for g in groups:
        w, r = S[g[y[g] == 1]], S[g[y[g] == 0]]
        if len(w) == 0 or len(r) == 0:
            continue
        d = w[:, None, :] - r[None, :, :]
        ok = np.isfinite(d)
        higher.append(((d > 0) & ok).sum((0, 1)) + 0.5 * ((d == 0) & ok).sum((0, 1)))
        pairs.append(ok.sum((0, 1)))
        gap.append(mean_ignoring_missing(w, 0) - mean_ignoring_missing(r, 0))
        ids.append(pid[g[0]])
    B = S.shape[1]
    return {"questions": np.asarray(ids), "higher": np.asarray(higher, dtype=np.float64).reshape(-1, B),
            "pairs": np.asarray(pairs, dtype=np.float64).reshape(-1, B),
            "gap": np.asarray(gap, dtype=np.float64).reshape(-1, B)}


def _share(h, p):
    return np.where(p > 0, h / np.maximum(p, 1), np.nan)


def summarize(st, n_boot, seed):
    """Per column: same-question AUROC and same-question gap, each with a 95% interval from resampling
    questions, and how many pairs and questions stand behind it."""
    higher, pairs, gap = st["higher"], st["pairs"], st["gap"]
    Q = len(higher)
    pick = np.random.default_rng(seed).integers(0, Q, size=(n_boot, Q))
    auroc_b = _share(higher[pick].sum(1), pairs[pick].sum(1))
    gap_b = mean_ignoring_missing(gap[pick], 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a_lo, a_hi = np.nanpercentile(auroc_b, [2.5, 97.5], axis=0)
        g_lo, g_hi = np.nanpercentile(gap_b, [2.5, 97.5], axis=0)
    return {"same_question_auroc": _share(higher.sum(0), pairs.sum(0)).tolist(),
            "auroc_interval": [a_lo.tolist(), a_hi.tolist()],
            "gap": mean_ignoring_missing(gap, 0).tolist(), "gap_interval": [g_lo.tolist(), g_hi.tolist()],
            "pairs": pairs.sum(0).astype(int).tolist(), "questions": (pairs > 0).sum(0).astype(int).tolist()}


def auroc_difference(st_a, st_b, n_boot, seed):
    """Same-question AUROC of a minus b (first column of each). Both are resampled on the same questions."""
    if not np.array_equal(st_a["questions"], st_b["questions"]):
        raise ValueError("the two detectors were not scored on the same questions")
    Q = len(st_a["higher"])
    pick = np.random.default_rng(seed).integers(0, Q, size=(n_boot, Q))

    def auroc(st, rows):
        return st["higher"][rows, 0].sum(-1) / st["pairs"][rows, 0].sum(-1)
    d = auroc(st_a, pick) - auroc(st_b, pick)
    lo, hi = np.percentile(d, [2.5, 97.5])
    every = np.arange(Q)
    return {"difference": float(auroc(st_a, every) - auroc(st_b, every)), "interval": [float(lo), float(hi)]}


# ---------------------------------------------------------------------------------------------
# where in the answer: token scores averaged over parts of an answer
# ---------------------------------------------------------------------------------------------

def by_tenth(s, offsets, idx, n_bins):
    """(answers, n_bins): mean token score in each bin of normalized position (t + 0.5) / T, as 73 bins it."""
    out = np.full((len(idx), n_bins), np.nan)
    for i, n in enumerate(idx):
        v = s[offsets[n]:offsets[n + 1]]
        T = len(v)
        b = np.minimum(((np.arange(T) + 0.5) / T * n_bins).astype(int), n_bins - 1)
        k = np.bincount(b, minlength=n_bins)
        hit = k > 0
        out[i, hit] = np.bincount(b, weights=v, minlength=n_bins)[hit] / k[hit]
    return out


def by_window(s, offsets, idx, window, n_windows, from_end):
    """(answers, n_windows): mean token score in consecutive windows of `window` tokens, counted from the
    start or back from the end. Only complete windows count, so every entry averages the same number of
    tokens; an answer too short to reach a window has NaN there."""
    out = np.full((len(idx), n_windows), np.nan)
    for i, n in enumerate(idx):
        v = s[offsets[n]:offsets[n + 1]]
        if from_end:
            v = v[::-1]
        k = min(n_windows, len(v) // window)
        if k:
            out[i, :k] = v[:k * window].reshape(k, window).mean(1)
    return out


def first_tokens(s, offsets, idx, ks):
    """(answers, len(ks) + 1): mean score of at most the first k tokens for each k; last column, all tokens."""
    out = np.empty((len(idx), len(ks) + 1))
    for i, n in enumerate(idx):
        v = s[offsets[n]:offsets[n + 1]]
        for j, k in enumerate(ks):
            out[i, j] = v[:k].mean()
        out[i, -1] = v.mean()
    return out


def position_tables(s, offsets, keep, s73):
    """{name: (matrix over the evaluated answers, column labels)} for one token scorer."""
    agg = s73.answer_aggregates(s, offsets, keep)
    return {
        "by_tokens_used": (np.stack([agg[a] for a in s73.AGGREGATES], axis=1), list(s73.AGGREGATES)),
        "by_tenth_of_answer": (by_tenth(s, offsets, keep, N_TENTHS), [str(b + 1) for b in range(N_TENTHS)]),
        "by_tokens_from_start": (by_window(s, offsets, keep, WINDOW, WINDOWS_FROM_START, False),
                                 ["%d-%d" % (k * WINDOW + 1, (k + 1) * WINDOW) for k in range(WINDOWS_FROM_START)]),
        "by_tokens_before_end": (by_window(s, offsets, keep, WINDOW, WINDOWS_BEFORE_END, True),
                                 ["%d-%d" % (k * WINDOW + 1, (k + 1) * WINDOW) for k in range(WINDOWS_BEFORE_END)]),
        "using_first_tokens": (first_tokens(s, offsets, keep, FIRST_TOKENS), [str(k) for k in FIRST_TOKENS] + ["all"]),
    }


# ---------------------------------------------------------------------------------------------
# the two token scorers
# ---------------------------------------------------------------------------------------------

def training_tokens(offsets, idx, per_answer, rng):
    """Token rows used for fitting: every token of a short answer, `per_answer` randomly placed tokens of a
    longer one. Returns (rows, offsets of the answers within rows)."""
    rows, n = [], []
    for a in idx:
        start, T = int(offsets[a]), int(offsets[a + 1] - offsets[a])
        pick = np.arange(T) if T <= per_answer else np.sort(rng.choice(T, size=per_answer, replace=False))
        rows.append(start + pick)
        n.append(len(pick))
    sub = np.zeros(len(n) + 1, dtype=np.int64)
    np.cumsum(n, out=sub[1:])
    return np.concatenate(rows), sub


def answer_means(tokens, offsets, idx):
    """(answers, P * r) float32: mean over all of an answer's tokens of its stored coordinates."""
    F = tokens.shape[1] * tokens.shape[2]
    out = np.empty((len(idx), F), dtype=np.float32)
    for i, n in enumerate(idx):
        out[i] = np.asarray(tokens[offsets[n]:offsets[n + 1]], dtype=np.float32).mean(0).reshape(F)
    return out


def same_question_direction(A, y, pid):
    """A: (answers, F) mean states on the common scale. For each question with both outcomes, mean of its
    wrong answers minus mean of its right answers; the direction is the average over those questions, at
    unit length. Returns (direction, questions used)."""
    y, pid = np.asarray(y), np.asarray(pid)
    order = np.argsort(pid, kind="stable")
    diffs = []
    for g in np.split(order, np.flatnonzero(np.diff(pid[order])) + 1):
        w, r = A[g[y[g] == 1]], A[g[y[g] == 0]]
        if len(w) and len(r):
            diffs.append(w.mean(0) - r.mean(0))
    if not diffs:
        raise ValueError("no training question has both a right and a wrong answer")
    d = np.mean(diffs, axis=0).astype(np.float64)
    return d / (np.linalg.norm(d) + 1e-12), len(diffs)


def to_standard_deviations(s, offsets, right_idx, rows):
    """Rescale s[rows] in place so that 0 is the average token of the right answers in right_idx and 1 is
    one standard deviation of their tokens, every answer counting once. Returns (shift, scale)."""
    mu = float(np.mean([s[offsets[n]:offsets[n + 1]].mean() for n in right_idx]))
    sd = float(np.sqrt(np.mean([np.mean((s[offsets[n]:offsets[n + 1]] - mu) ** 2) for n in right_idx])))
    s[rows] = (s[rows] - mu) / (sd + 1e-12)
    return mu, sd


def token_scores(tokens, offsets, y, pid, keep, fold, seed, per_answer, s73, log=print):
    """Held-out token scores of both scorers, in standard deviations of right-answer tokens.
    Returns ({scorer: (total_T,) float64, NaN outside the evaluated answers}, one note per fold)."""
    F = tokens.shape[1] * tokens.shape[2]
    means = answer_means(tokens, offsets, keep)
    scores = {name: np.full(int(offsets[-1]), np.nan) for name in SCORERS}
    notes = []
    for f in range(int(fold.max()) + 1):
        t0 = time.time()
        tr_l = np.flatnonzero(fold != f)
        tr, te = keep[tr_l], keep[fold == f]
        rows, sub = training_tokens(offsets, tr, per_answer, np.random.default_rng(seed + f))
        X = np.asarray(tokens[rows], dtype=np.float32).reshape(len(rows), F)
        w, yt = s73.token_weights(sub, np.arange(len(tr)), y[tr])
        sc, clf = s73.fit_probe(X, yt, w, seed)
        del X
        mu, sd = sc.mean_.astype(np.float32), sc.scale_.astype(np.float32)
        direction, n_q = same_question_direction((means[tr_l] - mu) / sd, y[tr], pid[tr])
        coef = clf.coef_.ravel().astype(np.float64)
        cosine = float(coef @ direction / (np.linalg.norm(coef) + 1e-12))

        rows_te = s73.token_rows(offsets, te)
        for a in range(0, len(rows_te), CHUNK):
            r = rows_te[a:a + CHUNK]
            Z = (np.asarray(tokens[r], dtype=np.float32).reshape(len(r), F) - mu) / sd
            scores["pooled_probe"][r] = clf.decision_function(Z)
            scores["same_question_direction"][r] = Z.astype(np.float64) @ direction
        for name in SCORERS:
            to_standard_deviations(scores[name], offsets, te[y[te] == 0], rows_te)
        notes.append({"fold": f, "training_answers": int(len(tr)), "training_tokens": int(len(rows)),
                      "training_questions_with_both_outcomes": int(n_q), "test_answers": int(len(te)),
                      "cosine_probe_weights_vs_same_question_direction": cosine,
                      "seconds": round(time.time() - t0, 1)})
        log("    token scorers, fold %d: fitted on %d tokens of %d answers; direction from %d questions; "
            "cosine with the probe's weights %.3f (%.0fs)" % (f, len(rows), len(tr), n_q, cosine, time.time() - t0))
    return scores, notes


# ---------------------------------------------------------------------------------------------
# LOTUS, through the pinned code
# ---------------------------------------------------------------------------------------------

def load_lotus(states_dir, keep):
    """The three tensors 44.load_new_dataset builds, for the evaluated answers only."""
    def g(k):
        return np.asarray(np.load(os.path.join(states_dir, "lotus_%s.npy" % k), mmap_mode="r")[keep], dtype=np.float32)
    return {"core": g("core"), "static": np.concatenate([g("q95"), g("q05")], axis=2),
            "velocity": np.concatenate([g("v95"), g("v05")], axis=2)}


def lotus_scores(feats, y, fold, seed, log=print):
    """Held-out LOTUS score of every answer: scaling, bases and forest fitted on the other folds."""
    s61 = _load("s61", "61_flatten_control.py")
    m = s61.mods()
    spec = m["s44"].CONDITION_SPECS[s61.DEFAULT_CONDITION]
    width = sum(r_l * r_d for _, r_l, r_d in spec)
    out = np.full(len(y), np.nan)
    for f in range(int(fold.max()) + 1):
        t0 = time.time()
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        Z = s61.build("hosvd", feats, spec, tr, seed)
        if Z.shape[1] != width:
            raise SystemExit("LOTUS features have width %d, expected %d" % (Z.shape[1], width))
        out[te] = m["s26"].fit_eval("RF", Z[tr], y[tr], Z[te], seed)
        log("    LOTUS, fold %d: %d features, fitted on %d answers (%.0fs)" % (f, Z.shape[1], len(tr), time.time() - t0))
    return out, width


# ---------------------------------------------------------------------------------------------
# descriptives
# ---------------------------------------------------------------------------------------------

def shared_start(answers, y, pid):
    """For every wrong answer whose question also has a right answer: how many leading tokens it shares with
    the right answer it follows longest, in tokens and as a share of its own length. Two answers with the
    same tokens so far have the same hidden states so far, so nothing can separate them before this point."""
    y, pid = np.asarray(y), np.asarray(pid)
    order = np.argsort(pid, kind="stable")
    tokens, share = [], []
    for g in np.split(order, np.flatnonzero(np.diff(pid[order])) + 1):
        right = [answers[i] for i in g[y[g] == 0]]
        if not right:
            continue
        for i in g[y[g] == 1]:
            a, best = answers[i], 0
            for b in right:
                n = min(len(a), len(b))
                differ = np.flatnonzero(a[:n] != b[:n])
                best = max(best, int(differ[0]) if len(differ) else n)
            tokens.append(best)
            share.append(best / max(len(a), 1))
    return np.asarray(tokens), np.asarray(share)


def quartiles(v):
    return {"p25": float(np.percentile(v, 25)), "median": float(np.median(v)), "p75": float(np.percentile(v, 75))}


def mean_std(v):
    v = [x for x in v if x is not None]
    return [float(np.mean(v)), float(np.std(v))] if v else [None, None]


def clean(o):
    """NaN is not valid JSON; store it as null."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ---------------------------------------------------------------------------------------------

def analyze(tokens, offsets, y, pid, keep, nll, answers, feats, s73, pooled_auroc, seed, n_folds, n_boot,
            per_answer, checkpoint=None, log=print):
    """Everything after loading. `keep` are the evaluated answers; feats (LOTUS tensors over keep) may be None."""
    y_k, pid_k = y[keep], pid[keep]
    fold, kind = question_folds(pid_k, y_k, n_folds, seed)
    both = kind == 1
    lengths = np.diff(offsets)[keep]
    out = {"n_folds": n_folds, "n_boot": n_boot, "seed": seed, "tokens_per_training_answer": per_answer,
           "counts": {"answers_evaluated": int(len(keep)), "questions_evaluated": int(len(np.unique(pid_k))),
                      "answers_wrong": int(y_k.sum()),
                      "questions_with_both_outcomes": int(len(np.unique(pid_k[both]))),
                      "wrong_answers_in_those_questions": int(y_k[both].sum()),
                      "right_answers_in_those_questions": int((y_k[both] == 0).sum()),
                      "questions_all_right": int(len(np.unique(pid_k[kind == 0]))),
                      "questions_all_wrong": int(len(np.unique(pid_k[kind == 2])))},
           "answer_tokens": {"all": quartiles(lengths), "wrong_in_questions_with_both_outcomes": quartiles(lengths[both & (y_k == 1)]),
                             "right_in_questions_with_both_outcomes": quartiles(lengths[both & (y_k == 0)])}}
    if answers is not None:
        tok, share = shared_start([answers[i] for i in keep], y_k, pid_k)
        out["tokens_shared_with_closest_right_answer"] = {"tokens": quartiles(tok), "share_of_answer": quartiles(share),
                                                          "wrong_answers": int(len(tok))}

    def answer_row(score):
        st = same_question_stats(score, y_k, pid_k)
        s = summarize(st, n_boot, seed)
        per_fold = [pooled_auroc(score[fold == f], y_k[fold == f]) for f in range(n_folds)]
        return {"same_question_auroc": s["same_question_auroc"][0],
                "interval": [s["auroc_interval"][0][0], s["auroc_interval"][1][0]],
                "pooled_auroc": mean_std(per_fold), "pairs": s["pairs"][0], "questions": s["questions"][0]}, st

    rows, stats = {}, {}
    rows["answer_length"], stats["answer_length"] = answer_row(lengths.astype(np.float64))
    if nll is not None:
        rows["answer_nll"], stats["answer_nll"] = answer_row(np.asarray(nll, dtype=np.float64)[keep])
    out["counts"]["same_question_pairs"] = rows["answer_length"]["pairs"]

    t0 = time.time()
    scores, notes = token_scores(tokens, offsets, y, pid, keep, fold, seed, per_answer, s73, log)
    out["token_scorer_folds"] = notes
    out["token_scorers"] = {}
    for name in SCORERS:
        res = {}
        for table, (M, columns) in position_tables(scores[name], offsets, keep, s73).items():
            res[table] = dict(summarize(same_question_stats(M, y_k, pid_k), n_boot, seed), columns=columns)
            if table == "using_first_tokens":
                rows[name + "_mean_of_tokens"], stats[name + "_mean_of_tokens"] = answer_row(M[:, -1])
            if table == "by_tokens_used":
                res[table]["pooled_auroc"] = [mean_std([pooled_auroc(M[fold == f, j], y_k[fold == f])
                                                        for f in range(n_folds)]) for j in range(M.shape[1])]
        out["token_scorers"][name] = res
    log("    token scorers and where-in-the-answer done (%.0fs)" % (time.time() - t0))
    out["answer_level"] = rows
    if checkpoint:
        checkpoint(out)

    if feats is not None:
        t0 = time.time()
        lotus, width = lotus_scores(feats, y_k, fold, seed, log)
        rows["lotus"], stats["lotus"] = answer_row(lotus)
        rows["lotus"]["features"] = int(width)
        out["answer_level_minus_lotus"] = {k: auroc_difference(stats[k], stats["lotus"], n_boot, seed)
                                           for k in rows if k != "lotus"}
        log("    LOTUS done (%.0fs)" % (time.time() - t0))
    return out, fold


def fmt(v, lo=None, hi=None):
    if v is None or not np.isfinite(v):
        return "  n/a"
    return "%.3f" % v if lo is None else "%.3f [%.3f, %.3f]" % (v, lo, hi)


def print_summary(out, title):
    c, t = out["counts"], out["answer_tokens"]
    print("\n  SUMMARY %s" % title)
    print("    evaluated: %d answers to %d questions; %d questions have both right and wrong answers"
          % (c["answers_evaluated"], c["questions_evaluated"], c["questions_with_both_outcomes"]))
    print("               (%d wrong, %d right, %d same-question pairs); %d all right, %d all wrong"
          % (c["wrong_answers_in_those_questions"], c["right_answers_in_those_questions"], c["same_question_pairs"],
             c["questions_all_right"], c["questions_all_wrong"]))
    print("    answer length in those questions, median tokens: wrong %.0f, right %.0f"
          % (t["wrong_in_questions_with_both_outcomes"]["median"], t["right_in_questions_with_both_outcomes"]["median"]))
    if "tokens_shared_with_closest_right_answer" in out:
        s = out["tokens_shared_with_closest_right_answer"]
        print("    a wrong answer repeats its closest right answer for a median %.0f tokens (%.0f%% of its length) "
              "before the text differs" % (s["tokens"]["median"], 100 * s["share_of_answer"]["median"]))
    print("\n    CAN A RIGHT AND A WRONG ANSWER TO THE SAME QUESTION BE TOLD APART?  (same-question AUROC; 0.5 = no)")
    names = [("lotus", "LOTUS"), ("pooled_probe_mean_of_tokens", "pooled token probe, mean of tokens"),
             ("same_question_direction_mean_of_tokens", "same-question direction, mean of tokens"),
             ("answer_length", "answer length alone"), ("answer_nll", "answer NLL alone")]
    for key, label in names:
        if key in out["answer_level"]:
            r = out["answer_level"][key]
            print("      %-40s %s   pooled %s +- %s" % (label, fmt(r["same_question_auroc"], *r["interval"]),
                                                       fmt(r["pooled_auroc"][0]), fmt(r["pooled_auroc"][1])))
    for key, label in names[1:3]:
        d = out.get("answer_level_minus_lotus", {}).get(key)
        if d:
            print("      %s minus LOTUS: %+.3f [%+.3f, %+.3f]" % (label, d["difference"], d["interval"][0], d["interval"][1]))
    for name, label in (("same_question_direction", "same-question direction"), ("pooled_probe", "pooled token probe")):
        r = out["token_scorers"][name]
        print("\n    WHERE IN THE ANSWER?  %s" % label)
        for table, head in (("by_tenth_of_answer", "tenth of the answer, first to last"),
                            ("by_tokens_from_start", "%d-token windows from the start" % WINDOW),
                            ("by_tokens_before_end", "%d-token windows back from the end, last window first" % WINDOW)):
            v = r[table]
            print("      %s" % head)
            print("        same-question AUROC      : %s" % " ".join(fmt(x) for x in v["same_question_auroc"]))
            print("        gap, standard deviations : %s" % " ".join("%+.2f" % x if x is not None and np.isfinite(x) else "  n/a" for x in v["gap"]))
            print("        pairs                    : %s" % " ".join("%5d" % x for x in v["pairs"]))
        v = r["using_first_tokens"]
        print("      using only the first %s tokens / all: %s" % (" / ".join(v["columns"][:-1]),
                                                                " ".join(fmt(x) for x in v["same_question_auroc"])))
        v = r["by_tokens_used"]
        print("      tokens used %s: %s" % (" / ".join(v["columns"]), " ".join(fmt(x) for x in v["same_question_auroc"])))


def run(model_folder, dataset, out_dir, seed, n_folds, n_boot, per_answer, skip_lotus):
    import torch
    import yaml
    import methods.base as B
    c = B.canonical()
    s73 = _load("s73", "73_token_structure.py")
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    root = os.path.normpath(os.path.join(HERE, cfg["math"]["out_dir"], model_folder))
    d = os.path.join(root, "%s_states" % dataset)
    seq_path = os.path.join(root, "%s_sequences_v1.pt" % dataset)
    if not os.path.exists(os.path.join(d, "meta.json")):
        raise SystemExit("%s is missing or incomplete -- run 77_extract_math_states.py first" % d)
    meta = json.load(open(os.path.join(d, "meta.json")))
    if any(meta["nonfinite_entries"].values()):
        raise SystemExit("the state store holds non-finite entries: %s" % meta["nonfinite_entries"])

    seq = torch.load(seq_path, weights_only=False)
    y = np.asarray(seq["all_hallucination_flag"]).astype(int)            # 1 = wrong final number
    pid = np.asarray(seq["prompt_id"], dtype=np.int64)
    answers = [np.asarray(ids)[int(pl):] for ids, pl in zip(seq["input_ids"], seq["prompt_len"])]
    offsets = np.load(os.path.join(d, "offsets.npy")).astype(np.int64)
    if not np.array_equal(pid, np.load(os.path.join(d, "prompt_id.npy")).astype(np.int64)):
        raise SystemExit("the state store and the sequences file disagree on question ids")
    if not np.array_equal(np.diff(offsets), np.array([len(a) for a in answers])):
        raise SystemExit("the state store and the sequences file disagree on answer lengths")
    keep = np.flatnonzero(~np.load(os.path.join(d, "basis_question.npy")))
    nll = np.load(os.path.join(d, "nll.npy"))
    print("  [%s/%s] %d answers, %.1f%% wrong | %d answers to basis questions left out, %d evaluated"
          % (model_folder, dataset, len(y), 100 * y.mean(), len(y) - len(keep), len(keep)), flush=True)

    t0 = time.time()
    tokens = np.load(os.path.join(d, "tokens.npy"))
    feats = None if skip_lotus else load_lotus(d, keep)
    print("  loaded %d tokens x %d blocks x %d directions%s (%.0fs)"
          % (tokens.shape + ("" if skip_lotus else ", and LOTUS's summaries", time.time() - t0)), flush=True)

    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "math_%s_%s.json" % (model_folder, dataset))
    head = {"model_folder": model_folder, "dataset": dataset, "blocks": meta["blocks"],
            "protocol": "%d folds of whole questions; basis questions left out" % n_folds,
            "store": {k: meta[k] for k in ("repo_head", "written", "basis") if k in meta},
            "decoding": seq.get("decoding_config"), "complete": False}

    def checkpoint(part):
        json.dump(clean(dict(head, **part)), open(dst, "w"), indent=1)

    def log(msg):
        print(msg, flush=True)

    out, fold = analyze(tokens, offsets, y, pid, keep, nll, answers, feats, s73, c["pooled_auroc"], seed, n_folds,
                        n_boot, per_answer, checkpoint, log)

    # the same-question AUROC here must be the paper's within-question AUROC
    for key in ("answer_length",):
        mine = out["answer_level"][key]["same_question_auroc"]
        ref = c["within_prompt_auroc"](np.diff(offsets)[keep].astype(float), y[keep], pid[keep])["within_prompt_auroc"]
        if abs(mine - ref) > 1e-9:
            raise SystemExit("same-question AUROC %.6f differs from the canonical within-question AUROC %.6f" % (mine, ref))
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    out = dict(head, **out)
    out["complete"] = True
    json.dump(clean(out), open(dst, "w"), indent=1)
    print_summary(clean(out), "%s / %s" % (model_folder, dataset))
    print("  wrote %s" % dst)


# ---------------------------------------------------------------------------------------------

def self_test():
    """Synthetic answers with planted properties; every measurement must recover what was planted."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print("    [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
        ok = ok and bool(cond)

    def pooled(s, y):
        return float(roc_auc_score(y, s)) if 0 < np.sum(y) < len(y) else None

    # --- folds
    pid = np.repeat(np.arange(100, 300), 10)
    y = np.zeros(len(pid), dtype=int)
    for q in range(200):
        k = 0 if q < 120 else (10 if q >= 180 else int(rng.integers(1, 10)))
        y[q * 10 + rng.permutation(10)[:k]] = 1
    fold, kind = question_folds(pid, y, 5, 0)
    per_q = np.array([len(np.unique(fold[pid == q])) for q in np.unique(pid)])
    check("a question's answers share a fold", (per_q == 1).all())
    both_per_fold = [len(np.unique(pid[(fold == f) & (kind == 1)])) for f in range(5)]
    check("questions with both outcomes are spread evenly", max(both_per_fold) - min(both_per_fold) <= 1, str(both_per_fold))
    check("folds are the same size", np.ptp(np.bincount(fold)) <= 10, str(np.bincount(fold).tolist()))
    check("folds change with the seed", not np.array_equal(fold, question_folds(pid, y, 5, 1)[0]))

    # --- same-question AUROC: equals the canonical one, and ignores what differs between questions
    s = rng.normal(size=len(pid))
    st = same_question_stats(s, y, pid)
    mine = summarize(st, 200, 0)
    try:
        import methods.base as B
        ref = B.canonical()["within_prompt_auroc"](s, y, pid)
        check("equals the canonical within-question AUROC", abs(mine["same_question_auroc"][0] - ref["within_prompt_auroc"]) < 1e-12
              and mine["pairs"][0] == ref["n_pairs"], "%.6f vs %.6f, %d pairs" % (mine["same_question_auroc"][0],
                                                                                 ref["within_prompt_auroc"], ref["n_pairs"]))
    except Exception as e:                                               # the canonical chain needs the full repo env
        print("    [SKIP] canonical within-question AUROC not importable here (%s)" % type(e).__name__)
    hard = np.repeat(np.array([y[pid == q].mean() for q in np.unique(pid)]), 10)
    s_question = 3.0 * hard + rng.normal(size=len(pid))
    m = summarize(same_question_stats(s_question, y, pid), 500, 0)
    check("a score that only knows the question: pooled high, same-question at chance",
          roc_auc_score(y, s_question) > 0.75 and abs(m["same_question_auroc"][0] - 0.5) < 0.06,
          "pooled %.3f, same-question %.3f" % (roc_auc_score(y, s_question), m["same_question_auroc"][0]))
    check("the interval contains the estimate", m["auroc_interval"][0][0] <= m["same_question_auroc"][0] <= m["auroc_interval"][1][0],
          "%.3f in [%.3f, %.3f]" % (m["same_question_auroc"][0], m["auroc_interval"][0][0], m["auroc_interval"][1][0]))
    s_answer = y + 0.8 * rng.normal(size=len(pid))
    d = auroc_difference(same_question_stats(s_answer, y, pid), st, 500, 0)
    check("difference between detectors: a real signal beats noise", d["interval"][0] > 0.1, "%+.3f [%+.3f, %+.3f]" % (
        d["difference"], d["interval"][0], d["interval"][1]))
    d0 = auroc_difference(st, st, 200, 0)
    check("difference of a detector with itself is zero", d0["difference"] == 0 and d0["interval"] == [0.0, 0.0])
    S2 = np.stack([s_answer, np.where(rng.random(len(pid)) < 0.3, np.nan, s_answer)], axis=1)
    m2 = summarize(same_question_stats(S2, y, pid), 200, 0)
    check("missing scores drop pairs, not answers", m2["pairs"][1] < m2["pairs"][0] and abs(m2["same_question_auroc"][1] - m2["same_question_auroc"][0]) < 0.05,
          "%d of %d pairs kept" % (m2["pairs"][1], m2["pairs"][0]))

    # --- parts of an answer
    offs = np.array([0, 40, 50, 53])
    v = np.arange(53, dtype=np.float64)
    idx = np.arange(3)
    w = by_window(v, offs, idx, 16, 4, False)
    check("windows from the start", np.allclose(w[0, :2], [7.5, 23.5]) and np.isnan(w[0, 2]) and np.isnan(w[1, 0]),
          str(np.round(w[0], 1).tolist()))
    w = by_window(v, offs, idx, 16, 4, True)
    check("windows back from the end", np.allclose(w[0, :2], [31.5, 15.5]) and np.isnan(w[0, 2]), str(np.round(w[0], 1).tolist()))
    t = by_tenth(v, offs, idx, 10)
    check("tenths of an answer", np.allclose(t[0], np.arange(10) * 4 + 1.5) and np.allclose(t[1], np.arange(40, 50))
          and np.isnan(t[2]).sum() == 7, str(np.round(t[0], 1).tolist()))
    ft = first_tokens(v, offs, idx, (8, 16))
    check("first tokens", np.allclose(ft[0], [3.5, 7.5, 19.5]) and np.allclose(ft[2], [51.0, 51.0, 51.0]))
    rows, sub = training_tokens(offs, idx, 8, np.random.default_rng(0))
    check("training tokens: at most 8 per answer, inside the answer, no repeats",
          np.diff(sub).tolist() == [8, 8, 3] and len(np.unique(rows)) == len(rows)
          and all(offs[i] <= r < offs[i + 1] for i in range(3) for r in rows[sub[i]:sub[i + 1]]), str(np.diff(sub).tolist()))

    # --- shared start
    ans = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 9, 9]), np.array([1, 2, 3, 7]), np.array([5, 5])]
    tok, share = shared_start(ans, np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]))
    check("shared start with the closest right answer", tok.tolist() == [3] and abs(share[0] - 0.75) < 1e-12, str(tok.tolist()))

    # --- end to end: wrong answers move along one direction, from 60% of the answer on. A second direction
    #     marks all-wrong questions and nothing else; the same-question direction must not pick it up.
    s73 = _load("s73", "73_token_structure.py")
    Q, P, r = 150, 3, 8
    pid = np.repeat(np.arange(Q), 10)
    y = np.zeros(Q * 10, dtype=int)
    all_wrong = np.zeros(Q, dtype=bool)
    for q in range(Q):
        if q < 60:
            y[q * 10 + rng.permutation(10)[:int(rng.integers(2, 6))]] = 1
        elif q >= 130:
            y[q * 10:(q + 1) * 10] = 1
            all_wrong[q] = True
    lengths = rng.integers(40, 81, size=Q * 10)
    offsets = np.zeros(Q * 10 + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    tokens = rng.normal(size=(offsets[-1], P, r))
    question_offset = rng.normal(size=(Q, P, r))
    for n in range(Q * 10):
        a, b = offsets[n], offsets[n + 1]
        tokens[a:b] += question_offset[pid[n]]
        if all_wrong[pid[n]]:
            tokens[a:b, 0, 1] += 3.0
        if y[n]:
            tokens[a + int(0.6 * (b - a)):b, 0, 0] += 1.5
    tokens = tokens.astype(np.float16)
    basis = np.isin(pid, rng.choice(Q, size=15, replace=False))
    keep = np.flatnonzero(~basis)
    quiet = lambda msg: None                                             # noqa: E731
    out, fold = analyze(tokens, offsets, y, pid, keep, None, None, None, s73, pooled, 0, 5, 300, 32, None, quiet)
    c = out["counts"]
    check("basis questions are left out", c["answers_evaluated"] == len(keep) and c["questions_evaluated"] == Q - 15,
          "%d answers, %d questions" % (c["answers_evaluated"], c["questions_evaluated"]))
    cos = [n["cosine_probe_weights_vs_same_question_direction"] for n in out["token_scorer_folds"]]
    d = out["token_scorers"]["same_question_direction"]
    tenth = np.array(d["by_tenth_of_answer"]["same_question_auroc"])
    gap = np.array(d["by_tenth_of_answer"]["gap"])
    check("early tenths: no separation", np.abs(tenth[:5] - 0.5).max() < 0.08, " ".join("%.2f" % x for x in tenth[:5]))
    check("late tenths: clear separation", tenth[7:].min() > 0.85, " ".join("%.2f" % x for x in tenth[7:]))
    check("gap in standard deviations: about zero early, about one late", np.abs(gap[:5]).max() < 0.2 and 0.7 < gap[7:].mean() < 1.4,
          "early %s, late %.2f" % (" ".join("%+.2f" % x for x in gap[:5]), gap[7:].mean()))
    first = d["using_first_tokens"]["same_question_auroc"]
    check("first 8 tokens tell nothing, the whole answer does", abs(first[0] - 0.5) < 0.08 and first[-1] > 0.8,
          "first 8: %.3f, all: %.3f" % (first[0], first[-1]))
    end, start = d["by_tokens_before_end"], d["by_tokens_from_start"]
    check("the last 16 tokens separate, the first 16 do not", end["same_question_auroc"][0] > 0.9 and abs(start["same_question_auroc"][0] - 0.5) < 0.08,
          "last %.3f, first %.3f" % (end["same_question_auroc"][0], start["same_question_auroc"][0]))
    check("windows no answer reaches are empty, not zero", start["pairs"][-1] == 0 and start["same_question_auroc"][-1] is not None
          and not np.isfinite(start["same_question_auroc"][-1]), "pairs %s" % start["pairs"][:6])
    a = out["answer_level"]
    check("answer-level rows are present", {"answer_length", "pooled_probe_mean_of_tokens", "same_question_direction_mean_of_tokens"} <= set(a))
    check("answer length alone tells nothing here", abs(a["answer_length"]["same_question_auroc"] - 0.5) < 0.08,
          "%.3f" % a["answer_length"]["same_question_auroc"])
    check("the same-question direction is not the pooled probe's", max(cos) < 0.9, "cosine per fold %s" % " ".join("%.2f" % x for x in cos))
    p = out["token_scorers"]["pooled_probe"]["by_tokens_used"]
    col = p["columns"].index("second_half")
    check("pooled probe: second half beats first half", p["same_question_auroc"][col] > p["same_question_auroc"][p["columns"].index("first_half")] + 0.1,
          "first %.3f, second %.3f" % (p["same_question_auroc"][p["columns"].index("first_half")], p["same_question_auroc"][col]))
    check("the result is valid JSON once cleaned", "NaN" not in json.dumps(clean(out)))

    # --- LOTUS through the pinned code, on a small tensor with a planted answer-level signal
    try:
        N, D = len(keep), 80
        y_k = y[keep]
        feats = {"core": rng.normal(size=(N, 9, D)).astype(np.float32), "static": rng.normal(size=(N, 9, 2 * D)).astype(np.float32),
                 "velocity": rng.normal(size=(N, 8, 2 * D)).astype(np.float32)}
        feats["core"][y_k == 1, :, :4] += 1.0
        lot, width = lotus_scores(feats, y_k, fold, 0, quiet)
        m = summarize(same_question_stats(lot, y_k, pid[keep]), 200, 0)
        check("LOTUS runs through the pinned code and finds a planted signal", width == 896 and np.isfinite(lot).all()
              and m["same_question_auroc"][0] > 0.8, "%d features, same-question AUROC %.3f" % (width, m["same_question_auroc"][0]))
    except ImportError as e:
        print("    [SKIP] pinned LOTUS code not importable here (%s)" % e)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model_folder")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--tokens-per-training-answer", type=int, default=TOKENS_PER_TRAINING_ANSWER)
    ap.add_argument("--skip-lotus", action="store_true", help="token scorers only (minutes); LOTUS is the slow part")
    a = ap.parse_args()
    if a.self_test:
        print("  self-test")
        raise SystemExit(0 if self_test() else 1)
    if not a.model_folder:
        raise SystemExit("--model_folder is required (or --self-test)")
    run(a.model_folder, a.dataset, a.out_dir, a.seed, a.folds, a.n_boot, a.tokens_per_training_answer, a.skip_lotus)


if __name__ == "__main__":
    main()
