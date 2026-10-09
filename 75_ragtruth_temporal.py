"""
75_ragtruth_temporal.py -- do a model's hidden states register the moment it starts to hallucinate? (T-024)
==================================================================================================

WHY. On short QA (73, T-023) hallucinated answers scored above correct ones by the same amount at every
position: the hallucination was the whole answer, so the token axis held a level shift and nothing temporal.
RAGTruth responses run to a few hundred tokens with human-labeled hallucinated spans, so a response can be
right for a while and then go wrong. This script asks whether the hidden states register that moment.
Reads 74's store; CPU only; writes results/token_structure/ragtruth_<generator>.json.

SPLIT. RAGTruth's own train/test split (about 450 test responses per generator). 74 fitted the projection on
train responses; both probes here are fitted on train tokens; everything reported is on test responses,
with 95% intervals from resampling test responses. There is one response per source per generator, so
within-question AUROC does not exist here. Pooled AUROC is also reported per task (QA, Data2txt, Summary):
hallucination rates differ by task, and a pooled number can reward recognising the task instead of the
hallucination.

TESTS.

  TOKEN-TO-TOKEN SMOOTHNESS. 73's test, at distances 1..32 because responses are long: excess cosine
      similarity of states k tokens apart over a within-response shuffle.

  WHERE IN THE ANSWER. 73's test: a probe on single tokens, each token carrying its RESPONSE's label (no span
      information), scored per test response by mean / max / first / last / first half / second half, plus
      the class gap in 10 bins of normalized position.

  ONSET. The new test. Each hallucinated test response is lined up on its first hallucinated token (position
      0, from the human spans) and the response-label probe's token scores are averaged at positions -24..+24.
      Correct responses have no onset, so each gets a pseudo-onset at a fraction of its length drawn from the
      hallucinated TRAIN responses' onsets; their curve is the no-event baseline at matched positions. The
      response-label probe never saw a span, so a jump at position 0 is found, not taught.
          jump = (mean score at 0..+7 minus mean at -8..-1) in hallucinated responses
                 minus the same quantity in correct responses at their pseudo-onsets
      Also: how far hallucinated responses already score above correct ones BEFORE the onset (-8..-1).

  FIRST VERSUS LATER HALLUCINATED TOKENS. A second probe trained on the TOKEN labels (inside a span or not).
      AUROC of the first token of each hallucinated run against all non-hallucinated test tokens, and of the
      later tokens of runs against the same -- the comparison of Snel and Oh (arXiv:2507.20836), who report
      the first hallucinated token as far easier to detect. Intervals resample responses, not tokens.

HOW TO READ IT.
  Clear jump at the onset, flat baseline  -> the hidden states register the moment the model starts to
      hallucinate: a temporal event on the token axis, which pooling blurs and a token-axis model could use.
  No jump, hallucinated responses high before and after  -> a whole-response shift, as on short QA, even when
      the text goes wrong partway through.
  High BEFORE the onset  -> the state anticipates the hallucination; the signal is prospective.

  python 75_ragtruth_temporal.py --self-test
  python 75_ragtruth_temporal.py --generator llama-2-7b-chat
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "token_structure")
MAX_LAG = 32
WINDOW = 24
BEFORE = (-8, -1)
AFTER = (0, 7)
N_BOOT = 2000
N_BOOT_TOKENS = 500
N_BINS = 10
TASK_NAMES = {0: "QA", 1: "Data2txt", 2: "Summary"}
SHORT_LENGTHS = (17, 8)    # median short-QA answer lengths: Qwen TyDiQA-GP 17, LLaMA TyDiQA-GP 6-8


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def boot_ci(stat, n_items, n_boot, seed):
    """Point value and 95% interval of stat(index array) under resampling of the items with replacement."""
    rng = np.random.default_rng(seed)
    point = stat(np.arange(n_items))
    vals = []
    for _ in range(n_boot):
        v = stat(rng.integers(0, n_items, size=n_items))
        if v is not None and np.isfinite(v):
            vals.append(v)
    lo, hi = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (None, None)
    return {"value": point, "lo": lo, "hi": hi}


def response_auroc(score, y, n_boot, seed):
    def stat(ix):
        return auroc(y[ix], score[ix]) if len(np.unique(y[ix])) == 2 else None
    return boot_ci(stat, len(y), n_boot, seed)


# ---------------------------------------------------------------------------------------------
# onset
# ---------------------------------------------------------------------------------------------

def pseudo_onsets(lengths, fractions, rng):
    """For responses without an onset: a position at a fraction of the length drawn from `fractions`."""
    f = rng.choice(np.asarray(fractions, dtype=float), size=len(lengths), replace=True)
    return np.minimum((f * lengths).astype(np.int64), lengths - 1)


def aligned_curve(s, offsets, idx, onset, window):
    """Mean of token scores s at positions onset-window..onset+window over responses idx (onset[k] is the
    aligned position within response idx[k]), and how many responses reach each position."""
    W = 2 * window + 1
    sums, cnt = np.zeros(W), np.zeros(W, dtype=np.int64)
    for k, n in enumerate(idx):
        v = s[offsets[n]:offsets[n + 1]]
        o = int(onset[k])
        lo, hi = max(0, o - window), min(len(v), o + window + 1)
        rel = np.arange(lo, hi) - o + window
        sums[rel] += v[lo:hi]
        cnt[rel] += 1
    return sums / np.maximum(cnt, 1), cnt


def chunk_spans(offsets, idx, chunk):
    """Cut each response in idx into consecutive pieces of `chunk` tokens, dropping pieces under 2 tokens.
    Returns (offsets-like array, piece ids) for 73.smoothness: piece i is Z[o[2i]:o[2i+1]].

    WHY. The smoothness baseline shuffles tokens WITHIN a response, so in a short response a random pair is
    itself only a few tokens apart and the baseline is high; in a long one it is low. Comparing short QA
    answers with whole RAGTruth responses therefore mixes smoothness with length. Measuring RAGTruth cut to
    the short answers' length puts both on the same baseline."""
    flat = []
    for n in idx:
        a, b = int(offsets[n]), int(offsets[n + 1])
        for s in range(a, b, chunk):
            e = min(s + chunk, b)
            if e - s >= 2:
                flat += [s, e]
    o = np.array(flat, dtype=np.int64)
    return o, np.arange(0, len(o), 2)


def aligned_matrix(s, offsets, idx, onset, window):
    """(responses, 2 * window + 1) token scores around each response's aligned position; NaN where the
    response does not reach."""
    M = np.full((len(idx), 2 * window + 1), np.nan)
    for k, n in enumerate(idx):
        v = s[offsets[n]:offsets[n + 1]]
        o = int(onset[k])
        lo, hi = max(0, o - window), min(len(v), o + window + 1)
        M[k, np.arange(lo, hi) - o + window] = v[lo:hi]
    return M


def curve_band(M, n_boot, seed):
    """95% band of the per-position mean under resampling of responses (rows of M)."""
    import warnings
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        draws = np.array([np.nanmean(M[rng.integers(0, len(M), len(M))], axis=0) for _ in range(n_boot)])
        return np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)


def window_mean(v, o, a, b):
    lo, hi = max(0, o + a), min(len(v), o + b + 1)
    return float(v[lo:hi].mean()) if hi > lo else np.nan


def onset_stats(s, offsets, hall_idx, hall_onset, corr_idx, corr_onset, n_boot, seed):
    """Jump at the onset (hallucinated minus pseudo-onset baseline) and the gap before it, with intervals that
    resample each class's responses independently. Responses with nothing before position 0 are left out of
    the jump and counted."""
    def per_response(idx, onset):
        before = np.array([window_mean(s[offsets[n]:offsets[n + 1]], int(o), *BEFORE) for n, o in zip(idx, onset)])
        after = np.array([window_mean(s[offsets[n]:offsets[n + 1]], int(o), *AFTER) for n, o in zip(idx, onset)])
        return before, after
    hb, ha = per_response(hall_idx, hall_onset)
    cb, ca = per_response(corr_idx, corr_onset)
    hj, cj = ha - hb, ca - cb
    hj, cj, hb_ok, cb_ok = hj[np.isfinite(hj)], cj[np.isfinite(cj)], hb[np.isfinite(hb)], cb[np.isfinite(cb)]
    rng = np.random.default_rng(seed)

    def two_sample(a, b):
        point = float(a.mean() - b.mean())
        vals = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
                for _ in range(n_boot)]
        return {"value": point, "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))}
    return {"jump": two_sample(hj, cj), "gap_before_onset": two_sample(hb_ok, cb_ok),
            "jump_hallucinated_only": float(hj.mean()), "jump_correct_baseline": float(cj.mean()),
            "responses_in_jump": {"hallucinated": int(len(hj)), "correct": int(len(cj))},
            "hallucinated_onset_at_first_token": int((np.asarray(hall_onset) == 0).sum())}


# ---------------------------------------------------------------------------------------------
# first versus later hallucinated tokens
# ---------------------------------------------------------------------------------------------

def run_positions(flags):
    """First token of each run of hallucinated tokens, and the later tokens of runs."""
    f = np.asarray(flags).astype(bool)
    prev = np.concatenate([[False], f[:-1]])
    return f & ~prev, f & prev


def first_vs_later(s, flags, offsets, idx, n_boot, seed):
    """Token AUROC of first-of-run and later-in-run hallucinated tokens against non-hallucinated tokens,
    resampling responses."""
    parts = []
    for n in idx:
        v, f = s[offsets[n]:offsets[n + 1]], flags[offsets[n]:offsets[n + 1]]
        first, later = run_positions(f)
        parts.append((v[first], v[later], v[~f.astype(bool)]))

    def stat_for(which):
        def stat(ix):
            pos = np.concatenate([parts[i][which] for i in ix])
            neg = np.concatenate([parts[i][2] for i in ix])
            if len(pos) == 0 or len(neg) == 0:
                return None
            return auroc(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
        return stat
    return {"first_hallucinated_token": boot_ci(stat_for(0), len(parts), n_boot, seed),
            "later_hallucinated_tokens": boot_ci(stat_for(1), len(parts), n_boot, seed + 1),
            "tokens": {"first": int(sum(len(p[0]) for p in parts)), "later": int(sum(len(p[1]) for p in parts)),
                       "not_hallucinated": int(sum(len(p[2]) for p in parts))}}


# ---------------------------------------------------------------------------------------------

def fmt(ci):
    return "%.3f [%.3f, %.3f]" % (ci["value"], ci["lo"], ci["hi"]) if ci["lo"] is not None else "%.3f" % ci["value"]


def run(generator, out_dir, seed, n_boot):
    s73 = _load("s73", "73_token_structure.py")
    s74 = _load("s74", "74_extract_ragtruth.py")
    c = s74.ragtruth_cfg()
    d = os.path.join(s74.resolve(c["out_dir"]), generator)
    if not os.path.exists(os.path.join(d, "meta.json")):
        raise SystemExit("%s is missing or incomplete -- run 74_extract_ragtruth.py first" % d)
    meta = json.load(open(os.path.join(d, "meta.json")))
    if meta["nonfinite_entries"]:
        raise SystemExit("store has %d non-finite entries" % meta["nonfinite_entries"])
    A = {k: np.load(os.path.join(d, k + ".npy")) for k in
         ("offsets", "token_halluc", "label", "first_halluc", "split", "task", "nll")}
    tokens = np.load(os.path.join(d, "tokens.npy"))
    offsets, flags, y, onset, split, task = (A["offsets"].astype(np.int64), A["token_halluc"], A["label"].astype(int),
                                             A["first_halluc"].astype(np.int64), A["split"], A["task"])
    lengths = np.diff(offsets)
    blocks = meta["blocks"]
    Z = tokens.reshape(len(tokens), -1)
    tr, te = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    print("  [%s] %d responses (%d train, %d test), %d tokens | test: %d hallucinated, %d correct | "
          "response tokens median %d" % (generator, len(y), len(tr), len(te), len(tokens), int(y[te].sum()),
                                        int((y[te] == 0).sum()), int(np.median(lengths))), flush=True)
    out = {"generator": generator, "model_id": meta["model_id"], "blocks": blocks, "store": meta,
           "response_tokens": {"median": float(np.median(lengths)), "mean": float(lengths.mean()),
                               "max": int(lengths.max())},
           "n_boot": n_boot, "seed": seed, "complete": False}
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "ragtruth_%s.json" % generator)
    t0 = time.time()

    # token-to-token smoothness
    allr = np.arange(len(y))
    sm = {str(b): s73.smoothness(np.ascontiguousarray(tokens[:, j, :]), offsets, allr, MAX_LAG, seed)
          for j, b in enumerate(blocks)}
    mid = len(blocks) // 2
    Zm = np.ascontiguousarray(tokens[:, mid, :])
    sm["by_class_at_block_%d" % blocks[mid]] = {
        "correct": s73.smoothness(Zm, offsets, allr[y == 0], MAX_LAG, seed),
        "hallucinated": s73.smoothness(Zm, offsets, allr[y == 1], MAX_LAG, seed)}
    out["token_smoothness"] = sm
    out["token_smoothness_cut_to_short_length"] = {"block": blocks[mid]}
    for chunk in SHORT_LENGTHS:
        o, pieces = chunk_spans(offsets, allr, chunk)
        r = s73.smoothness(Zm, o, pieces, min(8, chunk - 1), seed)
        out["token_smoothness_cut_to_short_length"]["%d_tokens" % chunk] = dict(r, pieces=int(len(pieces)))
    print("    smoothness done (%.0fs)" % (time.time() - t0), flush=True)

    # where in the answer: probe on response labels
    t1 = time.time()
    rows_tr = s73.token_rows(offsets, tr)
    w, yt = s73.token_weights(offsets, tr, y)
    sc, clf = s73.fit_probe(Z[rows_tr], yt, w, seed)
    s_resp = clf.decision_function(sc.transform(Z))
    agg = s73.answer_aggregates(s_resp, offsets, te)
    out["auroc_by_tokens_used"] = {a: response_auroc(agg[a], y[te], n_boot, seed) for a in s73.AGGREGATES}
    out["auroc_mean_by_task"] = {}
    for code, name in TASK_NAMES.items():
        m = task[te] == code
        if m.sum() and len(np.unique(y[te][m])) == 2:
            out["auroc_mean_by_task"][name] = dict(response_auroc(agg["mean"][m], y[te][m], n_boot, seed),
                                                   n=int(m.sum()), hallucinated=int(y[te][m].sum()))
    curves, cnt = s73.position_curves(s_resp, offsets, te, y, N_BINS)
    out["score_gap_by_position"] = {"bins": N_BINS, "correct": curves[0].tolist(), "hallucinated": curves[1].tolist(),
                                    "gap": (curves[1] - curves[0]).tolist(), "answers": cnt.tolist()}
    print("    response-label probe and where-in-the-answer done (%.0fs)" % (time.time() - t1), flush=True)

    # onset
    rng = np.random.default_rng(seed)
    tr_h = tr[(y[tr] == 1) & (onset[tr] >= 0)]
    fractions = onset[tr_h] / lengths[tr_h]
    hall_te = te[(y[te] == 1) & (onset[te] >= 0)]
    corr_te = te[y[te] == 0]
    corr_onset = pseudo_onsets(lengths[corr_te], fractions, rng)
    curve_h, n_h = aligned_curve(s_resp, offsets, hall_te, onset[hall_te], WINDOW)
    curve_c, n_c = aligned_curve(s_resp, offsets, corr_te, corr_onset, WINDOW)
    lo_h, hi_h = curve_band(aligned_matrix(s_resp, offsets, hall_te, onset[hall_te], WINDOW), n_boot, seed)
    lo_c, hi_c = curve_band(aligned_matrix(s_resp, offsets, corr_te, corr_onset, WINDOW), n_boot, seed + 1)
    out["onset"] = {"positions": list(range(-WINDOW, WINDOW + 1)),
                    "hallucinated_curve": curve_h.tolist(), "hallucinated_responses": n_h.tolist(),
                    "hallucinated_band": [lo_h.tolist(), hi_h.tolist()],
                    "correct_pseudo_onset_curve": curve_c.tolist(), "correct_responses": n_c.tolist(),
                    "correct_band": [lo_c.tolist(), hi_c.tolist()],
                    "onset_fraction_of_length": {"median": float(np.median(fractions)),
                                                 "p10": float(np.percentile(fractions, 10)),
                                                 "p90": float(np.percentile(fractions, 90))},
                    "probe": "response labels only (never saw a span)",
                    **onset_stats(s_resp, offsets, hall_te, onset[hall_te], corr_te, corr_onset, n_boot, seed)}

    # first versus later hallucinated tokens: probe on token labels
    t2 = time.time()
    ytok = flags[rows_tr].astype(int)
    wt = np.where(ytok == 1, 0.5 / max(ytok.sum(), 1), 0.5 / max((ytok == 0).sum(), 1))
    wt *= len(wt) / wt.sum()
    sc2, clf2 = s73.fit_probe(Z[rows_tr], ytok, wt, seed)
    s_tok = clf2.decision_function(sc2.transform(Z))
    out["first_vs_later_tokens"] = first_vs_later(s_tok, flags, offsets, te, N_BOOT_TOKENS, seed)
    rows_te = s73.token_rows(offsets, te)
    out["token_label_probe_token_auroc"] = auroc(flags[rows_te].astype(int), s_tok[rows_te])
    print("    token-label probe and first-versus-later done (%.0fs)" % (time.time() - t2), flush=True)

    out["elapsed_seconds"] = round(time.time() - t0, 1)
    out["complete"] = True
    json.dump(out, open(dst, "w"), indent=1)

    o = out["onset"]
    print("\n  SUMMARY %s (RAGTruth test, 95%% intervals)" % generator)
    print("    smoothness, excess similarity at block %d, distance 1 2 4 8 16 32: %s" % (blocks[mid], " ".join(
        "%.3f" % sm[str(blocks[mid])]["excess_similarity"][k - 1] for k in (1, 2, 4, 8, 16, 32))))
    for chunk in SHORT_LENGTHS:
        c = out["token_smoothness_cut_to_short_length"]["%d_tokens" % chunk]
        print("      cut into %2d-token pieces (same baseline as short answers), distance 1..%d: %s"
              % (chunk, len(c["excess_similarity"]), " ".join("%.3f" % v for v in c["excess_similarity"])))
    for a in s73.AGGREGATES:
        print("    tokens used %-12s AUROC %s" % (a, fmt(out["auroc_by_tokens_used"][a])))
    for name, v in out["auroc_mean_by_task"].items():
        print("    mean-of-tokens AUROC within %-8s %s  (%d responses, %d hallucinated)" % (name, fmt(v), v["n"], v["hallucinated"]))
    print("    score gap, hallucinated minus correct, by position (10 bins): %s"
          % " ".join("%.2f" % g for g in out["score_gap_by_position"]["gap"]))
    print("    onset sits at a median %.0f%% of the response; %d of %d hallucinated test responses start at token 0"
          % (100 * o["onset_fraction_of_length"]["median"], o["hallucinated_onset_at_first_token"], len(hall_te)))
    print("    ONSET jump (hallucinated minus baseline): %s   [hallucinated %.3f, baseline %.3f]"
          % (fmt(o["jump"]), o["jump_hallucinated_only"], o["jump_correct_baseline"]))
    print("    gap BEFORE the onset (hallucinated minus correct): %s" % fmt(o["gap_before_onset"]))
    pos = np.array(o["positions"])
    sel = np.isin(pos, [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
    print("    aligned score, position: %s" % " ".join("%+d" % p for p in pos[sel]))
    print("      hallucinated        : %s" % " ".join("%.2f" % v for v in np.array(o["hallucinated_curve"])[sel]))
    print("      correct (pseudo)    : %s" % " ".join("%.2f" % v for v in np.array(o["correct_pseudo_onset_curve"])[sel]))
    f = out["first_vs_later_tokens"]
    print("    token-label probe, token AUROC on test: %.3f" % out["token_label_probe_token_auroc"])
    print("      first hallucinated token of each run: %s  (%d tokens)" % (fmt(f["first_hallucinated_token"]), f["tokens"]["first"]))
    print("      later hallucinated tokens           : %s  (%d tokens)" % (fmt(f["later_hallucinated_tokens"]), f["tokens"]["later"]))
    print("  wrote %s" % dst)


# ---------------------------------------------------------------------------------------------

def self_test():
    """Synthetic token scores with a planted onset event, and with a planted whole-response shift: the onset
    test must tell them apart."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print("    [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
        ok = ok and bool(cond)

    rng = np.random.default_rng(0)
    n = 800
    lengths = rng.integers(40, 160, size=n)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    y = (rng.random(n) < 0.5).astype(int)
    onset = np.where(y == 1, (rng.uniform(0.2, 0.8, size=n) * lengths).astype(np.int64), -1)
    hall, corr = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    fractions = onset[hall] / lengths[hall]

    def scores(kind):
        s = rng.normal(size=offsets[-1])
        for i in hall:
            a = offsets[i] + (onset[i] if kind == "event" else 0)
            s[a:offsets[i + 1]] += 1.5
        return s
    for kind, want_jump, want_before in (("event", 1.5, 0.0), ("shift", 0.0, 1.5)):
        s = scores(kind)
        co = pseudo_onsets(lengths[corr], fractions, np.random.default_rng(1))
        st = onset_stats(s, offsets, hall, onset[hall], corr, co, 300, 2)
        check("%s: jump %.2f expected %.1f" % (kind, st["jump"]["value"], want_jump),
              abs(st["jump"]["value"] - want_jump) < 0.2)
        check("%s: gap before onset %.2f expected %.1f" % (kind, st["gap_before_onset"]["value"], want_before),
              abs(st["gap_before_onset"]["value"] - want_before) < 0.2)
        if kind == "event":
            curve, cnt = aligned_curve(s, offsets, hall, onset[hall], 24)
            check("event: aligned curve steps up at position 0", curve[24:].mean() - curve[:24].mean() > 1.2,
                  "before %.2f, after %.2f" % (curve[:24].mean(), curve[24:].mean()))

    first, later = run_positions([0, 1, 1, 0, 1])
    check("runs of hallucinated tokens", first.tolist() == [False, True, False, False, True]
          and later.tolist() == [False, False, True, False, False])

    # first tokens of runs boosted more than later ones -> higher AUROC for first
    flags = np.zeros(offsets[-1], dtype=np.uint8)
    for i in hall:
        a = offsets[i] + onset[i]
        flags[a:a + 6] = 1
    s = rng.normal(size=offsets[-1])
    fr, la = run_positions(flags)
    s[fr] += 2.0
    s[la] += 0.5
    fl = first_vs_later(s, flags, offsets, np.arange(n), 100, 3)
    check("first-of-run tokens score above later ones", fl["first_hallucinated_token"]["value"]
          > fl["later_hallucinated_tokens"]["value"] + 0.2,
          "first %.3f, later %.3f" % (fl["first_hallucinated_token"]["value"], fl["later_hallucinated_tokens"]["value"]))
    pse = pseudo_onsets(np.array([10, 50]), [0.5], np.random.default_rng(0))
    check("pseudo-onsets land inside the response", pse.tolist() == [5, 25], str(pse.tolist()))

    # cutting long smooth answers into short pieces reproduces what genuinely short answers give, and the
    # whole-length measurement reads higher -- the length effect the cut is there to remove
    s73 = _load("s73", "73_token_structure.py")
    r = 8

    def ar_store(lens, rho):
        off = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=off[1:])
        Z = np.empty((off[-1], r), dtype=np.float32)
        for i in range(len(lens)):
            base, e = rng.normal(size=r) * 2.0, rng.normal(size=r)
            for t in range(lens[i]):
                e = rho * e + np.sqrt(1 - rho ** 2) * rng.normal(size=r)
                Z[off[i] + t] = base + e
        return Z, off
    Zl, offl = ar_store(rng.integers(120, 200, size=300), 0.7)
    Zs, offs = ar_store(np.full(1500, 17), 0.7)
    o, pieces = chunk_spans(offl, np.arange(300), 17)
    cut = s73.smoothness(Zl, o, pieces, 4, 1)["excess_similarity"][0]
    short_ = s73.smoothness(Zs, offs, np.arange(1500), 4, 1)["excess_similarity"][0]
    whole = s73.smoothness(Zl, offl, np.arange(300), 4, 1)["excess_similarity"][0]
    check("long answers cut to 17 tokens match genuinely 17-token answers", abs(cut - short_) < 0.04,
          "cut %.3f, short %.3f" % (cut, short_))
    check("whole-length measurement reads higher (the length effect)", whole > cut + 0.05,
          "whole %.3f, cut %.3f" % (whole, cut))
    check("pieces are 17 tokens or the response's tail", all(2 <= o[2 * i + 1] - o[2 * i] <= 17 for i in range(len(pieces))))

    # the band brackets the aligned mean, and the matrix agrees with aligned_curve where both are defined
    s = scores("event")
    M = aligned_matrix(s, offsets, hall, onset[hall], 24)
    curve, cnt = aligned_curve(s, offsets, hall, onset[hall], 24)
    lo, hi = curve_band(M, 200, 5)
    check("aligned matrix matches the aligned curve", np.allclose(np.nanmean(M, axis=0), curve) and
          (np.isfinite(M).sum(axis=0) == cnt).all())
    check("band brackets the mean and is narrow", bool((lo <= curve).all() and (curve <= hi).all())
          and float(np.max(hi - lo)) < 0.6, "widest %.2f" % float(np.max(hi - lo)))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--generator")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    if a.self_test:
        print("  self-test")
        raise SystemExit(0 if self_test() else 1)
    if not a.generator:
        raise SystemExit("--generator is required (or --self-test)")
    run(a.generator, a.out_dir, a.seed, a.n_boot)


if __name__ == "__main__":
    main()
