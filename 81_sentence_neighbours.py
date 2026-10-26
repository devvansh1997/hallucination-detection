"""
81_sentence_neighbours.py -- do hallucinated sentences sit next to each other more than chance? (T-026)
==================================================================================================

THE CHECK. Labels only: no model, no hidden states. RAGTruth responses from all six generators are cut into
sentences, and a sentence counts as hallucinated if a human-labeled span STARTS in it. Then one contrast:

    real order       after a hallucinated sentence, how often is the next sentence hallucinated too?
    shuffled order   the same, with each response's sentences in random order (exact expectation)

Shuffling within a response keeps each response's own hallucination rate, so "this response is about an
obscure topic, everything in it is wrong" cannot make real order look clustered. If hallucinations snowball
(one wrong sentence makes the next one more likely to be wrong), real order beats shuffled; then a sequence
model over sentences has something to use. If they tie, the labels carry no order.

Also reported: the hallucination rate in the first, middle and last third of each response's sentences.

WHY "STARTS IN". A span that runs across a sentence boundary would mark two neighbouring sentences and make
real order look clustered by construction. Counting a span only where it starts removes that. The version
that counts every sentence a span touches is reported second, for comparison.

Reads RAGTruth's response.jsonl (config.yaml ragtruth.raw_dir). CPU, minutes.

  python 81_sentence_neighbours.py --self-test
  python 81_sentence_neighbours.py
"""

import argparse
import importlib.util
import json
import os
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "token_structure")
N_BOOT = 2000
BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sentence_starts(text):
    """Character position where each sentence starts. A sentence ends at . ! or ? followed by whitespace, or at
    a line break. Sentence k covers [starts[k], starts[k + 1]), so the whitespace between sentences belongs to
    the sentence before it."""
    starts, pos = [], 0
    for m in BOUNDARY.finditer(text):
        if text[pos:m.start()].strip():
            starts.append(pos)
        pos = m.end()
    if text[pos:].strip():
        starts.append(pos)
    return starts


def sentence_labels(text, spans):
    """(starts_in, touches): per sentence, 1 if a span starts in it / 1 if a span overlaps it."""
    st = sentence_starts(text)
    n = len(st)
    bounds = st + [len(text)]
    starts_in = np.zeros(n, dtype=np.int64)
    touches = np.zeros(n, dtype=np.int64)
    if n == 0:
        return starts_in, touches
    for sp in spans:
        a, b = int(sp["start"]), int(sp["end"])
        k = int(np.searchsorted(st, a, side="right")) - 1
        starts_in[max(k, 0)] = 1
        for j in range(n):
            if bounds[j + 1] > a and bounds[j] < b:
                touches[j] = 1
    return starts_in, touches


def neighbour_counts(lab):
    """For one response's 0/1 sentence labels: hallucinated sentences that have a next sentence, how many of
    those are followed by a hallucinated one, and the same two numbers expected with the sentences shuffled."""
    n, k = len(lab), int(lab.sum())
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    with_next = float(lab[:-1].sum())
    followed = float((lab[:-1] & lab[1:]).sum())
    exp_with_next = k * (n - 1) / n                 # each sentence is last with chance 1/n
    exp_followed = k * (k - 1) / n                  # (n - 1) neighbour pairs, each both-hallucinated w.p. k(k-1)/(n(n-1))
    return with_next, followed, exp_with_next, exp_followed


def thirds(lab):
    """Hallucinated and total sentences in the first, middle and last third of a response (by sentence
    index); empty for responses under 3 sentences."""
    n = len(lab)
    if n < 3:
        return np.zeros(3), np.zeros(3)
    b = np.minimum((np.arange(n) * 3) // n, 2)
    return np.bincount(b, weights=lab, minlength=3), np.bincount(b, minlength=3).astype(float)


def neighbour_stats(C, n_boot, seed):
    """C: (responses, 4) rows of neighbour_counts. Share of hallucinated sentences followed by another, real
    and shuffled, their ratio, with 95% intervals from resampling responses."""
    def stat(rows):
        s = rows.sum(0)
        real, shuf = s[1] / max(s[0], 1e-12), s[3] / max(s[2], 1e-12)
        return real, shuf, real / max(shuf, 1e-12)
    point = stat(C)
    rng = np.random.default_rng(seed)
    draws = np.array([stat(C[rng.integers(0, len(C), len(C))]) for _ in range(n_boot)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return {"real_order": point[0], "shuffled_order": point[1], "ratio": point[2],
            "interval": {"real_order": [lo[0], hi[0]], "shuffled_order": [lo[1], hi[1]], "ratio": [lo[2], hi[2]]},
            "responses": int(len(C)), "hallucinated_sentences_with_a_next": float(C[:, 0].sum())}


def third_stats(H, N, n_boot, seed):
    """H, N: (responses, 3) hallucinated and total sentences per third. Rate per third and last minus first."""
    def stat(h, n):
        r = h.sum(0) / np.maximum(n.sum(0), 1)
        return np.r_[r, r[2] - r[0]]
    point = stat(H, N)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(H), len(H))
        draws.append(stat(H[ix], N[ix]))
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
    return {"first_third": point[0], "middle_third": point[1], "last_third": point[2],
            "last_minus_first": point[3], "last_minus_first_interval": [lo[3], hi[3]], "responses": int(len(H))}


def analyze(recs, n_boot, seed):
    """recs: dicts with response, labels, model, task. Returns the result dict."""
    rows = []
    for r in recs:
        si, to = sentence_labels(r["response"], r["labels"])
        rows.append({"model": r["model"], "task": r["task"], "starts_in": si, "touches": to})
    out = {"responses": len(rows),
           "sentences": {"median": float(np.median([len(x["starts_in"]) for x in rows])),
                         "mean": float(np.mean([len(x["starts_in"]) for x in rows]))},
           "hallucinated_sentence_share": float(np.mean(np.concatenate([x["starts_in"] for x in rows])))}

    def block(sel):
        res = {}
        for key in ("starts_in", "touches"):
            # every response, including those with no hallucinated sentence (they add zeros): selecting rows on
            # the real-order counts would drop responses whose only hallucinated sentence is the last one and
            # bias the ratio downward
            C = np.array([neighbour_counts(x[key]) for x in sel])
            T = [thirds(x[key]) for x in sel]
            H, N = np.array([t[0] for t in T]), np.array([t[1] for t in T])
            keep = N.sum(1) > 0
            res[key] = {"neighbours": neighbour_stats(C, n_boot, seed),
                        "by_third": third_stats(H[keep], N[keep], n_boot, seed) if keep.any() else None}
        return res

    out["all"] = block(rows)
    out["by_generator"] = {m: block([x for x in rows if x["model"] == m]) for m in sorted({x["model"] for x in rows})}
    out["by_task"] = {t: block([x for x in rows if x["task"] == t]) for t in sorted({x["task"] for x in rows})}
    return out


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def load_all(raw_dir, keep_quality=("good",)):
    s74 = _load("s74", "74_extract_ragtruth.py")
    src = {str(s["source_id"]): s for s in s74.read_jsonl(os.path.join(raw_dir, "source_info.jsonl"))}
    recs = []
    for r in s74.read_jsonl(os.path.join(raw_dir, "response.jsonl")):
        if r.get("quality", "good") not in keep_quality or str(r["source_id"]) not in src:
            continue
        recs.append({"response": r["response"], "labels": r.get("labels") or [], "model": r["model"],
                     "task": src[str(r["source_id"])]["task_type"]})
    return recs


def run(out_dir, seed, n_boot):
    s74 = _load("s74", "74_extract_ragtruth.py")
    raw_dir = s74.resolve(s74.ragtruth_cfg()["raw_dir"])
    t0 = time.time()
    recs = load_all(raw_dir)
    print("  %d RAGTruth responses (quality good), %d generators" % (len(recs), len({r["model"] for r in recs})), flush=True)
    out = clean(analyze(recs, n_boot, seed))
    out.update({"seed": seed, "n_boot": n_boot, "elapsed_seconds": round(time.time() - t0, 1)})
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "sentence_neighbours_ragtruth.json")
    json.dump(out, open(dst, "w"), indent=1)

    def line(name, v):
        nb = v["neighbours"]
        th = v["by_third"]
        print("    %-26s %5.1f%%  %5.1f%%  %5.2f [%.2f, %.2f]   %5.1f%% %5.1f%% %5.1f%%"
              % (name, 100 * nb["real_order"], 100 * nb["shuffled_order"], nb["ratio"], nb["interval"]["ratio"][0],
                 nb["interval"]["ratio"][1], 100 * th["first_third"], 100 * th["middle_third"], 100 * th["last_third"]))

    print("\n  SUMMARY: after a hallucinated sentence, share of next sentences that are hallucinated")
    print("  (%d responses, median %d sentences, %.1f%% of sentences hallucinated)"
          % (out["responses"], out["sentences"]["median"], 100 * out["hallucinated_sentence_share"]))
    print("    %-26s %6s  %6s  %-18s   %s" % ("", "real", "shuf.", "ratio [95%]", "rate in 1st / 2nd / 3rd third"))
    line("all (span starts in it)", out["all"]["starts_in"])
    line("all (span touches it)", out["all"]["touches"])
    for m, v in out["by_generator"].items():
        line(m, v["starts_in"])
    for t, v in out["by_task"].items():
        line("task " + t, v["starts_in"])
    print("  ratio 1 = no clustering. Rows after the first two count a span where it starts.")
    print("  wrote %s" % dst)


# ---------------------------------------------------------------------------------------------

def self_test():
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print("    [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
        ok = ok and bool(cond)

    t = "The cafe opened in 2021. It serves tea!\nPrices are low? Yes. 3.5 stars."
    st = sentence_starts(t)
    check("sentence starts", [t[s:s + 3] for s in st] == ["The", "It ", "Pri", "Yes", "3.5"], str([t[s:s + 3] for s in st]))
    si, to = sentence_labels(t, [{"start": 19, "end": 30}])          # "2021. It se" crosses a boundary
    check("a span counts where it starts", si.tolist() == [1, 0, 0, 0, 0], str(si.tolist()))
    check("touching marks both sentences", to.tolist() == [1, 1, 0, 0, 0], str(to.tolist()))
    si, _ = sentence_labels(t, [{"start": 24, "end": 30}])          # starts in the space after "2021."
    check("a span starting between sentences belongs to the one before", si.tolist() == [1, 0, 0, 0, 0], str(si.tolist()))
    w, f, ew, ef = neighbour_counts(np.array([1, 1, 0, 0]))
    check("neighbour counts", (w, f) == (2.0, 1.0) and abs(ew - 1.5) < 1e-12 and abs(ef - 0.5) < 1e-12,
          "%s %s %s %s" % (w, f, ew, ef))
    # exact shuffled expectation equals the average over all permutations
    import itertools
    lab = np.array([1, 0, 1, 1, 0])
    perms = [neighbour_counts(lab[list(p)]) for p in itertools.permutations(range(5))]
    avg = np.mean(perms, axis=0)
    e = neighbour_counts(lab)
    check("shuffled expectation is exact", abs(avg[0] - e[2]) < 1e-9 and abs(avg[1] - e[3]) < 1e-9,
          "perm avg %.4f %.4f, formula %.4f %.4f" % (avg[0], avg[1], e[2], e[3]))

    def world(clustered):
        recs = []
        for i in range(600):
            n = int(rng.integers(4, 12))
            if clustered:
                lab = np.zeros(n, dtype=int)
                a = int(rng.integers(0, n - 1))
                lab[a:a + int(rng.integers(1, 4))] = 1
            else:
                lab = (rng.random(n) < 0.25).astype(int)
            text, spans = "", []
            for j in range(n):
                s = len(text)
                text += "Sentence number %d is here. " % j
                if lab[j]:
                    spans.append({"start": s + 9, "end": s + 15})
            recs.append({"response": text, "labels": spans, "model": "m", "task": "QA"})
        return analyze(recs, 200, 0)["all"]["starts_in"]["neighbours"]
    a, b = world(True), world(False)
    check("clustered labels: ratio above 1", a["ratio"] > 1.5 and a["interval"]["ratio"][0] > 1.2, "%.2f" % a["ratio"])
    check("independent labels: ratio near 1", abs(b["ratio"] - 1) < 0.15, "%.2f" % b["ratio"])
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    if a.self_test:
        print("  self-test")
        raise SystemExit(0 if self_test() else 1)
    run(a.out_dir, a.seed, a.n_boot)


if __name__ == "__main__":
    main()
