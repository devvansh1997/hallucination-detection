"""
84_math_last_part.py -- what is the late signal in GSM8K answers? (T-026)
==================================================================================================

Same data, folds and score as 78-80: Qwen2.5-7B-Instruct's 10 sampled answers per GSM8K question, same-question
AUROC on held-out questions, the 150 basis questions left out. Two plain checks on the end of the answer, where
78 found all of the separation (last 16 tokens alone: 0.69).

  A. DOES ORDER MATTER INSIDE THE LATE PART? 80's GRU, real order versus shuffled order, trained and scored
     on each answer's last 64 tokens only (shorter answers: all their tokens). 80 shuffled whole answers; here
     the shuffle stays inside the part that carries the signal.

  B. IS THE LATE SIGNAL THE FINAL NUMBER? The final number is located in each answer exactly as 76 reads it
     ('#### n', \\boxed{n}, else the last number). Rows:
        token probe (78's pooled probe), mean over the final number's tokens
        token probe, mean over every token BEFORE the final number
        token probe, mean over the last 16 / last 64 tokens (references)
        likelihood: negative log-likelihood of the final number (sum over its tokens)
        likelihood: mean negative log-likelihood of the last 16 tokens, and of the whole answer
     If the final number's tokens carry the separation and the rest does not, "the late signal" is the
     model's state at its answer -- a known effect (truthfulness concentrates on the exact answer tokens),
     not a temporal one. The likelihood rows say whether the model's own probability already shows it.

Per-token likelihoods are computed once on the GPU and kept beside the state store
(<dataset>_token_nll.npy, ~14 MB).

  python 84_math_last_part.py --self-test
  python 84_math_last_part.py --model_folder qwen-2.5-7b-instruct
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "token_structure")
LAST = 64
SHORT = 16
PER_ANSWER = 64            # 78's training tokens per answer


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def final_number_span(text, s76):
    """Character span of the number 76.read_final_number reads, or None."""
    best = None
    for rx in (s76.MARKED, s76.BOXED):
        ms = list(rx.finditer(text))
        if ms and s76.to_number(ms[-1].group(1)) is not None and (best is None or ms[-1].start() > best[0]):
            best = (ms[-1].start(), ms[-1].span(1))
    if best is not None:
        return best[1]
    nums = [m for m in s76.ANY_NUMBER.finditer(text) if s76.to_number(m.group(0)) is not None]
    return nums[-1].span() if nums else None


def token_char_ends(tok, ids):
    """End character of each token in tok.decode(ids, skip_special_tokens=True). Token-by-token decoding when it
    reproduces the full decode (plain ASCII text); otherwise prefix decoding."""
    full = tok.decode(ids, skip_special_tokens=True)
    pieces = [tok.decode([t], skip_special_tokens=True) for t in ids]
    if "".join(pieces) == full:
        return np.cumsum([len(p) for p in pieces]), full
    return np.array([len(tok.decode(ids[:t + 1], skip_special_tokens=True)) for t in range(len(ids))]), full


def number_tokens(ends, span, lead):
    """Positions of the tokens overlapping the character span [a, b) of the stripped text (lead = characters
    stripped from the front)."""
    if span is None:
        return np.zeros(0, dtype=np.int64)
    a, b = span[0] + lead, span[1] + lead
    starts = np.r_[0, ends[:-1]]
    return np.flatnonzero((ends > a) & (starts < b))


def window_rows(offsets, idx, w):
    """Token rows of the last w tokens of every answer in idx (all tokens of shorter ones), and offsets."""
    rows, n = [], []
    for a in idx:
        lo = max(int(offsets[a]), int(offsets[a + 1]) - w)
        rows.append(np.arange(lo, int(offsets[a + 1])))
        n.append(len(rows[-1]))
    o = np.zeros(len(n) + 1, dtype=np.int64)
    np.cumsum(n, out=o[1:])
    return np.concatenate(rows), o


def part_mean(v, offsets, idx, pick):
    """Per answer in idx: mean of v over the positions pick(n, T) returns (NaN if empty)."""
    out = np.full(len(idx), np.nan)
    for i, n in enumerate(idx):
        seg = v[offsets[n]:offsets[n + 1]]
        p = pick(int(n), len(seg))
        if len(p):
            out[i] = float(np.mean(seg[p]))
    return out


def compute_token_nll(model_id, input_ids, prompt_len, offsets, device):
    import torch
    from transformers import AutoModelForCausalLM
    s74 = _load("s74", "74_extract_ragtruth.py")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model.eval()
    out = np.empty(int(offsets[-1]), dtype=np.float32)
    t0 = time.time()
    for n in range(len(prompt_len)):
        out[offsets[n]:offsets[n + 1]] = s74.token_nll(model, [int(t) for t in input_ids[n]], int(prompt_len[n]), device)
        if (n + 1) % 1000 == 0 or n + 1 == len(prompt_len):
            el = time.time() - t0
            print("    likelihood pass %d/%d (%.0fs, ~%.0fs left)" % (n + 1, len(prompt_len), el,
                                                                   el / (n + 1) * (len(prompt_len) - n - 1)), flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def run(model_folder, dataset, out_dir, seed, n_boot):
    import torch
    import yaml
    from transformers import AutoTokenizer
    s73 = _load("s73", "73_token_structure.py")
    s76 = _load("s76", "76_generate_math.py")
    s78 = _load("s78", "78_math_same_question.py")
    s80 = _load("s80", "80_gru_real_vs_shuffled.py")
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)
    root = os.path.normpath(os.path.join(HERE, cfg["math"]["out_dir"], model_folder))
    d = os.path.join(root, "%s_states" % dataset)
    if not os.path.exists(os.path.join(d, "meta.json")):
        raise SystemExit("%s is missing or incomplete -- run 77_extract_math_states.py first" % d)
    seq = torch.load(os.path.join(root, "%s_sequences_v1.pt" % dataset), weights_only=False)
    y = np.asarray(seq["all_hallucination_flag"]).astype(int)
    pid = np.asarray(seq["prompt_id"], dtype=np.int64)
    offsets = np.load(os.path.join(d, "offsets.npy")).astype(np.int64)
    answers = [np.asarray(ids)[int(pl):] for ids, pl in zip(seq["input_ids"], seq["prompt_len"])]
    if not np.array_equal(np.diff(offsets), np.array([len(a) for a in answers])):
        raise SystemExit("the state store and the sequences file disagree on answer lengths")
    keep = np.flatnonzero(~np.load(os.path.join(d, "basis_question.npy")))
    fold, _ = s78.question_folds(pid[keep], y[keep], s78.N_FOLDS, seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.enabled = False        # cluster cuDNN clash, see 80
    t_all = time.time()

    # per-token likelihood, once
    nll_path = os.path.join(root, "%s_token_nll.npy" % dataset)
    if os.path.exists(nll_path):
        tnll = np.load(nll_path)
        if len(tnll) != offsets[-1]:
            raise SystemExit("%s does not match the store's token count" % nll_path)
    else:
        tnll = compute_token_nll(model_id, seq["input_ids"], seq["prompt_len"], offsets, device)
        np.save(nll_path, tnll)
    ref = np.load(os.path.join(d, "nll.npy"))
    check = np.array([tnll[offsets[n]:offsets[n + 1]].mean() for n in range(len(ref))])
    print("  likelihood check: largest difference from 77's mean answer NLL %.2e" % float(np.abs(check - ref).max()), flush=True)

    # where the final number is
    tok = AutoTokenizer.from_pretrained(model_id)
    num_pos, mismatch, misread = [], 0, 0
    for n, ids in enumerate(answers):
        ends, full = token_char_ends(tok, [int(t) for t in ids])
        text = full.strip()
        mismatch += int(text != seq["decoded_text"][n])
        span = final_number_span(text, s76)
        if span is not None and not s76.same_number(s76.to_number(text[span[0]:span[1]]), seq["pred"][n]):
            misread += 1
        num_pos.append(number_tokens(ends, span, len(full) - len(full.lstrip())))
    found = np.array([len(p) > 0 for p in num_pos])
    print("  final number located in %d of %d answers | decode mismatches %d | located number differs from 76's "
          "reading %d" % (int(found.sum()), len(answers), mismatch, misread), flush=True)

    # token probe scores (78's pooled probe, held out by fold)
    t0 = time.time()
    tokens = np.load(os.path.join(d, "tokens.npy"))
    scores, _ = s78.token_scores(tokens, offsets, y, pid, keep, fold, seed, PER_ANSWER, s73, lambda m: print(m, flush=True))
    probe = scores["pooled_probe"]
    print("  token probe scored (%.0fs)" % (time.time() - t0), flush=True)

    def before(n, T):
        return np.arange(num_pos[n][0]) if len(num_pos[n]) else np.zeros(0, dtype=np.int64)
    rows = {
        "probe_final_number": part_mean(probe, offsets, keep, lambda n, T: num_pos[n]),
        "probe_before_final_number": part_mean(probe, offsets, keep, before),
        "probe_last_16": part_mean(probe, offsets, keep, lambda n, T: np.arange(max(0, T - SHORT), T)),
        "probe_last_64": part_mean(probe, offsets, keep, lambda n, T: np.arange(max(0, T - LAST), T)),
        "nll_final_number": np.array([float(tnll[offsets[n]:offsets[n + 1]][num_pos[n]].sum()) if len(num_pos[n]) else np.nan
                                      for n in keep]),
        "nll_last_16": part_mean(tnll, offsets, keep, lambda n, T: np.arange(max(0, T - SHORT), T)),
        "nll_whole_answer": part_mean(tnll, offsets, keep, lambda n, T: np.arange(T)),
    }
    names = list(rows)
    st = s78.same_question_stats(np.stack([rows[k] for k in names], axis=1), y[keep], pid[keep])
    sb = s78.summarize(st, n_boot, seed)
    part_b = {k: {"auroc": sb["same_question_auroc"][j], "interval": [sb["auroc_interval"][0][j], sb["auroc_interval"][1][j]],
                  "pairs": sb["pairs"][j]} for j, k in enumerate(names)}
    final_tokens = np.array([len(num_pos[n]) for n in keep])
    del tokens

    # A: GRU on the last 64 tokens, real versus shuffled
    t0 = time.time()
    X = np.load(os.path.join(d, "tokens.npy"), mmap_mode="r")
    X = X.reshape(len(X), -1)
    allidx = np.arange(len(y))
    wrows, woff = window_rows(offsets, allidx, LAST)
    Xw = np.asarray(X[wrows])
    print("  last-%d-token store: %d tokens (%.1f GB) (%.0fs)" % (LAST, len(wrows), Xw.nbytes / 1e9, time.time() - t0), flush=True)
    gru = s80.evaluate(Xw, woff, y, pid, keep, fold, seed, device, lambda m: print(m, flush=True))
    part_a = {}
    for c, S in gru.items():
        s = s78.summarize(s78.same_question_stats(S, y[keep], pid[keep]), n_boot, seed)
        part_a[c] = {"auroc": s["same_question_auroc"], "interval": s["auroc_interval"], "pairs": s["pairs"],
                     "questions": s["questions"], "columns": ["first_half_of_window", "second_half_of_window", "whole_window"]}

    out = {"model_folder": model_folder, "dataset": dataset, "seed": seed, "n_boot": n_boot, "window": LAST,
           "last_part_gru": part_a, "final_number": part_b,
           "final_number_tokens": {"median": float(np.median(final_tokens[final_tokens > 0])),
                                   "answers_without_a_number": int((final_tokens == 0).sum())},
           "checks": {"decode_mismatches": mismatch, "located_number_differs_from_76": misread},
           "elapsed_seconds": round(time.time() - t_all, 1)}
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "math_last_part_%s_%s.json" % (model_folder, dataset))
    json.dump(s78.clean(out), open(dst, "w"), indent=1)

    r = part_a["real_order"]
    print("\n  SUMMARY %s / %s: same-question AUROC [95%%], %d questions, %d pairs" % (model_folder, dataset,
                                                                                    r["questions"][2], r["pairs"][2]))
    print("  A. GRU on the last %d tokens          whole window" % LAST)
    for c, label in (("real_order", "real order"), ("shuffled_order", "shuffled order")):
        v = part_a[c]
        print("    %-34s %.3f [%.3f, %.3f]" % (label, v["auroc"][2], v["interval"][0][2], v["interval"][1][2]))
    v = part_b["probe_last_64"]
    print("    %-34s %.3f [%.3f, %.3f]" % ("token probe, last 64 (reference)", v["auroc"], v["interval"][0], v["interval"][1]))
    print("  B. Where the late signal sits (final number: median %d tokens)" % out["final_number_tokens"]["median"])
    for k, label in (("probe_final_number", "token probe, final number"),
                     ("probe_before_final_number", "token probe, everything before it"),
                     ("probe_last_16", "token probe, last 16 tokens"),
                     ("nll_final_number", "likelihood of the final number"),
                     ("nll_last_16", "likelihood, last 16 tokens"),
                     ("nll_whole_answer", "likelihood, whole answer")):
        v = part_b[k]
        print("    %-34s %.3f [%.3f, %.3f]" % (label, v["auroc"], v["interval"][0], v["interval"][1]))
    print("  wrote %s" % dst)


# ---------------------------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print("    [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
        ok = ok and bool(cond)

    s76 = _load("s76", "76_generate_math.py")
    cases = [("So 3 + 4 = 7.\n#### 7", "7"), ("The total is \\boxed{1,250} dollars, for 4 hours.", "1,250"),
             ("#### 5\nActually \\boxed{6}", "6"), ("\\boxed{6} then\n#### 8", "8"), ("we get 12 and then 15.", "15"),
             ("no number here", None)]
    for text, want in cases:
        sp = final_number_span(text, s76)
        got = None if sp is None else text[sp[0]:sp[1]]
        same = (got is None and want is None) or (got is not None and s76.same_number(s76.to_number(got), s76.read_final_number(text)[0]))
        check("final number span: %r" % text[:24], got == want and same, "got %r" % got)

    class FakeTok:
        """One character per token, two leading spaces."""
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids)
    ids = [ord(c) for c in "  x = 42\n#### 42"]
    ends, full = token_char_ends(FakeTok(), ids)
    text = full.strip()
    p = number_tokens(ends, final_number_span(text, s76), len(full) - len(full.lstrip()))
    check("number tokens map back through the stripped text", "".join(chr(ids[i]) for i in p) == "42" and p.tolist() == [14, 15],
          str(p.tolist()))
    offs = np.array([0, 5, 105, 108])
    r, o = window_rows(offs, np.arange(3), 64)
    check("last-64 window", o.tolist() == [0, 5, 69, 72] and r[5] == 41 and r[68] == 104 and r[69] == 105, str(o.tolist()))
    v = np.arange(108, dtype=float)
    m = part_mean(v, offs, np.arange(3), lambda n, T: np.arange(max(0, T - 2), T))
    check("part means", m.tolist() == [3.5, 103.5, 106.5], str(m.tolist()))
    m = part_mean(v, offs, np.arange(3), lambda n, T: np.zeros(0, dtype=int) if n == 1 else np.arange(1))
    check("empty part is missing", np.isnan(m[1]) and m[0] == 0 and m[2] == 105)
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
