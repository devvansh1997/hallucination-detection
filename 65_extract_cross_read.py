"""
65_extract_cross_read.py -- one model reads another model's answers (cross-model transfer, T-013).
==================================================================================================

WHAT IT IS FOR. The detector is fitted per model: its scaling and Tucker bases live in that model's
hidden space (3584-d Qwen, 4096-d LLaMA), and the random forest sits on those coordinates. To apply
a detector trained on a SOURCE model to a TARGET model, the target's core coordinates have to be
mapped into the source's. This script produces the paired data that map is fitted on, without any
target labels: the target's own answers, read by the source model.

    generator = the model that wrote the answers (the target of the transfer)
    reader    = the model whose hidden states are recorded (the source)

For every generator answer, the reader sees exactly the prompt it saw when it generated its own
answers to that question, followed by the generator's answer text, and the window features are
pooled over the answer tokens exactly as the pinned pipeline pools the reader's own answers.

PROMPTS. No chat template anywhere in this pipeline (39_generate_dataset.py): both models were
prompted with the same raw template string, so the reader's prompt for question q is taken verbatim
from the reader's own sequences file. Both files index questions by the same prompt_id; that is
checked, not assumed -- the decoded prompt text must match for every question or the job stops.

ANSWER TEXT. The generator's completion ids (canonical window: content through the first stop token)
are decoded without stripping and re-encoded with the reader's tokenizer. If the generator's
completion ended in its literal EOS, the reader's EOS is appended, so a stop is a stop on both sides.

FEATURES. Hidden-state indices 16..24 (blocks 15..23, the reported window), pooled by
57_extract_all_layers.pool_answer: peak (9, D), range q95/q05 (9, D), update q95/q05 (8, D) -- the
same arrays as the pinned static_max / static_q95 / static_q05 / velocity_q95 / velocity_q05.

PREFLIGHT (every job, before the real work). The reader first reads ITS OWN answers for the first
--check-questions questions through this exact route (decode, re-encode, forward, pool) and the
result is compared with its pinned features. If the correlation is below 0.999 in any stream, the
route is wrong and the job stops.

  python 65_extract_cross_read.py --self-test
  python 65_extract_cross_read.py --reader qwen-2.5-7b-instruct --generator llama-3.1-8b --dataset truthfulqa
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, "..", "data-crossread"))
WINDOW = list(range(16, 25))       # hidden-state indices of blocks 15..23
CHECK_CORR = 0.999


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s57 = _load("s57", "57_extract_all_layers.py")


# ---------------------------------------------------------------------------------------------
# Pure helpers (self-tested without a model)
# ---------------------------------------------------------------------------------------------

def prompt_ids_by_question(seq):
    """{prompt_id: prompt token ids} from a sequences file. Every beam of a question must carry the
    same prompt; a file where they differ is not a file this script understands."""
    out = {}
    for ids, pl, q in zip(seq["input_ids"], seq["prompt_len"], seq["prompt_id"]):
        p = [int(t) for t in (ids.tolist() if hasattr(ids, "tolist") else list(ids))[:int(pl)]]
        q = int(q)
        if q in out and out[q] != p:
            raise SystemExit("question %d has beams with different prompts" % q)
        out[q] = p
    return out


def _prompt_text(tok, ids):
    """Decoded prompt, compared after Unicode NFC normalisation. Qwen2's tokenizer applies NFC when it
    encodes and LLaMA-3's does not, so a passage with decomposed accents (TyDiQA-GP has some) decodes to
    different code points from the two models although it is the same question. NFC is the only
    normalisation applied; anything else that differs is still a mismatch."""
    import unicodedata
    return unicodedata.normalize("NFC", tok.decode(ids, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)).strip()


def check_same_questions(reader_prompts, gen_prompts, reader_tok, gen_tok):
    """The two models must have been asked the same question under the same prompt_id. Returns the
    list of mismatching ids (empty when all agree)."""
    if set(reader_prompts) != set(gen_prompts):
        return sorted(set(reader_prompts) ^ set(gen_prompts))
    bad = []
    for q in sorted(reader_prompts):
        if _prompt_text(reader_tok, reader_prompts[q]) != _prompt_text(gen_tok, gen_prompts[q]):
            bad.append(q)
    return bad


def describe_mismatch(a, b, width=60):
    """Where two prompt texts first differ, as repr'd snippets -- so a failure names its cause."""
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    lo = max(0, i - width)
    return "first difference at character %d of %d/%d:\n    reader    %r\n    generator %r" % (
        i, len(a), len(b), a[lo:i + width], b[lo:i + width])


def reader_answer_ids(comp_ids, gen_tok, reader_tok):
    """Generator completion ids -> reader token ids for the same text. Returns (ids, eos_appended)."""
    comp_ids = [int(t) for t in comp_ids]
    ended_eos = len(comp_ids) > 0 and comp_ids[-1] == gen_tok.eos_token_id
    text = gen_tok.decode(comp_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ids = list(reader_tok.encode(text, add_special_tokens=False)) if text else []
    if ended_eos:
        ids.append(int(reader_tok.eos_token_id))
    return ids, ended_eos


def pool_window(hidden_states_j, s, e):
    """hidden_states_j: indexable by hidden-state index (list or dict) of (T_total, D) arrays for ONE row.
    Returns the pinned pipeline's five arrays for answer tokens s:e at indices 16..24."""
    h = np.stack([np.asarray(hidden_states_j[i][s:e], dtype=np.float32) for i in WINDOW])
    p = s57.pool_answer(h, ("core", "static", "velocity"))
    return {"static_max": p["core"], "static_q95": p["q95"], "static_q05": p["q05"],
            "velocity_q95": p["v95"], "velocity_q05": p["v05"]}


def agreement(a, b):
    x, z = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(z)
    x, z = x[ok], z[ok]
    corr = float(np.corrcoef(x, z)[0, 1]) if x.size > 1 and x.std() > 0 and z.std() > 0 else None
    return {"corr": corr, "max_abs_diff": float(np.max(np.abs(x - z))) if x.size else None}


ARRAYS = ("static_max", "static_q95", "static_q05", "velocity_q95", "velocity_q05")


# ---------------------------------------------------------------------------------------------
# GPU work
# ---------------------------------------------------------------------------------------------

def read_answers(model, reader_tok, reader_prompts, jobs, device, D, log_every=50, label=""):
    """jobs: list of (question, [(row, reader_answer_ids), ...]). Returns {array: (n_rows, L, D)}
    indexed by position in the flattened job order, plus the row order."""
    import torch
    rows = [r for _, items in jobs for r, _ in items]
    pos = {r: i for i, r in enumerate(rows)}
    store = {a: np.full((len(rows), 8 if a.startswith("velocity") else 9, D), np.nan, dtype=np.float16)
             for a in ARRAYS}
    pad = reader_tok.eos_token_id if reader_tok.pad_token_id is None else reader_tok.pad_token_id
    t0 = time.time()
    for gi, (q, items) in enumerate(jobs):
        prompt = reader_prompts[q]
        seqs = [prompt + ids for _, ids in items]
        L = max(len(s) for s in seqs)
        ids_t = torch.full((len(seqs), L), int(pad), dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids_t[j, :len(s)] = torch.tensor(s, dtype=torch.long)
            att[j, :len(s)] = 1
        with torch.no_grad():
            out = model(ids_t.to(device), attention_mask=att.to(device), use_cache=False,
                        output_hidden_states=True)
        for j, (r, ids) in enumerate(items):
            if not ids:
                continue                      # stays NaN: nothing to pool
            s, e = len(prompt), len(prompt) + len(ids)
            # Slice the answer tokens on the GPU; only nine (T, D) arrays cross to the host.
            hs = {i: out.hidden_states[i][j, s:e].float().cpu().numpy() for i in WINDOW}
            p = pool_window(hs, 0, e - s)
            for a in ARRAYS:
                store[a][pos[r]] = p[a]
        del out
        if log_every and (gi + 1) % log_every == 0:
            el = time.time() - t0
            print("    %s %d/%d questions (%.0fs, eta %.0fs)"
                  % (label, gi + 1, len(jobs), el, el / (gi + 1) * (len(jobs) - gi - 1)), flush=True)
    return store, rows


def run(reader, generator, dataset, data_dir, out_dir, device, check_questions, limit):
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if reader == generator:
        raise SystemExit("reader and generator are the same model; the preflight already covers that")
    dst_dir = os.path.join(out_dir, "%s_reads_%s" % (reader, generator))
    dst = os.path.join(dst_dir, "%s_window.npz" % dataset)
    if os.path.exists(dst):
        raise SystemExit("refusing: %s exists" % dst)
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    ids_of = {m["folder"]: m["id"] for m in cfg["models"]}

    def seqs(folder):
        p = os.path.join(data_dir, folder, "%s_sequences_v1.pt" % dataset)
        if not os.path.exists(p):
            raise SystemExit("%s not found" % p)
        return torch.load(p, weights_only=False)

    rseq, gseq = seqs(reader), seqs(generator)
    rtok = AutoTokenizer.from_pretrained(ids_of[reader])
    gtok = AutoTokenizer.from_pretrained(ids_of[generator])
    rprompts, gprompts = prompt_ids_by_question(rseq), prompt_ids_by_question(gseq)
    bad = check_same_questions(rprompts, gprompts, rtok, gtok)
    if bad:
        q = bad[0]
        raise SystemExit("%d questions differ between the two sequences files (first: %s) -- the "
                         "prompt_ids do not refer to the same questions. Question %d, %s"
                         % (len(bad), bad[:5], q, describe_mismatch(_prompt_text(rtok, rprompts[q]),
                                                                    _prompt_text(gtok, gprompts[q]))))
    print("  [%s reads %s / %s] %d questions, prompts identical in both files"
          % (reader, generator, dataset, len(rprompts)), flush=True)

    model = AutoModelForCausalLM.from_pretrained(ids_of[reader], dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(device)
    model.eval()
    D = model.config.hidden_size
    assert model.config.num_hidden_layers + 1 > max(WINDOW)

    def group(seq, tok_from, n_questions):
        by_q = {}
        stats = {"eos_appended": 0, "empty": 0}
        if tok_from is rtok:
            stats["roundtrip_same_ids"] = 0       # only meaningful when a model re-reads itself
        for r, (ids, pl, q) in enumerate(zip(seq["input_ids"], seq["prompt_len"], seq["prompt_id"])):
            q = int(q)
            comp = [int(t) for t in (ids.tolist() if hasattr(ids, "tolist") else list(ids))[int(pl):]]
            rid, eos = reader_answer_ids(comp, tok_from, rtok)
            stats["eos_appended"] += int(eos)
            stats["empty"] += int(len(rid) == 0)
            if tok_from is rtok:
                stats["roundtrip_same_ids"] += int(rid == comp)
            by_q.setdefault(q, []).append((r, rid))
        qs = sorted(by_q)[:n_questions] if n_questions else sorted(by_q)
        return [(q, by_q[q]) for q in qs], stats

    # Preflight: the reader reads its own answers through this route and must reproduce its pinned
    # features.
    t0 = time.time()
    pjobs, pstats = group(rseq, rtok, check_questions)
    pstore, prows = read_answers(model, rtok, rprompts, pjobs, device, D, log_every=0)
    pinned = np.load(os.path.join(data_dir, reader, "%s_phase2_features.npz" % dataset))
    if not np.array_equal(np.asarray(pinned["prompt_id"])[prows],
                          np.asarray([int(rseq["prompt_id"][r]) for r in prows])):
        raise SystemExit("pinned features are not in sequences order -- cannot run the preflight")
    pre = {a: agreement(pstore[a], np.asarray(pinned[a])[prows]) for a in ARRAYS}
    pre_ok = all(v["corr"] is not None and v["corr"] >= CHECK_CORR for v in pre.values())
    print("  PREFLIGHT (reader reads itself, %d questions, %d answers, %.0fs): %s -> %s"
          % (len(pjobs), len(prows), time.time() - t0, json.dumps(pre), "PASS" if pre_ok else "FAIL"),
          flush=True)
    if not pre_ok:
        raise SystemExit("preflight failed: the decode/re-encode/pool route does not reproduce the "
                         "reader's own pinned features")
    del pinned

    jobs, stats = group(gseq, gtok, limit)
    t1 = time.time()
    store, rows = read_answers(model, rtok, rprompts, jobs, device, D, label="cross")
    order = np.argsort(np.asarray(rows), kind="stable")
    rows = np.asarray(rows)[order]
    store = {a: v[order] for a, v in store.items()}
    nonfinite = {a: int((~np.isfinite(store[a])).sum()) for a in ARRAYS}

    os.makedirs(dst_dir, exist_ok=True)
    gy = np.asarray(gseq["all_hallucination_flag"], dtype=int)
    gp = np.asarray(gseq["prompt_id"], dtype=int)
    np.savez_compressed(dst, generator_row=rows, prompt_id=gp[rows], label=gy[rows], **store)
    meta = {"reader": reader, "generator": generator, "dataset": dataset,
            "n_answers": int(len(rows)), "n_generator_answers": int(len(gy)),
            "hidden_state_indices": WINDOW, "hidden_size": int(D),
            "answer_stats": stats, "preflight_self_read": {"answers": len(prows), "agreement": pre,
                                                          "roundtrip_same_ids": pstats["roundtrip_same_ids"]},
            "nonfinite_entries": nonfinite, "elapsed_seconds": round(time.time() - t1, 1),
            "note": "rows follow the generator's sequences order (generator_row); labels are the "
                    "generator's own; features are the reader's hidden states"}
    with open(dst.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\n  wrote %s  (%.1f GB, answers %d, empty %d, eos appended %d, non-finite %s; preflight "
          "re-encoded ids identical %d/%d)"
          % (dst, os.path.getsize(dst) / 1024 ** 3, len(rows), stats["empty"], stats["eos_appended"],
             nonfinite, pstats["roundtrip_same_ids"], len(prows)))


# ---------------------------------------------------------------------------------------------

class _FakeTok:
    """Character-level tokenizer: id = ord(c) + offset; eos and a special id that decode drops."""

    def __init__(self, offset, eos, special=None):
        self.offset, self.eos_token_id, self.pad_token_id = offset, eos, None
        self.special = {eos} | ({special} if special is not None else set())

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + self.offset for c in text]

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(i - self.offset) for i in ids
                       if not (skip_special_tokens and i in self.special))


def self_test():
    print("=" * 78)
    print("  SELF-TEST: 65_extract_cross_read")
    print("=" * 78)
    g, r = _FakeTok(1000, eos=5, special=7), _FakeTok(3000, eos=9)

    ids, eos = reader_answer_ids(g.encode(" Paris.") + [5], g, r)
    assert r.decode(ids) == " Paris." and ids[-1] == 9 and eos
    ids, eos = reader_answer_ids(g.encode(" no\n"), g, r)
    assert r.decode(ids) == " no\n" and not eos and ids[-1] != 9
    ids, eos = reader_answer_ids([7], g, r)
    assert ids == [] and not eos
    ids, eos = reader_answer_ids([5], g, r)
    assert ids == [9] and eos
    print("  [PASS] answer text keeps its leading space and stop; generator EOS becomes reader EOS; "
          "a special-only answer is empty")

    prompt = "Q: capital of France?\nA:"
    rseq = {"input_ids": [r.encode(prompt) + r.encode(" x"), r.encode(prompt) + r.encode(" y")],
            "prompt_len": [len(prompt)] * 2, "prompt_id": [0, 0]}
    gseq = {"input_ids": [[7] + g.encode(prompt) + g.encode(" z")], "prompt_len": [len(prompt) + 1],
            "prompt_id": [0]}
    rp, gp = prompt_ids_by_question(rseq), prompt_ids_by_question(gseq)
    assert r.decode(rp[0]) == prompt and check_same_questions(rp, gp, r, g) == []
    gseq_bad = {"input_ids": [g.encode("Q: capital of Spain?\nA:")], "prompt_len": [24], "prompt_id": [0]}
    assert check_same_questions(rp, prompt_ids_by_question(gseq_bad), r, g) == [0]
    assert "character 14" in describe_mismatch("Q: capital of France?", "Q: capital of Spain?")
    # The same passage with a precomposed accent on one side and a decomposed one on the other.
    composed, decomposed = "Q: café?\nA:", "Q: café?\nA:"
    rs = {"input_ids": [r.encode(composed)], "prompt_len": [len(composed)], "prompt_id": [5]}
    gs = {"input_ids": [g.encode(decomposed)], "prompt_len": [len(decomposed)], "prompt_id": [5]}
    assert check_same_questions(prompt_ids_by_question(rs), prompt_ids_by_question(gs), r, g) == []
    try:
        prompt_ids_by_question({"input_ids": [r.encode("ab"), r.encode("ac")], "prompt_len": [2, 2],
                                "prompt_id": [3, 3]})
        raise AssertionError("different prompts for one question must be refused")
    except SystemExit:
        pass
    print("  [PASS] prompts are taken per question, a BOS on one side is ignored, a different "
          "question under the same id is caught")

    # pool_window: indices 16..24, tokens s:e only. h[i][t] = i*100 + t, so the peak of index i over
    # tokens 5..7 is i*100 + 7 and the update is 100 everywhere.
    T, D = 10, 3
    hs = [np.stack([np.full(D, i * 100.0 + t) for t in range(T)]) for i in range(29)]
    p = pool_window(hs, 5, 8)
    assert p["static_max"].shape == (9, D) and list(p["static_max"][:, 0]) == [i * 100.0 + 7 for i in WINDOW]
    assert p["velocity_q95"].shape == (8, D) and np.allclose(p["velocity_q95"], 100.0)
    assert np.allclose(p["static_q05"][0], 1600.0 + 5 + 0.1)          # q05 of 5,6,7 is 5.1
    print("  [PASS] pooling reads indices 16..24 and answer tokens only; update is h[i+1]-h[i]")

    a = np.arange(12.0).reshape(3, 4)
    assert abs(agreement(a, a + 1e-6)["corr"] - 1.0) < 1e-12
    print("  [PASS] agreement")
    print("\n  ALL PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--reader")
    ap.add_argument("--generator")
    ap.add_argument("--dataset", choices=["truthfulqa", "tydiqa_gp", "nq_open", "triviaqa"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check-questions", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="first N questions (timing runs only)")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not (a.reader and a.generator and a.dataset):
        raise SystemExit("--reader, --generator and --dataset are required (or --self-test)")
    data_dir = a.data_dir
    if not data_dir:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]
    run(a.reader, a.generator, a.dataset, data_dir, a.out_dir, a.device, a.check_questions, a.limit)


if __name__ == "__main__":
    main()
