"""
74_extract_ragtruth.py -- token-level states of RAGTruth responses, read by the model that wrote them (T-024).
==================================================================================================

WHY. On short QA answers the hallucination IS the whole answer, so the token axis showed a flat shift and
nothing temporal (73, T-023). RAGTruth (Niu et al., ACL 2024) has responses of a few hundred tokens with
human-labeled hallucinated spans: an answer can be right for a while and then go wrong, and the labels say
where. That is data on which a temporal signal along the token axis can exist at all.

READ BY ITS OWN GENERATOR. A detector reads hidden states while a model writes. So each response is read by
the model that generated it -- Llama-2-7B-chat responses through Llama-2-7B-chat -- wrapped as RAGTruth's
open generators saw it, <s>[INST] {prompt} [/INST], followed by the response. Responses written by any other
model are never read.

WHAT IS STORED (../data-ragtruth/<folder>/), response tokens only, in the order of the kept responses:
    tokens.npy        (total_T, 9, 64) float32  per-token states of a 9-block window, robust-scaled and
                                                projected onto 64 directions per block
    offsets.npy       (N + 1,)                  response n occupies tokens[offsets[n]:offsets[n+1]]
    token_halluc.npy  (total_T,) uint8          1 if the token overlaps a human-labeled hallucinated span
    label.npy         (N,)                      1 if the response has any labeled span
    first_halluc.npy  (N,)                      position of the first hallucinated token, -1 if none
    split.npy         (N,)                      0 = RAGTruth train, 1 = RAGTruth test
    task.npy          (N,)                      0 = QA, 1 = Data2txt, 2 = Summary
    nll.npy           (N,)                      mean negative log-likelihood of the response tokens
    response_id.npy, source_id.npy (N,)
    meta.json                                   written LAST: a directory without it is incomplete

WHY PROJECTED, NOT RAW. Raw states for nine blocks would be ~40 GB per model; projected they are under
2 GB. The projection is fitted ONCE, on RAGTruth training responses only, so no test response shapes it.
Fit pass: read a random subset of training responses and fit 28's token scaler and a 64-direction basis per
block (73.fit_space -- the same fit the short-QA probe used). Store pass: read every kept response and store
its projection. The basis itself is not saved (AGENTS: no cached bases); meta.json records how it was fitted
and how much variance it keeps.

CHECKS, before the long pass:
  span check      every labeled span's text must equal response[start:end] (refuses above 1% mismatches)
  template check  on the first 32 kept responses, the mean response NLL with the [INST] wrapper must be
                  lower than without it. A wrong model or a wrong wrapper reads the text as someone else's
                  and fails here. (The same comparison across checkpoints is how an unknown generator
                  version would be identified.)

  python 74_extract_ragtruth.py --self-test
  python 74_extract_ragtruth.py --inspect                      # data only: no GPU, no model
  python 74_extract_ragtruth.py --generator llama-2-7b-chat
"""

import argparse
import importlib.util
import json
import os
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = 64
N_BLOCKS = 9
N_FIT = 40000              # tokens the scaler and basis are fitted on
FIT_POOL = 60000           # training-response tokens read in the fit pass before subsampling to N_FIT
N_TEMPLATE_CHECK = 32
SPAN_MISMATCH_MAX = 0.01
TASKS = {"QA": 0, "Data2txt": 1, "Summary": 2}
KEEP_QUALITY = ("good",)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ragtruth_cfg():
    import yaml
    with open(os.path.join(HERE, "config.yaml")) as f:
        return yaml.safe_load(f)["ragtruth"]


def resolve(path):
    return os.path.normpath(os.path.join(HERE, path))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------

def load_records(raw_dir, ragtruth_name, keep_quality=KEEP_QUALITY):
    """One generator's responses joined with their source prompts, training split first, then by id.
    Returns (records, counts of what was dropped and why)."""
    src = {str(s["source_id"]): s for s in read_jsonl(os.path.join(raw_dir, "source_info.jsonl"))}
    dropped = {"other_model": 0, "quality": 0, "no_source": 0}
    recs = []
    for r in read_jsonl(os.path.join(raw_dir, "response.jsonl")):
        if r["model"] != ragtruth_name:
            dropped["other_model"] += 1
            continue
        if r.get("quality", "good") not in keep_quality:
            dropped["quality"] += 1
            continue
        s = src.get(str(r["source_id"]))
        if s is None:
            dropped["no_source"] += 1
            continue
        recs.append({"id": str(r["id"]), "source_id": str(r["source_id"]), "split": r["split"],
                     "task": s["task_type"], "prompt": s["prompt"], "response": r["response"],
                     "labels": r.get("labels") or [], "temperature": r.get("temperature")})
    recs.sort(key=lambda x: (x["split"] != "train", x["id"]))
    return recs, dropped


def span_mismatches(recs):
    """Spans whose stored text does not equal response[start:end] (whitespace-trimmed)."""
    bad, total = [], 0
    for r in recs:
        for sp in r["labels"]:
            total += 1
            if r["response"][int(sp["start"]):int(sp["end"])].strip() != str(sp["text"]).strip():
                bad.append((r["id"], int(sp["start"]), int(sp["end"])))
    return bad, total


def token_flags(offset_mapping, spans):
    """1 for each token whose character range overlaps a labeled span [start, end)."""
    f = np.zeros(len(offset_mapping), dtype=np.uint8)
    for a, b in ((int(s["start"]), int(s["end"])) for s in spans):
        for i, (c0, c1) in enumerate(offset_mapping):
            if c1 > a and c0 < b:
                f[i] = 1
    return f


def first_true(flags):
    hit = np.flatnonzero(flags)
    return int(hit[0]) if len(hit) else -1


def build_ids(tok, prompt, response, wrap=True):
    """Token ids of the wrapped prompt followed by the response, the prompt length, and the response's
    character offsets. The response is tokenized on its own, so its first token carries the leading-space
    marker a generator emits right after [/INST], and the prompt/response boundary is exact."""
    head = "[INST] %s [/INST]" % prompt if wrap else prompt
    p = tok(head, add_special_tokens=True)["input_ids"]
    enc = tok(response, add_special_tokens=False, return_offsets_mapping=True)
    return list(p) + list(enc["input_ids"]), len(p), [tuple(o) for o in enc["offset_mapping"]]


# ---------------------------------------------------------------------------------------------
# model reads
# ---------------------------------------------------------------------------------------------

def read_response(model, ids, n_prompt, layer_idx, device):
    """(T, P, D) float32 states of the response tokens at hidden-state indices layer_idx, and the mean
    negative log-likelihood of the response tokens under the model."""
    import torch
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(input_ids=x, output_hidden_states=True)
        H = torch.stack([out.hidden_states[l][0, n_prompt:, :] for l in layer_idx], dim=1)
        logits = out.logits[0, n_prompt - 1:-1, :].float()
        nll = torch.nn.functional.cross_entropy(logits, x[0, n_prompt:]).item()
    return H.float().cpu().numpy(), nll


def project_response(H, space, s28):
    """(T, P, r): one response's states scaled and projected in each block's space (73.project_all's rule)."""
    Z = np.empty((H.shape[0], H.shape[1], space[0][2].shape[1]), dtype=np.float32)
    for j, (params, mu, V) in enumerate(space):
        Z[:, j] = (s28.apply_robust_scale(H[:, j, :], params) - mu) @ V
    return Z


def template_check(model, tok, recs, layer_idx, device, n):
    """Mean response NLL with and without the [INST] wrapper, on the first n records."""
    a, b = [], []
    for r in recs[:n]:
        ids, npr, _ = build_ids(tok, r["prompt"], r["response"], wrap=True)
        a.append(read_response(model, ids, npr, layer_idx[:1], device)[1])
        ids, npr, _ = build_ids(tok, r["prompt"], r["response"], wrap=False)
        b.append(read_response(model, ids, npr, layer_idx[:1], device)[1])
    return float(np.mean(a)), float(np.mean(b))


LLAMA2_DEFAULT_SYSTEM = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being "
    "safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
    "content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question "
    "does not make any sense, or is not factually coherent, explain why instead of answering something not "
    "correct. If you don't know the answer to a question, please don't share false information.")

TEMPLATE_VARIANTS = {
    "inst":                lambda p: "[INST] %s [/INST]" % p,
    "plain":               lambda p: p,
    "inst_no_spaces":      lambda p: "[INST]%s[/INST]" % p,
    "inst_default_system": lambda p: "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]" % (LLAMA2_DEFAULT_SYSTEM, p),
}


def response_token_nll(model, tok, head, response, device, extra_space=False):
    """Per-token negative log-likelihood of the response after `head` (BOS added to the head)."""
    import torch
    p = tok(head, add_special_tokens=True)["input_ids"]
    r = tok((" " + response) if extra_space else response, add_special_tokens=False)["input_ids"]
    x = torch.tensor([list(p) + list(r)], device=device)
    with torch.no_grad():
        logits = model(input_ids=x).logits[0, len(p) - 1:-1, :].float()
        nll = torch.nn.functional.cross_entropy(logits, x[0, len(p):], reduction="none")
    return nll.cpu().numpy()


def diagnose_template(generator, device, n, n_show=4):
    """Which way of presenting the prompt makes the generator's own responses most likely, and where in the
    response any difference comes from: the first token, tokens 2-5, or the rest. Writes nothing."""
    import collections
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    c = ragtruth_cfg()
    g = next(x for x in c["generators"] if x["folder"] == generator)
    recs, _ = load_records(resolve(c["raw_dir"]), g["ragtruth_name"])
    print("  [%s] temperature values in the release: %s" % (
        generator, dict(collections.Counter(str(r["temperature"]) for r in recs))), flush=True)
    starts = collections.Counter(" ".join(r["response"].split()[:2]) for r in recs)
    print("  most common first two words of responses: %s" % starts.most_common(8), flush=True)
    tok = AutoTokenizer.from_pretrained(g["id"])
    model = AutoModelForCausalLM.from_pretrained(g["id"], dtype=torch.bfloat16).to(device)
    model.eval()
    rng = np.random.default_rng(0)
    pick = [recs[i] for i in rng.choice(len(recs), size=min(n, len(recs)), replace=False)]

    variants = [(name, fn, False) for name, fn in TEMPLATE_VARIANTS.items()]
    variants.append(("inst_extra_space", TEMPLATE_VARIANTS["inst"], True))
    print("  NLL of the response tokens, mean over %d responses (lower = more likely):" % len(pick))
    print("    %-22s %8s %8s %8s %8s" % ("presentation", "all", "token 1", "2-5", "6+"))
    for name, fn, extra in variants:
        rows = [response_token_nll(model, tok, fn(r["prompt"]), r["response"], device, extra) for r in pick]
        print("    %-22s %8.3f %8.3f %8.3f %8.3f" % (
            name, np.mean([v.mean() for v in rows]), np.mean([v[0] for v in rows]),
            np.mean([v[1:5].mean() for v in rows if len(v) > 1]),
            np.mean([v[5:].mean() for v in rows if len(v) > 5])), flush=True)

    print("  what the model writes first, greedy, against what the release has:")
    for r in pick[:n_show]:
        for name in ("inst", "inst_default_system"):
            ids = tok(TEMPLATE_VARIANTS[name](r["prompt"]), add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=16, do_sample=False)
            print("    %-20s %r" % (name, tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)[:80]))
        print("    %-20s %r" % ("release", r["response"][:80]), flush=True)


def git_head():
    try:
        return subprocess.run(["git", "-C", HERE, "log", "--oneline", "-1"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------

def inspect(raw_dir):
    """What is in the release: responses per generator, quality, split, task, spans, span alignment."""
    resp = read_jsonl(os.path.join(raw_dir, "response.jsonl"))
    src = {str(s["source_id"]): s for s in read_jsonl(os.path.join(raw_dir, "source_info.jsonl"))}
    models = sorted({r["model"] for r in resp})
    print("  %d responses, %d sources" % (len(resp), len(src)))
    print("  %-24s %6s %6s %6s %6s %8s %7s" % ("model field", "total", "good", "train", "test", "halluc", "spans"))
    for m in models:
        rs = [r for r in resp if r["model"] == m]
        print("  %-24s %6d %6d %6d %6d %8d %7d" % (
            m, len(rs), sum(r.get("quality", "good") == "good" for r in rs),
            sum(r["split"] == "train" for r in rs), sum(r["split"] == "test" for r in rs),
            sum(bool(r.get("labels")) for r in rs), sum(len(r.get("labels") or []) for r in rs)))
    print("  quality values: %s" % sorted({str(r.get("quality")) for r in resp}))
    print("  task types: %s" % sorted({str(s["task_type"]) for s in src.values()}))
    for m in models:
        recs, _ = load_records(raw_dir, m, keep_quality=tuple({str(r.get("quality")) for r in resp}))
        bad, total = span_mismatches(recs)
        chars = np.array([len(r["response"]) for r in recs])
        print("  %-24s span check %d/%d mismatched | response chars median %d, 90th pct %d"
              % (m, len(bad), total, int(np.median(chars)), int(np.percentile(chars, 90))))


def run(generator, device, seed, max_len, n_fit, skip_template_check):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    c = ragtruth_cfg()
    g = next((x for x in c["generators"] if x["folder"] == generator), None)
    if g is None:
        raise SystemExit("unknown generator %r; config.yaml ragtruth.generators has %s"
                         % (generator, [x["folder"] for x in c["generators"]]))
    raw_dir = resolve(c["raw_dir"])
    out_dir = os.path.join(resolve(c["out_dir"]), generator)
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        raise SystemExit("refusing: %s is complete already" % out_dir)

    recs, dropped = load_records(raw_dir, g["ragtruth_name"])
    if not recs:
        names = sorted({r["model"] for r in read_jsonl(os.path.join(raw_dir, "response.jsonl"))})
        raise SystemExit("no responses with model field %r; the release has %s" % (g["ragtruth_name"], names))
    bad, total = span_mismatches(recs)
    print("  [%s] %d responses kept (dropped %s) | span check %d/%d mismatched"
          % (generator, len(recs), dropped, len(bad), total), flush=True)
    if total and len(bad) / total > SPAN_MISMATCH_MAX:
        raise SystemExit("refusing: %.1f%% of spans do not match their response text, e.g. %s"
                         % (100.0 * len(bad) / total, bad[:3]))

    tok = AutoTokenizer.from_pretrained(g["id"])
    if not tok.is_fast:
        raise SystemExit("need a fast tokenizer for character offsets")
    plan, skipped = [], {"empty": 0, "too_long": 0}
    for i, r in enumerate(recs):
        ids, npr, om = build_ids(tok, r["prompt"], r["response"])
        if len(ids) == npr:
            skipped["empty"] += 1
            continue
        if len(ids) > max_len:
            skipped["too_long"] += 1
            continue
        plan.append((i, ids, npr, token_flags(om, r["labels"])))
    lengths = np.array([len(ids) - npr for _, ids, npr, _ in plan], dtype=np.int64)
    offsets = np.zeros(len(plan) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    total_T = int(offsets[-1])
    print("  %d responses to read (skipped %s) | response tokens mean %.0f, median %.0f, max %d | "
          "tokens.npy %.2f GB" % (len(plan), skipped, lengths.mean(), np.median(lengths), lengths.max(),
                                  total_T * N_BLOCKS * R * 4 / 1e9), flush=True)
    shown = 0
    for i, ids, npr, _ in plan:
        if shown == 3 or not recs[i]["labels"]:
            continue
        sp = recs[i]["labels"][0]
        _, _, om = build_ids(tok, recs[i]["prompt"], recs[i]["response"])
        hit = np.flatnonzero(token_flags(om, [sp]))
        print("    alignment example: span %r -> tokens %r" % (str(sp["text"])[:60],
                                                             tok.decode([ids[npr + t] for t in hit])[:60]), flush=True)
        shown += 1

    model = AutoModelForCausalLM.from_pretrained(g["id"], dtype=torch.bfloat16).to(device)
    model.eval()
    n_layers = int(model.config.num_hidden_layers)
    first = int(g["first_block"])
    if first + N_BLOCKS > n_layers:
        raise SystemExit("window blocks %d..%d exceed the model's %d blocks" % (first, first + N_BLOCKS - 1, n_layers))
    layer_idx = list(range(first + 1, first + 1 + N_BLOCKS))      # hidden-state index of block b is b + 1
    kept = [recs[i] for i, _, _, _ in plan]

    t0 = time.time()
    nll_wrapped, nll_plain = template_check(model, tok, kept, layer_idx, device, N_TEMPLATE_CHECK)
    print("  template check: response NLL %.3f with the [INST] wrapper, %.3f without (%.0fs)"
          % (nll_wrapped, nll_plain, time.time() - t0), flush=True)
    if not nll_wrapped < nll_plain and not skip_template_check:
        raise SystemExit("refusing: the wrapper does not make the responses more likely -- wrong model or "
                         "wrong template, so the text would be read as someone else's")

    s28 = _load("s28", "28_eval_band.py")
    s73 = _load("s73", "73_token_structure.py")
    rng = np.random.default_rng(seed)
    train_pos = [k for k, (i, _, _, _) in enumerate(plan) if recs[i]["split"] == "train"]
    fit_H, fit_len, n_read = [], [], 0
    t0 = time.time()
    for k in rng.permutation(train_pos):
        _, ids, npr, _ = plan[k]
        H, _ = read_response(model, ids, npr, layer_idx, device)
        fit_H.append(H.astype(np.float16))
        fit_len.append(len(H))
        n_read += len(H)
        if n_read >= FIT_POOL:
            break
    fit_tokens = np.concatenate(fit_H, axis=0)
    del fit_H
    fit_off = np.zeros(len(fit_len) + 1, dtype=np.int64)
    np.cumsum(fit_len, out=fit_off[1:])
    space, share, used = s73.fit_space(fit_tokens, fit_off, np.arange(len(fit_len)), s28, n_fit, R, seed)
    del fit_tokens
    print("  fit pass: %d training responses, %d tokens read, fitted on %d | variance kept by 64 directions "
          "per block %s (%.0fs)" % (len(fit_len), n_read, used, " ".join("%.3f" % v for v in share),
                                    time.time() - t0), flush=True)

    os.makedirs(out_dir, exist_ok=True)
    mm = np.lib.format.open_memmap(os.path.join(out_dir, "tokens.npy"), mode="w+", dtype=np.float32,
                                   shape=(total_T, N_BLOCKS, R))
    nll = np.empty(len(plan), dtype=np.float64)
    t0 = time.time()
    for k, (i, ids, npr, _) in enumerate(plan):
        H, nll[k] = read_response(model, ids, npr, layer_idx, device)
        mm[offsets[k]:offsets[k + 1]] = project_response(H, space, s28)
        if (k + 1) % 200 == 0 or k + 1 == len(plan):
            el = time.time() - t0
            print("    store pass %d/%d (%.0fs, ~%.0fs left)" % (k + 1, len(plan), el,
                                                                el / (k + 1) * (len(plan) - k - 1)), flush=True)
    mm.flush()
    nonfinite = 0
    for a in range(0, total_T, 65536):
        nonfinite += int((~np.isfinite(mm[a:a + 65536])).sum())
    del mm

    flags = np.concatenate([fl for _, _, _, fl in plan])
    label = np.array([int(bool(recs[i]["labels"])) for i, _, _, _ in plan], dtype=np.int64)
    first_h = np.array([first_true(fl) for _, _, _, fl in plan], dtype=np.int64)
    disagree = int(((first_h >= 0).astype(int) != label).sum())
    arrays = {
        "offsets": offsets, "token_halluc": flags, "label": label, "first_halluc": first_h,
        "split": np.array([0 if recs[i]["split"] == "train" else 1 for i, _, _, _ in plan], dtype=np.int64),
        "task": np.array([TASKS.get(recs[i]["task"], -1) for i, _, _, _ in plan], dtype=np.int64),
        "nll": nll,
        "response_id": np.array([recs[i]["id"] for i, _, _, _ in plan]),
        "source_id": np.array([recs[i]["source_id"] for i, _, _, _ in plan]),
    }
    for name, arr in arrays.items():
        np.save(os.path.join(out_dir, "%s.npy" % name), arr)

    meta = {"generator": generator, "model_id": g["id"], "ragtruth_name": g["ragtruth_name"],
            "blocks": [first + j for j in range(N_BLOCKS)], "hidden_state_indices": layer_idx, "r": R,
            "n_responses": len(plan), "total_tokens": total_T, "dropped": dropped, "skipped": skipped,
            "max_len": max_len, "keep_quality": list(KEEP_QUALITY),
            "n_train": int((arrays["split"] == 0).sum()), "n_test": int((arrays["split"] == 1).sum()),
            "hallucinated_responses": int(label.sum()), "hallucinated_tokens": int(flags.sum()),
            "span_check": {"mismatched": len(bad), "total": total},
            "label_vs_token_flags_disagree": disagree,
            "template_check": {"nll_wrapped": nll_wrapped, "nll_plain": nll_plain, "n": N_TEMPLATE_CHECK,
                               "skipped": bool(skip_template_check)},
            "fit": {"responses": len(fit_len), "tokens_read": n_read, "tokens_fitted": used, "seed": seed,
                    "variance_kept_by_64_directions": share, "split_used": "RAGTruth train only"},
            "response_nll_mean": float(nll.mean()), "nonfinite_entries": nonfinite,
            "repo_head": git_head(), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("  done: %d responses, %d tokens (%d hallucinated), %d labeled hallucinated responses, "
          "label/flag disagreements %d, non-finite %d -> %s"
          % (len(plan), total_T, int(flags.sum()), int(label.sum()), disagree, nonfinite, out_dir), flush=True)


# ---------------------------------------------------------------------------------------------

def self_test():
    """Everything that does not need a GPU or the model, on synthetic records."""
    import tempfile
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print("    [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
        ok = ok and bool(cond)

    # spans to tokens: tokens [0,4) [4,9) [9,15) [15,20); span [5,12) covers tokens 1 and 2
    f = token_flags([(0, 4), (4, 9), (9, 15), (15, 20)], [{"start": 5, "end": 12}])
    check("span covers exactly the overlapping tokens", f.tolist() == [0, 1, 1, 0], str(f.tolist()))
    f = token_flags([(0, 4), (4, 9), (9, 15)], [{"start": 4, "end": 9}])
    check("span ending at a token boundary does not leak", f.tolist() == [0, 1, 0], str(f.tolist()))
    check("first hallucinated token", first_true(np.array([0, 0, 1, 1])) == 2 and first_true(np.zeros(3)) == -1)

    # loading, filtering and joining
    with tempfile.TemporaryDirectory() as d:
        src = [{"source_id": "s1", "task_type": "QA", "prompt": "P1", "source_info": {}},
               {"source_id": "s2", "task_type": "Summary", "prompt": "P2", "source_info": ""}]
        resp = [{"id": "9", "source_id": "s1", "model": "gen-a", "split": "test", "quality": "good",
                 "response": "The cafe opened in 2021.", "labels": [{"start": 19, "end": 23, "text": "2021"}]},
                {"id": "3", "source_id": "s2", "model": "gen-a", "split": "train", "quality": "good",
                 "response": "A summary.", "labels": []},
                {"id": "4", "source_id": "s2", "model": "gen-a", "split": "train", "quality": "truncated",
                 "response": "A summ", "labels": []},
                {"id": "5", "source_id": "s1", "model": "gen-b", "split": "train", "quality": "good",
                 "response": "x", "labels": []},
                {"id": "6", "source_id": "s1", "model": "gen-a", "split": "train", "quality": "good",
                 "response": "Wrong span here.", "labels": [{"start": 0, "end": 5, "text": "Span"}]}]
        for name, rows in (("source_info.jsonl", src), ("response.jsonl", resp)):
            with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
                fh.write("\n".join(json.dumps(x) for x in rows) + "\n")
        recs, dropped = load_records(d, "gen-a")
        check("keeps one generator, drops bad quality", [r["id"] for r in recs] == ["3", "6", "9"]
              and dropped == {"other_model": 1, "quality": 1, "no_source": 0}, "%s %s" % ([r["id"] for r in recs], dropped))
        check("training split comes first", [r["split"] for r in recs] == ["train", "train", "test"])
        check("prompt and task joined from the source", recs[2]["prompt"] == "P1" and recs[2]["task"] == "QA")
        bad, total = span_mismatches(recs)
        check("span check catches a span whose text does not match", bad == [("6", 0, 5)] and total == 2,
              "%s of %d" % (bad, total))

    # prompt / response boundary with a whitespace tokenizer that reports offsets
    class FakeTok:
        def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
            words, offs, pos = text.split(" "), [], 0
            for w in words:
                offs.append((pos, pos + len(w)))
                pos += len(w) + 1
            ids = [hash(w) % 1000 for w in words]
            out = {"input_ids": ([1] if add_special_tokens else []) + ids}
            if return_offsets_mapping:
                out["offset_mapping"] = offs
            return out
    ids, npr, om = build_ids(FakeTok(), "Summarize this", "It opened in 2021.")
    check("wrapped prompt length counts BOS and the wrapper", npr == 1 + 4, "n_prompt %d" % npr)
    check("response tokens follow the prompt", len(ids) == npr + 4 and om[3] == (13, 18), "%d ids, last offset %s" % (len(ids), om[3]))
    _, npr_plain, _ = build_ids(FakeTok(), "Summarize this", "x", wrap=False)
    check("unwrapped prompt is shorter", npr_plain == 1 + 2, "n_prompt %d" % npr_plain)

    # the store pass projects each response exactly as 73.project_all projects the concatenation
    s28 = _load("s28", "28_eval_band.py")
    s73 = _load("s73", "73_token_structure.py")
    rng = np.random.default_rng(0)
    lens = rng.integers(3, 40, size=60)
    off = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    D, P = 40, 3
    U = np.linalg.qr(rng.normal(size=(D, 5)))[0]
    raw = (rng.normal(size=(off[-1], P, 5)) @ U.T * 3.0 + 0.2 * rng.normal(size=(off[-1], P, D))).astype(np.float16)
    space, share, used = s73.fit_space(raw, off, np.arange(40), s28, 5000, 5, seed=0)
    ref = s73.project_all(raw, space, s28, chunk=500)
    per = np.concatenate([project_response(np.asarray(raw[off[n]:off[n + 1]], dtype=np.float32), space, s28)
                          for n in range(len(lens))])
    check("per-response projection equals the reference projection", np.allclose(per, ref, atol=1e-5),
          "max diff %.2e" % float(np.abs(per - ref).max()))
    check("planted directions keep most variance", min(share) > 0.8, str(["%.3f" % v for v in share]))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--inspect", action="store_true", help="summarize the release; needs no GPU or model")
    ap.add_argument("--generator", help="a folder from config.yaml ragtruth.generators")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=4096, help="prompt + response tokens (Llama-2 context)")
    ap.add_argument("--n-fit", type=int, default=N_FIT)
    ap.add_argument("--skip-template-check", action="store_true")
    ap.add_argument("--diagnose-template", action="store_true",
                    help="compare ways of presenting the prompt on --n-diagnose responses; writes nothing")
    ap.add_argument("--n-diagnose", type=int, default=64)
    a = ap.parse_args()
    if a.self_test:
        print("  self-test")
        raise SystemExit(0 if self_test() else 1)
    if a.inspect:
        inspect(resolve(ragtruth_cfg()["raw_dir"]))
        return
    if not a.generator:
        raise SystemExit("--generator is required (or --self-test / --inspect)")
    if a.diagnose_template:
        diagnose_template(a.generator, a.device, a.n_diagnose)
        return
    run(a.generator, a.device, a.seed, a.max_len, a.n_fit, a.skip_template_check)


if __name__ == "__main__":
    main()
