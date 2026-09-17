"""
68_diagnose_generation.py -- why does a new model produce looping answers under our decoding? (T-014)
==================================================================================================

The first Falcon-H1-7B-Base pilot (67) loaded, generated deterministically and returned clean hidden
states, but its answers under our decoding config (10-beam sampled search) were mostly loops --
"If If If ...", "The The spiciest part of the spiciest part ...", "W W W W" -- and generation ran at
44 s per TruthfulQA question against 2.55 s for Qwen and LLaMA. Before deciding whether the model can
share Table 1's protocol, find which ingredient breaks it. Same questions, several decodings:

  greedy            num_beams=1, no sampling                  -> is the model itself fine?
  sample            num_beams=1, sampling, 10 returns          -> does sampling alone loop?
  beam              num_beams=10, no sampling                  -> does beam search alone loop?
  pipeline          our exact config (beam + sampling)         -> the failing case, re-measured
  pipeline_nocache  our config with use_cache=False            -> if this is clean and `pipeline` is not,
                                                                  the recurrent-state cache is mis-handled
                                                                  when beams are reordered
  greedy_bos /      the same with the tokenizer's BOS prepended, only if the tokenizer has one and our
  pipeline_bos      raw tokenisation does not add it          -> is the prompt missing a start token?

Reported per decoding: seconds per question, and the share of answers that loop (67.degenerate), are
empty, or run to max_new_tokens -- plus the first three answers to each question.

  python 68_diagnose_generation.py --model_folder falcon-h1-7b-base
"""

import argparse
import importlib.util
import json
import os
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def variants(gen, has_bos_gap):
    base = dict(max_new_tokens=gen["max_new_tokens"], temperature=gen["temperature"], top_k=gen["top_k"],
                top_p=gen["top_p"], early_stopping=True)
    v = [("greedy", dict(base, do_sample=False, num_beams=1, num_return_sequences=1), False),
         ("sample", dict(base, do_sample=True, num_beams=1, num_return_sequences=gen["num_return_sequences"]), False),
         ("beam", dict(base, do_sample=False, num_beams=gen["num_beams"],
                       num_return_sequences=gen["num_return_sequences"]), False),
         ("pipeline", dict(base, do_sample=gen["do_sample"], num_beams=gen["num_beams"],
                           num_return_sequences=gen["num_return_sequences"]), False),
         ("pipeline_nocache", dict(base, do_sample=gen["do_sample"], num_beams=gen["num_beams"],
                                   num_return_sequences=gen["num_return_sequences"], use_cache=False), False)]
    if has_bos_gap:
        v += [("greedy_bos", dict(v[0][1]), True), ("pipeline_bos", dict(v[3][1]), True)]
    for _, kw, _ in v:            # greedy/beam ignore sampling knobs; drop them to avoid warnings
        if not kw["do_sample"]:
            for k in ("temperature", "top_k", "top_p"):
                kw.pop(k, None)
        if kw["num_beams"] == 1:
            kw.pop("early_stopping", None)
    return v


def run(model_folder, n_tqa, n_tydi, out_dir):
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    s39 = _load("s39", "39_generate_dataset.py")
    s67 = _load("s67", "67_pilot_new_model.py")
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)
    gen = cfg["generation"]
    ds_cfgs = {d["name"]: d for d in cfg["datasets"]}
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
    model.eval()
    eos_ids = s39.compute_eos_ids(tok)

    questions = []
    for ds, n in (("truthfulqa", n_tqa), ("tydiqa_gp", n_tydi)):
        samples, _ = s39.load_dataset_samples(ds_cfgs[ds])
        questions += [(ds, s) for s in samples[:n]]
    first_ids = tok(questions[0][1]["prompt_text"]).input_ids
    bos = tok.bos_token_id
    has_bos_gap = bos is not None and (len(first_ids) == 0 or first_ids[0] != bos)
    report = {"model_folder": model_folder, "model_id": model_id,
              "tokenizer": {"bos_token": tok.bos_token, "bos_token_id": bos, "eos_token": tok.eos_token,
                            "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id,
                            "add_bos_token": getattr(tok, "add_bos_token", None),
                            "first_prompt_ids_head": first_ids[:8],
                            "bos_missing_from_raw_prompt": bool(has_bos_gap),
                            "stop_ids": sorted(int(i) for i in eos_ids)},
              "variants": {}}
    print("  tokenizer: %s" % json.dumps(report["tokenizer"]), flush=True)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "diagnose_%s.json" % model_folder)

    for name, kw, add_bos in variants(gen, has_bos_gap):
        stats = {"answers": 0, "degenerate": 0, "empty": 0, "hit_max_new_tokens": 0}
        times, shown = {"truthfulqa": [], "tydiqa_gp": []}, []
        try:
            for ds, sample in questions:
                seed = s39.prompt_seed(s39.GEN_SEED_DEFAULT, sample["prompt_id"])
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                ids = tok(sample["prompt_text"], return_tensors="pt").input_ids
                if add_bos:
                    ids = torch.cat([torch.tensor([[bos]]), ids], dim=1)
                ids = ids.to(device)
                pl = ids.shape[1]
                torch.cuda.synchronize(); t0 = time.time()
                with torch.no_grad():
                    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         eos_token_id=list(eos_ids), pad_token_id=tok.eos_token_id,
                                         return_dict_in_generate=True, **kw)
                torch.cuda.synchronize(); times[ds].append(time.time() - t0)
                texts = []
                for b in range(out.sequences.shape[0]):
                    raw = out.sequences[b, pl:].tolist()
                    canon = raw[:s39.find_canonical_length(raw, eos_ids)]
                    t = tok.decode(canon, skip_special_tokens=True).strip()
                    stats["answers"] += 1
                    stats["degenerate"] += int(s67.degenerate(canon))
                    stats["empty"] += int(not t)
                    stats["hit_max_new_tokens"] += int(len(canon) >= gen["max_new_tokens"])
                    texts.append(t)
                shown.append({"dataset": ds, "prompt_id": sample["prompt_id"], "answers": texts[:3]})
            share = {k: round(stats[k] / max(stats["answers"], 1), 3) for k in stats if k != "answers"}
            rec = {"generate_kwargs": {k: v for k, v in kw.items()}, "bos_prepended": add_bos,
                   "s_per_question": {d: round(float(np.mean(v)), 2) for d, v in times.items() if v},
                   "counts": stats, "share": share, "samples": shown}
        except Exception:  # noqa: BLE001
            rec = {"generate_kwargs": kw, "bos_prepended": add_bos, "error": traceback.format_exc(limit=4)}
        report["variants"][name] = rec
        if "error" in rec:
            print("\n  %-17s ERROR %s" % (name, rec["error"].strip().splitlines()[-1]), flush=True)
        else:
            print("\n  %-17s s/q %s | loops %.0f%%  empty %.0f%%  hit max %.0f%%"
                  % (name, rec["s_per_question"], 100 * rec["share"]["degenerate"], 100 * rec["share"]["empty"],
                     100 * rec["share"]["hit_max_new_tokens"]), flush=True)
            for s in shown:
                print("      %s q%s: %s" % (s["dataset"][:4], s["prompt_id"], s["answers"]), flush=True)
        with open(dst, "w") as f:
            json.dump(report, f, indent=2, default=str)
    print("\n  wrote %s" % dst)


def self_test():
    gen = {"max_new_tokens": 64, "temperature": 0.5, "top_k": 5, "top_p": 0.99, "num_beams": 10,
           "num_return_sequences": 10, "do_sample": True}
    v = dict((n, (kw, b)) for n, kw, b in variants(gen, True))
    assert list(v) == ["greedy", "sample", "beam", "pipeline", "pipeline_nocache", "greedy_bos", "pipeline_bos"]
    assert v["pipeline"][0] == {"max_new_tokens": 64, "temperature": 0.5, "top_k": 5, "top_p": 0.99,
                                "early_stopping": True, "do_sample": True, "num_beams": 10,
                                "num_return_sequences": 10}, v["pipeline"][0]
    assert v["pipeline_nocache"][0]["use_cache"] is False and v["pipeline_bos"][1] and not v["pipeline"][1]
    assert "temperature" not in v["greedy"][0] and v["greedy"][0]["num_return_sequences"] == 1
    assert "top_k" not in v["beam"][0] and v["beam"][0]["num_beams"] == 10
    assert len(variants(gen, False)) == 5
    print("  [PASS] 'pipeline' is exactly config.yaml's decoding; the others change one ingredient each")
    print("\n  ALL PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model_folder")
    ap.add_argument("--n-truthfulqa", type=int, default=5)
    ap.add_argument("--n-tydiqa", type=int, default=3)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "pilot"))
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not a.model_folder:
        raise SystemExit("--model_folder is required")
    run(a.model_folder, a.n_truthfulqa, a.n_tydiqa, a.out_dir)


if __name__ == "__main__":
    main()
