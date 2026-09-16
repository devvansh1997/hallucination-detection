"""
67_pilot_new_model.py -- can a new model go through this pipeline unchanged? (T-014)
==================================================================================================

Before committing days of GPU time to a third model, check in about an hour that every model-specific
step of the pipeline works on it, and measure what the full run will cost. Written for Falcon-H1-7B-Base
(a hybrid Mamba-2 + attention model), whose cache and layer layout differ from the two transformers
already in the paper; nothing here is specific to it.

GATES, in order. A failed gate is reported and the later gates still run where they can, so one job
shows everything that needs fixing.
  G1 load        AutoModelForCausalLM loads natively (no trust_remote_code) in bf16 on one GPU.
  G2 generate    Our EXACT decoding config (config.yaml: 10-beam sampled search, stop tokens, per-prompt
                 seeds) runs on a few TruthfulQA and TyDiQA-GP questions, gives non-empty answers, and
                 is deterministic: regenerating the first question with the same seed gives the same
                 sequences. If this fails the model cannot share Table 1's protocol.
  G3 hidden      output_hidden_states returns num_hidden_layers + 1 states of shape (B, T, D), finite,
                 and the reported window (hidden-state indices 16..24) exists.
  G4 plumbing    27_extract_band.verify_post_norm_route (used by 42 before extraction) and
                 compute_bases (the lm_head SVD HARP also relies on) run on this model.
  G5 cost        Seconds per question for generation and for the extraction forward pass, extrapolated
                 to all four datasets.

Labels are not computed (BLEURT is the same for every model); answers are printed for a sanity read.

  python 67_pilot_new_model.py --model_folder falcon-h1-7b-base
"""

import argparse
import importlib.util
import json
import os
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_QUESTIONS = {"truthfulqa": 817, "tydiqa_gp": 440, "nq_open": 3610, "triviaqa": 9960}
SHORT_PROMPT_RATE_FROM = "truthfulqa"          # NQ-Open and TriviaQA use the same short template
WINDOW = list(range(16, 25))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def extrapolate(rates):
    """rates: {'truthfulqa': s/question, 'tydiqa_gp': s/question} -> hours per dataset."""
    out = {}
    for ds, n in N_QUESTIONS.items():
        r = rates.get(ds, rates.get(SHORT_PROMPT_RATE_FROM))
        out[ds] = None if r is None else round(n * r / 3600.0, 2)
    return out


def run(model_folder, n_tqa, n_tydi, out_dir):
    import torch
    import transformers
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    s39 = _load("s39", "39_generate_dataset.py")
    s27 = s39.band_mod
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)
    gen = cfg["generation"]
    ds_cfgs = {d["name"]: d for d in cfg["datasets"]}
    report = {"model_folder": model_folder, "model_id": model_id, "gates": {},
              "versions": {"transformers": transformers.__version__, "torch": torch.__version__}}
    for k in ("mamba_ssm", "causal_conv1d"):
        try:
            __import__(k)
            report["versions"][k] = "available"
        except Exception as e:  # noqa: BLE001
            report["versions"][k] = "missing (%s)" % type(e).__name__
    print("  versions: %s" % json.dumps(report["versions"]), flush=True)

    def gate(name, ok, **info):
        report["gates"][name] = dict(passed=bool(ok), **info)
        print("  [%s] %s %s" % ("PASS" if ok else "FAIL", name, json.dumps(info, default=str)), flush=True)

    # G1 --------------------------------------------------------------------------------------------
    device = torch.device("cuda")
    try:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
        model.eval()
        c = model.config
        gate("G1_load", True, cls=type(model).__name__, model_type=c.model_type,
             layers=c.num_hidden_layers, hidden=c.hidden_size, vocab=c.vocab_size,
             params_b=round(sum(p.numel() for p in model.parameters()) / 1e9, 2),
             tied_embeddings=bool(getattr(c, "tie_word_embeddings", False)),
             seconds=round(time.time() - t0, 1))
    except Exception:  # noqa: BLE001
        gate("G1_load", False, error=traceback.format_exc(limit=3))
        return report

    # G2 --------------------------------------------------------------------------------------------
    eos_ids = s39.compute_eos_ids(tok)
    kwargs = dict(max_new_tokens=gen["max_new_tokens"], eos_token_id=list(eos_ids), do_sample=gen["do_sample"],
                  temperature=gen["temperature"], top_k=gen["top_k"], top_p=gen["top_p"],
                  num_beams=gen["num_beams"], num_return_sequences=gen["num_return_sequences"],
                  return_dict_in_generate=True, pad_token_id=tok.eos_token_id, early_stopping=True)
    rates, samples_out, first = {}, {}, None
    for ds, n in (("truthfulqa", n_tqa), ("tydiqa_gp", n_tydi)):
        try:
            samples, _ = s39.load_dataset_samples(ds_cfgs[ds])
        except Exception:  # noqa: BLE001
            gate("G2_generate_%s" % ds, False, error="dataset load: " + traceback.format_exc(limit=2))
            continue
        times, comp_lens, hit_max, empty, shown = [], [], 0, 0, []
        try:
            for sample in samples[:n]:
                seed = s39.prompt_seed(s39.GEN_SEED_DEFAULT, sample["prompt_id"])
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                inputs = tok(sample["prompt_text"], return_tensors="pt").to(device)
                pl = inputs.input_ids.shape[1]
                torch.cuda.synchronize(); t0 = time.time()
                with torch.no_grad():
                    outs = model.generate(**inputs, **kwargs)
                torch.cuda.synchronize(); times.append(time.time() - t0)
                texts, fulls = [], []
                for b in range(outs.sequences.shape[0]):
                    raw = outs.sequences[b, pl:].tolist()
                    canon = raw[:s39.find_canonical_length(raw, eos_ids)]
                    hit_max += int(len(canon) >= gen["max_new_tokens"])
                    comp_lens.append(len(canon))
                    t = tok.decode(canon, skip_special_tokens=True).strip()
                    empty += int(not t)
                    texts.append(t)
                    fulls.append(inputs.input_ids[0].tolist() + canon)
                shown.append({"prompt_id": sample["prompt_id"], "answers": texts[:3]})
                if first is None:
                    first = (sample, seed, outs.sequences.cpu(), fulls, pl)
            n_ans = len(comp_lens)
            rates[ds] = float(np.mean(times))
            samples_out[ds] = shown
            gate("G2_generate_%s" % ds, n_ans > 0 and empty < n_ans, questions=len(times), answers=n_ans,
                 s_per_question=round(rates[ds], 2), mean_answer_tokens=round(float(np.mean(comp_lens)), 1),
                 hit_max_new_tokens=hit_max, empty=empty)
            for s in shown:
                print("      q%s: %s" % (s["prompt_id"], s["answers"]), flush=True)
        except Exception:  # noqa: BLE001
            gate("G2_generate_%s" % ds, False, error=traceback.format_exc(limit=4))

    if first is not None:
        sample, seed, seqs, fulls, pl = first
        try:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            inputs = tok(sample["prompt_text"], return_tensors="pt").to(device)
            with torch.no_grad():
                again = model.generate(**inputs, **kwargs).sequences.cpu()
            same = again.shape == seqs.shape and bool(torch.equal(again, seqs))
            gate("G2_determinism", same, prompt_id=sample["prompt_id"])
        except Exception:  # noqa: BLE001
            gate("G2_determinism", False, error=traceback.format_exc(limit=3))

    # G3 --------------------------------------------------------------------------------------------
    fwd_rates = {}
    if first is not None:
        _, _, _, fulls, pl = first
        try:
            L = max(len(x) for x in fulls)
            ids = torch.full((len(fulls), L), int(tok.pad_token_id), dtype=torch.long)
            att = torch.zeros((len(fulls), L), dtype=torch.long)
            for j, x in enumerate(fulls):
                ids[j, :len(x)] = torch.tensor(x)
                att[j, :len(x)] = 1
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad():
                out = model(ids.to(device), attention_mask=att.to(device), use_cache=False,
                            output_hidden_states=True)
            torch.cuda.synchronize(); fwd_rates["truthfulqa"] = time.time() - t0
            hs = out.hidden_states
            shapes = sorted({tuple(h.shape) for h in hs})
            finite = all(bool(torch.isfinite(h.float()).all()) for h in hs)
            n_expected = model.config.num_hidden_layers + 1
            ok = len(hs) == n_expected and len(shapes) == 1 and finite and n_expected > max(WINDOW)
            mx = max(float(h.float().abs().max()) for h in hs)
            gate("G3_hidden_states", ok, n_states=len(hs), expected=n_expected, shapes=shapes, finite=finite,
                 max_abs_activation=round(mx, 1), float16_safe=mx < 65504,
                 window_blocks="15-23 of %d (%.0f%%-%.0f%% depth)" % (
                     model.config.num_hidden_layers, 100 * 15 / model.config.num_hidden_layers,
                     100 * 23 / model.config.num_hidden_layers))
            del out, hs
        except Exception:  # noqa: BLE001
            gate("G3_hidden_states", False, error=traceback.format_exc(limit=4))

    # G4 --------------------------------------------------------------------------------------------
    try:
        s42 = _load("s42", "42_extract_phase2.py")
        route, agree = s27.verify_post_norm_route(model, tok, s42.A2_PROBE_TEXTS, device, min_tokens=128)
        gate("G4_post_norm_route", True, route=route, agreement=round(float(agree), 4))
    except Exception:  # noqa: BLE001
        gate("G4_post_norm_route", False, error=traceback.format_exc(limit=4))
    try:
        V_R, V_rand, spectrum = s27.compute_bases(model)
        gate("G4_lm_head_bases", True, V_R=tuple(V_R.shape), V_rand=tuple(V_rand.shape),
             lm_head=tuple(model.lm_head.weight.shape),
             harp_needs=dict(lm_head=hasattr(model, "lm_head"),
                             num_hidden_layers=hasattr(model.config, "num_hidden_layers")))
    except Exception:  # noqa: BLE001
        gate("G4_lm_head_bases", False, error=traceback.format_exc(limit=4))

    # G5 --------------------------------------------------------------------------------------------
    report["generation_hours_estimate"] = extrapolate(rates)
    report["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 1)
    report["samples"] = samples_out
    report["extraction_forward_s_first_question"] = fwd_rates
    gate("G5_cost", bool(rates), generation_s_per_question=rates,
         generation_hours_if_run_alone=report["generation_hours_estimate"],
         reference_rates_qwen_llama="TruthfulQA 2.55 s/q, short-answer datasets 1.29 s/q (39_generate_dataset.py)",
         peak_gpu_gb=report["peak_gpu_gb"])

    report["all_passed"] = all(g["passed"] for g in report["gates"].values())
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "pilot_%s.json" % model_folder)
    with open(dst, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n  ALL GATES PASSED: %s   (wrote %s)" % (report["all_passed"], dst))
    return report


def self_test():
    h = extrapolate({"truthfulqa": 2.0, "tydiqa_gp": 3.6})
    assert h["truthfulqa"] == round(817 * 2.0 / 3600, 2) and h["tydiqa_gp"] == round(440 * 3.6 / 3600, 2)
    assert h["triviaqa"] == round(9960 * 2.0 / 3600, 2) and h["nq_open"] == round(3610 * 2.0 / 3600, 2)
    assert extrapolate({})["triviaqa"] is None
    print("  [PASS] extrapolation uses the TruthfulQA rate for the short-prompt datasets, TyDiQA-GP its own")
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
        raise SystemExit("--model_folder is required (it must have an entry in config.yaml)")
    r = run(a.model_folder, a.n_truthfulqa, a.n_tydiqa, a.out_dir)
    raise SystemExit(0 if r.get("all_passed") else 1)


if __name__ == "__main__":
    main()
