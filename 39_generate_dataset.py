"""
39_generate_dataset.py -- Session 06 Phase 1, Steps 1+2: Dataset Creation (TriviaQA/NQ-Open/TyDiQA-GP)
===========================================================================================================
GPU (generation), with a CPU-only --audit-only preview mode for Step 1. Scope is strictly
generate + label + pin -- NO hidden-state capture, no pooling, no feature extraction this session
(the raw-state-store pattern from sessions 04/05 does not apply here; we only need sequences,
decoded text, and labels this time, which is why this is far cheaper than 34_gate_reconstruct_or_
regenerate.py's run_route_n).

Dataset loading (Step 1 audit): reuses 01_generate_full_beams.py's loader logic verbatim
(TriviaQA dedup by question_id, NQ-Open, TyDiQA-GP English-filter + context handling) --
re-implemented here rather than imported, since that file is a top-level script with
argparse.parse_args() at import time (same reuse constraint as every other numbered script this
project). Cross-checked against 01_generate_full_beams.py:60-69: both loaders read the SAME
config.yaml hf_path/hf_config entries (neither hardcodes a different source), confirmed by
inspection, not just assumption. Labeling uses the SIMPLE two-threshold formula that actually
produced the TruthfulQA v3 data (21_generate_maxpool_datasets.py / 29_generate_extract_band.py /
34_gate_reconstruct_or_regenerate.py's run_route_n), NOT 01_generate_full_beams.py's more complex
contrastive judge -- per this session's explicit instruction.

*** CORRECTION to this session's brief: there is no chat template anywhere in this pipeline's
actual generation code (01_generate_full_beams.py:222, 21_generate_maxpool_datasets.py,
29_generate_extract_band.py, 34_gate_reconstruct_or_regenerate.py's run_route_n all call
tokenizer(prompt_text) directly on the raw config.yaml template string). "Same chat template as
v3" is replicated as "the same NO-chat-template raw-string tokenization", not as introducing one
that was never actually used -- see Deviations.

Fixes the window bug session05b diagnosed: completion token ids are truncated to the CANONICAL
window (content through the first stop-token inclusive) at save time, not v2's buggy
literal-eos_token_id-only filter.

*** ADDENDUM 2 (full validation splits, no subsampling): supersedes the original 2,000-prompt
cap. Every dataset now uses its FULL standard validation split -- no apply_cap, no subsample
seed. Raw HF split length is hard-asserted for triviaqa (17,944) and nq_open (3,610) as a
config/version sanity check (catches loading the wrong hf_config/split silently); tydiqa_gp has
no fixed expected raw length (multi-language split), so its English-filtered count is reported
only, informally checked against ~440. TriviaQA's existing question_id dedup (inherited from
01_generate_full_beams.py, confirmed identical config source) is KEPT: it reduces the raw 17,944
rows to ~9,960 unique questions (multiple rows share a question_id across different evidence
documents even in the "nocontext" config). This means the actual generation N is ~9,960, not
17,944 -- the addendum's "~22x TruthfulQA" figure (17,944/817) appears to assume the undeduped
raw count. Flagging this explicitly rather than silently picking a side: both the raw (asserted)
and post-dedup (actual) counts are printed and pinned in the manifest so this is never ambiguous
after the fact. Alias-max scoring: TriviaQA's correct_answers now includes the full
answer.aliases list (deduped, value first), not just answer.value -- NQ-Open and TyDiQA-GP
already used their full multi-answer fields and are unchanged.

Usage:
  python 39_generate_dataset.py --self-test
  python 39_generate_dataset.py --dataset triviaqa --model_folder llama-3.1-8b-instruct --audit-only
  python 39_generate_dataset.py --dataset triviaqa --model_folder llama-3.1-8b-instruct
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
_job_id = os.environ.get("SLURM_JOB_ID", "local")
os.environ["HF_METRICS_CACHE"] = f"/tmp/rouge_cache_{_job_id}"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


band_mod = _load("s02_extract", "27_extract_band.py")
window_mod = _load("s05b_window", "38_window_forensics.py")

compute_eos_ids = window_mod.compute_eos_ids
find_canonical_length = window_mod.find_canonical_length

# v3's TruthfulQA regeneration's recorded environment (from raw_state_meta.json's decoding_config,
# quoted in reports/session05_extraction_forensics.md) -- asserted against, not silently accepted.
EXPECTED_VERSIONS = {"transformers": "5.13.0", "torch": "2.13.0+cu126", "cuda": "12.6"}
GEN_SEED_DEFAULT = 0

# Empirically measured per-prompt generation rates (real cluster runs, same decoding config).
# TruthfulQA's open-ended answers are much longer than TriviaQA/NQ-Open/TyDiQA-GP's short
# factoid answers, so 2.55 is a conservative upper bound for the latter three, not a good
# central estimate -- both are shown when printing resource estimates rather than picking one.
EMPIRICAL_S_PER_PROMPT_LONG_ANSWER = 2.55    # TruthfulQA v3 real run: 2080s / 817 prompts
EMPIRICAL_S_PER_PROMPT_SHORT_ANSWER = 1.29   # TriviaQA real run: 2581s / 2000 prompts

# Raw HF validation-split length, hard-asserted at load time as a config/version sanity check
# (session06 Phase-1 addendum). tydiqa_gp has no fixed expected value (multi-language split;
# its English-filtered count is reported informally instead -- see load_dataset_samples).
EXPECTED_SPLIT_LENGTH = {"triviaqa": 17944, "nq_open": 3610}


# ==============================================================================
# SMALL PURE HELPERS (independently self-testable, no network)
# ==============================================================================

def assert_split_length(name, raw_len):
    """Hard-fails if the raw loaded HF split doesn't match the expected, pinned length -- catches
    silently loading the wrong hf_config/split/dataset version. No expected value -> info-only."""
    expected = EXPECTED_SPLIT_LENGTH.get(name)
    if expected is not None:
        assert raw_len == expected, (
            f"{name}: raw HF validation split length {raw_len} != expected {expected}. This "
            f"likely means a different hf_config/split/dataset version is being loaded than "
            f"intended -- check config.yaml's hf_path/hf_config for '{name}'.")
        print(f"  [PASS] {name} raw validation split length == {expected} (asserted)")
    else:
        print(f"  [INFO] {name} raw validation split length = {raw_len} (no fixed expected value; reported only)")
    return raw_len


def dedup_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_tydiqa_language(example_id):
    """TyDiQA's secondary_task config has NO 'language' field (real schema, confirmed by
    inspection: answers/context/id/question/title only) -- language is encoded as the prefix of
    'id' instead (e.g. "arabic-2387335860751143628-1" -> "arabic"). The originally inherited
    filter (ex.get("language", "english") != "english", copied verbatim from
    01_generate_full_beams.py) was a silent no-op: since the key never exists, .get()'s default
    always won and every row passed through regardless of actual language -- caught only after
    a real cluster run's eyeball examples showed Arabic/Korean/Telugu/Finnish/Swahili/Indonesian
    text in what was supposed to be an English-only dataset. This is a real, pre-existing bug in
    the old loader too; fixed here (39_*) only, per the additive-only rule against 01-25."""
    return example_id.split("-")[0]


# ==============================================================================
# DATASET LOADING (Step 1) -- verbatim re-implementation of 01_generate_full_beams.py's loader,
# extended per the Phase-1 addendum: full split (no cap), alias-max for TriviaQA.
# ==============================================================================

def load_dataset_samples(ds_cfg):
    from datasets import load_dataset
    name = ds_cfg["name"]
    ds = load_dataset(ds_cfg["hf_path"], ds_cfg["hf_config"], split="validation")
    raw_split_len = len(ds)
    assert_split_length(name, raw_split_len)
    template = ds_cfg["prompt_template"]
    samples = []

    if name == "triviaqa":
        seen = set()
        i = 0
        for ex in ds:
            qid = ex["question_id"]
            if qid in seen:
                continue
            seen.add(qid)
            aliases = ex["answer"].get("aliases", [])
            correct_answers = dedup_preserve_order([str(ex["answer"]["value"])] + [str(a) for a in aliases])
            samples.append({"prompt_id": i, "prompt_text": template.format(question=ex["question"]),
                             "correct_answers": correct_answers, "incorrect_answers": []})
            i += 1
        print(f"  [INFO] triviaqa: {raw_split_len} raw rows -> {len(samples)} unique questions "
              f"after question_id dedup (inherited from 01_generate_full_beams.py's loader; "
              f"multiple rows per question exist even in 'rc.nocontext' since it strips context "
              f"text only, not the per-evidence-document row structure)")
    elif name == "nq_open":
        for i, ex in enumerate(ds):
            samples.append({"prompt_id": i, "prompt_text": template.format(question=ex["question"]),
                             "correct_answers": [str(a) for a in ex["answer"]], "incorrect_answers": []})
    elif name == "tydiqa_gp":
        i = 0
        for ex in ds:
            if extract_tydiqa_language(ex["id"]) != "english":
                continue
            ctx = ex["context"][0] if isinstance(ex["context"], list) else ex["context"]
            samples.append({"prompt_id": i,
                             "prompt_text": template.format(context=str(ctx), question=ex["question"]),
                             "correct_answers": [str(a) for a in ex["answers"]["text"] if a],
                             "incorrect_answers": []})
            i += 1
        print(f"  [INFO] tydiqa_gp: {raw_split_len} raw (all-language) rows -> {len(samples)} "
              f"English-filtered questions (expected ~440 per Phase-1 addendum; informal check, not asserted)")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return samples, raw_split_len


# ==============================================================================
# DETERMINISTIC PER-PROMPT SEEDING
# ==============================================================================

def prompt_seed(global_seed, prompt_id):
    """SHA256-based, not Python's built-in hash() -- avoids any PYTHONHASHSEED/version-dependent
    hash-algorithm risk for a function whose whole point is bit-for-bit reproducibility."""
    h = hashlib.sha256(f"{global_seed}:{prompt_id}".encode()).hexdigest()
    return int(h, 16) % (2 ** 31)


# ==============================================================================
# LABELING -- the simple two-threshold formula that produced the actual TruthfulQA v3 data
# ==============================================================================

def is_correct_simple(gen_text, correct, incorrect, rouge, bleurt, rouge_threshold=0.7,
                       sen_sim_threshold=0.5):
    """correct may now be a multi-alias list (TriviaQA) -- ROUGE-L and BLEURT are each computed
    against every reference and the MAX is taken, per the alias-max scoring rule."""
    r = rouge.compute(predictions=[gen_text] * len(correct), references=correct) if correct else {"rougeL": 0.0}
    rl = r["rougeL"]
    all_refs = correct + incorrect
    max_correct_b = 0.0
    if all_refs:
        bs = bleurt.compute(predictions=[gen_text] * len(all_refs), references=all_refs)
        max_correct_b = max(bs["scores"][:len(correct)], default=0.0)
    return (rl >= rouge_threshold) or (max_correct_b > sen_sim_threshold)


# ==============================================================================
# VERSION ASSERTION
# ==============================================================================

def check_versions(force=False):
    import transformers
    actual = {"transformers": transformers.__version__, "torch": torch.__version__,
              "cuda": torch.version.cuda if torch.cuda.is_available() else None}
    mismatches = []
    if actual["transformers"] != EXPECTED_VERSIONS["transformers"]:
        mismatches.append(f"transformers: expected {EXPECTED_VERSIONS['transformers']}, got {actual['transformers']}")
    if actual["torch"] != EXPECTED_VERSIONS["torch"]:
        mismatches.append(f"torch: expected {EXPECTED_VERSIONS['torch']}, got {actual['torch']}")
    if actual["cuda"] != EXPECTED_VERSIONS["cuda"]:
        mismatches.append(f"cuda: expected {EXPECTED_VERSIONS['cuda']}, got {actual['cuda']}")
    if mismatches and not force:
        raise RuntimeError(
            "Library version mismatch vs the TruthfulQA v3 regeneration (asserted per this "
            "session's explicit instruction, since v2's original extraction's unlogged versions "
            "were a real contributing factor to session05b's forensics investigation):\n  " +
            "\n  ".join(mismatches) + "\nRe-run with --force-version-mismatch if this is intentional.")
    return actual, mismatches


# ==============================================================================
# RESOURCE ESTIMATE -- printed BEFORE launching generation (Step 1 spec requirement), shared by
# both --audit-only and the real run (so it's satisfied even in a single combined invocation).
# ==============================================================================

def print_resource_estimate(name, n_final):
    est_s_short = n_final * EMPIRICAL_S_PER_PROMPT_SHORT_ANSWER
    est_s_long = n_final * EMPIRICAL_S_PER_PROMPT_LONG_ANSWER
    est_disk_mb = n_final * 10 * 0.002   # ~2KB/beam empirically for token ids + decoded text
    print(f"  N (full validation split, no subsampling): {n_final}")
    print(f"  Estimated GPU time: {est_s_short/3600:.2f}-{est_s_long/3600:.2f} hours "
          f"({n_final} prompts x 1.29-2.55 s/prompt -- 1.29 is TriviaQA's real measured rate for "
          f"short factoid answers, 2.55 is TruthfulQA's real measured rate for longer open-ended "
          f"answers; NQ-Open/TyDiQA-GP are structurally closer to TriviaQA's short-answer style)")
    print(f"  Estimated disk: ~{est_disk_mb:.0f} MB (sequences+text+labels only -- no hidden "
          f"states captured this session)")
    return est_s_short, est_s_long, est_disk_mb


# ==============================================================================
# STEP 1 -- AUDIT-ONLY PREVIEW (CPU, no model)
# ==============================================================================

def run_audit(ds_cfg):
    print(f"[Step 1 audit] {ds_cfg['name']}: loading from {ds_cfg['hf_path']} "
          f"(config={ds_cfg['hf_config']}) ...")
    print(f"  [Cross-check] 01_generate_full_beams.py:60-69 reads the SAME config.yaml entry "
          f"(hf_path={ds_cfg['hf_path']}, hf_config={ds_cfg['hf_config']}) -- confirmed identical "
          f"by inspection, not independently hardcoded elsewhere.")
    samples, raw_split_len = load_dataset_samples(ds_cfg)
    n_final = len(samples)
    print_resource_estimate(ds_cfg["name"], n_final)
    return {"raw_split_len": raw_split_len, "n_final": n_final}


# ==============================================================================
# STEP 2 -- GENERATE + LABEL + PIN (GPU)
# ==============================================================================

def run_generation(ds_cfg, model_folder, global_seed, out_dir, force_versions=False,
                    n_determinism_check=5):
    import evaluate
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)
    gen_cfg = cfg["generation"]

    versions, mismatches = check_versions(force=force_versions)
    if mismatches:
        print(f"  [WARN] proceeding despite version mismatches (--force-version-mismatch): {mismatches}")
    else:
        print(f"  Version check PASSED: {versions}")

    print(f"  [Cross-check] 01_generate_full_beams.py:60-69 reads the SAME config.yaml entry "
          f"(hf_path={ds_cfg['hf_path']}, hf_config={ds_cfg['hf_config']}) -- confirmed identical "
          f"by inspection, not independently hardcoded elsewhere.")
    samples, raw_split_len = load_dataset_samples(ds_cfg)
    print(f"[Step 2] {ds_cfg['name']}: {len(samples)} prompts (full validation split, no subsampling; "
          f"raw split length {raw_split_len})")
    print_resource_estimate(ds_cfg["name"], len(samples))

    rouge = evaluate.load("rouge")
    bleurt = evaluate.load("bleurt", config_name=cfg["judge"]["bleurt_model"])

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                  device_map=device, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    eos_ids_set = compute_eos_ids(tokenizer)
    print(f"  Stop-token set: {len(eos_ids_set)} ids")

    all_sequences, all_prompt_lens, all_decoded_text = [], [], []
    all_flags, all_is_known, all_prompt_idx, all_gen_seeds = [], [], [], []
    t0 = time.time()

    bar = tqdm(samples, desc=f"[{ds_cfg['name']}] generation", unit="prompt")
    for sample in bar:
        pid = sample["prompt_id"]
        seed_i = prompt_seed(global_seed, pid)
        torch.manual_seed(seed_i); torch.cuda.manual_seed_all(seed_i)

        inputs = tokenizer(sample["prompt_text"], return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=gen_cfg["max_new_tokens"], eos_token_id=list(eos_ids_set),
                do_sample=gen_cfg["do_sample"], temperature=gen_cfg["temperature"],
                top_k=gen_cfg["top_k"], top_p=gen_cfg["top_p"], num_beams=gen_cfg["num_beams"],
                num_return_sequences=gen_cfg["num_return_sequences"], return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id, early_stopping=True)

        gen_ids_full = outputs.sequences[:, prompt_len:]
        any_correct = False
        for b in range(gen_ids_full.shape[0]):
            raw_comp_ids = gen_ids_full[b].tolist()
            canon_len = find_canonical_length(raw_comp_ids, eos_ids_set)
            comp_ids = raw_comp_ids[:canon_len]
            full_ids = inputs.input_ids[0].tolist() + comp_ids
            gen_text = tokenizer.decode(comp_ids, skip_special_tokens=True).strip()

            is_correct = is_correct_simple(gen_text, sample["correct_answers"],
                                            sample["incorrect_answers"], rouge, bleurt,
                                            cfg["judge"]["rouge_threshold"], cfg["judge"]["sen_sim_threshold"])
            if is_correct:
                any_correct = True

            all_sequences.append(torch.tensor(full_ids, dtype=torch.long))
            all_prompt_lens.append(prompt_len)
            all_decoded_text.append(gen_text)
            all_flags.append(not is_correct)
            all_prompt_idx.append(pid)
            all_gen_seeds.append(seed_i)

        all_is_known.append(any_correct)
        bar.set_postfix_str(f"{sum(all_flags)}/{len(all_flags)} hallucinated so far")
        del outputs
        torch.cuda.empty_cache(); gc.collect()
    bar.close()

    n_beams = len(all_decoded_text)
    n_empty = sum(1 for t in all_decoded_text if not t)
    print(f"\n[Step 2] Generation complete: {len(samples)} prompts, {n_beams} beams "
          f"({time.time()-t0:.0f}s total). Empty completions: {n_empty} ({n_empty/n_beams*100:.2f}%)")

    decoding_config = {"do_sample": gen_cfg["do_sample"], "num_beams": gen_cfg["num_beams"],
                        "temperature": gen_cfg["temperature"], "top_p": gen_cfg["top_p"],
                        "top_k": gen_cfg["top_k"], "max_new_tokens": gen_cfg["max_new_tokens"],
                        "global_seed": global_seed, "prompt_seed_scheme": "sha256(global_seed:prompt_id)",
                        "chat_template": None, "prompt_template": ds_cfg["prompt_template"],
                        "rouge_threshold": cfg["judge"]["rouge_threshold"],
                        "sen_sim_threshold": cfg["judge"]["sen_sim_threshold"],
                        "bleurt_model": cfg["judge"]["bleurt_model"],
                        **versions}

    seq_path = os.path.join(out_dir, f"{ds_cfg['name']}_sequences_v1.pt")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"input_ids": all_sequences, "prompt_len": all_prompt_lens, "prompt_id": all_prompt_idx,
                "decoded_text": all_decoded_text, "all_hallucination_flag": all_flags,
                "all_is_known": all_is_known, "gen_seeds": all_gen_seeds,
                "decoding_config": decoding_config, "git_commit": band_mod.git_commit_hash()}, seq_path)
    print(f"Saved: {seq_path}")

    # -- assert D (determinism), inline, model still loaded --
    print(f"\n[Assert D] Regenerating {n_determinism_check} random prompts to check determinism ...")
    rng = random.Random(global_seed)
    check_prompts = rng.sample(samples, min(n_determinism_check, len(samples)))
    d_results = []
    for sample in check_prompts:
        pid = sample["prompt_id"]
        seed_i = prompt_seed(global_seed, pid)
        torch.manual_seed(seed_i); torch.cuda.manual_seed_all(seed_i)
        inputs = tokenizer(sample["prompt_text"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs2 = model.generate(
                **inputs, max_new_tokens=gen_cfg["max_new_tokens"], eos_token_id=list(eos_ids_set),
                do_sample=gen_cfg["do_sample"], temperature=gen_cfg["temperature"],
                top_k=gen_cfg["top_k"], top_p=gen_cfg["top_p"], num_beams=gen_cfg["num_beams"],
                num_return_sequences=gen_cfg["num_return_sequences"], return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id, early_stopping=True)
        first_idx = next(i for i, p in enumerate(all_prompt_idx) if p == pid)
        pinned_seq = all_sequences[first_idx]
        fresh_seq = outputs2.sequences[0].cpu()
        match = (len(pinned_seq) <= len(fresh_seq)) and torch.equal(pinned_seq, fresh_seq[:len(pinned_seq)])
        d_results.append({"prompt_id": pid, "match": bool(match)})
        print(f"  prompt {pid}: {'PASS' if match else 'FAIL'}")
    d_pass = all(r["match"] for r in d_results)
    print(f"[Assert D] {'PASS' if d_pass else 'FAIL'} ({sum(r['match'] for r in d_results)}/{len(d_results)})")

    del model
    torch.cuda.empty_cache()

    return {"seq_path": seq_path, "n_prompts": len(samples), "n_beams": n_beams,
            "n_empty": n_empty, "decoding_config": decoding_config, "assert_d": d_results,
            "assert_d_pass": d_pass, "raw_split_len": raw_split_len}


# ==============================================================================
# MANIFEST
# ==============================================================================

def build_manifest(gen_result, ds_cfg, out_dir):
    seq_data = torch.load(gen_result["seq_path"], weights_only=False)
    labels = np.asarray(seq_data["all_hallucination_flag"], dtype=np.int64)
    labels_hash = hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()
    text_hash = hashlib.sha256("".join(seq_data["decoded_text"]).encode()).hexdigest()
    ids_hash = hashlib.sha256(b"".join(t.numpy().tobytes() for t in seq_data["input_ids"])).hexdigest()

    manifest = {
        "dataset": ds_cfg["name"], "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": ds_cfg["hf_path"], "subset": ds_cfg["hf_config"], "split": "validation",
        "split_length": gen_result["raw_split_len"],
        "sampling": "full validation split",
        "sampling_note": ("HARP does not report per-dataset sample counts; we evaluate on the "
                           "full standard validation split of each dataset."),
        "n_prompts": gen_result["n_prompts"], "n_beams": gen_result["n_beams"],
        "n_empty_completions": gen_result["n_empty"],
        "sequences_path": os.path.abspath(gen_result["seq_path"]),
        "ids_sha256": ids_hash, "text_sha256": text_hash, "labels_sha256": labels_hash,
        "decoding_config": gen_result["decoding_config"],
        "git_commit": seq_data.get("git_commit"),
        "assert_d_determinism_pass": gen_result["assert_d_pass"],
        "assert_d_results": gen_result["assert_d"],
    }
    manifest_path = os.path.join(out_dir, f"manifest_{ds_cfg['name']}_v1.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest, manifest_path


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: prompt seeding, split-length assert, manifest building (no model/GPU/network)")
    print("=" * 70)

    s1 = prompt_seed(0, 42)
    s2 = prompt_seed(0, 42)
    s3 = prompt_seed(0, 43)
    s4 = prompt_seed(1, 42)
    assert s1 == s2, "prompt_seed must be deterministic for identical inputs"
    assert s1 != s3, "different prompt_id should (almost certainly) give a different seed"
    assert s1 != s4, "different global_seed should (almost certainly) give a different seed"
    assert 0 <= s1 < 2 ** 31
    print(f"  [PASS] prompt_seed deterministic and seed/prompt-sensitive: {s1}, {s3}, {s4}")

    assert dedup_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert dedup_preserve_order([]) == []
    print("  [PASS] dedup_preserve_order: order-preserving, duplicates removed")

    # regression test for the real TyDiQA-GP bug found on the first cluster run: the language
    # filter must actually filter (the old ex.get("language","english") approach was a silent
    # no-op since that key doesn't exist in the real schema).
    assert extract_tydiqa_language("arabic-2387335860751143628-1") == "arabic"
    assert extract_tydiqa_language("english-12345-1") == "english"
    assert extract_tydiqa_language("korean-999-3") == "korean"
    print("  [PASS] extract_tydiqa_language: correctly parses language from the real id format "
          "(regression test for the real TyDiQA-GP no-op-filter bug)")

    assert_split_length("triviaqa", 17944)   # must not raise
    try:
        assert_split_length("triviaqa", 12345)
        raise AssertionError("assert_split_length should have raised on a length mismatch")
    except AssertionError as e:
        assert "17944" in str(e) or "12345" in str(e)
    assert_split_length("nq_open", 3610)
    assert_split_length("tydiqa_gp", 999)   # no fixed expected value -- must not raise (info-only)
    print("  [PASS] assert_split_length: raises on mismatch for pinned datasets, "
          "info-only (no raise) for datasets with no fixed expected length")

    # -- fabricate a minimal generation result and build a manifest from it --
    tmp_dir = os.path.join(HERE, "results", "_selftest_gen_dataset")
    os.makedirs(tmp_dir, exist_ok=True)
    n_beams = 20
    seq_data = {
        "input_ids": [torch.arange(5 + (i % 4)) for i in range(n_beams)],
        "prompt_len": [5] * n_beams,
        "prompt_id": [i // 10 for i in range(n_beams)],
        "decoded_text": [f"fake answer {i}" for i in range(n_beams)],
        "all_hallucination_flag": [bool(i % 3 == 0) for i in range(n_beams)],
        "all_is_known": [True, False],
        "gen_seeds": [prompt_seed(0, i // 10) for i in range(n_beams)],
        "decoding_config": {"do_sample": True, "transformers": EXPECTED_VERSIONS["transformers"]},
        "git_commit": "selftest",
    }
    seq_path = os.path.join(tmp_dir, "fake_sequences_v1.pt")
    torch.save(seq_data, seq_path)

    gen_result = {"seq_path": seq_path, "n_prompts": 2, "n_beams": n_beams, "n_empty": 0,
                  "decoding_config": seq_data["decoding_config"], "assert_d": [{"prompt_id": 0, "match": True}],
                  "assert_d_pass": True, "raw_split_len": 17944}
    manifest, manifest_path = build_manifest(gen_result, {"name": "faketest", "hf_path": "fake/path",
                                                            "hf_config": "fake_config"}, tmp_dir)
    assert os.path.exists(manifest_path)
    assert manifest["n_beams"] == n_beams
    assert manifest["split_length"] == 17944
    assert manifest["sampling"] == "full validation split"
    assert "HARP does not report" in manifest["sampling_note"]
    assert manifest["repo"] == "fake/path" and manifest["subset"] == "fake_config"
    with open(manifest_path) as f:
        reloaded = json.load(f)
    assert reloaded["labels_sha256"] == manifest["labels_sha256"]
    print(f"  [PASS] build_manifest: wrote and reloaded {manifest_path}, hashes consistent, "
          f"repo/subset/split_length/sampling/sampling_note fields present")

    est_short, est_long, est_disk = print_resource_estimate("faketest", 1000)
    assert est_short < est_long, "short-answer estimate should be less than the long-answer estimate"
    assert est_disk > 0
    print(f"  [PASS] print_resource_estimate: short={est_short:.0f}s < long={est_long:.0f}s, disk={est_disk:.0f}MB")

    mismatched_versions, mismatches = check_versions(force=True)
    # this environment almost certainly does NOT match EXPECTED_VERSIONS -- confirms the check fires
    print(f"  [INFO] check_versions on this (local, non-cluster) environment: "
          f"{len(mismatches)} mismatch(es) detected (expected, since this isn't the cluster) -- "
          f"force=True allowed it through without raising")
    try:
        check_versions(force=False)
        if mismatches:
            raise AssertionError("check_versions(force=False) should have raised given mismatches exist")
    except RuntimeError:
        print("  [PASS] check_versions(force=False) correctly raises on a version mismatch")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["triviaqa", "nq_open", "tydiqa_gp"], default=None)
    parser.add_argument("--model_folder", type=str, default="llama-3.1-8b-instruct")
    parser.add_argument("--global-seed", type=int, default=GEN_SEED_DEFAULT)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--force-version-mismatch", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.dataset:
        print("ERROR: --dataset required."); sys.exit(1)

    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    ds_cfg = next(d for d in cfg["datasets"] if d["name"] == args.dataset)

    if args.audit_only:
        run_audit(ds_cfg)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for generation (only --audit-only is CPU-only).")

    out_dir = args.output_dir or os.path.join(cfg["output"]["data_dir"], args.model_folder)
    gen_result = run_generation(ds_cfg, args.model_folder, args.global_seed, out_dir,
                                 force_versions=args.force_version_mismatch)
    manifest, manifest_path = build_manifest(gen_result, ds_cfg, out_dir)
    print(f"\nWrote manifest: {manifest_path}")
    print(f"Next: run 40_validate_dataset.py --manifest {manifest_path}")


if __name__ == "__main__":
    main()
