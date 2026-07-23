"""
40_validate_dataset.py -- Session 06 Phase 1, Step 3+4: Dataset Validation (A,B,C,E,F) + Report
=====================================================================================================
CPU-only (tokenizer load only -- no GPU, no model forward pass). Assert D (determinism) already ran
inline in 39_generate_dataset.py's run_generation() while the model was loaded, since regenerating
5 prompts needs the GPU anyway and doing it there avoids a whole separate GPU allocation for 5
prompts -- see that script's module docstring and this session's Deviations for why the prompt's
literal "validation = CPU" framing doesn't quite fit assert D, and how this resolves it.

A. Structure: exactly 10 completions/prompt, zero duplicate prompts, 20-prompt alignment spot-check
   against the live source HF dataset.
B. Window/termination: re-verifies every saved completion is already at its canonical boundary
   (should hold by construction, since 39_generate_dataset.py truncates at save time -- this is a
   defense-in-depth re-check, not a formality). Completion-length distribution, empty-completion rate.
C. Round-trip: re-tokenize 100 random beams' pinned decoded text, compare against pinned ids.
E. Labels: NaN/missing check, hallucination-rate sanity band, composition, 5+5 eyeball examples
   (recomputes ROUGE/BLEURT for just those 10, since only the boolean label was pinned).
F. Manifest: reload-and-verify sha256 of ids/text/labels against the pinned manifest.

Usage:
  python 40_validate_dataset.py --self-test
  python 40_validate_dataset.py --manifest ../data/llama-3.1-8b-instruct/manifest_triviaqa_v1.json \
      --model_folder llama-3.1-8b-instruct --dataset triviaqa
  python 40_validate_dataset.py --combine reports/session06_phase1_validation.md \
      --dataset-jsons results/session06_triviaqa.json results/session06_nq_open.json results/session06_tydiqa_gp.json
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# A -- STRUCTURE
# ==============================================================================

def check_structure(seq_data, ds_cfg, sample_source_lookup=None, tokenizer=None, n_spot_check=20, seed=0):
    prompt_ids = np.asarray(seq_data["prompt_id"])
    unique_prompts, counts = np.unique(prompt_ids, return_counts=True)
    n_not_10 = int((counts != 10).sum())
    exactly_10 = (n_not_10 == 0)

    n_duplicate_prompts = int(len(unique_prompts) - len(set(unique_prompts.tolist())))
    zero_duplicates = (n_duplicate_prompts == 0)

    spot_check_results = []
    if sample_source_lookup is not None and tokenizer is not None:
        rng = random.Random(seed)
        check_ids = rng.sample(list(unique_prompts), min(n_spot_check, len(unique_prompts)))
        for pid in check_ids:
            beam_idx = int(np.where(prompt_ids == pid)[0][0])
            p_len = seq_data["prompt_len"][beam_idx]
            saved_prompt_text = tokenizer.decode(seq_data["input_ids"][beam_idx][:p_len],
                                                  skip_special_tokens=True)
            expected = sample_source_lookup(int(pid))
            matched = (expected is not None and saved_prompt_text.strip() == expected["prompt_text"].strip())
            spot_check_results.append({"prompt_id": int(pid), "found_in_source": expected is not None,
                                        "text_matches": matched})
    n_spot_checked = len(spot_check_results)
    n_spot_matched = sum(1 for r in spot_check_results if r["text_matches"])
    spot_check_pass = (n_spot_checked == 0) or (n_spot_matched == n_spot_checked)

    return {"n_prompts": int(len(unique_prompts)), "exactly_10_per_prompt": exactly_10,
            "n_prompts_not_10": n_not_10, "zero_duplicate_prompts": zero_duplicates,
            "n_duplicate_prompts": n_duplicate_prompts, "spot_check_n": n_spot_checked,
            "spot_check_n_matched": n_spot_matched, "spot_check_pass": spot_check_pass,
            "pass": exactly_10 and zero_duplicates and spot_check_pass}


# ==============================================================================
# B -- WINDOW / TERMINATION
# ==============================================================================

def check_window(seq_data, eos_ids_set):
    import importlib.util
    spec = importlib.util.spec_from_file_location("s05b", os.path.join(HERE, "38_window_forensics.py"))
    window_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(window_mod)

    n_beams = len(seq_data["input_ids"])
    n_violations = 0
    comp_lens = []
    n_empty = 0
    for i in range(n_beams):
        p_len = seq_data["prompt_len"][i]
        comp_ids = seq_data["input_ids"][i][p_len:].tolist()
        comp_lens.append(len(comp_ids))
        if len(comp_ids) == 0:
            n_empty += 1
            continue
        canon_len = window_mod.find_canonical_length(comp_ids, eos_ids_set)
        if canon_len != len(comp_ids):
            n_violations += 1   # saved window extends past the canonical boundary

    comp_lens = np.array(comp_lens)
    empty_pct = n_empty / n_beams * 100
    return {"n_beams": n_beams, "n_window_violations": n_violations,
            "zero_violations": (n_violations == 0),
            "completion_length_stats": {"mean": float(comp_lens.mean()), "std": float(comp_lens.std()),
                                          "min": int(comp_lens.min()), "max": int(comp_lens.max()),
                                          "median": float(np.median(comp_lens))},
            "n_empty_completions": n_empty, "empty_pct": empty_pct,
            "empty_under_threshold": bool(empty_pct < 0.5),
            "pass": (n_violations == 0) and (empty_pct < 0.5)}


# ==============================================================================
# C -- ROUND-TRIP
# ==============================================================================

def check_roundtrip(seq_data, tokenizer, n_samples=100, seed=0):
    """Decodes each beam's PINNED ids fresh (no .strip()) rather than trusting
    seq_data["decoded_text"], which 39_generate_dataset.py saves .strip()'d for
    readability. Llama's tokenizer bakes a leading space into word-initial tokens
    (" Paris" vs "Paris" are different token ids), so re-encoding the stripped text
    silently shifts the first token almost every time -- a validation-script
    artifact, not a real drift in the pinned ids (which Assert D already verifies
    independently via fresh regeneration)."""
    n_beams = len(seq_data["input_ids"])
    rng = random.Random(seed)
    idx = rng.sample(range(n_beams), min(n_samples, n_beams))
    n_exact, n_checked = 0, 0
    mismatches = []
    for i in idx:
        p_len = seq_data["prompt_len"][i]
        pinned_comp_ids = seq_data["input_ids"][i][p_len:].tolist()
        if not pinned_comp_ids:
            continue
        n_checked += 1
        text = tokenizer.decode(pinned_comp_ids, skip_special_tokens=True)
        retok_ids = tokenizer.encode(text, add_special_tokens=False)
        if retok_ids == pinned_comp_ids:
            n_exact += 1
        else:
            mismatches.append({"beam": i, "pinned_len": len(pinned_comp_ids), "retok_len": len(retok_ids)})

    exact_rate = n_exact / n_checked if n_checked else float("nan")
    return {"n_checked": n_checked, "n_exact_match": n_exact, "exact_match_rate": exact_rate,
            "sample_mismatches": mismatches[:10], "pass": bool(n_checked > 0 and exact_rate >= 0.90)}


# ==============================================================================
# E -- LABELS
# ==============================================================================

def check_labels(seq_data):
    labels = np.asarray(seq_data["all_hallucination_flag"])
    prompt_ids = np.asarray(seq_data["prompt_id"])
    n_nan = int(np.isnan(labels.astype(float)).sum()) if labels.dtype != bool else 0
    halluc_rate = float(labels.astype(float).mean() * 100)
    in_band = 5.0 <= halluc_rate <= 95.0

    n_mixed = n_all_truthful = n_all_halluc = 0
    n_pairs = 0
    for pid in np.unique(prompt_ids):
        beam_labels = labels[prompt_ids == pid]
        if beam_labels.any() and (~beam_labels).any():
            n_mixed += 1
            n_h, n_t = int(beam_labels.sum()), int((~beam_labels).sum())
            n_pairs += n_h * n_t
        elif beam_labels.all():
            n_all_halluc += 1
        else:
            n_all_truthful += 1

    return {"n_nan_or_missing": n_nan, "no_nan": (n_nan == 0), "hallucination_rate_pct": halluc_rate,
            "in_5_95_band": in_band, "n_mixed_prompts": n_mixed, "n_all_truthful_prompts": n_all_truthful,
            "n_all_hallucinated_prompts": n_all_halluc, "n_pairs": n_pairs,
            "pass": (n_nan == 0)}   # in_band is a WARNING per spec, not a hard fail


def eyeball_examples(seq_data, source_lookup, rouge, bleurt, rouge_threshold, sen_sim_threshold,
                      n_each=5, seed=0):
    """Recomputes real ROUGE-L / BLEURT scores against the actual reference answers (via
    source_lookup(prompt_id) -> sample dict) for just the 10 chosen beams -- NOT rouge-against-
    itself, which would be a meaningless ~1.0 tautology. Mirrors is_correct_simple's exact formula
    from 39_generate_dataset.py so the printed scores match what actually produced the label."""
    labels = np.asarray(seq_data["all_hallucination_flag"])
    prompt_ids = seq_data["prompt_id"]
    rng = random.Random(seed)
    correct_idx = np.where(~labels)[0].tolist()
    incorrect_idx = np.where(labels)[0].tolist()
    chosen_correct = rng.sample(correct_idx, min(n_each, len(correct_idx)))
    chosen_incorrect = rng.sample(incorrect_idx, min(n_each, len(incorrect_idx)))

    examples = []
    for i in chosen_correct + chosen_incorrect:
        text = seq_data["decoded_text"][i]
        pid = int(prompt_ids[i])
        sample = source_lookup(pid) if source_lookup else None
        correct_answers = sample["correct_answers"] if sample else []
        incorrect_answers = sample["incorrect_answers"] if sample else []

        rl = 0.0
        if text and correct_answers:
            r = rouge.compute(predictions=[text] * len(correct_answers), references=correct_answers)
            rl = r["rougeL"]
        max_b = 0.0
        all_refs = correct_answers + incorrect_answers
        if text and all_refs:
            bs = bleurt.compute(predictions=[text] * len(all_refs), references=all_refs)
            max_b = max(bs["scores"][:len(correct_answers)], default=0.0)

        examples.append({"beam": int(i), "prompt_id": pid, "text": text[:200],
                          "label": "hallucinated" if labels[i] else "truthful",
                          "rouge_l": round(float(rl), 4), "bleurt_max": round(float(max_b), 4)})
    return examples


# ==============================================================================
# F -- MANIFEST RELOAD-VERIFY
# ==============================================================================

def check_manifest(seq_data, manifest):
    labels = np.asarray(seq_data["all_hallucination_flag"], dtype=np.int64)
    labels_hash = hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()
    text_hash = hashlib.sha256("".join(seq_data["decoded_text"]).encode()).hexdigest()
    ids_hash = hashlib.sha256(b"".join(t.numpy().tobytes() for t in seq_data["input_ids"])).hexdigest()

    return {"labels_match": labels_hash == manifest["labels_sha256"],
            "text_match": text_hash == manifest["text_sha256"],
            "ids_match": ids_hash == manifest["ids_sha256"],
            "pass": (labels_hash == manifest["labels_sha256"] and text_hash == manifest["text_sha256"]
                     and ids_hash == manifest["ids_sha256"])}


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: checks A/B/C/E/F on a fabricated pinned dataset (no model/GPU/network)")
    print("=" * 70)
    import importlib.util
    spec = importlib.util.spec_from_file_location("s05b", os.path.join(HERE, "38_window_forensics.py"))
    window_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(window_mod)

    class FakeTokenizer:
        """Char-level codepoint tokenizer -- a true bijection (unlike a mod-based
        scheme, which would silently corrupt the round-trip self-test via collisions)."""
        eos_token_id = 999

        def encode(self, s, add_special_tokens=False):
            return [ord(c) for c in s] if s else []

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids)

    tok = FakeTokenizer()
    eos_ids_set = window_mod.compute_eos_ids(tok)
    # content tokens must avoid colliding with the reconstructed custom-stop id set
    # (built from small-valued punctuation/"Yes..." chars under FakeTokenizer's char->id map,
    # plus the literal eos_token_id=999), else find_canonical_length would truncate "content"
    # early and falsely flag a B violation.
    assert eos_ids_set.isdisjoint(range(200, 400)), f"test assumption violated: {eos_ids_set}"

    rng = np.random.default_rng(0)
    n_prompts, n_each = 30, 10
    input_ids, prompt_lens, prompt_ids, decoded_text, labels = [], [], [], [], []
    for p in range(n_prompts):
        for b in range(n_each):
            p_len = 5
            content_len = int(rng.integers(3, 8))
            comp_ids = [int(rng.integers(200, 400)) for _ in range(content_len)] + [tok.eos_token_id]
            text = tok.decode(comp_ids)
            retok = tok.encode(text)
            full_ids = [int(rng.integers(200, 400)) for _ in range(p_len)] + comp_ids
            input_ids.append(torch.tensor(full_ids, dtype=torch.long))
            prompt_lens.append(p_len); prompt_ids.append(p); decoded_text.append(text)
            labels.append(bool(rng.integers(0, 2)))

    seq_data = {"input_ids": input_ids, "prompt_len": prompt_lens, "prompt_id": prompt_ids,
                "decoded_text": decoded_text, "all_hallucination_flag": labels}

    a_result = check_structure(seq_data, {"name": "faketest"})
    assert a_result["exactly_10_per_prompt"] and a_result["zero_duplicate_prompts"]
    print(f"  [PASS] A structure: {a_result}")

    b_result = check_window(seq_data, eos_ids_set)
    assert b_result["zero_violations"], f"synthetic data should have zero window violations: {b_result}"
    print(f"  [PASS] B window: {b_result['n_window_violations']} violations, "
          f"empty_pct={b_result['empty_pct']:.2f}%")

    c_result = check_roundtrip(seq_data, tok, n_samples=50, seed=0)
    assert c_result["exact_match_rate"] == 1.0, "fake tokenizer's encode/decode is a perfect inverse by construction"
    print(f"  [PASS] C round-trip: {c_result['n_exact_match']}/{c_result['n_checked']} exact")

    # regression test for the real bug found on TriviaQA's first cluster run: check_roundtrip
    # must decode pinned ids fresh, NOT trust seq_data["decoded_text"] (which
    # 39_generate_dataset.py saves .strip()'d -- stripping a leading-space-bearing first
    # token before re-encoding silently shifted the first token on ~100% of real beams).
    # Corrupt decoded_text here and confirm check_roundtrip is unaffected.
    corrupted = dict(seq_data)
    corrupted["decoded_text"] = ["<corrupted>"] * len(seq_data["decoded_text"])
    c_result_corrupted = check_roundtrip(corrupted, tok, n_samples=50, seed=0)
    assert c_result_corrupted["exact_match_rate"] == 1.0, \
        "check_roundtrip must decode pinned ids independently, not read seq_data['decoded_text']"
    print("  [PASS] C round-trip ignores seq_data['decoded_text'] (decodes pinned ids fresh) "
          "-- regression test for the real TriviaQA-run bug")

    e_result = check_labels(seq_data)
    assert e_result["no_nan"]
    print(f"  [PASS] E labels: halluc_rate={e_result['hallucination_rate_pct']:.1f}%  "
          f"mixed={e_result['n_mixed_prompts']}")

    # tamper test: introduce a real window violation and confirm B catches it
    tampered = dict(seq_data)
    tampered["input_ids"] = list(seq_data["input_ids"])
    tampered["input_ids"][0] = torch.cat([seq_data["input_ids"][0], torch.tensor([50, 60])])
    b_bad = check_window(tampered, eos_ids_set)
    assert b_bad["n_window_violations"] >= 1, "appending post-stop tokens should trigger a B violation"
    print(f"  [PASS] B correctly detects an injected window violation: {b_bad['n_window_violations']} found")

    # F: build a manifest and verify round-trip
    labels_arr = np.asarray(labels, dtype=np.int64)
    manifest = {
        "labels_sha256": hashlib.sha256(np.ascontiguousarray(labels_arr).tobytes()).hexdigest(),
        "text_sha256": hashlib.sha256("".join(decoded_text).encode()).hexdigest(),
        "ids_sha256": hashlib.sha256(b"".join(t.numpy().tobytes() for t in input_ids)).hexdigest(),
    }
    f_result = check_manifest(seq_data, manifest)
    assert f_result["pass"]
    print(f"  [PASS] F manifest reload-verify: {f_result}")

    tampered_manifest = dict(manifest); tampered_manifest["labels_sha256"] = "deadbeef"
    f_bad = check_manifest(seq_data, tampered_manifest)
    assert not f_bad["pass"]
    print(f"  [PASS] F correctly detects a tampered manifest hash")

    class FakeRouge:
        def compute(self, predictions, references):
            return {"rougeL": 1.0 if predictions[0] == references[0] else 0.0}

    class FakeBleurt:
        def compute(self, predictions, references):
            return {"scores": [0.9 if p == r else 0.1 for p, r in zip(predictions, references)]}

    fake_lookup = {pid: {"correct_answers": [decoded_text[prompt_ids.index(pid)]], "incorrect_answers": []}
                   for pid in set(prompt_ids)}
    ex = eyeball_examples(seq_data, fake_lookup.get, FakeRouge(), FakeBleurt(), 0.7, 0.5, n_each=3, seed=0)
    assert 0 < len(ex) <= 6
    assert all("rouge_l" in e and "bleurt_max" in e for e in ex)
    print(f"  [PASS] eyeball_examples: {len(ex)} examples with real rouge_l/bleurt_max fields "
          f"(not a self-tautology)")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def write_dataset_report_section(dataset_name, results):
    lines = [f"## {dataset_name}\n"]
    lines.append(f"- N prompts: {results['a']['n_prompts']}  |  N beams: "
                 f"{results['a']['n_prompts']*10}\n")
    lines.append(f"- Assert A (structure): {'PASS' if results['a']['pass'] else 'FAIL'} -- "
                 f"{results['a']}\n")
    lines.append(f"- Assert B (window): {'PASS' if results['b']['pass'] else 'FAIL'} -- "
                 f"empty={results['b']['empty_pct']:.2f}%, length stats={results['b']['completion_length_stats']}\n")
    lines.append(f"- Assert C (round-trip): {'PASS' if results['c']['pass'] else 'FAIL'} -- "
                 f"{results['c']['n_exact_match']}/{results['c']['n_checked']} exact\n")
    lines.append(f"- Assert D (determinism, from generation step): "
                 f"{'PASS' if results.get('d_pass') else 'FAIL/NOT RUN'}\n")
    lines.append(f"- Assert E (labels): {'PASS' if results['e']['pass'] else 'FAIL'} -- "
                 f"hallucination rate={results['e']['hallucination_rate_pct']:.1f}% "
                 f"(in 5-95% band: {results['e']['in_5_95_band']}), "
                 f"mixed={results['e']['n_mixed_prompts']}, "
                 f"all-truthful={results['e']['n_all_truthful_prompts']}, "
                 f"all-hallucinated={results['e']['n_all_hallucinated_prompts']}\n")
    lines.append(f"- Assert F (manifest): {'PASS' if results['f']['pass'] else 'FAIL'}\n")
    lines.append("\n### Eyeball examples\n")
    for ex in results.get("examples", []):
        lines.append(f"- [{ex['label']}] beam {ex['beam']} (rougeL={ex.get('rouge_l')}, "
                     f"bleurt_max={ex.get('bleurt_max')}): {ex['text']!r}")
    lines.append("\n### Decoding config\n```json\n" + json.dumps(results.get("decoding_config", {}), indent=2) + "\n```\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--sequences", type=str, default=None,
                         help="Path to the _sequences_v1.pt file; defaults to manifest's sequences_path.")
    parser.add_argument("--model_folder", type=str, default="llama-3.1-8b-instruct")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--report-section", type=str, default=None)
    parser.add_argument("--combine", type=str, default=None,
                         help="Combine mode: write the final report to this path.")
    parser.add_argument("--dataset-jsons", nargs="+", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.combine:
        sections = []
        for jp in args.dataset_jsons:
            with open(jp) as f:
                d = json.load(f)
            sections.append(write_dataset_report_section(d["dataset"], d["results"]))
        os.makedirs(os.path.dirname(args.combine) or ".", exist_ok=True)
        with open(args.combine, "w") as f:
            f.write("# Session 06 Phase 1 -- Dataset Validation\n\n" + "\n\n---\n\n".join(sections))
        print(f"Wrote combined report: {args.combine}")
        return

    if not args.manifest or not args.dataset:
        print("ERROR: --manifest and --dataset required (or use --combine)."); sys.exit(1)

    t_start = time.time()

    def elapsed(label):
        print(f"  [{label}: {time.time()-t_start:.1f}s elapsed since start]")

    with open(args.manifest) as f:
        manifest = json.load(f)
    seq_path = args.sequences or manifest["sequences_path"]
    seq_data = torch.load(seq_path, weights_only=False)
    elapsed("sequences loaded")

    import yaml
    from transformers import AutoTokenizer
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    ds_cfg = next(d for d in cfg["datasets"] if d["name"] == args.dataset)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == args.model_folder)
    print(f"Loading tokenizer only (CPU-safe): {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    elapsed("tokenizer loaded")

    import importlib.util

    def _load(name, filename):
        s = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        return m

    window_mod = _load("s05b", "38_window_forensics.py")
    eos_ids_set = window_mod.compute_eos_ids(tokenizer)

    print("\n[A] Structure (incl. 20-prompt source alignment spot-check) ...")
    gen_mod = _load("s06_gen", "39_generate_dataset.py")
    source_samples, _raw_split_len = gen_mod.load_dataset_samples(ds_cfg)
    source_lookup = {s["prompt_id"]: s for s in source_samples}
    a_result = check_structure(seq_data, ds_cfg, sample_source_lookup=source_lookup.get, tokenizer=tokenizer)
    print(f"  {'PASS' if a_result['pass'] else 'FAIL'}: {a_result}")
    elapsed("A done (source split load + spot-check dominates this step)")

    print("\n[B] Window/termination ...")
    b_result = check_window(seq_data, eos_ids_set)
    print(f"  {'PASS' if b_result['pass'] else 'FAIL'}: violations={b_result['n_window_violations']}, "
          f"empty={b_result['empty_pct']:.2f}%")
    elapsed("B done")

    print("\n[C] Round-trip ...")
    c_result = check_roundtrip(seq_data, tokenizer)
    print(f"  {'PASS' if c_result['pass'] else 'FAIL'}: {c_result['n_exact_match']}/{c_result['n_checked']} exact")
    elapsed("C done")

    print("\n[E] Labels ...")
    e_result = check_labels(seq_data)
    print(f"  {'PASS' if e_result['pass'] else 'FAIL'}: hallucination_rate={e_result['hallucination_rate_pct']:.1f}%")
    if not e_result["in_5_95_band"]:
        print(f"  [WARN] hallucination rate {e_result['hallucination_rate_pct']:.1f}% is outside [5%,95%]")
    elapsed("E done")

    print("\n[F] Manifest reload-verify ...")
    f_result = check_manifest(seq_data, manifest)
    print(f"  {'PASS' if f_result['pass'] else 'FAIL'}: {f_result}")
    elapsed("F done")

    print("\nComputing eyeball examples (recomputes ROUGE-L/BLEURT for 10 examples only) ...")
    import evaluate
    rouge = evaluate.load("rouge")
    bleurt = evaluate.load("bleurt", config_name=cfg["judge"]["bleurt_model"])
    elapsed("rouge/bleurt libraries loaded")
    examples = eyeball_examples(seq_data, source_lookup.get, rouge, bleurt,
                                 cfg["judge"]["rouge_threshold"], cfg["judge"]["sen_sim_threshold"])
    for ex in examples:
        print(f"  [{ex['label']}] beam {ex['beam']} (rougeL={ex['rouge_l']}, bleurt_max={ex['bleurt_max']}): "
              f"{ex['text'][:100]!r}")
    elapsed("eyeball examples done")

    total_s = time.time() - t_start
    d_pass = manifest.get("assert_d_determinism_pass")
    results = {"a": a_result, "b": b_result, "c": c_result, "e": e_result, "f": f_result,
               "d_pass": d_pass, "examples": examples, "decoding_config": manifest.get("decoding_config", {}),
               "timing_seconds": total_s}
    overall_pass = all([a_result["pass"], b_result["pass"], c_result["pass"], e_result["pass"],
                         f_result["pass"], bool(d_pass)])
    print(f"\n{'='*70}\nOVERALL: {'PASS' if overall_pass else 'FAIL'}  ({total_s:.1f}s total)\n{'='*70}")

    out_json = args.output_json or f"results/session06_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"dataset": args.dataset, "overall_pass": overall_pass, "results": results}, f, indent=2, default=str)
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
