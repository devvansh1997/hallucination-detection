"""
34_gate_reconstruct_or_regenerate.py -- Session 04 Parts 1+2: Recovery Gate + Raw-State Store
==================================================================================================
GPU-side. Resolves the blocker from the original session04 attempt: no raw sequences were ever
pinned for the seeded dataset. This script first searches exhaustively for cached generated TEXT
(labels/judge caches require decoded strings, so text may have existed transiently even though
token ids never did), and only falls back to a fresh generation pass if explicitly authorized.

--mode auto (default):
  1a. INVENTORY -- read-only search of the data directories for anything that could be per-beam
      completion text. Prints a verdict: COMPLETE (all 8170 beams) / PARTIAL / NONE.
  1b. ROUTE R (only if COMPLETE) -- reconstructs each beam's full sequence by tokenizing the
      exact prompt template 29_generate_extract_band.py used (NOT a chat template -- that script
      calls tokenizer(prompt_text) directly on config.yaml's "Q: {question}\nA:" template, no
      apply_chat_template; verified by re-reading that script rather than assuming) plus the
      recovered completion text. Pre-verifies per-beam token counts against the pinned band
      npz's offsets (>99% match required), then re-forwards every beam and fingerprints
      recomputed band coordinates against the stored npz (per-beam mean cosine > 0.999,
      required for >99% of beams). PASS -> proceeds to Part 2 (raw-state store) using the
      verified reconstruction. FAIL/PARTIAL -> prints a full report and exits, instructing a
      re-run with --authorize-new-pass. Does NOT proceed to regeneration on its own.

--authorize-new-pass:
  1c. ROUTE N -- one comprehensive fresh generation pass (dataset v3). Pins the literal decoding
      config + library versions, saves raw sequences, prompt lengths, decoded text, and
      recomputed ROUGE/BLEURT labels this time (the gap that caused this whole detour), then
      proceeds to Part 2.

Part 2 (both routes): during the same forward pass, persists for every completion token the
post-block residual states for layers 15..23 AND the final post-norm state (10 slices total),
bf16, sharded, plus the V_R (trailing-320 unembedding SVD) and rand256 bases actually saved this
time (session02 only ever recomputed them on the fly). This is the permanent fix -- future
feature ideas derive from this store on CPU; the model is only reloaded for a genuinely new
decoding protocol.

Usage:
  python 34_gate_reconstruct_or_regenerate.py --self-test
  python 34_gate_reconstruct_or_regenerate.py --model_folder llama-3.1-8b-instruct --dataset truthfulqa --mode auto
  python 34_gate_reconstruct_or_regenerate.py --model_folder llama-3.1-8b-instruct --dataset truthfulqa --authorize-new-pass
"""

import argparse
import csv
import fnmatch
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


band_mod = _load("s02_extract", "27_extract_band.py")
pin_mod = _load("s03_pin", "30_pin_manifest.py")

W_START, W_END = 15, 24
RAW_SLICE_LAYERS = list(range(W_START, W_END)) + ["final_norm"]   # 9 mid-layers + 1 final-norm = 10
INVENTORY_PATTERNS = ["*label*", "*rouge*", "*bleurt*", "*answer*", "*gen*", "*.jsonl", "*.csv", "*.json"]
SHARD_BYTES_LIMIT = int(1.5 * 1e9)


# ==============================================================================
# 1a -- INVENTORY SEARCH
# ==============================================================================

def inventory_search(search_roots):
    """Read-only. Returns a sorted list of candidate file paths matching text-cache-like globs,
    excluding files already known to be pooled/projected artifacts (not raw text)."""
    known_non_text = {"pooled", "band", "manifest", "core", "ads_btd_features"}
    candidates = set()
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if any(fnmatch.fnmatch(fn.lower(), pat) for pat in INVENTORY_PATTERNS):
                    if any(tok in fn.lower() for tok in known_non_text) and fn.endswith(".pt"):
                        continue
                    if fn.endswith((".pt", ".npz", ".png")):
                        continue
                    candidates.add(os.path.join(dirpath, fn))
    return sorted(candidates)


def try_extract_beam_texts(path, expected_n):
    """Best-effort heuristic loader. Returns a list of length expected_n (str or None per slot)
    if the file plausibly contains per-beam completion text, else None. Never raises."""
    try:
        if path.endswith(".jsonl"):
            with open(path, encoding="utf-8", errors="replace") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        elif path.endswith(".json"):
            with open(path, encoding="utf-8", errors="replace") as f:
                obj = json.load(f)
            rows = obj if isinstance(obj, list) else [obj]
        elif path.endswith(".csv"):
            with open(path, encoding="utf-8", errors="replace", newline="") as f:
                rows = list(csv.DictReader(f))
        else:
            return None
    except Exception:
        return None

    if not isinstance(rows, list) or len(rows) == 0:
        return None

    text_field_candidates = ("generated", "generated_text", "completion", "text", "gen_text", "answer")

    def extract_text(row):
        if isinstance(row, str):
            return row
        if isinstance(row, dict):
            for k in text_field_candidates:
                if k in row and isinstance(row[k], str):
                    return row[k]
        return None

    texts = [extract_text(r) for r in rows]
    n_found = sum(1 for t in texts if t)
    if n_found == 0:
        return None
    if len(rows) == expected_n:
        return texts
    return texts   # partial -- caller judges completeness by n_found vs expected_n


def run_inventory(search_roots, dataset, expected_n_beams):
    """search_roots: caller-supplied list of directories to search -- kept explicit rather than
    hardcoded here so behavior is fully determined by the argument (and testable in isolation)."""
    candidates = inventory_search(search_roots)
    print(f"  [1a] Searched {len(search_roots)} root(s), found {len(candidates)} candidate file(s) "
          f"matching {INVENTORY_PATTERNS}")

    best = None
    for path in candidates:
        texts = try_extract_beam_texts(path, expected_n_beams)
        if texts is None:
            continue
        n_found = sum(1 for t in texts if t)
        print(f"    candidate: {path}  ({n_found}/{expected_n_beams} plausible text entries)")
        if best is None or n_found > best[1]:
            best = (path, n_found, texts)

    if best is None:
        print("  [1a] VERDICT: NONE -- no candidate file contained plausible per-beam text.")
        return "NONE", None
    path, n_found, texts = best
    if n_found >= expected_n_beams:
        print(f"  [1a] VERDICT: COMPLETE -- {path} has text for all {expected_n_beams} beams.")
        return "COMPLETE", (path, texts)
    print(f"  [1a] VERDICT: PARTIAL -- best candidate {path} covers {n_found}/{expected_n_beams} beams.")
    return "PARTIAL", (path, texts)


# ==============================================================================
# 1b -- ROUTE R: RECONSTRUCT + VERIFY
# ==============================================================================

def reconstruct_sequence(tokenizer, prompt_template, question, completion_text):
    """Matches 29_generate_extract_band.py's tokenization EXACTLY: tokenizer(prompt_text) on the
    raw "Q: {question}\nA:" template -- no chat template is used by that script (confirmed by
    reading it, not assumed)."""
    prompt_text = prompt_template.format(question=question)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
    completion_ids = tokenizer(completion_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full_ids = torch.cat([prompt_ids, completion_ids])
    return full_ids, len(prompt_ids), len(completion_ids)


def preverify_token_counts(reconstructed_lens, band_offsets):
    """reconstructed_lens: list of per-beam completion-token counts from reconstruction.
    band_offsets: the pinned band npz's offsets array. Returns match_rate in [0,1]."""
    expected_lens = np.diff(band_offsets)
    matches = sum(1 for r, e in zip(reconstructed_lens, expected_lens) if r == e)
    return matches / len(expected_lens)


def fingerprint_verify(recomputed_z_band_mean, stored_z_band_mean, threshold=0.999):
    """Per-beam mean cosine similarity between recomputed and stored band coordinates.
    Returns (pass_rate, per_beam_cosine)."""
    num = (recomputed_z_band_mean * stored_z_band_mean).sum(axis=1)
    denom = (np.linalg.norm(recomputed_z_band_mean, axis=1) *
              np.linalg.norm(stored_z_band_mean, axis=1) + 1e-12)
    cos = num / denom
    pass_rate = float((cos > threshold).mean())
    return pass_rate, cos


# ==============================================================================
# RAW-STATE STORE PACKING (shared by both routes)
# ==============================================================================

def pack_raw_state_beam(h_by_layer_and_final_norm):
    """h_by_layer_and_final_norm: dict with keys 15..23 (mid-layer post-block residual, (T,D))
    and 'final_norm' (post-final-norm, (T,D)). Returns (T, 10, D) bf16 tensor, layer order
    [15,16,...,23,final_norm]."""
    T = h_by_layer_and_final_norm[W_START].shape[0]
    D = h_by_layer_and_final_norm[W_START].shape[1]
    if T == 0:
        return torch.zeros(0, len(RAW_SLICE_LAYERS), D, dtype=torch.bfloat16)
    slices = [h_by_layer_and_final_norm[l].to(torch.bfloat16) for l in RAW_SLICE_LAYERS]
    return torch.stack(slices, dim=1)   # (T, 10, D)


def pack_and_save_raw_store(per_beam_raw, prompt_ids, beam_idxs, labels, V_R, V_rand, out_dir,
                             checkpoint_id, seed, decoding_config, git_commit):
    """per_beam_raw: list of (T_i, 10, D) bf16 tensors. Sharded npz (bf16 preserved via a
    view-as-uint16 round trip, since np.savez doesn't natively store bfloat16)."""
    n_beams = len(per_beam_raw)
    offsets = np.zeros(n_beams + 1, dtype=np.int64)
    for i, t in enumerate(per_beam_raw):
        offsets[i + 1] = offsets[i] + t.shape[0]
    total_T = int(offsets[-1])
    D = per_beam_raw[0].shape[2] if n_beams > 0 else 0

    raw_all = torch.zeros(total_T, len(RAW_SLICE_LAYERS), D, dtype=torch.bfloat16)
    for i, t in enumerate(per_beam_raw):
        s, e = offsets[i], offsets[i + 1]
        raw_all[s:e] = t
    # bf16 has no native numpy dtype -- store as raw uint16 bit-pattern, documented in metadata
    raw_u16 = raw_all.view(torch.uint16).numpy()

    os.makedirs(out_dir, exist_ok=True)
    n_shards = max(1, int(np.ceil(raw_u16.nbytes / SHARD_BYTES_LIMIT)))
    beams_per_shard = max(1, int(np.ceil(n_beams / n_shards)))
    shard_paths = []
    b0 = 0
    shard_i = 0
    while b0 < n_beams:
        b1 = min(b0 + beams_per_shard, n_beams)
        t0, t1 = offsets[b0], offsets[b1]
        shard_path = os.path.join(out_dir, f"raw_state_shard{shard_i}.npz")
        np.savez_compressed(shard_path,
                             raw_u16=raw_u16[t0:t1], offsets=offsets[b0:b1 + 1] - offsets[b0],
                             prompt_id=np.asarray(prompt_ids[b0:b1], dtype=np.int64),
                             beam_idx=np.asarray(beam_idxs[b0:b1], dtype=np.int64),
                             label=np.asarray(labels[b0:b1], dtype=np.int64))
        shard_paths.append(shard_path)
        b0 = b1; shard_i += 1

    bases_path = os.path.join(out_dir, "bases.npz")
    # defensive .detach().cpu() regardless of what device the caller's tensors are on --
    # compute_bases() returns V_band on whatever device lm_head.weight lives on (GPU, once a
    # model is loaded), so callers passing straight through without an explicit .cpu() call
    # crash here otherwise (exactly what happened on the first real run).
    np.savez_compressed(bases_path, V_R=V_R.detach().cpu().numpy().astype(np.float32),
                         V_rand=V_rand.detach().cpu().numpy().astype(np.float32))

    meta = {
        "checkpoint_id": checkpoint_id, "seed": seed, "git_commit": git_commit,
        "decoding_config": decoding_config, "n_beams": n_beams, "total_tokens": total_T,
        "layer_order": [str(l) for l in RAW_SLICE_LAYERS], "dtype": "bfloat16 (stored as uint16 bit-pattern)",
        "shards": [os.path.basename(p) for p in shard_paths], "bases_file": "bases.npz",
    }
    meta_path = os.path.join(out_dir, "raw_state_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return shard_paths, bases_path, meta_path


def load_raw_state_shard(shard_path):
    d = dict(np.load(shard_path))
    raw = torch.from_numpy(d["raw_u16"]).view(torch.bfloat16)
    return raw, d["offsets"], d["prompt_id"], d["beam_idx"], d["label"]


def git_commit_hash():
    return band_mod.git_commit_hash()


# ==============================================================================
# ROUTE N -- comprehensive fresh generation (extends 29_generate_extract_band.py's pattern:
# THIS TIME saves raw sequences / prompt lengths / decoded text / labels, the gap that caused
# this whole detour)
# ==============================================================================

def library_versions():
    import transformers
    return {"torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "transformers": transformers.__version__}


def run_route_n(model_folder, dataset, gen_seed, out_dir, batch_size=8):
    import random
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import evaluate

    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)
    ds_cfg = next(d for d in cfg["datasets"] if d["name"] == dataset)
    gen_cfg = cfg["generation"]

    random.seed(gen_seed); np.random.seed(gen_seed)
    torch.manual_seed(gen_seed); torch.cuda.manual_seed_all(gen_seed)

    ds = load_dataset(ds_cfg["hf_path"], ds_cfg["hf_config"], split="validation")
    samples = [{"prompt_text": ds_cfg["prompt_template"].format(question=ex["question"]),
                "correct_answers": [str(ex["best_answer"])],
                "incorrect_answers": [str(a) for a in ex["incorrect_answers"]]}
               for ex in ds] if dataset == "truthfulqa" else None
    if samples is None:
        raise NotImplementedError(f"dataset {dataset} not wired up")

    rouge = evaluate.load("rouge")
    bleurt = evaluate.load("bleurt", config_name="BLEURT-20")

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                  device_map=device, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    D = model.config.hidden_size

    print("[Route N] Computing bases from lm_head.weight (fp32 SVD) ...")
    V_R, V_rand, spectrum = band_mod.compute_bases(model)

    route, agreement = band_mod.verify_post_norm_route(
        model, tokenizer, ["The capital of France is", "Water boils at a temperature of"], device)
    print(f"[Route N] A2 post-norm route: {route}  agreement={agreement:.4f}")
    apply_norm_manually = (route == "manual_norm")

    eos_strs = [".", "!", "?", ".\n", "!\n", "?\n", "\n", "\n\n"]
    eos_ids = {tokenizer.eos_token_id}
    for s in eos_strs:
        eos_ids.update(tokenizer.encode(s, add_special_tokens=False))
        eos_ids.update(tokenizer.encode("Yes" + s, add_special_tokens=False)[1:])

    decoding_config = {"do_sample": gen_cfg["do_sample"], "num_beams": gen_cfg["num_beams"],
                        "temperature": gen_cfg["temperature"], "top_p": gen_cfg["top_p"],
                        "top_k": gen_cfg["top_k"], "max_new_tokens": gen_cfg["max_new_tokens"],
                        "gen_seed": gen_seed, **library_versions()}

    all_sequences, all_prompt_lens, all_decoded_text = [], [], []
    all_flags, all_is_known, all_prompt_idx = [], [], []
    per_beam_raw = []
    t0 = time.time()

    prompt_bar = tqdm(list(enumerate(samples)), desc="[Route N] generation", unit="prompt")
    for idx, sample in prompt_bar:
        inputs = tokenizer(sample["prompt_text"], return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]
        torch.manual_seed(gen_seed * 100003 + idx); torch.cuda.manual_seed_all(gen_seed * 100003 + idx)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=gen_cfg["max_new_tokens"], eos_token_id=list(eos_ids),
                do_sample=gen_cfg["do_sample"], temperature=gen_cfg["temperature"],
                top_k=gen_cfg["top_k"], top_p=gen_cfg["top_p"], num_beams=gen_cfg["num_beams"],
                num_return_sequences=gen_cfg["num_return_sequences"], output_hidden_states=True,
                return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id, early_stopping=True)

        hidden_states = outputs.hidden_states
        gen_ids_full = outputs.sequences[:, prompt_len:]
        num_gen = len(hidden_states) - 1
        any_correct = False
        for b in range(gen_ids_full.shape[0]):
            gids = gen_ids_full[b]
            gids = gids[gids != tokenizer.eos_token_id]
            gen_text = tokenizer.decode(gids, skip_special_tokens=True).strip()
            all_sequences.append(outputs.sequences[b, :prompt_len + len(gids)].cpu())
            all_prompt_lens.append(prompt_len)
            all_decoded_text.append(gen_text)

            r = rouge.compute(predictions=[gen_text] * len(sample["correct_answers"]),
                               references=sample["correct_answers"]) if sample["correct_answers"] else {"rougeL": 0.0}
            all_refs = sample["correct_answers"] + sample["incorrect_answers"]
            bs = bleurt.compute(predictions=[gen_text] * len(all_refs), references=all_refs)
            max_correct_b = max(bs["scores"][:len(sample["correct_answers"])], default=0)
            is_correct = (r["rougeL"] >= 0.7) or (max_correct_b > 0.5)
            if is_correct:
                any_correct = True
            all_flags.append(not is_correct)

            T_real = min(len(gids), num_gen)
            h_by_layer = {}
            for l in range(W_START, W_END):
                toks = [hidden_states[step][l + 1][b, -1, :] if hidden_states[step][l + 1].dim() == 3
                        else hidden_states[step][l + 1][b] for step in range(1, T_real + 1)]
                h_by_layer[l] = torch.stack(toks, dim=0).float() if toks else torch.zeros(0, D)
            final_toks = []
            for step in range(1, T_real + 1):
                h = hidden_states[step][-1]
                h = h[:, -1, :] if h.dim() == 3 else h
                h = h[b].float()
                if apply_norm_manually:
                    h = model.model.norm(h)
                final_toks.append(h)
            h_by_layer["final_norm"] = torch.stack(final_toks, dim=0) if final_toks else torch.zeros(0, D)

            per_beam_raw.append(pack_raw_state_beam(h_by_layer))

        all_is_known.append(any_correct)
        all_prompt_idx.extend([idx] * gen_ids_full.shape[0])
        del outputs, hidden_states
        torch.cuda.empty_cache(); gc.collect()
        n_beams_so_far = len(all_decoded_text)
        n_halluc_so_far = sum(all_flags)
        prompt_bar.set_postfix_str(f"{n_beams_so_far} beams, {n_halluc_so_far} hallucinated so far")
    prompt_bar.close()
    print(f"[Route N] Generation complete: {len(samples)} prompts, {len(all_decoded_text)} beams "
          f"({time.time()-t0:.0f}s total)")

    n_beams = len(all_decoded_text)
    labels = all_flags
    prompt_ids = all_prompt_idx
    beam_idxs = list(range(n_beams))

    os.makedirs(out_dir, exist_ok=True)
    seq_path = os.path.join(out_dir, f"{dataset}_v3_sequences.pt")
    torch.save({"input_ids": all_sequences, "prompt_len": all_prompt_lens, "prompt_id": prompt_ids,
                "decoded_text": all_decoded_text, "all_hallucination_flag": labels,
                "all_is_known": all_is_known, "decoding_config": decoding_config}, seq_path)
    print(f"[Route N] Saved raw sequences/text/labels: {seq_path}")

    shard_paths, bases_path, meta_path = pack_and_save_raw_store(
        per_beam_raw, prompt_ids, beam_idxs, labels, V_R, V_rand, out_dir,
        model_id, gen_seed, decoding_config, git_commit_hash())

    return {"seq_path": seq_path, "shard_paths": shard_paths, "bases_path": bases_path,
            "meta_path": meta_path, "n_beams": n_beams, "decoding_config": decoding_config}


# ==============================================================================
# RECOVERY: resume raw-state-store extraction from an already-saved sequences file, without
# redoing generation/judging. Only needed because of the V_R/V_rand device bug above -- the
# sequences file is already complete and correct once it exists.
# ==============================================================================

def resume_route_n_raw_state(seq_path, model_folder, out_dir, batch_size=8):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)

    print(f"Loading saved sequences: {seq_path}")
    data = torch.load(seq_path, weights_only=False)
    input_ids_list = data["input_ids"]
    prompt_lens = data["prompt_len"]
    prompt_ids = data["prompt_id"]
    labels = data["all_hallucination_flag"]
    decoding_config = data["decoding_config"]
    n_beams = len(input_ids_list)
    print(f"  {n_beams} beams loaded, generation/judging NOT re-run")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                  device_map=device, trust_remote_code=True)
    model.eval()

    print("Computing bases from lm_head.weight (fp32 SVD) ...")
    V_R, V_rand, spectrum = band_mod.compute_bases(model)
    V_R, V_rand = V_R.detach().cpu(), V_rand.detach().cpu()   # the fix -- always CPU before saving
    route, agreement = band_mod.verify_post_norm_route(
        model, tokenizer, ["The capital of France is", "Water boils at a temperature of"], device)
    apply_norm_manually = (route == "manual_norm")
    print(f"A2 post-norm route: {route}  agreement={agreement:.4f}")

    per_beam_raw = []
    batch_starts = list(range(0, n_beams, batch_size))
    batch_bar = tqdm(batch_starts, desc="[Resume] re-forwarding saved sequences", unit="batch")
    for start in batch_bar:
        end = min(start + batch_size, n_beams)
        lengths = [len(input_ids_list[i]) for i in range(start, end)]
        T_max = max(lengths)
        padded = torch.zeros((end - start, T_max), dtype=torch.long)
        attn = torch.zeros((end - start, T_max), dtype=torch.long)
        for j, i in enumerate(range(start, end)):
            ids = input_ids_list[i]
            padded[j, :len(ids)] = ids
            attn[j, :len(ids)] = 1
        padded, attn = padded.to(device), attn.to(device)
        with torch.no_grad():
            out = model(input_ids=padded, attention_mask=attn, use_cache=False, output_hidden_states=True)

        for j, i in enumerate(range(start, end)):
            p_len = prompt_lens[i]
            comp_start, comp_end = p_len, lengths[j]
            h_by_layer = {l: out.hidden_states[l + 1][j, comp_start:comp_end, :].float()
                          for l in range(W_START, W_END)}
            h_final = out.hidden_states[-1][j, comp_start:comp_end, :].float()
            if apply_norm_manually:
                h_final = model.model.norm(h_final)
            h_by_layer["final_norm"] = h_final
            per_beam_raw.append(pack_raw_state_beam(h_by_layer))
        batch_bar.set_postfix_str(f"{end}/{n_beams} beams")
    batch_bar.close()

    shard_paths, bases_path, meta_path = pack_and_save_raw_store(
        per_beam_raw, prompt_ids, list(range(n_beams)), labels, V_R, V_rand, out_dir,
        model_id, decoding_config.get("gen_seed", 0), decoding_config, git_commit_hash())

    return {"seq_path": seq_path, "shard_paths": shard_paths, "bases_path": bases_path,
            "meta_path": meta_path, "n_beams": n_beams, "decoding_config": decoding_config}


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: inventory heuristics, fingerprint math, raw-state packing (no model)")
    print("=" * 70)
    tmp_dir = os.path.join(HERE, "results", "_selftest_gate")
    os.makedirs(tmp_dir, exist_ok=True)

    # -- 1a inventory: fabricate a COMPLETE candidate and a decoy that shouldn't match --
    complete_path = os.path.join(tmp_dir, "fake_generation_labels.jsonl")
    with open(complete_path, "w") as f:
        for i in range(20):
            f.write(json.dumps({"beam": i, "generated": f"fake completion {i}"}) + "\n")
    decoy_path = os.path.join(tmp_dir, "truthfulqa_pooled_maxenergy.pt")
    open(decoy_path, "wb").close()

    candidates = inventory_search([tmp_dir])
    assert complete_path in candidates
    assert decoy_path not in candidates, "known pooled .pt artifact should be excluded from inventory"
    print(f"  [PASS] inventory_search found the text candidate and excluded the pooled-tensor decoy")

    verdict, payload = run_inventory([tmp_dir], "faketest", expected_n_beams=20)
    assert verdict == "COMPLETE"
    print(f"  [PASS] run_inventory verdict=COMPLETE for a fully-covered fake candidate")

    verdict2, _ = run_inventory([tmp_dir], "faketest", expected_n_beams=50)
    assert verdict2 == "PARTIAL"
    print(f"  [PASS] run_inventory verdict=PARTIAL when coverage is incomplete (20/50)")

    empty_dir = os.path.join(tmp_dir, "empty")
    os.makedirs(empty_dir, exist_ok=True)
    verdict3, _ = run_inventory([empty_dir], "faketest", expected_n_beams=20)
    assert verdict3 == "NONE"
    print(f"  [PASS] run_inventory verdict=NONE when nothing is found (isolated root, no hardcoded fallback)")

    # -- pre-verify token counts --
    offsets = np.array([0, 5, 5, 12, 20])   # 3 beams: lens 5, 0, 7, 8
    match_rate = preverify_token_counts([5, 0, 7, 8], offsets)
    assert match_rate == 1.0
    match_rate_bad = preverify_token_counts([5, 1, 7, 8], offsets)
    assert abs(match_rate_bad - 0.75) < 1e-9
    print(f"  [PASS] preverify_token_counts: exact=1.0, one mismatch=0.75")

    # -- fingerprint verify --
    rng = np.random.default_rng(0)
    stored = rng.normal(0, 1, size=(50, 16))
    recomputed_good = stored + rng.normal(0, 1e-4, size=stored.shape)   # near-identical
    pass_rate, cos = fingerprint_verify(recomputed_good, stored)
    assert pass_rate > 0.99, f"near-identical vectors should pass fingerprint check, got {pass_rate}"
    recomputed_bad = rng.normal(0, 1, size=stored.shape)                # unrelated
    pass_rate_bad, _ = fingerprint_verify(recomputed_bad, stored)
    assert pass_rate_bad < 0.5, "unrelated vectors should mostly fail fingerprint check"
    print(f"  [PASS] fingerprint_verify: near-identical pass_rate={pass_rate:.3f}, "
          f"unrelated pass_rate={pass_rate_bad:.3f}")

    # -- raw-state store packing/sharding round-trip --
    D = 24
    n_beams = 30
    per_beam_raw = []
    prompt_ids, beam_idxs, labels = [], [], []
    for i in range(n_beams):
        T_i = int(rng.integers(0, 8))
        h_by_layer = {l: torch.tensor(rng.normal(0, 1, size=(T_i, D)).astype(np.float32))
                      for l in range(W_START, W_END)}
        h_by_layer["final_norm"] = torch.tensor(rng.normal(0, 1, size=(T_i, D)).astype(np.float32))
        per_beam_raw.append(pack_raw_state_beam(h_by_layer))
        prompt_ids.append(i // 5); beam_idxs.append(i % 5); labels.append(int(rng.integers(0, 2)))

    assert per_beam_raw[0].dtype == torch.bfloat16
    V_R = torch.randn(320, D); V_rand = torch.randn(256, D)
    out_dir = os.path.join(tmp_dir, "raw_store")
    global SHARD_BYTES_LIMIT
    orig_limit = SHARD_BYTES_LIMIT
    SHARD_BYTES_LIMIT = 500   # force multi-shard for this test
    shard_paths, bases_path, meta_path = pack_and_save_raw_store(
        per_beam_raw, prompt_ids, beam_idxs, labels, V_R, V_rand, out_dir,
        "self-test", 0, {"do_sample": True}, "selftest")
    SHARD_BYTES_LIMIT = orig_limit
    assert len(shard_paths) >= 2, "expected forced sharding to produce >=2 shards"

    total_reloaded = 0
    for i, sp in enumerate(shard_paths):
        raw, off, pid, bidx, lab = load_raw_state_shard(sp)
        assert raw.dtype == torch.bfloat16
        total_reloaded += raw.shape[0]
    expected_total = sum(t.shape[0] for t in per_beam_raw)
    assert total_reloaded == expected_total
    print(f"  [PASS] raw-state store: {len(shard_paths)} shards, bf16 round-trip preserved, "
          f"{total_reloaded} total tokens match")

    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["n_beams"] == n_beams
    assert meta["dtype"].startswith("bfloat16")
    print(f"  [PASS] raw_state_meta.json: {meta['n_beams']} beams, layer_order={meta['layer_order']}")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# 1b (continued) -- ROUTE R DRIVER: full reconstruct + re-forward + fingerprint, then Part 2
# ==============================================================================

def run_route_r(manifest, model_folder, dataset, found_texts, out_dir, batch_size=12):
    """found_texts: list from run_inventory's payload, len == n_beams, per-beam completion text.
    Prompts are reconstructed independently from the ORIGINAL dataset + the already-pinned
    prompt_indices (no dependency on whatever schema the found text file happens to have)."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)
    ds_cfg = next(d for d in cfg["datasets"] if d["name"] == dataset)

    pooled = torch.load(manifest["pooled_pt_path"], weights_only=False)
    prompt_idx = np.asarray(pooled["prompt_indices"], dtype=np.int64)
    labels = np.asarray([int(f) for f in pooled["all_hallucination_flag"]], dtype=np.int64)
    n_beams = len(prompt_idx)

    band_meta_path = os.path.join(os.path.dirname(manifest["pooled_pt_path"]), f"{dataset}_band_meta.json")
    with open(band_meta_path) as f:
        band_meta = json.load(f)
    shard_dir = os.path.dirname(band_meta_path)
    band_shard_paths = [os.path.join(shard_dir, s) for s in band_meta["shards"]]
    packed = band_mod.load_packed(band_shard_paths)
    if not np.array_equal(packed["label"], labels):
        raise ValueError("Band npz labels != pooled labels -- refusing Route R on a possibly "
                          "misaligned dataset.")

    ds = load_dataset(ds_cfg["hf_path"], ds_cfg["hf_config"], split="validation")
    questions = ds["question"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Route R's re-forward step.")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Reconstructing sequences and pre-verifying token counts against the pinned band npz ...")
    all_ids, reconstructed_lens = [], []
    for i in range(n_beams):
        q = questions[int(prompt_idx[i])]
        text = found_texts[i] or ""
        full_ids, p_len, c_len = reconstruct_sequence(tokenizer, ds_cfg["prompt_template"], q, text)
        all_ids.append((full_ids, p_len))
        reconstructed_lens.append(c_len)

    match_rate = preverify_token_counts(reconstructed_lens, packed["offsets"])
    print(f"  Pre-verify token-count match rate: {match_rate*100:.2f}%")
    if match_rate <= 0.99:
        return {"status": "FAIL", "stage": "preverify", "match_rate": match_rate}

    print("  Pre-verify passed. Loading model and re-forwarding all beams (also extracts the "
          "raw-state store in the same pass) ...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                  device_map=device, trust_remote_code=True)
    model.eval()
    D = model.config.hidden_size

    V_R, V_rand, spectrum = band_mod.compute_bases(model)
    V_R_dev, V_rand_dev = V_R.to(device), V_rand.to(device)
    route, agreement = band_mod.verify_post_norm_route(
        model, tokenizer, ["The capital of France is", "Water boils at a temperature of"], device)
    apply_norm_manually = (route == "manual_norm")
    print(f"  A2 post-norm route: {route}  agreement={agreement:.4f}")

    stored_z_band_mean = np.zeros((n_beams, packed["z_band"].shape[1]), dtype=np.float32)
    for i in range(n_beams):
        s, e = packed["offsets"][i], packed["offsets"][i + 1]
        stored_z_band_mean[i] = packed["z_band"][s:e].mean(axis=0) if e > s else 0.0

    recomputed_z_band_mean = np.zeros_like(stored_z_band_mean)
    per_beam_raw = []
    batch_starts = list(range(0, n_beams, batch_size))
    batch_bar = tqdm(batch_starts, desc="[Route R] re-forward + fingerprint", unit="batch")
    for start in batch_bar:
        end = min(start + batch_size, n_beams)
        batch = all_ids[start:end]
        lengths = [len(ids) for ids, _ in batch]
        T_max = max(lengths)
        padded = torch.zeros((len(batch), T_max), dtype=torch.long)
        attn = torch.zeros((len(batch), T_max), dtype=torch.long)
        for j, (ids, _) in enumerate(batch):
            padded[j, :len(ids)] = ids
            attn[j, :len(ids)] = 1
        padded, attn = padded.to(device), attn.to(device)
        with torch.no_grad():
            out = model(input_ids=padded, attention_mask=attn, use_cache=False, output_hidden_states=True)

        for j, (ids, p_len) in enumerate(batch):
            beam_i = start + j
            comp_start, comp_end = p_len, lengths[j]
            h_by_layer = {l: out.hidden_states[l + 1][j, comp_start:comp_end, :].float()
                          for l in range(W_START, W_END)}
            h_final = out.hidden_states[-1][j, comp_start:comp_end, :].float()
            if apply_norm_manually:
                h_final = model.model.norm(h_final)
            h_by_layer["final_norm"] = h_final
            per_beam_raw.append(pack_raw_state_beam(h_by_layer))

            if h_final.shape[0] > 0:
                z_band_tok = h_final @ V_R_dev.T.to(h_final.dtype)
                recomputed_z_band_mean[beam_i] = z_band_tok.mean(dim=0).float().cpu().numpy()
        batch_bar.set_postfix_str(f"{end}/{n_beams} beams")
    batch_bar.close()

    pass_rate, cos = fingerprint_verify(recomputed_z_band_mean, stored_z_band_mean)
    print(f"  Fingerprint verification: {pass_rate*100:.2f}% of beams have cosine > 0.999 "
          f"(mean cosine={float(cos.mean()):.5f})")
    if pass_rate <= 0.99:
        return {"status": "FAIL", "stage": "fingerprint", "pass_rate": pass_rate}

    decoding_config = {"do_sample": cfg["generation"]["do_sample"], "num_beams": cfg["generation"]["num_beams"],
                        "temperature": cfg["generation"]["temperature"], "top_p": cfg["generation"]["top_p"]}
    shard_paths, bases_path, meta_path = pack_and_save_raw_store(
        per_beam_raw, prompt_idx.tolist(), list(range(n_beams)), labels.tolist(), V_R, V_rand,
        out_dir, model_id, 0, decoding_config, git_commit_hash())

    return {"status": "PASS", "match_rate": match_rate, "fingerprint_pass_rate": pass_rate,
            "shard_paths": shard_paths, "bases_path": bases_path, "meta_path": meta_path}


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="truthfulqa")
    parser.add_argument("--manifest", type=str, default="data/manifest_seeded_v1.json")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto"])
    parser.add_argument("--authorize-new-pass", action="store_true")
    parser.add_argument("--resume-raw-state", type=str, default=None,
                         help="Recovery: path to an already-saved {dataset}_v3_sequences.pt "
                              "(from a Route N run that crashed after saving sequences but "
                              "before finishing the raw-state store). Re-forwards the saved "
                              "sequences directly -- does NOT re-run generation/judging.")
    parser.add_argument("--gen-seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.model_folder:
        print("ERROR: --model_folder required."); sys.exit(1)

    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    data_dir = cfg["output"]["data_dir"]
    out_dir = os.path.join(data_dir, args.model_folder, "raw_state_store")

    if args.resume_raw_state:
        print("=" * 70)
        print("  RESUME: rebuilding the raw-state store from already-saved sequences")
        print("=" * 70)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required.")
        result = resume_route_n_raw_state(args.resume_raw_state, args.model_folder, out_dir)
        print(f"\nResume complete. n_beams={result['n_beams']}")
        print(f"Raw-state store: {result['shard_paths']}")
        print("Next: run 35_derive_streams.py with --route N, then update the manifest to v3 "
              "and recompute canonical references per Part 3d.")
        return

    if args.authorize_new_pass:
        print("=" * 70)
        print("  ROUTE N: authorized fresh generation pass (dataset v3)")
        print("=" * 70)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required.")
        result = run_route_n(args.model_folder, args.dataset, args.gen_seed, out_dir)
        print(f"\nRoute N complete. n_beams={result['n_beams']}")
        print(f"Sequences: {result['seq_path']}")
        print(f"Raw-state store: {result['shard_paths']}")
        print("Next: run 35_derive_streams.py with --route N, then update the manifest to v3 "
              "and recompute canonical references per Part 3d.")
        return

    manifest_path = os.path.join(HERE, args.manifest) if not os.path.isabs(args.manifest) else args.manifest
    manifest = pin_mod.verify_manifest(manifest_path)
    n_beams = manifest["counts"]["n_beams"]
    print(f"Manifest verified. n_beams={n_beams}")

    print("\n[Part 1a] Inventory search ...")
    search_roots = [os.path.join(data_dir, args.model_folder),
                     os.path.join(os.path.dirname(data_dir.rstrip("/\\")), "results"),
                     os.path.join(HERE, "results")]
    verdict, payload = run_inventory(search_roots, args.dataset, n_beams)

    if verdict != "COMPLETE":
        print(f"\nRECONSTRUCTION IMPOSSIBLE/FAILED: inventory verdict was {verdict}, not COMPLETE. "
              f"Re-run with --authorize-new-pass to regenerate (creates dataset v3).")
        sys.exit(1)

    print("\n[Part 1b] Route R: reconstruct + verify ...")
    found_path, found_texts = payload
    print(f"  Using candidate: {found_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Route R's re-forward/fingerprint step.")

    result = run_route_r(manifest, args.model_folder, args.dataset, found_texts, out_dir)

    if result["status"] == "FAIL":
        if result["stage"] == "preverify":
            detail = f"token-count pre-verify match rate {result['match_rate']*100:.2f}% (need >99%)"
        else:
            detail = f"fingerprint pass rate {result['pass_rate']*100:.2f}% (need >99%)"
        print(f"\nRECONSTRUCTION IMPOSSIBLE/FAILED: Route R failed at the {result['stage']} stage "
              f"-- {detail}. Re-run with --authorize-new-pass to regenerate (creates dataset v3).")
        sys.exit(1)

    print(f"\nRoute R PASSED: dataset exactly recovered (match_rate={result['match_rate']*100:.2f}%, "
          f"fingerprint_pass_rate={result['fingerprint_pass_rate']*100:.2f}%). Labels and label "
          f"composition (406/111/300) unchanged. Raw-state store: {result['shard_paths']}")
    print("Next: run 35_derive_streams.py with --route R, then extend the manifest to v2.1 with "
          "sequence + store hashes, per Part 1b.")


if __name__ == "__main__":
    main()
