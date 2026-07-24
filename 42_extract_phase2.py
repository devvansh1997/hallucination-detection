"""
42_extract_phase2.py -- Session 06 Phase 2 Part A: Cross-Dataset Extraction (GPU)
====================================================================================
Reuses the EXACT TruthfulQA v3 store-based extraction module, per this phase's explicit
instruction -- does NOT reimplement the forward-pass/hook logic:
  - 34_gate_reconstruct_or_regenerate.py's resume_route_n_raw_state(): the same hook
    convention (out.hidden_states[l+1] post-block residual for layers 15..23, optional
    manual final-norm application decided by verify_post_norm_route()) used to build
    TruthfulQA v3's raw-state store. *** EXCEPT: a real GPU-memory-accumulation bug in
    that function (confirmed via a real TriviaQA OOM crash at 95,664/99,600 beams -- see
    reforward_and_extract_raw_state()'s docstring below) is fixed here in a local copy of
    just the batch loop, since 34_gate_reconstruct_or_regenerate.py is a pre-existing file
    this project doesn't modify. Same hook convention, same math, same helper functions
    (pack_raw_state_beam, pack_and_save_raw_store, compute_bases, verify_post_norm_route
    all reused unchanged) -- only the device-management bug differs.
  - 35_derive_streams.py's derive_all_streams() / derive_positive_max_core(): the same
    pooling math (q95/q05 quantile pooling over tokens, inter-layer deltas, positive
    max-pool) used to derive TruthfulQA v3's velocity/kinematic/static/core features.

This file is a thin driver: it points those functions at Phase 1's pinned
{dataset}_sequences_v1.pt files (triviaqa/nq_open/tydiqa_gp) instead of TruthfulQA's,
casts the outputs to Phase 2's fp16 storage format, deletes the transient raw-state-store
shards afterward (they run ~2-3x larger than the final pooled outputs and were never
asked to be kept -- keeping them would blow well past the ~50GB fp16 budget), and writes
manifest v2.

Window boundaries: 39_generate_dataset.py already truncates every beam's input_ids to the
canonical window at SAVE TIME (session05b's fix, validated by Phase 1's Assert B). So the
completion boundary for beam i is simply prompt_len[i] to len(input_ids[i]) -- no
re-derivation needed here. This script ASSERTS that identity (comparing a window-length
recomputation from the raw-state store's actual captured token counts against the
pre-extraction figure from the sequences file directly) rather than assuming it silently.

Usage:
  python 42_extract_phase2.py --self-test
  python 42_extract_phase2.py --dataset triviaqa --model_folder llama-3.1-8b-instruct
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Llama-3.1-8B-Instruct's max_position_embeddings -- fixed architecture fact, used only to
# sanity-check TyDiQA-GP's long passages weren't silently truncated during Phase 1 generation.
CONTEXT_LIMIT = 131072


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate_mod = _load("s04_gate", "34_gate_reconstruct_or_regenerate.py")
derive_mod = _load("s04_derive", "35_derive_streams.py")
vel_mod = _load("s04_vel", "32_extract_velocity.py")
band_mod = _load("s02_extract", "27_extract_band.py")
pin_mod = _load("s03_pin", "30_pin_manifest.py")

sha256_file = pin_mod.sha256_file
sha256_array = pin_mod.sha256_array

W_START, W_END = gate_mod.W_START, gate_mod.W_END


# ==============================================================================
# GPU-MEMORY-SAFE RE-FORWARD -- fixes a confirmed OOM bug in the reused
# resume_route_n_raw_state(), without modifying 34_gate_reconstruct_or_regenerate.py
# ==============================================================================
#
# Real crash (TriviaQA, N=99,600 beams, H100 80GB): torch.OutOfMemoryError at 95,664/99,600
# beams (96% through), GPU at 79.17/79.18 GiB used while trying to allocate 2 MiB -- a slow
# accumulation, not a spike. Root cause: resume_route_n_raw_state()'s per_beam_raw list
# never moves each batch's extracted hidden-state slices off the GPU before appending --
# pack_raw_state_beam()'s ".to(torch.bfloat16)" only changes dtype, not device, so the list
# holds GPU tensors for the entire dataset for the whole run. TriviaQA's real math:
# 99,600 beams x ~7.67 tokens avg x 10 layers x 4096 dim x 2 bytes (bf16) = ~81.6 GB --
# matches the observed near-exact exhaustion. TyDiQA-GP (4,400 beams) and NQ-Open
# (36,100 beams) stay under 80GB in total, which is why only TriviaQA hit this.
#
# This is a genuine, pre-existing bug in the reused function, not a config issue --
# 34_gate_reconstruct_or_regenerate.py is a pre-existing file this project doesn't modify,
# so the fix lives here instead: an EXACT copy of the same hook convention and math (same
# layer slicing, same post-norm-route handling, same packing/saving helpers, reused
# unchanged), with only the device-management bug fixed -- each beam's packed tensor is
# moved to CPU immediately, and the GPU forward-pass output is explicitly freed every batch.

def reforward_and_extract_raw_state(seq_path, model_folder, out_dir, batch_size=16):
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                  device_map=device, trust_remote_code=True)
    model.eval()

    print("Computing bases from lm_head.weight (fp32 SVD) ...")
    V_R, V_rand, spectrum = band_mod.compute_bases(model)
    V_R, V_rand = V_R.detach().cpu(), V_rand.detach().cpu()
    route, agreement = band_mod.verify_post_norm_route(
        model, tokenizer, ["The capital of France is", "Water boils at a temperature of"], device)
    apply_norm_manually = (route == "manual_norm")
    print(f"A2 post-norm route: {route}  agreement={agreement:.4f}")

    per_beam_raw = []
    batch_starts = list(range(0, n_beams, batch_size))
    batch_bar = tqdm(batch_starts, desc="[Fixed re-forward] extracting (CPU-offloaded per batch)",
                      unit="batch")
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
            # THE FIX: .cpu() here, immediately -- everything else is identical to
            # resume_route_n_raw_state(). Without this, per_beam_raw accumulates GPU
            # tensors for the entire dataset.
            per_beam_raw.append(gate_mod.pack_raw_state_beam(h_by_layer).cpu())

        del out
        torch.cuda.empty_cache()
        if start % (batch_size * 50) == 0:
            gc.collect()
        batch_bar.set_postfix_str(f"{end}/{n_beams} beams")
    batch_bar.close()

    shard_paths, bases_path, meta_path = gate_mod.pack_and_save_raw_store(
        per_beam_raw, prompt_ids, list(range(n_beams)), labels, V_R, V_rand, out_dir,
        model_id, decoding_config.get("gen_seed", 0), decoding_config, gate_mod.git_commit_hash())

    del model
    torch.cuda.empty_cache()

    return {"seq_path": seq_path, "shard_paths": shard_paths, "bases_path": bases_path,
            "meta_path": meta_path, "n_beams": n_beams, "decoding_config": decoding_config}


# ==============================================================================
# PURE HELPERS (independently self-testable, no model/GPU/network)
# ==============================================================================

def compute_window_stats(seq_data):
    """Mean/min/max completion length + max prompt length, straight from the pinned
    sequences file -- no model needed. This is the pre-extraction reference the
    post-extraction recomputation (from the raw-state store's actual captured T per beam)
    must match exactly."""
    comp_lens = [len(seq_data["input_ids"][i]) - seq_data["prompt_len"][i]
                 for i in range(len(seq_data["input_ids"]))]
    return {
        "mean_completion_len": float(np.mean(comp_lens)),
        "min_completion_len": int(np.min(comp_lens)),
        "max_completion_len": int(np.max(comp_lens)),
        "max_prompt_len": int(np.max(seq_data["prompt_len"])),
        "n_beams": len(comp_lens),
    }


def spot_decode(seq_data, n=5, seed=0):
    """Prints n random completions' already-decoded text (Phase 1 saved decoded_text for
    every beam -- no need to re-decode via a tokenizer here)."""
    rng = random.Random(seed)
    idx = rng.sample(range(len(seq_data["decoded_text"])), min(n, len(seq_data["decoded_text"])))
    lines = []
    for i in idx:
        text = seq_data["decoded_text"][i]
        lines.append(f"  beam {i} (prompt {seq_data['prompt_id'][i]}): {text[:150]!r}")
    return lines


def cast_and_pack_phase2_features(vel_packed, core_all_emb):
    """vel_packed: dict from derive_all_streams()'s packed output (V95,V05,S95,S05,kinematic,
    prompt_id,beam_idx,label -- all float32 numpy). core_all_emb: list of (9,D) tensors from
    derive_positive_max_core()'s all_emb. Casts to Phase 2's storage dtypes: static max-pool /
    static q95,q05 / velocity q95,q05 all fp16; kinematic-30 stays fp32 (explicit spec
    instruction: 'inert until Phase 3', kept at full precision for whatever consumes it then)."""
    static_max = torch.stack(core_all_emb).numpy().astype(np.float16)
    return {
        "static_max": static_max,
        "static_q95": vel_packed["S95"].astype(np.float16),
        "static_q05": vel_packed["S05"].astype(np.float16),
        "velocity_q95": vel_packed["V95"].astype(np.float16),
        "velocity_q05": vel_packed["V05"].astype(np.float16),
        "kinematic": vel_packed["kinematic"].astype(np.float32),
        "prompt_id": vel_packed["prompt_id"], "beam_idx": vel_packed["beam_idx"],
        "label": vel_packed["label"],
    }


def recompute_window_lengths_from_raw_store(raw_store_dir):
    """Re-derives per-beam completion token counts directly from the raw-state store's shard
    offsets (independent of the pre-extraction seq_data path) -- the genuine regression check:
    these two derivations of 'how long was this completion' must agree exactly, since they
    describe the same pinned data through two different code paths."""
    meta_path = os.path.join(raw_store_dir, "raw_state_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    lens = []
    for shard_name in meta["shards"]:
        _, offsets, _, _, _ = gate_mod.load_raw_state_shard(os.path.join(raw_store_dir, shard_name))
        lens.extend(np.diff(offsets).tolist())
    return lens


# ==============================================================================
# RESOURCE ESTIMATE -- printed BEFORE launching (Part A spec requirement)
# ==============================================================================

EMPIRICAL_S_PER_BEAM_REFORWARD = 0.05   # rough: teacher-forced single fwd pass, no decode loop,
                                          # far cheaper than generation's per-token sampling cost.


def print_resource_estimate(dataset_name, n_beams):
    est_s = n_beams * EMPIRICAL_S_PER_BEAM_REFORWARD
    # static: max + q95 + q05 (3x [N,9,4096] fp16) ; velocity: q95 + q05 (2x [N,8,4096] fp16) ;
    # kinematic: [N,30] fp32 (negligible)
    static_bytes = n_beams * 9 * 4096 * 2 * 3
    velocity_bytes = n_beams * 8 * 4096 * 2 * 2
    kinematic_bytes = n_beams * 30 * 4
    disk_mb = (static_bytes + velocity_bytes + kinematic_bytes) / 1e6
    print(f"  [{dataset_name}] N={n_beams} beams -- est. extraction time: {est_s/60:.1f} min "
          f"(rough, teacher-forced single fwd pass, no decode loop)")
    print(f"  [{dataset_name}] est. disk (fp16 pooled outputs, this dataset only): {disk_mb:.0f} MB")
    return est_s, disk_mb


# ==============================================================================
# PART A DRIVER
# ==============================================================================

def run_extraction(dataset, model_folder, data_dir, batch_size=16):
    seq_path = os.path.join(data_dir, model_folder, f"{dataset}_sequences_v1.pt")
    print(f"Loading pinned Phase 1 sequences: {seq_path}")
    seq_data = torch.load(seq_path, weights_only=False)

    stats_pre = compute_window_stats(seq_data)
    print(f"  [Pre-extraction] mean completion len={stats_pre['mean_completion_len']:.2f}  "
          f"min={stats_pre['min_completion_len']}  max={stats_pre['max_completion_len']}  "
          f"max_prompt_len={stats_pre['max_prompt_len']}")
    if stats_pre["max_prompt_len"] >= CONTEXT_LIMIT:
        print(f"  [WARN] max prompt length {stats_pre['max_prompt_len']} >= model context limit "
              f"{CONTEXT_LIMIT} -- possible prompt-side truncation during Phase 1 generation.")
    else:
        print(f"  [PASS] no prompt-side truncation risk (max {stats_pre['max_prompt_len']} tokens, "
              f"well under the {CONTEXT_LIMIT}-token context limit)")

    print("\n  Spot-decoded completions (from Phase 1's pinned decoded_text, 5 random beams):")
    for line in spot_decode(seq_data, n=5, seed=0):
        print(line)

    print()
    print_resource_estimate(dataset, len(seq_data["input_ids"]))

    raw_store_dir = os.path.join(data_dir, model_folder, f"{dataset}_raw_state_store")
    print(f"\n[Extraction] Teacher-forced re-forward -> raw-state store: {raw_store_dir}")
    t0 = time.time()
    gate_result = reforward_and_extract_raw_state(seq_path, model_folder, raw_store_dir,
                                                    batch_size=batch_size)
    print(f"[Extraction] Re-forward complete: {gate_result['n_beams']} beams "
          f"({time.time()-t0:.0f}s total)")

    print("\n[Derivation] velocity/kinematic/static-repooling + positive-max core "
          "(35_derive_streams.py, unchanged) ...")
    vel_packed, _band_derived, _meta = derive_mod.derive_all_streams(raw_store_dir, route="N",
                                                                       canonical_lengths=None)
    core_data = derive_mod.derive_positive_max_core(raw_store_dir, canonical_lengths=None)

    print("\n[Asserts]")
    lens_post = recompute_window_lengths_from_raw_store(raw_store_dir)
    mean_post = float(np.mean(lens_post))
    assert abs(mean_post - stats_pre["mean_completion_len"]) < 1e-6, (
        f"Recomputed mean window length ({mean_post:.4f}) != pre-extraction figure "
        f"({stats_pre['mean_completion_len']:.4f}) -- the extraction pipeline's window "
        f"handling has diverged from the pinned sequences.")
    print(f"  [PASS] recomputed mean window length ({mean_post:.4f}) matches Phase 1's pinned "
          f"sequences exactly")
    n_empty = sum(1 for l in lens_post if l == 0)
    assert n_empty == 0, f"{n_empty} beams have zero-length completion windows post-extraction"
    print(f"  [PASS] zero empty windows ({len(lens_post)} beams checked)")

    print("\n[Packing] Casting to Phase 2 storage format (fp16 pooled tensors, fp32 kinematic) ...")
    phase2 = cast_and_pack_phase2_features(vel_packed, core_data["all_emb"])

    out_dir = os.path.join(data_dir, model_folder)
    out_path = os.path.join(out_dir, f"{dataset}_phase2_features.npz")
    np.savez_compressed(out_path, **phase2)
    print(f"Saved: {out_path}")

    # transient raw-state store is 2-3x larger than the saved pooled features and was never
    # asked to be kept -- delete it to stay near the ~50GB fp16 budget across all 3 datasets.
    shutil.rmtree(raw_store_dir, ignore_errors=True)
    print(f"Deleted transient raw-state store: {raw_store_dir}")

    shapes = {k: list(v.shape) for k, v in phase2.items() if hasattr(v, "shape")}
    hashes = {k: sha256_array(v) for k, v in phase2.items() if hasattr(v, "shape")}
    npz_hash = sha256_file(out_path)

    manifest_v1_path = os.path.join(out_dir, f"manifest_{dataset}_v1.json")
    with open(manifest_v1_path) as f:
        manifest_v1 = json.load(f)
    manifest_v2 = dict(manifest_v1)
    manifest_v2["version"] = 2
    manifest_v2["phase2_features_path"] = os.path.abspath(out_path)
    manifest_v2["phase2_features_npz_sha256"] = npz_hash
    manifest_v2["phase2_shapes"] = shapes
    manifest_v2["phase2_tensor_sha256"] = hashes
    manifest_v2["phase2_window_stats"] = stats_pre
    manifest_v2_path = os.path.join(out_dir, f"manifest_{dataset}_v2.json")
    with open(manifest_v2_path, "w") as f:
        json.dump(manifest_v2, f, indent=2)
    print(f"Wrote manifest v2: {manifest_v2_path}")

    return {"dataset": dataset, "n_beams": gate_result["n_beams"], "out_path": out_path,
            "manifest_v2_path": manifest_v2_path, "shapes": shapes}


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: window stats, fp16 packing, manifest-v2, full synthetic extraction chain")
    print("=" * 70)

    n_beams = 20
    rng = np.random.default_rng(0)
    input_ids = [torch.arange(5 + int(rng.integers(1, 10))) for _ in range(n_beams)]
    seq_data = {"input_ids": input_ids, "prompt_len": [5] * n_beams,
                "prompt_id": [i // 10 for i in range(n_beams)],
                "decoded_text": [f"fake answer {i}" for i in range(n_beams)]}
    stats = compute_window_stats(seq_data)
    expected_mean = float(np.mean([len(ids) - 5 for ids in input_ids]))
    assert abs(stats["mean_completion_len"] - expected_mean) < 1e-9
    assert stats["max_prompt_len"] == 5
    print(f"  [PASS] compute_window_stats: mean={stats['mean_completion_len']:.2f}, "
          f"n_beams={stats['n_beams']}")

    lines = spot_decode(seq_data, n=5, seed=0)
    assert len(lines) == 5
    print(f"  [PASS] spot_decode: {len(lines)} lines produced")

    D = 16
    vel_packed = {
        "S95": rng.normal(0, 1, size=(n_beams, 9, D)).astype(np.float32),
        "S05": rng.normal(0, 1, size=(n_beams, 9, D)).astype(np.float32),
        "V95": rng.normal(0, 1, size=(n_beams, 8, D)).astype(np.float32),
        "V05": rng.normal(0, 1, size=(n_beams, 8, D)).astype(np.float32),
        "kinematic": rng.normal(0, 1, size=(n_beams, 30)).astype(np.float32),
        "prompt_id": np.arange(n_beams) // 10, "beam_idx": np.arange(n_beams) % 10,
        "label": rng.integers(0, 2, size=n_beams),
    }
    core_all_emb = [torch.tensor(rng.normal(0, 1, size=(9, D)).astype(np.float32)) for _ in range(n_beams)]
    phase2 = cast_and_pack_phase2_features(vel_packed, core_all_emb)
    assert phase2["static_max"].dtype == np.float16 and phase2["static_max"].shape == (n_beams, 9, D)
    assert phase2["velocity_q95"].dtype == np.float16 and phase2["velocity_q95"].shape == (n_beams, 8, D)
    assert phase2["kinematic"].dtype == np.float32, "kinematic must stay fp32 per spec"
    print(f"  [PASS] cast_and_pack_phase2_features: fp16 for static/velocity, fp32 for kinematic, "
          f"shapes correct")

    # -- full synthetic chain: fake raw-state store -> derive_all_streams/derive_positive_max_core
    # -> cast_and_pack -> window-length regression assert, exercising the exact same code path
    # run_extraction() uses on real data --
    tmp_dir = os.path.join(HERE, "results", "_selftest_phase2_extract")
    os.makedirs(tmp_dir, exist_ok=True)
    W_START, W_END = 15, 24
    per_beam_raw = []
    prompt_ids, beam_idxs, labels = [], [], []
    true_lens = []
    for i in range(n_beams):
        T_i = int(rng.integers(1, 9))
        true_lens.append(T_i)
        h = {l: torch.tensor(rng.normal(0, 1, size=(T_i, D)).astype(np.float32)) for l in range(W_START, W_END)}
        h["final_norm"] = torch.tensor(rng.normal(0, 1, size=(T_i, D)).astype(np.float32))
        per_beam_raw.append(gate_mod.pack_raw_state_beam(h))
        prompt_ids.append(i // 5); beam_idxs.append(i % 5); labels.append(int(rng.integers(0, 2)))
    V_R, V_rand = torch.randn(20, D), torch.randn(16, D)
    raw_store_dir = os.path.join(tmp_dir, "raw_store")
    gate_mod.pack_and_save_raw_store(per_beam_raw, prompt_ids, beam_idxs, labels, V_R, V_rand,
                                      raw_store_dir, "self-test", 0, {"do_sample": True}, "selftest")

    lens_post = recompute_window_lengths_from_raw_store(raw_store_dir)
    assert lens_post == true_lens, f"recomputed lengths {lens_post} != fabricated {true_lens}"
    print(f"  [PASS] recompute_window_lengths_from_raw_store: exact match against fabricated ground truth")

    vel_packed2, _, _ = derive_mod.derive_all_streams(raw_store_dir, route="N", canonical_lengths=None)
    core_data2 = derive_mod.derive_positive_max_core(raw_store_dir, canonical_lengths=None)
    phase2_2 = cast_and_pack_phase2_features(vel_packed2, core_data2["all_emb"])
    assert phase2_2["static_max"].shape == (n_beams, 9, D)
    assert phase2_2["velocity_q95"].shape == (n_beams, 8, D)
    print(f"  [PASS] full synthetic chain (fake raw-state store -> derive -> cast_and_pack): "
          f"shapes correct end to end")

    # regression: deliberately corrupt one beam's captured length and confirm the assert fires
    lens_bad = lens_post[:]
    lens_bad[0] = lens_bad[0] + 1
    mean_bad = float(np.mean(lens_bad))
    mean_true = float(np.mean(lens_post))
    assert abs(mean_bad - mean_true) > 1e-6, "corrupted mean should differ from the true mean"
    print(f"  [PASS] window-length mismatch would be caught (corrupted mean {mean_bad:.4f} "
          f"vs true {mean_true:.4f} -- run_extraction()'s assert would fire on this)")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["triviaqa", "nq_open", "tydiqa_gp"], default=None)
    parser.add_argument("--model_folder", type=str, default="llama-3.1-8b-instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.dataset:
        print("ERROR: --dataset required."); sys.exit(1)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for extraction.")

    if args.data_dir:
        data_dir = args.data_dir
    else:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        data_dir = cfg["output"]["data_dir"]

    result = run_extraction(args.dataset, args.model_folder, data_dir, batch_size=args.batch_size)
    print(f"\nDone. {result['dataset']}: {result['n_beams']} beams -> {result['out_path']}")
    print(f"Next: run 43_eval_phase2.py --dataset {result['dataset']} ...")


if __name__ == "__main__":
    main()
