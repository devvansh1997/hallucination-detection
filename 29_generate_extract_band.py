"""
29_generate_extract_band.py -- Session 02: Seeded Regeneration + Core/Band Extraction
==========================================================================================
Resolves the blocker found while building 27_extract_band.py: no raw token sequence /
prompt-length cache exists anywhere for the beams behind the Phase-1 baseline (confirmed
against the actual cluster `data/` listing on 2026-07-20 -- there is no data_unpooled/
directory at all, and every cached file is a pooled activation or a small derived-feature
file). do_sample=True with no fixed seed anywhere in the generation code also means those
exact beams could never be reproduced even if we tried.

This script does ONE fixed-seed generation pass per prompt and, using the hidden states
generate() already computes, produces BOTH outputs a fresh, internally-consistent dataset
needs, with no raw sequences or unpooled per-layer states ever touching disk:
  (a) pooled core activations, SAME SCHEMA as 21_generate_maxpool_datasets.py's output
      ({dataset}_pooled_maxenergy_seeded.pt) -- drop-in for 26_grouped_baseline.py /
      28_eval_band.py via --pooled-suffix _maxenergy_seeded
  (b) reasoning-band + random-control token streams, SAME SCHEMA as 27_extract_band.py's
      packed output ({dataset}_band.npz + _meta.json) -- consumed directly by
      28_eval_band.py's --band-meta, no changes needed to either downstream script.

Expected combined size for TruthfulQA (817 prompts x 10 beams): ~1.1-1.3 GB (pooled core
~0.6 GB, band streams ~0.5 GB assuming ~25 completion tokens/beam average) -- NOT the
~50+ GB a literal unpooled all-layers dump would cost. Printed at the end of a real run.

Additive: does not modify 21_generate_maxpool_datasets.py or 27_extract_band.py -- reuses
27's basis/verification/packing functions read-only via import, and re-implements (does
not import, since 21 has no function boundaries to import) the same pooling logic 21 uses.

Usage:
  python 29_generate_extract_band.py --self-test
  python 29_generate_extract_band.py --model_folder llama-3.1-8b-instruct --dataset truthfulqa --gen-seed 0
"""

import argparse
import gc
import importlib.util
import os
import random
import sys
import time

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# --- read-only import of 27_extract_band.py's bases/verification/packing (never modified) ---
_spec = importlib.util.spec_from_file_location("s02_extract", os.path.join(HERE, "27_extract_band.py"))
band_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(band_mod)

W_START, W_END = 15, 24   # same 9-layer reasoning window as 21_generate_maxpool_datasets.py


# ==============================================================================
# CORE POOLING + BAND PROJECTION (same generate() call, one pass, no raw sequences saved)
# ==============================================================================

def pool_and_project_prompt(hidden_states, gen_ids, num_beams, D, V_band, V_rand,
                             apply_norm_manually, norm_fn):
    """hidden_states: tuple over generation steps, each a tuple over (L+1) layers,
    each (num_beams, seq_at_step, D) -- the exact structure generate(output_hidden_states=True)
    returns. gen_ids: (num_beams, T_gen) already-trimmed generated ids (for length only, to
    know how many steps are real vs padding artifacts of early_stopping).
    Returns per beam: H_pooled (9, D) float32, z_band (T_i, band_dim), z_rand (T_i, rand_dim),
    rms (T_i,)."""
    num_gen = len(hidden_states) - 1

    # precompute (optionally normed) final-layer state per generation step, ONCE per prompt
    final_per_step = []
    for step in range(1, len(hidden_states)):
        h = hidden_states[step][-1]                  # (num_beams, 1, D) or (num_beams, D)
        if h.dim() == 3:
            h = h[:, -1, :]
        h = h.float()
        if apply_norm_manually:
            h = norm_fn(h)
        final_per_step.append(h)                     # (num_beams, D)

    results = []
    for b in range(num_beams):
        gids = gen_ids[b]
        T_real = min(len(gids), num_gen)

        # --- core pooling: max over completion tokens, layers W_START:W_END ---
        if T_real == 0:
            H_pooled = torch.zeros(W_END - W_START, D)
        else:
            layers = []
            for l in range(W_START, W_END):
                tokens = [hidden_states[step][l + 1][b, -1, :] if hidden_states[step][l + 1].dim() == 3
                          else hidden_states[step][l + 1][b]
                          for step in range(1, T_real + 1)]
                layer_cat = torch.stack(tokens, dim=0).float()
                layers.append(layer_cat.max(dim=0).values)
            H_pooled = torch.stack(layers, dim=0)      # (9, D)

        # --- band/rand projection: per completion token, final layer only ---
        if T_real == 0:
            z_band = np.zeros((0, V_band.shape[0]), dtype=np.float32)
            z_rand = np.zeros((0, V_rand.shape[0]), dtype=np.float32)
            rms = np.zeros((0,), dtype=np.float32)
        else:
            h_tok = torch.stack([final_per_step[s][b] for s in range(T_real)], dim=0)   # (T_real, D)
            z_band = (h_tok @ V_band.T.to(h_tok.dtype)).cpu().numpy().astype(np.float32)
            z_rand = (h_tok @ V_rand.T.to(h_tok.dtype)).cpu().numpy().astype(np.float32)
            rms = torch.sqrt((h_tok.pow(2)).mean(dim=-1)).cpu().numpy().astype(np.float32)

        results.append((H_pooled.cpu(), z_band, z_rand, rms))
    return results


# ==============================================================================
# SELF-TEST -- fabricates a fake generate()-shaped hidden_states structure, no model needed
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: pooling + band projection from a synthetic generate()-shaped structure")
    print("=" * 70)
    torch.manual_seed(0)
    D, L = 16, 24  # small synthetic hidden dim / layer count (real: D=4096, L=32)
    num_beams = 10
    n_prompts = 6

    V_band, V_rand, _ = band_mod.compute_bases_from_matrix(
        torch.randn(500, D), band_rank=6, rand_rank=4, seed=0)

    class FakeNorm:
        def __call__(self, x):
            return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

    all_core, all_band, all_rand, all_rms_lens = [], [], [], []
    prompt_ids, beam_idxs, labels = [], [], []

    for p in range(n_prompts):
        n_steps = int(torch.randint(3, 12, (1,)))
        gen_len = torch.randint(1, n_steps + 1, (num_beams,))   # per-beam real length <= n_steps
        hidden_states = [tuple(torch.randn(num_beams, 1, D) for _ in range(L + 1))]  # step 0: prompt
        for _ in range(n_steps):
            hidden_states.append(tuple(torch.randn(num_beams, 1, D) for _ in range(L + 1)))
        gen_ids = [torch.zeros(int(gen_len[b])) for b in range(num_beams)]

        results = pool_and_project_prompt(hidden_states, gen_ids, num_beams, D, V_band, V_rand,
                                           apply_norm_manually=True, norm_fn=FakeNorm())
        assert len(results) == num_beams
        for b, (H_pooled, z_band, z_rand, rms) in enumerate(results):
            assert H_pooled.shape == (W_END - W_START, D)
            T_i = int(gen_len[b])
            assert z_band.shape == (T_i, 6), f"z_band shape {z_band.shape} != ({T_i},6)"
            assert z_rand.shape == (T_i, 4)
            assert rms.shape == (T_i,)
            all_core.append(H_pooled)
            all_band.append(z_band); all_rand.append(z_rand); all_rms_lens.append(len(rms))
            prompt_ids.append(p); beam_idxs.append(b)
            labels.append(int(torch.randint(0, 2, (1,))))

    print(f"  [PASS] pooled+projected {n_prompts} prompts x {num_beams} beams, "
          f"shapes consistent with variable per-beam completion lengths")

    # -- pack via 27_extract_band.py's exact packing/offset logic (read-only reuse) --
    per_beam = [(all_band[i], all_rand[i], np.zeros(all_rms_lens[i], dtype=np.float32))
                for i in range(len(all_band))]
    meta = {"checkpoint_id": "self-test-29", "dtype_route": "manual_norm", "seed": 0, "git_commit": "n/a"}
    packed, meta_full = band_mod.pack_beams(per_beam, prompt_ids, beam_idxs, labels, meta)
    assert packed["z_band"].shape[0] == sum(all_rms_lens)
    print(f"  [PASS] band packing produced {packed['z_band'].shape[0]} total tokens, "
          f"offsets consistent with per-beam lengths")

    # -- pooled core output alignment: same beam count/order as band output --
    core_stack = torch.stack(all_core)   # (n_beams, 9, D)
    assert core_stack.shape[0] == len(packed["prompt_id"])
    assert np.array_equal(np.array(prompt_ids), packed["prompt_id"])
    print(f"  [PASS] pooled-core beam count/order matches band-output prompt_id/beam_idx exactly")

    out_dir = os.path.join(HERE, "results", "_selftest_generate_band")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"all_emb": [core_stack[i] for i in range(core_stack.shape[0])],
                "all_hallucination_flag": labels, "prompt_indices": prompt_ids},
               os.path.join(out_dir, "core_selftest.pt"))
    shard_paths, meta_path = band_mod.save_packed(packed, meta_full, os.path.join(out_dir, "band_selftest.npz"))
    assert os.path.exists(os.path.join(out_dir, "core_selftest.pt"))
    assert all(os.path.exists(p) for p in shard_paths)
    print(f"  [PASS] both outputs written and alignable: core .pt and band .npz/{os.path.basename(meta_path)}")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="truthfulqa")
    parser.add_argument("--gen-seed", type=int, default=0)
    parser.add_argument("--output-suffix", type=str, default="_maxenergy_seeded")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.model_folder:
        print("ERROR: --model_folder required."); sys.exit(1)

    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next((m["id"] for m in cfg["models"] if m["folder"] == args.model_folder), None)
    if model_id is None:
        print(f"ERROR: unknown model folder {args.model_folder}"); sys.exit(1)
    ds_cfg = next((d for d in cfg["datasets"] if d["name"] == args.dataset), None)
    if ds_cfg is None:
        print(f"ERROR: unknown dataset {args.dataset}"); sys.exit(1)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    random.seed(args.gen_seed)
    np.random.seed(args.gen_seed)
    torch.manual_seed(args.gen_seed)
    torch.cuda.manual_seed_all(args.gen_seed)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import evaluate

    ds = load_dataset(ds_cfg["hf_path"], ds_cfg["hf_config"], split="validation")
    samples = []
    if args.dataset == "truthfulqa":
        for ex in ds:
            samples.append({"prompt_text": ds_cfg["prompt_template"].format(question=ex["question"]),
                             "correct_answers": [str(ex["best_answer"])],
                             "incorrect_answers": [str(a) for a in ex["incorrect_answers"]]})
    else:
        raise NotImplementedError(f"dataset {args.dataset} not wired up in this script yet")
    print(f"Dataset: {args.dataset}, prompts: {len(samples)}")

    rouge = evaluate.load("rouge")
    bleurt = evaluate.load("bleurt", config_name="BLEURT-20")

    print(f"Loading: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                  device_map=device, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    D = model.config.hidden_size
    print(f"  Hidden: {D}  Window: {W_START}:{W_END}")

    print("[A1] Computing bases from lm_head.weight (fp32 SVD) ...")
    V_band, V_rand, spectrum = band_mod.compute_bases(model)
    V_band, V_rand = V_band.to(device), V_rand.to(device)
    print(f"  sigma_1={spectrum['sigma_1']:.4f}  band_energy_mass={spectrum['band_energy_mass']:.6f}")

    print("[A2] Post-norm verification ...")
    route, agreement = band_mod.verify_post_norm_route(
        model, tokenizer, ["The capital of France is", "Water boils at a temperature of"], device)
    print(f"  route={route}  agreement={agreement:.4f}")
    apply_norm_manually = (route == "manual_norm")
    norm_fn = model.model.norm

    eos_strs = [".", "!", "?", ".\n", "!\n", "?\n", "\n", "\n\n"]
    eos_ids = {tokenizer.eos_token_id}
    for s in eos_strs:
        eos_ids.update(tokenizer.encode(s, add_special_tokens=False))
        eos_ids.update(tokenizer.encode("Yes" + s, add_special_tokens=False)[1:])

    gen_cfg = cfg["generation"]
    all_emb, all_flags, all_is_known, all_prompt_idx = [], [], [], []
    per_beam_band = []
    n_empty = 0
    t0 = time.time()

    for idx, sample in enumerate(samples):
        prompt_text = sample["prompt_text"]
        correct, wrong = sample["correct_answers"], sample["incorrect_answers"]
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        # this transformers version's generate() has no generator= kwarg (it's not part
        # of its public API here -- _validate_model_kwargs rejects it); seed the global
        # RNG deterministically per-prompt instead, which gives the same reproducibility.
        prompt_seed = args.gen_seed * 100003 + idx
        torch.manual_seed(prompt_seed)
        torch.cuda.manual_seed_all(prompt_seed)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=gen_cfg["max_new_tokens"], eos_token_id=list(eos_ids),
                do_sample=gen_cfg["do_sample"], temperature=gen_cfg["temperature"],
                top_k=gen_cfg["top_k"], top_p=gen_cfg["top_p"],
                num_beams=gen_cfg["num_beams"], num_return_sequences=gen_cfg["num_return_sequences"],
                output_hidden_states=True, return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id, early_stopping=True)

        hidden_states = outputs.hidden_states
        gen_ids_full = outputs.sequences[:, prompt_len:]
        num_beams_here = gen_ids_full.shape[0]

        gen_ids_trimmed, texts, any_correct = [], [], False
        prompt_flags = []
        for b in range(num_beams_here):
            gids = gen_ids_full[b]
            gids = gids[gids != tokenizer.eos_token_id]
            gen_ids_trimmed.append(gids)
            gen_text = tokenizer.decode(gids, skip_special_tokens=True).strip()
            texts.append(gen_text)

            r = rouge.compute(predictions=[gen_text] * len(correct), references=correct) if correct else {"rougeL": 0.0}
            rl = r["rougeL"]
            all_refs = correct + wrong
            bs = bleurt.compute(predictions=[gen_text] * len(all_refs), references=all_refs)
            max_correct_b = max(bs["scores"][:len(correct)], default=0)
            is_correct = (rl >= 0.7) or (max_correct_b > 0.5)
            if is_correct:
                any_correct = True
            prompt_flags.append(not is_correct)

        results = pool_and_project_prompt(hidden_states, gen_ids_trimmed, num_beams_here, D,
                                           V_band, V_rand, apply_norm_manually, norm_fn)
        for b, (H_pooled, z_band, z_rand, rms) in enumerate(results):
            all_emb.append(H_pooled)
            per_beam_band.append((z_band, z_rand, rms))
            if z_band.shape[0] == 0:
                n_empty += 1

        all_flags.extend(prompt_flags)
        all_is_known.append(any_correct)
        all_prompt_idx.extend([idx] * num_beams_here)

        if idx < 5:
            print(f"  [sample] prompt {idx}: beam0 T={len(gen_ids_trimmed[0])} tokens -> "
                  f"{texts[0][:80]!r}")
        if idx % 50 == 0:
            print(f"  {idx}/{len(samples)}  ({time.time()-t0:.0f}s elapsed)")

        del outputs, hidden_states
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nEmpty completions: {n_empty}/{len(all_emb)}")

    # -- save pooled core, same schema as 21_generate_maxpool_datasets.py --
    data_dir = cfg["output"]["data_dir"]
    out_dir = os.path.join(data_dir, args.model_folder)
    os.makedirs(out_dir, exist_ok=True)
    core_path = os.path.join(out_dir, f"{args.dataset}_pooled{args.output_suffix}.pt")
    torch.save({"all_emb": all_emb, "all_hallucination_flag": all_flags,
                "all_is_known": all_is_known, "prompt_indices": all_prompt_idx}, core_path)
    core_size_gb = os.path.getsize(core_path) / 1e9
    print(f"Saved core: {core_path}  ({core_size_gb:.2f} GB)")

    # -- save band streams, same schema as 27_extract_band.py --
    meta = {"checkpoint_id": model_id, "dtype_route": route, "seed": args.gen_seed,
            "git_commit": band_mod.git_commit_hash(), "spectrum": spectrum,
            "n_beams_total": len(all_emb)}
    packed, meta_full = band_mod.pack_beams(per_beam_band, all_prompt_idx, list(range(len(all_emb))),
                                             all_flags, meta)
    band_path = os.path.join(out_dir, f"{args.dataset}_band.npz")
    shard_paths, meta_path = band_mod.save_packed(packed, meta_full, band_path)
    band_size_gb = sum(os.path.getsize(p) for p in shard_paths) / 1e9
    print(f"Saved band: {shard_paths}  ({band_size_gb:.2f} GB combined)")
    print(f"Metadata: {meta_path}")
    print(f"\nTotal on-disk size: {core_size_gb + band_size_gb:.2f} GB "
          f"(vs. ~50+ GB a literal all-layers unpooled dump would have cost)")


if __name__ == "__main__":
    main()
