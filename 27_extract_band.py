"""
27_extract_band.py -- Session 02: Reasoning-Band Token Stream Extraction
=============================================================================
GPU-only. For every beam behind the Phase-1 baseline (the 8170 beams in
{dataset}_pooled_maxenergy.pt), re-forwards the full prompt+completion
sequence, takes final-layer post-norm states, projects each completion
token onto (a) the trailing right-singular subspace of the unembedding
matrix ("reasoning band") and (b) a random orthogonal control basis of the
same rank, and saves per-token projected streams keyed back to the same
beam/prompt ids and labels used in session 01.

*** BLOCKED on real data as of session01's audit (see reports/session01_repo_audit.md,
section A1): the current pipeline does not persist raw generated token
sequences or per-beam prompt lengths anywhere -- only pooled activations,
labels, and prompt_indices survive. This script therefore REFUSES to run
its real-data path unless given --sequence-cache pointing at a file with
that information (see load_sequence_cache() below for the required
schema). It will not regenerate sequences: do_sample=True with no fixed
seed anywhere in the generation code means a fresh run would not reproduce
the exact beams the cached labels describe. See A1 in the audit report.

Usage:
  python 27_extract_band.py --self-test
  python 27_extract_band.py --model_folder llama-3.1-8b-instruct --dataset truthfulqa \
      --sequence-cache <path-to-be-confirmed> --output ../data_unpooled/llama-3.1-8b-instruct/truthfulqa_band.npz
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

BAND_RANK = 320
RAND_RANK = 256
SEED = 0
SHARD_BYTES_LIMIT = int(1.8 * 1e9)   # split into shards if packed arrays exceed this


# ==============================================================================
# A1 -- BASES
# ==============================================================================

def compute_bases_from_matrix(W: torch.Tensor, band_rank=BAND_RANK, rand_rank=RAND_RANK, seed=SEED):
    """W: [V, D] unembedding-like matrix (or any matrix for self-test). fp32.
    Returns V_band [band_rank, D], V_rand [rand_rank, D], spectrum summary."""
    W = W.float()
    _, S, Vh = torch.linalg.svd(W, full_matrices=False)   # Vh: [min(V,D), D]
    D = W.shape[1]
    n_sv = S.shape[0]
    br = min(band_rank, n_sv)
    V_band = Vh[-br:]                                      # [br, D] -- trailing (smallest-sigma) directions

    g = torch.Generator().manual_seed(seed)
    A = torch.randn(rand_rank, D, generator=g)
    Q, _ = torch.linalg.qr(A.T)                             # Q: [D, rand_rank], orthonormal columns
    V_rand = Q.T                                             # [rand_rank, D]

    band_energy_mass = float((S[-br:] ** 2).sum() / (S ** 2).sum())
    spectrum = {
        "sigma_1": float(S[0]),
        f"sigma_-{br}": float(S[-br]),
        "band_energy_mass": band_energy_mass,
        "n_singular_values": int(n_sv),
        "band_rank_used": int(br),
    }
    return V_band, V_rand, spectrum


def compute_bases(model, band_rank=BAND_RANK, rand_rank=RAND_RANK, seed=SEED):
    W = model.lm_head.weight.data
    return compute_bases_from_matrix(W, band_rank, rand_rank, seed)


# ==============================================================================
# A2 -- POST-NORM VERIFICATION
# ==============================================================================

def verify_post_norm_route(model, tokenizer, sample_texts, device):
    """Returns (route, agreement_frac). route in {"hidden_states[-1]", "manual_norm"}.
    Hard-fails (raises) if neither route reaches >= 99.9% top-1 agreement."""
    inputs = tokenizer(sample_texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    logits_model = out.logits.float()
    h_last = out.hidden_states[-1].float()
    mask = inputs["attention_mask"].bool()

    # nn.Linear requires input/weight dtypes to match exactly (unlike elementwise ops,
    # which promote automatically) -- model.lm_head's weight is bf16 since the model is
    # loaded in bf16, so compute the "in fp32" check via an explicit fp32-cast weight
    # rather than calling the bf16 module directly on an fp32 input.
    lm_head_w = model.lm_head.weight.float()
    lm_head_b = model.lm_head.bias.float() if model.lm_head.bias is not None else None

    def lm_head_fp32(h):
        return torch.nn.functional.linear(h, lm_head_w, lm_head_b)

    def agreement(logits_check):
        pred_check = logits_check.argmax(dim=-1)
        pred_model = logits_model.argmax(dim=-1)
        agree = (pred_check == pred_model) & mask
        return float(agree.sum()) / float(mask.sum())

    logits_direct = lm_head_fp32(h_last)
    frac_direct = agreement(logits_direct)
    if frac_direct >= 0.999:
        return "hidden_states[-1]", frac_direct

    h_normed = model.model.norm(h_last)
    logits_normed = lm_head_fp32(h_normed)
    frac_normed = agreement(logits_normed)
    if frac_normed >= 0.999:
        return "manual_norm", frac_normed

    raise RuntimeError(
        f"A2 post-norm verification FAILED both routes: direct hidden_states[-1] "
        f"agreement={frac_direct:.4f}, manual model.model.norm(...) agreement={frac_normed:.4f}. "
        f"Neither reaches the required 0.999 top-1 threshold -- refusing to extract with an "
        f"unverified projection basis. Inspect this transformers version's generate() hidden-state "
        f"convention before proceeding.")


# ==============================================================================
# REQUIRED (MISSING) INPUT -- raw sequences + prompt lengths for the cached beams
# ==============================================================================

def load_sequence_cache(path, expected_n_beams):
    """Required schema (not currently produced by any script in this repo -- see
    module docstring and reports/session01_repo_audit.md A1):
      {
        "input_ids":   list[LongTensor], len == expected_n_beams, one full
                       prompt+completion token sequence per beam, in the SAME
                       order as prompt_indices in {dataset}_pooled_maxenergy.pt
        "prompt_len":  list[int], len == expected_n_beams
      }
    Hard-fails with an explanatory message rather than guessing an alignment or
    re-tokenizing prompt text (which would not guarantee an exact match and is
    explicitly disallowed by this session's guardrails).
    """
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            "\n\n"
            "BLOCKED: no raw per-beam token sequences / prompt lengths are available.\n"
            "Session 01's audit (reports/session01_repo_audit.md, A1) established that the\n"
            "current pipeline does NOT persist generated token sequences or prompt lengths\n"
            "anywhere -- only pooled activations (all_emb), labels (all_hallucination_flag),\n"
            "and prompt_indices survive in {dataset}_pooled_maxenergy.pt. The one script whose\n"
            "name suggested otherwise (20a_pilot_raw_tokens.py) is an independent 100-prompt\n"
            "pilot run with its own do_sample=True generation call -- it does not cover the\n"
            "8170 beams behind the Phase-1 baseline, and does not save raw token ids either\n"
            "(it saves hidden states directly).\n\n"
            "This script refuses to regenerate sequences: do_sample=True with no fixed seed\n"
            "anywhere in the generation code means a fresh run will not reproduce the exact\n"
            "beams the cached labels describe (see A1's reproducibility note).\n\n"
            "To proceed: point --sequence-cache at a file matching the schema in this "
            "function's docstring -- most likely something that would need to live in "
            "data_unpooled/ on the cluster (gitignored, never synced locally) IF it was "
            "retained from the original generation run. If no such artifact exists anywhere, "
            "extracting the reasoning-band stream for these specific 8170 beams is not "
            "possible without redoing generation+labeling from scratch, which is out of "
            "scope for an additive session02 script.")
    data = torch.load(path, weights_only=False)
    for k in ("input_ids", "prompt_len"):
        if k not in data:
            raise ValueError(f"--sequence-cache is missing required key '{k}'. Found: {list(data.keys())}")
    if len(data["input_ids"]) != expected_n_beams or len(data["prompt_len"]) != expected_n_beams:
        raise ValueError(
            f"--sequence-cache beam count mismatch: input_ids={len(data['input_ids'])}, "
            f"prompt_len={len(data['prompt_len'])}, expected={expected_n_beams} "
            f"(from {{dataset}}_pooled_maxenergy.pt). Refusing to guess an alignment.")
    return data


# ==============================================================================
# A3 -- EXTRACTION (per beam, batched)
# ==============================================================================

@torch.no_grad()
def extract_batch(model, input_ids_list, prompt_lens, V_band, V_rand, apply_norm_manually, device):
    """input_ids_list: list of 1D LongTensors (full prompt+completion, variable length).
    Right-pads into a batch, forwards with use_cache=False, slices completion tokens
    per-beam using each beam's own prompt_len and true (unpadded) length.
    Returns list of (z_band [T_i,320], z_rand [T_i,256], rms [T_i]) fp32 numpy arrays,
    one per beam in the input order."""
    B = len(input_ids_list)
    lengths = [len(ids) for ids in input_ids_list]
    T_max = max(lengths)
    pad_id = 0
    padded = torch.full((B, T_max), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T_max), dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        padded[i, :len(ids)] = ids
        attn[i, :len(ids)] = 1
    padded, attn = padded.to(device), attn.to(device)

    out = model(input_ids=padded, attention_mask=attn, use_cache=False, output_hidden_states=True)
    h = out.hidden_states[-1].float()
    if apply_norm_manually:
        h = model.model.norm(h)

    results = []
    for i in range(B):
        full_len = lengths[i]
        p_len = prompt_lens[i]
        comp_start, comp_end = p_len, full_len
        if comp_end <= comp_start:
            results.append((np.zeros((0, V_band.shape[0]), dtype=np.float32),
                             np.zeros((0, V_rand.shape[0]), dtype=np.float32),
                             np.zeros((0,), dtype=np.float32)))
            continue
        h_comp = h[i, comp_start:comp_end, :]                    # (T_i, D)
        z_band = (h_comp @ V_band.T.to(h_comp.dtype)).cpu().numpy().astype(np.float32)
        z_rand = (h_comp @ V_rand.T.to(h_comp.dtype)).cpu().numpy().astype(np.float32)
        rms = torch.sqrt((h_comp.pow(2)).mean(dim=-1)).cpu().numpy().astype(np.float32)
        results.append((z_band, z_rand, rms))
    return results


# ==============================================================================
# A4 -- PACKING / OUTPUT FORMAT
# ==============================================================================

def pack_beams(per_beam, prompt_ids, beam_idxs, labels, extraction_meta):
    """per_beam: list of (z_band [T_i,320], z_rand [T_i,256], rms [T_i]).
    Builds concatenated token arrays + int64 offset arrays (CSR-style: beam i's
    tokens are rows offsets[i]:offsets[i+1]) + per-beam metadata."""
    n_beams = len(per_beam)
    offsets = np.zeros(n_beams + 1, dtype=np.int64)
    for i, (zb, zr, rms) in enumerate(per_beam):
        offsets[i + 1] = offsets[i] + zb.shape[0]

    total_T = int(offsets[-1])
    band_dim = per_beam[0][0].shape[1] if n_beams > 0 else BAND_RANK
    rand_dim = per_beam[0][1].shape[1] if n_beams > 0 else RAND_RANK

    band_all = np.zeros((total_T, band_dim), dtype=np.float32)
    rand_all = np.zeros((total_T, rand_dim), dtype=np.float32)
    rms_all = np.zeros((total_T,), dtype=np.float32)
    for i, (zb, zr, rms) in enumerate(per_beam):
        s, e = offsets[i], offsets[i + 1]
        band_all[s:e] = zb
        rand_all[s:e] = zr
        rms_all[s:e] = rms

    n_empty = int(sum(1 for zb, _, _ in per_beam if zb.shape[0] == 0))

    packed = {
        "z_band": band_all, "z_rand": rand_all, "rms": rms_all,
        "offsets": offsets,
        "prompt_id": np.asarray(prompt_ids, dtype=np.int64),
        "beam_idx": np.asarray(beam_idxs, dtype=np.int64),
        "label": np.asarray(labels, dtype=np.int64),
    }
    meta = dict(extraction_meta)
    meta["n_beams"] = n_beams
    meta["n_empty_completions"] = n_empty
    meta["total_tokens"] = total_T
    return packed, meta


def save_packed(packed, meta, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nbytes = sum(a.nbytes for k, a in packed.items())
    meta_path = os.path.splitext(out_path)[0] + "_meta.json"

    if nbytes <= SHARD_BYTES_LIMIT:
        np.savez_compressed(out_path, **packed)
        meta["shards"] = [os.path.basename(out_path)]
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return [out_path], meta_path

    # shard by beam index range
    n_beams = len(packed["prompt_id"])
    offsets = packed["offsets"]
    approx_shard_beams = max(1, int(n_beams * SHARD_BYTES_LIMIT / max(nbytes, 1)))
    shard_paths = []
    base, ext = os.path.splitext(out_path)
    b0 = 0
    shard_i = 0
    while b0 < n_beams:
        b1 = min(b0 + approx_shard_beams, n_beams)
        t0, t1 = offsets[b0], offsets[b1]
        shard = {
            "z_band": packed["z_band"][t0:t1], "z_rand": packed["z_rand"][t0:t1],
            "rms": packed["rms"][t0:t1],
            "offsets": offsets[b0:b1 + 1] - offsets[b0],
            "prompt_id": packed["prompt_id"][b0:b1], "beam_idx": packed["beam_idx"][b0:b1],
            "label": packed["label"][b0:b1],
        }
        shard_path = f"{base}_shard{shard_i}{ext}"
        np.savez_compressed(shard_path, **shard)
        shard_paths.append(shard_path)
        b0 = b1
        shard_i += 1

    meta["shards"] = [os.path.basename(p) for p in shard_paths]
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return shard_paths, meta_path


def load_packed(shard_paths):
    """Concatenate shards back into one packed dict with globally-consistent offsets."""
    parts = [dict(np.load(p)) for p in shard_paths]
    if len(parts) == 1:
        return parts[0]
    z_band = np.concatenate([p["z_band"] for p in parts], axis=0)
    z_rand = np.concatenate([p["z_rand"] for p in parts], axis=0)
    rms = np.concatenate([p["rms"] for p in parts], axis=0)
    prompt_id = np.concatenate([p["prompt_id"] for p in parts], axis=0)
    beam_idx = np.concatenate([p["beam_idx"] for p in parts], axis=0)
    label = np.concatenate([p["label"] for p in parts], axis=0)
    offsets = [np.array([0], dtype=np.int64)]
    tok_base = 0
    for p in parts:
        offsets.append(p["offsets"][1:] + tok_base)
        tok_base += p["z_band"].shape[0]
    offsets = np.concatenate(offsets)
    return {"z_band": z_band, "z_rand": z_rand, "rms": rms, "offsets": offsets,
            "prompt_id": prompt_id, "beam_idx": beam_idx, "label": label}


def git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__) or ".",
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ==============================================================================
# A5 -- SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: bases math + packing/unpacking/offset logic (no model)")
    print("=" * 70)
    rng = np.random.default_rng(0)

    # -- bases --
    W = torch.tensor(rng.normal(0, 1, size=(2000, 128)).astype(np.float32))
    V_band, V_rand, spectrum = compute_bases_from_matrix(W, band_rank=32, rand_rank=24, seed=SEED)
    assert V_band.shape == (32, 128) and V_rand.shape == (24, 128)
    orth_band = (V_band @ V_band.T)
    assert torch.allclose(orth_band, torch.eye(32), atol=1e-4), "V_band rows not orthonormal"
    orth_rand = (V_rand @ V_rand.T)
    assert torch.allclose(orth_rand, torch.eye(24), atol=1e-4), "V_rand rows not orthonormal"
    V_rand2, _, _ = compute_bases_from_matrix(W, band_rank=32, rand_rank=24, seed=SEED)[1], None, None
    print("  [PASS] V_band / V_rand orthonormal, correct shapes")

    # determinism of V_rand given fixed seed
    _, V_rand_again, _ = compute_bases_from_matrix(W, band_rank=32, rand_rank=24, seed=SEED)
    assert torch.allclose(V_rand, V_rand_again), "V_rand not deterministic under fixed seed"
    print("  [PASS] V_rand deterministic under seed=0")

    # -- packing / unpacking --
    n_beams = 37
    per_beam = []
    prompt_ids, beam_idxs, labels = [], [], []
    for i in range(n_beams):
        T_i = 0 if i % 9 == 0 else rng.integers(1, 20)   # some empty completions
        zb = rng.normal(0, 1, size=(T_i, 32)).astype(np.float32)
        zr = rng.normal(0, 1, size=(T_i, 24)).astype(np.float32)
        rms = rng.uniform(0.1, 5.0, size=(T_i,)).astype(np.float32)
        per_beam.append((zb, zr, rms))
        prompt_ids.append(i // 5)
        beam_idxs.append(i % 5)
        labels.append(int(rng.integers(0, 2)))

    meta = {"checkpoint_id": "self-test", "dtype_route": "hidden_states[-1]",
            "seed": SEED, "git_commit": git_commit_hash()}
    packed, meta_full = pack_beams(per_beam, prompt_ids, beam_idxs, labels, meta)

    n_empty_expected = sum(1 for zb, _, _ in per_beam if zb.shape[0] == 0)
    assert meta_full["n_empty_completions"] == n_empty_expected
    print(f"  [PASS] packed {n_beams} beams, {n_empty_expected} empty completions flagged correctly")

    # round-trip through disk, including forced sharding
    tmp_dir = os.path.join(os.path.dirname(__file__) or ".", "results", "_selftest_band")
    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, "band_selftest.npz")

    global SHARD_BYTES_LIMIT
    orig_limit = SHARD_BYTES_LIMIT
    SHARD_BYTES_LIMIT = 2000   # force sharding for this test regardless of actual size
    shard_paths, meta_path = save_packed(packed, meta_full, out_path)
    SHARD_BYTES_LIMIT = orig_limit
    assert len(shard_paths) >= 2, "expected forced sharding to produce >=2 shards"
    reloaded = load_packed(shard_paths)

    for i in range(n_beams):
        s, e = packed["offsets"][i], packed["offsets"][i + 1]
        s2, e2 = reloaded["offsets"][i], reloaded["offsets"][i + 1]
        assert (e - s) == (e2 - s2), f"beam {i}: token count mismatch after shard round-trip"
        np.testing.assert_allclose(packed["z_band"][s:e], reloaded["z_band"][s2:e2], atol=1e-6)
        np.testing.assert_allclose(packed["z_rand"][s:e], reloaded["z_rand"][s2:e2], atol=1e-6)
    assert np.array_equal(packed["prompt_id"], reloaded["prompt_id"])
    assert np.array_equal(packed["label"], reloaded["label"])
    print(f"  [PASS] sharded round-trip ({len(shard_paths)} shards): every beam's tokens "
          f"recovered exactly via offsets")

    with open(meta_path) as f:
        meta_check = json.load(f)
    assert meta_check["n_beams"] == n_beams
    print(f"  [PASS] metadata JSON written and readable: {meta_path}")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="truthfulqa")
    parser.add_argument("--sequence-cache", type=str, default=None,
                         help="Required. See load_sequence_cache() docstring for schema. "
                              "No default exists in this repo as of session01's audit.")
    parser.add_argument("--pooled-suffix", type=str, default="_maxenergy")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.model_folder:
        print("ERROR: --model_folder required for real extraction."); sys.exit(1)

    with open(os.path.join(os.path.dirname(__file__) or ".", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next((m["id"] for m in cfg["models"] if m["folder"] == args.model_folder), None)
    if model_id is None:
        print(f"ERROR: unknown model folder {args.model_folder}"); sys.exit(1)

    data_dir = cfg["output"]["data_dir"]
    pooled_path = os.path.join(data_dir, args.model_folder, f"{args.dataset}_pooled{args.pooled_suffix}.pt")
    if not os.path.exists(pooled_path):
        print(f"ERROR: cached pooled file not found: {pooled_path}"); sys.exit(1)
    pooled = torch.load(pooled_path, weights_only=False)
    n_beams = len(pooled["all_emb"])
    prompt_idx = pooled["prompt_indices"]
    labels = [int(f) for f in pooled["all_hallucination_flag"]]

    # This will raise with a full explanation if the sequence cache is missing --
    # see module docstring and reports/session01_repo_audit.md A1.
    seq_cache = load_sequence_cache(args.sequence_cache, n_beams)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for extraction (A1-A3 need the real model).")
    device = torch.device("cuda")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                  device_map=device, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    print("[A1] Computing bases from lm_head.weight (fp32 SVD) ...")
    V_band, V_rand, spectrum = compute_bases(model)
    V_band, V_rand = V_band.to(device), V_rand.to(device)
    print(f"  sigma_1={spectrum['sigma_1']:.4f}  band_energy_mass={spectrum['band_energy_mass']:.6f}")

    print("[A2] Post-norm verification ...")
    sample_texts = ["The capital of France is", "Water boils at a temperature of"]
    route, agreement = verify_post_norm_route(model, tokenizer, sample_texts, device)
    print(f"  route={route}  agreement={agreement:.4f}")
    apply_norm_manually = (route == "manual_norm")

    print("[A3] Extraction loop ...")
    per_beam = [None] * n_beams
    bs = args.batch_size
    t0 = time.time()
    for start in range(0, n_beams, bs):
        end = min(start + bs, n_beams)
        batch_ids = [seq_cache["input_ids"][i] for i in range(start, end)]
        batch_plens = [seq_cache["prompt_len"][i] for i in range(start, end)]
        batch_out = extract_batch(model, batch_ids, batch_plens, V_band, V_rand,
                                   apply_norm_manually, device)
        for j, res in enumerate(batch_out):
            per_beam[start + j] = res
        if (start // bs) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {end}/{n_beams}  ({elapsed:.0f}s elapsed)")

    n_empty = sum(1 for zb, _, _ in per_beam if zb.shape[0] == 0)
    print(f"  Empty completions: {n_empty}/{n_beams}")
    import random
    sample_idx = random.sample(range(n_beams), min(5, n_beams))
    for i in sample_idx:
        T_i = per_beam[i][0].shape[0]
        print(f"  beam {i}: T={T_i} tokens")

    meta = {"checkpoint_id": model_id, "dtype_route": route, "seed": SEED,
            "git_commit": git_commit_hash(), "spectrum": spectrum,
            "n_beams_total": n_beams}
    packed, meta_full = pack_beams(per_beam, prompt_idx, list(range(n_beams)), labels, meta)

    out_path = args.output or os.path.join(data_dir + "_unpooled", args.model_folder,
                                            f"{args.dataset}_band.npz")
    shard_paths, meta_path = save_packed(packed, meta_full, out_path)
    print(f"\nSaved {len(shard_paths)} shard(s): {shard_paths}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
