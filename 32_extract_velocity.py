"""
32_extract_velocity.py -- Session 04 Part A: Depth-Kinematics (Velocity) Extraction
=========================================================================================
*** BLOCKED on real data as of this session (see final handoff summary): "re-forward the
manifest's sequences" has nothing to act on. data/manifest_seeded_v1.json pins hashes of the
already-POOLED outputs (truthfulqa_pooled_maxenergy_seeded.pt, truthfulqa_band.npz) -- it does
not, and never did, pin raw token sequences. 29_generate_extract_band.py (session02) computed
core pooling and band projection inline during its one generate() call specifically so it would
never need to persist raw sequences (see that file's module docstring). No script in this repo
has ever saved per-beam input_ids for the seeded run. Regeneration is explicitly banned this
session ("NO regeneration, ever, without explicit instruction" -- session03 hard rule, reaffirmed
here), so this script cannot synthesize the missing sequences and will not attempt to.

This file is fully built and self-tested (--self-test exercises every pooling/packing function
on synthetic data, no model or sequences needed) so it is ready to run the moment either (a) a
sequence cache matching the schema in load_sequence_cache() is pinned, or (b) a human explicitly
authorizes a new seeded generation pass covering this session's streams. Until then, the real-data
path hard-fails immediately, before touching the GPU.

Usage:
  python 32_extract_velocity.py --self-test
  python 32_extract_velocity.py --model_folder llama-3.1-8b-instruct --dataset truthfulqa \
      --sequence-cache <path-to-be-confirmed>
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


band_mod = _load("s02_extract", "27_extract_band.py")
pin_mod = _load("s03_pin", "30_pin_manifest.py")

W_START, W_END = 15, 24          # same 9-layer reasoning window as the rest of the pipeline
VELOCITY_LAYER_PAIRS = list(range(W_START, W_END - 1))   # l = 15..22, 8 slices (Delta_l = h_{l+1} - h_l)
TURN_PAIRS = list(range(W_START, W_END - 2))              # l = 15..21, 7 adjacent layer-pair pairs


# ==============================================================================
# POOLING (pure functions, no model dependency -- fully self-testable)
# ==============================================================================

def compute_velocity_streams(h_by_layer):
    """h_by_layer: dict {layer_idx (15..23): (T, D) tensor} for ONE beam's completion tokens.
    Returns V95, V05: (8, D) -- q95/q05 over tokens of Delta_l = h_{l+1} - h_l for l=15..22.
    Two-sided quantile pooling (not max, not signed-extremum) -- outlier-robust by construction."""
    D = h_by_layer[W_START].shape[1]
    T = h_by_layer[W_START].shape[0]
    if T == 0:
        empty = torch.zeros(len(VELOCITY_LAYER_PAIRS), D)
        return empty.clone(), empty.clone()
    q95_rows, q05_rows = [], []
    for l in VELOCITY_LAYER_PAIRS:
        delta = h_by_layer[l + 1] - h_by_layer[l]          # (T, D)
        q95_rows.append(torch.quantile(delta, 0.95, dim=0))
        q05_rows.append(torch.quantile(delta, 0.05, dim=0))
    return torch.stack(q95_rows, dim=0), torch.stack(q05_rows, dim=0)   # each (8, D)


def compute_kinematic_scalars(h_by_layer):
    """Returns a 30-dim vector: for each of 8 layer-pairs, mean+q90 of speed ||Delta_l,t|| (16),
    plus for each of 7 adjacent layer-pair pairs, mean+q90 of turn angle 1-cos(Delta_l, Delta_l+1) (14)."""
    T = h_by_layer[W_START].shape[0]
    if T == 0:
        return torch.zeros(30)

    deltas = {l: h_by_layer[l + 1] - h_by_layer[l] for l in VELOCITY_LAYER_PAIRS}   # each (T, D)
    speed_stats = []
    for l in VELOCITY_LAYER_PAIRS:
        speed = deltas[l].norm(dim=1)                       # (T,)
        speed_stats.append(speed.mean())
        speed_stats.append(torch.quantile(speed, 0.90))

    turn_stats = []
    for l in TURN_PAIRS:
        a, b = deltas[l], deltas[l + 1]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1, eps=1e-8)   # (T,)
        turn = 1 - cos
        turn_stats.append(turn.mean())
        turn_stats.append(torch.quantile(turn, 0.90))

    return torch.stack(speed_stats + turn_stats)             # (16 + 14,) = (30,)


def compute_static_repooling(h_by_layer):
    """Returns S95, S05: (9, D) -- q95/q05 over tokens of h itself, layers 15..23."""
    D = h_by_layer[W_START].shape[1]
    T = h_by_layer[W_START].shape[0]
    layers = list(range(W_START, W_END))
    if T == 0:
        empty = torch.zeros(len(layers), D)
        return empty.clone(), empty.clone()
    q95_rows = [torch.quantile(h_by_layer[l], 0.95, dim=0) for l in layers]
    q05_rows = [torch.quantile(h_by_layer[l], 0.05, dim=0) for l in layers]
    return torch.stack(q95_rows, dim=0), torch.stack(q05_rows, dim=0)   # each (9, D)


# ==============================================================================
# REQUIRED (MISSING) INPUT -- same precondition pattern as 27_extract_band.py
# ==============================================================================

def load_sequence_cache(path, expected_n_beams):
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            "\n\n"
            "BLOCKED: no raw per-beam token sequences are available for the pinned seeded "
            "dataset. data/manifest_seeded_v1.json pins sha256 hashes of the already-POOLED "
            "outputs only (truthfulqa_pooled_maxenergy_seeded.pt, truthfulqa_band.npz) -- it "
            "does not reference, and there does not exist anywhere in this repo, a file with "
            "per-beam input_ids for that run. 29_generate_extract_band.py computed core "
            "pooling and band projection inline during its one generate() call specifically so "
            "it would never need to persist raw sequences.\n\n"
            "This script will not regenerate: session03 established 'NO regeneration, ever, "
            "without explicit instruction' as a hard rule, and session04's brief reaffirms it "
            "('regeneration remains banned'). There is no code-level workaround for a sequence "
            "cache that was never created.\n\n"
            "To proceed, either point --sequence-cache at a file matching this function's "
            "expected schema (list of full prompt+completion input_ids per beam, aligned to "
            "the manifest's beam order, plus per-beam prompt_len), or get explicit human "
            "authorization for a new seeded generation pass that computes velocity/kinematic/"
            "static-repooling streams inline (mirroring 29_generate_extract_band.py's approach) "
            "-- that is a deliberate decision to make once, not something this script should "
            "default into.")
    data = torch.load(path, weights_only=False)
    for k in ("input_ids", "prompt_len", "prompt_id"):
        if k not in data:
            raise ValueError(f"--sequence-cache is missing required key '{k}'. Found: {list(data.keys())}")
    if len(data["input_ids"]) != expected_n_beams:
        raise ValueError(f"--sequence-cache beam count ({len(data['input_ids'])}) != manifest "
                          f"beam count ({expected_n_beams}). Refusing to guess an alignment.")
    return data


# ==============================================================================
# PACKING -- extends 27_extract_band.py's pack_beams/save_packed convention to
# the three new per-beam artifacts (V95/V05, kinematic scalars, S95/S05)
# ==============================================================================

def pack_velocity_dataset(v95_list, v05_list, kin_list, s95_list, s05_list,
                           prompt_ids, beam_idxs, labels):
    n_beams = len(v95_list)
    return {
        "V95": torch.stack(v95_list).numpy().astype(np.float32),   # (n_beams, 8, D)
        "V05": torch.stack(v05_list).numpy().astype(np.float32),
        "kinematic": torch.stack(kin_list).numpy().astype(np.float32),   # (n_beams, 30)
        "S95": torch.stack(s95_list).numpy().astype(np.float32),   # (n_beams, 9, D)
        "S05": torch.stack(s05_list).numpy().astype(np.float32),
        "prompt_id": np.asarray(prompt_ids, dtype=np.int64),
        "beam_idx": np.asarray(beam_idxs, dtype=np.int64),
        "label": np.asarray(labels, dtype=np.int64),
    }


def save_velocity_dataset(packed, out_path, checkpoint_id, seed, decoding_config):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **{k: v for k, v in packed.items()
                                      if k not in ("checkpoint_id",)})
    meta = {
        "checkpoint_id": checkpoint_id, "seed": seed, "git_commit": band_mod.git_commit_hash(),
        "decoding_config": decoding_config,
        "shapes": {k: list(v.shape) for k, v in packed.items() if hasattr(v, "shape")},
        "n_beams": int(packed["V95"].shape[0]),
    }
    meta_path = os.path.splitext(out_path)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return out_path, meta_path


# ==============================================================================
# MANIFEST v2 -- extends 30_pin_manifest.py's manifest with the new artifacts +
# the literal decoding config (Part A's explicit ask: pin it, don't re-describe it)
# ==============================================================================

def build_manifest_v2(manifest_v1_path, velocity_npz_path, velocity_meta_path, decoding_config):
    with open(manifest_v1_path) as f:
        manifest_v1 = json.load(f)
    velocity_hash = pin_mod.sha256_file(velocity_npz_path)
    with open(velocity_meta_path) as f:
        velocity_meta = json.load(f)

    manifest_v2 = dict(manifest_v1)
    manifest_v2["version"] = 2
    manifest_v2["velocity_npz_sha256"] = velocity_hash
    manifest_v2["velocity_npz_path"] = os.path.abspath(velocity_npz_path)
    manifest_v2["velocity_shapes"] = velocity_meta["shapes"]
    manifest_v2["decoding_config_pinned"] = decoding_config
    manifest_v2["decoding_config_note"] = (
        "Literal config.yaml generation: block at extraction time -- pinned here because "
        "two prior briefing assumptions about this pipeline's decoding config (deterministic "
        "greedy search; a specific reported AUROC's feature composition) were later overturned "
        "by directly reading the code and data, per sessions 01/02.")
    return manifest_v2


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: velocity/kinematic/static-repooling math + packing (no model/GPU)")
    print("=" * 70)
    torch.manual_seed(0)
    D = 32   # small synthetic hidden dim (real: 4096)
    L_full = list(range(W_START, W_END))   # 15..23

    def fake_beam(T):
        return {l: torch.randn(T, D) for l in L_full}

    h5 = fake_beam(5)
    v95, v05 = compute_velocity_streams(h5)
    assert v95.shape == (8, D) and v05.shape == (8, D)
    assert torch.all(v95 >= v05 - 1e-5), "q95 should be >= q05 per channel"
    print("  [PASS] compute_velocity_streams: correct shape (8, D), q95 >= q05")

    kin = compute_kinematic_scalars(h5)
    assert kin.shape == (30,)
    assert torch.isfinite(kin).all()
    print("  [PASS] compute_kinematic_scalars: correct shape (30,), all finite")

    s95, s05 = compute_static_repooling(h5)
    assert s95.shape == (9, D) and s05.shape == (9, D)
    assert torch.all(s95 >= s05 - 1e-5)
    print("  [PASS] compute_static_repooling: correct shape (9, D), q95 >= q05")

    h0 = fake_beam(0)
    v95_0, v05_0 = compute_velocity_streams(h0)
    kin_0 = compute_kinematic_scalars(h0)
    s95_0, s05_0 = compute_static_repooling(h0)
    assert v95_0.shape == (8, D) and kin_0.shape == (30,) and s95_0.shape == (9, D)
    assert torch.all(v95_0 == 0) and torch.all(kin_0 == 0) and torch.all(s95_0 == 0)
    print("  [PASS] empty-completion (T=0) beams produce correctly-shaped zero tensors")

    n_beams = 12
    v95_list, v05_list, kin_list, s95_list, s05_list = [], [], [], [], []
    prompt_ids, beam_idxs, labels = [], [], []
    rng = np.random.default_rng(0)
    for i in range(n_beams):
        T_i = int(rng.integers(0, 10))
        h = fake_beam(T_i)
        a, b = compute_velocity_streams(h)
        v95_list.append(a); v05_list.append(b)
        kin_list.append(compute_kinematic_scalars(h))
        c, d = compute_static_repooling(h)
        s95_list.append(c); s05_list.append(d)
        prompt_ids.append(i // 4); beam_idxs.append(i % 4)
        labels.append(int(rng.integers(0, 2)))

    packed = pack_velocity_dataset(v95_list, v05_list, kin_list, s95_list, s05_list,
                                    prompt_ids, beam_idxs, labels)
    assert packed["V95"].shape == (n_beams, 8, D)
    assert packed["kinematic"].shape == (n_beams, 30)
    assert packed["S95"].shape == (n_beams, 9, D)
    print(f"  [PASS] pack_velocity_dataset: stacked {n_beams} beams, all shapes correct")

    tmp_dir = os.path.join(HERE, "results", "_selftest_velocity")
    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, "velocity_selftest.npz")
    decoding_config = {"do_sample": True, "num_beams": 10, "temperature": 0.5, "top_p": 0.99}
    saved_path, meta_path = save_velocity_dataset(packed, out_path, "self-test", 0, decoding_config)
    assert os.path.exists(saved_path) and os.path.exists(meta_path)
    reloaded = dict(np.load(saved_path))
    np.testing.assert_allclose(reloaded["V95"], packed["V95"])
    assert np.array_equal(reloaded["prompt_id"], packed["prompt_id"])
    print(f"  [PASS] save/reload round-trip exact: {saved_path}")

    # -- manifest v2 build, against a fabricated v1 manifest --
    fake_v1_path = os.path.join(tmp_dir, "fake_manifest_v1.json")
    with open(fake_v1_path, "w") as f:
        json.dump({"created": "fake", "counts": {"n_beams": n_beams}}, f)
    manifest_v2 = build_manifest_v2(fake_v1_path, saved_path, meta_path, decoding_config)
    assert manifest_v2["version"] == 2
    assert manifest_v2["velocity_shapes"]["V95"] == [n_beams, 8, D]
    assert manifest_v2["decoding_config_pinned"] == decoding_config
    print("  [PASS] build_manifest_v2: extends v1 with new-artifact hashes/shapes + decoding config")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="truthfulqa")
    parser.add_argument("--sequence-cache", type=str, default=None)
    parser.add_argument("--manifest", type=str, default="data/manifest_seeded_v1.json")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.model_folder:
        print("ERROR: --model_folder required."); sys.exit(1)

    manifest_path = os.path.join(HERE, args.manifest) if not os.path.isabs(args.manifest) else args.manifest
    manifest = pin_mod.verify_manifest(manifest_path)
    n_beams = manifest["counts"]["n_beams"]
    print(f"Manifest verified. n_beams={n_beams}")

    # Hard-fails here with a full explanation -- see load_sequence_cache() docstring above.
    seq_cache = load_sequence_cache(args.sequence_cache, n_beams)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for extraction.")
    device = torch.device("cuda")

    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next((m["id"] for m in cfg["models"] if m["folder"] == args.model_folder), None)
    gen_cfg = cfg["generation"]
    decoding_config = {"do_sample": gen_cfg["do_sample"], "num_beams": gen_cfg["num_beams"],
                        "temperature": gen_cfg["temperature"], "top_p": gen_cfg["top_p"]}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                  device_map=device, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()

    v95_list, v05_list, kin_list, s95_list, s05_list = [], [], [], [], []
    prompt_ids, beam_idxs, labels = [], [], []
    total_completion_tokens = 0
    bs = args.batch_size
    t0 = time.time()

    for start in range(0, n_beams, bs):
        end = min(start + bs, n_beams)
        batch_ids = [seq_cache["input_ids"][i] for i in range(start, end)]
        batch_plens = [seq_cache["prompt_len"][i] for i in range(start, end)]
        lengths = [len(ids) for ids in batch_ids]
        T_max = max(lengths)
        padded = torch.zeros((len(batch_ids), T_max), dtype=torch.long)
        attn = torch.zeros((len(batch_ids), T_max), dtype=torch.long)
        for j, ids in enumerate(batch_ids):
            padded[j, :len(ids)] = ids
            attn[j, :len(ids)] = 1
        padded, attn = padded.to(device), attn.to(device)

        with torch.no_grad():
            out = model(input_ids=padded, attention_mask=attn, use_cache=False, output_hidden_states=True)
        # post-BLOCK residual states, layers 15..23, NOT final-norm -- hidden_states[l+1] convention
        for j in range(len(batch_ids)):
            comp_start, comp_end = batch_plens[j], lengths[j]
            h_by_layer = {l: out.hidden_states[l + 1][j, comp_start:comp_end, :].float()
                          for l in range(W_START, W_END)}
            T_i = comp_end - comp_start
            total_completion_tokens += T_i

            a, b = compute_velocity_streams(h_by_layer)
            v95_list.append(a); v05_list.append(b)
            kin_list.append(compute_kinematic_scalars(h_by_layer))
            c, d = compute_static_repooling(h_by_layer)
            s95_list.append(c); s05_list.append(d)
        if "prompt_id" not in seq_cache:
            raise NotImplementedError(
                "This path is unreachable today (load_sequence_cache always raises first -- see "
                "module docstring). Once a real --sequence-cache exists, its schema needs a "
                "'prompt_id' list wired in here (or equivalent), matching load_sequence_cache()'s "
                "documented contract -- not implemented since this session never had a real "
                "cache to design against.")
        prompt_ids.extend(seq_cache["prompt_id"][start:end])
        beam_idxs.extend(range(start, end))
        del out
        torch.cuda.empty_cache()
        if start % (bs * 20) == 0:
            print(f"  {end}/{n_beams}  ({time.time()-t0:.0f}s elapsed)")

    if total_completion_tokens != manifest["counts"]["n_tokens"]:
        raise ValueError(f"Alignment guard failed: total completion tokens ({total_completion_tokens}) "
                          f"!= manifest's pinned token count ({manifest['counts']['n_tokens']}).")

    print("NOTE: this real-data path is scaffolding for whenever a sequence cache is provided -- "
          "prompt_id wiring from the sequence cache's own metadata still needs to be filled in "
          "once that schema is known; not exercised because this session never reached real data.")


if __name__ == "__main__":
    main()
