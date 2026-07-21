"""
35_derive_streams.py -- Session 04 Part 2 (CPU derivation): Velocity/Kinematic/Re-pooling
================================================================================================
CPU-only. Reads the raw-state store produced by 34_gate_reconstruct_or_regenerate.py (Route R
or Route N) and derives, in raw values with NO normalization (scaling happens downstream in the
eval harness):
  2a. Velocity: V95/V05 -- q95/q05 pooling of inter-layer deltas, layers 15..22 (8 pairs).
  2b. Kinematic scalars: 30-dim per-beam profile (speed + turn-angle stats).
  2c. Static re-pooling: S95/S05 -- q95/q05 pooling of h itself, layers 15..23.
  2d. Route N only: re-derives the band/rand token-coordinate npz from the stored final_norm
      slice using the saved V_R/rand256 bases (Route R reuses the existing band npz unchanged).

Reuses 32_extract_velocity.py's pooling/packing functions directly (read-only import) -- the
math doesn't change based on how the raw states were obtained.

Usage:
  python 35_derive_streams.py --self-test
  python 35_derive_streams.py --raw-state-dir <path> --route R
  python 35_derive_streams.py --raw-state-dir <path> --route N
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vel_mod = _load("s04_extract", "32_extract_velocity.py")
gate_mod = _load("s04_gate", "34_gate_reconstruct_or_regenerate.py")
band_mod = _load("s02_extract", "27_extract_band.py")

W_START, W_END = 15, 24


# ==============================================================================
# DERIVATION
# ==============================================================================

def h_by_layer_from_raw(raw_beam):
    """raw_beam: (T, 10, D) bf16 tensor, layer order [15..23, final_norm]. Returns the dict
    32_extract_velocity.py's pooling functions expect (fp32 on load, per the store's own
    documented convention -- bf16 stays the on-disk dtype, math always happens in fp32)."""
    d = {}
    for i, l in enumerate(range(W_START, W_END)):
        d[l] = raw_beam[:, i, :].float()
    d["final_norm"] = raw_beam[:, len(range(W_START, W_END)), :].float()
    return d


def derive_all_streams(raw_store_dir, route):
    meta_path = os.path.join(raw_store_dir, "raw_state_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    shard_paths = [os.path.join(raw_store_dir, s) for s in meta["shards"]]

    v95_list, v05_list, kin_list, s95_list, s05_list = [], [], [], [], []
    prompt_ids, beam_idxs, labels = [], [], []
    band_final_list = []   # for route N's 2d, kept in beam order

    t0 = time.time()
    n_done = 0
    for shard_path in shard_paths:
        raw, offsets, pid, bidx, lab = gate_mod.load_raw_state_shard(shard_path)
        n_beams_shard = len(pid)
        for i in range(n_beams_shard):
            s, e = offsets[i], offsets[i + 1]
            raw_beam = raw[s:e]
            h = h_by_layer_from_raw(raw_beam)

            v95, v05 = vel_mod.compute_velocity_streams(h)
            v95_list.append(v95); v05_list.append(v05)
            kin_list.append(vel_mod.compute_kinematic_scalars(h))
            s95, s05 = vel_mod.compute_static_repooling(h)
            s95_list.append(s95); s05_list.append(s05)

            if route == "N":
                band_final_list.append(h["final_norm"])

            prompt_ids.append(int(pid[i])); beam_idxs.append(int(bidx[i])); labels.append(int(lab[i]))
            n_done += 1
        if n_done % 500 < n_beams_shard:
            print(f"  {n_done} beams derived ({time.time()-t0:.0f}s elapsed)")

    packed = vel_mod.pack_velocity_dataset(v95_list, v05_list, kin_list, s95_list, s05_list,
                                            prompt_ids, beam_idxs, labels)

    band_derived = None
    if route == "N":
        bases_path = os.path.join(raw_store_dir, meta["bases_file"])
        bases = np.load(bases_path)
        V_R, V_rand = torch.tensor(bases["V_R"]), torch.tensor(bases["V_rand"])
        band_derived = derive_band_from_final_norm(band_final_list, V_R, V_rand, prompt_ids, beam_idxs, labels)

    return packed, band_derived, meta


def derive_positive_max_core(raw_store_dir):
    """Route N only: reconstructs the {dataset}_pooled_maxenergy-equivalent 'all_emb' tensor
    (positive max-pool over completion tokens, layers 15-23 ONLY -- excludes the final_norm
    slice) from the raw-state store, since Route N's run_route_n() never produced this file on
    its own. Route R does not need this -- it reuses the existing pinned pooled .pt unchanged."""
    meta_path = os.path.join(raw_store_dir, "raw_state_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    shard_paths = [os.path.join(raw_store_dir, s) for s in meta["shards"]]
    n_mid_layers = W_END - W_START

    all_emb, prompt_ids, beam_idxs, labels = [], [], [], []
    for shard_path in shard_paths:
        raw, offsets, pid, bidx, lab = gate_mod.load_raw_state_shard(shard_path)
        for i in range(len(pid)):
            s, e = offsets[i], offsets[i + 1]
            raw_beam = raw[s:e].float()
            if raw_beam.shape[0] == 0:
                D = raw_beam.shape[2]
                H_pooled = torch.zeros(n_mid_layers, D)
            else:
                mid_layers = raw_beam[:, :n_mid_layers, :]   # exclude the final_norm slice (last index)
                H_pooled = mid_layers.max(dim=0).values       # (9, D) -- same convention as
                                                                # 21_generate_maxpool_datasets.py
            all_emb.append(H_pooled)
            prompt_ids.append(int(pid[i])); beam_idxs.append(int(bidx[i])); labels.append(int(lab[i]))

    prompt_ids_arr = np.array(prompt_ids)
    labels_arr = np.array(labels)
    unique_prompts = sorted(set(prompt_ids))
    is_known_by_prompt = {p: bool((labels_arr[prompt_ids_arr == p] == 0).any()) for p in unique_prompts}
    all_is_known = [is_known_by_prompt[p] for p in unique_prompts]

    return {"all_emb": all_emb, "all_hallucination_flag": labels,
            "all_is_known": all_is_known, "prompt_indices": prompt_ids}


def derive_band_from_final_norm(final_norm_list, V_R, V_rand, prompt_ids, beam_idxs, labels):
    """Route N's 2d: re-derives the band/rand token-coordinate npz from the final_norm slice
    and the bases actually saved this time -- same schema as 27_extract_band.py's pack_beams."""
    per_beam = []
    for h_final in final_norm_list:
        z_band = (h_final @ V_R.T).numpy().astype(np.float32) if h_final.shape[0] > 0 \
            else np.zeros((0, V_R.shape[0]), dtype=np.float32)
        z_rand = (h_final @ V_rand.T).numpy().astype(np.float32) if h_final.shape[0] > 0 \
            else np.zeros((0, V_rand.shape[0]), dtype=np.float32)
        rms = torch.sqrt((h_final.pow(2)).mean(dim=-1)).numpy().astype(np.float32) if h_final.shape[0] > 0 \
            else np.zeros((0,), dtype=np.float32)
        per_beam.append((z_band, z_rand, rms))
    meta = {"checkpoint_id": "route-N-derived", "dtype_route": "n/a", "seed": 0, "git_commit": band_mod.git_commit_hash()}
    packed, meta_full = band_mod.pack_beams(per_beam, prompt_ids, beam_idxs, labels, meta)
    return packed, meta_full


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: derivation from a synthetic raw-state store (no model)")
    print("=" * 70)
    tmp_dir = os.path.join(HERE, "results", "_selftest_derive")
    os.makedirs(tmp_dir, exist_ok=True)

    rng = np.random.default_rng(0)
    D = 32
    n_beams = 25
    per_beam_raw = []
    prompt_ids, beam_idxs, labels = [], [], []
    for i in range(n_beams):
        T_i = int(rng.integers(0, 9))
        h = {l: torch.tensor(rng.normal(0, 1, size=(T_i, D)).astype(np.float32))
             for l in range(W_START, W_END)}
        h["final_norm"] = torch.tensor(rng.normal(0, 1, size=(T_i, D)).astype(np.float32))
        per_beam_raw.append(gate_mod.pack_raw_state_beam(h))
        prompt_ids.append(i // 5); beam_idxs.append(i % 5); labels.append(int(rng.integers(0, 2)))

    V_R = torch.randn(20, D); V_rand = torch.randn(16, D)
    out_dir = os.path.join(tmp_dir, "raw_store")
    shard_paths, bases_path, meta_path = gate_mod.pack_and_save_raw_store(
        per_beam_raw, prompt_ids, beam_idxs, labels, V_R, V_rand, out_dir,
        "self-test", 0, {"do_sample": True}, "selftest")
    print(f"  [PASS] fabricated raw-state store: {len(shard_paths)} shard(s)")

    packed, band_derived, meta = derive_all_streams(out_dir, route="R")
    assert packed["V95"].shape == (n_beams, 8, D)
    assert packed["kinematic"].shape == (n_beams, 30)
    assert packed["S95"].shape == (n_beams, 9, D)
    assert band_derived is None, "route R should not derive a band npz"
    print(f"  [PASS] Route R derivation: V95={packed['V95'].shape}  kinematic={packed['kinematic'].shape}  "
          f"S95={packed['S95'].shape}, band_derived correctly None")

    packed_n, band_derived_n, _ = derive_all_streams(out_dir, route="N")
    assert band_derived_n is not None
    band_packed, band_meta = band_derived_n
    assert band_packed["z_band"].shape[1] == 20
    assert band_packed["z_rand"].shape[1] == 16
    assert np.array_equal(band_packed["label"], np.asarray(labels))
    print(f"  [PASS] Route N derivation additionally derives band npz: "
          f"z_band dim={band_packed['z_band'].shape[1]}, z_rand dim={band_packed['z_rand'].shape[1]}")

    core_data = derive_positive_max_core(out_dir)
    assert len(core_data["all_emb"]) == n_beams
    assert core_data["all_emb"][0].shape == (W_END - W_START, D)
    assert len(core_data["all_hallucination_flag"]) == n_beams
    assert len(core_data["all_is_known"]) == len(set(prompt_ids))
    # cross-check against a hand-computed max-pool on beam 0's raw mid-layer slices
    raw0_check = per_beam_raw[0].float()
    if raw0_check.shape[0] > 0:
        expected_core0 = raw0_check[:, : W_END - W_START, :].max(dim=0).values
        torch.testing.assert_close(core_data["all_emb"][0], expected_core0)
    print(f"  [PASS] derive_positive_max_core: {len(core_data['all_emb'])} beams, shape "
          f"{tuple(core_data['all_emb'][0].shape)}, matches a hand-computed max-pool check")

    # sanity: derived velocity should be internally consistent with a hand-computed check on beam 0
    raw0 = per_beam_raw[0].float()
    if raw0.shape[0] > 0:
        h0 = {l: raw0[:, i, :] for i, l in enumerate(range(W_START, W_END))}
        v95_expected, _ = vel_mod.compute_velocity_streams(h0)
        assert torch.allclose(v95_expected, torch.tensor(packed["V95"][0]), atol=1e-4)
        print("  [PASS] beam-0 velocity matches a direct hand-computed check")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-state-dir", type=str, default=None)
    parser.add_argument("--route", type=str, choices=["R", "N"], default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.raw_state_dir or not args.route:
        print("ERROR: --raw-state-dir and --route required."); sys.exit(1)

    print(f"Deriving streams from {args.raw_state_dir} (route {args.route}) ...")
    packed, band_derived, meta = derive_all_streams(args.raw_state_dir, args.route)

    out_dir = args.output_dir or args.raw_state_dir
    out_path = os.path.join(out_dir, "velocity_kinematic_repooling.npz")
    decoding_config = meta.get("decoding_config", {})
    vel_mod.save_velocity_dataset(packed, out_path, meta.get("checkpoint_id", "unknown"),
                                   meta.get("seed", 0), decoding_config)
    print(f"Saved derived streams: {out_path}")

    if band_derived is not None:
        band_packed, band_meta = band_derived
        band_out_path = os.path.join(out_dir, "band_derived_v3.npz")
        shard_paths, band_meta_path = band_mod.save_packed(band_packed, band_meta, band_out_path)
        print(f"Saved Route N-derived band npz: {shard_paths}")

    if args.route == "N":
        # run_route_n() never produced a pooled 'all_emb' core tensor on its own -- reconstruct
        # it here from the raw-state store (positive max-pool, layers 15-23 only) so
        # 33_eval_session04.py --v3-pooled-pt has something real to point at.
        core_data = derive_positive_max_core(args.raw_state_dir)
        core_out_path = os.path.join(out_dir, "truthfulqa_v3_pooled.pt")
        torch.save(core_data, core_out_path)
        print(f"Saved Route N pooled core tensor (--v3-pooled-pt target): {core_out_path}")


if __name__ == "__main__":
    main()
