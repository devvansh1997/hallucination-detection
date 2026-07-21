"""
30_pin_manifest.py -- Session 03 Part 0: Pin the Canonical Seeded Dataset
=============================================================================
CPU-only. Does not load the LLM, regenerate beams, or recompute labels.
Hashes the session02 seeded artifacts (truthfulqa_band.npz shards +
truthfulqa_pooled_maxenergy_seeded.pt) and the labels array, records counts
and prompt composition, and recomputes the two reference numbers session04+
scripts compare against: core-only RF/LR (grouped) and the HARP-protocol
core-only row session02 omitted.

verify_manifest() is imported by every later eval entrypoint (starting with
31_eval_session03.py) and hard-fails if the on-disk files no longer match
what was pinned here.

Usage:
  python 30_pin_manifest.py --self-test
  python 30_pin_manifest.py --model_folder llama-3.1-8b-instruct --dataset truthfulqa
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

_spec01 = importlib.util.spec_from_file_location("s01", os.path.join(HERE, "26_grouped_baseline.py"))
s01 = importlib.util.module_from_spec(_spec01)
_spec01.loader.exec_module(s01)

_spec_band = importlib.util.spec_from_file_location("s02_extract", os.path.join(HERE, "27_extract_band.py"))
band_mod = importlib.util.module_from_spec(_spec_band)
_spec_band.loader.exec_module(band_mod)

_spec02 = importlib.util.spec_from_file_location("s02", os.path.join(HERE, "28_eval_band.py"))
s02 = importlib.util.module_from_spec(_spec02)
_spec02.loader.exec_module(s02)

SEED = 0
N_SPLITS = 5
ORIGINAL_SEED = 42
# session02's own recomputed core-only numbers (results/session02_metrics.json ->
# core_only_recomputed), quoted directly in this session's prompt as the expected
# reproduction target on the canonical seeded set.
EXPECTED_CORE_RF_POOLED = 0.8347
EXPECTED_CORE_RF_WITHIN_PROMPT = 0.7365


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def compute_prompt_composition(y, prompt_idx):
    n_mixed = n_all_truthful = n_all_halluc = 0
    for p in np.unique(prompt_idx):
        labels = y[prompt_idx == p]
        if (labels == 1).any() and (labels == 0).any():
            n_mixed += 1
        elif (labels == 1).all():
            n_all_halluc += 1
        else:
            n_all_truthful += 1
    return {"n_mixed_prompts": n_mixed, "n_all_truthful_prompts": n_all_truthful,
            "n_all_hallucinated_prompts": n_all_halluc}


# ==============================================================================
# MANIFEST CREATION
# ==============================================================================

def build_manifest(band_meta_path, pooled_pt_path, r_l, r_d,
                    expected_composition=(406, 111, 300), skip_composition_check=False):
    with open(band_meta_path) as f:
        band_meta = json.load(f)
    shard_dir = os.path.dirname(band_meta_path)
    shard_paths = [os.path.join(shard_dir, s) for s in band_meta["shards"]]

    packed = band_mod.load_packed(shard_paths)
    pooled = torch.load(pooled_pt_path, weights_only=False)

    y_band = np.asarray(packed["label"], dtype=np.int64)
    y_pooled = np.array([int(f) for f in pooled["all_hallucination_flag"]], dtype=np.int64)
    if not np.array_equal(y_band, y_pooled):
        raise ValueError("Band-file labels and pooled-file labels do not match beam-for-beam. "
                          "Refusing to pin a manifest over a possibly misaligned dataset.")
    prompt_idx = np.asarray(pooled["prompt_indices"], dtype=np.int64)
    n_beams = len(y_band)

    composition = compute_prompt_composition(y_band, prompt_idx)
    if not skip_composition_check:
        expected = {"n_mixed_prompts": expected_composition[0],
                    "n_all_truthful_prompts": expected_composition[1],
                    "n_all_hallucinated_prompts": expected_composition[2]}
        if composition != expected:
            raise ValueError(
                f"Prompt composition {composition} does not match the expected canonical "
                f"composition {expected}. This dataset is not the one session02/03 assume -- "
                f"refusing to pin it as manifest_seeded_v1. Pass --skip-composition-check "
                f"explicitly if this is deliberately a different dataset.")
    print(f"  Composition: {composition}")

    band_hashes = {s: sha256_file(os.path.join(shard_dir, s)) for s in band_meta["shards"]}
    pooled_hash = sha256_file(pooled_pt_path)
    labels_hash = sha256_array(y_band)

    print("  Recomputing core-only RF/LR (grouped, fold-pure Tucker) reference ...")
    X = torch.stack(pooled["all_emb"]).float().numpy()
    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(n_splits=N_SPLITS).split(X, y_band, groups=prompt_idx))
    core_results, core_by_fold = s02.run_core_only(X, y_band, prompt_idx, folds, r_l, r_d, seed=SEED)
    print(f"    RF pooled={core_results['RF']['pooled_oof_auroc']:.4f}  "
          f"within-prompt={core_results['RF']['within_prompt']['within_prompt_auroc']:.4f}")
    delta_rf_pooled = abs(core_results["RF"]["pooled_oof_auroc"] - EXPECTED_CORE_RF_POOLED)
    delta_rf_within = abs(core_results["RF"]["within_prompt"]["within_prompt_auroc"] - EXPECTED_CORE_RF_WITHIN_PROMPT)
    if delta_rf_pooled > 0.005 or delta_rf_within > 0.005:
        print(f"    [WARN] deviates from session02's reported 0.8347/0.7365 by "
              f"{delta_rf_pooled:.4f}/{delta_rf_within:.4f} (> 0.005) -- known sklearn/threading "
              f"nondeterminism (see session01 audit A5); not treated as fatal.")

    print("  Recomputing HARP-protocol core-only (session02 omitted this row) ...")
    is_known = np.asarray(pooled["all_is_known"], dtype=bool)
    data_for_e1 = {"X": X, "y": y_band, "prompt_idx": prompt_idx, "is_known": is_known,
                   "n_beams": n_beams}
    harp_core = s01.eval_E1(data_for_e1, r_l, r_d)

    manifest = {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "extraction_git_commit": band_meta.get("git_commit", "unknown"),
        "band_npz_sha256": band_hashes,
        "pooled_pt_sha256": pooled_hash,
        "labels_sha256": labels_hash,
        "band_meta_path": os.path.abspath(band_meta_path),
        "pooled_pt_path": os.path.abspath(pooled_pt_path),
        "counts": {
            "n_beams": n_beams, "n_prompts": int(len(np.unique(prompt_idx))),
            "n_tokens": int(packed["z_band"].shape[0]),
            "n_hallucinated": int(y_band.sum()), "n_truthful": int((y_band == 0).sum()),
            **composition,
        },
        "references": {
            "core_only_grouped": core_results,
            "harp_protocol_core_only": harp_core,
        },
        "config": {"r_l": r_l, "r_d": r_d, "seed": SEED, "n_splits": N_SPLITS},
    }
    return manifest


def verify_manifest(manifest_path, band_meta_path=None, pooled_pt_path=None):
    """Recomputes sha256 of the pinned files and raises if anything drifted.
    Returns the loaded manifest dict on success."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run 30_pin_manifest.py first.")
    with open(manifest_path) as f:
        manifest = json.load(f)

    band_meta_path = band_meta_path or manifest["band_meta_path"]
    pooled_pt_path = pooled_pt_path or manifest["pooled_pt_path"]
    if not os.path.exists(band_meta_path) or not os.path.exists(pooled_pt_path):
        raise FileNotFoundError(
            f"Manifest points at files that no longer exist: {band_meta_path}, {pooled_pt_path}")

    with open(band_meta_path) as f:
        band_meta = json.load(f)
    shard_dir = os.path.dirname(band_meta_path)
    for shard_name, expected_hash in manifest["band_npz_sha256"].items():
        shard_path = os.path.join(shard_dir, shard_name)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Manifest expects shard {shard_path}, not found.")
        actual = sha256_file(shard_path)
        if actual != expected_hash:
            raise ValueError(f"MANIFEST MISMATCH: {shard_path} hash changed "
                              f"({expected_hash[:12]}... -> {actual[:12]}...). The pinned dataset "
                              f"has drifted -- do not evaluate against it silently.")

    actual_pooled_hash = sha256_file(pooled_pt_path)
    if actual_pooled_hash != manifest["pooled_pt_sha256"]:
        raise ValueError(f"MANIFEST MISMATCH: {pooled_pt_path} hash changed. The pinned dataset "
                          f"has drifted -- do not evaluate against it silently.")

    return manifest


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: manifest build + verify (tamper detection), no real data")
    print("=" * 70)
    tmp_dir = os.path.join(HERE, "results", "_selftest_manifest")
    os.makedirs(tmp_dir, exist_ok=True)

    data = s01.generate_synthetic_data(n_prompts=30, beams_per_prompt=10, L=9, D=32, seed=0)
    X, y, prompt_idx = data["X"], data["y"], data["prompt_idx"]
    is_known = data["is_known"]

    pooled_path = os.path.join(tmp_dir, "fake_pooled.pt")
    torch.save({"all_emb": [torch.tensor(X[i]) for i in range(X.shape[0])],
                "all_hallucination_flag": [bool(v) for v in y],
                "all_is_known": [bool(v) for v in is_known],
                "prompt_indices": prompt_idx.tolist()}, pooled_path)

    per_beam = []
    rng = np.random.default_rng(0)
    for i in range(len(y)):
        T_i = int(rng.integers(2, 8))
        per_beam.append((rng.normal(0, 1, size=(T_i, 6)).astype(np.float32),
                          rng.normal(0, 1, size=(T_i, 4)).astype(np.float32),
                          np.zeros(T_i, dtype=np.float32)))
    meta = {"checkpoint_id": "self-test", "dtype_route": "hidden_states[-1]", "seed": 0,
            "git_commit": "selftest"}
    packed, meta_full = band_mod.pack_beams(per_beam, prompt_idx.tolist(), list(range(len(y))),
                                             y.tolist(), meta)
    band_path = os.path.join(tmp_dir, "fake_band.npz")
    shard_paths, band_meta_path = band_mod.save_packed(packed, meta_full, band_path)

    manifest = build_manifest(band_meta_path, pooled_path, r_l=5, r_d=10,
                               expected_composition=None, skip_composition_check=True)
    manifest_path = os.path.join(tmp_dir, "manifest_selftest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [PASS] manifest built: {manifest['counts']}")

    verified = verify_manifest(manifest_path)
    assert verified["counts"]["n_beams"] == len(y)
    print("  [PASS] verify_manifest succeeds on an untampered dataset")

    # tamper: flip one byte in the pooled file, verify hard-fails
    with open(pooled_path, "r+b") as f:
        f.seek(0)
        b = f.read(1)
        f.seek(0)
        f.write(bytes([b[0] ^ 0xFF]))
    try:
        verify_manifest(manifest_path)
        raise AssertionError("verify_manifest should have raised on a tampered pooled file")
    except ValueError as e:
        assert "MANIFEST MISMATCH" in str(e)
        print("  [PASS] verify_manifest correctly hard-fails on a tampered file")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="truthfulqa")
    parser.add_argument("--pooled-suffix", type=str, default="_maxenergy_seeded")
    parser.add_argument("--r_l", type=int, default=5)
    parser.add_argument("--r_d", type=int, default=64)
    parser.add_argument("--output", type=str, default="data/manifest_seeded_v1.json")
    parser.add_argument("--skip-composition-check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.model_folder:
        print("ERROR: --model_folder required."); sys.exit(1)

    import yaml
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    data_dir = cfg["output"]["data_dir"]
    band_meta_path = os.path.join(data_dir, args.model_folder, f"{args.dataset}_band_meta.json")
    pooled_pt_path = os.path.join(data_dir, args.model_folder, f"{args.dataset}_pooled{args.pooled_suffix}.pt")

    manifest = build_manifest(band_meta_path, pooled_pt_path, args.r_l, args.r_d,
                               skip_composition_check=args.skip_composition_check)

    out_path = os.path.join(HERE, args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote: {out_path}")
    print(f"Counts: {manifest['counts']}")


if __name__ == "__main__":
    main()
