"""
01b_merge_shards.py -- Merge Distributed Shards
=================================================
Finds all faizul_shard_*.pt and hosvd_shard_*.pt files for a given
model/dataset, concatenates their contents, and saves the final
aggregated files.

Usage:
  python 01b_merge_shards.py --model meta-llama/Llama-3.2-3B-Instruct --dataset triviaqa --num_shards 8
"""

import argparse
import glob
import os
import torch


# ==============================================================================
# ARGPARSE
# ==============================================================================

parser = argparse.ArgumentParser(description="Merge distributed shard files")
parser.add_argument("--model", type=str, required=True,
                    help="HuggingFace model ID (must match generation)")
parser.add_argument("--dataset", type=str, required=True,
                    choices=["truthfulqa", "triviaqa", "tydiqa"])
parser.add_argument("--num_shards", type=int, default=8,
                    help="Number of shards that were generated")
args = parser.parse_args()

MODEL_KEY = args.model.split("/")[-1].replace("-", "").replace(".", "_").lower()
DATA_DIR  = "../data"
BASE      = f"{MODEL_KEY}_{args.dataset}"


# ==============================================================================
# MERGE HELPER
# ==============================================================================

def merge_shards(prefix_suffix: str, keys: list[str]) -> dict:
    """Find all shard files matching *prefix_suffix*.pt, load them,
    concatenate each key's list, and return the merged dict.

    prefix_suffix = e.g. "faizul_shard" or "hosvd_shard"
    """
    pattern = os.path.join(DATA_DIR, f"{BASE}_{prefix_suffix}_*.pt")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No shard files found matching: {pattern}")

    print(f"  Found {len(files)} {prefix_suffix} shard files")

    merged = {k: [] for k in keys}
    total_samples = 0

    for fpath in files:
        data = torch.load(fpath, weights_only=False)
        for k in keys:
            merged[k].extend(data[k])
        total_samples += len(data[keys[0]])
        print(f"    {os.path.basename(fpath):50s}  ->  {total_samples:6d} total")

    print(f"  Merged: {total_samples} samples across {len(files)} shards")
    return merged


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"  MERGE SHARDS  |  {MODEL_KEY}  |  {args.dataset.upper()}")
    print("=" * 60)

    # ── Faizul shards ───────────────────────────────────────────────────
    print(f"\n[1/2] Merging Faizul shards ...")
    faizul_merged = merge_shards("faizul_shard",
                                 keys=["features", "labels"])

    faizul_out = os.path.join(DATA_DIR, f"{BASE}_faizul_full.pt")
    torch.save(faizul_merged, faizul_out)
    print(f"  Saved -> {faizul_out}")

    # ── HOSVD shards ────────────────────────────────────────────────────
    print(f"\n[2/2] Merging HOSVD shards ...")
    hosvd_merged = merge_shards("hosvd_shard",
                                keys=["all_emb", "all_hallucination_flag"])

    hosvd_out = os.path.join(DATA_DIR, f"{BASE}_hosvd_full.pt")
    torch.save(hosvd_merged, hosvd_out)
    print(f"  Saved -> {hosvd_out}")

    # ── Cleanup: delete individual shard files ───────────────────────────
    print(f"\n  Deleting individual shard files ...")
    for tag in ["faizul_shard", "hosvd_shard"]:
        pattern = os.path.join(DATA_DIR, f"{BASE}_{tag}_*.pt")
        for fpath in glob.glob(pattern):
            os.remove(fpath)
            print(f"    deleted: {os.path.basename(fpath)}")

    n_faizul = len(faizul_merged["labels"])
    n_hosvd  = len(hosvd_merged["all_emb"])
    assert n_faizul == n_hosvd, \
        f"Mismatch: Faizul {n_faizul} samples vs HOSVD {n_hosvd} samples"

    print(f"\n{'=' * 60}")
    print(f"  MERGE COMPLETE")
    print(f"  Samples:   {n_faizul}")
    print(f"  Faizul:    {faizul_out}")
    print(f"  HOSVD:     {hosvd_out}")
    print(f"{'=' * 60}")
