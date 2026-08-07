"""
01b_merge_shards.py — Merge Distributed Shard Files
=====================================================
Concatenates per-shard .pt files into final _pooled.pt, preserving
prompt_indices, all_emb, all_hallucination_flag, all_is_known.

Usage:
  python 01b_merge_shards.py --model llama-3.1-8b-instruct --dataset triviaqa --num_shards 8
"""

import argparse
import glob
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True,
                    help="Model folder name, e.g. llama-3.1-8b-instruct")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--num_shards", type=int, default=8)
args = parser.parse_args()

DATA_DIR = f"../data/{args.model}"
BASE = f"{args.dataset}_shard"

# Find shard files
pattern = os.path.join(DATA_DIR, f"{BASE}_*.pt")
files = sorted(glob.glob(pattern))
print(f"Found {len(files)} shard files for {args.model}/{args.dataset}")

keys = ["all_emb", "all_hallucination_flag", "all_is_known", "prompt_indices"]
merged = {k: [] for k in keys}

for fpath in files:
    data = torch.load(fpath, weights_only=False)
    for k in keys:
        merged[k].extend(data[k])
    print(f"  {os.path.basename(fpath):40s} -> {len(merged[keys[0]]):6d} total")

out_path = os.path.join(DATA_DIR, f"{args.dataset}_pooled.pt")
torch.save(merged, out_path)

n_beams = len(merged["all_emb"])
n_prompts = len(set(merged["prompt_indices"]))
print(f"\nSaved: {out_path}")
print(f"  Beams: {n_beams}  |  Unique prompts: {n_prompts}")

# Delete shard files
for fpath in files:
    os.remove(fpath)
print("  Shard files deleted.")
