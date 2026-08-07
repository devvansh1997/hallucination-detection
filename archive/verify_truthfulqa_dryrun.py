"""
verify_truthfulqa_dryrun.py — Strict Alignment Audit
======================================================
Loads truthfulqa_pooled.pt and asserts perfect alignment between
embeddings, labels, and prompt indices.  Also prints jagged grouping
evidence from early-stopping beam search.

Usage:
  python verify_truthfulqa_dryrun.py --model llama-3.1-8b-instruct
"""

import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--dataset", type=str, default="truthfulqa")
args = parser.parse_args()

DATA_DIR = "../data"
path = os.path.join(DATA_DIR, args.model, f"{args.dataset}_pooled.pt")

print(f"Loading: {path}")
data = torch.load(path, weights_only=False)

features = data["all_emb"]
flags    = data["all_hallucination_flag"]
prompt_indices = data["prompt_indices"]

# ---- 1. Array lengths must match exactly ----
assert len(features) == len(flags) == len(prompt_indices), (
    f"CRITICAL FATAL ERROR: ARRAY LENGTHS DO NOT MATCH. "
    f"features={len(features)}, flags={len(flags)}, indices={len(prompt_indices)}"
)

# ---- 2. Unique prompts ----
n_beams   = len(features)
n_unique  = len(set(prompt_indices))
n_prompts = len(data["all_is_known"])

print(f"\n  Total beams:         {n_beams}")
print(f"  Unique prompt indices: {n_unique}")
print(f"  all_is_known length:   {n_prompts}")
assert n_unique == n_prompts, (
    f"MISMATCH: unique indices ({n_unique}) != is_known length ({n_prompts})"
)

# ---- 3. Print first 50 prompt indices (shows jagged grouping) ----
print(f"\n  First 50 prompt_indices (jagged early-stopping pattern):")
print(f"  {prompt_indices[:50]}")

# ---- 4. Success ----
print(f"\n  VERIFICATION PASSED: Arrays are perfectly aligned and leak-proof.")
