"""
04_verify_data_integrity.py — Audit Generated Tensor Files
============================================================
Scans data/*/ for all .pt files, checks shapes, labels, numerical sanity.
"""

import argparse
import os
import glob

import torch
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None,
                    help="e.g. llama-3.1-8b-instruct")
parser.add_argument("--dataset", type=str, default=None,
                    help="e.g. triviaqa")
args = parser.parse_args()

DATA_DIR = "../data"

print("=" * 72)
print("  DATA INTEGRITY AUDIT")
print("=" * 72)

if args.model and args.dataset:
    fpath = os.path.join(DATA_DIR, args.model, f"{args.dataset}_pooled.pt")
    if not os.path.exists(fpath):
        print(f"  File not found: {fpath}")
        exit(1)
    files = [fpath]
elif args.model:
    files = sorted(glob.glob(os.path.join(DATA_DIR, args.model, "*_pooled.pt")))
else:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*", "*_pooled.pt")))

if not files:
    print("  No pooled files found.")
    exit(1)

print(f"\n  Found {len(files)} pooled files:\n")

issues = []

for fpath in files:
    rel = os.path.relpath(fpath, DATA_DIR)
    folder = rel.split(os.sep)[0]
    fname = os.path.basename(fpath)
    dataset = fname.replace("_pooled.pt", "")

    data = torch.load(fpath, weights_only=False)
    emb = data["all_emb"]
    flags = data["all_hallucination_flag"]

    N = len(emb)
    shape = emb[0].shape if N > 0 else None
    L, D = shape if shape else (0, 0)
    n_hall = sum(flags)
    rate = n_hall / N * 100 if N > 0 else 0
    has_is_known = "all_is_known" in data
    has_prompt_idx = "prompt_indices" in data
    n_known = sum(data.get("all_is_known", [])) if has_is_known else "N/A"

    # Numerical sanity
    n_nan, n_inf, n_zero = 0, 0, 0
    for t in tqdm(emb, desc=f"    scanning", leave=False):
        if torch.isnan(t).any():
            n_nan += 1
        if torch.isinf(t).any():
            n_inf += 1
        if (t == 0).all():
            n_zero += 1

    flag_ok = (len(flags) == N)

    print(f"  [{folder}] {dataset}")
    print(f"    Samples: {N:6d}  |  shape: {tuple(shape) if shape else 'EMPTY'}"
          f"  |  L={L}  D={D}")
    print(f"    Labels:  {n_hall} hallucinated, {N - n_hall} truthful"
          f"  ({rate:.1f}%)  |  len(flags)==N: {flag_ok}")
    print(f"    Known prompts: {str(n_known):>6s}  "
          f"|  has_is_known: {has_is_known}  |  has_prompt_indices: {has_prompt_idx}")
    print(f"    NaN: {n_nan}  |  Inf: {n_inf}  |  Zero-filled: {n_zero}")

    if n_nan > 0 or n_inf > 0:
        issues.append(f"{rel}: {n_nan} NaN, {n_inf} Inf")
    if n_zero > 0.1 * N:
        issues.append(f"{rel}: {n_zero} zero-filled tensors ({n_zero/N*100:.1f}%)")
    if not flag_ok:
        issues.append(f"{rel}: label length mismatch {len(flags)} vs {N}")
    if not has_prompt_idx:
        issues.append(f"{rel}: MISSING prompt_indices (needed for known/unknown eval)")

    print()

# Cross-check: beam counts should be >= prompt count
print("  " + "=" * 72)
print("  CROSS-CHECKS")
print("  " + "=" * 72)
for fpath in files:
    data = torch.load(fpath, weights_only=False)
    if "all_is_known" in data and "prompt_indices" in data:
        n_beams = len(data["all_emb"])
        n_prompts = len(data["all_is_known"])
        n_unique = len(set(data["prompt_indices"]))
        rel = os.path.relpath(fpath, DATA_DIR)
        beams_per = n_beams / n_prompts if n_prompts > 0 else 0
        ok = (n_unique == n_prompts)
        print(f"  {rel:50s}  beams={n_beams:6d}  prompts={n_prompts:5d}  "
              f"avg_beams={beams_per:.1f}  unique_prompts={n_unique}  ok={ok}")
    elif "all_is_known" in data:
        rel = os.path.relpath(fpath, DATA_DIR)
        n_beams = len(data["all_emb"])
        n_prompts = len(data["all_is_known"])
        print(f"  {rel:50s}  beams={n_beams:6d}  prompts={n_prompts:5d}  "
              f"NO prompt_indices (cannot do known/unknown split)")

print()
if issues:
    print(f"  ISSUES FOUND ({len(issues)}):")
    for i in issues:
        print(f"    - {i}")
else:
    print("  All files clean — no NaN, Inf, shape mismatches, or label alignment issues.")

print(f"\n{'=' * 72}")
