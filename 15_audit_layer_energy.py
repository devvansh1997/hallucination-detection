"""
15_audit_layer_energy.py — Layer-wise Energy Divergence Audit
===============================================================
Computes per-layer L2 energy for truthful vs hallucinated beams,
plots the divergence, and identifies the layer of maximum separation.

Usage:
  python 15_audit_layer_energy.py --model_folder llama-3.1-8b-instruct --dataset triviaqa
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless-safe
import matplotlib.pyplot as plt
import yaml

import torch

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = cfg["output"]["data_dir"]

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--suffix", type=str, default="")
args = parser.parse_args()

# ── Load ──
path = os.path.join(DATA_DIR, args.model_folder,
                    f"{args.dataset}_pooled{args.suffix}.pt")
print(f"Loading: {path}")
data = torch.load(path, weights_only=False)
X = torch.stack(data["all_emb"])               # (N, L, D)
flags = np.array([int(f) for f in data["all_hallucination_flag"]])

N, L, D = X.shape
print(f"  Beams: {N}  Layers: {L}  Hidden: {D}")
print(f"  Truthful: {(flags == 0).sum()}  Hallucinated: {(flags == 1).sum()}")

# ── Layer-wise L2 energy ──
X = X.float()                                   # (N, L, D)
energy = X.norm(dim=2)                          # (N, L) — L2 norm per layer

truth_mask = (flags == 0)
hall_mask  = (flags == 1)

energy_truth = energy[truth_mask].mean(dim=0)   # (L,)
energy_hall  = energy[hall_mask].mean(dim=0)    # (L,)
delta        = (energy_truth - energy_hall).abs()  # (L,)

best_layer = int(delta.argmax().item())

print(f"\n  Layer of max divergence: {best_layer}")
print(f"    Truthful energy: {energy_truth[best_layer]:.4f}")
print(f"    Hallucinated:    {energy_hall[best_layer]:.4f}")
print(f"    Delta:           {delta[best_layer]:.4f}")

# ── Plot ──
layers = np.arange(L)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(layers, energy_truth.numpy(), "b-", label="Truthful", linewidth=1.5)
ax1.plot(layers, energy_hall.numpy(),  "r-", label="Hallucinated", linewidth=1.5)
ax1.set_ylabel("Avg L2 Norm")
ax1.set_title(f"{args.model_folder} / {args.dataset} — Layer-wise Energy")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.bar(layers, delta.numpy(), color="purple", alpha=0.7)
ax2.axvline(x=best_layer, color="black", linestyle="--",
            label=f"Max at layer {best_layer}")
ax2.set_xlabel("Layer Index")
ax2.set_ylabel("|Truthful − Hallucinated|")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(DATA_DIR, args.model_folder,
                        f"{args.dataset}_layer_energy_audit.png")
plt.savefig(out_path, dpi=150)
print(f"\n  Saved: {out_path}")
