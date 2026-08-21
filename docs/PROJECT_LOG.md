# Project log — hallucination detection via multilinear decomposition

Running record of findings, corrections and operational knowledge, kept current as work
progresses. Every claim here carries its evidence; anything retracted stays visible with the
reason, because knowing what we got wrong is as load-bearing as knowing what we got right.

Last updated: 2026-08-20.

---

## 1. Where things stand

**Models.** `qwen-2.5-7b-instruct` (complete), `llama-3.1-8b` **base** (3 of 4 datasets),
`llama-3.1-8b-instruct` (legacy, kept — *not* the checkpoint HARP evaluates).

**Datasets.** TruthfulQA (817), TriviaQA (9,960 after question_id dedup), NQ-Open (3,610),
TyDiQA-GP (440, English-filtered). 10 beams per question throughout.

**Why base LLaMA matters.** HARP's `main.py:20` resolves its `"Llama-3.1-8B"` key to
`meta-llama/Llama-3.1-8B` — the base model — and §5.1 of the paper agrees. Our earlier LLaMA rows
used Instruct, so they were never a like-for-like comparison.

| model | dataset | halluc % | known Q | ours (question-level) | ours (answer-level) | leakage cost |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | TruthfulQA | 43.3 | 666 | 87.80 | 94.11 | +6.31 |
| Qwen2.5-7B | TriviaQA | 74.0 | 6,316 | 92.70 | 94.05 (5/6) | +1.35 |
| Qwen2.5-7B | NQ-Open | 96.9 | 360 | 79.40 | 91.15 | +11.75 |
| Qwen2.5-7B | TyDiQA-GP | 59.1 | 302 | 88.30 | 94.72 | +6.42 |
| LLaMA-3.1-8B base | TruthfulQA | 63.0 | 506 | 89.32 | 94.47 | +5.15 |
| LLaMA-3.1-8B base | TriviaQA | — | — | *running* | *running* | — |
| LLaMA-3.1-8B base | NQ-Open | 88.8 | 1,267 | 85.36 | 91.96 | +6.60 |
| LLaMA-3.1-8B base | TyDiQA-GP | 48.9 | 404 | 83.23 | 91.00 | +7.77 |

TriviaQA/Qwen answer-level covers 5 of 6 conditions (`joint_tensor` OOM'd at 200 GB). It is a
maximum, so the true value can only be higher and the +1.35 is a lower bound.

**HARP reproduction (Qwen, their code on our data, proj_dim 256).** TruthfulQA 86.27 vs published
88.1; TriviaQA 92.89 vs 92.8; NQ-Open 83.12 vs 84.0; TyDiQA-GP 91.87 vs 88.4. Mean signed
deviation **+0.21** — no systematic bias, which is what licenses everything downstream.

**HARP on LLaMA base** — runs completed on the cluster (jobs 763901–903); numbers not yet pulled
into the reports.

---

## 2. Findings

### 2.1 HARP's released code splits over answers, not questions

`main.py:259` calls `utils.split_data` on a flat per-answer list, so the 75/25 split inside the
known group is over **answers**. The paper (§3.1, Appendix A) describes a split over **questions**
("the set of all inputs"). Consequence: a known question keeps all ten answers on one side only
0.75^10 = 5.6% of the time, so **~94% of known questions appear on both sides**.

Observed: 626/666, 5,956/6,316, 334/360, 278/302 — **92–94%**, against the 94.4% predicted.

The known/unknown grouping itself is **correct** (`get_known.py:115`, `main.py:264`). Only the
sub-split within the known group is at the wrong granularity. Stating it more broadly than that
would be wrong.

### 2.2 The mismatch is worth 3.8–13.3 AUROC points, and it replicates

Same hidden states, same SVD, same MLP, same seeds — only the split unit varied.

| | TruthfulQA | TriviaQA | NQ-Open | TyDiQA-GP |
|---|---|---|---|---|
| HARP answer-level | 85.16 | 92.50 | 81.03 | 91.75 |
| HARP question-level | 77.54 | 88.75 | 67.73 | 79.90 |
| **cost** | −7.62 | −3.75 | −13.30 | −11.85 |

The `answer` arm calls **their** `utils.split_data` and reproduces what their `main.py` printed,
so it is a validated control. Our method loses 1.35–11.75 (Qwen) and 5.15–7.77 (LLaMA base) under
the same correction — **this is a property of the protocol, not of their method, and we are not
exempt from it.**

### 2.3 Under matched protocols we beat HARP on all four datasets

Qwen, question-level: 87.8 vs 77.54, 92.7 vs 88.75, 79.4 vs 67.73, 88.3 vs 79.90.
Qwen, answer-level: 94.11 vs 85.16, 94.05 vs 92.50, 91.15 vs 81.03, 94.72 vs 91.75.

Both protocols, all four datasets. The earlier impression that we lost 3 of 4 came **only** from
comparing our question-level numbers against their answer-level ones — the mismatched comparison
the literature table invites.

### 2.4 Our six conditions are indistinguishable within-prompt

Across three Qwen datasets, **10 of 24** pairwise condition comparisons have pooled-AUROC CIs
excluding zero. **0 of 24** do under within-prompt AUROC.

Whatever separates `triple_concat` from `core_max` on pooled numbers, it is not the ability to rank
answers *to the same question*. This constrains any theory we build: a story explaining only the
pooled number is mostly explaining 2.5.

### 2.5 A large share of pooled AUROC is question difficulty

Qwen/NQ-Open, `core_max` + RF: pooled 0.875, within-prompt 0.745, same score vector.

### 2.6 A memorisation signature

Under the question-level split, LR beats RF on TriviaQA for **all six** conditions. Under the
answer-level split the ranking **inverts** — RF wins there and on every other dataset
(`q_static`: 94.03 RF / 93.02 LR answer-level, against 91.7 RF / 92.7 LR question-level).

A forest can carve per-question regions a linear model structurally cannot, so the classifier
better at memorising is exactly the one that gains when the split permits it. This fell out of
data collected for another purpose, which is what makes it worth something.

---

## 3. Retracted / corrected

Kept deliberately. Each cost time and each would have been caught by a reviewer.

| claim | status | why |
|---|---|---|
| TyDiQA's negative pooled/within gap is because "pooling mixes questions across languages" | **wrong** | We filter to English (`39_generate_dataset.py`, `extract_tydiqa_language`). So does HARP (`DatasetInit.py:65-71`). Both are 440 English questions. |
| "HARP's code randomly splits the data — there is no known/unknown set" | **overstated** | The grouping is correct; only the sub-split within the known group is at answer granularity. |
| Leakage effect scales with the *share of test answers* drawn from seen questions | **refuted by data** | NQ-Open has the smallest share (2.7%) and the largest effect (−13.30). Every AUROC pair needs a correct answer, and correct answers exist only inside known questions — so unknown questions contribute no discriminative pairs at all, and the leakage touches nearly all usable signal everywhere. |
| Effect size tracks the *number of known questions*, inversely | **does not replicate** | Fitted to 4 Qwen points. LLaMA base gives 404→7.77, 506→5.15, 1,267→6.60 — non-monotonic. What modulates the size is unexplained. |
| HARP's subspace is the trailing `d − 0.95d` directions | **imprecise** | That is their §4.3 rule (~179/205 dims). Their §5.3 fixes **256** globally and every Table-1 number is at it. |

---

## 4. Open questions

1. **Is the layer mode really low-rank?** Sweep `r_L` over 1..9. If `r_L=1` matches `r_L=5`, the
   multi-layer premise collapses. Cheap; should precede any writing.
2. **Is the signal separable?** Tucker core vs unrestricted PCA on `vec(H)` at matched dim.
3. **Learned vs weight-derived subspace.** Principal angles between `span(U_F)` and HARP's
   `span(V_R)`. Only immediately comparable for `core_max` — the other streams live in R^(2D).
4. **The layer window is absolute and probably shouldn't be.** `W = {15..23}` is hardcoded
   regardless of depth: 47–72% on LLaMA (32 layers), 54–82% on Qwen (28). Every cross-architecture
   comparison is confounded by window placement. Going into ablations per Devansh.
5. **Does two-sided pooling earn its width?** `S` and `V` cost 2D per layer; ablate against q95
   alone.
6. **Judge agreement** — does HARP's own `DatasetJudge` reproduce our labels? Decides whether we
   can say "HARP's honest TyDiQA number is 79.9" or only "on our labels, their split is worth
   11.85 points". Script `51_judge_agreement.py` written, not yet run.

---

## 5. Method reference

Per answer, over completion tokens only, layers `W = {15..23}`:

- **static** `S` in R^(9 x 2D) — `[q0.95(h) ; q0.05(h)]`, two-sided per layer
- **velocity** `V` in R^(8 x 2D) — `[q0.95(D) ; q0.05(D)]`, `D_l,t = h_{l+1,t} − h_{l,t}`
- **core** `C` in R^(9 x D) — `max_t h_{l,t}`, elementwise (one-sided, hence width D not 2D)
- **kinematic** `k` in R^30 — unused by the six conditions

Decomposition: winsorise [0.5, 99.5] then median/(IQR/1.349) per (layer, feature), train-fold only;
then truncated HOSVD giving `G_n = U_L^T H_n U_F` in R^(r_L x r_F), flattened, RF or LR readout.

| condition | input | r_L | r_F | dim |
|---|---|---|---|---|
| `core_max` | C (9 x D) | 5 | 64 | 320 |
| `q_static` | S (9 x 2D) | 5 | 64 | 320 |
| `q_velocity` | V (8 x 2D) | 4 | 64 | 256 |
| `core_concat` | C + V | 5,4 | 64 | 576 |
| `joint_tensor` | [S;V] (17 x 2D) | 8 | 64 | 512 |
| `triple_concat` | C + S + V | 5,5,4 | 64 | 896 |

**Naming trap.** The npz field `static_max` is the **core** stream (max-pool), not `static`.
`44_eval_phase3.py:724` maps `feats["core"] = d["static_max"]`.

---

## 6. Operational knowledge

Things that cost real time. Written down so they cost it once.

**Clusters.** Newton = GPU jobs (`highgpu`, `normal`). Stokes = high-RAM jobs. Shared
filesystem — same paths, no copying — but **separate schedulers**, so `sbatch` from a terminal
on the target cluster.

**Stokes partitions** (measured 2026-08-21): `highmem` ~3 TB/node — the only one that fits the
TriviaQA evals; `normal*` ~187 GB/node; `preemptable` ~187 GB and preemptible. Note Newton
*also* has a `normal`, with a different ceiling — same name, different machine.

**Login nodes cap you at 100 processes** (`RLIMIT_NPROC`). numpy/OpenBLAS spawns 32 threads on
import and dies, reported misleadingly as `KeyboardInterrupt`. Set
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, or use a CPU job.

**`sbatch --wrap` runs under `sh`**, where `conda activate` (a bash function) does not exist. Use a
real `#!/bin/bash` script.

**HARP needs transformers 4.5x**; our pipeline runs 5.13. `main.py:105` uses `model.model.config`
and `bleurt_pytorch` needs `pytorch_utils.find_pruneable_heads_and_indices`, both gone in 5.x.
Solution: a venv built with `--system-site-packages` over `hal-det`, so transformers 4.51 shadows
5.13 while torch 2.13+cu126 comes through — ~1 GB instead of ~8 GB for a standalone env.

**HARP's `main.py` hardcodes three GPUs** (`data_device=cuda:1`, `model_device=cuda:2`), and
`cuda:1` does **no compute** — it is a tensor holding pen. `data_device="cpu"` costs one PCIe copy
per beam and frees a whole GPU. That is the only edit we make to their code.

**Races in HARP's code**: the lm_head SVD cache is keyed on **model**, not dataset
(`main.py:109`), and `mkdir()` is check-then-create (`main.py:64-68`). Run one dataset alone
first, then fan out.

**Our own race**: `49_harp_adapter.py:287` writes one `adapter_summary.json` per model, so the
adapter must run once with `--dataset all`, never per-dataset in parallel.

**Model-keyed dicts are landmines.** `CONTEXT_LIMITS` in `42_extract_phase2.py` had no entry for a
new model and killed extraction with a bare `KeyError` seconds in — after generation had spent
GPU-hours. When adding a model, grep for dicts keyed on `model_folder`.

**Generation has no checkpointing.** A timeout discards the whole run. TriviaQA gen needs 16 h
(observed 7h40m projected; a 6 h budget died at 77.6%).

**TriviaQA is the only dataset with a multi-GB download** (2.14 GB, `rc.nocontext`) and compute
nodes get ~1.5 MB/s. Prefetch it, with `HF_HUB_DOWNLOAD_TIMEOUT` raised.

**The A2 post-norm gate compares a bf16 argmax against an fp32 one.** The model is loaded in bf16
so `out.logits` is bf16; the check recomputes `lm_head` in fp32. Near-ties disagree legitimately.
With a 13-token probe, one near-tie = 12/13 = 0.9231 and the gate fails. Now tie-tolerant, on 200+
tokens. Base models trip this more than Instruct — flatter distributions on generic probe text.

**Every pipeline stage is resumable** — each checks its *last* artifact (a half-finished stage
leaves early files but never the final one). `FORCE=1` overrides.
