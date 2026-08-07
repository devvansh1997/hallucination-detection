# Session 01 — Repo Audit

Repo root for all paths below: `hallucination-detection/` (git repo, branch `main`, clean except two
untracked files `23_evaluate_phase2_head_tucker_and_lookback.py` and `run_debug_attn.slurm` — not
touched by this session). Verified locally against the actual cached tensors on disk using the
`viewpoint` conda env (torch 2.11.0+cu128, scikit-learn 1.7.2, numpy 2.2.6) — no GPU/model access
was needed for verification, only cached `.pt` files.

## What's missing / blocked

Nothing in A1–A5 is unanswerable from the repo — all five were fully resolved, including running the
actual evaluator against the actual cached data. A6/A7 are report-only and fully resolved. The one
real gap: **there is no dedicated `.slurm` template for `21_generate_maxpool_datasets.py`**, the
script that produced the cached data behind the Phase-1 baseline (see A7). Not a blocker for Part B,
since Part B only needs the already-cached pooled tensors and runs entirely on CPU.

## Critical corrections to the session brief

Three claims in the session brief do not match the actual repo/data and materially change how Part B
should be read:

1. **Generation is NOT greedy/deterministic.** `config.yaml` sets `do_sample: true` (temperature 0.5,
   top_k 5, top_p 0.99), not `do_sample=False`. The 10 "beams" are samples drawn during a
   `num_beams=10` search with sampling enabled, and no RNG seed is set before any `model.generate()`
   call anywhere in the repo — so re-running generation is **not** reproducible run-to-run, only the
   downstream analysis (Tucker/classifiers) is seeded. See A1.
2. **Class balance is inverted from the brief.** The actual cached beam-level labels are
   **61.18% hallucinated / 38.82% truthful** (4998/8170 vs 3172/8170), not "~62% truthful / 38%
   hallucinated." See A2.
3. **0.8094 does not come from the plain 320-dim Tucker core.** It comes from a 346-dim variant (core
   + 26 raw geometric features) in `22_evaluate_phase1_kinematics_and_q.py`, and the exact figure is
   sensitive to sklearn version even with identical seed/data/split. See A5 — this is the most
   important finding in this audit and directly shapes the E1 anchor in Part B.

---

## A1 — Generation

**Three separate generation code paths exist; the one that actually produced the data behind the
0.8094 baseline is `21_generate_maxpool_datasets.py`, not either `01_*` script.**

- `01_generate_and_extract.py` (early-stop variant, saves `{dataset}_pooled.pt`) — superseded, no
  matching cache file exists on disk for truthfulqa.
- `01_generate_full_beams.py` (no early-stop, saves `{dataset}_pooled_fullbeams.pt`, **mean**-pools
  over **all** layers) — produced `data/llama-3.1-8b-instruct/truthfulqa_pooled_fullbeams.pt` (4.0 GB,
  2026-07-08). This is what scripts `02`–`19` consume (via a `--suffix _fullbeams` convention). Per
  `AGENTS.md:55` ("Mean-pooling deprecated → max-energy pooling preferred") this line is legacy.
- `21_generate_maxpool_datasets.py` (standalone, self-contained — no shared `data.py`) — produced
  `data/llama-3.1-8b-instruct/truthfulqa_pooled_maxenergy.pt` (577 MB, 2026-07-16). **This is the file
  that `22_evaluate_phase1_kinematics_and_q.py` consumes and that reproduces the ≈0.8094 number
  (see A5).** All A1/A2/A3 answers below describe this path since it's the one behind the number in
  the session brief.

**Generation call**, `21_generate_maxpool_datasets.py:116-124`:
```python
outputs = model.generate(
    **inputs, max_new_tokens=gen_cfg["max_new_tokens"],
    eos_token_id=list(eos_ids),
    do_sample=gen_cfg["do_sample"], temperature=gen_cfg["temperature"],
    top_k=gen_cfg["top_k"], top_p=gen_cfg["top_p"],
    num_beams=gen_cfg["num_beams"],
    num_return_sequences=gen_cfg["num_return_sequences"],
    output_hidden_states=True, return_dict_in_generate=True,
    pad_token_id=tokenizer.eos_token_id, early_stopping=True)
```
kwargs pulled straight from `config.yaml:42-49`: `max_new_tokens: 64, temperature: 0.5, top_k: 5,
top_p: 0.99, num_beams: 10, num_return_sequences: 10, do_sample: true`. No `torch.manual_seed`/
`set_seed` call anywhere in the file.

**Storage.** Generated token sequences themselves are **not** persisted — only the pooled activations
survive. Per-beam prompt length (`inputs.input_ids.shape[1]`, line 113) is used transiently to slice
`gen_ids = outputs.sequences[:, prompt_len:]` (line 128) and then discarded — it is **not** written to
disk. What *is* saved (`21_generate_maxpool_datasets.py:178-183`):
```python
torch.save({
    "all_emb": all_emb,                       # list[Tensor(9, 4096), bfloat16], len 8170
    "all_hallucination_flag": all_flags,       # list[bool], len 8170
    "all_is_known": all_is_known,              # list[bool], len 817 (per-prompt)
    "prompt_indices": all_prompt_idx,          # list[int], len 8170, aligned with all_emb
}, out_path)
```
at `../data/{model_folder}/{dataset}_pooled_maxenergy.pt`. Verified by loading the actual file:
817 prompts × 10 beams = 8170 beams exactly, `prompt_indices` is a deterministic `[0]*10 + [1]*10 +
...` ordering aligned positionally with `all_emb`/`all_hallucination_flag` (no reconstruction needed
for Part B — see A4/B1).

## A2 — Labeling

Computed **inline** in `21_generate_maxpool_datasets.py:136-146` (own reimplementation, not a call
into `HARP-Code/DatasetJudge.py`, which is legacy/unused):
```python
r = rouge.compute(predictions=r_candidates, references=correct) if correct else {"rougeL": 0.0}
rl = r["rougeL"]
...
bs = bleurt.compute(predictions=candidates, references=all_refs)
max_correct_b = max(bs["scores"][:len(correct)], default=0)
is_correct = (rl >= 0.7) or (max_correct_b > 0.5)
```
via `evaluate.load("rouge")` / `evaluate.load("bleurt", config_name="BLEURT-20")` (lines 79-80). This
matches the session brief's stated formula exactly: `ROUGE-L >= 0.7 OR BLEURT > 0.5`. (Note: the
sibling `01_generate_full_beams.py` has a *different*, more complex contrastive judge with an extra
"advantage" margin term — `judge_contrastive()` — but that script did not produce the data behind the
0.8094 number, so it's not the operative labeling rule here.)

Labels are cached in the same `.pt` file as generation, key `all_hallucination_flag` (bool, `True` =
hallucinated, set via `all_flags.append(not is_correct)`, line 166), positionally aligned with
`all_emb` and `prompt_indices`.

**Actual class balance (measured directly from the cached file, no GPU needed):**
- 8170 beams total, 817 prompts × 10 beams each (uniform — confirmed by prompt-count distribution).
- **4998 hallucinated (61.18%), 3172 truthful (38.82%)** — inverted from the brief's "~62%
  truthful / 38% hallucinated."
- 505/817 prompts (61.8%) are "known" (at least one of their 10 beams is correct).

## A3 — Extraction

Layers hooked: `W_START, W_END = 15, 24` (`21_generate_maxpool_datasets.py:93`) →
`range(15, 24)` = layers **15–23 inclusive, 9 layers**. Indexing convention confirmed:
`hidden_states[step][l + 1][b]` (line 158) — the `+1` means **`hidden_states[0]` is the embedding
layer**, so transformer layer `l`'s output sits at hidden_states index `l+1`, consistent across the
whole repo.

Dtype: model loaded `dtype=torch.bfloat16` (line 85); hidden states are produced and pooled in
bfloat16 and saved to disk in bfloat16 (verified directly on the cached file: `all_emb[0].dtype ==
torch.bfloat16`). Cast to float32 happens later, inside the evaluate scripts (e.g.
`22_evaluate_phase1_kinematics_and_q.py:145`: `X = torch.stack(data["all_emb"]).float()`), not at
extraction time.

**Pooling window is completion tokens only — prompt tokens are excluded.** Exact code,
`21_generate_maxpool_datasets.py:148-163`:
```python
if num_gen == 0 or len(gids) == 0:
    H_pooled = torch.zeros(W_END - W_START, D)
else:
    layers = []
    for l in range(W_START, W_END):
        tokens = []
        for step in range(1, len(hidden_states)):
            if step - 1 >= len(gids):
                break
            tokens.append(hidden_states[step][l + 1][b].cpu())
        layer_cat = torch.cat(tokens, dim=0) if tokens else torch.zeros(0, D)
        # Max-pool across token dimension
        layers.append(layer_cat.max(dim=0).values if layer_cat.shape[0] > 0
                      else torch.zeros(D))
    H_pooled = torch.stack(layers, dim=0)  # (9, D)
```
`hidden_states[0]` (the prompt forward pass) is never touched — the loop starts at `step=1`, which in
HF's `generate(..., output_hidden_states=True)` convention is the first *generated* token's step. This
is genuinely completion-only, positive **max**-pooling (not mean) across the token dimension per
layer, matching the session brief and `AGENTS.md:55`'s "max-energy pooling preferred."

## A4 — Phase-1 features

**No separately-cached 320-dim core-feature file exists anywhere in the repo.** The Tucker/HOSVD
projection is computed **on the fly, inside each evaluate script**, from the cached *pre-Tucker*
pooled tensors (`{dataset}_pooled_maxenergy.pt`, shape-per-beam `(9, 4096)`). The canonical Phase-1
evaluator is `22_evaluate_phase1_kinematics_and_q.py`.

**Fit on train only — confirmed, and confirmed consistently across every one of the ~18 evaluate
scripts in the repo** (grepped all `compute_ul_ud`/`gram_factor_matrices` call sites): every single
one is called as `compute_ul_ud(X[train_idx])` / `compute_ul_ud(X_all[train_idx_arr])` /
`compute_ul_ud(X[t_idx])` — never on the full population. For the canonical script:
```python
# 22_evaluate_phase1_kinematics_and_q.py:176
U_L, U_D = compute_ul_ud(X[t_idx])   # t_idx = TRAIN beam indices only
```
No fit-on-all leak exists in this codebase. `compute_ul_ud` (lines 98-111) does eigh-on-Gram exactly
as `[[project-codebase-v1]]` describes: `torch.linalg.eigh` on `X_f @ X_f.T` for both the layer-mode
(`R_L=5`) and hidden-mode (`R_D=64`, chunked accumulation for memory) Grams, giving `U_L (9,5)` /
`U_D (4096,64)`, then `G = (X @ U_D).transpose(1,2) @ U_L` reshaped to `(N, 320)` (line 183) —
`5 * 64 = 320`, matching the brief.

**Row keying:** cores are produced by projecting the *entire* stacked `X` (all 8170 beams, train and
valid together) through the train-only `U_L, U_D` in one shot (line 180-183: `X.float() @ U_D` uses
the full `X`, not `X[t_idx]`) — so ordering is inherited directly and deterministically from
`all_emb`/`prompt_indices` in the source `.pt` file. Beam→prompt mapping is exact and requires no
reconstruction: `prompt_idx[i]` for beam `i` is stored explicitly.

## A5 — Split protocol

**Prompt-grouped, not beam-level** — this is the single most important correction for Part B's
premise. Exact code, `22_evaluate_phase1_kinematics_and_q.py:153-163`:
```python
# HARP split
known_idx = np.where(is_known)[0]                      # PROMPT indices (len 817), not beam indices
np.random.seed(RANDOM_SEED); np.random.shuffle(known_idx)
s = int(len(known_idx) * 0.75)
tp = set(known_idx[:s]); vp = set(known_idx[s:])
vp.update(np.where(~is_known)[0])                       # all "unknown" prompts go to valid
t_mask = np.array([prompt_idx[i] in tp for i in range(N)])   # per-BEAM mask, built from prompt sets
v_mask = np.array([prompt_idx[i] in vp for i in range(N)])
t_idx = np.where(t_mask)[0]; v_idx = np.where(v_mask)[0]
```
`RANDOM_SEED = 42` (line 22). This *is* a `GroupKFold`-style split in spirit (all 10 beams of a given
prompt land on the same side), just implemented as a single fixed 75/25-of-known partition (HARP's
"known/unknown" protocol) rather than sklearn's `GroupKFold`. **So the classic "sibling beams split
across train/val" leakage that Part B's premise hypothesizes does NOT occur in the script that
produced 0.8094** — verified empirically too (see below): loading the actual data gives Train=3780 /
Valid=4390 beams, and every beam's prompt-mask is prompt-consistent by construction. The second half
of the session's hypothesis — that features may track question difficulty/prompt identity rather than
per-beam truthfulness — is untouched by this finding and remains the live question; a prompt-disjoint
split doesn't rule out the classifier keying off cross-prompt "hard question" signatures that
generalize to unseen prompts. E2–E4 in Part B are still exactly the right diagnostic for that.

Classifier: `RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42,
n_jobs=-1)` (`22_evaluate_phase1_kinematics_and_q.py:230-231`), features standardized per-variant with
`StandardScaler` fit on train only (lines 225-227).

**Where "0.8094" actually comes from — verified by literally running the script against the cached
data, twice, in two different environments:**

| Variant | Features | dim | RF AUROC (sklearn 1.9.0, system Python) | RF AUROC (sklearn 1.7.2, `viewpoint`) |
|---|---|---:|---:|---:|
| V1 | Tucker core only | 320 | 0.7985 | 0.8054 |
| V2 | Core + raw Q-stat/kinematics | 346 | 0.8037 | **0.8094** |
| V3 | Core + Gram-Schmidt-orthogonalized Q/kinematics | 346 | **0.8093** | 0.8020 |
| V4 | Orthogonalized geometry alone | 26 | 0.6176 | 0.6159 |

(Full commands: `python 22_evaluate_phase1_kinematics_and_q.py --model_folder llama-3.1-8b-instruct
--dataset truthfulqa --suffix _maxenergy`, CPU-only, ~15s.)

**Conclusion: the exact "0.8094" figure is not the pure 320-dim core (V1) — it lands on V2 or V3
depending on which of two very similar 346-dim geometric-augmented variants and which sklearn version
you use.** With `viewpoint`'s sklearn (1.7.2, closer to the user's real working environment), V2 hits
0.8094 to 4 decimal places; with a newer sklearn (1.9.0) V3 gets closest (0.8093) and V2 drops to
0.8037. The plain 320-dim core alone is consistently the *lowest* of the three real variants (0.7985–
0.8054) — a 0.004–0.011 AUROC gap from "the" reported number, i.e. right at or just outside the task's
own ±0.01 replication tolerance depending on sklearn version. **This is a real reproducibility
sensitivity in the existing pipeline** (RF internals differ enough between sklearn 1.7.x and 1.9.x to
move AUROC by ~0.005–0.01 with identical seed/data/split), not a bug introduced by this audit.

**Handling in Part B:** Since Part B's spec (B1–B4) explicitly scopes "core features" to the 320-dim
Tucker core, E1's replication anchor targets **V1 (pure 320-dim core)** and reports the real number
obtained (~0.80–0.81 depending on sklearn version) rather than forcing a fabricated exact 0.8094 — the
metrics JSON records which sklearn/variant combination the human's cluster run lands closest to. This
is flagged again inline in the script's output.

## A6 — Phase-4 forensics (report only)

From `25_evaluate_phase4_ads_btd.py` (subagent-verified, spot-checked):

- **Layer window:** the same layer 15–23 reasoning window as Phase 1, not final-layer/all-layer —
  `LAYERS = list(range(15, 24))` (line 21), hooks registered only on those layers
  (lines 456-461), and the real-feature beam tensor is explicitly `(1, 1, 9, 4096)` (line 512), the
  `9` being `len(LAYERS)`.
- **Pooling happens before the `V_R`/`P_R` projection.** Tokens are mean-pooled per layer first
  (lines 504-511: `torch.cat(tok_vecs, dim=0).mean(dim=0, keepdim=True)`), producing a token axis of
  size 1, and only then is the pooled `X_beam` multiplied by `P_R`/`P_S` (lines 514-516:
  `X_beam.float() @ P_R_gpu`, inside `dual_stream_btd`). So: pool-then-project, not project-then-pool.
- **Dtype:** float32 for every projection matmul. `P_S`/`P_R` are stored bfloat16 but explicitly cast
  via `.float()` at every use site (lines 126-127, 464-465, 359-360/365).
- **`V_R` itself is weight-derived, not activation-fit** — built once from `lm_head.weight` × final
  RMSNorm gamma via SVD (`build_vocab_anchor()`, lines 28-58), so there's no train/valid leakage in
  the basis construction. The *downstream* per-beam Tucker factors inside `dual_stream_btd` are fit
  independently per beam with no visible train-only gating, but since each beam gets its own
  independent fit (not a shared population basis), this doesn't create cross-sample leakage the way a
  fit-on-all population Tucker would.

## A7 — Cluster run conventions

Seven `.slurm` files, one consistent template (`-p gpu --gpus=1 --mem-per-cpu=8G -C gmem80`, `-c 4`
except `run_debug_attn.slurm` at `-c 2`), identical conda/module block:
```
module load anaconda3/2022.05
module load cuda/12.6
source /share/apps/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate hal-det
cd /home/devansh/Hallucination-Detection/hallucination-detection
```
Env vars (`TF_FORCE_GPU_ALLOW_GROWTH`, `HF_METRICS_CACHE`) are **not** set in the `.slurm` files —
they're set inside the Python entry scripts themselves, before any TF/HF import (e.g.
`21_generate_maxpool_datasets.py:14`). No Makefile or other launch abstraction exists; it's raw
`sbatch run_*.slurm`, one file per dataset/model.

**Gap:** no `.slurm` file launches `21_generate_maxpool_datasets.py` (the script that produced the
data behind the 0.8094 baseline) — the six dataset-specific `.slurm` files all invoke
`01_generate_and_extract.py` or `01_generate_full_beams.py`. Not a blocker here since the new script
in Part B is CPU-only and reads already-cached data, but worth flagging: **the human should run the
new script on a CPU-only node/allocation, no `sbatch` GPU template needed** — a plain
`python 26_grouped_baseline.py ...` after `conda activate hal-det` (or `viewpoint`, if that's the
intended env going forward) and `cd` to the repo is sufficient.

README.md is stale (describes a `data.py`/`main.py` toy pipeline that no longer exists — deleted in
commit `73aa118`); the numbered scripts + `AGENTS.md` are the actual source of truth.
