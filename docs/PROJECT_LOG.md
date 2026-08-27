# Project log — hallucination detection via multilinear decomposition

Running record of findings, corrections and operational knowledge, kept current as work
progresses. Every claim here carries its evidence; anything retracted stays visible with the
reason, because knowing what we got wrong is as load-bearing as knowing what we got right.

Last updated: 2026-08-24.

---

## 1. Where things stand

**Models.** `qwen-2.5-7b-instruct` (complete), `llama-3.1-8b` **base** (4 of 4 datasets, both protocols, all six conditions — complete),
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
| LLaMA-3.1-8B base | TriviaQA | 52.5 | 8,220 | 84.79 | 89.52 | +4.72 |
| LLaMA-3.1-8B base | NQ-Open | 88.8 | 1,267 | 85.36 | 91.96 | +6.60 |
| LLaMA-3.1-8B base | TyDiQA-GP | 48.9 | 404 | 83.23 | 91.00 | +7.77 |

TriviaQA/Qwen answer-level covers 5 of 6 conditions: `joint_tensor` was OOM-killed in
`slurm/phase3_answersplit_trivia.slurm`, which requested `--mem=128G`. It is a maximum, so the
true value can only be higher and the +1.35 is a lower bound. LLaMA TriviaQA has all six on
both arms (180G).

**HARP reproduction (Qwen, their code on our data, proj_dim 256).** TruthfulQA 86.27 vs published
88.1; TriviaQA 92.89 vs 92.8; NQ-Open 83.12 vs 84.0; TyDiQA-GP 91.87 vs 88.4. Mean signed
deviation **+0.21** — no systematic bias, which is what licenses everything downstream.

**HARP on LLaMA base**, their code on our data, proj_dim 256 — all four datasets:

| | TruthfulQA | TriviaQA | NQ-Open | TyDiQA-GP |
|---|---|---|---|---|
| answer-level (their split) | 84.84 | 88.08 | 86.56 | 85.65 |
| question-level (paper) | 79.58 | 82.54 | 80.56 | 74.99 |
| **cost** | −5.26 | −5.54 | −5.99 | −10.66 |
| published | 88.5 | 92.9 | 89.4 | 86.6 |
| vs published | −3.66 | −4.82 | −2.84 | −0.95 |

**Unexplained asymmetry.** Qwen reproduced at mean signed deviation **+0.21**; LLaMA base is
**−3.07**, low on all four. Could be our generations, our labels, or the base checkpoint. It
does not threaten the A/B (both arms share everything), but it weakens any claim about their
*published* LLaMA numbers specifically, and a reviewer would ask.

LLaMA known-question counts: TruthfulQA 506, TriviaQA 8,220, NQ-Open 1,267, TyDiQA-GP 404 —
TriviaQA at 82.5% known vs Qwen's 63.4%.

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

### 2.2b ~~The control for §2.2~~ — **RETRACTED 2026-08-24, see §2.9**

> The argument below rests on HalluGuard being training-free. It is not: the paper's
> Appendix C.1 says *"We train only HALLUGUARD's lightweight projection layers using
> AdamW."* A fitted score carries a leakage term, so it cannot isolate the composition
> term. The three-way AUROC reporting in `53_halluguard_score.py` is still useful for
> comparability, but the **control claim is withdrawn**. Kept below for the record.

#### (retracted) The control for §2.2: does the gap come from leakage or a different test set?

The obvious objection to §2.2, and the first one a reviewer will raise: the two arms are not
evaluated on the same rows, so maybe the 3.8–13.3 points is just the two test populations being
different rather than train/test contamination. Decompose it:

> answer-level − question-level = **leakage** + **population composition**

We could not separate those two terms with HARP and our method alone, because both are fitted and
both carry a leakage term. HalluGuard can, because **it fits nothing** — its per-beam score is a
deterministic function of (prompt, answer) and does not change between arms. Its leakage term is
identically zero, so its delta estimates the composition term on its own. If that delta is small,
the whole of HARP's gap is leakage.

Two facts make the control tighter than it first looks, both verified numerically in
`53_halluguard_score.py --self-test` on 200 synthetic prompts × 10 beams:

- The two arms' valid sets hold **the same number of known-prompt rows** (25% of them) and both
  take **all** unknown rows. They differ only in *which* known rows — so composition differs at
  second order, not first.
- Where they differ sharply is question coverage: the question-level valid set touches **35 of 140**
  known questions, the answer-level one **130 of 140** — 25% vs 93%, against the predicted
  1 − 0.75¹⁰ = 94.4%.

On synthetic data with a fixed score, the measured delta was **−0.12 points**. That is the number
HalluGuard on real data has to be compared against; a similarly small value converts §2.2 from
"the number moves when we change the protocol" into "the number moves *because of leakage*".

Caveat to keep: this assumes the composition effect is of comparable magnitude across methods. Not
guaranteed — but it is a bound where we currently have nothing.

Implemented as three AUROCs from one deterministic scoring pass (all beams / question-level test
rows / answer-level test rows), using the project's own `original_harp_split` and
`answer_level_harp_split` and the same `HARP_SEEDS = [42, 0, 1, 2, 3]`, so the row sets are
literally the ones the other two methods were scored on. Skipped under `--limit`, where truncation
cuts a prompt in half and the partitions would no longer correspond.

### 2.2c LLaMA TriviaQA: our smallest margin, on the dataset with the most known questions

Question-level, LLaMA-3.1-8B base, best of six, RF: **84.79 ±0.23** (`q_static`) against HARP's
**82.54** — a margin of **+2.25**, our narrowest anywhere. Every other cell sits at +4.80 to +9.74.

TriviaQA is also, by a wide distance, the dataset with the most known questions (8,220, against
404–1,267 elsewhere; 82.5% of its prompts are known vs Qwen's 63.4%). Two readings, and we cannot
yet separate them: either the margin narrows where there is enough question diversity that HARP's
detector is pushed toward a transferable rule, or TriviaQA is simply the dataset where their
features suit the task. Note this is the *question-level* margin, so it is not a leakage artifact.

Do not fold this into §2.7's known-question story — that relationship was about the *size of the
leakage cost* and has already been retracted as non-replicating (§3). This is a different quantity
(our margin, not their inflation) and one data point. It is recorded as an observation to check
against the answer-level cell when it lands, not as a mechanism.

**Answer-level landed 2026-08-24: `q_static` 89.52 ±0.04, cost −4.72, margin +1.44.** TriviaQA is
now our weakest cell on *both* protocols (+2.25 question-level, +1.44 answer-level, against +4.80 to
+9.74 and +5.35 to +9.63 elsewhere), which strengthens the observation above rather than resolving it.

**The RF/LR flip replicates, and more sharply than on Qwen.** Question-level, LR wins 5 of 6
(core_max +0.35, q_static +0.25, core_concat +0.65, joint_tensor +0.91, triple_concat +0.48;
`q_velocity` the exception at −0.19). Answer-level, RF wins **6 of 6** by **3.12–5.28** points. Under
the paper's protocol the readouts are within a point of each other in either direction; under the
released one the forest pulls ahead by three to five points on every condition. That is exactly what
the memorisation account predicts, now on a second model and architecture.

**Provenance note.** The two `session06_phase3_partA_triviaqa.json` files are distinguishable without
metadata: the `grouped` block is bit-identical between them (5-fold CV does not depend on the split
unit) while every `harp` condition gains +4.56 to +5.62. `n_train`/`n_valid` do **not** discriminate
— both arms give 61,650/37,950, since each puts 25% of known rows plus all unknown rows in valid.

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

### 2.7 Our labels ARE HARP's labels (audit #1, closed)

Ran HARP's own `DatasetJudge` -- their class, their thresholds, constructed as `main.py:133-137`
does -- over our generations and compared label for label. Qwen2.5-7B, all four datasets,
148,270 beams.

| dataset | beam agreement | known-status agreement | ours / theirs known | flips |
|---|---|---|---|---|
| NQ-Open | 99.94% | 99.89% | 360 / 358 | 4 |
| TriviaQA | 99.71% | 99.67% | 6,316 / 6,331 | 33 |
| TruthfulQA | 99.57% | 99.27% | 666 / 670 | 6 |
| TyDiQA-GP | 99.27% | 99.77% | 302 / 301 | 1 |

**44 of 14,827 questions (0.30%) change known/unknown status.**

This was the load-bearing uncertainty. It decides between two claims:

- *internal* -- "changing only the split unit costs 3.8-13.3 points" -- held regardless, since
  both arms share whatever labels we have.
- *external* -- "HARP's honest TyDiQA number is 79.9, not the published 88.4" -- needed our
  partition to be the one their judge produces. **It is.** The claim stands as written.

**The two judges are independent implementations, which makes the agreement mean more.** Ours
(`is_correct_simple`, `39_generate_dataset.py:240`) was written from the paper's §5.1 description
before we had their code, and uses HuggingFace `evaluate` for both ROUGE and BLEURT-20. Theirs
(`DatasetJudge.judge`) uses the `rouge` pip package and `bleurt_pytorch`. They share no code, no
metric library, and no BLEURT wrapper; they differ even in `>=` vs `>` on the ROUGE threshold and
in how scores are aggregated over a question's alias list. Agreement at 99.3-99.9% is therefore
evidence the *rubric* is applied faithfully, not that one implementation was run twice.

Worth a look sometime: our ROUGE path calls `rouge.compute()` over all references at once and
reads `r["rougeL"]`, which under `evaluate`'s default aggregator is a MEAN across references,
while the docstring says max and their judge takes an explicit max. It evidently does not matter
empirically -- 99%+ agreement -- but the code and its comment disagree.

The judge is a function of (generated text, reference set), not of the model that produced the
text, so agreement across 148,270 Qwen beams is strong evidence the implementation matches for
LLaMA's generations too. Not separately verified there.

**It does NOT explain the LLaMA -3.07 asymmetry.** Labels were the leading suspect and are now
ruled out on Qwen. That asymmetry remains open.

### 2.6 A memorisation signature

Under the question-level split, LR beats RF on TriviaQA for **all six** conditions. Under the
answer-level split the ranking **inverts** — RF wins there and on every other dataset
(`q_static`: 94.03 RF / 93.02 LR answer-level, against 91.7 RF / 92.7 LR question-level).

A forest can carve per-question regions a linear model structurally cannot, so the classifier
better at memorising is exactly the one that gains when the split permits it. This fell out of
data collected for another purpose, which is what makes it worth something.

---

### 2.8 `halluguard_true.py`: the file implementing the paper's formula cannot scale

> **Scope corrected 2026-08-24.** Everything in this section is about the file
> `Score/halluguard_true.py`, which is **not** the code path behind the paper's published
> numbers. See §2.9. The findings stand as statements about that file; they are *not* a
> critique of HalluGuard's reported results.

Pilot 1 (job 774134) and pilot 2 (774273), TyDiQA-GP, Qwen2.5-7B-Instruct, our pinned generations,
their `Score/halluguard_true.py` imported **unmodified**.

**(a) The released path does not execute.** Their line 126 calls `torch.linalg.eigvalsh`, which has
no half-precision kernel on either device — `NotImplementedError` for `Half` and for `BFloat16`,
verified locally on CPU and observed on 200/200 beams on CUDA. Their only call site,
`Beam Search/reward_model/ntk_reward.py:26`, loads `torch_dtype=torch.float16 if device == "cuda"`
and then calls with `param_subset="last_block"` at line 72. That combination crashes on the first
beam. The same call site passes `input_ids=prompt_ids[:1], generated_ids=prompt_ids[1:]` — the
entire prompt treated as generated — so T is hundreds of tokens, needing hundreds of GB.

**(b) Memory is linear in completion length, so coverage is length-selected.** Line 124 stacks the
T per-token gradients into `[T, P]` while the list of T separate vectors is still alive, giving a
peak of ~2·T·P·4 bytes above a 30.5 GB fp32 model. The allocator log pins P: single-gradient
requests of 933,232,640 bytes ⇒ **0.93 GB per generated token**. Observed cutoff on an 80 GB H100
is a hard **T ≤ 27** (`expandable_segments:True`; T ≤ ~20 without it). Coverage 81% at 400 beams.

The selection turned out **benign on class balance**: hallucination rate 57.7% among scored vs
57.9% among failed, a 0.2 point shift. That was not guaranteed and is the reason the numbers below
are readable at all.

**(c) `det(K)` is inert — and my predicted mechanism for why was wrong.** I expected collinear
consecutive-token gradients to push eigenvalues onto the `1e-8` clamp so `det_k = exp(log_det)`
would underflow to 0. It does not: **median det(K) = 0.407, median κ = 3.92**. The gradients are
nearly *orthogonal*, not collinear — which is what unit vectors in 233M dimensions do. The
conclusion survives the wrong reasoning: `det(K)` contributes **0.0821** of the score's spread, and
`log(σ_max) − 2log(κ)` alone scores **0.3360** against the full score's **0.3370**. The determinant
moves the AUROC by 0.001. Note the gradients are still needed for κ; it is `det(K)` specifically,
the term the method is named for, that does no work.

**(d) Completion length beats the method.** Oriented as a hallucination detector, the score reaches
**0.6630** pooled; **raw token count reaches 0.6763**. ρ(score, T) = **−0.692**, so the score is
substantially an inverse length proxy. Restricting to T ≤ 27 truncates the range of T, which
*suppresses* length's AUROC — so the full population can only favour length more.

Raw (unflipped) AUROC is 0.3370 pooled, 0.2043 within-prompt: their score runs opposite to our
convention (1 = hallucinated). Reported both ways rather than flipped silently; which direction
their paper intends still needs checking against the paper.

| | pooled | within-prompt |
|---|---|---|
| HalluGuard, as computed | 0.3370 | 0.2043 |
| HalluGuard, oriented as a detector | 0.6630 | 0.7957 |
| completion length alone | **0.6763** | *pending* |

**Open:** the within-prompt length baseline was the gap — 0.7957 is where the method looks
strongest and we had nothing to compare it to. Added in `e64595c`, not yet measured on real data.

---

### 2.9 HalluGuard's repo contains three scores — **partially retracted, see §2.11**

> The claim that the default path is the proxy is **wrong**. The default path calls
> `halluguard_true`. What survives: the three implementations exist and differ; the legacy
> helpers discard the covariance matrix; the method is not training-free. Kept for the record.

Devansh asked the obvious question — *they report Llama2-70B, so how did they afford this?* The
answer is that they did not run the expensive thing. Their repo holds two implementations:

| | `Score/func/metric.py::getNTKS3Score` (default) | `Score/halluguard_true.py` |
|---|---|---|
| what K is | covariance over the **10 sampled generations** | `G Gᵀ` over the **T decoding steps** |
| needs gradients | **no** — forward hidden states only | yes, `∇_θ log p(y_t)` per token |
| cost at 7B | negligible | **0.93 GB per generated token** |
| runs on 70B | yes | no |

**The default path never computes the published formula.** Read `func/metric.py:196-215`: it builds
`CovMatrix = np.cov(...) + 1e-3·I` — and then **discards it**. The next line is

```python
# For now, use a simplified approach: just the norm of the embedding
# This avoids the matrix dimension issues while maintaining the concept
residual = np.sqrt(emb @ emb)
```

so `det(K)` is never computed, `κ(K)` is never computed, and the score reduces to

> `‖mean-pooled hidden state at layer L/2‖ × mean_t exp(‖h_t − h_{t−1}‖)`

against the paper's `det(K) + log σ_max − log κ(K)²`. Note also **mean** where the paper specifies
**max** over t (`metric.py:240`).

**Their own repo documents this.** `Score/TECHNICAL_SPEC_VERIFICATION.md` grades `getNTKS3Score` as
"❌ Not NTK: no θ-Jacobians; over sequences, not steps; amplification is mean, not max", and
concludes: *"Satisfied only when using `halluguard_true.py`. Default scripts still use the proxy."*

**Which one produced the tables?** Not verified, but the evidence points one way: the paper's own
Appendix C.1 describes the proxy's structure, not the true one — middle layer `L/2`, ridge
`α = 1e-3`, `K = 10` generations per input, fp16, "for each **set** of generations we form a
task-specific NTK feature matrix". The README says K is "the NTK Gram matrix (**over generated
outputs**)". And the proxy is the only path that can run Llama2-70B at all.

**Also: HalluGuard is not training-free.** Appendix C.1: *"We train only HALLUGUARD's lightweight
projection layers using AdamW, learning rate from {1e-5, 5e-5, 1e-4}, weight decay from {0, 0.01}.
The best setting is chosen on a held-out validation split."* No such layer exists anywhere in the
released code, and no weights are published. So the released code is missing a trained component
that the paper's numbers depend on.

**Useful alignments for us.** Their decoding config is temperature 0.5, top-p 0.95, top-k 10, and
**K = 10 candidates per input** — very close to our 10 beams. They evaluate on NQ-Open and
TruthfulQA, both of which we have. And they label with the dual ROUGE / LLM-judge regime from
Janiak et al. 2025, the same "Illusion of Progress" paper already in our landscape notes.

**Consequence for our plan.** There is no "rerun their code" option, because the two candidate
codes compute different things and neither is the paper. The choices are: reimplement the proxy
(cheap — needs only forward hidden states we may already have), reimplement the paper's formula, or
report the discrepancy itself. My recommendation is the third plus the first: the discrepancy is
the finding, and the proxy is cheap enough to measure alongside it.

---

### 2.10 Running HalluGuard's **legacy proxy** on our data: no score has a consistent direction

> Title corrected 2026-08-27. `54_halluguard_proxy.py` implements the legacy helper, which is
> **not** the default path. The measurements stand; what they are measurements *of* is narrower
> than originally stated.

`54_halluguard_proxy.py`, four of eight cells landed (Qwen 3 datasets, LLaMA-3.1-8B base TriviaQA).
Each cell shows raw AUROC / sign-flipped, per beam except B which is per question.

| model | dataset | beam HR | A as-coded | C norm alone | D length alone | B as-published (Q) |
|---|---|---|---|---|---|---|
| llama | triviaqa | 52.5% | 0.410 / 0.590 | 0.387 / 0.613 | 0.665 / 0.335 | 0.394 / 0.606 |
| qwen | nq_open | 96.9% | 0.526 / 0.474 | 0.263 / 0.737 | 0.768 / 0.232 | 0.302 / 0.698 |
| qwen | truthfulqa | 43.3% | 0.519 / 0.481 | 0.616 / 0.384 | 0.358 / 0.642 | 0.559 / 0.441 |
| qwen | tydiqa_gp | 59.1% | 0.523 / 0.477 | 0.291 / 0.709 | 0.634 / 0.366 | 0.357 / 0.643 |

**The finding is not "it scores X". It is that no score here has a fixed sign.** For every one of
A, B, C and D there is no single orientation clearing 0.5 in all four cells — the direction reverses
between datasets and between models. A detector whose sign depends on which dataset you point it at
is not a detector, whatever its magnitude.

**A is at chance on Qwen (0.519–0.526 across three datasets with hallucination rates of 43%, 59% and
97%) but not on LLaMA (0.410).** So it is not uniformly inert; it is inconsistent.

**The amplification term destroys the signal the norm carries.** On Qwen, C alone sits at 0.263,
0.616 and 0.291 — well off chance — while A sits at 0.52 in all three. Multiplying by
`mean_t exp(‖Δh‖)` washes a strong (if inconsistently-signed) quantity into noise. On LLaMA it does
not: |A − C| = 0.023 there against 0.10–0.26 on Qwen.

**κ is an artifact of the ridge, and more so on LLaMA.** With N=10 generations in D≈3.5–4k dims the
covariance has rank ≤ 9, so λ_min is the ridge and κ = λ_max/α. λ_min sits within 5% of the ridge on
**94.0%** of LLaMA prompts and 31–64% of Qwen's. The "spectral instability penalty" is therefore
measuring λ_max times a constant, not conditioning. κ ranges 85–194 on LLaMA against 830–7586 on
Qwen — a 40× difference in the spread of the 10-generation cloud between the two models.

**What this does NOT establish.** Every cell was computed on OUR generations, which come from
sampled beam search (`config.yaml`: `num_beams: 10`, `do_sample: true`, top-p 0.99, top-k 5) while
their Appendix C.1 uses plain nucleus sampling (top-p 0.95, top-k 10, K=10). Ten beams from one beam
search are far more alike than ten independent samples, and this method measures precisely the
spread of those ten. The cloud may be degenerate before the method sees it. Temperature 0.5 does
match.

One thing the confound does not obviously explain is the **sign reversal**: attenuation from
degenerate inputs should pull scores toward 0.5, not flip them between +0.26 and +0.62. But C's
values are far enough from chance that this is worth testing rather than asserting.

**Open:** the nucleus-sampling rerun on TyDiQA (440 prompts, ~40 min) closes the one objection that
is ours. Remaining cells: Qwen TriviaQA, LLaMA on tydiqa/truthfulqa/nq_open.

---

### 2.11 The default path cannot return a number on a GPU, and that settles it

Corrects §2.9. The default scripts **do** call `halluguard_true`. But:

- `gpu_evaluation_all.py:78` loads `torch_dtype=torch.float16 if device.type == "cuda"`.
- `torch.linalg.eigvalsh` has no fp16 kernel, so `halluguard_true.py:126` raises every call. Observed
  directly: 200/200 beams, `NotImplementedError: "linalg_eigh_cuda" not implemented for 'Half'`.
- `gpu_evaluation_all.py:443-446` catches **every** exception and returns `0.0`. Same handler at
  `gpu_evaluation_llm.py:534-536` and `evaluation.py:185-187`.
- The Beam Search driver `run/run_score.py:75` uses `bfloat16` — also no kernel.

**The argument that needs no speculation:** a constant score has an AUROC of exactly 0.5. The paper
reports 71–90. Therefore the published numbers were not produced by the released default path on a
GPU. We are *not* claiming their results are zeros — we are claiming that whatever produced them is
not in the repository in runnable form.

Stronger than the retracted §2.9 story, because it does not depend on guessing which script they ran.
With the scale arithmetic alongside it (Llama2-70B needs 261 GB for the fp32 model alone, before
3.19 GB per token of gradients), no released configuration yields a real number at the reported scales.

**Open — the decisive experiment.** Run *their own documented quickstart* and look at the output.
`Score/README_PIPELINE.md` gives it verbatim: `./Score/run_pipeline.sh --model gpt2 --dataset coqa
--device cuda --num_generations_per_prompt 2 --fraction_of_data_to_use 0.01`. GPT-2 is small enough
that memory is not a factor, so this isolates the dtype/exception path cleanly. If every
`halluguard_score` in the output pickle is 0.0, the finding is settled with their command, their
model, their data — not our adaptation of anything. Runs on a laptop in minutes.

---

## 3. Retracted / corrected

Kept deliberately. Each cost time and each would have been caught by a reviewer.

| claim | status | why |
|---|---|---|
| TyDiQA's negative pooled/within gap is because "pooling mixes questions across languages" | **wrong** | We filter to English (`39_generate_dataset.py`, `extract_tydiqa_language`). So does HARP (`DatasetInit.py:65-71`). Both are 440 English questions. |
| "HARP's code randomly splits the data — there is no known/unknown set" | **overstated** | The grouping is correct; only the sub-split within the known group is at answer granularity. |
| Leakage effect scales with the *share of test answers* drawn from seen questions | **refuted by data** | NQ-Open has the smallest share (2.7%) and the largest effect (−13.30). Every AUROC pair needs a correct answer, and correct answers exist only inside known questions — so unknown questions contribute no discriminative pairs at all, and the leakage touches nearly all usable signal everywhere. |
| Effect size tracks the *number of known questions*, inversely | **does not replicate** | Fitted to 4 Qwen points. LLaMA base gives 404→7.77, 506→5.15, 1,267→6.60 — non-monotonic. What modulates the size is unexplained. |
| Qwen TriviaQA `joint_tensor` OOM'd at **200 GB** | **wrong figure** | The run that produced that cell is `slurm/phase3_answersplit_trivia.slurm`, which requested `--mem=128G`. 200G is `harp_qwen_triviaqa.slurm`, a different job; the per-condition question-level jobs ran 48G–192G (`joint_tensor` at 160G). Quoted wrong in the log and in `harp_paper_vs_code.tex`; both fixed 2026-08-24. Noted as caught in a prior session but never actually corrected in the artifacts — the same follow-through failure as the truncated-npz guard. |
| "HalluGuard is training-free — nothing is fitted" | **wrong** | Appendix C.1: *"We train only HALLUGUARD's lightweight projection layers using AdamW."* I searched the repo for checkpoints, found none, and concluded training-free. The correct conclusion was that the **released code omits a trained component**. This was the sole basis for the §2.2b control, which is withdrawn. |
| "HalluGuard's released path cannot run as shipped, so they never ran it at scale" | **wrong framing** | True of `halluguard_true.py`; false of the method. The default proxy needs no gradients and runs on 70B trivially. I generalised from the one file I had chosen to the paper as a whole. Devansh caught it by checking the results section against my claim. |
| "det(K) is inert / completion length beats HalluGuard" | **scope corrected** | Measured on `halluguard_true.py`, which is not the code behind the published numbers. Stands as a statement about that file only. See §2.8, §2.9. |
| "HalluGuard's default scripts call the legacy proxy, so their numbers came from it" | **wrong** | Resolved from the imports, not the prose: `evaluation.py:175`, `gpu_evaluation_all.py:434`, `gpu_evaluation_llm.py:524` and all three `pipeline/generate*.py` call `halluguard_true`. Their README says so plainly. I quoted the verdict line of their `TECHNICAL_SPEC_VERIFICATION.md`, which is stale and contradicts its own table two rows above, and did not check the imports. Devansh caught it from the README. **Lesson: their documents disagree with their code; take the code.** Conclusion unchanged, reason stronger — see §2.11. |
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
