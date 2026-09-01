# Repo orientation

For someone joining the codebase. Read `AGENTS.md` first (house rules), then this, then
`docs/PROJECT_LOG.md` (what we have found and what we got wrong).

---

## 1. The one-paragraph version

We detect hallucinations from a model's **internal states**, not its text. For each question we
generate 10 answers, label each right/wrong with an automatic judge, capture hidden states while
re-running those answers, pool them into per-answer feature vectors, and train a small classifier to
predict the label. The headline metric is **AUROC**. Most of the subtlety is in *how the data is
split* and *which AUROC you report* — see §5 and §6, which is where every mistake on this project
has happened.

---

## 2. Layout

```
hallucination-detection/          <- the repo (this is what you clone)
  NN_*.py                         <- numbered pipeline scripts, run in order
  config.yaml                     <- models, datasets, decoding, judge thresholds. NOT hardcoded.
  AGENTS.md                       <- house rules. Read first.
  docs/PROJECT_LOG.md             <- findings register, including retractions
  slurm/                          <- one .slurm per stage
  submit_*.sh                     <- SLURM drivers that queue whole pipelines
  results/                        <- JSON outputs, committed
  archive/                        <- superseded code. Do not use.
../data/{model_folder}/           <- generations + features (NOT in git, too large)
../data-nucleus/{model_folder}/   <- alternate-decoding generations, kept separate on purpose
../HARP-Code/                     <- their code, read-only, unmodified
../HalluGuard-ICLR2026/           <- their code, read-only, unmodified
```

**Numbers are chronological, not a dependency order.** `26_` is older than `55_`; it is not a
prerequisite of it. The live pipeline is roughly `39 -> 40 -> 42 -> 44`.

---

## 3. The files that matter

### Core pipeline — this is the path a new model takes

| file | what it does | cost |
|---|---|---|
| `39_generate_dataset.py` | Generate 10 answers per question **and label them** (judge runs inline). Writes `{dataset}_sequences_v1.pt`. | 8h for TriviaQA |
| `40_validate_dataset.py` | Integrity checks on what 39 produced. **Needs `--manifest`** — forgetting it has silently failed three gen jobs before. | seconds |
| `42_extract_phase2.py` | Re-run the pinned answers, capture hidden states, pool into feature streams. | 12h / 256 GB for TriviaQA |
| `44_eval_phase3.py` | **The main evaluator.** Trains RF and LR on the pooled features and reports AUROC under both split protocols. | up to 7h per condition |

### Baselines and audits

| file | what it does |
|---|---|
| `49_harp_adapter.py` | Converts our generations into HARP's input format, so their unmodified code runs on our data |
| `51_judge_agreement.py` | Re-labels everything with HARP's own judge and measures disagreement (audit #1, closed: 0.30% flip) |
| `53_halluguard_score.py` | HalluGuard via their gradient implementation. **Superseded** — see PROJECT_LOG §2.11 |
| `54_halluguard_proxy.py` | HalluGuard via their legacy helper. **Superseded** — see §2.9 |
| `55_halluguard_rebuttal.py` | HalluGuard as the authors describe it in their OpenReview rebuttal. **Current.** |
| `audit_split_protocols.py` | Verifies our split implementations against HARP's own `utils.split_data` |

53 and 54 are kept, not deleted, because the *reason* they were superseded is itself a finding.

### Figures

`45_analysis_loader.py` is the shared loader; `46_`, `47_`, `48_` produce figures 1-3.

---

## 3b. Adding a new detection method — start here

**Do not write a standalone script.** Use the harness:

```bash
python 56_run_method.py --list
python 56_run_method.py --method halluguard --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct
```

`--method` is the only thing that changes between rows of the comparison table. The runner owns the
data loading, both split protocols, the AUROC and the output schema; a method owns exactly one
thing — turning the pinned generations into a number per row. It is never handed the split or the
metric, so it cannot accidentally define its own.

To add one:

1. `methods/<name>.py` with a `Method` subclass — see `methods/halluguard.py`, which is the
   reference implementation and is commented as such.
2. One line in `methods/__init__.py`.

The interface:

```python
class MyMethod(Method):
    name        = "my_method"
    granularity = "beam"          # or "question"
    def precompute(self, data): ...                      # expensive, split-free, runs ONCE
    def score(self, data, pre, train_idx, test_idx): ... # cheap, runs per split
    def self_test(self): ...                             # required
```

`precompute`/`score` are separate on purpose. A training-free scorer does all its work in
`precompute` and `score` just indexes — otherwise the forward pass repeats once per seed, five times
over. A trained probe loads features in `precompute` and does fit/predict in `score`. Both fit.

**`data.model()` and `data.features()` are lazy.** Most methods need neither; the features file is
1.3–33 GB per dataset, so it is only read if you ask.

**Granularity is not cosmetic.** `beam` scores answers (label: 1 = hallucinated). `question` scores
questions (label: 1 = the model never got it right). They are not comparable to each other and must
not share a column. The runner records which in every output file.

**The harness self-test checks the plumbing, not just the methods.** It runs a perfect oracle and an
inverted oracle through the full path and asserts 1.000 and 0.000 under both protocols, and asserts
that the two protocols really do differ in how many questions straddle the split. Run it after any
change:

```bash
python 56_run_method.py --self-test
```

## 4. Where AUROC actually lives

Three implementations exist. They are not redundant — know which is which.

**Pooled AUROC** — every pair of (hallucinated, truthful) answers, across all questions.

- `sklearn.metrics.roc_auc_score`, used directly in `43_eval_phase2.py:204` and
  `44_eval_phase3.py:199`. This is the number in the main results tables.
- `53_halluguard_score.py:95` `pooled_auroc()` — written out by hand so its tie convention matches
  the within-prompt function exactly. Its self-test asserts agreement with sklearn to 1e-9.

**Within-prompt AUROC** — the same score vector, but only comparing answers **to the same question**.

- `26_grouped_baseline.py:211` `within_prompt_auroc()` is the canonical definition.
- `53_halluguard_score.py:107` reimplements it so 53 stays importable without the `26 -> 27 -> ...`
  chain, and its self-test asserts the two agree to 1e-12.
- `55_halluguard_rebuttal.py` imports 53's helpers rather than writing a third copy.

**Why both.** Pooled AUROC includes cross-question pairs, so a score that merely knows *which
questions are hard* scores well without judging any answer. Within-prompt strips that out. On
TriviaQA the same features give **0.84 pooled and 0.73 within-prompt** — that gap is question
difficulty, not detection. If you report one number, report both.

> If you add a fourth AUROC implementation, cross-check it against `26_grouped_baseline` in a
> self-test. Two of the three already do.

---

## 5. The split protocols — the single most important thing here

Each question has 10 answers. When you split into train/test, **what is the unit?**

| | `original_harp_split` (`26_grouped_baseline.py:130`) | `answer_level_harp_split` (`44_eval_phase3.py:225`) |
|---|---|---|
| unit | **question** | **answer** |
| a known question... | lands entirely on one side | has ~94% chance of appearing on **both** |
| what it is | what HARP's paper describes | what HARP's released code does |

`harp_split()` (`44_eval_phase3.py:249`) dispatches on the `SPLIT_UNIT` global, set by
`--split-unit {question,answer}`.

Both arms put **all** unknown questions in test, and both give the same train/test row *counts* —
61,650 / 37,950 on TriviaQA. So **you cannot tell which protocol produced a result file by looking
at its sizes.** The only difference is *which* known rows land where. This has caught us out; the
reliable tell is that the `grouped` block is identical between arms while every `harp` entry differs.

Changing only the split unit is worth **3.8-13.3 AUROC points** to HARP and 1.3-11.8 to us. That is
the paper's headline result. See PROJECT_LOG §2.1-2.2.

`derive_is_known()` (`43_eval_phase2.py:163`) defines the known/unknown grouping: a question is
*known* if **any** of its 10 answers is judged correct.

---

## 6. Conventions that are not obvious

**Labels.** `1 = hallucinated`, `0 = truthful`. A score with AUROC below 0.5 is not broken, it is
inverted — report both orientations rather than silently flipping.

**Every script has `--self-test`.** All 31 of them. It runs on synthetic data, needs no cluster, no
GPU and no model, and it is the first thing to run after any change:

```bash
python 44_eval_phase3.py --self-test
```

These are not decoration. The self-tests on this project have caught a planted-signal bug, a
scale-relative epsilon that collapsed to zero, an arithmetic error in an expected value, and a wrong
degeneracy metric. **Write a failing case before trusting a number.**

**`config.yaml` is the source of truth** for models, datasets, decoding and judge thresholds. Do not
hardcode. The `model_folder` string (`qwen-2.5-7b-instruct`, `llama-3.1-8b`) is the key that ties
config, data directories and result filenames together.

**Layer window is `{15..23}` plus the final normed layer, hardcoded** at
`34_gate_reconstruct_or_regenerate.py:73`. Layers outside it are computed during the forward pass and
**thrown away** — they are not on disk. Any experiment about layer choice needs a re-extraction.

**Non-ASCII characters are banned in Python files.** See AGENTS.md.

---

## 7. Running things

Never `python script.py` on a cluster login node for real work — submit via SLURM.

```bash
bash submit_pipeline.sh <model_folder>     # full pipeline for one model
bash submit_trivia_eval.sh <model_folder>  # TriviaQA evaluation fan-out (run from STOKES)
bash submit_hgreb.sh                       # HalluGuard, current version
```

**Newton for GPU jobs, Stokes for high-RAM CPU jobs.** They share a filesystem but have separate
schedulers — a job lands on whichever cluster you submitted from. Evaluation is CPU-and-memory-heavy
and belongs on Stokes; anything touching a model belongs on Newton.

Every stage carries a **skip guard** that verifies existing output before redoing work, so
resubmitting after a failure redoes only what is missing. The guards check *integrity*, not just
existence — an OOM once left an 8.6 GB stub of a 35 GB file and a guard that only tested existence
reported success.

**Contention is real:** roughly 4x slowdown at 7 concurrent jobs. Do not size time budgets from an
uncontended run.

---

## 8. Things that have bitten us

Read PROJECT_LOG §3 for the full list with reasons. The recurring shapes:

- **A guard that checks existence rather than integrity** lets a truncated file through silently.
- **`sub()` called as `J=$(sub ...)`** runs in a subshell, so `exit` on a rejected `sbatch` kills only
  the subshell and the driver reports success for jobs it never queued.
- **Trusting a document over the code.** Both HARP's paper and HalluGuard's spec document disagree
  with their own repositories. Always read the imports.
- **Concurrent jobs racing a shared cache.** Two generation jobs starting one second apart both
  re-downloaded a 4.2 GB judge model and both timed out.
- **Quoting a number from memory.** Every figure in a document should be traceable to a file and
  line. We have shipped a wrong memory figure into a report twice.

---

## 9. First day

```bash
git pull
conda activate hal-det
python 44_eval_phase3.py --self-test      # ~1 min, no GPU. Should end "[PASS] All ... passed."
python 55_halluguard_rebuttal.py --self-test
cat docs/PROJECT_LOG.md                    # sections 1 and 2 are the state of play
python -c "import json;d=json.load(open('results/qwen-2.5-7b-instruct/session06_phase3_partA_tydiqa_gp.json'));print(json.dumps(d['harp']['q_static'],indent=2))"
```

That last one shows the shape of a result file: per-seed RF and LR numbers under the HARP protocol,
with `n_train` / `n_valid`, which is what every table in the paper is built from.
