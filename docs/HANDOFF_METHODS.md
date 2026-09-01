# Handoff: adding detection methods

**Your task.** Rebuild the comparison table — several detection methods × our models × our datasets —
using the generations we have already produced. You do not need to generate data, label anything,
or write an evaluator. All three exist and are shared.

**The one thing that matters.** Every method must be scored on identical data, with identical
splits, by identical metric code. If each method brings its own evaluation the table is not a
comparison. The harness enforces this: it hands your method the data and takes back a number per
row, and never lets you see the split or the metric.

Read `docs/ONBOARDING.md` for how the repo is laid out. This document is only about adding methods.

---

## 1. Setup (once)

```bash
git clone <repo>                       # your own clone
cd hallucination-detection
conda activate hal-det                 # existing env on the cluster; nothing to install

export HD_REPO=$PWD
export HD_DATA=/home/de807845/Hallucination-Detection/data
```

Put those two exports in your `~/.bashrc`. **`HD_DATA` points at Devansh's directory and is read
only** — the harness never writes there. That is how you use the pinned generations without copying
91 GB. If you get permission denied, ask him to run:

```bash
chmod o+X /home/de807845 /home/de807845/Hallucination-Detection
chmod -R o+rX /home/de807845/Hallucination-Detection/data
```

Verify everything works before writing anything — this needs no GPU and takes about a minute:

```bash
python 56_run_method.py --list        # should print 4 methods
python 56_run_method.py --self-test   # should end "[PASS] All self-test assertions passed."
```

Then a real run on the smallest dataset, which needs no model at all and finishes in seconds:

```bash
python 56_run_method.py --method lexical_similarity \
    --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct --data-dir $HD_DATA
```

If that prints AUROCs, your data access and environment are both correct.

---

## 2. What already exists

| method | granularity | needs | source |
|---|---|---|---|
| `perplexity` | beam | one forward pass | Ren et al. 2023 |
| `ln_entropy` | question | the same forward pass | Malinin & Gales 2021 |
| `lexical_similarity` | question | **nothing — no model** | Lin et al. 2022b |
| `eigenscore` | question | one forward pass | Chen et al. 2024 (INSIDE) |

Also in the repo but **not** in the table: `methods/halluguard.py`. It is deregistered on purpose —
see `docs/PROJECT_LOG.md` §2.9–2.12. Do not re-enable it without asking.

Not yet written, and the two that matter most for the paper: **our method** (currently evaluated by
`44_eval_phase3.py`) and **HARP** (currently run through `49_harp_adapter.py`). Those are ours to
port; check before starting on them.

---

## 3. Adding a method

```bash
cp methods/_template.py methods/my_method.py
```

Edit it, then add two lines to `methods/__init__.py`:

```python
from .my_method import MyMethod
register(MyMethod)
```

That is the whole integration. Then:

```bash
python 56_run_method.py --self-test                      # your self-test runs automatically
python 56_run_method.py --method my_method --dataset tydiqa_gp \
    --model_folder qwen-2.5-7b-instruct --data-dir $HD_DATA
```

### The interface

```python
class MyMethod(Method):
    name        = "my_method"
    granularity = "beam"                                  # or "question"
    def precompute(self, data): ...                       # slow, runs ONCE
    def score(self, data, pre, train_idx, test_idx): ...  # fast, runs 10+ times
    def self_test(self): ...                              # required
```

`precompute` and `score` are separate because `score` runs once per seed per protocol — ten times or
more. A forward pass in `score` costs ten times what it should. Training-free methods do everything
in `precompute` and index in `score`; trained probes load features in `precompute` and fit in
`score`.

What `data` gives you is listed in `methods/_template.py` and defined in `methods/base.py`.
`data.model()` and `data.features()` are lazy — nothing is loaded unless you call them.

---

## 4. Five rules

Breaking any of these corrupts the table **silently** rather than raising. Each has already caught
somebody on this project.

**1. High score = predicted hallucination. Always.** If your metric naturally points the other way —
agreement, confidence, similarity — negate it inside `score()` and say so in `meta()`. An inverted
sign produces a plausible-looking AUROC that is exactly wrong. This happened while writing
`eigenscore`, and only the self-test caught it.

**2. Never compute your own AUROC, and never touch the split** beyond the indices you are handed.

**3. Return `NaN` for rows you cannot score, never `0.0`.** Zero is a real score and gets ranked as
one. The harness counts NaNs and reports them.

**4. Expensive work goes in `precompute()`.**

**5. Question-granularity methods return one value per unique question in `test_idx`, in sorted
question order.** Use `np.searchsorted(data.question_ids, qs)`. Indexing by raw question id is wrong
whenever ids do not start at zero — and TriviaQA's do not.

---

## 5. Writing the self-test

It is required, and it is not a formality — **every method in this repo found a real bug this way,
and two of them found an inverted sign.** It must run in seconds with no GPU, no model and no
cluster.

Test the things that fail silently:

- **orientation** — assert the confident case scores *below* the uncertain one
- **alignment** — assert `score()` returns the requested rows in the requested order, using question
  ids that do not start at zero
- **NaN** — assert unscoreable rows stay NaN
- **closed form** — if your maths has one, assert it exactly

Have your self-test *print* the direction it measured. That is literally how the `eigenscore` sign
bug was caught: the printed sentence contradicted itself.

---

## 6. Running the grid

```bash
bash submit_methods.sh                                    # everything
METHODS="my_method" bash submit_methods.sh                # just yours
DATASETS=tydiqa_gp MODELS=qwen-2.5-7b-instruct bash submit_methods.sh   # one cell, to smoke-test
```

Newton, not Stokes — these are GPU jobs. Jobs are independent, and each has a skip guard, so a
resubmit after a failure redoes only what is missing.

Results land in `$HD_REPO/results/methods/<method>_<model>_<dataset>.json`.

---

## 7. Reading the output

Each file has three AUROC blocks. **They are different quantities and must not be mixed in one
column.**

| block | what it is |
|---|---|
| `all_rows` | every row, no split. Meaningful only for training-free methods |
| `protocols.question` | question-level split — what HARP's **paper** describes |
| `protocols.answer` | answer-level split — what HARP's released **code** does |

The gap between the last two is 3.8–13.3 AUROC points, and it is the paper's central finding. A
number reported without saying which protocol produced it is not a result.

Also check before trusting anything:

- `method_meta` — your own diagnostics, and any caveat you recorded
- `n_non_finite` — how many rows you failed to score
- `granularity` — beam and question methods are **not** comparable to each other
- `source_decoding_config` — confirms which generations were scored

---

## 8. Things that will bite you

- **`AGENTS.md` has a stale cluster path** (`/home/devansh/...`) and some slurm files hardcode
  `/home/de807845/...`. Use `$HD_REPO` and `$HD_DATA`; do not copy paths out of older scripts.
- **No non-ASCII characters in Python files.** House rule, see `AGENTS.md`.
- **Do not write to `$HD_DATA`.** It is somebody else's directory and the pinned generations behind
  every existing result.
- **Concurrent jobs can race a shared model cache.** Two jobs loading the same large checkpoint at
  once have re-downloaded it and timed out. If you see that, serialise with
  `--dependency=singleton`.
- **`data.labels` is for diagnostics only.** Using it to build a score is cheating and the harness
  cannot detect it.

---

## 9. When you are stuck

Send the command you ran, the last 30 lines of `slurm_logs/<jobname>_<jobid>/`, and the output of
`python 56_run_method.py --self-test`. Those three answer most questions immediately.
