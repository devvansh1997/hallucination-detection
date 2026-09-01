"""methods/_template.py -- copy this file to start a new method.

    cp methods/_template.py methods/my_method.py

Then edit it, add two lines to methods/__init__.py, and run:

    python 56_run_method.py --self-test
    python 56_run_method.py --method my_method --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct

The leading underscore keeps this out of the registry. It is a template, not a method.

FIVE RULES. Breaking any one of them silently corrupts the comparison table rather than raising.

  1. HIGH SCORE = PREDICTED HALLUCINATION. Always. If your metric naturally points the other way
     (agreement, confidence, similarity), negate it INSIDE score() and say so in meta(). Getting
     this backwards inverts your AUROC and the number still looks plausible.

  2. NEVER compute your own AUROC, and never touch the train/test split beyond the indices you are
     handed. The harness owns both. That is the only reason rows of the table can be compared.

  3. Return NaN for rows you genuinely cannot score. Do NOT return 0.0 -- zero is a real score and
     will be ranked as one. The harness counts NaNs and reports them.

  4. Put expensive work in precompute(), not score(). score() runs 10+ times (5 seeds x 2
     protocols); precompute() runs once. A forward pass in score() costs 10x for nothing.

  5. Question-granularity methods return one value per UNIQUE question in test_idx, in SORTED
     question order. Use np.searchsorted(data.question_ids, qs) to map -- indexing by raw question
     id is wrong whenever ids do not start at 0, and TriviaQA's do not.
"""

import numpy as np

from .base import Method


class TemplateMethod(Method):
    name = "template"                 # what --method takes. Must be unique.
    granularity = "beam"              # "beam" (one score per answer) or "question"
    description = "one line, shown by --method-list, cite the paper"

    def __init__(self):
        self.some_setting = 1.0
        self._diag = {}

    # -- optional: method-specific CLI. The harness namespaces these as --m-<name>. --
    def add_args(self, parser):
        parser.add_argument("--m-some-setting", type=float, default=1.0)

    def configure(self, args):
        self.some_setting = args.m_some_setting

    def precompute(self, data):
        """Runs ONCE. Put the forward pass, the feature load, anything slow, here.

        What `data` gives you (see methods/base.py):
            data.input_ids[i]        token ids of answer i (prompt + completion)
            data.prompt_len[i]       where the completion starts
            data.prompt_id           (n_answers,) which question each answer belongs to
            data.labels              (n_answers,) 1 = hallucinated. FOR DIAGNOSTICS ONLY --
                                     using these to build your score is cheating, and the harness
                                     cannot detect it.
            data.decoded_text[i]     the answer as text
            data.question_ids        (n_questions,) sorted unique question ids
            data.question_labels     (n_questions,) 1 = model never got this question right
            data.is_known            (n_questions,) the complement of the above
            data.groups()            [(question_id, row_indices)] -- batch by question
            data.model()             the frozen LLM, lazily loaded and cached
            data.features()          pooled hidden-state features from 42 (1.3-33 GB, lazy)
            data.decoding_config     how this data was generated -- check it if you care
        """
        import torch                                    # noqa: F401  (delete if unused)
        scores = np.full(len(data.labels), np.nan)
        for gi, (q, idx) in enumerate(data.groups()):
            # ... compute something for this question's answers ...
            scores[idx] = 0.0
            if (gi + 1) % 200 == 0:
                print("      %d questions" % (gi + 1), flush=True)
        self._diag = {"anything_worth_recording": 0}
        return {"scores": scores}

    def score(self, data, pre, train_idx, test_idx):
        """Runs per split. Cheap. Return higher = more likely hallucinated.

        train_idx is there for methods that fit something. Training-free methods ignore it."""
        if self.granularity == "beam":
            return pre["scores"][test_idx]
        qs = np.unique(data.prompt_id[test_idx])
        return pre["scores"][np.searchsorted(data.question_ids, qs)]   # see rule 5

    def meta(self):
        """Goes into the results JSON. Record settings, diagnostics, and any caveat a reader of the
        table would need -- e.g. 'implemented from the paper, not verified against their code'."""
        return {"some_setting": self.some_setting,
                "orientation": "higher = ...", **self._diag}

    def self_test(self):
        """REQUIRED. Synthetic only: no cluster, no GPU, no model, runs in seconds.

        Test the things that fail silently, not the things that raise:
          - orientation: assert the confident case scores BELOW the uncertain one
          - alignment:   assert score() returns the requested rows in the requested order
          - NaN:         assert unscoreable rows stay NaN rather than becoming 0.0
          - any closed form your maths has, asserted exactly

        Every method in this repo found a real bug this way. Two of them found an inverted sign."""
        m = TemplateMethod()

        class D:
            prompt_id = np.array([0, 0, 1, 1])
            question_ids = np.array([0, 1])
        pre = {"scores": np.array([0.1, 0.9, 0.2, 0.8])}
        got = m.score(D(), pre, np.array([0]), np.array([3, 1]))
        assert list(got) == [0.8, 0.9], list(got)
        print("    [PASS] score() returns test rows in the requested order")
