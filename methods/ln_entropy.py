"""methods/ln_entropy.py -- length-normalised predictive entropy (Malinin & Gales, ICLR 2021).

    score(question) = (1/K) * sum_k [ -(1/T_k) * sum_t log p(y_t^k | y_<t^k, x) ]

i.e. the mean, over the K sampled answers, of each answer's length-normalised NLL. High = the model
was uncertain across its own samples = more likely to have hallucinated on this question.

QUESTION granularity, and that is the whole difference from `perplexity`. Perplexity asks "is THIS
answer surprising"; LN-entropy asks "was the model unsure about this QUESTION". They share a forward
pass -- this file reuses `perplexity.nll_per_beam` rather than repeating it, so the two can never
drift apart.

ONE HONEST NOTE ON THE NAME. Malinin & Gales derive this as an approximation to the predictive
entropy, estimated by Monte Carlo over sampled sequences; it is not the exact token-level entropy
over the vocabulary. Every implementation in this literature uses the sampled-NLL form, including
the "LN-Entropy" rows of HARP's and HalluGuard's tables, so that is what we compute. A true
vocabulary-entropy variant would be a different row, not this one.
"""

import numpy as np

from .base import Method
from .perplexity import nll_per_beam


def aggregate_to_question(nll, prompt_id, question_ids):
    """Mean of the per-answer NLLs within each question, NaNs skipped rather than propagated.

    Skipping matters: one empty completion would otherwise turn the whole question's score into
    NaN and drop it from the evaluation entirely, silently shrinking the test set."""
    nll = np.asarray(nll, dtype=float)
    p = np.asarray(prompt_id)
    out = np.full(len(question_ids), np.nan)
    for i, q in enumerate(question_ids):
        v = nll[p == q]
        v = v[np.isfinite(v)]
        if v.size:
            out[i] = v.mean()
    return out


class LNEntropy(Method):
    name = "ln_entropy"
    granularity = "question"
    description = "length-normalised predictive entropy over the K samples (Malinin & Gales 2021)"

    def precompute(self, data):
        print("    [ln_entropy] %d questions, %d answers -- shares perplexity's forward pass"
              % (len(data.question_ids), len(data.labels)), flush=True)
        nll, ntok = nll_per_beam(data)
        per_q = aggregate_to_question(nll, data.prompt_id, data.question_ids)
        self._diag = {
            "n_answers_with_no_generated_tokens": int((ntok == 0).sum()),
            "n_questions_with_no_finite_answer": int(np.isnan(per_q).sum()),
            "mean_answers_per_question": float(len(nll) / max(len(data.question_ids), 1)),
        }
        return {"per_question": per_q}

    def score(self, data, pre, train_idx, test_idx):
        qs = np.unique(data.prompt_id[test_idx])
        pos = np.searchsorted(data.question_ids, qs)
        return pre["per_question"][pos]

    def meta(self):
        return {"orientation": "higher = model was less certain across its samples",
                **getattr(self, "_diag", {})}

    def self_test(self):
        # Aggregation is hand-computable: question 0 averages 1 and 3, question 1 averages 2 and 4.
        nll = np.array([1.0, 3.0, 2.0, 4.0])
        pid = np.array([0, 0, 1, 1])
        got = aggregate_to_question(nll, pid, np.array([0, 1]))
        assert list(got) == [2.0, 3.0], list(got)
        print("    [PASS] aggregate_to_question: mean within question")

        # A NaN answer is skipped, not propagated -- otherwise one empty completion would delete a
        # whole question from the evaluation.
        got = aggregate_to_question(np.array([1.0, np.nan, np.nan, np.nan]), pid, np.array([0, 1]))
        assert got[0] == 1.0 and np.isnan(got[1])
        print("    [PASS] NaN answers skipped; a question with none stays NaN")

        # score() must map test ROWS to the right QUESTION positions. Using a question_ids array
        # that does not start at 0 catches the classic bug of indexing by id instead of position.
        class D:
            prompt_id = np.array([5, 5, 7, 7, 9, 9])
            question_ids = np.array([5, 7, 9])
        pre = {"per_question": np.array([10.0, 20.0, 30.0])}
        out = LNEntropy().score(D(), pre, np.array([0]), np.array([4, 2]))
        assert list(out) == [20.0, 30.0], list(out)
        print("    [PASS] score(): maps rows to question POSITIONS, not raw ids")
