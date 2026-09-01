"""methods/lexical_similarity.py -- mean pairwise ROUGE-L among the K sampled answers.

    score(question) = - (2 / K(K-1)) * sum_{i<j} ROUGE-L(y_i, y_j)

If the model's ten answers agree with each other it is probably confident and probably right; if
they disagree it is probably guessing. Negated so that HIGH = predicted hallucination, matching the
orientation of every other method in the harness.

QUESTION granularity. The "Lexical Similarity" row of HalluGuard's baseline table (Lin et al. 2022b).

WHY THIS ROW MATTERS MORE THAN ITS AGE SUGGESTS. It needs no model, no GPU and no forward pass --
it reads `decoded_text` and nothing else, so it runs in seconds on a laptop. It is therefore the
cheapest possible check on any expensive method: if a detector that costs a GPU-hour cannot beat
string overlap, that is the finding. We have already seen HalluGuard reduce to a diversity count on
TyDiQA, and this is the row that makes such a claim concrete rather than rhetorical.

ROUGE-L IS IMPLEMENTED HERE, not imported. `evaluate.load("rouge")` pulls a 4 GB metrics stack and
is far too slow for K(K-1)/2 = 45 pairs x 14,827 questions. The longest-common-subsequence F-measure
is fifteen lines and is exercised against hand-computed values in the self-test.
"""

import numpy as np

from .base import Method


def lcs_length(a, b):
    """Length of the longest common subsequence. O(len(a) * len(b)) with a rolling row."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(a_tokens, b_tokens):
    """ROUGE-L F-measure. Symmetric, in [0, 1]; 1.0 iff the token sequences are identical."""
    if not a_tokens or not b_tokens:
        return 1.0 if (not a_tokens and not b_tokens) else 0.0
    l = lcs_length(a_tokens, b_tokens)
    if l == 0:
        return 0.0
    p, r = l / len(b_tokens), l / len(a_tokens)
    return 2 * p * r / (p + r)


def mean_pairwise_rouge(texts):
    """Mean ROUGE-L over all unordered pairs. Returns NaN when there are fewer than two answers."""
    toks = [t.strip().lower().split() for t in texts]
    n = len(toks)
    if n < 2:
        return np.nan
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += rouge_l(toks[i], toks[j])
            cnt += 1
    return tot / cnt


class LexicalSimilarity(Method):
    name = "lexical_similarity"
    granularity = "question"
    description = "negated mean pairwise ROUGE-L among the K answers (Lin et al. 2022b) -- no model"

    def precompute(self, data):
        if data.decoded_text is None:
            raise SystemExit("this dataset has no decoded_text; regenerate with 39, which saves it")
        print("    [lexical_similarity] %d questions, no model needed"
              % len(data.question_ids), flush=True)
        sims = np.full(len(data.question_ids), np.nan)
        distinct = np.zeros(len(data.question_ids))
        for i, (_, idx) in enumerate(data.groups()):
            texts = [data.decoded_text[k] for k in idx]
            sims[i] = mean_pairwise_rouge(texts)
            distinct[i] = len(set(t.strip() for t in texts))
        self._diag = {
            "mean_pairwise_rouge": float(np.nanmean(sims)),
            "mean_distinct_answers_per_question": float(distinct.mean()),
            "n_questions_all_answers_identical": int((distinct == 1).sum()),
        }
        return {"sim": sims}

    def score(self, data, pre, train_idx, test_idx):
        qs = np.unique(data.prompt_id[test_idx])
        pos = np.searchsorted(data.question_ids, qs)
        return -pre["sim"][pos]      # negate: high = disagreement = predicted hallucination

    def meta(self):
        return {"orientation": "score = -similarity, so higher = answers disagree = hallucination",
                **getattr(self, "_diag", {})}

    def self_test(self):
        assert lcs_length(["a", "b", "c", "d", "e"], ["a", "b", "c", "d", "e"]) == 5
        assert lcs_length(list("abc"), list("axbyc")) == 3
        assert lcs_length([], list("abc")) == 0
        print("    [PASS] lcs_length on hand-computed cases")

        assert rouge_l(["a", "b"], ["a", "b"]) == 1.0
        assert rouge_l(["a", "b"], ["c", "d"]) == 0.0
        assert abs(rouge_l(["a", "b"], ["a"]) - 2 * (1 / 1) * (1 / 2) / (1 / 1 + 1 / 2)) < 1e-12
        # Symmetry is not automatic for an F-measure if precision and recall get swapped.
        assert abs(rouge_l(list("abcd"), list("abx")) - rouge_l(list("abx"), list("abcd"))) < 1e-12
        print("    [PASS] rouge_l: identical=1, disjoint=0, symmetric")

        assert mean_pairwise_rouge(["same text", "same text", "same text"]) == 1.0
        assert mean_pairwise_rouge(["alpha beta", "gamma delta"]) == 0.0
        assert np.isnan(mean_pairwise_rouge(["only one"]))
        print("    [PASS] mean_pairwise_rouge: identical=1, disjoint=0, single answer=NaN")

        # Orientation. Agreement must produce a LOWER score than disagreement, since high means
        # hallucination. Getting this backwards is the single easiest mistake in this file.
        class D:
            prompt_id = np.array([0, 0, 1, 1])
            question_ids = np.array([0, 1])
        pre = {"sim": np.array([1.0, 0.0])}          # q0 answers agree, q1 disagree
        out = LexicalSimilarity().score(D(), pre, np.array([]), np.array([0, 2]))
        assert out[0] < out[1], "agreeing answers must score lower than disagreeing ones"
        assert list(out) == [-1.0, 0.0]
        print("    [PASS] orientation: agreement scores %.1f, disagreement %.1f" % (out[0], out[1]))
