"""methods/base.py -- the contract every detector implements, and the data it is handed.

WHY THIS EXISTS
    We are rebuilding the HARP comparison table across models x datasets x methods. If each method
    brings its own splitting or its own AUROC, the table is not a comparison -- it is six different
    experiments in one grid. Two published papers in this area already disagree with their own
    documentation about exactly these details, and we are not going to add a third.

    So the harness owns the data, the split and the metric. A method owns ONE thing: turning the
    pinned generations into a number per row. It cannot reach the split or the metric, because it is
    never given them.

THE CONTRACT
    class MyMethod(Method):
        name        = "my_method"
        granularity = "beam"        # or "question"
        def precompute(self, data): ...            # expensive, split-independent, called ONCE
        def score(self, data, pre, train_idx, test_idx): ...   # cheap, called per split
        def self_test(self): ...                   # required

    precompute/score is split in two deliberately. A training-free scorer (HalluGuard, perplexity,
    semantic entropy) does all its work in precompute and score() just indexes -- otherwise the
    forward pass would be repeated once per seed, five times over, for no reason. A trained probe
    loads features in precompute and does fit/predict in score(). Both fit the same shape.

GRANULARITY
    "beam"     one score per generated answer. Label: 1 = hallucinated. Comparable to HARP and ours.
    "question" one score per question. Label: 1 = the model never got this question right, i.e.
               NOT is_known -- the same definition 43_eval_phase2.py:163 uses. HalluGuard is here.
    A question-level method cannot be compared beam-to-beam with a beam-level one. The runner
    reports the granularity in every output file so this is never silently mixed.
"""

import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def load_numbered(name, filename):
    """Import one of the repo's numbered scripts. They cannot be imported normally because a module
    name may not begin with a digit, so every consumer does this dance; it lives here once."""
    path = os.path.join(REPO, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------------------------
# THE CANONICAL METRIC AND SPLIT, IMPORTED -- NEVER REIMPLEMENTED
#
# These are the definitions the paper's existing numbers were produced with. Importing rather than
# copying is the entire point of the harness: if a method could define its own AUROC, comparability
# would be a matter of everyone being careful, which it demonstrably is not.
# ---------------------------------------------------------------------------------------------

_cache = {}


def canonical():
    """{pooled_auroc, within_prompt_auroc, question_split, answer_split, derive_is_known, seeds}."""
    if "c" in _cache:
        return _cache["c"]
    s53 = load_numbered("s53", "53_halluguard_score.py")   # AUROC, self-tested against sklearn
    s26 = load_numbered("s26", "26_grouped_baseline.py")   # question-level split (paper protocol)
    s44 = load_numbered("s44", "44_eval_phase3.py")        # answer-level split (their released code)
    assert list(s44.HARP_SEEDS) == list(s53.HARP_SEEDS), (
        "seed lists diverged between 44 and 53; every cross-method comparison assumes the SAME "
        "partitions, so this must be fixed rather than tolerated")
    _cache["c"] = {
        "pooled_auroc": s53.pooled_auroc,
        "within_prompt_auroc": s53.within_prompt_auroc,
        "question_split": lambda ik, p, n, seed: s26.original_harp_split(ik, p, n, seed=seed),
        "answer_split": lambda ik, p, n, seed: s44.answer_level_harp_split(ik, p, n, seed),
        "derive_is_known": s53.derive_is_known,
        "seeds": list(s53.HARP_SEEDS),
    }
    return _cache["c"]


# ---------------------------------------------------------------------------------------------

class DataBundle:
    """Everything a method is allowed to see. Constructed once by the runner from the pinned data.

    Deliberately does NOT carry the split or the metric. A method that wants to know which rows are
    in training is handed train_idx by score(); it cannot go looking for anything else."""

    def __init__(self, dataset, model_folder, model_id, data_dir, seq, device, dtype):
        self.dataset = dataset
        self.model_folder = model_folder
        self.model_id = model_id
        self.data_dir = data_dir
        self.device = device
        self.dtype = dtype

        self.input_ids = seq["input_ids"]
        self.prompt_len = seq["prompt_len"]
        self.prompt_id = np.asarray(seq["prompt_id"])
        self.labels = np.asarray(seq["all_hallucination_flag"], dtype=int)   # 1 = hallucinated
        self.decoded_text = seq.get("decoded_text")
        self.decoding_config = seq.get("decoding_config", {}) or {}

        c = canonical()
        self.is_known = c["derive_is_known"](self.labels, self.prompt_id)
        self.question_ids = np.unique(self.prompt_id)
        # A question is hallucinated iff NO answer was right -- the complement of is_known, so both
        # granularities rest on one definition rather than two that could drift apart.
        self.question_labels = (~self.is_known).astype(int)

        self._model = None
        self._features = None

    # -- lazily loaded, because most methods need neither and both are expensive --

    def model(self):
        """The frozen LLM. ~30s-7min to load; cached so precompute() pays it at most once."""
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM
            td = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                  "float32": torch.float32}[self.dtype]
            m = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=td,
                                                     trust_remote_code=True).to(self.device)
            m.eval()
            self._model = m
        return self._model

    def features(self):
        """The pooled hidden-state features from 42_extract_phase2.py. 1.3-33 GB per dataset, so
        this is loaded only if a method actually asks for it."""
        if self._features is None:
            p = os.path.join(self.data_dir, self.model_folder,
                             "%s_phase2_features.npz" % self.dataset)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    "%s not found -- run 42_extract_phase2.py for this model/dataset first." % p)
            self._features = np.load(p)
        return self._features

    def groups(self):
        """[(question_id, row_indices)] in stable question order. Most methods that need a forward
        pass want to batch by question, since the ten answers share a prompt."""
        order = np.argsort(self.prompt_id, kind="stable")
        return [(int(q), order[self.prompt_id[order] == q]) for q in self.question_ids]

    def __repr__(self):
        return ("DataBundle(%s/%s, %d answers, %d questions, %.1f%% hallucinated)"
                % (self.model_folder, self.dataset, len(self.labels), len(self.question_ids),
                   100.0 * self.labels.mean()))


class Method:
    """Subclass this. See methods/halluguard.py for a worked example."""

    name = None
    granularity = "beam"        # "beam" or "question"
    description = ""

    # Optional: declare method-specific CLI flags. The runner namespaces them under --m-<flag>.
    def add_args(self, parser):
        pass

    def configure(self, args):
        pass

    def precompute(self, data):
        """Expensive, split-independent work. Called ONCE per (model, dataset). Return anything."""
        return None

    def score(self, data, pre, train_idx, test_idx):
        """Return scores for test_idx (beam granularity) or for the test QUESTIONS (question
        granularity), higher = more likely hallucinated. Training-free methods ignore train_idx.

        For question granularity, test_idx is still row indices; the runner maps to questions and
        expects one score per UNIQUE question present in test_idx, in sorted question order."""
        raise NotImplementedError

    def meta(self):
        """Anything worth recording in the results file -- settings, diagnostics, warnings."""
        return {}

    def self_test(self):
        """Required. Synthetic, no cluster, no GPU, no model."""
        raise NotImplementedError("every method must ship a self-test")
