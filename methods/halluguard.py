"""methods/halluguard.py -- HalluGuard (ICLR 2026), as its authors specify it in rebuttal.

Reference implementation of the Method contract. Read this before writing a new one.

THE SCORE. From the OpenReview rebuttal: no gradients, no back-propagation. Collect the final-layer
hidden states of the K sampled trajectories into H, form K = H H^T / d, then

    score = log det K + log sigma_max - 2 log kappa
          = sum_i log L_i - log L_max + 2 log L_min

The maths lives in 55_halluguard_rebuttal.py and is imported, not copied -- 55 remains runnable
standalone and its self-test is the authority on the arithmetic. This file is the adapter.

WHY granularity = "question". The score is a property of the SET of ten trajectories, not of any
one answer, so there is one number per question. The runner labels a question hallucinated iff no
trajectory was correct. This is not comparable beam-to-beam with HARP or with our method, which is
exactly why the harness records granularity in every output file.

WHAT WE CANNOT DO. The paper trains "lightweight projection layers" (Appendix C.1) that are absent
from the release -- architecture, dimensions and objective all unstated. We run identity projection
and say so. The authors describe them as calibration for comparability ACROSS backbones, so running
one model at a time makes this a smaller assumption than it sounds; but the score is not
scale-invariant, so it is not a free one either.

THE CONFOUND. This method measures how spread out the trajectories are. Beam search returns ten
DISTINCT sequences by construction, so K is full rank whatever the answers look like; nucleus
sampling at low temperature returns duplicates and the rank collapses. On TyDiQA the numerical rank
was 10.00 under beam search and 3.98 under nucleus. mean_numerical_rank is reported for exactly
this reason -- read it before the AUROC.
"""

import numpy as np

from .base import Method, load_numbered

_s55 = None


def s55():
    global _s55
    if _s55 is None:
        _s55 = load_numbered("s55", "55_halluguard_rebuttal.py")
    return _s55


class HalluGuard(Method):
    name = "halluguard"
    granularity = "question"
    description = "NTK Gram over the K sampled trajectories (OpenReview rebuttal spec)"

    def __init__(self):
        self.layer = "final"
        self.pool = "mean"
        self.ridge = 0.0
        self._diag = {}

    def add_args(self, parser):
        parser.add_argument("--m-layer", default="final", choices=["final", "middle"],
                            help="rebuttal says final; Appendix C.1 says middle (L/2). They differ.")
        parser.add_argument("--m-pool", default="mean", choices=["mean", "last"],
                            help="rebuttal says 'hidden states'; C.1 says 'final token'.")
        parser.add_argument("--m-ridge", type=float, default=0.0,
                            help="0 = rebuttal spec (K is full rank uncentered). 1e-3 = C.1.")

    def configure(self, args):
        self.layer, self.pool, self.ridge = args.m_layer, args.m_pool, args.m_ridge

    def precompute(self, data):
        """One no-grad forward pass per question. Split-independent, so this runs once even though
        the runner evaluates five seeds x two protocols."""
        import torch
        m = data.model()
        n_layers = m.config.num_hidden_layers
        sel = -1 if self.layer == "final" else (n_layers + 1) // 2
        groups = data.groups()
        print("    [halluguard] %d questions | layer=%s (hidden_states[%s]) pool=%s ridge=%g"
              % (len(groups), self.layer, sel, self.pool, self.ridge), flush=True)
        if int(data.decoding_config.get("num_beams", 0) or 0) > 1:
            print("    [halluguard] WARNING: these generations came from beam search "
                  "(num_beams=%s). This score measures trajectory spread, and beam search returns "
                  "distinct sequences by construction -- read mean_numerical_rank before the AUROC."
                  % data.decoding_config.get("num_beams"), flush=True)

        out, ranks = {}, []
        for gi, (q, idx) in enumerate(groups):
            batch = [data.input_ids[i] for i in idx]
            pls = [int(data.prompt_len[i]) for i in idx]
            L = max(len(b) for b in batch)
            pad = m.config.eos_token_id or 0
            ids = torch.full((len(batch), L), pad, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, b in enumerate(batch):
                t = b if torch.is_tensor(b) else torch.as_tensor(b)
                ids[j, :len(t)] = t
                att[j, :len(t)] = 1
            with torch.no_grad():
                o = m(ids.to(data.device), attention_mask=att.to(data.device),
                      output_hidden_states=True)
            Hs = o.hidden_states[sel].float()
            rows = []
            for j in range(len(batch)):
                v = s55().pool_sequence(
                    Hs[j, pls[j]:len(batch[j]), :].cpu().numpy(), self.pool)
                if v is not None:
                    rows.append(v)
            del o, Hs
            r = s55().rebuttal_score(np.stack(rows), ridge=self.ridge) if len(rows) >= 2 else None
            out[q] = r["score"] if r else np.nan
            if r:
                ranks.append(r["numerical_rank"])
            if (gi + 1) % 200 == 0:
                print("      %d/%d questions" % (gi + 1, len(groups)), flush=True)

        self._diag = {
            "mean_numerical_rank": float(np.mean(ranks)) if ranks else None,
            "n_rank_deficient": int(sum(1 for r in ranks if r < 10)),
            "n_questions_scored": len(ranks),
        }
        return out

    def score(self, data, pre, train_idx, test_idx):
        """Training-free: train_idx is ignored. One score per unique question in test_idx, in
        sorted question order -- the order the runner expects."""
        qs = np.unique(data.prompt_id[test_idx])
        return np.array([pre.get(int(q), np.nan) for q in qs], dtype=float)

    def meta(self):
        return {"layer": self.layer, "pool": self.pool, "ridge": self.ridge,
                "projection": "identity (the trained calibration layer is unreleased)",
                **self._diag}

    def self_test(self):
        m = HalluGuard()

        # Orthonormal trajectories -> K = I -> every term vanishes.
        rng = np.random.default_rng(0)
        Q = np.linalg.qr(rng.normal(size=(8, 8)))[0]
        assert abs(s55().rebuttal_score(Q, normalise_by_d=False)["score"]) < 1e-9
        print("    [PASS] score of an orthonormal cloud is 0")

        # Ten identical trajectories: rank 1, score collapses. This is the failure mode the
        # decoding confound produces, so it is asserted rather than assumed.
        dup = np.repeat(rng.normal(size=(1, 32)), 10, axis=0)
        rd = s55().rebuttal_score(dup, normalise_by_d=False)
        assert rd["numerical_rank"] == 1 and rd["score"] < -100
        print("    [PASS] duplicate cloud: rank 1, score %.0f" % rd["score"])

        # score() must return one value per unique test question, in sorted order, and must not
        # look at train_idx -- a training-free method that quietly used it would leak.
        class D:
            prompt_id = np.array([0, 0, 1, 1, 2, 2])
        pre = {0: -1.0, 1: -2.0, 2: -3.0}
        got = m.score(D(), pre, train_idx=np.array([0, 1]), test_idx=np.array([5, 2, 4]))
        assert list(got) == [-2.0, -3.0], list(got)
        same = m.score(D(), pre, train_idx=np.array([]), test_idx=np.array([5, 2, 4]))
        assert list(got) == list(same), "score() must ignore train_idx"
        print("    [PASS] score(): one value per unique test question, sorted, ignores train_idx")
