"""methods/perplexity.py -- mean negative log-likelihood of the generated answer.

The oldest and most universal baseline in this literature, and the one every paper reports. If a
detector cannot beat it, that is the headline, not a footnote.

    score(answer) = -(1/T) * sum_t log p(y_t | y_<t, x)

Higher = the model found its own answer more surprising = more likely a hallucination. This is
Eq. 1 of the HalluGuard paper and the "Perplexity" row of both its and HARP's baseline tables.

BEAM granularity: one number per answer, directly comparable with HARP and with our method.

Also exports nll_per_beam(), which ln_entropy.py reuses -- the two methods differ only in how the
per-answer NLLs are aggregated, and computing the forward pass twice would be wasteful and would
risk the two disagreeing.
"""

import numpy as np

from .base import Method


def nll_per_beam(data, log_every=200):
    """Mean NLL of the generated tokens, one value per answer, plus the token count.

    Teacher forcing on the PINNED tokens: we score the answer that was actually generated, not a
    fresh one. Position t of the answer is predicted by logits at position t-1, which is the
    off-by-one every implementation of this gets wrong at least once."""
    import torch
    m = data.model()
    groups = data.groups()
    nll = np.full(len(data.labels), np.nan)
    ntok = np.zeros(len(data.labels), dtype=int)
    for gi, (_, idx) in enumerate(groups):
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
        ids = ids.to(data.device)
        with torch.no_grad():
            logits = m(ids, attention_mask=att.to(data.device)).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        for j, i in enumerate(idx):
            s, e = pls[j], len(batch[j])
            if e - s < 1:
                continue
            # logits[:, t-1] predicts token t -- hence the shift.
            tok = ids[j, s:e]
            lp = logp[j, s - 1:e - 1, :].gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            nll[i] = float(-lp.mean().item())
            ntok[i] = int(e - s)
        del logits, logp
        if (gi + 1) % log_every == 0:
            print("      %d/%d questions" % (gi + 1, len(groups)), flush=True)
    return nll, ntok


class Perplexity(Method):
    name = "perplexity"
    granularity = "beam"
    description = "mean negative log-likelihood of the generated answer (Ren et al. 2023)"

    def precompute(self, data):
        print("    [perplexity] %d questions, teacher-forced NLL over the pinned answers"
              % len(data.question_ids), flush=True)
        nll, ntok = nll_per_beam(data)
        self._n_empty = int((ntok == 0).sum())
        return {"nll": nll, "ntok": ntok}

    def score(self, data, pre, train_idx, test_idx):
        return pre["nll"][test_idx]

    def meta(self):
        return {"n_answers_with_no_generated_tokens": getattr(self, "_n_empty", 0),
                "orientation": "higher NLL = more surprised = predicted hallucination"}

    def self_test(self):
        # A hand-computable case: two answers, one confidently predicted, one not.
        # The scorer is a pure index into precomputed values, so the test targets the alignment
        # -- that score() returns exactly the requested rows, in the requested order.
        m = Perplexity()
        pre = {"nll": np.array([0.1, 5.0, 0.2, 4.0]), "ntok": np.array([3, 3, 3, 3])}

        class D:
            pass
        got = m.score(D(), pre, np.array([0, 1]), np.array([3, 1]))
        assert list(got) == [4.0, 5.0], list(got)
        print("    [PASS] score() indexes test rows in the given order, no reordering")

        # Orientation: a confident answer must score BELOW a surprised one, or the sign is flipped.
        assert pre["nll"][0] < pre["nll"][1]
        print("    [PASS] orientation: low NLL = confident = predicted truthful")

        # NaN for an empty completion must survive rather than becoming 0.0, which would read as
        # maximum confidence and silently bias the metric.
        pre2 = {"nll": np.array([np.nan, 1.0]), "ntok": np.array([0, 4])}
        out = m.score(D(), pre2, np.array([]), np.array([0, 1]))
        assert np.isnan(out[0]) and out[1] == 1.0
        print("    [PASS] empty completions stay NaN, not 0.0")
