"""methods/eigenscore.py -- EigenScore / INSIDE (Chen et al., ICLR 2024).

    Sigma = Z^T J_d Z          Z is (d, K): the K answers' sentence embeddings as COLUMNS
                               J_d = I_d - (1/d) 1 1^T is the centering matrix
    score =  (1/K) * log det(Sigma + alpha I_K)

Large covariance volume = the K answers occupy a wide region of embedding space = the model was
uncertain. The published form already points the right way for us: HIGH = spread = predicted
hallucination, matching every other method in the harness. It is NOT negated -- an earlier draft
negated it, which inverted every AUROC, and the self-test's orientation check caught it.

QUESTION granularity. The "Inside" row of both HARP's and HalluGuard's baseline tables, and by
their own runtime numbers the cheapest strong baseline in the field.

WHY THIS ROW IS WORTH MORE THAN A TABLE ENTRY. We found that HalluGuard's score is carried entirely
by its log-det term and reduces to counting distinct answers. EigenScore is a log-det of a
covariance over the same K embeddings. If the two correlate near 1, then HalluGuard's contribution
over a 2024 baseline is the two terms that we measured contributing nothing -- which is a claim we
can support with a number instead of an argument. Running them on identical data through the same
harness is what makes that comparison legitimate.

IMPLEMENTED FROM THE PAPER, NOT FROM THEIR CODE. We have not cloned INSIDE, so this is the formula
as published. Given that both papers we HAVE cloned turned out to disagree with their own
documentation, treat this as our reading until it is checked against their release. Two choices the
paper fixes and we follow: centering is across the EMBEDDING DIMENSIONS via J_d (not across the K
samples), and the regulariser alpha is inside the determinant. The layer is not fixed by the paper;
--m-layer exposes it and defaults to the middle layer, which is what their released configs use.
"""

import numpy as np

from .base import Method


def eigenscore(Z, alpha=1e-3):
    """Z is (d, K) -- embeddings as columns, matching the paper's convention.

    Returns (1/K) log det(Z^T J_d Z + alpha I). J_d centers each embedding across its dimensions.
    NOT negated: log det grows with the spread of the K answers, and spread is the uncertain case,
    so the published orientation already matches this harness's convention of high = hallucination."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[1] < 2:
        return np.nan
    d, K = Z.shape
    Zc = Z - Z.mean(axis=0, keepdims=True)        # J_d Z: subtract each column's own mean
    S = Zc.T @ Zc + alpha * np.eye(K)
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        return np.nan
    return float(logdet / K)


class EigenScore(Method):
    name = "eigenscore"
    granularity = "question"
    description = "INSIDE: log-det of the covariance of the K answer embeddings (Chen et al. 2024)"

    def __init__(self):
        self.layer = "middle"
        self.alpha = 1e-3
        self._diag = {}

    def add_args(self, parser):
        parser.add_argument("--m-layer", default="middle", choices=["middle", "final"],
                            help="which hidden layer to embed from; the paper does not fix it")
        parser.add_argument("--m-alpha", type=float, default=1e-3,
                            help="regulariser inside the determinant")

    def configure(self, args):
        self.layer, self.alpha = args.m_layer, args.m_alpha

    def precompute(self, data):
        import torch
        m = data.model()
        n_layers = m.config.num_hidden_layers
        sel = -1 if self.layer == "final" else (n_layers + 1) // 2
        groups = data.groups()
        print("    [eigenscore] %d questions | layer=%s (hidden_states[%s]) alpha=%g"
              % (len(groups), self.layer, sel, self.alpha), flush=True)

        out = np.full(len(data.question_ids), np.nan)
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
            with torch.no_grad():
                o = m(ids.to(data.device), attention_mask=att.to(data.device),
                      output_hidden_states=True)
            H = o.hidden_states[sel].float()
            cols = []
            for j in range(len(batch)):
                h = H[j, pls[j]:len(batch[j]), :]
                if h.shape[0]:
                    cols.append(h.mean(dim=0).cpu().numpy())     # one embedding per answer
            del o, H
            if len(cols) >= 2:
                out[gi] = eigenscore(np.stack(cols, axis=1), self.alpha)   # (d, K)
            if (gi + 1) % 200 == 0:
                print("      %d/%d questions" % (gi + 1, len(groups)), flush=True)

        self._diag = {"n_questions_scored": int(np.isfinite(out).sum()),
                      "n_questions_nan": int(np.isnan(out).sum())}
        return {"per_question": out}

    def score(self, data, pre, train_idx, test_idx):
        qs = np.unique(data.prompt_id[test_idx])
        pos = np.searchsorted(data.question_ids, qs)
        return pre["per_question"][pos]

    def meta(self):
        return {"layer": self.layer, "alpha": self.alpha,
                "orientation": "score = (1/K) log det, so higher = wider spread = hallucination",
                "provenance": "implemented from the paper; NOT verified against their release",
                **self._diag}

    def self_test(self):
        rng = np.random.default_rng(0)

        # Identical answers, in exact closed form. Centering is across DIMENSIONS, not across
        # samples, so K identical columns do NOT give a zero covariance -- each column becomes the
        # same centered vector c, Zc^T Zc is ||c||^2 * ones(K,K), and its eigenvalues are
        # K||c||^2 (once) and 0 (K-1 times). With the regulariser:
        #     log det = log(K||c||^2 + alpha) + (K-1) log(alpha)
        # My first version of this test asserted -log(alpha), on the assumption that identical
        # columns center to zero. They do not, and the assertion caught it.
        d, K, a = 32, 10, 1e-3
        z = rng.normal(size=(d, 1))
        dup = np.repeat(z, K, axis=1)
        c = z - z.mean()
        expect = (np.log(K * float(c.T @ c) + a) + (K - 1) * np.log(a)) / K
        s_dup = eigenscore(dup, a)
        assert abs(s_dup - expect) < 1e-8, (s_dup, expect)
        print("    [PASS] identical answers match the closed form exactly: %.4f" % s_dup)

        # A spread cloud must score HIGHER than a degenerate one: log det grows with volume,
        # spread means the model was uncertain, and uncertain means predicted hallucination. If
        # this inequality flips, every AUROC in the row inverts. It DID flip in the first draft.
        s_spread = eigenscore(rng.normal(size=(d, K)), a)
        assert s_spread > s_dup, (s_spread, s_dup)
        print("    [PASS] orientation: spread %.2f > identical %.2f -- high = uncertain = "
              "predicted hallucination, same direction as every other method" % (s_spread, s_dup))

        # Permuting the answers must not change the score -- their order is arbitrary. Centering is
        # per-column so it commutes with a column permutation, and Sigma -> P^T Sigma P leaves the
        # determinant alone.
        Z = rng.normal(size=(d, K))
        perm = rng.permutation(K)
        assert abs(eigenscore(Z[:, perm], a) - eigenscore(Z, a)) < 1e-9
        print("    [PASS] invariant to the order of the K answers")

        # NOT rotation-invariant, and that is a property of the PUBLISHED formula rather than a bug
        # here. Sigma = Z^T J_d Z with J_d = I - (1/d)11^T, and Q^T J_d Q = J_d only when Q fixes
        # the all-ones direction. So EigenScore depends on the coordinate frame of the embedding
        # space. Worth knowing when comparing it against HalluGuard, whose uncentered H H^T IS
        # rotation-invariant -- the two are not measuring the same geometric quantity.
        # I first asserted invariance here; the assertion failed and was right to.
        Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
        assert abs(eigenscore(Q @ Z, a) - eigenscore(Z, a)) > 1e-6
        print("    [PASS] NOT rotation-invariant -- a property of the published formula, asserted "
              "so nobody 'fixes' it later")

        # Degenerate inputs return NaN rather than raising or silently returning 0.
        assert np.isnan(eigenscore(rng.normal(size=(d, 1)), a))
        print("    [PASS] fewer than two answers returns NaN")

        # score() maps rows to question POSITIONS, with ids that do not start at zero.
        class D:
            prompt_id = np.array([4, 4, 6, 6, 8, 8])
            question_ids = np.array([4, 6, 8])
        out = EigenScore().score(D(), {"per_question": np.array([1.0, 2.0, 3.0])},
                                 np.array([0]), np.array([5, 3]))
        assert list(out) == [2.0, 3.0], list(out)
        print("    [PASS] score(): maps rows to question positions, not raw ids")
