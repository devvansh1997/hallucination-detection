"""
54_halluguard_proxy.py -- HalluGuard the way their published numbers were actually produced.
=====================================================================================================
WHY THIS EXISTS, AND WHY IT IS NOT 53.

53 runs `Score/halluguard_true.py`, which implements the paper's formula literally: per-token
gradients w.r.t. the last block, K = G G^T over T decoding steps. It costs ~0.93 GB per generated
token and cannot run past ~27 tokens on an 80GB H100. The paper reports Llama2-70B. Those two facts
cannot both be true of the same code path, and the resolution is that their repo holds TWO scores:

  Score/func/metric.py::getNTKS3Score   -- the DEFAULT. No gradients at all: mean-pooled hidden
                                           states at layer L/2 over the 10 sampled generations.
                                           Trivially cheap, scales to 70B.
  Score/halluguard_true.py              -- the paper's formula. Does not scale.

Their own `Score/TECHNICAL_SPEC_VERIFICATION.md` grades the default "NOT NTK: no theta-Jacobians;
over sequences, not steps; amplification is mean, not max", and concludes "Satisfied only when using
halluguard_true.py. Default scripts still use the proxy."

THE DEFAULT NEVER COMPUTES THE PUBLISHED FORMULA.
    metric.py:203-204 builds `CovMatrix = np.cov(...) + 1e-3*I` and then DISCARDS it. Line 212 is
        residual = np.sqrt(emb @ emb)      # "For now, use a simplified approach: just the norm"
    so det(K) is never computed and kappa(K) is never computed. What ships is
        ||mean-pooled h at layer L/2||  x  mean_t exp(||h_t - h_{t-1}||)
    against the paper's  det(K) + log sigma_max - log kappa(K)^2.  Also `mean` where the paper
    specifies `max` (metric.py:240).

    Two further defects worth recording, both harmless to the score only because the matrix is
    discarded: `np.cov(sequence_embeddings.T)` on an (N, D) array returns a (D, D) matrix, not the
    (N, N) their comment claims -- 3584x3584 of rank <= 9 for a 7B model. And it is built at full
    cost before being thrown away.

SO WE COMPUTE FOUR THINGS FROM ONE FORWARD PASS, AND THE COMPARISON IS THE POINT:
    A  as-coded      ||emb_i|| * mean_t exp(||delta||)      what their numbers most likely came from
    B  as-published  det(K) + log sigma_max - 2 log kappa   what the paper says the method is
    C  norm alone    ||emb_i||                              null: is the amplification term inert?
    D  length alone  T                                      null: is any of it beating token count?

WHAT WE CANNOT DO, STATED UP FRONT
    Appendix C.1: "We train only HALLUGUARD's lightweight projection layers using AdamW." No such
    layer exists in the release and no weights are published, so we run with an identity projection.
    Our numbers are therefore not their numbers and are not presented as a reproduction.

GRANULARITY -- the mismatch that no split protocol fixes
    Their score comes out PER QUESTION: getNTKS3Score returns np.mean over the 10 generations, and in
    getNTKS3ScoreOutput the amplification term is one scalar shared by all 10. So HalluGuard answers
    "did the model hallucinate on this question", while ours and HARP answer "is this answer a
    hallucination". We therefore report BOTH granularities for every score, with the per-question
    label being `not is_known` (the model never got this question right).

Usage:
  python 54_halluguard_proxy.py --self-test
  python 54_halluguard_proxy.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RIDGE = 1e-3          # metric.py:166 `alpha = 1e-3`, and Appendix C.1 "a small ridge alpha = 1e-3"
# Absolute floor on eigenvalues before log/ratio. Must NOT be tied to the ridge: at ridge=0 a
# relative floor collapses to 0, log_det goes to -inf and kappa divides by zero. With the ridge
# applied every eigenvalue exceeds it anyway, so this only ever guards numerical negatives.
EIG_FLOOR = 1e-12


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _s53():
    """Reuse 53's AUROC helpers rather than writing a third copy. 53's own self-test already checks
    them against 26_grouped_baseline and sklearn, so importing keeps all three scripts on one
    definition of the metric."""
    return _load_module("s53", os.path.join(HERE, "53_halluguard_score.py"))


# ==============================================================================
# PURE SCORE MATH -- no torch, no model, independently testable
# ==============================================================================

def log_mean_exp(d):
    """log( mean_t exp(d_t) ), computed stably.

    Their amplification term is `np.mean([np.exp(x) for x in deltas])` (metric.py:240). Hidden-state
    step norms in a 7B model are routinely 20-80, and exp(80) = 5.5e34 while exp(710) overflows to
    inf outright -- so the raw form loses beams to overflow. Since the final score is a PRODUCT,
    log(score) = log(norm) + log_mean_exp(deltas) is a strictly monotone transform of it, so every
    AUROC is identical while nothing overflows. We report the raw value too, and count how many
    beams it would have lost, because that is a fact about their implementation."""
    a = np.asarray(d, dtype=np.float64)
    if a.size == 0:
        return 0.0
    m = a.max()
    return float(m + np.log(np.mean(np.exp(a - m))))


def gram_spectrum(emb, ridge=RIDGE):
    """Eigen-spectrum of the Gram matrix over the GENERATIONS of one prompt.

    emb is (N, D): N generations, D hidden dims. K is (N, N) -- the paper's "NTK Gram matrix (over
    generated outputs)". Note their code computes np.cov(emb.T), which is (D, D), not what its own
    comment says; we take the (N, N) form their comment intends and the paper describes.

    Returns log|det K| rather than det K. det K is a product of N eigenvalues each of order
    ||h||^2 ~ 1e3-1e5, so it overflows float64 for N=10 -- their Appendix B anticipates this
    ("Use log det(K) via Cholesky for stability; replace det in the score with log det if desired
    (monotone equivalent)"). That parenthetical is wrong once det is SUMMED with the other two
    terms -- monotonicity of log does not survive addition -- so we return both and score both."""
    e = np.asarray(emb, dtype=np.float64)
    if e.ndim != 2 or e.shape[0] < 2:
        return None
    K = np.cov(e) + ridge * np.eye(e.shape[0])
    lam = np.linalg.eigvalsh(K)
    lam = np.clip(lam, EIG_FLOOR, None)         # cov of N samples has rank <= N-1; ridge floors it
    log_det = float(np.sum(np.log(lam)))
    with np.errstate(over="ignore"):
        det = float(np.exp(log_det))
    return {"log_det": log_det, "det": det,
            "kappa": float(lam.max() / lam.min()),
            "lambda_max": float(lam.max()), "lambda_min": float(lam.min()), "n": int(e.shape[0])}


def published_score(spec, sigma_max, use_log_det):
    """det(K) + log sigma_max - log kappa(K)^2, i.e. -2 log kappa. Eq. 7 of the paper."""
    if spec is None or not np.isfinite(sigma_max) or sigma_max <= 0:
        return None
    head = spec["log_det"] if use_log_det else spec["det"]
    if not np.isfinite(head):
        return None
    return float(head + np.log(sigma_max) - 2.0 * np.log(spec["kappa"]))


def lipschitz_ratios(deltas, eps=1e-8):
    """sigma_max as halluguard_true.py:44-51 defines it -- max over t of the ratio of consecutive
    step sizes, not the raw step size. Needs at least 3 generated tokens, exactly as theirs does;
    below that theirs returns a degenerate 1.0 and we record NaN so it can be excluded knowingly."""
    d = np.asarray(deltas, dtype=np.float64)
    if d.size < 2:
        return np.nan
    return float(np.max(d[1:] / (d[:-1] + eps)))


def prompt_level_labels(labels, prompt_ids):
    """A question is 'hallucinated' iff the model never got it right -- the complement of the
    `is_known` rule already used everywhere else in this project, so the two granularities rest on
    one definition rather than two."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(prompt_ids)
    uniq = np.unique(p)
    return uniq, np.array([int((y[p == q] == 1).all()) for q in uniq], dtype=int)


def aggregate_to_prompt(values, prompt_ids, how="mean"):
    """metric.py:252 returns np.mean over the prompt's generations. Replicated so the per-question
    numbers are theirs, not an aggregation of our choosing."""
    v = np.asarray(values, dtype=float)
    p = np.asarray(prompt_ids)
    uniq = np.unique(p)
    out = np.full(len(uniq), np.nan)
    for i, q in enumerate(uniq):
        sel = v[p == q]
        sel = sel[np.isfinite(sel)]
        if sel.size:
            out[i] = sel.mean() if how == "mean" else sel.max()
    return uniq, out


# ==============================================================================

def extract(dataset, model_folder, data_dir, device, dtype, limit=None, log_every=50):
    """One no-grad forward pass per prompt-group. We need exactly two things per beam from layer
    L/2: the mean-pooled hidden state over generated tokens, and the consecutive-token step norms."""
    import torch
    import yaml
    from transformers import AutoModelForCausalLM

    seq_path = os.path.join(data_dir, model_folder, "%s_sequences_v1.pt" % dataset)
    if not os.path.exists(seq_path):
        raise FileNotFoundError("%s not found -- run 39_generate_dataset.py first." % seq_path)
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)

    seq = torch.load(seq_path, weights_only=False)
    input_ids, prompt_lens = seq["input_ids"], seq["prompt_len"]
    prompt_ids = np.asarray(seq["prompt_id"])
    labels = np.asarray(seq["all_hallucination_flag"], dtype=int)

    order = np.argsort(prompt_ids, kind="stable")
    groups = []
    for q in np.unique(prompt_ids):
        rows = order[prompt_ids[order] == q]
        groups.append((int(q), rows))
    if limit is not None:
        groups = groups[:limit]
    n_beams = sum(len(r) for _, r in groups)

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype,
                                                 trust_remote_code=True).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    # metric.py:167 -- `int(len(hidden_states[0])/2)` over a tuple of n_layers+1 tensors.
    sel = (n_layers + 1) // 2
    print("  [%s] %d prompts / %d beams | %d layers, layer L/2 -> hidden_states[%d]"
          % (dataset, len(groups), n_beams, n_layers, sel), flush=True)

    rows, embs = [], []
    t0 = time.time()
    for gi, (q, idx) in enumerate(groups):
        batch = [input_ids[i] for i in idx]
        pls = [int(prompt_lens[i]) for i in idx]
        L = max(len(b) for b in batch)
        pad = model.config.eos_token_id or 0
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, b in enumerate(batch):
            ids[j, :len(b)] = b if torch.is_tensor(b) else torch.as_tensor(b)
            att[j, :len(b)] = 1
        ids, att = ids.to(device), att.to(device)
        with torch.no_grad():
            out = model(ids, attention_mask=att, output_hidden_states=True)
        H = out.hidden_states[sel].float()               # (B, L, D)
        for j, i in enumerate(idx):
            pl, tot = pls[j], len(batch[j])
            h = H[j, pl:tot, :]                           # generated tokens only
            T = int(h.shape[0])
            if T == 0:
                rows.append({"row": int(i), "prompt_id": int(q), "T": 0, "norm": np.nan,
                             "log_amp": np.nan, "sigma_max": np.nan, "max_delta": np.nan,
                             "raw_amp_overflow": False})
                embs.append(None)
                continue
            emb = h.mean(dim=0)
            d = torch.norm(h[1:] - h[:-1], dim=-1).cpu().numpy().astype(np.float64) if T >= 2 \
                else np.zeros(0)
            with np.errstate(over="ignore"):
                raw_amp = float(np.mean(np.exp(d))) if d.size else 1.0
            rows.append({"row": int(i), "prompt_id": int(q), "T": T,
                         "norm": float(torch.norm(emb).item()),
                         "log_amp": log_mean_exp(d) if d.size else 0.0,
                         "sigma_max": lipschitz_ratios(d),
                         "max_delta": float(d.max()) if d.size else np.nan,
                         "raw_amp_overflow": not np.isfinite(raw_amp)})
            embs.append(emb.cpu().numpy())
        del out, H
        if (gi + 1) % log_every == 0:
            el = time.time() - t0
            print("    %s %d/%d prompts (%.0fs, eta %.0fs)"
                  % (dataset, gi + 1, len(groups), el, el / (gi + 1) * (len(groups) - gi - 1)),
                  flush=True)

    keep = np.array([r["row"] for r in rows], dtype=int)
    return rows, embs, labels[keep], prompt_ids[keep], model_id, sel, n_layers


def assemble(rows, embs, y, pid, use_log_det=True):
    """Build the four scores and evaluate each at both granularities."""
    s53 = _s53()
    T = np.array([r["T"] for r in rows], dtype=float)
    norm = np.array([r["norm"] for r in rows], dtype=float)
    log_amp = np.array([r["log_amp"] for r in rows], dtype=float)
    sig = np.array([r["sigma_max"] for r in rows], dtype=float)

    with np.errstate(invalid="ignore"):
        # A, in log space: log(||emb|| * mean_t exp(delta)) -- rank-identical to their raw product.
        A = np.log(np.maximum(norm, 1e-300)) + log_amp
    C = norm

    # B is per PROMPT: the Gram matrix is over that prompt's generations.
    uniq = np.unique(pid)
    B_by_prompt, spec_rows = np.full(len(uniq), np.nan), []
    for i, q in enumerate(uniq):
        sel = np.where(pid == q)[0]
        E = [embs[k] for k in sel if embs[k] is not None]
        spec = gram_spectrum(np.stack(E)) if len(E) >= 2 else None
        s_valid = sig[sel][np.isfinite(sig[sel])]
        sm = float(s_valid.max()) if s_valid.size else np.nan
        B_by_prompt[i] = published_score(spec, sm, use_log_det) if spec else np.nan
        spec_rows.append({"prompt_id": int(q), "sigma_max": sm,
                          **({k: spec[k] for k in ("log_det", "kappa", "lambda_min", "n")}
                             if spec else {})})

    _, y_prompt = prompt_level_labels(y, pid)
    res = {"n_beams": int(len(y)), "n_prompts": int(len(uniq)),
           "beam_hallucination_rate_pct": round(100.0 * float(y.mean()), 3),
           "prompt_hallucination_rate_pct": round(100.0 * float(y_prompt.mean()), 3),
           "n_raw_amp_overflow": int(sum(r["raw_amp_overflow"] for r in rows)),
           "det_head": "log_det" if use_log_det else "det"}

    def beam_block(v):
        f = np.isfinite(v)
        if f.sum() < 2:
            return None
        a = s53.pooled_auroc(v[f], y[f])
        w = s53.within_prompt_auroc(v[f], y[f], pid[f])
        return {"pooled_auroc": a, "pooled_auroc_flipped": None if a is None else 1.0 - a,
                "within_prompt_auroc": w["within_prompt_auroc"], "n_pairs": w["n_pairs"],
                "n_scored": int(f.sum())}

    def prompt_block(v):
        f = np.isfinite(v)
        if f.sum() < 2:
            return None
        a = s53.pooled_auroc(v[f], y_prompt[f])
        return {"auroc": a, "auroc_flipped": None if a is None else 1.0 - a,
                "n_scored": int(f.sum())}

    res["per_beam"] = {"A_as_coded": beam_block(A), "C_norm_alone": beam_block(C),
                       "D_length_alone": beam_block(T)}
    res["per_prompt"] = {"B_as_published": prompt_block(B_by_prompt)}
    for name, v in (("A_as_coded", A), ("C_norm_alone", C), ("D_length_alone", T)):
        res["per_prompt"][name] = prompt_block(aggregate_to_prompt(v, pid)[1])
    return res, {"A": A, "C": C, "T": T, "sigma_max": sig, "B_by_prompt": B_by_prompt,
                 "prompt_ids_unique": uniq, "y_prompt": y_prompt}, spec_rows


def self_test():
    print("=" * 74)
    print("  SELF-TEST: 54_halluguard_proxy (synthetic, no model, no cluster files)")
    print("=" * 74)
    rng = np.random.default_rng(0)

    # log_mean_exp against the naive form where the naive form still works
    d = np.array([1.0, 2.0, 3.0])
    assert abs(log_mean_exp(d) - np.log(np.mean(np.exp(d)))) < 1e-12
    big = np.array([800.0, 801.0])                      # exp() of these is inf in float64
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.mean(np.exp(big))), "test premise: the naive form must overflow"
    # 800 + log((1 + e)/2), not 800 + log 2 -- the two entries differ, so the mean of the
    # shifted exponentials is (1 + e)/2. Caught by this assertion on first run.
    assert np.isfinite(log_mean_exp(big))
    assert abs(log_mean_exp(big) - (800.0 + np.log((1.0 + np.e) / 2.0))) < 1e-9, log_mean_exp(big)
    print("  [PASS] log_mean_exp: matches the naive mean-of-exp, and survives where it overflows")

    # gram_spectrum on a case with a known answer: two orthogonal directions, N=3
    E = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    sp = gram_spectrum(E, ridge=0.0)
    K = np.cov(E)
    ref = float(np.sum(np.log(np.clip(np.linalg.eigvalsh(K), EIG_FLOOR, None))))
    assert abs(sp["log_det"] - ref) < 1e-9, (sp["log_det"], ref)
    assert np.isfinite(sp["log_det"]) and np.isfinite(sp["kappa"]),         "ridge=0 must still yield finite values -- the eigenvalue floor is absolute, not relative"
    assert sp["n"] == 3
    # cov of N points has rank <= N-1, so without a ridge the spectrum is singular -- the ridge is
    # doing real work, not decoration, and kappa must fall when it is applied.
    k_no = gram_spectrum(E, ridge=1e-9)["kappa"]
    k_yes = gram_spectrum(E, ridge=1e-3)["kappa"]
    assert k_yes < k_no, (k_yes, k_no)
    print("  [PASS] gram_spectrum: N x N Gram, rank deficiency real, ridge lowers kappa %.3g -> %.3g"
          % (k_no, k_yes))

    # the paper's "monotone equivalent" parenthetical is false once det is summed with other terms
    # Chosen so the det gap (13) clears the kappa penalty gap (2 log 100 = 9.21) but the log-det
    # gap (2.01) does not. A case where both agree would not test the claim at all.
    sp_a = {"log_det": np.log(2.0), "det": 2.0, "kappa": 10.0}
    sp_b = {"log_det": np.log(15.0), "det": 15.0, "kappa": 1000.0}
    d_a, d_b = published_score(sp_a, 2.0, False), published_score(sp_b, 2.0, False)
    l_a, l_b = published_score(sp_a, 2.0, True), published_score(sp_b, 2.0, True)
    assert (d_a > d_b) != (l_a > l_b), (d_a, d_b, l_a, l_b)
    print("  [PASS] published_score: det and log-det ORDER PROMPTS DIFFERENTLY -- their Appendix B "
          "calls the swap 'monotone equivalent', which holds only for the term alone")

    # lipschitz_ratios matches halluguard_true's definition on a hand case
    assert abs(lipschitz_ratios(np.array([1.0, 2.0, 8.0])) - 4.0) < 1e-6
    assert not np.isfinite(lipschitz_ratios(np.array([1.0])))
    print("  [PASS] lipschitz_ratios: max consecutive ratio; NaN when there are too few steps")

    # prompt labels: hallucinated iff NO beam is truthful
    uq, yp = prompt_level_labels([1, 1, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])
    assert list(uq) == [0, 1] and list(yp) == [0, 1]
    print("  [PASS] prompt_level_labels: a question counts as hallucinated only if every beam is")

    # aggregate_to_prompt replicates their np.mean, and ignores NaNs rather than propagating
    _, agg = aggregate_to_prompt([1.0, 3.0, np.nan, 10.0], [0, 0, 1, 1])
    assert agg[0] == 2.0 and agg[1] == 10.0
    print("  [PASS] aggregate_to_prompt: mean over a prompt's beams, NaNs skipped not propagated")

    # end-to-end on planted data: the norm carries the signal, length does not.
    n_q, n_b = 60, 10
    pid = np.repeat(np.arange(n_q), n_b)
    y = np.zeros(n_q * n_b, dtype=int)
    for q in range(n_q // 2):                            # half the questions fully hallucinated
        y[pid == q] = 1
    rows, embs = [], []
    for i in range(n_q * n_b):
        base = 3.0 if y[i] else 1.0
        e = rng.normal(size=8) * 0.1 + base
        rows.append({"row": i, "prompt_id": int(pid[i]), "T": int(rng.integers(5, 20)),
                     "norm": float(np.linalg.norm(e)), "log_amp": float(rng.normal()),
                     "sigma_max": float(abs(rng.normal()) + 1.0), "max_delta": 1.0,
                     "raw_amp_overflow": False})
        embs.append(e)
    res, _, _ = assemble(rows, embs, y, pid)
    assert res["per_beam"]["C_norm_alone"]["pooled_auroc"] > 0.95, res["per_beam"]["C_norm_alone"]
    assert 0.4 < res["per_beam"]["D_length_alone"]["pooled_auroc"] < 0.6, "length must be at chance"
    assert res["per_prompt"]["B_as_published"]["auroc"] is not None
    assert res["prompt_hallucination_rate_pct"] == 50.0
    print("  [PASS] assemble: planted norm signal recovered (%.3f), length at chance (%.3f), "
          "per-prompt B produced" % (res["per_beam"]["C_norm_alone"]["pooled_auroc"],
                                     res["per_beam"]["D_length_alone"]["pooled_auroc"]))

    # within-prompt must be UNDEFINED here: every question is single-class by construction, which
    # is exactly the regime a prompt-level detector lives in. If this ever returns a number, the
    # planted data stopped testing what it was built to test.
    assert res["per_beam"]["C_norm_alone"]["within_prompt_auroc"] is None
    print("  [PASS] assemble: within-prompt is None when no question is mixed, not silently 0.5")

    print("\n[PASS] All self-test assertions passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tydiqa_gp",
                    choices=["truthfulqa", "triviaqa", "nq_open", "tydiqa_gp"])
    ap.add_argument("--model_folder", default="qwen-2.5-7b-instruct")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "halluguard_proxy"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--limit", type=int, default=None, help="first N PROMPTS (not beams)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--use-raw-det", action="store_true",
                    help="score with det(K) rather than log det(K). Overflows for N=10; kept "
                         "because it is what Eq. 7 literally says.")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return

    data_dir = a.data_dir
    if not data_dir:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]

    print("=" * 78)
    print("  HALLUGUARD PROXY -- %s / %s" % (a.model_folder, a.dataset))
    print("  the default path from Score/func/metric.py, which needs no gradients")
    print("  identity projection: the trained layer of Appendix C.1 is not in their release")
    print("=" * 78, flush=True)

    rows, embs, y, pid, model_id, sel, n_layers = extract(
        a.dataset, a.model_folder, data_dir, a.device, a.dtype, a.limit, a.log_every)
    res, arr, spec_rows = assemble(rows, embs, y, pid, use_log_det=not a.use_raw_det)
    res.update({"dataset": a.dataset, "model_folder": a.model_folder, "model_id": model_id,
                "dtype": a.dtype, "layer_index_used": sel, "n_layers": n_layers,
                "ridge": RIDGE, "projection": "identity (theirs is unreleased)"})

    os.makedirs(a.out_dir, exist_ok=True)
    stem = "hgproxy_%s_%s" % (a.model_folder, a.dataset)
    np.savez_compressed(os.path.join(a.out_dir, stem + "_scores.npz"),
                        label=y, prompt_id=pid, **{k: v for k, v in arr.items()})
    with open(os.path.join(a.out_dir, stem + ".json"), "w") as f:
        json.dump({**res, "per_prompt_spectra": spec_rows[:200]}, f, indent=2)

    print("\n  beams %d | prompts %d | halluc %.1f%% (beam) %.1f%% (prompt)" % (
        res["n_beams"], res["n_prompts"], res["beam_hallucination_rate_pct"],
        res["prompt_hallucination_rate_pct"]))
    if res["n_raw_amp_overflow"]:
        print("  NOTE: their raw mean-of-exp amplification overflows on %d beams; we rank in log "
              "space, which is order-identical" % res["n_raw_amp_overflow"])
    print("\n  PER BEAM (comparable to ours and HARP)          pooled   flipped   within-prompt")
    for k, v in res["per_beam"].items():
        if v:
            print("    %-18s %8.4f  %8.4f   %s" % (
                k, v["pooled_auroc"], v["pooled_auroc_flipped"],
                "%.4f" % v["within_prompt_auroc"] if v["within_prompt_auroc"] is not None else "n/a"))
    print("\n  PER QUESTION (the granularity their code returns)   auroc   flipped")
    for k, v in res["per_prompt"].items():
        if v:
            print("    %-18s %8.4f  %8.4f" % (k, v["auroc"], v["auroc_flipped"]))
    print("\nWrote: %s{.json,_scores.npz}" % os.path.join(a.out_dir, stem))


if __name__ == "__main__":
    main()
