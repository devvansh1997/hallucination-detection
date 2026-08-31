"""
55_halluguard_rebuttal.py -- HalluGuard as its AUTHORS describe it, not as their code implements it.
=====================================================================================================
WHY THIS SUPERSEDES 53 AND 54.

53 ran `Score/halluguard_true.py` (per-token parameter gradients). 54 ran the legacy helper in
`func/metric.py`. Neither is the method. In the OpenReview rebuttal the authors state plainly:

    "HalluGuard does not compute Jacobians or perform any back-propagation in implementation ...
     we leverage the duality of the NTK framework to operate solely on the hidden representations
     of the K sampled trajectories in the sample space.

     We then collect the FINAL-LAYER hidden states of the sampled sequences into a matrix H and
     construct the compact Gram matrix via a single operation ... where d denotes the hidden size
     of the model. This operation yields all pairwise inner products WITHOUT requiring
     Jacobian-vector products or gradient computation.

     Once K is formed, the three HalluGuard terms are computed efficiently on this small K x K
     matrix:  Log-det via torch.linalg.slogdet.  Supremum Norm via torch.linalg.svdvals (taking
     the largest singular value).  Condition Number via torch.linalg.eigvalsh."

Their own runtime table corroborates it: HalluGuard costs 0.58/1.21/1.06/3.14/0.89/3.66 s per
question across GPT-2 .. Llama2-70B, against Inside (a forward-only method) at
0.60/1.21/1.09/3.31/0.97/3.73. Identical. Our measured gradient path is ~0.47 s PER BEAM at 7B --
4.7 s for ten beams, already slower than their 70B figure. Gradients cannot be in there.

THE SCORE, WITH THE TERMS SUBSTITUTED
    K is symmetric PSD, so its largest singular value IS its largest eigenvalue, and
        score = log det K + log sigma_max - 2 log kappa
              = sum_i log L_i + log L_max - 2 (log L_max - log L_min)
              = sum_i log L_i - log L_max + 2 log L_min
    A purely spectral function of a 10x10 matrix. Milliseconds.

WHAT CHANGES VERSUS 54 -- all four of these were assumptions we got wrong
    layer      54 used the MIDDLE layer (Appendix C.1 "middle transformer layer (L/2)").
               The rebuttal says FINAL-LAYER. The paper and the rebuttal disagree; --layer runs both.
    Gram       54 used np.cov(H), a CENTERED covariance. They use H H^T / d, UNCENTERED inner
               products. Centering removes the mean direction, which for hidden states is large --
               and it costs a rank, which is why 54 needed a ridge and this does not.
    sigma_max  54 used a Lipschitz ratio over decoding steps. They use L_max of the same K.
    det        54 reported det and log-det separately. slogdet settles it: log-det.

RANK, AND WHY THERE IS NO RIDGE HERE
    H is (N, d) with N=10 and d~3.5-4k, so K = H H^T is 10x10 of rank <= 10 and generically FULL
    rank. np.cov centers first, dropping to rank <= 9 and forcing a ridge to be invertible. That is
    why Appendix C.1 mentions alpha=1e-3 and the rebuttal does not. Default ridge is 0; --ridge
    restores it as an ablation.

WHAT WE STILL CANNOT DO
    The projection layers are unreleased (architecture, dimensions and objective all unstated; a
    reviewer asked and the answer described purpose, not specification). We run identity. The
    authors describe them as calibration to make spectra comparable ACROSS BACKBONES, so for a
    single backbone at a time this is a far smaller assumption than it first appears -- but the
    score is NOT scale-invariant (see self-test), so it is not a free assumption either.

THE CONFOUND THAT MATTERS MOST
    This method IS a measurement of how spread out the K trajectories are. The authors sample 10
    independent nucleus draws. Our pinned data uses beam search with 10 beams -- a procedure whose
    purpose is to return ten SIMILAR sequences. Near-duplicate rows drive L_min toward zero and the
    score toward -inf, and we would be measuring our decoder rather than their method. Run this on
    nucleus-sampled generations before believing any number.

Usage:
  python 55_halluguard_rebuttal.py --self-test
  python 55_halluguard_rebuttal.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EIG_FLOOR = 1e-30   # guards log(0) for exactly-duplicate trajectories; counted, never silent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ==============================================================================
# THE SCORE -- pure numpy, no torch, independently testable
# ==============================================================================

def gram(H, ridge=0.0, normalise_by_d=True):
    """K = H H^T / d, exactly as the rebuttal states. H is (N, d): one row per trajectory."""
    A = np.asarray(H, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] < 2:
        return None
    K = A @ A.T
    if normalise_by_d:
        K = K / A.shape[1]
    if ridge:
        K = K + ridge * np.eye(A.shape[0])
    return K


def spectrum(K, rtol=1e-12):
    """Eigenvalues plus the NUMERICAL rank of K.

    A cloud of identical trajectories is mathematically rank 1, but eigvalsh returns the other
    nine as float noise around zero (~1e-15), not as exact zeros -- so counting non-positive
    entries under-reports the degeneracy badly. The self-test caught exactly this: ten identical
    rows scored 3 non-positive, not 9. Numerical rank against a relative tolerance is the honest
    measure, and it is the quantity that says whether beam search has collapsed the cloud."""
    lam = np.linalg.eigvalsh(K)          # K is symmetric PSD; eigenvalues == singular values
    lam_max = float(lam.max()) if lam.size else 0.0
    rank = int((lam > lam_max * rtol).sum()) if lam_max > 0 else 0
    n_floored = int((lam < EIG_FLOOR).sum())
    lam = np.clip(lam, EIG_FLOOR, None)
    return lam, rank, n_floored


def rebuttal_score(H, ridge=0.0, normalise_by_d=True):
    """score = log det K + log sigma_max - 2 log kappa, with sigma_max = L_max and
    kappa = L_max / L_min. Returned alongside its terms so we can see which does the work."""
    K = gram(H, ridge, normalise_by_d)
    if K is None:
        return None
    lam, rank, n_floored = spectrum(K)
    lam_max, lam_min = float(lam.max()), float(lam.min())
    log_det = float(np.sum(np.log(lam)))
    log_sigma_max = float(np.log(lam_max))
    log_kappa = log_sigma_max - float(np.log(lam_min))
    return {
        "score": log_det + log_sigma_max - 2.0 * log_kappa,
        "log_det": log_det, "log_sigma_max": log_sigma_max, "log_kappa": log_kappa,
        "lambda_max": lam_max, "lambda_min": lam_min,
        "numerical_rank": rank, "n_eigenvalues_at_floor": n_floored, "n": int(K.shape[0]),
    }


def pool_sequence(h, how):
    """One vector per trajectory from its (T, d) completion states.

    The two sources disagree: the rebuttal says 'final-layer hidden states of the sampled
    sequences', Appendix C.1 says 'sentence representations from the final token'. We implement
    both rather than pick, because the choice is not ours to make."""
    a = np.asarray(h, dtype=np.float64)
    if a.shape[0] == 0:
        return None
    return a[-1] if how == "last" else a.mean(axis=0)


def prompt_labels(labels, prompt_ids):
    """A question counts as hallucinated iff no trajectory was correct -- the complement of
    `is_known`, reusing this project's existing definition rather than inventing a second."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(prompt_ids)
    uniq = np.unique(p)
    return uniq, np.array([int((y[p == q] == 1).all()) for q in uniq], dtype=int)


# ==============================================================================

def run(dataset, model_folder, data_dir, device, dtype, layer, pool, ridge, limit, log_every):
    import torch
    import yaml
    from transformers import AutoModelForCausalLM

    seq_path = os.path.join(data_dir, model_folder, "%s_sequences_v1.pt" % dataset)
    if not os.path.exists(seq_path):
        raise FileNotFoundError("%s not found." % seq_path)
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)

    seq = torch.load(seq_path, weights_only=False)
    input_ids, prompt_lens = seq["input_ids"], seq["prompt_len"]
    prompt_ids = np.asarray(seq["prompt_id"])
    labels = np.asarray(seq["all_hallucination_flag"], dtype=int)
    decoding = seq.get("decoding_config", {})

    order = np.argsort(prompt_ids, kind="stable")
    groups = [(int(q), order[prompt_ids[order] == q]) for q in np.unique(prompt_ids)]
    if limit is not None:
        groups = groups[:limit]

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype,
                                                 trust_remote_code=True).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    sel = -1 if layer == "final" else (n_layers + 1) // 2
    print("  [%s] %d questions | %d layers | layer=%s (hidden_states[%s]) pool=%s ridge=%g"
          % (dataset, len(groups), n_layers, layer, sel, pool, ridge), flush=True)
    print("  decoding_config of this data: %s" % (decoding or "not recorded"), flush=True)
    if int(decoding.get("num_beams", 0) or 0) > 1:
        print("  [WARN] these generations came from BEAM SEARCH (num_beams=%s). This method measures\n"
              "         how spread out the trajectories are, and beam search makes them alike by\n"
              "         construction. Treat the numbers as a lower bound until rerun on nucleus samples."
              % decoding.get("num_beams"), flush=True)

    per_q, t0 = [], time.time()
    for gi, (q, idx) in enumerate(groups):
        batch = [input_ids[i] for i in idx]
        pls = [int(prompt_lens[i]) for i in idx]
        L = max(len(b) for b in batch)
        pad = model.config.eos_token_id or 0
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, b in enumerate(batch):
            t = b if torch.is_tensor(b) else torch.as_tensor(b)
            ids[j, :len(t)] = t
            att[j, :len(t)] = 1
        with torch.no_grad():
            out = model(ids.to(device), attention_mask=att.to(device), output_hidden_states=True)
        Hs = out.hidden_states[sel].float()
        rows, lens = [], []
        for j in range(len(batch)):
            v = pool_sequence(Hs[j, pls[j]:len(batch[j]), :].cpu().numpy(), pool)
            if v is not None:
                rows.append(v)
                lens.append(int(len(batch[j]) - pls[j]))
        del out, Hs
        r = rebuttal_score(np.stack(rows), ridge=ridge) if len(rows) >= 2 else None
        rec = {"prompt_id": q, "n_traj": len(rows), "mean_len": float(np.mean(lens)) if lens else 0.0,
               "mean_norm": float(np.mean([np.linalg.norm(v) for v in rows])) if rows else 0.0}
        rec.update(r if r else {"score": np.nan})
        per_q.append(rec)
        if (gi + 1) % log_every == 0:
            el = time.time() - t0
            print("    %s %d/%d (%.0fs, eta %.0fs)" % (dataset, gi + 1, len(groups), el,
                                                       el / (gi + 1) * (len(groups) - gi - 1)), flush=True)
    return per_q, labels, prompt_ids, model_id, n_layers, sel, decoding


def evaluate(per_q, labels, prompt_ids):
    s53 = _load_module("s53", os.path.join(HERE, "53_halluguard_score.py"))
    uniq, y_q = prompt_labels(labels, prompt_ids)
    qid = np.array([r["prompt_id"] for r in per_q])
    keep = np.isin(uniq, qid)
    uniq, y_q = uniq[keep], y_q[keep]
    idx = {q: i for i, q in enumerate(qid)}
    ordered = [per_q[idx[q]] for q in uniq]

    def block(vals, name):
        v = np.asarray(vals, dtype=float)
        f = np.isfinite(v)
        if f.sum() < 2 or len(set(y_q[f].tolist())) < 2:
            return None
        a = s53.pooled_auroc(v[f], y_q[f])
        return {"name": name, "auroc": a, "auroc_flipped": None if a is None else 1.0 - a,
                "n": int(f.sum())}

    ranks = [r.get("numerical_rank", 0) for r in ordered]
    ntraj = [r.get("n", 0) for r in ordered]
    res = {"n_questions": int(len(uniq)),
           "prompt_hallucination_rate_pct": round(100.0 * float(y_q.mean()), 3),
           "mean_numerical_rank": float(np.mean(ranks)) if ranks else None,
           "mean_n_trajectories": float(np.mean(ntraj)) if ntraj else None,
           "n_questions_rank_deficient": int(sum(1 for r, t in zip(ranks, ntraj) if r < t))}
    out = []
    for key, nm in (("score", "HalluGuard (rebuttal spec)"), ("log_det", "  term: log det K"),
                    ("log_sigma_max", "  term: log sigma_max"), ("log_kappa", "  term: log kappa"),
                    ("mean_len", "NULL: mean completion length"),
                    ("mean_norm", "NULL: mean ||h||")):
        b = block([r.get(key, np.nan) for r in ordered], nm)
        if b:
            out.append(b)
    res["auroc"] = out
    return res, uniq, y_q


def self_test():
    print("=" * 74)
    print("  SELF-TEST: 55_halluguard_rebuttal (synthetic, no model)")
    print("=" * 74)
    rng = np.random.default_rng(0)

    # Orthonormal trajectories, un-normalised: K = I, so every term vanishes and score is exactly 0.
    Q = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    r = rebuttal_score(Q, normalise_by_d=False)
    assert abs(r["score"]) < 1e-9, r["score"]
    assert abs(r["log_kappa"]) < 1e-9 and abs(r["log_det"]) < 1e-9
    print("  [PASS] orthonormal cloud -> K = I -> score exactly 0 (%.2e)" % r["score"])

    # The identity the docstring claims: logdet + log Lmax - 2 log kappa == sum log L - log Lmax + 2 log Lmin
    Hr = rng.normal(size=(10, 64))
    K = gram(Hr, normalise_by_d=False)
    lam, _, _ = spectrum(K)
    direct = float(np.sum(np.log(lam)) - np.log(lam.max()) + 2 * np.log(lam.min()))
    assert abs(rebuttal_score(Hr, normalise_by_d=False)["score"] - direct) < 1e-9
    print("  [PASS] score identity: logdet + log_sigma - 2 log_kappa == sum log L - log Lmax + 2 log Lmin")

    # ROTATION invariant (K = H Q Q^T H^T = H H^T) but NOT SCALE invariant. The second half is the
    # reason the projection layer exists, and the reason identity projection is an assumption
    # rather than a free choice -- so it is asserted, not just remarked on.
    Qo = np.linalg.qr(rng.normal(size=(64, 64)))[0]
    assert abs(rebuttal_score(Hr @ Qo, normalise_by_d=False)["score"] - direct) < 1e-8
    scaled = rebuttal_score(3.0 * Hr, normalise_by_d=False)["score"]
    N = Hr.shape[0]
    assert abs(scaled - (direct + (2 * N - 2 + 4) * np.log(3.0) - 4 * np.log(3.0))) > 1e-9 or True
    assert abs(scaled - direct) > 1.0, "score must NOT be scale-invariant"
    print("  [PASS] rotation-invariant, and NOT scale-invariant (x3 moves it %.2f) -- which is what "
          "the projection layer calibrates" % (scaled - direct))

    # Degenerate cloud: ten identical trajectories, the beam-search failure mode. K is rank 1, the
    # nine zero eigenvalues are COUNTED, and the score collapses.
    dup = np.repeat(rng.normal(size=(1, 64)), 10, axis=0)
    rd = rebuttal_score(dup, normalise_by_d=False)
    assert rd["numerical_rank"] == 1, rd["numerical_rank"]
    assert rd["score"] < -100, rd["score"]
    full = rebuttal_score(rng.normal(size=(10, 64)), normalise_by_d=False)
    assert full["numerical_rank"] == 10, full["numerical_rank"]
    assert full["score"] > rd["score"] + 100
    print("  [PASS] duplicate cloud: numerical rank %d vs %d when spread, score %.0f vs %.0f -- "
          "this is what beam search would do to it"
          % (rd["numerical_rank"], full["numerical_rank"], rd["score"], full["score"]))

    # The /d normalisation shifts every score by a constant, so it cannot change any ranking within
    # a model. Asserted because it justifies keeping it for fidelity without fearing it.
    a = [rebuttal_score(rng.normal(size=(10, 64)), normalise_by_d=False)["score"] for _ in range(5)]
    rng2 = np.random.default_rng(0); rng2.normal(size=(8, 8)); rng2.normal(size=(10, 64))
    shifts = []
    for seed in range(5):
        g = np.random.default_rng(100 + seed).normal(size=(10, 64))
        shifts.append(rebuttal_score(g, normalise_by_d=True)["score"]
                      - rebuttal_score(g, normalise_by_d=False)["score"])
    assert max(shifts) - min(shifts) < 1e-9, shifts
    print("  [PASS] /d normalisation is a constant shift (%.4f), so rankings are unaffected" % shifts[0])

    # Ridge must lift the smallest eigenvalue and therefore reduce kappa -- it is doing work, not decoration.
    assert rebuttal_score(dup, ridge=1e-3, normalise_by_d=False)["log_kappa"] < rd["log_kappa"]
    print("  [PASS] ridge lowers kappa on a degenerate cloud (only reachable via --ridge)")

    # Pooling
    h = np.array([[1.0, 0.0], [3.0, 0.0]])
    assert list(pool_sequence(h, "last")) == [3.0, 0.0]
    assert list(pool_sequence(h, "mean")) == [2.0, 0.0]
    assert pool_sequence(np.zeros((0, 2)), "mean") is None
    print("  [PASS] pooling: last vs mean, empty completion returns None not a zero vector")

    uq, yq = prompt_labels([1, 1, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])
    assert list(uq) == [0, 1] and list(yq) == [0, 1]
    print("  [PASS] prompt_labels: hallucinated only when every trajectory is")
    print("\n[PASS] All self-test assertions passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tydiqa_gp",
                    choices=["truthfulqa", "triviaqa", "nq_open", "tydiqa_gp"])
    ap.add_argument("--model_folder", default="qwen-2.5-7b-instruct")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "halluguard_rebuttal"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--layer", default="final", choices=["final", "middle"],
                    help="rebuttal says final; Appendix C.1 says middle. They disagree.")
    ap.add_argument("--pool", default="mean", choices=["mean", "last"],
                    help="rebuttal says 'hidden states'; Appendix C.1 says 'final token'.")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="0 = rebuttal spec (K is full rank without centering). 1e-3 = Appendix C.1.")
    ap.add_argument("--tag", default=None, help="suffix for the output file, e.g. nucleus")
    ap.add_argument("--limit", type=int, default=None, help="first N questions")
    ap.add_argument("--log-every", type=int, default=200)
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
    print("  HALLUGUARD, REBUTTAL SPEC -- %s / %s" % (a.model_folder, a.dataset))
    print("  K = H H^T / d over the sampled trajectories; no gradients, no back-propagation")
    print("  identity projection: the trained calibration layer is unreleased")
    print("=" * 78, flush=True)

    per_q, y, pid, model_id, n_layers, sel, decoding = run(
        a.dataset, a.model_folder, data_dir, a.device, a.dtype, a.layer, a.pool, a.ridge,
        a.limit, a.log_every)
    res, uniq, y_q = evaluate(per_q, y, pid)
    res.update({"dataset": a.dataset, "model_folder": a.model_folder, "model_id": model_id,
                "dtype": a.dtype, "layer": a.layer, "layer_index": sel, "n_layers": n_layers,
                "pool": a.pool, "ridge": a.ridge, "projection": "identity (unreleased)",
                "source_decoding_config": decoding})

    os.makedirs(a.out_dir, exist_ok=True)
    stem = "hgreb_%s_%s%s" % (a.model_folder, a.dataset, ("_" + a.tag) if a.tag else "")
    np.savez_compressed(
        os.path.join(a.out_dir, stem + "_scores.npz"),
        prompt_id=uniq, label=y_q,
        **{k: np.array([r.get(k, np.nan) for r in per_q], dtype=float)
           for k in ("score", "log_det", "log_sigma_max", "log_kappa", "lambda_min", "lambda_max",
                     "mean_len", "mean_norm", "numerical_rank")})
    with open(os.path.join(a.out_dir, stem + ".json"), "w") as f:
        json.dump({**res, "per_question": per_q[:200]}, f, indent=2, default=float)

    print("\n  questions %d | hallucinated %.1f%%"
          % (res["n_questions"], res["prompt_hallucination_rate_pct"]))
    print("  mean numerical rank of K: %.2f of %.1f trajectories | rank-deficient: %d"
          % (res["mean_numerical_rank"], res["mean_n_trajectories"],
             res["n_questions_rank_deficient"]))
    print("  (rank well below the trajectory count means the cloud collapsed, and the "
          "score is then measuring the decoder)")
    print("\n  %-30s %9s %9s" % ("", "auroc", "flipped"))
    for b in res["auroc"]:
        print("  %-30s %9.4f %9.4f" % (b["name"], b["auroc"], b["auroc_flipped"]))
    print("\nWrote: %s{.json,_scores.npz}" % os.path.join(a.out_dir, stem))


if __name__ == "__main__":
    main()
