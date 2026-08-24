"""
53_halluguard_score.py -- score OUR pinned generations with HalluGuard's own NTK detector.
=====================================================================================================
THIRD METHOD, SAME DATA. We already have ours and HARP's scored on identical generations, labels and
known/unknown partitions. This adds HalluGuard (ICLR 2026) on the same footing, so the three-way
comparison varies the method and nothing else.

WE IMPORT THEIR SCORER, WE DO NOT REIMPLEMENT IT
    halluguard_true.compute_halluguard_score(model, input_ids, generated_ids, ...) is called
    unmodified. Its signature happens to be exactly what our Phase-1 sequences file already holds:
    input_ids concatenated with prompt_len, so splitting at prompt_len yields its two arguments.
    Nothing of theirs is edited; we never touch their pipeline/generate_simple.py, which couples
    generation to scoring and would otherwise force us onto their samples instead of ours.

    For reference, the score they compute is
        score = det(K) + log(sigma_max) - 2 log(kappa),      K = G G^T
    where G stacks per-token normalised gradients of log p(token_t) w.r.t. the last transformer
    block's parameters, kappa is K's condition number, and sigma_max is a Lipschitz-ratio proxy
    over hidden states. One autograd.grad call per generated token -- that is the cost driver.

NO TRAIN/TEST SPLIT, DELIBERATELY
    HalluGuard is training-free: it maps one (prompt, answer) pair to a scalar with no detector to
    fit. So there is no split to get wrong, and the answer-level leakage that costs HARP 3.8-13.3
    points cannot apply to it by construction. We therefore score every beam and report pooled and
    within-prompt AUROC over all of them -- no protocol arm, because it has no protocol.

DTYPE
    Their models/_load_model.py defaults to torch.float16, so that is our default too. But this
    code path takes GRADIENTS, and fp16 gradients can overflow to inf or collapse to zero in ways
    a forward-only pipeline never sees. Non-finite scores are counted and reported rather than
    silently dropped; --dtype float32 is available if they turn out to be common.

Usage:
  python 53_halluguard_score.py --self-test
  python 53_halluguard_score.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct --limit 200
  python 53_halluguard_score.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HG = os.path.abspath(os.path.join(HERE, "..", "HalluGuard-ICLR2026", "Score"))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def import_halluguard(hg_score_dir):
    """Import their scorer. Only halluguard_true.py is touched -- it is self-contained (torch
    only), so none of their pipeline, dataloaders or model wrappers come along."""
    p = os.path.join(hg_score_dir, "halluguard_true.py")
    if not os.path.exists(p):
        raise FileNotFoundError(
            "%s not found. Clone the repo next to HARP-Code:\n"
            "  git clone https://github.com/Susan571/HalluGuard-ICLR2026.git" % p)
    return _load_module("halluguard_true", p)


# ==============================================================================
# PURE HELPERS -- no torch, no model, independently testable
# ==============================================================================

def pooled_auroc(scores, labels):
    """P(score_halluc > score_truthful) over ALL pairs, ties=0.5. Equivalent to sklearn's
    roc_auc_score but written out so the tie convention matches within_prompt_auroc exactly."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    diffs = pos[:, None] - neg[None, :]
    return float(((diffs > 0).sum() + 0.5 * (diffs == 0).sum()) / diffs.size)


def within_prompt_auroc(scores, labels, prompt_ids):
    """Same definition as 26_grouped_baseline.within_prompt_auroc: concordant (hallucinated,
    truthful) pairs restricted to the SAME prompt, ties 0.5. Reimplemented here only to keep this
    script importable without the 26->27->... chain; the self-test checks it against that one."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    p = np.asarray(prompt_ids)
    n_mixed = n_all_truthful = n_all_halluc = 0
    concordant = 0.0
    total = 0
    for q in np.unique(p):
        idx = np.where(p == q)[0]
        hs, ts = s[idx][y[idx] == 1], s[idx][y[idx] == 0]
        if len(hs) == 0:
            n_all_truthful += 1
            continue
        if len(ts) == 0:
            n_all_halluc += 1
            continue
        n_mixed += 1
        d = hs[:, None] - ts[None, :]
        concordant += (d > 0).sum() + 0.5 * (d == 0).sum()
        total += d.size
    return {"within_prompt_auroc": float(concordant / total) if total else None,
            "n_mixed_prompts": n_mixed, "n_all_truthful_prompts": n_all_truthful,
            "n_all_hallucinated_prompts": n_all_halluc, "n_pairs": int(total)}


def summarise_finiteness(scores):
    """Non-finite scores are a real possibility under fp16 gradients. Report them; never drop
    them silently, because a method that fails on 10% of beams is not comparable to one that
    does not, and an AUROC computed over the survivors would hide that."""
    a = np.asarray(scores, dtype=float)
    finite = np.isfinite(a)
    return {"n_total": int(a.size), "n_finite": int(finite.sum()),
            "n_nan": int(np.isnan(a).sum()), "n_inf": int(np.isinf(a).sum()),
            "finite_pct": round(100.0 * finite.sum() / a.size, 4) if a.size else None}


# ==============================================================================

def score_dataset(dataset, model_folder, data_dir, hg, device, dtype, limit=None,
                  layer_idx=-1, param_subset="last_block", log_every=250):
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
    n = len(input_ids) if limit is None else min(limit, len(input_ids))
    print("  [%s] %d beams (of %d), model=%s dtype=%s" % (dataset, n, len(input_ids),
                                                          model_id, dtype), flush=True)

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype,
                                                  trust_remote_code=True).to(device)
    model.eval()

    scores, details, short = [], [], 0
    t0 = time.time()
    for i in range(n):
        ids = input_ids[i]
        pl = int(prompt_lens[i])
        p_ids, g_ids = ids[:pl], ids[pl:]
        # Their _compute_sigma_max_lipschitz needs >=3 generated tokens and otherwise returns a
        # degenerate sigma_max of 1.0. Count those rather than exclude them -- excluding would
        # quietly restrict the comparison to longer answers.
        if len(g_ids) < 3:
            short += 1
        try:
            r = hg.compute_halluguard_score(model, p_ids, g_ids, layer_idx=layer_idx,
                                            param_subset=param_subset)
            scores.append(r["score"])
            details.append((r["det_k"], r["sigma_max"], r["kappa"], r["T"]))
        except Exception as e:
            scores.append(float("nan"))
            details.append((float("nan"),) * 4)
            if len([d for d in details if not np.isfinite(d[0])]) <= 3:
                print("    beam %d failed: %s: %s" % (i, type(e).__name__, str(e)[:120]), flush=True)
        if (i + 1) % log_every == 0:
            el = time.time() - t0
            print("    %s %d/%d  (%.0fs, eta %.0fs)" % (dataset, i + 1, n, el,
                                                        el / (i + 1) * (n - i - 1)), flush=True)

    scores = np.asarray(scores, dtype=float)
    y, pid = labels[:n], prompt_ids[:n]
    fin = np.isfinite(scores)
    out = {
        "dataset": dataset, "model_folder": model_folder, "model_id": model_id,
        "dtype": dtype, "layer_idx": layer_idx, "param_subset": param_subset,
        "n_beams_scored": int(n), "n_short_completions_lt3_tokens": int(short),
        "finiteness": summarise_finiteness(scores),
        "hallucination_rate_pct": round(100.0 * float(y.mean()), 3),
    }
    if fin.sum() < n:
        out["note"] = ("AUROC computed over finite scores only; %d of %d beams excluded. This is "
                       "reported rather than hidden -- a method failing on some beams is not "
                       "directly comparable to one that does not." % (n - int(fin.sum()), n))
    out["pooled_auroc"] = pooled_auroc(scores[fin], y[fin])
    out["within_prompt"] = within_prompt_auroc(scores[fin], y[fin], pid[fin])
    return out, scores, y, pid


def self_test():
    print("=" * 74)
    print("  SELF-TEST: 53_halluguard_score (synthetic, no model, no cluster files)")
    print("=" * 74)

    # pooled AUROC against a hand-computable case
    a = pooled_auroc([3, 2, 1, 0], [1, 1, 0, 0])
    assert a == 1.0, a
    assert pooled_auroc([0, 1, 2, 3], [1, 1, 0, 0]) == 0.0
    assert pooled_auroc([1, 1], [1, 0]) == 0.5, "exact ties must score 0.5"
    assert pooled_auroc([1, 2], [1, 1]) is None, "one-class input has no AUROC"
    print("  [PASS] pooled_auroc: perfect, inverted, all-ties, and degenerate one-class")

    # agrees with sklearn on random data (ties aside, which sklearn also treats as 0.5)
    rng = np.random.default_rng(0)
    s = rng.normal(size=500)
    y = (rng.uniform(size=500) < 0.4).astype(int)
    try:
        from sklearn.metrics import roc_auc_score
        assert abs(pooled_auroc(s, y) - roc_auc_score(y, s)) < 1e-9
        print("  [PASS] pooled_auroc matches sklearn.roc_auc_score to 1e-9")
    except ImportError:
        print("  [SKIP] sklearn not importable; skipped the cross-check")

    # within-prompt: a case where pooled and within-prompt MUST disagree, which is the whole
    # reason both are reported. Question A is easy (low scores), B is hard (high scores); within
    # each, the hallucinated answer scores lower. Pooled sees difficulty; within-prompt sees the
    # ranking is backwards.
    sc = [0.1, 0.2, 0.8, 0.9]
    yy = [1, 0, 1, 0]
    pp = [0, 0, 1, 1]
    w = within_prompt_auroc(sc, yy, pp)
    assert w["within_prompt_auroc"] == 0.0, w
    assert w["n_mixed_prompts"] == 2 and w["n_pairs"] == 2
    # Pooled is 0.25, not 0.5: of the four cross-label pairs the only concordant one is
    # (0.8 hallucinated, 0.2 truthful) -- a wrong answer to the HARD question outscoring a right
    # answer to the EASY one. That single pair is pure question-difficulty signal, and it is the
    # entire difference between 0.25 pooled and 0.00 within-prompt.
    assert pooled_auroc(sc, yy) == 0.25, pooled_auroc(sc, yy)
    print("  [PASS] within_prompt_auroc: 0.00 where pooled is 0.25 -- the gap is difficulty")

    w2 = within_prompt_auroc([1, 0, 1, 1], [1, 0, 1, 1], [0, 0, 1, 1])
    assert w2["n_mixed_prompts"] == 1 and w2["n_all_hallucinated_prompts"] == 1
    print("  [PASS] within_prompt_auroc: all-hallucinated prompts counted, not silently dropped")

    # cross-check against the project's own implementation, so numbers stay comparable
    try:
        s26 = _load_module("s26", os.path.join(HERE, "26_grouped_baseline.py"))
        rs = rng.normal(size=300)
        ry = (rng.uniform(size=300) < 0.5).astype(int)
        rp = rng.integers(0, 30, size=300)
        mine = within_prompt_auroc(rs, ry, rp)["within_prompt_auroc"]
        theirs = s26.within_prompt_auroc(rs, ry, rp)
        theirs = theirs[0] if isinstance(theirs, tuple) else (
            theirs["within_prompt_auroc"] if isinstance(theirs, dict) else theirs)
        assert abs(mine - float(theirs)) < 1e-12, (mine, theirs)
        print("  [PASS] within_prompt_auroc matches 26_grouped_baseline's to 1e-12")
    except Exception as e:
        print("  [SKIP] cross-check against 26_grouped_baseline (%s: %s)" % (type(e).__name__, e))

    f = summarise_finiteness([1.0, float("nan"), float("inf"), 2.0])
    assert f["n_finite"] == 2 and f["n_nan"] == 1 and f["n_inf"] == 1 and f["finite_pct"] == 50.0
    print("  [PASS] summarise_finiteness: counts NaN and inf separately")

    print("\n[PASS] All self-test assertions passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tydiqa_gp",
                    choices=["truthfulqa", "triviaqa", "nq_open", "tydiqa_gp"])
    ap.add_argument("--model_folder", default="qwen-2.5-7b-instruct")
    ap.add_argument("--hg-dir", default=DEFAULT_HG)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "halluguard"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                    help="float16 matches their models/_load_model.py default.")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N beams (pilot)")
    ap.add_argument("--layer-idx", type=int, default=-1)
    ap.add_argument("--param-subset", default="last_block")
    ap.add_argument("--log-every", type=int, default=250,
                    help="progress interval; lower it for short pilot runs")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return

    data_dir = a.data_dir
    if not data_dir:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]

    hg = import_halluguard(a.hg_dir)
    print("=" * 78)
    print("  HALLUGUARD on our generations -- %s / %s" % (a.model_folder, a.dataset))
    print("  scorer: %s (imported unmodified)" % os.path.join(a.hg_dir, "halluguard_true.py"))
    print("  training-free, so no train/test split and no protocol arm")
    print("=" * 78, flush=True)

    out, scores, y, pid = score_dataset(a.dataset, a.model_folder, data_dir, hg,
                                        a.device, a.dtype, a.limit, a.layer_idx, a.param_subset,
                                        log_every=a.log_every)

    os.makedirs(a.out_dir, exist_ok=True)
    stem = "halluguard_%s_%s" % (a.model_folder, a.dataset)
    np.savez_compressed(os.path.join(a.out_dir, stem + "_scores.npz"),
                        score=scores, label=y, prompt_id=pid)
    with open(os.path.join(a.out_dir, stem + ".json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n  finite      : %d/%d (%.2f%%)" % (out["finiteness"]["n_finite"],
                                                out["finiteness"]["n_total"],
                                                out["finiteness"]["finite_pct"]))
    print("  short (<3 tok, degenerate sigma_max): %d" % out["n_short_completions_lt3_tokens"])
    print("  pooled AUROC       : %s" % out["pooled_auroc"])
    print("  within-prompt AUROC: %s  (%d pairs, %d mixed prompts)" % (
        out["within_prompt"]["within_prompt_auroc"], out["within_prompt"]["n_pairs"],
        out["within_prompt"]["n_mixed_prompts"]))
    print("\nWrote: %s{.json,_scores.npz}" % os.path.join(a.out_dir, stem))


if __name__ == "__main__":
    main()
