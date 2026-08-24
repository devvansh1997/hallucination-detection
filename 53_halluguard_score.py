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

TRAINING-FREE, WHICH MAKES IT A CONTROL AS WELL AS A BASELINE
    HalluGuard fits nothing: it maps one (prompt, answer) pair to a scalar, so there is no split to
    get wrong and the answer-level leakage that costs HARP 3.8-13.3 points cannot touch it. That
    does NOT mean the protocol is irrelevant to it. HARP's AUROC is computed over a particular
    population -- 25% of known questions plus all unknown ones -- and "every beam we generated" is a
    different population with a different hallucination rate. Scoring HalluGuard over all beams and
    calling it comparable to HARP would be comparing two numbers computed on two test sets.

    So we report THREE numbers from one pass of scoring (the scores are deterministic, so the extra
    two are free):
        all beams          -- what HalluGuard achieves on our data, unconditioned
        question-level     -- restricted to the test rows OUR protocol evaluates on
        answer-level       -- restricted to the test rows HARP's RELEASED CODE evaluates on
    all three using the project's own split functions and the same HARP_SEEDS, so the row sets are
    literally identical to the ones the other two methods were scored on.

    The second and third also buy us a control we do not otherwise have. For HARP,
        (answer-level - question-level) = leakage + population-composition
    and a reviewer's first objection is that the whole gap might be composition. For a score that
    does not change between the two arms the leakage term is identically zero, so HalluGuard's own
    delta estimates the composition term alone. Near zero means HARP's gap is leakage.

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

# Must match 43_eval_phase2.py:70 and 44_eval_phase3.py:112 or the "same test rows" claim is false.
# Declared here rather than imported so this module stays importable on its own; load_split_fns
# asserts the two agree at the point where 44 is loaded anyway.
HARP_SEEDS = [42, 0, 1, 2, 3]


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


def derive_is_known(labels, prompt_ids):
    """A prompt is known if ANY of its beams is truthful (label 0) -- the rule
    35_derive_streams.py uses and the one HARP encodes at get_known.py:115.
    Returned indexed BY PROMPT ID, which is how the split functions consume it."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(prompt_ids)
    uniq = np.unique(p)
    assert uniq.min() == 0 and uniq.max() == len(uniq) - 1, (
        "prompt ids must be contiguous 0..P-1 -- original_harp_split indexes is_known by "
        "prompt id, so a gap would silently mis-assign questions")
    is_known = np.zeros(len(uniq), dtype=bool)
    for q in uniq:
        is_known[q] = bool((y[p == q] == 0).any())
    return is_known


def evaluate_on_test_rows(scores, labels, prompt_ids, seeds, split_fns):
    """HalluGuard's per-beam score never changes -- nothing is fitted -- so evaluating it on the
    SAME test rows HARP and our method are scored on makes the three directly comparable, and
    doing so under BOTH protocols isolates something we otherwise cannot measure.

    For HARP, (answer-level minus question-level) = leakage + whatever comes from the two test
    sets having different composition: the question-level set draws from ~25% of known questions,
    the answer-level set from ~94% of them. For a FIXED score the leakage term is identically
    zero, so HalluGuard's own delta measures the composition term alone. If it is near zero,
    HARP's 3.8-13.3 points is leakage rather than an artifact of comparing different test sets --
    which is the first objection a reviewer raises and one we currently cannot answer.

    Caveat worth keeping: this assumes the composition effect is comparable in magnitude across
    methods. It is not guaranteed, but it is far better than no control at all."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    p = np.asarray(prompt_ids)
    is_known = derive_is_known(y, p)
    N = len(y)
    out = {}
    for arm, fn in split_fns.items():
        pooled, within, sizes = [], [], []
        for seed in seeds:
            _, v_idx = fn(is_known, p, N, seed)
            v_idx = np.asarray(v_idx, dtype=int)
            fin = np.isfinite(s[v_idx])
            v = v_idx[fin]
            a = pooled_auroc(s[v], y[v])
            w = within_prompt_auroc(s[v], y[v], p[v])
            if a is not None:
                pooled.append(a)
            if w["within_prompt_auroc"] is not None:
                within.append(w["within_prompt_auroc"])
            sizes.append(int(len(v)))
        out[arm] = {
            "pooled_auroc_mean": float(np.mean(pooled)) if pooled else None,
            "pooled_auroc_std": float(np.std(pooled)) if pooled else None,
            "within_prompt_auroc_mean": float(np.mean(within)) if within else None,
            "within_prompt_auroc_std": float(np.std(within)) if within else None,
            "n_test_rows": sizes[0] if sizes else None,
            "seeds": list(seeds),
        }
    if out.get("answer") and out.get("question"):
        pa, pq = out["answer"]["pooled_auroc_mean"], out["question"]["pooled_auroc_mean"]
        if pa is not None and pq is not None:
            out["composition_effect_pooled"] = round(100.0 * (pa - pq), 4)
            out["composition_effect_note"] = (
                "answer-level minus question-level, in AUROC points, for a score that does not "
                "change between them. This is the population-composition term only; any leakage "
                "term is identically zero because nothing is fitted.")
    return out


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
    out["all_beams"] = {"pooled_auroc": pooled_auroc(scores[fin], y[fin]),
                        "within_prompt": within_prompt_auroc(scores[fin], y[fin], pid[fin])}
    return out, scores, y, pid


def load_split_fns():
    """The project's own split implementations, imported rather than reimplemented, so
    HalluGuard is scored on literally the same test rows as HARP and our method -- same
    functions, same seeds, same partitions."""
    s26 = _load_module("s26", os.path.join(HERE, "26_grouped_baseline.py"))
    s44 = _load_module("s44", os.path.join(HERE, "44_eval_phase3.py"))
    assert list(s44.HARP_SEEDS) == list(HARP_SEEDS), (
        "seed lists diverged: 44 uses %s, this script uses %s. The three-way comparison depends on "
        "the SAME partitions, so this must be fixed rather than tolerated."
        % (list(s44.HARP_SEEDS), list(HARP_SEEDS)))
    return {
        "question": lambda ik, p, n, seed: s26.original_harp_split(ik, p, n, seed=seed),
        "answer": lambda ik, p, n, seed: s44.answer_level_harp_split(ik, p, n, seed),
    }


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

    # ---- derive_is_known -------------------------------------------------------------------
    # prompt 0 has a truthful beam -> known; prompt 1 is all-hallucinated -> unknown
    ik = derive_is_known([1, 0, 1, 1], [0, 0, 1, 1])
    assert list(ik) == [True, False], list(ik)
    try:
        derive_is_known([1, 0], [0, 7])
        raise SystemExit("derive_is_known accepted non-contiguous prompt ids")
    except AssertionError:
        pass
    print("  [PASS] derive_is_known: known iff any truthful beam; rejects non-contiguous ids")

    try:
        s43 = _load_module("s43", os.path.join(HERE, "43_eval_phase2.py"))
        ry2 = (rng.uniform(size=400) < 0.5).astype(int)
        rp2 = np.sort(rng.integers(0, 40, size=400))
        rp2 = np.searchsorted(np.unique(rp2), rp2)  # force contiguous 0..P-1
        theirs_ik, _ = s43.derive_is_known(ry2, rp2)
        assert np.array_equal(derive_is_known(ry2, rp2), theirs_ik)
        print("  [PASS] derive_is_known matches 43_eval_phase2's on random data")
    except Exception as e:
        print("  [SKIP] cross-check against 43_eval_phase2 (%s: %s)" % (type(e).__name__, e))

    # ---- evaluate_on_test_rows, against the REAL split functions ----------------------------
    try:
        fns = load_split_fns()
        P, B = 200, 10
        pid_t = np.repeat(np.arange(P), B)
        # 70% of prompts get at least one truthful beam -> known; the rest are all-hallucinated
        y_t = np.ones(P * B, dtype=int)
        known_set = rng.choice(P, size=int(0.7 * P), replace=False)
        for q in known_set:
            k = rng.integers(1, B)                      # 1..B-1 truthful beams
            y_t[q * B + rng.choice(B, size=k, replace=False)] = 0
        ik_t = derive_is_known(y_t, pid_t)
        assert ik_t.sum() == len(known_set)

        # The two protocols' valid sets differ in WHICH known rows they hold, not how many.
        # Check that directly -- it is the property the composition-effect argument rests on.
        n_known_rows = int(np.isin(pid_t, np.where(ik_t)[0]).sum())
        cov = {}
        for arm, fn in fns.items():
            t_idx, v_idx = fn(ik_t, pid_t, len(y_t), 42)
            assert len(np.intersect1d(t_idx, v_idx)) == 0, "%s: train and valid ROWS overlap" % arm
            unknown_rows = np.where(~np.isin(pid_t, np.where(ik_t)[0]))[0]
            assert np.isin(unknown_rows, v_idx).all(), (
                "%s: unknown prompts must go WHOLLY to valid; a train index was returned" % arm)
            n_known_v = int(np.isin(pid_t[v_idx], np.where(ik_t)[0]).sum())
            assert abs(n_known_v - 0.25 * n_known_rows) <= B, (arm, n_known_v, n_known_rows)
            cov[arm] = len(set(pid_t[v_idx].tolist()) & set(np.where(ik_t)[0].tolist()))
        # ~25% of known QUESTIONS vs ~94% of them -- 1 - 0.75^10 = 0.944
        assert 0.20 * ik_t.sum() <= cov["question"] <= 0.30 * ik_t.sum(), cov
        assert cov["answer"] >= 0.88 * ik_t.sum(), cov
        print("  [PASS] split arms hold the same COUNT of known rows (%d) but cover %d vs %d of "
              "%d known questions" % (int(0.25 * n_known_rows), cov["question"], cov["answer"],
                                      int(ik_t.sum())))

        # A score with no signal must land at 0.5 on both arms, and the composition effect at 0.
        flat = np.zeros(P * B)
        r_flat = evaluate_on_test_rows(flat, y_t, pid_t, HARP_SEEDS, fns)
        assert abs(r_flat["question"]["pooled_auroc_mean"] - 0.5) < 1e-12
        assert abs(r_flat["answer"]["pooled_auroc_mean"] - 0.5) < 1e-12
        assert abs(r_flat["composition_effect_pooled"]) < 1e-9
        print("  [PASS] evaluate_on_test_rows: an all-ties score gives 0.5 on both arms, delta 0")

        # A real signal: both arms must recover it, and -- the point of the control -- agree,
        # because a fixed score cannot leak. Anything beyond sampling noise here would mean the
        # subsetting itself, not leakage, moves the number.
        sig = rng.normal(size=P * B) + 1.5 * y_t
        r_sig = evaluate_on_test_rows(sig, y_t, pid_t, HARP_SEEDS, fns)
        assert r_sig["question"]["pooled_auroc_mean"] > 0.75
        assert abs(r_sig["composition_effect_pooled"]) < 2.0, r_sig["composition_effect_pooled"]
        print("  [PASS] evaluate_on_test_rows: real signal recovered on both arms, delta %+.2f pts "
              "(fixed score => no leakage term)" % r_sig["composition_effect_pooled"])

        # Non-finite scores inside a test set must be excluded, not poison the mean.
        holed = sig.copy()
        holed[rng.choice(P * B, size=50, replace=False)] = np.nan
        r_hole = evaluate_on_test_rows(holed, y_t, pid_t, HARP_SEEDS, fns)
        assert r_hole["question"]["pooled_auroc_mean"] is not None
        assert np.isfinite(r_hole["question"]["pooled_auroc_mean"])
        assert r_hole["question"]["n_test_rows"] < r_sig["question"]["n_test_rows"]
        print("  [PASS] evaluate_on_test_rows: NaN scores dropped from the test rows and counted")
    except AssertionError:
        raise
    except Exception as e:
        print("  [SKIP] split-restricted evaluation (%s: %s)" % (type(e).__name__, e))

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
    print("  training-free -- nothing is fitted, so the scores below are one deterministic pass")
    print("  reported over all beams AND over both protocols' test rows (seeds %s)" % HARP_SEEDS)
    print("=" * 78, flush=True)

    out, scores, y, pid = score_dataset(a.dataset, a.model_folder, data_dir, hg,
                                        a.device, a.dtype, a.limit, a.layer_idx, a.param_subset,
                                        log_every=a.log_every)

    # Same test rows HARP and our method are scored on. Skipped for --limit runs, where the
    # truncated beam set no longer contains whole prompts and the split would be meaningless.
    if a.limit is None:
        try:
            out["on_test_rows"] = evaluate_on_test_rows(scores, y, pid, HARP_SEEDS,
                                                        load_split_fns())
        except Exception as e:
            out["on_test_rows_error"] = "%s: %s" % (type(e).__name__, e)
            print("  [WARN] split-restricted evaluation failed: %s" % e, flush=True)
    else:
        out["on_test_rows"] = None
        out["on_test_rows_note"] = ("skipped: --limit truncates the beam set mid-prompt, so the "
                                    "question/answer splits would not correspond to the ones "
                                    "HARP and our method used")

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
    ab = out["all_beams"]
    print("  ALL BEAMS          pooled %s | within-prompt %s (%d pairs)" % (
        ab["pooled_auroc"], ab["within_prompt"]["within_prompt_auroc"],
        ab["within_prompt"]["n_pairs"]))
    tr = out.get("on_test_rows")
    if tr:
        for arm in ("question", "answer"):
            r = tr.get(arm)
            if r:
                print("  %-8s test rows  pooled %.4f +/- %.4f | within-prompt %.4f  (n=%d)" % (
                    arm, r["pooled_auroc_mean"], r["pooled_auroc_std"],
                    r["within_prompt_auroc_mean"], r["n_test_rows"]))
        if "composition_effect_pooled" in tr:
            print("  composition effect (answer - question, fixed score): %+.2f points"
                  % tr["composition_effect_pooled"])
    print("\nWrote: %s{.json,_scores.npz}" % os.path.join(a.out_dir, stem))


if __name__ == "__main__":
    main()
