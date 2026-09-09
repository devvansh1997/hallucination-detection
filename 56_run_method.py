"""
56_run_method.py -- one runner, every detection method, identical evaluation.
=====================================================================================================
    python 56_run_method.py --method halluguard --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct

Changing --method is the ONLY thing that changes between rows of the comparison table. The data,
the train/test split, the metric and the output schema are owned by this file and are not reachable
from a method. That is the point: comparability becomes structural instead of a matter of everyone
being careful. Two published papers in this area disagree with their own documentation about exactly
these details; we are not adding a third.

WHAT THIS FILE OWNS
    - loading the pinned generations, and deriving is_known / question labels ONE way
    - the two split protocols, from their canonical definitions in 26 and 44
    - pooled and within-prompt AUROC, from the canonical definition in 53
    - the results schema

WHAT A METHOD OWNS
    - turning the data into one number per row. Nothing else. See methods/base.py.

THE TWO PROTOCOLS, because this is where the field goes wrong
    question-level   a question's ten answers stay together. What HARP's PAPER describes.
    answer-level     answers are split individually, so ~94% of known questions appear on BOTH
                     sides. What HARP's released CODE does. Worth 3.8-13.3 AUROC points.
    Both are reported for every method, always. A single number under an unnamed protocol is not a
    result.

GRANULARITY
    Beam-level methods score answers; question-level methods score questions. They are NOT
    comparable to each other and the output file says which it is in three places. Do not put a
    question-level AUROC in the same column as a beam-level one.

Usage:
  python 56_run_method.py --list
  python 56_run_method.py --self-test
  python 56_run_method.py --method halluguard --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct
  python 56_run_method.py --method halluguard --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct \
      --data-dir ../data-nucleus --tag nucleus --m-layer final --m-pool mean
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import methods as M   # noqa: E402


def load_bundle(dataset, model_folder, data_dir, device, dtype):
    import torch
    import yaml
    seq_path = os.path.join(data_dir, model_folder, "%s_sequences_v1.pt" % dataset)
    if not os.path.exists(seq_path):
        raise SystemExit(
            "%s not found.\nThe harness scores PINNED generations; it does not create them.\n"
            "Run 39_generate_dataset.py for this model/dataset first." % seq_path)
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    try:
        model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)
    except StopIteration:
        raise SystemExit("model_folder %r is not in config.yaml. Add it there rather than "
                         "hardcoding an id." % model_folder)
    seq = torch.load(seq_path, weights_only=False)
    return M.DataBundle(dataset, model_folder, model_id, data_dir, seq, device, dtype)


def evaluate(method, data, pre, seeds=None):
    """Score under both protocols and over all rows, with one metric definition throughout."""
    c = M.canonical()
    seeds = seeds or c["seeds"]
    n = len(data.labels)
    q_gran = method.granularity == "question"
    y_all = data.question_labels if q_gran else data.labels

    def measure(scores, rows):
        """rows: the beam indices this evaluation covers. For question granularity the scores are
        already per-question and aligned to sorted unique question ids within `rows`."""
        s = np.asarray(scores, dtype=float)
        if q_gran:
            qs = np.unique(data.prompt_id[rows])
            y = data.question_labels[np.searchsorted(data.question_ids, qs)]
            pid = qs
        else:
            y = data.labels[rows]
            pid = data.prompt_id[rows]
        f = np.isfinite(s)
        if f.sum() < 2 or len(set(y[f].tolist())) < 2:
            return None
        a = c["pooled_auroc"](s[f], y[f])
        w = c["within_prompt_auroc"](s[f], y[f], pid[f])
        return {"pooled_auroc": a,
                "pooled_auroc_flipped": None if a is None else 1.0 - a,
                # Within-prompt is undefined at question granularity -- one row per question means
                # no same-question pairs exist. Reported as None rather than a misleading 0.5.
                "within_prompt_auroc": None if q_gran else w["within_prompt_auroc"],
                "n_pairs": None if q_gran else w["n_pairs"],
                "n_scored": int(f.sum()), "n_non_finite": int((~f).sum())}

    # all_rows scores every row with train_idx == test_idx. For a training-free method that is a
    # convenient full-data pass. For a method that FITS in score() it is train-on-test, and the
    # number it produces is both inflated and expensive -- one extra full training run. Skipped,
    # with the reason recorded in its place so nobody reads the absence as an oversight.
    all_rows = np.arange(n)
    if getattr(method, "trains", False):
        out = {"all_rows": {"skipped": True,
                            "reason": ("%s fits on train_idx, so scoring with train_idx == "
                                       "test_idx would be train-on-test. Use the protocol arms."
                                       % method.name)}}
    else:
        out = {"all_rows": measure(method.score(data, pre, all_rows, all_rows), all_rows)}

    out["protocols"] = {}
    t_eval = time.time()
    for arm, fn in (("question", c["question_split"]), ("answer", c["answer_split"])):
        per_seed = []
        for seed in seeds:
            t_idx, v_idx = fn(data.is_known, data.prompt_id, n, seed)
            t_idx, v_idx = np.asarray(t_idx, dtype=int), np.asarray(v_idx, dtype=int)
            assert len(np.intersect1d(t_idx, v_idx)) == 0, \
                "%s split returned overlapping train/test ROWS at seed %s" % (arm, seed)
            t_split = time.time()
            r = measure(method.score(data, pre, t_idx, v_idx), v_idx)
            t_split = time.time() - t_split
            if r:
                r["seed"] = seed
                r["score_seconds"] = round(t_split, 1)
                r["n_train_rows"] = int(len(t_idx))
                r["n_test_rows"] = int(len(v_idx))
                r["n_test_questions"] = int(len(np.unique(data.prompt_id[v_idx])))
                # The leakage signature: under answer-level, a known question appears on both sides.
                r["known_questions_on_both_sides"] = int(
                    len(set(data.prompt_id[t_idx].tolist()) & set(data.prompt_id[v_idx].tolist())))
                per_seed.append(r)
        if per_seed:
            vals = [p["pooled_auroc"] for p in per_seed if p["pooled_auroc"] is not None]
            out["protocols"][arm] = {
                "pooled_auroc_mean": float(np.mean(vals)) if vals else None,
                "pooled_auroc_std": float(np.std(vals)) if vals else None,
                "per_seed": per_seed, "seeds": list(seeds),
            }
    out["evaluate_seconds"] = round(time.time() - t_eval, 1)
    return out


def self_test():
    print("=" * 74)
    print("  SELF-TEST: 56_run_method + methods registry")
    print("=" * 74)
    c = M.canonical()

    assert M.REGISTRY, "registry is empty"
    for name, gran, _ in M.describe():
        assert gran in ("beam", "question")
    print("  [PASS] registry: %s" % ", ".join(n for n, _, _ in M.describe()))

    for name in sorted(M.REGISTRY):
        print("  -- self-test of method %r" % name)
        M.get(name).self_test()

    # A synthetic bundle: 60 questions x 10 answers, with a planted per-question signal.
    n_q, n_b = 60, 10
    pid = np.repeat(np.arange(n_q), n_b)
    rng = np.random.default_rng(0)
    y = np.ones(n_q * n_b, dtype=int)
    for q in range(n_q // 2):                       # half the questions get some correct answers
        y[q * n_b + rng.choice(n_b, size=4, replace=False)] = 0

    class FakeSeq(dict):
        pass
    seq = {"input_ids": [None] * (n_q * n_b), "prompt_len": [0] * (n_q * n_b),
           "prompt_id": pid, "all_hallucination_flag": y, "decoding_config": {"num_beams": 1}}
    data = M.DataBundle("synthetic", "m", "id", ".", seq, "cpu", "float32")
    assert data.is_known.sum() == n_q // 2
    assert data.question_labels.sum() == n_q // 2
    print("  [PASS] DataBundle: is_known and question_labels agree (%d known of %d)"
          % (data.is_known.sum(), n_q))

    # A perfect beam-level oracle must score 1.0 under BOTH protocols. If it does not, the
    # plumbing -- indexing, alignment, label lookup -- is wrong, independent of any method.
    class Oracle(M.Method):
        name, granularity = "oracle", "beam"
        def score(self, d, pre, tr, te): return d.labels[te].astype(float)
    r = evaluate(Oracle(), data, None)
    assert abs(r["all_rows"]["pooled_auroc"] - 1.0) < 1e-12
    for arm in ("question", "answer"):
        assert abs(r["protocols"][arm]["pooled_auroc_mean"] - 1.0) < 1e-12, arm
    print("  [PASS] beam oracle scores 1.000 under both protocols -- indexing and alignment correct")

    # A TRAINED oracle, exercising the trains=True path. The skipped-all_rows branch was added
    # without a test that runs it, and the summary printer then crashed on the cluster with
    # KeyError: 'within_prompt_auroc' after a 40-minute run. Every branch gets a caller now.
    class TrainedOracle(Oracle):
        name, trains = "trained_oracle", True
    rt = evaluate(TrainedOracle(), data, None)
    assert rt["all_rows"].get("skipped") is True, rt["all_rows"]
    assert "pooled_auroc" not in rt["all_rows"]
    for arm in ("question", "answer"):
        assert abs(rt["protocols"][arm]["pooled_auroc_mean"] - 1.0) < 1e-12, arm
        assert all("score_seconds" in sd for sd in rt["protocols"][arm]["per_seed"])
    assert "evaluate_seconds" in rt
    print("  [PASS] trains=True skips all_rows, keeps both protocol arms, records timing")

    # An inverted oracle must score 0.0, not 1.0. Catches an accidental abs() or sign flip.
    class Inv(Oracle):
        name = "inv"
        def score(self, d, pre, tr, te): return -d.labels[te].astype(float)
    assert abs(evaluate(Inv(), data, None)["all_rows"]["pooled_auroc"] - 0.0) < 1e-12
    print("  [PASS] inverted oracle scores 0.000 -- orientation is preserved, not absorbed")

    # A question-level oracle, exercising the other alignment path.
    class QOracle(M.Method):
        name, granularity = "qoracle", "question"
        def score(self, d, pre, tr, te):
            qs = np.unique(d.prompt_id[te])
            return d.question_labels[np.searchsorted(d.question_ids, qs)].astype(float)
    rq = evaluate(QOracle(), data, None)
    assert abs(rq["all_rows"]["pooled_auroc"] - 1.0) < 1e-12
    assert rq["all_rows"]["within_prompt_auroc"] is None, \
        "within-prompt is undefined at question granularity and must be None, not a number"
    print("  [PASS] question oracle scores 1.000; within-prompt correctly reported as None")

    # The leakage signature must actually differ between arms, or the two protocols are not doing
    # what the whole comparison rests on.
    qb = r["protocols"]["question"]["per_seed"][0]["known_questions_on_both_sides"]
    ab = r["protocols"]["answer"]["per_seed"][0]["known_questions_on_both_sides"]
    assert qb == 0, "question-level split must keep a question wholly on one side, got %d" % qb
    assert ab >= 0.8 * data.is_known.sum(), \
        "answer-level split should put ~94%% of known questions on both sides, got %d of %d" % (
            ab, data.is_known.sum())
    print("  [PASS] protocols differ as designed: %d vs %d of %d known questions on both sides"
          % (qb, ab, data.is_known.sum()))

    print("\n[PASS] All self-test assertions passed.")


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--method", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    known, _ = ap.parse_known_args()

    if known.list:
        print("%-14s %-10s %s" % ("method", "granularity", "description"))
        for n, g, d in M.describe():
            print("%-14s %-10s %s" % (n, g, d))
        return
    if known.self_test:
        self_test(); return
    if not known.method:
        raise SystemExit("--method is required. See --list.")

    method = M.get(known.method)
    p = argparse.ArgumentParser(parents=[ap])
    p.add_argument("--dataset", required=True,
                   choices=["truthfulqa", "triviaqa", "nq_open", "tydiqa_gp"])
    p.add_argument("--model_folder", required=True)
    p.add_argument("--data-dir", default=None,
                   help="defaults to config.yaml output.data_dir. Point at ../data-nucleus to "
                        "score alternate-decoding generations.")
    p.add_argument("--out-dir", default=os.path.join(HERE, "results", "methods"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--tag", default=None, help="suffix for the output filename")
    method.add_args(p)
    a = p.parse_args()
    method.configure(a)

    data_dir = a.data_dir
    if not data_dir:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]

    print("=" * 78)
    print("  METHOD %s (%s-level) -- %s / %s" % (known.method, method.granularity,
                                                 a.model_folder, a.dataset))
    print("  data: %s" % os.path.abspath(data_dir))
    print("=" * 78, flush=True)

    data = load_bundle(a.dataset, a.model_folder, data_dir, a.device, a.dtype)
    print("  %r" % data)
    print("  decoding of this data: %s" % (data.decoding_config or "not recorded"), flush=True)

    t0 = time.time()
    pre = method.precompute(data)
    t_pre = time.time() - t0
    res = evaluate(method, data, pre)
    res.update({
        "method": known.method, "granularity": method.granularity,
        "dataset": a.dataset, "model_folder": a.model_folder, "model_id": data.model_id,
        "data_dir": os.path.abspath(data_dir), "dtype": a.dtype,
        "source_decoding_config": data.decoding_config,
        "n_beams": int(len(data.labels)), "n_questions": int(len(data.question_ids)),
        "beam_hallucination_rate_pct": round(100.0 * float(data.labels.mean()), 3),
        "question_hallucination_rate_pct": round(100.0 * float(data.question_labels.mean()), 3),
        "precompute_seconds": round(t_pre, 1), "method_meta": method.meta(),
        "total_seconds": round(time.time() - t0, 1),
    })

    os.makedirs(a.out_dir, exist_ok=True)
    stem = "%s_%s_%s%s" % (known.method, a.model_folder, a.dataset,
                           ("_" + a.tag) if a.tag else "")
    with open(os.path.join(a.out_dir, stem + ".json"), "w") as f:
        json.dump(res, f, indent=2, default=float)

    unit = "questions" if method.granularity == "question" else "answers"
    print("\n  %d %s | hallucinated %.1f%% | precompute %.0fs"
          % (res["n_questions"] if method.granularity == "question" else res["n_beams"], unit,
             res["question_hallucination_rate_pct"] if method.granularity == "question"
             else res["beam_hallucination_rate_pct"], t_pre))
    if res["method_meta"]:
        print("  method: %s" % json.dumps(res["method_meta"], default=float))
    print("\n  %-22s %9s %9s %14s" % ("", "pooled", "flipped", "within-prompt"))
    ar = res["all_rows"] or {}
    if ar.get("skipped"):
        print("  %-22s %s" % ("all rows", "skipped (%s)" % ar.get("reason", "")))
    elif ar:
        wp = "n/a" if ar["within_prompt_auroc"] is None else "%.4f" % ar["within_prompt_auroc"]
        print("  %-22s %9.4f %9.4f %14s" % ("all rows", ar["pooled_auroc"],
                                            ar["pooled_auroc_flipped"], wp))
    for arm in ("question", "answer"):
        b = res["protocols"].get(arm)
        if b:
            print("  %-22s %9.4f %9.4f   (+/- %.4f over %d seeds)"
                  % (arm + "-level split", b["pooled_auroc_mean"],
                     1.0 - b["pooled_auroc_mean"], b["pooled_auroc_std"], len(b["seeds"])))
    print("\n  precompute %.0fs | evaluate %.0fs | total %.0fs"
          % (t_pre, res.get("evaluate_seconds", 0.0), res["total_seconds"]))
    print("Wrote: %s" % os.path.join(a.out_dir, stem + ".json"))


if __name__ == "__main__":
    main()
