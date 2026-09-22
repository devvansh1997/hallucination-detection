"""
72_empty_answers.py -- how many answers are empty, and how much would "the answer is empty" alone score? (T-014)
==================================================================================================

An answer is empty when the first generated token is a stop token: its decoded text is "" and its
window holds only that token (the canonical window is content through the first stop token,
inclusive). Empty answers are labelled hallucinated by the judge, so they are trivially separable.
Falcon-H1 on TyDiQA-GP has no question with all ten answers correct, which is what one empty beam per
question would produce. This measures, per model and dataset, from the pinned sequences files:

  empty          number and share of empty answers
  of_halluc      share of hallucinated answers that are empty
  empty_correct  empty answers labelled correct (expected 0)
  q_with_empty   questions with at least one empty answer
  auroc_empty    pooled AUROC of the score 1{answer is empty} against the labels, over all answers --
                 what a detector gets from emptiness alone (0.5 = nothing)

Reads ../data/<model>/<dataset>_sequences_v1.pt (keys decoded_text, all_hallucination_flag, prompt_id);
writes results/empty_answers.json. Missing files are reported and skipped.

  python 72_empty_answers.py
  python 72_empty_answers.py --models falcon-h1-7b-base --datasets tydiqa_gp
  python 72_empty_answers.py --self-test
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["qwen-2.5-7b-instruct", "llama-3.1-8b", "falcon-h1-7b-base"]
DATASETS = ["truthfulqa", "tydiqa_gp", "nq_open", "triviaqa"]


def summarise(texts, labels, prompt_ids):
    """texts: decoded answers; labels: 1 = hallucinated; prompt_ids: question of each answer."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray([int(v) for v in labels])
    e = np.asarray([int(not str(t).strip()) for t in texts])
    q = np.asarray([int(p) for p in prompt_ids])
    per_q = Counter(q[e == 1].tolist())
    n_h = int(y.sum())
    auroc = float(roc_auc_score(y, e)) if 0 < n_h < len(y) else None
    return {"answers": int(len(y)), "empty": int(e.sum()), "empty_pct": round(100 * e.mean(), 2),
            "of_halluc_pct": round(100 * float((e & y).sum()) / max(n_h, 1), 2),
            "empty_correct": int((e & (1 - y)).sum()),
            "q_with_empty": len(per_q), "questions": int(len(np.unique(q))),
            "empties_per_question": dict(sorted(Counter(per_q.values()).items())),
            "auroc_empty": None if auroc is None else round(100 * auroc, 2)}


def run(models, datasets, out_path):
    import torch
    import yaml
    with open(os.path.join(HERE, "config.yaml")) as f:
        data_dir = os.path.join(HERE, yaml.safe_load(f)["output"]["data_dir"])
    out = {}
    for m in models:
        for ds in datasets:
            p = os.path.join(data_dir, m, "%s_sequences_v1.pt" % ds)
            if not os.path.exists(p):
                print("  %-22s %-11s no sequences file (%s)" % (m, ds, p), flush=True)
                continue
            d = torch.load(p, weights_only=False)
            r = summarise(d["decoded_text"], d["all_hallucination_flag"], d["prompt_id"])
            out["%s/%s" % (m, ds)] = r
            print("  %-22s %-11s answers %6d  empty %5d (%5.2f%%)  = %5.2f%% of hallucinated  "
                  "empty+correct %d  questions with an empty answer %d/%d  AUROC(is empty) %s"
                  % (m, ds, r["answers"], r["empty"], r["empty_pct"], r["of_halluc_pct"], r["empty_correct"],
                     r["q_with_empty"], r["questions"], r["auroc_empty"]), flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\n  wrote %s" % out_path)


def self_test():
    # 4 questions x 3 answers; question 0 and 1 have one empty (hallucinated) answer each
    texts = ["", "a", "b", " ", "c", "d", "e", "f", "g", "h", "i", "j"]
    labels = [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    pids = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    r = summarise(texts, labels, pids)
    assert r["empty"] == 2 and r["empty_correct"] == 0 and r["q_with_empty"] == 2, r
    assert r["of_halluc_pct"] == round(100 * 2 / 7, 2), r
    # 1{empty} is 1 on 2 of 7 hallucinated and 0 on all 5 correct: AUROC = 0.5 + 0.5 * 2/7
    assert abs(r["auroc_empty"] - round(100 * (0.5 + 0.5 * 2 / 7), 2)) < 1e-9, r
    assert r["empties_per_question"] == {1: 2}, r
    print("  [PASS] summarise: counts, share of hallucinated, per-question counts, AUROC(is empty) = "
          "0.5 + 0.5 x share when no correct answer is empty")
    print("\n  ALL PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "empty_answers.json"))
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    run(a.models, a.datasets, a.out)


if __name__ == "__main__":
    main()
