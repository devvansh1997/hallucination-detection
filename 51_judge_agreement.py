"""
51_judge_agreement.py -- do OUR correctness labels match what HARP's own judge would produce?
=====================================================================================================
WHY THIS IS THE LOAD-BEARING AUDIT
    Our known/unknown partition comes from OUR judge. We matched their thresholds exactly
    (sen_sim 0.5, rouge 0.7, correct_advantage -1, config.yaml judge block), but nobody has run
    THEIR DatasetJudge on our generations and compared. That distinction decides which claim we
    are entitled to make:

      internal   "changing only the split unit moves AUROC by 3.8-13.3 points"
                 -- holds regardless: both arms use identical labels, whatever those labels are.
      external   "HARP's honest TyDiQA number is 79.9, not the published 88.4"
                 -- needs our partition to be the one their paper was computed over.

    If the labels disagree materially, the second claim weakens to "on our labels".

NO REGENERATION NEEDED
    49_harp_adapter.py already wrote gen_text for every beam, plus correct_answers /
    incorrect_answers, into harp_input/{model}/[known|unknown]{ds}.jsonl -- that is exactly the
    input DatasetJudge.judge() takes. This script re-judges those strings. It never loads the
    LLM, never generates, and never writes into HARP-Code.

THEIR JUDGE, THEIR PARAMETERS
    We import DatasetJudge and construct it exactly as main.py:133-137 does:
        DatasetJudge(bleurt_model_path=..., sen_sim_threshold=0.5, rouge_threshold=0.7)
    correct_advantage is deliberately NOT passed, so it keeps their default of -1
    (DatasetJudge.py:26) -- the value main.py actually runs with. Note their get_known.py:136
    entry point passes 0.05 instead; main.py is the one that produced the published numbers.

DEVICE
    Their judge never moves BLEURT off the CPU, so --device cpu is the faithful setting and the
    default. GPU is far faster but fp arithmetic can differ, and a score that shifts by 1e-4
    across the 0.5 threshold flips a label -- which is the very quantity being measured. Rather
    than assume it does not matter, --check-device N judges the same N beams on both and reports
    how many labels actually flip.

Usage:
  python 51_judge_agreement.py --self-test
  python 51_judge_agreement.py --dataset tydiqa --limit-prompts 50        # pilot
  python 51_judge_agreement.py --dataset tydiqa --check-device 200        # cpu-vs-gpu flips
  python 51_judge_agreement.py --dataset tydiqa
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HARP = os.path.abspath(os.path.join(HERE, "..", "HARP-Code"))
BLEURT_ID = "lucadiliello/BLEURT-20"          # MODEL_PATH["BLEURT-20"], main.py:21
SEN_SIM_THRESHOLD = 0.5                        # main.py:135
ROUGE_THRESHOLD = 0.7                          # main.py:136
HARP_DATASETS = ["tydiqa", "truthful_qa", "nq_open", "trivia_qa"]


# ==============================================================================
# PURE HELPERS -- no model, no files, independently testable
# ==============================================================================

def confusion(ours, theirs):
    """Beam-level agreement. Both are lists of bool is_correct."""
    o = np.asarray(ours, dtype=bool)
    t = np.asarray(theirs, dtype=bool)
    both = int(np.sum(o & t))
    neither = int(np.sum(~o & ~t))
    ours_only = int(np.sum(o & ~t))
    theirs_only = int(np.sum(~o & t))
    n = len(o)
    return {"n_beams": n, "both_correct": both, "both_incorrect": neither,
            "ours_correct_theirs_not": ours_only, "theirs_correct_ours_not": theirs_only,
            "agreement_pct": round(100.0 * (both + neither) / n, 4) if n else None}


def known_from_labels(prompt_ids, is_correct):
    """HARP's rule: a question is known if ANY of its beams is correct (get_known.py:115)."""
    known = {}
    for p, c in zip(prompt_ids, is_correct):
        known[p] = known.get(p, False) or bool(c)
    return known


def partition_delta(prompt_ids, ours, theirs):
    """How the known/unknown split itself moves -- the quantity the external claim rests on."""
    ko = known_from_labels(prompt_ids, ours)
    kt = known_from_labels(prompt_ids, theirs)
    keys = sorted(ko)
    agree = sum(1 for k in keys if ko[k] == kt[k])
    ours_known = sum(1 for k in keys if ko[k])
    theirs_known = sum(1 for k in keys if kt[k])
    return {"n_prompts": len(keys),
            "ours_known": ours_known, "theirs_known": theirs_known,
            "prompts_with_same_status": agree,
            "prompts_that_flip": len(keys) - agree,
            "status_agreement_pct": round(100.0 * agree / len(keys), 4) if keys else None,
            "known_only_ours": sum(1 for k in keys if ko[k] and not kt[k]),
            "known_only_theirs": sum(1 for k in keys if kt[k] and not ko[k])}


def load_jsonl_beams(path):
    """Flatten one of the adapter's jsonl files into per-beam records."""
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            for b in item["result"]:
                recs.append({"prompt_id": int(item["ind"]), "gen_text": b["gen_text"],
                             "ours_is_correct": bool(b["score"]["is_correct"]),
                             "correct_answers": item["correct_answers"],
                             "incorrect_answers": item["incorrect_answers"]})
    return recs


# ==============================================================================

def build_judge(harp_dir, device):
    if harp_dir not in sys.path:
        sys.path.insert(0, harp_dir)
    from DatasetJudge import DatasetJudge          # THEIR class, unmodified
    judge = DatasetJudge(bleurt_model_path=BLEURT_ID,
                         sen_sim_threshold=SEN_SIM_THRESHOLD,
                         rouge_threshold=ROUGE_THRESHOLD)
    # correct_advantage intentionally left at their default of -1 (main.py never passes it)
    if device != "cpu":
        judge.bleurt_model = judge.bleurt_model.to(device)
    return judge


def judge_beams(judge, recs, device, label=""):
    out = []
    t0 = time.time()
    for i, r in enumerate(recs):
        if device != "cpu":
            import torch
            _orig = judge.bleurt_tokenizer

            def _tok(*a, **k):
                enc = _orig(*a, **k)
                return {kk: vv.to(device) for kk, vv in enc.items()}
            judge.bleurt_tokenizer = _tok
            try:
                res = judge.judge(generated_text=r["gen_text"],
                                  correct_answer_list=r["correct_answers"],
                                  incorrect_answer_list=r["incorrect_answers"])
            finally:
                judge.bleurt_tokenizer = _orig
        else:
            res = judge.judge(generated_text=r["gen_text"],
                              correct_answer_list=r["correct_answers"],
                              incorrect_answer_list=r["incorrect_answers"])
        out.append(bool(res.is_correct))
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print("    %s judged %d/%d  (%.1fs, eta %.0fs)" % (
                label, i + 1, len(recs), el, el / (i + 1) * (len(recs) - i - 1)), flush=True)
    return out


def run(dataset, model_folder, harp_dir, in_root, out_dir, device,
        limit_prompts=None, check_device=0):
    recs = []
    for side in ("known", "unknown"):
        p = os.path.join(in_root, model_folder, "[%s]%s.jsonl" % (side, dataset))
        if not os.path.exists(p):
            raise FileNotFoundError("%s not found -- run 49_harp_adapter.py first." % p)
        recs.extend(load_jsonl_beams(p))
    if limit_prompts:
        keep = sorted({r["prompt_id"] for r in recs})[:limit_prompts]
        keep = set(keep)
        recs = [r for r in recs if r["prompt_id"] in keep]
    print("  [%s] %d beams / %d prompts  device=%s" % (
        dataset, len(recs), len({r["prompt_id"] for r in recs}), device), flush=True)

    judge = build_judge(harp_dir, device)
    theirs = judge_beams(judge, recs, device, label=dataset)
    ours = [r["ours_is_correct"] for r in recs]
    pids = [r["prompt_id"] for r in recs]

    result = {"dataset": dataset, "model_folder": model_folder, "device": device,
              "n_prompts_limited_to": limit_prompts,
              "judge_params": {"sen_sim_threshold": SEN_SIM_THRESHOLD,
                               "rouge_threshold": ROUGE_THRESHOLD,
                               "correct_advantage": judge.correct_advantage,
                               "bleurt": BLEURT_ID},
              "beam_level": confusion(ours, theirs),
              "prompt_level": partition_delta(pids, ours, theirs)}

    if check_device and device != "cpu":
        sub = recs[:check_device]
        print("  [%s] device check: re-judging %d beams on cpu" % (dataset, len(sub)), flush=True)
        cpu_judge = build_judge(harp_dir, "cpu")
        cpu_lab = judge_beams(cpu_judge, sub, "cpu", label=dataset + "/cpu")
        flips = int(np.sum(np.asarray(cpu_lab) != np.asarray(theirs[:len(sub)])))
        result["device_check"] = {"n": len(sub), "label_flips_cpu_vs_%s" % device: flips}
        print("  [%s] device check: %d/%d labels differ" % (dataset, flips, len(sub)), flush=True)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "judge_agreement_%s_%s.json" % (model_folder, dataset))
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    b, q = result["beam_level"], result["prompt_level"]
    print("  [%s] BEAM  agreement %.2f%%  (ours-only %d, theirs-only %d)" % (
        dataset, b["agreement_pct"], b["ours_correct_theirs_not"], b["theirs_correct_ours_not"]))
    print("  [%s] KNOWN agreement %.2f%%  ours_known=%d theirs_known=%d  flips=%d" % (
        dataset, q["status_agreement_pct"], q["ours_known"], q["theirs_known"],
        q["prompts_that_flip"]))
    print("  wrote %s" % out, flush=True)
    return result


def self_test():
    print("=" * 74)
    print("  SELF-TEST: 51_judge_agreement (synthetic, no BLEURT, no files)")
    print("=" * 74)

    c = confusion([True, True, False, False], [True, False, True, False])
    assert c == {"n_beams": 4, "both_correct": 1, "both_incorrect": 1,
                 "ours_correct_theirs_not": 1, "theirs_correct_ours_not": 1,
                 "agreement_pct": 50.0}, c
    print("  [PASS] confusion: all four cells and the agreement rate")

    assert confusion([True] * 5, [True] * 5)["agreement_pct"] == 100.0
    assert confusion([True] * 5, [False] * 5)["agreement_pct"] == 0.0
    print("  [PASS] confusion: perfect agreement and perfect disagreement")

    k = known_from_labels([0, 0, 1, 1, 2, 2], [False, True, False, False, True, True])
    assert k == {0: True, 1: False, 2: True}, k
    print("  [PASS] known_from_labels: any-correct rule, matching get_known.py:115")

    # one beam flipping can flip a whole question's status -- and only in one direction
    pids = [0, 0, 1, 1]
    d = partition_delta(pids, [True, False, False, False], [False, False, False, False])
    assert d["ours_known"] == 1 and d["theirs_known"] == 0
    assert d["prompts_that_flip"] == 1 and d["known_only_ours"] == 1 and d["known_only_theirs"] == 0
    print("  [PASS] partition_delta: a single beam disagreement moves a question's known status")

    same = partition_delta([0, 0, 1, 1], [True, False, True, False], [False, True, False, True])
    assert same["prompts_that_flip"] == 0, \
        "beams can disagree while every question keeps its status -- that is the case that matters"
    print("  [PASS] partition_delta: beam disagreement need NOT change the partition")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for ind, flags in [(3, [True, False]), (4, [False])]:
                f.write(json.dumps({"ind": ind, "prompt": "p", "correct_answers": ["a"],
                                    "incorrect_answers": [],
                                    "result": [{"beam_id": i, "gen_text": "g%d" % i,
                                                "score": {"is_correct": c}}
                                               for i, c in enumerate(flags)]}) + "\n")
        recs = load_jsonl_beams(p)
    assert [r["prompt_id"] for r in recs] == [3, 3, 4]
    assert [r["ours_is_correct"] for r in recs] == [True, False, False]
    assert recs[0]["gen_text"] == "g0"
    print("  [PASS] load_jsonl_beams: flattens the adapter's format, preserves order and labels")
    print("\n[PASS] All self-test assertions passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tydiqa", choices=HARP_DATASETS + ["all"])
    ap.add_argument("--model_folder", default="qwen-2.5-7b-instruct")
    ap.add_argument("--harp-dir", default=DEFAULT_HARP)
    ap.add_argument("--in-root", default=os.path.join(HERE, "harp_input"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "judge_agreement"))
    ap.add_argument("--device", default="cpu",
                    help="cpu is what their judge actually uses and is the faithful default.")
    ap.add_argument("--limit-prompts", type=int, default=None)
    ap.add_argument("--check-device", type=int, default=0,
                    help="re-judge this many beams on cpu and report label flips (needs --device cuda)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return

    datasets = HARP_DATASETS if a.dataset == "all" else [a.dataset]
    print("=" * 78)
    print("  JUDGE AGREEMENT -- their DatasetJudge vs our stored labels")
    print("  reads  : %s" % os.path.join(a.in_root, a.model_folder))
    print("  writes : %s" % a.out_dir)
    print("  HARP-Code is read-only; no model is loaded and nothing is regenerated.")
    print("=" * 78, flush=True)
    for ds in datasets:
        run(ds, a.model_folder, a.harp_dir, a.in_root, a.out_dir, a.device,
            a.limit_prompts, a.check_device)


if __name__ == "__main__":
    main()
