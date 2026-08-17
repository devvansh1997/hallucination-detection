"""
52_pipeline_summary.py -- one consolidated report for a whole model's pipeline run.
=====================================================================================================
Collects every artifact a full run produces and prints one table, so a fire-and-forget job ends
with data stats AND inference stats together instead of scattered across a dozen logs.

Reads (all optional -- anything missing is reported as missing, never guessed):
    data/{model}/{ds}_sequences_v1.pt              generation: counts, hallucination rate, known/unknown
    data/{model}/manifest_{ds}_v1.json             extraction: shapes, hashes
    results/{model}/session06_phase3_partA_{ds}.json           our eval, question-level
    results/{model}-answersplit/...partA_{ds}.json             our eval, answer-level
    harp_input/{model}/adapter_summary.json        handoff to HARP
    results/judge_agreement/judge_agreement_{model}_{hds}.json audit #1

Writes results/{model}/pipeline_summary.json and prints a human table.

Usage:
  python 52_pipeline_summary.py --self-test
  python 52_pipeline_summary.py --model_folder llama-3.1-8b
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS_OURS_TO_HARP = {"truthfulqa": "truthful_qa", "triviaqa": "trivia_qa",
                   "nq_open": "nq_open", "tydiqa_gp": "tydiqa"}
CONDITIONS = ["core_max", "q_velocity", "q_static", "core_concat", "joint_tensor", "triple_concat"]


def _read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return "UNREADABLE"


def gen_stats(seq_path):
    """Generation-stage facts, straight from the pinned sequences file."""
    if not os.path.exists(seq_path):
        return None
    import torch
    seq = torch.load(seq_path, weights_only=False)
    hall = np.asarray(seq["all_hallucination_flag"], dtype=bool)
    pid = np.asarray(seq["prompt_id"])
    n_prompts = len(set(pid.tolist()))
    out = {"n_prompts": n_prompts, "n_beams": int(len(hall)),
           "beams_per_prompt": round(len(hall) / n_prompts, 3) if n_prompts else None,
           "hallucination_rate_pct": round(100.0 * float(hall.mean()), 2)}
    if "all_is_known" in seq:
        known = np.asarray(seq["all_is_known"], dtype=bool)
        out["n_known_prompts"] = int(known.sum())
        out["n_unknown_prompts"] = int(n_prompts - known.sum())
        out["known_pct"] = round(100.0 * float(known.sum()) / n_prompts, 2) if n_prompts else None
    return out


def best_condition(partA, protocol_key="harp"):
    """Best pooled AUROC over conditions, and which readout won. Handles both file shapes:
    a combined part-A file (conditions nested under 'harp') and a single-condition file."""
    if not partA or partA == "UNREADABLE":
        return None
    blk = partA.get(protocol_key)
    if blk is None:
        return None
    # single-condition file: {"condition": "...", "harp": {"RF_mean": ...}}
    if "RF_mean" in blk:
        blk = {partA.get("condition", "?"): blk}
    # Compare on the RAW fraction and only convert to a percentage at the end. Comparing a
    # fraction against an already-scaled percentage silently makes the first entry unbeatable.
    best_raw, best = None, None
    for cond, v in blk.items():
        if not isinstance(v, dict) or "RF_mean" not in v:
            continue
        for clf in ("RF", "LR"):
            m = v.get(clf + "_mean")
            if m is None:
                continue
            if best_raw is None or m > best_raw:
                best_raw = m
                best = {"auroc": round(100.0 * m, 2), "condition": cond, "readout": clf,
                        "std": round(100.0 * v.get(clf + "_std", 0.0), 2)}
    return best


def collect(model_folder, data_dir, results_root, harp_input_root, judge_root, datasets):
    rows = []
    for ds in datasets:
        hds = DS_OURS_TO_HARP[ds]
        r = {"dataset": ds, "harp_dataset": hds}
        r["generation"] = gen_stats(os.path.join(data_dir, model_folder, f"{ds}_sequences_v1.pt"))
        man = _read_json(os.path.join(data_dir, model_folder, f"manifest_{ds}_v1.json"))
        r["extraction_manifest"] = "present" if man not in (None, "UNREADABLE") else (
            "unreadable" if man == "UNREADABLE" else "missing")
        qs = _read_json(os.path.join(results_root, model_folder,
                                     f"session06_phase3_partA_{ds}.json"))
        an = _read_json(os.path.join(results_root, model_folder + "-answersplit",
                                     f"session06_phase3_partA_{ds}.json"))
        r["ours_question_level"] = best_condition(qs)
        r["ours_answer_level"] = best_condition(an)
        if r["ours_question_level"] and r["ours_answer_level"]:
            r["leakage_cost"] = round(r["ours_answer_level"]["auroc"]
                                      - r["ours_question_level"]["auroc"], 2)
        r["judge_agreement"] = _read_json(os.path.join(
            judge_root, f"judge_agreement_{model_folder}_{hds}.json"))
        rows.append(r)
    adapter = _read_json(os.path.join(harp_input_root, model_folder, "adapter_summary.json"))
    return {"model_folder": model_folder, "datasets": rows,
            "adapter": "present" if adapter not in (None, "UNREADABLE") else "missing"}


def render(summary):
    print("=" * 96)
    print("  PIPELINE SUMMARY -- %s" % summary["model_folder"])
    print("=" * 96)
    print("\n-- DATA --")
    print("  %-12s %8s %8s %9s %9s %9s" % ("dataset", "prompts", "beams", "halluc%", "known", "unknown"))
    for r in summary["datasets"]:
        g = r["generation"]
        if not g:
            print("  %-12s %s" % (r["dataset"], "NOT GENERATED"))
            continue
        print("  %-12s %8d %8d %8.1f%% %9s %9s" % (
            r["dataset"], g["n_prompts"], g["n_beams"], g["hallucination_rate_pct"],
            g.get("n_known_prompts", "-"), g.get("n_unknown_prompts", "-")))

    print("\n-- INFERENCE (best of %d conditions, pooled AUROC %%) --" % len(CONDITIONS))
    print("  %-12s %-22s %-22s %10s" % ("dataset", "question-level", "answer-level", "leakage"))
    for r in summary["datasets"]:
        def fmt(b):
            return "%6.2f  %s/%s" % (b["auroc"], b["condition"][:9], b["readout"]) if b else "--"
        print("  %-12s %-22s %-22s %10s" % (
            r["dataset"], fmt(r["ours_question_level"]), fmt(r["ours_answer_level"]),
            ("%+.2f" % r["leakage_cost"]) if r.get("leakage_cost") is not None else "--"))

    ja = [r for r in summary["datasets"] if r.get("judge_agreement") not in (None, "UNREADABLE")]
    if ja:
        print("\n-- JUDGE AGREEMENT (their DatasetJudge vs our labels) --")
        print("  %-12s %10s %10s" % ("dataset", "beam%", "known%"))
        for r in ja:
            j = r["judge_agreement"]
            print("  %-12s %9.2f%% %9.2f%%" % (
                r["dataset"], j["beam_level"]["agreement_pct"],
                j["prompt_level"]["status_agreement_pct"]))

    missing = [r["dataset"] for r in summary["datasets"] if not r["generation"]]
    print("\n-- STATUS --")
    print("  adapter handoff : %s" % summary["adapter"])
    print("  not generated   : %s" % (", ".join(missing) if missing else "none"))
    print("=" * 96)


def self_test():
    print("=" * 70)
    print("  SELF-TEST: 52_pipeline_summary (synthetic, no cluster files)")
    print("=" * 70)

    assert _read_json("/definitely/not/here.json") is None
    print("  [PASS] _read_json: missing file returns None rather than raising")

    combined = {"harp": {"core_max": {"RF_mean": 0.87, "RF_std": 0.01, "LR_mean": 0.83,
                                      "LR_std": 0.01},
                         "q_static": {"RF_mean": 0.85, "RF_std": 0.01, "LR_mean": 0.91,
                                      "LR_std": 0.02}}}
    b = best_condition(combined)
    assert b["auroc"] == 91.0 and b["condition"] == "q_static" and b["readout"] == "LR", b
    print("  [PASS] best_condition: picks the best across BOTH conditions and readouts")

    single = {"condition": "triple_concat",
              "harp": {"RF_mean": 0.94, "RF_std": 0.005, "LR_mean": 0.90, "LR_std": 0.004}}
    b2 = best_condition(single)
    assert b2["auroc"] == 94.0 and b2["condition"] == "triple_concat" and b2["readout"] == "RF"
    print("  [PASS] best_condition: handles single-condition files too (the TriviaQA shape)")

    assert best_condition(None) is None and best_condition("UNREADABLE") is None
    assert best_condition({"grouped": {}}) is None, "a file with no harp block must not crash"
    print("  [PASS] best_condition: degrades to None on missing/absent/unreadable input")

    assert set(DS_OURS_TO_HARP) == {"truthfulqa", "triviaqa", "nq_open", "tydiqa_gp"}
    print("  [PASS] dataset mapping matches the adapter's")
    print("\n[PASS] All self-test assertions passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_folder", default="llama-3.1-8b")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-root", default=os.path.join(HERE, "results"))
    ap.add_argument("--harp-input-root", default=os.path.join(HERE, "harp_input"))
    ap.add_argument("--judge-root", default=os.path.join(HERE, "results", "judge_agreement"))
    ap.add_argument("--datasets", default="tydiqa_gp,truthfulqa,nq_open")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return

    data_dir = a.data_dir
    if not data_dir:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]

    datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]
    summary = collect(a.model_folder, data_dir, a.results_root, a.harp_input_root,
                      a.judge_root, datasets)
    render(summary)

    out_dir = os.path.join(a.results_root, a.model_folder)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "pipeline_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nWrote: %s" % out)


if __name__ == "__main__":
    main()
