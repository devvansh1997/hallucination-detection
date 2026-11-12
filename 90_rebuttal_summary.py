"""
90_rebuttal_summary.py -- one printout of every rebuttal result that exists so far.
=====================================================================================================
Reads the outputs of submit_rebuttal.sh and prints compact tables. Missing files are listed as missing,
never guessed. Read-only and light: run it on the login node inside hal-det, then deactivate.

  module load anaconda/anaconda-2024.10 && conda activate hal-det
  python 90_rebuttal_summary.py                 # everything
  python 90_rebuttal_summary.py > results/rebuttal/summary.txt
  conda deactivate
  python 90_rebuttal_summary.py --self-test
"""

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ("qwen-2.5-7b-instruct", "llama-3.1-8b")
FALCON = "falcon-h1-7b-base"
SMALL = ("truthfulqa", "tydiqa_gp")


def _j(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def pct(x, nd=2):
    return "  --  " if x is None else ("%.*f" % (nd, 100.0 * x))


def kept_table(root, out):
    out.append("\n== Kept subspace vs discarded residual (85; pooled AUROC %, within-question in []) ==")
    out.append("%-30s %16s %16s %16s %16s" % ("cell", "kept", "discarded", "discarded_rand", "kept+discarded"))
    for m in MODELS:
        for d in SMALL:
            r = _j(os.path.join(root, "results", "rebuttal", "kept_vs_discarded_%s_%s_RF.json" % (m, d)))
            if not r:
                out.append("%-30s missing" % ("%s/%s" % (m[:5], d)))
                continue
            s = r["summary"]
            cells = []
            for a in ("kept", "discarded", "discarded_random", "kept_plus_discarded"):
                e = s.get(a)
                cells.append("--" if not e else "%s+-%s [%s]" % (pct(e["pooled_mean"]), pct(e["pooled_std"], 1),
                                                                  pct(e["within_mean"], 1)))
            out.append("%-30s %16s %16s %16s %16s%s" % ("%s/%s" % (m[:5], d), *cells,
                                                        "" if r.get("complete") else "  (partial)"))


def responses_table(root, out):
    out.append("\n== Answers per question, K (86; pooled AUROC %, change vs K=10, known questions) ==")
    labels = ["reference"] + ["%s_k%d" % (a, k) for k in (5, 3)
                              for a in ("fewer_answers", "known_from_k", "as_if_k")]
    for m in MODELS:
        for d in SMALL:
            r = _j(os.path.join(root, "results", "rebuttal", "responses_per_question_%s_%s_RF.json" % (m, d)))
            if not r:
                out.append("  %s/%s missing" % (m[:5], d))
                continue
            out.append("  %s/%s%s" % (m[:5], d, "" if r.get("complete") else "  (partial)"))
            for lab in labels:
                e = r["summary"].get(lab)
                if e:
                    out.append("    %-20s %6s  (%+.2f)  known %5.0f  train %6.0f  test %6.0f rows"
                               % (lab, pct(e["pooled_mean"]), e.get("delta_vs_reference_pts", 0.0),
                                  e["mean_known_questions"], e["mean_train_rows"], e["mean_test_rows"]))


def fewer_labels_table(root, out):
    out.append("\n== ACT-ViT vs LOTUS with fewer labelled questions (87; pooled AUROC %) ==")
    for m in MODELS:
        for d in SMALL:
            r = _j(os.path.join(root, "results", "rebuttal", "actvit_fewer_labels_%s_%s.json" % (m, d)))
            if not r:
                out.append("  %s/%s missing" % (m[:5], d))
                continue
            out.append("  %s/%s%s   frac: ACT-ViT | LOTUS | LOTUS-ACT-ViT | LOTUS ahead (seeds)"
                       % (m[:5], d, "" if r.get("complete") else "  (partial)"))
            for k in sorted(r["summary"], key=float):
                e = r["summary"][k]
                lo = e.get("lotus", {})
                out.append("    %5g%%: %6s | %6s | %s | %s" % (100 * float(k), pct(e["act_vit"]["pooled_mean"]),
                                                          pct(lo.get("pooled_mean")),
                                                          "%+.2f" % e["lotus_minus_act_vit_pts"]
                                                          if "lotus_minus_act_vit_pts" in e else "  -- ",
                                                          e.get("lotus_wins_seeds", "--")))


def _actvit_q(path):
    r = _j(path)
    if not r:
        return None
    q = r.get("protocols", {}).get("question")
    return None if not q else (q["pooled_auroc_mean"], q["pooled_auroc_std"])


def _gb_from_meta(path):
    r = _j(path)
    if not r:
        return None
    if r.get("gb") is not None:
        return float(r["gb"])
    try:
        return r["n_beams"] * r["l_pool"] * r["n_pool"] * r["hidden_size"] * 2 / 1024.0 ** 3
    except KeyError:
        return None


def storage_table(root, out):
    out.append("\n== ACT-ViT at matched storage (56 on 88/59 tensors; question-level pooled AUROC %, stored GB) ==")
    at = os.path.join(os.path.dirname(root), "data-acttensors")
    for m in MODELS:
        for d in SMALL + ("triviaqa",):
            row = []
            for tag, meta in (("", "%s_at_L8_N20.json" % d), ("_L8N5", "%s_at_L8_N5.json" % d),
                              ("_L4N10", "%s_at_L4_N10.json" % d)):
                v = _actvit_q(os.path.join(root, "results", "methods", "act_vit_%s_%s%s.json" % (m, d, tag)))
                gb = _gb_from_meta(os.path.join(at, m, meta))
                row.append("%s%s" % ("--" if not v else "%s+-%s" % (pct(v[0]), pct(v[1], 1)),
                                     "" if gb is None else " (%.1f GB)" % gb))
            lotus_gb = None
            f = os.path.join(os.path.dirname(root), "data", m, "%s_phase2_features.npz" % d)
            if os.path.exists(f):
                lotus_gb = os.path.getsize(f) / 1024.0 ** 3
            out.append("  %-24s N20 %-22s L8N5 %-22s L4N10 %-22s LOTUS stored %s"
                       % ("%s/%s" % (m[:5], d), row[0], row[1], row[2],
                          "--" if lotus_gb is None else "%.1f GB" % lotus_gb))


def falcon_tables(root, out):
    out.append("\n== Transfer with Falcon-H1 (66; target's test answers, pooled AUROC %, within in []) ==")
    files = sorted(glob.glob(os.path.join(root, "results", "cross_model", "cross_*%s*_RF.json" % FALCON)))
    if not files:
        out.append("  none yet")
    for p in files:
        r = _j(p)
        if not r or "summary" not in r:
            out.append("  %s (partial)" % os.path.basename(p))
            continue
        s = r["summary"]
        out.append("  %-12s -> %-12s %-10s in-model %s [%s] | aligned %s [%s] | naive %s | proxy %s | map R2 %.3f"
                   % (r["source"][:12], r["target"][:12], r["dataset"],
                      pct(s["in_model"]["pooled_mean"]), pct(s["in_model"]["within_mean"], 1),
                      pct(s["aligned"]["pooled_mean"]), pct(s["aligned"]["within_mean"], 1),
                      pct(s["naive"]["pooled_mean"]), pct(s["proxy"]["pooled_mean"]),
                      s.get("alignment_test_r2_mean", float("nan"))))
    out.append("\n== ACT-ViT on Falcon-H1 (56; question-level pooled AUROC %) ==")
    for d in SMALL:
        v = _actvit_q(os.path.join(root, "results", "methods", "act_vit_%s_%s.json" % (FALCON, d)))
        out.append("  %-12s ACT-ViT %s   (LOTUS reported: TruthfulQA 89.69, TyDiQA-GP 90.21)"
                   % (d, "missing" if not v else "%s +- %s" % (pct(v[0]), pct(v[1]))))


def cost_table(root, out):
    out.append("\n== Cost profile (89) ==")
    files = sorted(glob.glob(os.path.join(root, "results", "rebuttal", "cost_*.json")))
    if not files:
        out.append("  none yet")
    keep = ("offline_total_seconds", "offline_basis_fit", "offline_classifier_fit", "per_answer_ms",
            "per_answer_ms_without_classifier", "fitted_basis_parameters", "projection_macs_per_answer",
            "offline_training_seconds", "peak_gpu_gb_training", "peak_gpu_gb_inference", "peak_gpu_gb",
            "tensor_load_seconds", "stored_gb_loaded", "peak_rss_gb_process", "peak_rss_gb_after_load",
            "offline_ridge_fit_with_cv_seconds", "source_reading_ms_per_answer",
            "projections_seconds_both_models", "map_parameters", "stored_kb_per_answer_float16",
            "max_abs_diff_gpu_vs_cpu_first_50", "test_pooled_auroc")
    for p in files:
        r = _j(p) or {}
        out.append("  %s" % os.path.basename(p))
        for k in keep:
            if k in r:
                out.append("      %-36s %s" % (k, json.dumps(r[k])))


def build(root):
    out = ["REBUTTAL RESULTS under %s" % root]
    for fn in (kept_table, responses_table, fewer_labels_table, storage_table, falcon_tables, cost_table):
        try:
            fn(root, out)
        except Exception as e:                      # a malformed file must not hide the other tables
            out.append("  [%s failed: %s: %s]" % (fn.__name__, type(e).__name__, e))
    return "\n".join(out)


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "repo")
        os.makedirs(os.path.join(root, "results", "rebuttal"))
        os.makedirs(os.path.join(root, "results", "methods"))
        os.makedirs(os.path.join(root, "results", "cross_model"))
        with open(os.path.join(root, "results", "rebuttal",
                               "kept_vs_discarded_qwen-2.5-7b-instruct_truthfulqa_RF.json"), "w") as f:
            json.dump({"complete": True, "summary": {a: {"pooled_mean": 0.8776, "pooled_std": 0.009,
                                                         "within_mean": 0.82}
                                                     for a in ("kept", "discarded", "discarded_random",
                                                               "kept_plus_discarded")}}, f)
        with open(os.path.join(root, "results", "methods",
                               "act_vit_llama-3.1-8b_truthfulqa_L8N5.json"), "w") as f:
            json.dump({"protocols": {"question": {"pooled_auroc_mean": 0.8812, "pooled_auroc_std": 0.012}}}, f)
        with open(os.path.join(root, "results", "cross_model",
                               "cross_%s_to_llama-3.1-8b_truthfulqa_RF.json" % FALCON), "w") as f:
            json.dump({"source": FALCON, "target": "llama-3.1-8b", "dataset": "truthfulqa",
                       "summary": dict({k: {"pooled_mean": 0.85, "within_mean": 0.8} for k in
                                        ("in_model", "aligned", "naive", "proxy")},
                                       alignment_test_r2_mean=0.6)}, f)
        with open(os.path.join(root, "results", "cross_model", "cross_x_to_%s_tydiqa_gp_RF.json" % FALCON), "w") as f:
            f.write("{not json")
        txt = build(root)
    assert "87.76+-0.9 [82.0]" in txt, txt
    assert "88.12+-1.2" in txt and "aligned 85.00" in txt and "(partial)" in txt
    assert "missing" in txt and "none yet" in txt
    print(txt)
    print("\n  [PASS] tables render from partial results; missing and malformed files are reported, not fatal")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--root", default=HERE)
    a = p.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    print(build(a.root))


if __name__ == "__main__":
    main()
