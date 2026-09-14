"""
63_rank_sweep.py -- does the reported detector depend on its ranks?
==================================================================================================

Section 4 reports layer rank r_L = 5 (4 for the eight-layer update summary) and feature rank r_F = 64.
Those values came from an early exploratory search, so the paper needs evidence that they are not a
lucky pick. This script varies ONE rank at a time and keeps everything else exactly as the main
results: triple_concat, the question-level split, five seeds, random forest.

  layer rank    k = 1..9 at r_F = 64     (the eight-layer update summary gets k - 1, at least 1)
  feature rank  r_F in {8,16,32,64,128} at k = 5

That is 13 settings. Each one is the flatten control's hosvd arm with different ranks -- 61.build and
26.fit_eval, nothing reimplemented -- so (k = 5, r_F = 64) IS the reported detector and must reproduce
results/flatten_control exactly. The run checks that.

READING IT. Flat around (5, 64): the choice is justified, because nearby ranks give the same result.
A sharp peak at (5, 64): this curve alone cannot rule out that the ranks were fitted to the test split.

  python 63_rank_sweep.py --self-test
  python 63_rank_sweep.py --dataset tydiqa_gp --model_folder llama-3.1-8b
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "rank_sweep")
CONDITION = "triple_concat"
K_GRID = list(range(1, 10))
RF_GRID = [8, 16, 32, 64, 128]
K_REPORTED, RF_REPORTED = 5, 64


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_mods = {}


def mods():
    if not _mods:
        _mods["s61"] = _load("s61", "61_flatten_control.py")
        _mods.update(_mods["s61"].mods())            # s43, s26, s44 -- the modules 61 itself uses
    return _mods


def layer_ranks(k, block_layers):
    """The longest summary gets k; a shorter one gets k minus the difference, at least 1, at most its
    own length. For triple_concat's (9, 9, 8): k = 5 gives the reported 5, 5, 4."""
    top = max(block_layers)
    return [max(1, min(L, k - (top - L))) for L in block_layers]


def settings():
    """One rank at a time. The reported point (5, 64) appears once."""
    out = [(k, RF_REPORTED) for k in K_GRID]
    out += [(K_REPORTED, rf) for rf in RF_GRID if rf != RF_REPORTED]
    return out


def spec_for(base_spec, block_layers, k, rf):
    return [(key, r, rf) for (key, _, _), r in zip(base_spec, layer_ranks(k, block_layers))]


def score_setting(m, c, feats, spec, y, pid, is_known, seeds, readout):
    """The flatten control's hosvd arm under a given spec, seed by seed."""
    rows = []
    for seed in seeds:
        tr, te = c["question_split"](is_known, pid, len(y), seed)
        tr, te = np.asarray(tr, dtype=int), np.asarray(te, dtype=int)
        Z = m["s61"].build("hosvd", feats, spec, tr, seed,
                           {"y": y, "prompt_id": pid, "readout": readout})
        sc = m["s26"].fit_eval(readout, Z[tr], y[tr], Z[te], seed)
        # (scores, labels), in that order: reversed, the metric silently returns 1 - AUROC.
        wp = c["within_prompt_auroc"](sc, y[te], pid[te])
        rows.append({"seed": int(seed), "pooled_auroc": float(c["pooled_auroc"](sc, y[te])),
                     "within_prompt_auroc": wp["within_prompt_auroc"]})
    p = np.array([r["pooled_auroc"] for r in rows])
    return {"pooled_mean": float(p.mean()), "pooled_std": float(p.std()),
            "width": int(Z.shape[1]), "per_seed": rows}


def reproduction_check(res, model_folder, dataset, readout):
    path = os.path.join(HERE, "results", "flatten_control",
                        "flatten_%s_%s_%s.json" % (model_folder, dataset, readout))
    key = "k%d_rf%d" % (K_REPORTED, RF_REPORTED)
    if not os.path.exists(path) or key not in res:
        return "not checked (no flatten control result for this cell)"
    with open(path) as f:
        ref = {r["seed"]: r["pooled_auroc"] for r in json.load(f)["summary"]["hosvd"]["per_seed"]}
    diffs = [abs(r["pooled_auroc"] - ref[r["seed"]]) for r in res[key]["per_seed"] if r["seed"] in ref]
    if not diffs:
        return "not checked (no overlapping seeds)"
    return "EXACT" if max(diffs) == 0.0 else "DIFFERS by up to %.2e" % max(diffs)


def summarize(res):
    rep = res["k%d_rf%d" % (K_REPORTED, RF_REPORTED)]
    best_key = max(res, key=lambda s: res[s]["pooled_mean"])
    best = res[best_key]
    # Plain bool and float throughout: a numpy bool in here makes the final json.dump raise.
    return {"reported_mean": float(rep["pooled_mean"]), "reported_std": float(rep["pooled_std"]),
            "best_setting": best_key, "best_mean": float(best["pooled_mean"]),
            "best_std": float(best["pooled_std"]),
            "reported_within_one_std_of_best": bool(rep["pooled_mean"]
                                                    >= best["pooled_mean"] - best["pooled_std"]),
            "layer_curve": [[k, float(res["k%d_rf%d" % (k, RF_REPORTED)]["pooled_mean"]),
                             float(res["k%d_rf%d" % (k, RF_REPORTED)]["pooled_std"])] for k in K_GRID],
            "feature_curve": [[rf, float(res["k%d_rf%d" % (K_REPORTED, rf)]["pooled_mean"]),
                               float(res["k%d_rf%d" % (K_REPORTED, rf)]["pooled_std"])]
                              for rf in RF_GRID]}


def report(summ, repro):
    """The end-of-run table. A function so the self-test can exercise it: on this project a summary
    printer has already crashed at the end of a finished run and lost the output."""
    print("\n  layer rank at r_F=64          feature rank at k=5")
    for j in range(max(len(K_GRID), len(RF_GRID))):
        left = ("  k=%d  %.4f +- %.4f" % tuple(summ["layer_curve"][j])) if j < len(K_GRID) else " " * 25
        right = ("   r_F=%-3d %.4f +- %.4f" % tuple(summ["feature_curve"][j])) if j < len(RF_GRID) else ""
        print(left + "   " + right)
    print("\n  reported (5, 64): %.4f +- %.4f | best %s: %.4f +- %.4f | reported within one std of best: %s"
          % (summ["reported_mean"], summ["reported_std"], summ["best_setting"], summ["best_mean"],
             summ["best_std"], summ["reported_within_one_std_of_best"]))
    print("  reproduction of the flatten control at (5, 64): %s" % repro)


def run(dataset, model_folder, data_dir, out_dir, readout, seeds=None):
    m = mods()
    import methods.base as B
    c = B.canonical()
    s61 = m["s61"]
    seeds = [int(x) for x in (seeds or c["seeds"])]
    base = m["s44"].CONDITION_SPECS[CONDITION]
    dst = os.path.join(out_dir, "ranks_%s_%s_%s.json" % (model_folder, dataset, readout))
    s61._retry_os(lambda: os.makedirs(out_dir, exist_ok=True), "creating %s" % out_dir)

    feats, y, pid, is_known = m["s44"].load_new_dataset(dataset, data_dir, model_folder,
                                                        condition=CONDITION)
    y, pid = np.asarray(y, dtype=int), np.asarray(pid)
    L_blocks = [feats[key].shape[1] for key, _, _ in base]
    print("  [%s/%s] %d answers | %s | %d settings x %d seeds | readout %s"
          % (model_folder, dataset, len(y), CONDITION, len(settings()), len(seeds), readout), flush=True)

    def payload(res, complete, extra=None):
        d = {"dataset": dataset, "model_folder": model_folder, "readout": readout,
             "condition": CONDITION, "seeds": seeds, "protocol": "question-level (paper protocol)",
             "design": "one rank at a time: k=1..9 at r_F=64, r_F in %s at k=5" % RF_GRID,
             "layer_rank_rule": "max(1, k - (9 - L)), capped at L", "results": res, "complete": complete}
        d.update(extra or {})
        return d

    res, t_start = {}, time.time()
    for i, (k, rf) in enumerate(settings(), 1):
        spec = spec_for(base, L_blocks, k, rf)
        t0 = time.time()
        r = score_setting(m, c, feats, spec, y, pid, is_known, seeds, readout)
        r.update({"k": k, "r_F": rf, "layer_ranks": [s[1] for s in spec],
                  "seconds": round(time.time() - t0, 1)})
        res["k%d_rf%d" % (k, rf)] = r
        print("  [%2d/%d] k=%d r_L=%-6s r_F=%-3d width %-5d  AUROC %.4f +- %.4f  (%.0fs)"
              % (i, len(settings()), k, ",".join(str(s[1]) for s in spec), rf, r["width"],
                 r["pooled_mean"], r["pooled_std"], r["seconds"]), flush=True)
        s61._write_result(dst, payload(res, False))      # a partial result survives a timeout

    summ = summarize(res)
    repro = reproduction_check(res, model_folder, dataset, readout)
    report(summ, repro)
    s61._write_result(dst, payload(res, True, {"summary": summ, "reproduction_check": repro,
                                               "elapsed_seconds": round(time.time() - t_start, 1)}))


def self_test():
    print("=" * 78)
    print("  SELF-TEST: 63_rank_sweep")
    print("=" * 78)
    m = mods()
    import methods.base as B
    c = B.canonical()

    base = m["s44"].CONDITION_SPECS[CONDITION]
    assert layer_ranks(5, [9, 9, 8]) == [5, 5, 4] and layer_ranks(1, [9, 9, 8]) == [1, 1, 1]
    assert layer_ranks(9, [9, 9, 8]) == [9, 9, 8]
    assert spec_for(base, [9, 9, 8], K_REPORTED, RF_REPORTED) == [tuple(s) for s in base], base
    grid = settings()
    assert len(grid) == 13 and grid.count((K_REPORTED, RF_REPORTED)) == 1, grid
    print("  [PASS] 13 settings, one rank at a time; (5, 64) reproduces the reported spec %s" % base)

    # End to end on a planted low-rank signal: the widths must follow the ranks, and the detector
    # must find the signal, or the run is not measuring anything.
    rng = np.random.default_rng(0)
    nq, per_q, F = 40, 5, 24
    n = nq * per_q
    pid = np.repeat(np.arange(nq), per_q)
    y = (np.arange(n) % 2).astype(int)
    ul = rng.standard_normal(9); ul /= np.linalg.norm(ul)
    feats = {}
    for key, L, width in (("core", 9, F), ("static", 9, 2 * F), ("velocity", 8, 2 * F)):
        uf = rng.standard_normal(width); uf /= np.linalg.norm(uf)
        A = rng.standard_normal((n, L, width)).astype(np.float32)
        A += (2.5 * y[:, None, None] * np.outer(ul[:L], uf)[None]).astype(np.float32)
        feats[key] = A
    is_known, _ = m["s44"].derive_is_known(y, pid)
    res = {}
    for k, rf in ((1, 8), (5, 8), (5, 4)):
        spec = spec_for(base, [9, 9, 8], k, rf)
        r = score_setting(m, c, feats, spec, y, pid, is_known, [42, 0], "RF")
        assert r["width"] == sum(s[1] * rf for s in spec), (r["width"], spec)
        assert r["pooled_mean"] > 0.8, "planted signal not found at k=%d r_F=%d: %.3f" % (k, rf, r["pooled_mean"])
        res["k%d_rf%d" % (k, rf)] = r
    print("  [PASS] widths follow the ranks and the planted signal is found at every setting "
          "(AUROC %s)" % ", ".join("%.3f" % v["pooled_mean"] for v in res.values()))

    fake = {}
    for k, rf in settings():
        mean = 0.80 + 0.01 * min(k, 5) + 0.001 * np.log2(rf)
        fake["k%d_rf%d" % (k, rf)] = {"pooled_mean": mean, "pooled_std": 0.01, "width": 1,
                                      "per_seed": [{"seed": 42, "pooled_auroc": mean}]}
    summ = summarize(fake)
    repro = reproduction_check(fake, "no-such-model", "no-such-dataset", "RF")
    report(summ, repro)
    assert len(summ["layer_curve"]) == 9 and len(summ["feature_curve"]) == 5
    assert isinstance(summ["reported_within_one_std_of_best"], bool)
    json.dumps(summ)
    assert repro.startswith("not checked")
    print("  [PASS] summary, end-of-run table and reproduction check run on a full 13-setting grid, "
          "and the summary serialises to JSON")

    print("\n  ALL PASS")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dataset")
    p.add_argument("--model_folder")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--readout", default="RF", choices=["RF", "LR"])
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    a = p.parse_args()

    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not a.dataset or not a.model_folder:
        raise SystemExit("--dataset and --model_folder are required (or use --self-test)")
    data_dir = a.data_dir
    if not data_dir:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]
    run(a.dataset, a.model_folder, data_dir, a.out_dir, a.readout, seeds=a.seeds)


if __name__ == "__main__":
    main()
