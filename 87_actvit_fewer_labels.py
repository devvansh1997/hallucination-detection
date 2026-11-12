"""
87_actvit_fewer_labels.py -- the strongest supervised baseline, trained on fewer labelled questions.
=====================================================================================================
THE QUESTION (ICLR 2027 review, pjMe Q4). "Could the authors also compare supervised baselines under
varying amounts of labeled data to further assess LOTUS's dependence on training sample size?"

LOTUS already has this curve: 61_flatten_control.py --train-frac 0.05 0.1 0.25 0.5 1.0 (the paper's
limited-label table, results/flatten_control/sampeff_<model>_<dataset>_RF.json). This script puts
ACT-ViT through the SAME curve:

  * the same question-level split and seeds {42,0,1,2,3};
  * the same thinning, imported from 61 (61._subsample_train): whole training QUESTIONS are kept with
    all their answers, the test set is held fixed, and each fraction is an independent draw. Because
    the function and its seeding are shared, ACT-ViT and LOTUS train on exactly the same questions at
    every fraction and seed;
  * ACT-ViT exactly as in the main table: methods/act_vit.py, their architecture and recipe, the
    (L=8, N=20) pooled tensors, early stopping on a grouped carve of the (thinned) training rows.

The output puts LOTUS's numbers from the matching sampeff file next to ACT-ViT's, fraction by fraction.
Results are written after every (fraction, seed), and a rerun skips what is already done.

  python 87_actvit_fewer_labels.py --self-test
  python 87_actvit_fewer_labels.py --dataset truthfulqa --model_folder qwen-2.5-7b-instruct
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEFAULT_OUT = os.path.join(HERE, "results", "rebuttal")
FRACS = (0.05, 0.1, 0.25, 0.5, 1.0)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_mods = {}


def mods():
    if not _mods:
        _mods["s61"] = _load("s61", "61_flatten_control.py")
    return _mods


def frac_key(frac):
    """The key 61 writes for a fraction ("%g"), so the two files line up."""
    return "%g" % float(frac)


def lotus_curve(model_folder, dataset, readout="RF"):
    """{frac_key: {"pooled_mean", "pooled_std", "within_mean", "per_seed": {seed: pooled}}} from 61's
    sampeff file, hosvd arm (the reported detector). Empty if the file is not there."""
    path = os.path.join(HERE, "results", "flatten_control",
                        "sampeff_%s_%s_%s.json" % (model_folder, dataset, readout))
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for k, v in d.get("by_fraction", {}).items():
        h = v["summary"].get("hosvd")
        if h:
            out[k] = {"pooled_mean": h["pooled_mean"], "pooled_std": h["pooled_std"],
                      "within_mean": h.get("within_mean"),
                      "per_seed": {int(r["seed"]): r["pooled_auroc"] for r in h["per_seed"]}}
    return out


def evaluate(method, data, pre, fracs, seeds, c, done=None, on_result=None, log=print):
    """ACT-ViT (or any trained Method) at each fraction and seed. done: {(frac_key, seed): row} already
    computed, skipped. Returns the same dict, completed."""
    s61 = mods()["s61"]
    done = dict(done or {})
    n = len(data.labels)
    for frac in fracs:
        for seed in seeds:
            key = (frac_key(frac), int(seed))
            if key in done:
                log("    %5s%%  seed %-3d already done (%.4f)" % (frac_key(100 * frac), seed,
                                                                  done[key]["pooled_auroc"]))
                continue
            tr_full, te = (np.asarray(v, dtype=int) for v in
                           c["question_split"](data.is_known, data.prompt_id, n, seed))
            tr = s61._subsample_train(tr_full, data.prompt_id, frac, seed)
            assert not np.intersect1d(tr, te).size, "thinning leaked test rows into train"
            if len(np.unique(data.labels[tr])) < 2:
                log("    %5s%%  seed %-3d SKIPPED: one class in the training rows"
                    % (frac_key(100 * frac), seed))
                continue
            t = time.time()
            s = np.asarray(method.score(data, pre, tr, te), dtype=float)
            y, pid = data.labels[te], data.prompt_id[te]
            wp = c["within_prompt_auroc"](s, y, pid)
            row = {"frac": float(frac), "seed": int(seed),
                   "n_train_questions": int(len(np.unique(data.prompt_id[tr]))),
                   "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
                   "pooled_auroc": float(c["pooled_auroc"](s, y)),
                   "within_prompt_auroc": wp["within_prompt_auroc"], "n_pairs": wp["n_pairs"],
                   "seconds": round(time.time() - t, 1)}
            done[key] = row
            log("    %5s%%  seed %-3d %4d questions %5d rows | pooled %.4f  within %s  (%.0fs)"
                % (frac_key(100 * frac), seed, row["n_train_questions"], row["n_train_rows"],
                   row["pooled_auroc"],
                   "n/a" if row["within_prompt_auroc"] is None else "%.4f" % row["within_prompt_auroc"],
                   row["seconds"]))
            if on_result:
                on_result(done)
    return done


def summarize(done, fracs, lotus):
    out = {}
    for frac in fracs:
        k = frac_key(frac)
        rows = sorted((r for (fk, _), r in done.items() if fk == k), key=lambda r: r["seed"])
        if not rows:
            continue
        p = np.array([r["pooled_auroc"] for r in rows])
        w = [r["within_prompt_auroc"] for r in rows if r["within_prompt_auroc"] is not None]
        entry = {"act_vit": {"pooled_mean": float(p.mean()), "pooled_std": float(p.std()),
                             "within_mean": float(np.mean(w)) if w else None,
                             "n_seeds": len(rows), "per_seed": rows},
                 "n_train_questions_mean": float(np.mean([r["n_train_questions"] for r in rows]))}
        if k in lotus:
            entry["lotus"] = {kk: lotus[k][kk] for kk in ("pooled_mean", "pooled_std", "within_mean")}
            paired = [(r["pooled_auroc"], lotus[k]["per_seed"][r["seed"]]) for r in rows
                      if r["seed"] in lotus[k]["per_seed"]]
            entry["lotus_minus_act_vit_pts"] = round(100.0 * (lotus[k]["pooled_mean"] - p.mean()), 2)
            entry["lotus_wins_seeds"] = "%d/%d" % (sum(b > a for a, b in paired), len(paired))
        out[k] = entry
    return out


def run(dataset, model_folder, data_dir, out_dir, device, dtype, fracs=FRACS, seeds=None,
        at_dir=None, n_eff=20, l_eff=8, epochs=None):
    s56 = _load("s56", "56_run_method.py")
    import methods as M
    s61 = mods()["s61"]
    c = M.canonical()
    seeds = [int(s) for s in (seeds or c["seeds"])]
    fracs = sorted(float(f) for f in fracs)
    dst = os.path.join(out_dir, "actvit_fewer_labels_%s_%s.json" % (model_folder, dataset))
    s61._retry_os(lambda: os.makedirs(out_dir, exist_ok=True), "creating %s" % out_dir)

    method = M.get("act_vit")
    from methods.act_vit import DEFAULT_AT_DIR, EPOCHS
    method.configure(argparse.Namespace(m_at_dir=at_dir or DEFAULT_AT_DIR, m_n_eff=n_eff,
                                        m_l_eff=l_eff, m_epochs=epochs or EPOCHS))
    data = s56.load_bundle(dataset, model_folder, data_dir, device, dtype)
    print("  %r" % data, flush=True)
    lotus = lotus_curve(model_folder, dataset)
    print("  LOTUS curve from 61: %s" % (", ".join(sorted(lotus, key=float)) if lotus else
                                         "NOT FOUND (ACT-ViT only)"), flush=True)

    done = {}
    if os.path.exists(dst):
        with open(dst) as f:
            prev = json.load(f)
        for k, e in prev.get("summary", {}).items():
            for r in e["act_vit"]["per_seed"]:
                done[(frac_key(r["frac"]), int(r["seed"]))] = r
        print("  resuming: %d (fraction, seed) results already in %s" % (len(done), dst), flush=True)

    t0 = time.time()
    pre = method.precompute(data)
    t_pre = time.time() - t0

    def save(d, complete=False):
        s61._write_result(dst, {"dataset": dataset, "model_folder": model_folder, "method": "act_vit",
                                "fractions": fracs, "seeds": seeds,
                                "protocol": "question-level split; training questions thinned with "
                                            "61._subsample_train (same draws as LOTUS); test fixed",
                                "method_meta": method.meta(), "precompute_seconds": round(t_pre, 1),
                                "summary": summarize(d, fracs, lotus), "complete": complete,
                                "elapsed_seconds": round(time.time() - t0, 1)})

    done = evaluate(method, data, pre, fracs, seeds, c, done=done, on_result=save,
                    log=lambda s: print(s, flush=True))
    save(done, complete=True)

    summ = summarize(done, fracs, lotus)
    print("\n  pooled AUROC by fraction of training questions (mean over seeds):")
    print("    %6s  %8s  %8s  %8s  %s" % ("frac", "ACT-ViT", "LOTUS", "L - A", "LOTUS ahead"))
    for frac in fracs:
        e = summ.get(frac_key(frac))
        if not e:
            continue
        a = e["act_vit"]["pooled_mean"]
        if "lotus" in e:
            print("    %5s%%  %8.4f  %8.4f  %+8.2f  %s seeds" % (frac_key(100 * frac), a,
                                                          e["lotus"]["pooled_mean"],
                                                          e["lotus_minus_act_vit_pts"],
                                                          e["lotus_wins_seeds"]))
        else:
            print("    %5s%%  %8.4f" % (frac_key(100 * frac), a))
    return summ


# ---------------------------------------------------------------------------------------------

def self_test():
    print("=" * 78)
    print("  SELF-TEST: 87_actvit_fewer_labels")
    print("=" * 78)
    import methods as M
    c = M.canonical()
    s61 = mods()["s61"]
    rng = np.random.default_rng(0)

    n_q, K = 80, 10
    pid = np.repeat(np.arange(n_q), K)
    y = np.ones(n_q * K, dtype=int)
    for q in range(n_q):
        if q % 4:
            y[q * K + rng.choice(K, size=rng.integers(1, 6), replace=False)] = 0
    seq = {"input_ids": [None] * len(y), "prompt_len": [0] * len(y), "prompt_id": pid,
           "all_hallucination_flag": y, "decoding_config": {}}
    data = M.DataBundle("synthetic", "m", "id", ".", seq, "cpu", "float32")

    seen = []

    class Recorder(M.Method):
        """Records the training rows it is handed and scores test rows by a noisy label."""
        name, granularity, trains = "recorder", "beam", True

        def score(self, d, pre, tr, te):
            seen.append(np.array(tr))
            r = np.random.default_rng(len(seen))
            return d.labels[te] + r.normal(0, 0.8, len(te))

    fracs, seeds = (0.25, 1.0), [42, 0]
    done = evaluate(Recorder(), data, None, fracs, seeds, c, log=lambda s: None)
    assert set(done) == {(frac_key(f), s) for f in fracs for s in seeds}

    # The training rows ACT-ViT gets must be EXACTLY the rows LOTUS got at the same fraction and seed.
    for i, (f, s) in enumerate([(f, s) for f in fracs for s in seeds]):
        tr_full, _ = c["question_split"](data.is_known, data.prompt_id, len(y), s)
        want = s61._subsample_train(np.asarray(tr_full, dtype=int), data.prompt_id, f, s)
        assert np.array_equal(seen[i], want), "fraction %g seed %d: different training rows" % (f, s)
    print("  [PASS] training rows are exactly 61._subsample_train's draw at every fraction and seed")

    # Resume: nothing already done is recomputed.
    n_calls = len(seen)
    done2 = evaluate(Recorder(), data, None, fracs + (0.5,), seeds, c, done=done, log=lambda s: None)
    assert len(seen) == n_calls + len(seeds), "resume recomputed finished results"
    assert all(done2[k] == done[k] for k in done)
    print("  [PASS] a rerun skips finished (fraction, seed) results and adds only the new ones")

    # LOTUS side by side, read from a sampeff file in 61's format.
    import tempfile
    lot = {"0.25": {"pooled_mean": 0.80, "pooled_std": 0.01, "within_mean": 0.7,
                    "per_seed": {42: 0.81, 0: 0.79}}}
    summ = summarize(done2, fracs + (0.5,), lot)
    e = summ["0.25"]
    assert "lotus" in e and e["lotus_wins_seeds"].endswith("/2")
    assert abs(e["lotus_minus_act_vit_pts"] - 100 * (0.80 - e["act_vit"]["pooled_mean"])) < 0.01
    assert "lotus" not in summ["1"]
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "results", "flatten_control"))
        payload = {"by_fraction": {"0.05": {"summary": {"hosvd": {
            "pooled_mean": 0.77, "pooled_std": 0.02, "within_mean": 0.6,
            "per_seed": [{"seed": 42, "pooled_auroc": 0.78}]}}}}}
        with open(os.path.join(td, "results", "flatten_control",
                               "sampeff_m_d_RF.json"), "w") as f:
            json.dump(payload, f)
        global HERE
        saved, HERE = HERE, td
        try:
            got = lotus_curve("m", "d")
        finally:
            HERE = saved
    assert got["0.05"]["per_seed"] == {42: 0.78} and got["0.05"]["pooled_mean"] == 0.77
    assert frac_key(0.05) == "0.05" and frac_key(1.0) == "1" and frac_key(0.1) == "0.1"
    print("  [PASS] LOTUS's curve is read from 61's sampeff format and paired seed by seed")

    print("\n  ALL PASS")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dataset", choices=["truthfulqa", "tydiqa_gp", "nq_open", "triviaqa"])
    p.add_argument("--model_folder")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--fracs", nargs="+", type=float, default=list(FRACS))
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--at-dir", default=None, help="ACT-ViT tensors (default ../data-acttensors)")
    p.add_argument("--n-eff", type=int, default=20)
    p.add_argument("--l-eff", type=int, default=8)
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
    run(a.dataset, a.model_folder, data_dir, a.out_dir, a.device, a.dtype, a.fracs, a.seeds,
        a.at_dir, a.n_eff, a.l_eff)


if __name__ == "__main__":
    main()
