"""
86_responses_per_question.py -- how much do the results depend on sampling K = 10 answers per question?
=====================================================================================================
THE QUESTION (ICLR 2027 review, zjPm Q2). "How sensitive are the results to the number of sampled
responses? Since [K] affects both training-set size and the identification of model-known questions,
separating these effects would help assess the robustness of the evaluation protocol."

K does two things at once:
  (a) training-set size: each training question contributes K answers;
  (b) which questions count: a question is model-known if at least one of its K answers is correct,
      and only known questions are split into train/test (unknown ones all go to test).

For each k in {5, 3}, a seeded random k of each question's 10 answers is drawn, and four arms change
one thing at a time. The detector is always the reported one (61's hosvd arm: triple_concat, ranks
(5,64)/(5,64)/(4,64), random forest, question-level split, seeds {42,0,1,2,3}).

  reference       K = 10 as reported. Bit-identical to the flatten control's hosvd arm.
  fewer_answers   effect (a) alone: the known set and the split come from all 10 answers; training uses
                  only the k drawn answers per question; the test set is untouched.
  known_from_k    effect (b) alone: the known set comes from the k drawn answers; every answer of every
                  question is kept on its side of the split.
  as_if_k         both: what the protocol would have produced had we sampled k answers. Known set from
                  the k answers, and only those k answers in training AND test.

Test sets differ between arms whenever the known set changes, so each row records its own question and
answer counts and the test hallucination rate next to the AUROC.

  python 86_responses_per_question.py --self-test
  python 86_responses_per_question.py --dataset truthfulqa --model_folder qwen-2.5-7b-instruct
"""

import argparse
import importlib.util
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "rebuttal")
CONDITION = "triple_concat"
KS = (5, 3)
ARMS = ("fewer_answers", "known_from_k", "as_if_k")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_mods = {}


def mods():
    if not _mods:
        _mods["s61"] = _load("s61", "61_flatten_control.py")
        _mods.update(_mods["s61"].mods())            # s43, s26, s44
    return _mods


def draw_answers(pid, k, seed):
    """A seeded random k answers of every question (all of them if it has fewer). Sorted row indices.
    The seed mixes in k, so the k = 3 draw is not a truncation of the k = 5 draw."""
    rng = np.random.default_rng(7919 * int(seed) + 31 * int(k) + 17)
    out = []
    for q in np.unique(pid):
        rows = np.flatnonzero(pid == q)
        out.append(rows if len(rows) <= k else rng.choice(rows, size=k, replace=False))
    return np.sort(np.concatenate(out))


def known_from(y, pid, rows, n_questions):
    """Model-known from the given rows only: a question is known if one of ITS rows is correct."""
    known = np.zeros(n_questions, dtype=bool)
    ok = rows[y[rows] == 0]
    known[pid[ok]] = True
    return known


def arm_rows(arm, split, y, pid, is_known, sub, seed):
    """(train rows, test rows, known set used) for one arm."""
    n = len(y)
    if arm == "reference":
        tr, te = split(is_known, pid, n, seed)
        return np.asarray(tr, dtype=int), np.asarray(te, dtype=int), is_known
    if arm == "fewer_answers":
        tr, te = split(is_known, pid, n, seed)
        tr = np.asarray(tr, dtype=int)
        return tr[np.isin(tr, sub)], np.asarray(te, dtype=int), is_known
    known_k = known_from(y, pid, sub, len(is_known))
    tr, te = (np.asarray(v, dtype=int) for v in split(known_k, pid, n, seed))
    if arm == "known_from_k":
        return tr, te, known_k
    if arm == "as_if_k":
        return tr[np.isin(tr, sub)], te[np.isin(te, sub)], known_k
    raise ValueError(arm)


def evaluate(feats, y, pid, is_known, spec, seeds, ks, readout, c, on_seed=None, log=print):
    """Every arm for every k and seed. Returns {label: [row per seed]}."""
    m = mods()
    width = sum(r_l * r_d for _, r_l, r_d in spec)
    split = c["question_split"]
    res = {}
    for seed in seeds:
        jobs = [("reference", None)] + [(arm, k) for k in ks for arm in ARMS]
        for arm, k in jobs:
            label = arm if k is None else "%s_k%d" % (arm, k)
            sub = None if k is None else draw_answers(pid, k, seed)
            tr, te, known = arm_rows(arm, split, y, pid, is_known, sub, seed)
            assert not np.intersect1d(tr, te).size, "%s: train and test share rows" % label
            assert not (set(pid[tr].tolist()) & set(pid[te].tolist())), \
                "%s: train and test share questions" % label
            if len(np.unique(y[tr])) < 2:
                log("    seed %-3d %-22s SKIPPED: one class in the training rows" % (seed, label))
                continue
            r = m["s61"]._score_one(m, c, "hosvd", feats, spec, tr, te, y, pid, readout, seed, width)
            r.update({"k": k, "n_known_questions": int(known.sum()),
                      "n_train_rows": int(len(tr)), "n_train_questions": int(len(np.unique(pid[tr]))),
                      "n_test_rows": int(len(te)), "n_test_questions": int(len(np.unique(pid[te]))),
                      "test_hallucination_rate": float(y[te].mean())})
            res.setdefault(label, []).append(r)
            log("    seed %-3d %-22s known %4d | train %5d rows | test %5d rows (%.0f%% halluc) | "
                "pooled %.4f  within %s  (%.0fs)"
                % (seed, label, r["n_known_questions"], r["n_train_rows"], r["n_test_rows"],
                   100 * r["test_hallucination_rate"], r["pooled_auroc"],
                   "n/a" if r["within_prompt_auroc"] is None else "%.4f" % r["within_prompt_auroc"],
                   r["seconds"]))
        if on_seed:
            on_seed(res)
    return res


def summarize(res):
    out = {}
    for label, rows in res.items():
        p = np.array([r["pooled_auroc"] for r in rows])
        w = [r["within_prompt_auroc"] for r in rows if r["within_prompt_auroc"] is not None]
        out[label] = {"pooled_mean": float(p.mean()), "pooled_std": float(p.std()),
                      "within_mean": float(np.mean(w)) if w else None, "n_seeds": len(rows),
                      "mean_known_questions": float(np.mean([r["n_known_questions"] for r in rows])),
                      "mean_train_rows": float(np.mean([r["n_train_rows"] for r in rows])),
                      "mean_test_rows": float(np.mean([r["n_test_rows"] for r in rows])),
                      "per_seed": rows}
    if "reference" in out:
        base = out["reference"]["pooled_mean"]
        for label, s in out.items():
            s["delta_vs_reference_pts"] = round(100.0 * (s["pooled_mean"] - base), 2)
    return out


def run(dataset, model_folder, data_dir, out_dir, readout, seeds=None, ks=KS):
    m = mods()
    s61 = m["s61"]
    import methods.base as B
    c = B.canonical()
    seeds = [int(s) for s in (seeds or c["seeds"])]
    spec = m["s44"].CONDITION_SPECS[CONDITION]
    dst = os.path.join(out_dir, "responses_per_question_%s_%s_%s.json" % (model_folder, dataset, readout))
    s61._retry_os(lambda: os.makedirs(out_dir, exist_ok=True), "creating %s" % out_dir)

    feats, y, pid, is_known = m["s44"].load_new_dataset(dataset, data_dir, model_folder,
                                                         condition=CONDITION)
    y, pid, is_known = np.asarray(y, dtype=int), np.asarray(pid, dtype=np.int64), np.asarray(is_known)
    # The canonical split indexes is_known by question id, so ids must be 0..Q-1. Checked, not assumed.
    if not np.array_equal(np.unique(pid), np.arange(len(is_known))):
        raise SystemExit("question ids are not 0..%d; the split's is_known indexing would be wrong"
                         % (len(is_known) - 1))
    per_q = np.bincount(pid)
    print("  [%s/%s] %d answers, %d questions (%d-%d answers each), %d known | ks %s | seeds %s"
          % (model_folder, dataset, len(y), len(is_known), per_q.min(), per_q.max(),
             int(is_known.sum()), list(ks), seeds), flush=True)

    t0 = time.time()

    def save(res, complete=False):
        summ = summarize(res)
        s61._write_result(dst, {"dataset": dataset, "model_folder": model_folder, "readout": readout,
                                "condition": CONDITION, "seeds": seeds, "ks": list(ks),
                                "arms": ["reference"] + list(ARMS), "summary": summ,
                                "complete": complete, "elapsed_seconds": round(time.time() - t0, 1)})
        return summ

    res = evaluate(feats, y, pid, is_known, spec, seeds, ks, readout, c,
                   on_seed=lambda r: save(r), log=lambda s: print(s, flush=True))
    summ = save(res, complete=True)

    print("\n  pooled AUROC, mean over seeds (change vs K = 10 in points):")
    for label in ["reference"] + ["%s_k%d" % (a, k) for k in ks for a in ARMS]:
        if label in summ:
            s = summ[label]
            print("    %-22s %.4f +- %.4f (%+.2f)  known %.0f  train %.0f  test %.0f rows"
                  % (label, s["pooled_mean"], s["pooled_std"], s["delta_vs_reference_pts"],
                     s["mean_known_questions"], s["mean_train_rows"], s["mean_test_rows"]))
    return summ


# ---------------------------------------------------------------------------------------------

def self_test():
    print("=" * 78)
    print("  SELF-TEST: 86_responses_per_question")
    print("=" * 78)
    import methods.base as B
    c = B.canonical()
    rng = np.random.default_rng(0)
    n_q, K = 80, 10
    pid = np.repeat(np.arange(n_q), K).astype(np.int64)
    y = np.ones(n_q * K, dtype=int)
    for q in range(n_q):                       # 3/4 of questions have some correct answers
        if q % 4:
            y[q * K + rng.choice(K, size=rng.integers(1, 6), replace=False)] = 0
    is_known = known_from(y, pid, np.arange(len(y)), n_q)

    # Draws: exactly k per question, inside the question, deterministic, and independent across seeds
    # and across k (not truncations of each other).
    s5 = draw_answers(pid, 5, 0)
    assert len(s5) == 5 * n_q and all((pid[s5] == q).sum() == 5 for q in range(n_q))
    assert np.array_equal(s5, draw_answers(pid, 5, 0)), "draw is not deterministic"
    assert not np.array_equal(s5, draw_answers(pid, 5, 1)), "draws do not change with the seed"
    s3 = draw_answers(pid, 3, 0)
    assert not set(s3.tolist()) <= set(s5.tolist()), "the k=3 draw is a subset of the k=5 draw"
    print("  [PASS] draws keep exactly k answers per question, are seeded, and independent across k")

    # Known-from-k can only lose questions: if one of k answers is correct, one of 10 is.
    kk = known_from(y, pid, s3, n_q)
    assert not (kk & ~is_known).any() and kk.sum() <= is_known.sum()
    print("  [PASS] known from 3 answers is a subset of known from 10 (%d of %d)"
          % (kk.sum(), is_known.sum()))

    split = c["question_split"]
    tr0, te0, _ = arm_rows("reference", split, y, pid, is_known, None, 42)
    tr1, te1, _ = arm_rows("fewer_answers", split, y, pid, is_known, s5, 42)
    assert np.array_equal(te0, te1), "fewer_answers changed the test set"
    assert set(tr1.tolist()) <= set(tr0.tolist()) and len(tr1) == len(tr0) // 2
    print("  [PASS] fewer_answers keeps the reference test set and half its training rows at k=5")

    tr2, te2, k2 = arm_rows("known_from_k", split, y, pid, is_known, s3, 42)
    for rows in (tr2, te2):
        for q in np.unique(pid[rows]):
            assert (pid[rows] == q).sum() == K, "known_from_k must keep every answer of a question"
    assert set(np.unique(pid[tr2]).tolist()) <= set(np.flatnonzero(k2).tolist())
    print("  [PASS] known_from_k keeps whole questions and trains only on questions known from k")

    tr3, te3, _ = arm_rows("as_if_k", split, y, pid, is_known, s3, 42)
    assert set(tr3.tolist()) <= set(s3.tolist()) and set(te3.tolist()) <= set(s3.tolist())
    assert np.array_equal(np.unique(pid[tr3]), np.unique(pid[tr2]))
    print("  [PASS] as_if_k uses only the drawn answers, on the same questions as known_from_k")

    # End to end with the real detector on a tiny synthetic tensor: a planted signal must be found
    # by every arm, the reference must equal 61's hosvd arm exactly, and every label must appear.
    # The signal is a dense Kronecker pattern, as in 61's self-test: a single planted channel would be
    # flattened by the per-channel robust scaling and missed by the bases, which is a property of the
    # detector, not of this script.
    L, D = 4, 12
    ul = rng.standard_normal(L); ul /= np.linalg.norm(ul)
    uf = rng.standard_normal(D); uf /= np.linalg.norm(uf)
    feats = {"core": rng.standard_normal((len(y), L, D)).astype(np.float32) * 0.6}
    feats["core"] += (y[:, None, None] * 3.0) * np.outer(ul, uf)[None].astype(np.float32)
    spec = [("core", 2, 3)]
    res = evaluate(feats, y, pid, is_known, spec, [42, 0], (5, 3), "LR", c, log=lambda s: None)
    want = {"reference"} | {"%s_k%d" % (a, k) for k in (5, 3) for a in ARMS}
    assert set(res) == want, set(res) ^ want
    for label, rows in res.items():
        assert all(r["pooled_auroc"] > 0.9 for r in rows), (label, [r["pooled_auroc"] for r in rows])
    m = mods()
    ref = m["s61"]._score_one(m, c, "hosvd", feats, spec, tr0, te0, y, pid, "LR", 42, 6)
    assert ref["pooled_auroc"] == res["reference"][0]["pooled_auroc"]
    s = summarize(res)
    assert s["reference"]["delta_vs_reference_pts"] == 0.0
    print("  [PASS] end to end: all %d arm/k labels scored, reference identical to 61's hosvd arm"
          % len(res))

    print("\n  ALL PASS")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dataset")
    p.add_argument("--model_folder")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--readout", default="RF", choices=["RF", "LR"])
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--ks", nargs="+", type=int, default=list(KS))
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
    run(a.dataset, a.model_folder, data_dir, a.out_dir, a.readout, a.seeds, tuple(a.ks))


if __name__ == "__main__":
    main()
