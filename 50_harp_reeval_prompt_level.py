"""
50_harp_reeval_prompt_level.py -- re-score HARP's method under the split the PAPER describes
=====================================================================================================
THE EXPERIMENT
    HARP's released code splits the known group over individual ANSWERS; the paper describes a
    split over QUESTIONS ("the set of all inputs", S3.1). We hold everything else fixed -- same
    hidden states, same lm_head SVD, same projection, same MLP, same hyperparameters -- and vary
    ONLY the unit of the split. The difference is the leakage effect, measured rather than argued.

    arm "answer"    their split: torch.randperm over the flat answer list   (utils.split_data)
    arm "question"  paper split: 75/25 over known QUESTIONS, taken whole
    arm "groupkfold" our protocol: 5-fold OOF grouped by prompt, pooled AUROC over all data

    Arm "answer" is the CONTROL, and it matters more than it looks: if it does not reproduce the
    number their own main.py printed, this harness is wrong and arm "question" means nothing.
    Always read the two together.

NOTHING OF THEIRS IS REIMPLEMENTED OR MODIFIED
    We import their functions and call them:
        utils.get_proj        the reasoning-subspace projection
        utils.split_data      arm "answer" uses their splitter verbatim
        utils.seed_everything their seeding, so arm "answer" is seed-comparable to main.py
        mlp_trainer.train     their detector, their optimiser, their max-over-tokens scoring
        mlp_trainer.validate  their AUROC
    We never import main.py (its argparse runs at module level) and never import DatasetJudge
    (it pulls in bleurt_pytorch, which we do not need -- the judging step never runs here).

INPUTS -- all produced by their own main.py, nothing extra to compute
    HARP-Code/hidden_state/{model}/[known|unknown]{ds}.pt   per-token last-layer states + flags
    HARP-Code/svd/[{model}]unembedding_svd.pt               their lm_head SVD
    HARP-Code/dataset/[known|unknown]{ds}.jsonl             what we fed in; supplies prompt ids

RECOVERING THE QUESTION GROUPING
    Their .pt stores a FLAT list with no prompt id. main.py:184-204 builds it by walking the jsonl
    records in file order and, within each, that record's beams in order. We replay exactly that
    walk over the same jsonl to label every row with its prompt id, then hard-check the result
    against their stored hallucination flags element for element. A silent mis-alignment here
    would invent a leakage effect out of nothing, so it is an assertion, not a comment.

Usage:
  python 50_harp_reeval_prompt_level.py --self-test
  python 50_harp_reeval_prompt_level.py --dataset tydiqa --model_name Qwen2.5-7B-Instruct
  python 50_harp_reeval_prompt_level.py --dataset tydiqa --arms answer,question,groupkfold
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HARP = os.path.abspath(os.path.join(HERE, "..", "HARP-Code"))

PROJ_DIMS = [32, 64, 128, 192, 256, 512, 1024]   # main.py:59, unchanged
# Their S5.3 fixes the reasoning subspace dimension at 256 for BOTH models and ALL four datasets
# ("a dimension of 256 yields the best performance"), and every Table 1 value is reported at it.
# So 256 -- not the sweep maximum -- is the like-for-like cell.
PAPER_PROJ_DIM = 256
HIDDEN_DIM = 1024                                 # main.py:61
TRAIN_RATIO = 0.75                                # their default / README
LR, WEIGHT_DECAY, EPOCHS, BATCH = 1e-4, 3e-4, 50, 128   # main.py:286-288 + README
# We never call mlp_trainer.validate ourselves -- train() already reports valid AUROC, and it
# passes batch_size through (mlp_trainer.py:344), so the whole valid set is never padded at once.


# ==============================================================================
# SMALL PURE HELPERS (independently self-testable, no torch, no HARP, no files)
# ==============================================================================

def walk_records(records):
    """Replay main.py:184-204's iteration order over jsonl records.

    Returns (prompt_ids, hallucination_flags) aligned row-for-row with the flat list their
    get_hidden_state produced. Deliberately does NOT assume 10 beams per prompt -- it walks
    whatever each record actually holds, exactly as their loop does."""
    pids, flags = [], []
    for item in records:
        for beam in item["result"]:
            pids.append(int(item["ind"]))
            flags.append(not bool(beam["score"]["is_correct"]))
    return pids, flags


def split_by_question(pids, train_ratio, rng):
    """The paper's split: choose 75% of the distinct QUESTIONS, take all of their answers.

    Returns (train_rows, valid_rows) as row-index lists. A question lands wholly on one side --
    which is the entire point, and is asserted by the caller."""
    uniq = sorted(set(pids))
    order = rng.permutation(len(uniq))
    n_train = int(len(uniq) * train_ratio)
    train_q = {uniq[i] for i in order[:n_train]}
    train_rows = [r for r, p in enumerate(pids) if p in train_q]
    valid_rows = [r for r, p in enumerate(pids) if p not in train_q]
    return train_rows, valid_rows


def group_kfold_indices(pids, n_splits, rng):
    """Prompt-grouped K-fold: yields (train_rows, test_rows) with no prompt spanning the two."""
    uniq = sorted(set(pids))
    order = rng.permutation(len(uniq))
    folds = [set() for _ in range(n_splits)]
    for i, idx in enumerate(order):
        folds[i % n_splits].add(uniq[idx])
    for f in folds:
        test_rows = [r for r, p in enumerate(pids) if p in f]
        train_rows = [r for r, p in enumerate(pids) if p not in f]
        yield train_rows, test_rows


def questions_on_both_sides(pids, train_rows, valid_rows):
    """How many distinct questions have at least one answer on each side. Zero is the paper's
    protocol; a large number is the released code's."""
    a = {pids[r] for r in train_rows}
    b = {pids[r] for r in valid_rows}
    return len(a & b)


def summarise(per_seed):
    """mean/std over seeds, reported as percentages like their printout."""
    a = np.asarray(per_seed, dtype=float)
    return float(a.mean()), float(a.std())


# ==============================================================================
# HARP INTEROP
# ==============================================================================

def import_harp(harp_dir):
    """Import their modules without touching them. main.py is never imported: its
    parser.parse_args() sits at module level (main.py:40-49) and would eat our argv."""
    if not os.path.isdir(harp_dir):
        raise FileNotFoundError("HARP-Code not found at %s" % harp_dir)
    if harp_dir not in sys.path:
        sys.path.insert(0, harp_dir)
    import utils as harp_utils
    import mlp_trainer as harp_mlp
    return harp_utils, harp_mlp


def load_side(harp_dir, model_name, dataset, side):
    """One of [known]/[unknown]: their hidden states + our prompt ids, cross-checked."""
    import torch
    pt = os.path.join(harp_dir, "hidden_state", model_name, "[%s]%s.pt" % (side, dataset))
    js = os.path.join(harp_dir, "dataset", "[%s]%s.jsonl" % (side, dataset))
    for p in (pt, js):
        if not os.path.exists(p):
            raise FileNotFoundError(
                "%s not found. Run their main.py for this dataset first -- it writes the hidden "
                "states we re-score here (main.py:217-224)." % p)

    res = torch.load(pt, map_location="cpu", weights_only=True)
    keys = list(res["all_emb"].keys())
    assert len(keys) == 1, "expected exactly one hooked layer, got %s" % keys
    emb = res["all_emb"][keys[0]]
    stored_flags = [bool(f) for f in res["all_hallucination_flag"]]

    with open(js, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    pids, flags = walk_records(records)

    # --- the alignment must be exact; a silent shift here would fabricate a leakage effect ---
    assert len(pids) == len(emb), (
        "%s: jsonl implies %d answers but their hidden-state file holds %d. The two are out of "
        "sync -- do not trust any number computed from this pairing." % (side, len(pids), len(emb)))
    assert flags == stored_flags, (
        "%s: our replayed labels disagree with the flags stored alongside their hidden states, "
        "so the row order is not what main.py:184-204 produced." % side)
    return emb, pids, flags, keys[0]


def load_projection(harp_dir, model_name, device):
    """Their saved lm_head SVD (main.py:109-116). V's LAST columns are the reasoning subspace."""
    import torch
    p = os.path.join(harp_dir, "svd", "[%s]unembedding_svd.pt" % model_name)
    if not os.path.exists(p):
        raise FileNotFoundError("%s not found -- their main.py writes it on first run." % p)
    _, _, V = torch.load(p, map_location=device, weights_only=False)
    return V


def project(harp_utils, V, emb, proj_dim, device):
    """Their get_proj, their subspace choice. One tensor at a time to bound peak memory."""
    Vk = V[:, -proj_dim:]
    return [harp_utils.get_proj(V=Vk, emb=e.to(device), norm=None).clone() for e in emb]


def run_one(harp_mlp, sentences, flags_t, train_rows, valid_rows, proj_dim, seed, device, tmp):
    """Their train() on our row selection. Hyperparameters are theirs, verbatim."""
    import torch
    from matplotlib import pyplot as plt
    train_data = [(sentences[r], flags_t[r]) for r in train_rows]
    valid_data = [(sentences[r], flags_t[r]) for r in valid_rows]
    res = harp_mlp.train(
        train_data=train_data, valid_data=valid_data,
        input_dim=proj_dim, hidden_dim=HIDDEN_DIM,
        batch_size=BATCH, step_log=10 ** 9, epochs=EPOCHS, epoch_log=10 ** 9,
        max_grad_norm=1.0, random_seed=seed, lr=LR, weight_decay=WEIGHT_DECAY,
        device=device, auroc_img_save_folder=tmp)
    # their plt_auroc never closes its figures (mlp_trainer.py:227-240); across a sweep that
    # leaks. Closing from OUR side rather than editing theirs.
    plt.close("all")
    model = res.model
    auroc = float(res.valid_auroc)
    del res
    torch.cuda.empty_cache() if device.startswith("cuda") else None
    return auroc, model


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 74)
    print("  SELF-TEST: 50_harp_reeval_prompt_level (synthetic, no model, no cluster files)")
    print("=" * 74)

    recs = [{"ind": 5, "result": [{"score": {"is_correct": True}},
                                  {"score": {"is_correct": False}}]},
            {"ind": 9, "result": [{"score": {"is_correct": False}}]}]
    pids, flags = walk_records(recs)
    assert pids == [5, 5, 9], pids
    assert flags == [False, True, True], flags
    print("  [PASS] walk_records: main.py's record-then-beam order, is_correct -> hallucinated")

    ragged = [{"ind": 0, "result": [{"score": {"is_correct": True}}] * 3},
              {"ind": 1, "result": [{"score": {"is_correct": True}}] * 7}]
    assert walk_records(ragged)[0] == [0] * 3 + [1] * 7, "must not assume 10 beams per prompt"
    print("  [PASS] walk_records: ragged beam counts handled (no stride-10 assumption)")

    rng = np.random.default_rng(0)
    pids4 = [q for q in range(40) for _ in range(10)]
    tr, va = split_by_question(pids4, 0.75, rng)
    assert len(tr) + len(va) == len(pids4) and not (set(tr) & set(va))
    assert questions_on_both_sides(pids4, tr, va) == 0, "question split must leave NO overlap"
    assert len({pids4[r] for r in tr}) == 30, "75% of 40 questions is 30"
    print("  [PASS] split_by_question: partitions all rows, zero questions on both sides")

    folds = list(group_kfold_indices(pids4, 5, np.random.default_rng(1)))
    assert len(folds) == 5
    seen = []
    for trr, ter in folds:
        assert questions_on_both_sides(pids4, trr, ter) == 0
        seen.extend(ter)
    assert sorted(seen) == list(range(len(pids4))), "every row must be tested exactly once"
    print("  [PASS] group_kfold_indices: 5 disjoint prompt-grouped folds, full OOF coverage")

    m, s = summarise([0.90, 0.92, 0.94])
    assert abs(m - 0.92) < 1e-9 and s > 0
    print("  [PASS] summarise: mean/std over seeds")

    # ---- the real test: data whose ONLY signal is question identity ----------------------
    # Each question gets a difficulty; the embedding encodes the difficulty and nothing about
    # the individual answer. A detector can only score above chance by recognising the question.
    # Under the answer-level split it can (it trained on other answers to the same question);
    # under the question-level split it cannot. If this harness reports the two as equal, it is
    # not actually separating the protocols and every number it produces is meaningless.
    try:
        import torch
        harp_utils, harp_mlp = import_harp(DEFAULT_HARP)
    except Exception as e:
        print("\n  [SKIP] planted-leakage check needs HARP-Code + torch importable (%s: %s)"
              % (type(e).__name__, e))
        print("\n[PASS] All runnable self-test assertions passed.")
        return

    rng = np.random.default_rng(7)
    D, NQ, NB = 24, 80, 10
    # Each question gets a RANDOM signature that carries no information about its difficulty.
    # This is the crux: the signature must be memorizable but NOT generalizable. An earlier
    # version of this test encoded difficulty along a single fixed direction -- which a model
    # learns as a rule that transfers to unseen questions, so both splits scored the same and
    # there was no leakage to detect. Signature and difficulty must be independent.
    sig = rng.normal(size=(NQ, D)) * 3.0
    diff = rng.choice([0.1, 0.9], size=NQ)             # per-question difficulty, independent
    sents, labels, pids_s = [], [], []
    for q in range(NQ):
        for _ in range(NB):
            T = int(rng.integers(4, 9))
            base = np.tile(sig[q], (T, 1))
            sents.append(torch.tensor(base + rng.normal(scale=0.5, size=(T, D)),
                                      dtype=torch.float32))
            labels.append(bool(rng.uniform() < diff[q]))
            pids_s.append(q)
    flags_t = torch.tensor(labels)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sents = [s.to(dev) for s in sents]
    flags_t = flags_t.to(dev)
    tmp = tempfile.mkdtemp(prefix="harp_selftest_")
    try:
        r = np.random.default_rng(3)
        tr_q, va_q = split_by_question(pids_s, 0.75, r)
        n = len(sents)
        perm = np.random.default_rng(3).permutation(n)
        cut = int(n * 0.75)
        tr_a, va_a = list(perm[:cut]), list(perm[cut:])

        assert questions_on_both_sides(pids_s, tr_a, va_a) > 0.8 * NQ, \
            "answer-level split should put most questions on both sides"
        assert questions_on_both_sides(pids_s, tr_q, va_q) == 0

        kw = dict(harp_mlp=harp_mlp, sentences=sents, flags_t=flags_t, proj_dim=D,
                  seed=42, device=dev, tmp=tmp)
        auroc_a, _ = run_one(train_rows=tr_a, valid_rows=va_a, **kw)
        auroc_q, _ = run_one(train_rows=tr_q, valid_rows=va_q, **kw)
        print("\n  planted question-identity-only signal:")
        print("    answer-level split  (leaky) : AUROC %.3f" % auroc_a)
        print("    question-level split (clean): AUROC %.3f" % auroc_q)
        assert auroc_a > 0.65, \
            "answer-level split should exploit question identity, got %.3f" % auroc_a
        assert auroc_a - auroc_q > 0.10, (
            "the two protocols must separate on data whose only signal is question identity "
            "(answer %.3f vs question %.3f) -- otherwise this harness cannot measure leakage"
            % (auroc_a, auroc_q))
        print("  [PASS] planted leakage is visible to the answer split and hidden from the "
              "question split")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tydiqa",
                    help="HARP's dataset key: truthful_qa | trivia_qa | nq_open | tydiqa")
    ap.add_argument("--model_name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--harp-dir", default=DEFAULT_HARP)
    ap.add_argument("--arms", default="answer,question")
    ap.add_argument("--proj-dims", default=",".join(str(d) for d in PROJ_DIMS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5, help="groupkfold arm only")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "harp_reeval"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    dims = [int(s) for s in a.proj_dims.split(",") if s.strip()]
    harp_utils, harp_mlp = import_harp(a.harp_dir)

    print("=" * 78)
    print("  HARP re-evaluation -- %s / %s   device=%s" % (a.model_name, a.dataset, device))
    print("  arms=%s  proj_dims=%s  seeds=%d" % (arms, dims, a.seeds))
    print("  Reads only: %s   Writes only: %s" % (a.harp_dir, a.out_dir))
    print("=" * 78, flush=True)

    emb_k, pid_k, fl_k, layer = load_side(a.harp_dir, a.model_name, a.dataset, "known")
    emb_u, pid_u, fl_u, layer_u = load_side(a.harp_dir, a.model_name, a.dataset, "unknown")
    assert layer == layer_u, "known/unknown were hooked at different layers: %s vs %s" % (
        layer, layer_u)
    n_k = len(emb_k)
    emb_all = emb_k + emb_u
    pid_all = pid_k + pid_u
    fl_all = fl_k + fl_u
    assert not (set(pid_k) & set(pid_u)), \
        "a prompt appears in both the known and unknown files -- the grouping is corrupt"
    print("  layer hooked: %s" % layer)
    print("  known:   %5d answers / %4d questions" % (n_k, len(set(pid_k))))
    print("  unknown: %5d answers / %4d questions" % (len(emb_u), len(set(pid_u))))
    print("  hallucinated overall: %.1f%%" % (100.0 * np.mean(fl_all)), flush=True)

    V = load_projection(a.harp_dir, a.model_name, device)
    flags_all = torch.tensor(fl_all).to(device)
    tmp = tempfile.mkdtemp(prefix="harp_reeval_")
    results = {}

    try:
        for pd_ in dims:
            print("\n--- proj_dim %d ---" % pd_, flush=True)
            sents = project(harp_utils, V, emb_all, pd_, device)

            for arm in arms:
                per_seed, overlaps = [], []
                for s in range(a.seeds):
                    seed = 42 + s
                    harp_utils.seed_everything(seed=seed)
                    rng = np.random.default_rng(seed)

                    if arm == "groupkfold":
                        # Prompt-grouped 5-fold. NOTE: this is the MEAN OF PER-FOLD AUROCs, not
                        # the pooled-OOF AUROC our own pipeline reports. Pooling would need every
                        # held-out row's raw score, and their validate() returns only an
                        # AurocResult -- extracting per-row scores would mean rewriting their
                        # max-over-tokens scoring, which is exactly what we refuse to do. So this
                        # arm is indicative, and is NOT drop-in comparable with the master table.
                        fold_aurocs = []
                        for trr, ter in group_kfold_indices(pid_all, a.folds, rng):
                            assert questions_on_both_sides(pid_all, trr, ter) == 0
                            auroc_f, _ = run_one(harp_mlp, sents, flags_all, trr, ter,
                                                 pd_, seed, device, tmp)
                            fold_aurocs.append(auroc_f)
                        per_seed.append(float(np.mean(fold_aurocs)))
                        overlaps.append(0)
                        continue

                    if arm == "answer":
                        # their splitter, verbatim, over the known rows only
                        rows_known = list(range(n_k))
                        tr_idx, va_idx = harp_utils.split_data(
                            rows_known, train_ratio=TRAIN_RATIO, shuffle=True)
                        tr = list(tr_idx)
                        va = list(va_idx) + list(range(n_k, len(sents)))
                    elif arm == "question":
                        tr_k, va_k = split_by_question(pid_k, TRAIN_RATIO, rng)
                        tr = tr_k
                        va = va_k + list(range(n_k, len(sents)))
                    else:
                        raise ValueError("unknown arm %r" % arm)

                    assert not (set(tr) & set(va)), "train/valid rows overlap"
                    overlaps.append(questions_on_both_sides(pid_all, tr, va))
                    auroc, _ = run_one(harp_mlp, sents, flags_all, tr, va,
                                       pd_, seed, device, tmp)
                    per_seed.append(auroc)

                m, sd = summarise(per_seed)
                results.setdefault(arm, {})[pd_] = {
                    "mean": m, "std": sd, "per_seed": per_seed,
                    "questions_on_both_sides": overlaps}
                print("  %-11s AUROC %.4f +/- %.4f   (questions on both sides: %s)"
                      % (arm, m, sd, overlaps[0] if overlaps else "n/a"), flush=True)

            del sents
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "%s_%s_reeval.json" % (a.model_name, a.dataset))
    with open(out, "w") as f:
        json.dump({"model": a.model_name, "dataset": a.dataset, "layer": layer,
                   "n_known_answers": n_k, "n_unknown_answers": len(emb_u),
                   "n_known_questions": len(set(pid_k)),
                   "n_unknown_questions": len(set(pid_u)),
                   "arms": results}, f, indent=2)
    print("\nWrote: %s" % out)

    print("\n" + "=" * 78)
    print("  SUMMARY")
    # 256 is THE number to compare against the paper. Their S5.3 fixes the reasoning subspace
    # dimension at 256 globally -- across both models and all four datasets -- and every value in
    # their Table 1 is reported at it. They do not take a per-dataset maximum over the sweep, so
    # neither should we when quoting a like-for-like figure.
    print("  at proj_dim 256 (the paper's fixed setting, S5.3 -- use this for comparison):")
    for arm in arms:
        if arm in results and PAPER_PROJ_DIM in results[arm]:
            r = results[arm][PAPER_PROJ_DIM]
            print("    %-11s %.2f%% +/- %.2f" % (arm, 100 * r["mean"], 100 * r["std"]))
    if all(a in results and PAPER_PROJ_DIM in results[a] for a in ("answer", "question")):
        d = (results["answer"][PAPER_PROJ_DIM]["mean"]
             - results["question"][PAPER_PROJ_DIM]["mean"])
        print("    delta (answer - question): %+.2f points" % (100 * d))
        print("    Read this ONLY if the 'answer' arm reproduces their main.py number.")

    print("\n  best over the sweep (context only -- NOT the paper's reported quantity):")
    for arm in arms:
        if arm not in results:
            continue
        best = max(results[arm].items(), key=lambda kv: kv[1]["mean"])
        print("    %-11s %.2f%%  at proj_dim %d" % (arm, 100 * best[1]["mean"], best[0]))
    print("=" * 78)


if __name__ == "__main__":
    main()
