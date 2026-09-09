"""
62_act_vit_joint.py -- ACT-ViT's multi-source variant, trained across our model/dataset combinations.
=====================================================================================================
WHY. What we report so far is ACT-ViT(s), their SINGLE-dataset variant. Their headline ACT-ViT is one
model trained jointly over fifteen LLM-dataset combinations, with a per-LLM linear adapter feeding a
shared ViT backbone, and their Table 1 has it ahead of ACT-ViT(s) in twelve of fifteen cases. We
cannot reproduce fifteen: four of their five datasets are not in our pipeline, and of their fifteen
combinations exactly one (Qwen2.5-7B-Instruct x TriviaQA) is reproducible here. Their LLaMA is
3-8B-Instruct; ours is 3.1-8B base.

What we CAN do is run their joint recipe over OUR combinations -- two models x two datasets. That is
not their result, and this file does not pretend otherwise. It answers the question a reviewer will
ask ("you compared against the weaker variant") with a measurement rather than a paragraph.

EXPECT IT TO SHOW LITTLE. Their gain comes from corpus diversity: three model families and five task
types including sentiment classification and movie QA. Ours is two model families and two QA
datasets. If the joint arm does not beat ACT-ViT(s) here, that is consistent with the gain coming
from diversity we cannot reproduce, and it is a reportable result either way.

WHAT IS SHARED AND WHAT IS NOT. One ViT backbone, trained once per split over the union of every
combination's training rows. Each row carries the index of its source model, so their
ModuleListPerLLMLinear routes it to that model's adapter -- this is the first place in our pipeline
where that machinery is used as designed rather than pinned to a constant. Feature widths differ
(Qwen 3584, LLaMA 4096) so the arrays are kept separate and gathered per batch, then zero-padded to
their adapter width of 4096.

SPLITS ARE PER COMBINATION AND THEN UNIONED. Each combination is split on its own questions using the
canonical protocol, the training halves are concatenated, and the model is scored on each
combination's test half separately. A single global split would let one dataset's questions leak into
another's training set, which is meaningless here but would still be wrong.

ROW ORDER. Splits are computed from the extraction's own prompt_id, which 59 wrote alongside the
rows, so no reordering against the pinned generations is needed and none is done.

SANITY CHECK. Run with a single combination and the result should track
results/methods/act_vit_<model>_<dataset>.json. If it does not, the joint path has a bug the
self-test did not catch.

Usage:
  python 62_act_vit_joint.py --self-test
  python 62_act_vit_joint.py --combos qwen-2.5-7b-instruct:tydiqa_gp qwen-2.5-7b-instruct:truthfulqa \
                                      llama-3.1-8b:tydiqa_gp llama-3.1-8b:truthfulqa
  python 62_act_vit_joint.py --combos qwen-2.5-7b-instruct:tydiqa_gp     # single, as a check
"""

import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AT_DIR = os.path.abspath(os.path.join(HERE, "..", "data-acttensors"))
DEFAULT_OUT = os.path.join(HERE, "results", "act_vit_joint")


def _av():
    """methods/act_vit.py holds the architecture import, the recipe constants and pad_features.
    Imported rather than duplicated so the joint arm cannot drift from the single-dataset one."""
    import methods.act_vit as A
    return A


class Combo:
    """One (model, dataset) source: its activation tensors, labels and question ids."""

    def __init__(self, model_folder, dataset, at_dir, l_eff, n_eff):
        self.model_folder, self.dataset = model_folder, dataset
        path = os.path.join(at_dir, model_folder,
                            "%s_at_L%d_N%d.npz" % (dataset, l_eff, n_eff))
        if not os.path.exists(path):
            raise SystemExit(
                "%s not found. Extract it first:\n"
                "    python 59_extract_act_tensors.py --dataset %s --model_folder %s --n-pool %d"
                % (path, dataset, model_folder, n_eff))
        z = np.load(path)
        self.at = z["at"]
        self.y = np.asarray(z["label"], dtype=int)
        self.pid = np.asarray(z["prompt_id"])
        self.path = path
        self.n = len(self.y)
        if self.at.shape[0] != self.n:
            raise SystemExit("%s: %d rows of activations against %d labels"
                             % (path, self.at.shape[0], self.n))

    @property
    def key(self):
        return "%s/%s" % (self.model_folder, self.dataset)

    def __repr__(self):
        return ("Combo(%s, %d answers, %d questions, %.1f%% hallucinated, %s, %.1f GB)"
                % (self.key, self.n, len(np.unique(self.pid)), 100.0 * self.y.mean(),
                   self.at.shape[1:], self.at.nbytes / 1024 ** 3))


def gather(combos, global_rows, index):
    """Rows from the global index space -> (padded batch, per-row adapter index).

    Widths differ across models, so the arrays cannot be concatenated. Rows are grouped by source,
    gathered from each source's own array, padded to the adapter width and stacked in the order the
    caller asked for -- getting that order wrong would pair activations with the wrong labels."""
    import torch
    A = _av()
    src, loc = index["src"][global_rows], index["loc"][global_rows]
    out = torch.empty((len(global_rows), combos[0].at.shape[1], combos[0].at.shape[2],
                       A.FEATURE_DIM_MAX), dtype=torch.float32)
    idx = torch.empty(len(global_rows), dtype=torch.long)
    for ci in np.unique(src):
        m = src == ci
        x = torch.from_numpy(np.asarray(combos[ci].at[loc[m]], dtype=np.float32))
        out[torch.from_numpy(np.flatnonzero(m))] = A.pad_features(x)
        idx[torch.from_numpy(np.flatnonzero(m))] = int(
            A.LLM_INDEX.get(combos[ci].model_folder, 0))
    return out, idx


def build_index(combos):
    """Flat row space over every combination, with the source and within-source row of each."""
    src = np.concatenate([np.full(c.n, i, dtype=np.int64) for i, c in enumerate(combos)])
    loc = np.concatenate([np.arange(c.n, dtype=np.int64) for c in combos])
    off = np.cumsum([0] + [c.n for c in combos])
    return {"src": src, "loc": loc, "offset": off, "n": int(off[-1])}


def per_combo_split(combos, index, split_fn, derive_is_known, seed):
    """Split EACH combination on its own questions, then union the training halves.

    A single global split would mix questions across datasets. That cannot leak anything meaningful,
    since the datasets are disjoint, but it would no longer be the protocol we report."""
    tr, te = [], []
    for i, c in enumerate(combos):
        is_known = derive_is_known(c.y, c.pid)
        t, v = split_fn(is_known, c.pid, c.n, seed)
        tr.append(np.asarray(t, dtype=np.int64) + index["offset"][i])
        te.append(np.asarray(v, dtype=np.int64) + index["offset"][i])
    return np.concatenate(tr), te


def train_joint(combos, index, tr, seed, epochs, device, log=None):
    """One backbone over the union of training rows. Returns the fitted model."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import get_scheduler
    A = _av()

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Validation carved from TRAINING rows, whole questions at a time, per combination so no
    # dataset is left out of the early-stopping signal.
    fit, val = [], []
    for i, c in enumerate(combos):
        loc = index["loc"][tr[index["src"][tr] == i]]
        f, v = A.grouped_val_split(c.pid, loc, seed)
        fit.append(f + index["offset"][i])
        if v is not None:
            val.append(v + index["offset"][i])
    fit = np.concatenate(fit)
    val = np.concatenate(val) if val else None

    l_p, n_p = combos[0].at.shape[1], combos[0].at.shape[2]
    model = A.build_model(l_p, n_p, device)
    opt = torch.optim.AdamW(model.parameters(), lr=A.LR, weight_decay=A.WEIGHT_DECAY)
    steps = max(1, int(np.ceil(len(fit) / A.BATCH))) * epochs
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=int(0.1 * steps),
                          num_training_steps=steps)
    criterion = torch.nn.BCELoss()
    labels = np.concatenate([c.y for c in combos])

    def loader(rows, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(rows.astype(np.int64))),
                          batch_size=A.BATCH, shuffle=shuffle)

    def forward(rows):
        x, li = gather(combos, rows, index)
        return model(x.to(device), li.to(device)).reshape(-1)

    best, best_state, stale = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for (b,) in loader(fit, True):
            rows = b.numpy()
            opt.zero_grad()
            loss = criterion(forward(rows),
                             torch.from_numpy(labels[rows].astype(np.float32)).to(device))
            loss.backward()
            opt.step()
            sched.step()
        if val is None:
            continue
        model.eval()
        with torch.no_grad():
            vs = np.concatenate([forward(b.numpy()).cpu().numpy() for (b,) in loader(val, False)])
        a = A._auc(labels[val], vs)
        if log and (ep == 0 or ep == epochs - 1):
            log("      epoch %2d/%d  val AUROC %.4f  (%d fit / %d val rows)"
                % (ep + 1, epochs, a, len(fit), len(val)))
        if np.isnan(a) or a <= best:
            stale += 1
            if stale >= A.PATIENCE:
                break
        else:
            best, stale = a, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def score(model, combos, index, rows, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    A = _av()
    out = []
    with torch.no_grad():
        for (b,) in DataLoader(TensorDataset(torch.from_numpy(rows.astype(np.int64))),
                               batch_size=A.BATCH, shuffle=False):
            x, li = gather(combos, b.numpy(), index)
            out.append(model(x.to(device), li.to(device)).reshape(-1).cpu().numpy())
    return np.concatenate(out)


def run(combo_specs, at_dir, out_dir, l_eff, n_eff, epochs, device, tag=None):
    import methods.base as B
    c = B.canonical()
    A = _av()

    combos = [Combo(m, d, at_dir, l_eff, n_eff) for m, d in combo_specs]
    widths = {cb.at.shape[1:3] for cb in combos}
    if len(widths) != 1:
        raise SystemExit("all combinations must share (L_p, N_p); got %s. Re-extract at one "
                         "pooling." % widths)
    index = build_index(combos)
    for cb in combos:
        print("  %r" % cb, flush=True)
    print("  joint: %d rows, %.1f GB resident, adapter indices %s"
          % (index["n"], sum(cb.at.nbytes for cb in combos) / 1024 ** 3,
             {cb.key: A.LLM_INDEX.get(cb.model_folder, 0) for cb in combos}), flush=True)

    res = {cb.key: {"question": [], "answer": []} for cb in combos}
    t0 = time.time()
    for arm, fn in (("question", c["question_split"]), ("answer", c["answer_split"])):
        for seed in c["seeds"]:
            t1 = time.time()
            tr, te_list = per_combo_split(combos, index, fn, c["derive_is_known"], seed)
            model = train_joint(combos, index, tr, seed, epochs, device, log=print)
            for i, cb in enumerate(combos):
                s = score(model, combos, index, te_list[i], device)
                loc = index["loc"][te_list[i]]
                y, pid = cb.y[loc], cb.pid[loc]
                f = np.isfinite(s)
                wp = c["within_prompt_auroc"](s[f], y[f], pid[f])
                res[cb.key][arm].append({
                    "seed": int(seed),
                    "pooled_auroc": float(c["pooled_auroc"](s[f], y[f])),
                    "within_prompt_auroc": wp["within_prompt_auroc"],
                    "n_test_rows": int(len(te_list[i])), "n_non_finite": int((~f).sum())})
            print("    %-8s seed %-3d  %s  (%.0fs)"
                  % (arm, seed,
                     "  ".join("%s %.4f" % (cb.dataset[:6], res[cb.key][arm][-1]["pooled_auroc"])
                               for cb in combos), time.time() - t1), flush=True)

    summary = {}
    for cb in combos:
        summary[cb.key] = {}
        for arm in ("question", "answer"):
            p = np.array([r["pooled_auroc"] for r in res[cb.key][arm]])
            summary[cb.key][arm] = {"pooled_mean": float(p.mean()), "pooled_std": float(p.std()),
                                    "per_seed": res[cb.key][arm]}
        q, a = summary[cb.key]["question"]["pooled_mean"], summary[cb.key]["answer"]["pooled_mean"]
        summary[cb.key]["leakage_pts"] = round(100.0 * (a - q), 2)

    print("\n  %-34s %10s %10s %8s" % ("", "question", "answer", "leakage"))
    for cb in combos:
        s = summary[cb.key]
        print("  %-34s %10.4f %10.4f %+8.2f"
              % (cb.key, s["question"]["pooled_mean"], s["answer"]["pooled_mean"],
                 s["leakage_pts"]))

    os.makedirs(out_dir, exist_ok=True)
    stem = "act_vit_joint_%dcombo_N%d%s" % (len(combos), n_eff, ("_" + tag) if tag else "")
    dst = os.path.join(out_dir, stem + ".json")
    with open(dst, "w") as f:
        json.dump({"combos": [cb.key for cb in combos],
                   "sources": {cb.key: cb.path for cb in combos},
                   "l_eff": l_eff, "n_eff": n_eff, "epochs": epochs,
                   "n_rows_total": index["n"], "seeds": c["seeds"],
                   "config": A.ActViT().meta()["config"],
                   "note": ("ACT-ViT's multi-source variant over OUR combinations. Not their "
                            "ACT-ViT, which is trained over fifteen LLM-dataset combinations "
                            "spanning three model families and five task types."),
                   "summary": summary,
                   "elapsed_seconds": round(time.time() - t0, 1)}, f, indent=2)
    print("\n  wrote %s" % dst)
    return summary


def self_test():
    print("=" * 78)
    print("  SELF-TEST: 62_act_vit_joint")
    print("=" * 78)
    import torch
    rng = np.random.default_rng(0)
    A = _av()

    class Fake:
        def __init__(self, key, n, D, sig, seed):
            self.model_folder, self.dataset = key.split("/")
            r = np.random.default_rng(seed)
            self.n = n
            self.y = np.array([0, 1] * (n // 2))
            self.pid = np.repeat(np.arange(n // 2), 2)
            self.at = (r.standard_normal((n, 8, 4, D)).astype(np.float32) * 0.05).astype(np.float32)
            self.at[self.y == 1] += sig
            self.path = "<synthetic>"
        key = property(lambda s: "%s/%s" % (s.model_folder, s.dataset))

    # Two sources with DIFFERENT feature widths, which is the case that made this file necessary.
    combos = [Fake("qwen-2.5-7b-instruct/a", 64, 3584, 3.0, 1),
              Fake("llama-3.1-8b/b", 48, 4096, 3.0, 2)]
    index = build_index(combos)
    assert index["n"] == 112 and list(index["offset"]) == [0, 64, 112]
    print("  [PASS] flat index over 2 sources of differing width: 112 rows, offsets [0, 64, 112]")

    # gather must return rows in the ORDER ASKED, padded to the adapter width, with each row
    # carrying its own source's adapter index. Order is the part that would silently misalign
    # activations against labels.
    ask = np.array([100, 3, 70, 0])          # deliberately interleaved across sources
    x, li = gather(combos, ask, index)
    assert x.shape == (4, 8, 4, A.FEATURE_DIM_MAX), x.shape
    assert list(li.numpy()) == [A.LLM_INDEX["llama-3.1-8b"], A.LLM_INDEX["qwen-2.5-7b-instruct"],
                               A.LLM_INDEX["llama-3.1-8b"], A.LLM_INDEX["qwen-2.5-7b-instruct"]]
    ref = torch.from_numpy(np.asarray(combos[0].at[3], dtype=np.float32))
    assert torch.allclose(x[1, :, :, :3584], ref) and torch.all(x[1, :, :, 3584:] == 0)
    print("  [PASS] gather preserves request order, pads to 4096, routes per-row adapter indices")

    # Splits are per combination and must not leak: a test row of one source must never appear in
    # the union of training rows.
    import methods.base as B
    c = B.canonical()
    tr, te = per_combo_split(combos, index, c["question_split"], c["derive_is_known"], 42)
    for i, t in enumerate(te):
        assert not np.intersect1d(tr, t).size, "combo %d has test rows inside the joint train set" % i
        assert set(index["src"][t].tolist()) == {i}, "test rows of combo %d span sources" % i
    print("  [PASS] per-combination splits: no test row enters the joint training set, and each "
          "test set stays within its own source")

    if not os.path.isdir(A.ACT_REPO):
        print("  [SKIP] ../ACT-ViT not cloned -- joint training not exercised")
        print("\n  ALL PASS")
        return True

    # End to end. Both sources carry the same separable signal, so a correctly wired joint model
    # must score high on BOTH test sets. A model that routed every row through one adapter, or
    # misaligned rows against labels, would not.
    model = train_joint(combos, index, tr, seed=0, epochs=4, device="cpu")
    aucs = []
    for i, cb in enumerate(combos):
        s = score(model, combos, index, te[i], "cpu")
        loc = index["loc"][te[i]]
        aucs.append(A._auc(cb.y[loc], s))
    assert all(a > 0.9 for a in aucs), (
        "joint training did not learn a separable signal on both sources: %s. High score must mean "
        "hallucinated, and both adapters must be reached." % aucs)
    print("  [PASS] joint training scores %.3f / %.3f on the two held-out sources, orientation "
          "correct" % (aucs[0], aucs[1]))

    print("\n  ALL PASS")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--combos", nargs="*", default=[],
                   help="model_folder:dataset, one per source")
    p.add_argument("--at-dir", default=DEFAULT_AT_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--l-eff", type=int, default=8)
    p.add_argument("--n-eff", type=int, default=20)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default=None)
    a = p.parse_args()

    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not a.combos:
        raise SystemExit("--combos is required, e.g. "
                         "--combos qwen-2.5-7b-instruct:tydiqa_gp llama-3.1-8b:tydiqa_gp")
    specs = []
    for s in a.combos:
        if ":" not in s:
            raise SystemExit("combo %r must be model_folder:dataset" % s)
        specs.append(tuple(s.split(":", 1)))
    epochs = a.epochs if a.epochs is not None else _av().EPOCHS
    run(specs, a.at_dir, a.out_dir, a.l_eff, a.n_eff, epochs, a.device, a.tag)


if __name__ == "__main__":
    main()
