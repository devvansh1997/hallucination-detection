"""
83_processbench_steps.py -- does reading the earlier steps IN ORDER help spot the first wrong step? (T-026)
==================================================================================================

DATA. ProcessBench (Zheng et al., 2024): 3,400 math solutions (GSM8K, MATH, OlympiadBench, Omni-MATH) written
by 12 open models and cut into steps; human experts marked the first wrong step (or none). Qwen2.5-7B-Instruct
reads every solution as text after the problem (a reader, not the writer). Each step is kept as the mean of
its tokens' states (blocks 15..23, projected onto 64 directions per block as in 74 and 77).

THE CHECK. Every step up to and including the first wrong one is an item: label 1 for the first wrong step,
0 for a correct step before it (steps after the first error carry no label and are never read). Three readers
score each item:

    probe                 logistic regression on the step's own state
    GRU, real order       reads steps 0..k in order, scores step k from its own state plus the GRU's summary
    GRU, shuffled order   reads the same steps, steps 0..k-1 in a random order, then step k

Both GRUs see exactly the same steps; only the order of the earlier ones differs. Each GRU's score has a direct
path from step k's own state, so it is the probe plus context and cannot fall below the probe by underfitting.

SCORE. Same-position AUROC: a first wrong step at position k is compared only with correct steps at position k
(of any solution), positions 10 and later pooled. Errors happen later in long solutions, so comparing across
positions would reward knowing the position; within a position it cannot help. 95% intervals resample
solutions. 5 folds of whole problems.

READING. Real order above shuffled: the order of the earlier steps helps spot the error. Tie: the step's own
state and the set of steps before it are all the readers use.

  python 83_processbench_steps.py --self-test
  python 83_processbench_steps.py --prefetch          # dataset into the HF cache; no GPU (login node)
  python 83_processbench_steps.py                     # read the solutions once (GPU), then the check
"""

import argparse
import hashlib
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "token_structure")
R = 64
N_BLOCKS = 9
N_FIT = 40000
FIT_POOL = 60000
N_BASIS = 200              # solutions read only to fit the scaler and basis (no labels used), never evaluated
MAX_LEN = 4096
N_FOLDS = 5
MAX_GROUP = 10             # positions 10 and later share one comparison group
HIDDEN = 64
EPOCHS = 15
BATCH = 64
LR = 1e-3
SEEDS = (0, 1, 2)
N_BOOT = 2000


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pb_cfg():
    import yaml
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    return cfg, cfg["processbench"]


def store_dir():
    cfg, pc = pb_cfg()
    return os.path.normpath(os.path.join(HERE, pc["out_dir"], pc["reader"]))


def load_rows(pc):
    from datasets import load_dataset
    rows = []
    for split in pc["splits"]:
        for r in load_dataset(pc["hf_path"], split=split):
            rows.append({"id": str(r["id"]), "split": split, "generator": str(r["generator"]),
                         "problem": r["problem"], "steps": list(r["steps"]), "label": int(r["label"])})
    return rows


# ---------------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------------

def step_starts(steps, sep="\n\n"):
    """Character position where each step starts in sep.join(steps)."""
    starts, pos = [], 0
    for s in steps:
        starts.append(pos)
        pos += len(s) + len(sep)
    return starts


def token_steps(offset_mapping, starts):
    """Step of each token: the last step that starts at or before the token's first character. The separator
    after a step belongs to that step."""
    c0 = np.array([a for a, _ in offset_mapping], dtype=np.int64)
    return np.searchsorted(np.asarray(starts), c0, side="right") - 1


def build_ids(tok, problem, steps):
    head = tok.apply_chat_template([{"role": "user", "content": problem}], tokenize=False, add_generation_prompt=True)
    p = list(tok(head, add_special_tokens=False)["input_ids"])
    enc = tok("\n\n".join(steps), add_special_tokens=False, return_offsets_mapping=True)
    return p + list(enc["input_ids"]), len(p), token_steps(enc["offset_mapping"], step_starts(steps))


def step_means(Z, tstep, n_steps):
    """(n_steps, P, r) mean of the projected token states of each step, and the token count per step."""
    cnt = np.bincount(tstep, minlength=n_steps)
    out = np.zeros((n_steps,) + Z.shape[1:], dtype=np.float64)
    np.add.at(out, tstep, Z)
    return (out / np.maximum(cnt, 1)[:, None, None]).astype(np.float32), cnt


def extract(device, seed):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg, pc = pb_cfg()
    out_dir = store_dir()
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        raise SystemExit("refusing: %s is complete already" % out_dir)
    model_id = next((m["id"] for m in cfg["models"] if m["folder"] == pc["reader"]), None)
    if model_id is None:
        raise SystemExit("unknown reader %r" % pc["reader"])
    rows = load_rows(pc)
    tok = AutoTokenizer.from_pretrained(model_id)
    plan, skipped = [], {"too_long": 0, "empty_step": 0}
    for i, r in enumerate(rows):
        ids, npr, ts = build_ids(tok, r["problem"], r["steps"])
        if len(ids) > MAX_LEN:
            skipped["too_long"] += 1
            continue
        if len(ts) == 0 or np.bincount(ts, minlength=len(r["steps"])).min() == 0 or ts.min() < 0:
            skipped["empty_step"] += 1
            continue
        plan.append((i, ids, npr, ts))
    n_steps = np.array([len(rows[i]["steps"]) for i, _, _, _ in plan], dtype=np.int64)
    step_off = np.zeros(len(plan) + 1, dtype=np.int64)
    np.cumsum(n_steps, out=step_off[1:])
    print("  %d solutions, %d to read (skipped %s), %d steps | response tokens median %d"
          % (len(rows), len(plan), skipped, int(step_off[-1]), int(np.median([len(ids) - npr for _, ids, npr, _ in plan]))),
          flush=True)

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model.eval()
    first = int(pc["first_block"])
    layer_idx = list(range(first + 1, first + 1 + N_BLOCKS))
    s28 = _load("s28", "28_eval_band.py")
    s73 = _load("s73", "73_token_structure.py")
    s74 = _load("s74", "74_extract_ragtruth.py")

    rng = np.random.default_rng(seed)
    basis = np.zeros(len(plan), dtype=bool)
    basis[rng.choice(len(plan), size=min(N_BASIS, len(plan)), replace=False)] = True
    fit_H, fit_len, n_read = [], [], 0
    t0 = time.time()
    for k in rng.permutation(np.flatnonzero(basis)):
        _, ids, npr, _ = plan[k]
        H, _ = s74.read_response(model, ids, npr, layer_idx, device)
        fit_H.append(H.astype(np.float16))
        fit_len.append(len(H))
        n_read += len(H)
        if n_read >= FIT_POOL:
            break
    fit_tokens = np.concatenate(fit_H, axis=0)
    del fit_H
    fit_off = np.zeros(len(fit_len) + 1, dtype=np.int64)
    np.cumsum(fit_len, out=fit_off[1:])
    space, share, used = s73.fit_space(fit_tokens, fit_off, np.arange(len(fit_len)), s28, N_FIT, R, seed)
    del fit_tokens
    print("  fit pass: %d basis solutions read, %d tokens, fitted on %d | variance kept %s (%.0fs)"
          % (len(fit_len), n_read, used, " ".join("%.3f" % v for v in share), time.time() - t0), flush=True)

    states = np.zeros((int(step_off[-1]), N_BLOCKS, R), dtype=np.float32)
    step_tokens = np.zeros(int(step_off[-1]), dtype=np.int64)
    t0 = time.time()
    for k, (i, ids, npr, ts) in enumerate(plan):
        H, _ = s74.read_response(model, ids, npr, layer_idx, device)
        Z = s74.project_response(H, space, s28)
        states[step_off[k]:step_off[k + 1]], step_tokens[step_off[k]:step_off[k + 1]] = step_means(Z, ts, int(n_steps[k]))
        if (k + 1) % 200 == 0 or k + 1 == len(plan):
            el = time.time() - t0
            print("    store pass %d/%d (%.0fs, ~%.0fs left)" % (k + 1, len(plan), el, el / (k + 1) * (len(plan) - k - 1)), flush=True)
    if not np.isfinite(states).all():
        raise SystemExit("non-finite step states")
    os.makedirs(out_dir, exist_ok=True)
    kept = [rows[i] for i, _, _, _ in plan]
    arrays = {"step_states": states, "step_offsets": step_off, "step_tokens": step_tokens,
              "first_error": np.array([r["label"] for r in kept], dtype=np.int64),
              "split": np.array([r["split"] for r in kept]), "generator": np.array([r["generator"] for r in kept]),
              "problem_key": np.array([hashlib.sha1(r["problem"].encode("utf-8")).hexdigest()[:16] for r in kept]),
              "solution_id": np.array([r["id"] for r in kept]), "basis": basis}
    for name, arr in arrays.items():
        np.save(os.path.join(out_dir, name + ".npy"), arr)
    meta = {"reader": pc["reader"], "model_id": model_id, "hf_path": pc["hf_path"], "splits": pc["splits"],
            "blocks": [first + j for j in range(N_BLOCKS)], "r": R, "solutions": len(plan), "steps": int(step_off[-1]),
            "skipped": skipped, "basis_solutions": int(basis.sum()), "variance_kept": share,
            "prompt": "reader's chat template, one user turn = the problem; steps joined by blank lines as the reply",
            "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print("  stored %d solutions, %d steps -> %s" % (len(plan), int(step_off[-1]), out_dir), flush=True)


# ---------------------------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------------------------

def make_items(step_off, first_error, sols):
    """Items (solution, position, label): every step before the first error (0) and the first error (1);
    every step of a solution with no error (0)."""
    s_, k_, y_ = [], [], []
    for s in sols:
        n = int(step_off[s + 1] - step_off[s])
        e = int(first_error[s])
        last = n - 1 if e < 0 else min(e, n - 1)
        for k in range(last + 1):
            s_.append(s)
            k_.append(k)
            y_.append(1 if k == e else 0)
    return np.array(s_), np.array(k_), np.array(y_)


def problem_folds(problem_key, sols, n_folds, seed):
    keys = problem_key[sols]
    u, inv = np.unique(keys, return_inverse=True)
    f = np.random.default_rng(seed).permutation(len(u)) % n_folds
    return f[inv]


def same_position_auroc(S, y, group, unit, n_boot, seed):
    """S: (items, C). For each group, pairs of a positive and a negative item (ties half); pooled over groups.
    Interval: resample units (solutions) with replacement; an item counts as often as its unit is drawn."""
    S = np.asarray(S, dtype=np.float64)
    if S.ndim == 1:
        S = S[:, None]
    uu, uinv = np.unique(unit, return_inverse=True)
    prep = []
    for g in np.unique(group):
        pos, neg = np.flatnonzero((group == g) & (y == 1)), np.flatnonzero((group == g) & (y == 0))
        if len(pos) == 0 or len(neg) == 0:
            continue
        cols = []
        for c in range(S.shape[1]):
            o = np.argsort(S[neg, c], kind="stable")
            ns = S[neg, c][o]
            lo = np.searchsorted(ns, S[pos, c], side="left")
            hi = np.searchsorted(ns, S[pos, c], side="right")
            cols.append((neg[o], lo, hi))
        prep.append((pos, cols))

    def stat(wu):
        w = wu[uinv]
        higher, pairs = np.zeros(S.shape[1]), np.zeros(S.shape[1])
        for pos, cols in prep:
            wp = w[pos]
            for c, (nsort, lo, hi) in enumerate(cols):
                cw = np.concatenate([[0.0], np.cumsum(w[nsort])])
                higher[c] += float(np.sum(wp * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))))
                pairs[c] += float(wp.sum() * cw[-1])
        return higher / np.maximum(pairs, 1e-12), pairs
    point, pairs = stat(np.ones(len(uu)))
    rng = np.random.default_rng(seed)
    draws = np.array([stat(np.bincount(rng.integers(0, len(uu), len(uu)), minlength=len(uu)).astype(float))[0]
                      for _ in range(n_boot)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return point, np.stack([lo, hi], axis=1), pairs


def sequences(X, step_off, s, k, orders):
    """(items, k_max + 1, F) step states of steps 0..k of each item's solution, step k last; with orders, the
    earlier steps follow orders[i]. Returns the padded array and lengths."""
    lens = k + 1
    out = np.zeros((len(s), int(lens.max()), X.shape[1]), dtype=np.float32)
    for i in range(len(s)):
        prev = np.arange(k[i]) if orders is None else orders[i]
        r = step_off[s[i]] + np.r_[prev, k[i]].astype(np.int64)
        out[i, :lens[i]] = X[r]
    return out, lens


def build_reader(F, hidden):
    import torch

    class Reader(torch.nn.Module):
        """Score of step k = linear layer on [GRU state after step k, step k's own state] (probe + context)."""
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(F, hidden, batch_first=True)
            self.head = torch.nn.Linear(hidden + F, 1)

        def forward(self, x, lens):
            packed = torch.nn.utils.rnn.pack_padded_sequence(x, torch.as_tensor(lens), batch_first=True,
                                                             enforce_sorted=False)
            _, h = self.gru(packed)
            last = x[torch.arange(x.shape[0], device=x.device), torch.as_tensor(lens, device=x.device) - 1]
            return self.head(torch.cat([h[-1], last], dim=-1)).squeeze(-1)
    return Reader()


def train_reader(X, step_off, s, k, y, tr, te, orders, seed, device, init=None, epochs=EPOCHS, hidden=HIDDEN,
                 batch=BATCH, lr=LR):
    """Train on items tr (indices into s/k/y), return scores for items te. init = (scaler, model) of the step
    probe: the reader starts exactly at the probe (direct path set to its weights, GRU path at zero) and
    training adds what the earlier steps contribute."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows = np.unique(np.concatenate([step_off[a] + np.arange(step_off[a + 1] - step_off[a]) for a in np.unique(s[tr])]))
    mu, sd = X[rows].mean(0).astype(np.float64), X[rows].std(0).astype(np.float64) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    pw = torch.tensor(float((y[tr] == 0).sum()) / max(int(y[tr].sum()), 1), device=device)
    model = build_reader(X.shape[1], hidden).to(device)
    if init is not None:
        sc, clf = init
        coef = clf.coef_.ravel()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.weight[0, hidden:] = torch.from_numpy((coef * sd / sc.scale_).astype(np.float32))
            model.head.bias.fill_(float(clf.intercept_[0] + np.sum(coef * (mu - sc.mean_) / sc.scale_)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        order = tr[np.argsort(k[tr], kind="stable")]
        chunks = [order[a:a + batch] for a in range(0, len(order), batch)]
        for b in (chunks[i] for i in rng.permutation(len(chunks))):
            xb, lb = sequences(Xs, step_off, s[b], k[b], None if orders is None else [orders[i] for i in b])
            logit = model(torch.from_numpy(xb).to(device), lb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, torch.as_tensor(y[b], dtype=torch.float32, device=device), pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    out = np.empty(len(te))
    with torch.no_grad():
        for a in range(0, len(te), 512):
            b = te[a:a + 512]
            xb, lb = sequences(Xs, step_off, s[b], k[b], None if orders is None else [orders[i] for i in b])
            out[a:a + len(b)] = model(torch.from_numpy(xb).to(device), lb).cpu().numpy()
    return out


def evaluate(X, step_off, first_error, problem_key, sols, seed, device, n_boot, log=print, seeds=SEEDS, n_folds=N_FOLDS, **kw):
    """Held-out scores of the probe and both GRUs for every item; same-position AUROC of each."""
    s73 = _load("s73", "73_token_structure.py")
    s, k, y = make_items(step_off, first_error, sols)
    sol_fold = dict(zip(sols.tolist(), problem_folds(problem_key, sols, n_folds, seed).tolist()))
    fold = np.array([sol_fold[int(a)] for a in s])
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(int(kk)) for kk in k]          # one fixed order of the earlier steps per item
    names = ["probe"] + ["gru_real_order_seed%d" % a for a in seeds] + ["gru_shuffled_order_seed%d" % a for a in seeds]
    S = np.full((len(s), len(names)), np.nan)
    for f in range(n_folds):
        t0 = time.time()
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        Xi = X[step_off[s] + k]
        w = np.where(y[tr] == 1, 0.5 / max(y[tr].sum(), 1), 0.5 / max((y[tr] == 0).sum(), 1))
        w *= len(w) / w.sum()
        sc, clf = s73.fit_probe(Xi[tr], y[tr], w, seed)
        S[te, 0] = clf.decision_function(sc.transform(Xi[te]))
        j = 1
        for o in (None, orders):
            for a in seeds:
                S[te, j] = train_reader(X, step_off, s, k, y, tr, te, o, a, device, init=(sc, clf), **kw)
                j += 1
        log("    fold %d: %d training items, %d test items (%.0fs)" % (f, len(tr), len(te), time.time() - t0))
    group = np.minimum(k, MAX_GROUP)
    point, ci, pairs = same_position_auroc(S, y, group, s, n_boot, seed)
    res = {n: {"auroc": float(point[j]), "interval": ci[j].tolist()} for j, n in enumerate(names)}
    return res, {"items": int(len(s)), "first_errors": int(y.sum()), "pairs": float(pairs[0]),
                 "solutions": int(len(np.unique(s)))}, (S, s, k, y, names)


def run(out_dir, seed, n_boot, device_pref):
    import torch
    d = store_dir()
    if not os.path.exists(os.path.join(d, "meta.json")):
        extract(device_pref, seed)
    meta = json.load(open(os.path.join(d, "meta.json")))
    A = {n: np.load(os.path.join(d, n + ".npy")) for n in
         ("step_states", "step_offsets", "first_error", "split", "generator", "problem_key", "basis")}
    X = A["step_states"].reshape(len(A["step_states"]), -1).astype(np.float32)
    step_off = A["step_offsets"].astype(np.int64)
    sols = np.flatnonzero(~A["basis"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.enabled = False        # cluster cuDNN clash, see 80
    fe = A["first_error"][sols]
    print("  [%s reads ProcessBench] %d solutions evaluated (%d basis left out), %d with an error | device %s"
          % (meta["reader"], len(sols), int(A["basis"].sum()), int((fe >= 0).sum()), device), flush=True)
    print("    first wrong step at position 0 1 2 3 4 5 6 7 8 9 10+: %s"
          % " ".join(str(int(v)) for v in np.bincount(np.minimum(fe[fe >= 0], MAX_GROUP), minlength=MAX_GROUP + 1)), flush=True)

    res, counts, (S, s, k, y, names) = evaluate(X, step_off, A["first_error"], A["problem_key"], sols, seed, device, n_boot,
                                                lambda m: print(m, flush=True))
    by_split = {}
    for sp in np.unique(A["split"][sols]):
        m = A["split"][s] == sp
        p, ci, pairs = same_position_auroc(S[m], y[m], np.minimum(k[m], MAX_GROUP), s[m], 200, seed)
        by_split[str(sp)] = {n: {"auroc": float(p[j]), "interval": ci[j].tolist()} for j, n in enumerate(names)}
        by_split[str(sp)]["pairs"] = float(pairs[0])
    out = {"reader": meta["reader"], "store": meta, "seed": seed, "n_boot": n_boot, "hidden": HIDDEN, "epochs": EPOCHS,
           "gru_seeds": list(SEEDS), "same_position_auroc": res, "counts": counts, "by_split": by_split}
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "processbench_steps_%s.json" % meta["reader"])
    s78 = _load("s78", "78_math_same_question.py")
    json.dump(s78.clean(out), open(dst, "w"), indent=1)

    def mean_runs(r, cond):
        return np.mean([r["%s_seed%d" % (cond, a)]["auroc"] for a in SEEDS])
    print("\n  SUMMARY: same-position AUROC, first wrong step vs correct steps at the same position [95%]")
    print("  (%d solutions, %d items, %d first wrong steps, %.0f pairs)"
          % (counts["solutions"], counts["items"], counts["first_errors"], counts["pairs"]))
    v = res["probe"]
    print("    %-32s %.3f [%.3f, %.3f]" % ("probe, the step alone", v["auroc"], v["interval"][0], v["interval"][1]))
    for cond, label in (("gru_real_order", "GRU, earlier steps in order"), ("gru_shuffled_order", "GRU, earlier steps shuffled")):
        print("    %-32s %.3f  (runs %s)" % (label, mean_runs(res, cond),
                                             " ".join("%.3f" % res["%s_seed%d" % (cond, a)]["auroc"] for a in SEEDS)))
    v = res["gru_real_order_seed0"]
    print("    (interval of one GRU run, real order seed 0: [%.3f, %.3f])" % tuple(v["interval"]))
    print("    by split            probe   real   shuffled")
    for sp, r in by_split.items():
        print("      %-16s %6.3f %6.3f %6.3f" % (sp, r["probe"]["auroc"], mean_runs(r, "gru_real_order"), mean_runs(r, "gru_shuffled_order")))
    print("  wrote %s" % dst)


# ---------------------------------------------------------------------------------------------

def self_test():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print("    [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
        ok = ok and bool(cond)

    steps = ["First.", "Then 2.", "So 4."]
    st = step_starts(steps)
    text = "\n\n".join(steps)
    check("step starts", [text[a:a + 2] for a in st] == ["Fi", "Th", "So"], str(st))
    om = [(0, 5), (5, 6), (6, 8), (8, 12), (12, 15), (15, 17), (17, 19), (19, 20)]
    check("tokens to steps; a separator belongs to the step before",
          token_steps(om, st).tolist() == [0, 0, 0, 1, 1, 1, 2, 2], str(token_steps(om, st).tolist()))
    Z = np.arange(12, dtype=np.float32).reshape(4, 1, 3)
    m, c = step_means(Z, np.array([0, 0, 1, 1]), 2)
    check("step means", np.allclose(m[:, 0], [[1.5, 2.5, 3.5], [7.5, 8.5, 9.5]]) and c.tolist() == [2, 2])
    off = np.array([0, 4, 7, 9])
    s, k, y = make_items(off, np.array([2, -1, 0]), np.arange(3))
    check("items stop at the first error", list(zip(s.tolist(), k.tolist(), y.tolist())) ==
          [(0, 0, 0), (0, 1, 0), (0, 2, 1), (1, 0, 0), (1, 1, 0), (1, 2, 0), (2, 0, 1)], str(list(zip(s, k, y))))

    S = rng.normal(size=(300, 2))
    yy = (rng.random(300) < 0.3).astype(int)
    S[yy == 1, 0] += 1.0
    p, ci, _ = same_position_auroc(S, yy, np.zeros(300, dtype=int), np.arange(300), 200, 0)
    check("one group: equals the ordinary AUROC", abs(p[0] - roc_auc_score(yy, S[:, 0])) < 1e-9 and ci[0, 0] < p[0] < ci[0, 1],
          "%.4f vs %.4f" % (p[0], roc_auc_score(yy, S[:, 0])))
    g = np.repeat([0, 1], 150)
    S2 = rng.normal(size=300) + 3.0 * g
    y2 = np.r_[(rng.random(150) < 0.2), (rng.random(150) < 0.8)].astype(int)
    p2, _, _ = same_position_auroc(S2, y2, g, np.arange(300), 50, 0)
    check("a score that only knows the group is 0.5 within groups", abs(p2[0] - 0.5) < 0.08 and roc_auc_score(y2, S2) > 0.7,
          "within %.3f, pooled %.3f" % (p2[0], roc_auc_score(y2, S2)))
    p3, _, _ = same_position_auroc(np.r_[S[:, 0], S[:, 0]], np.r_[yy, yy], np.zeros(600, dtype=int), np.r_[np.arange(300), np.arange(300)], 10, 0)
    check("duplicating every unit leaves the AUROC unchanged", abs(p3[0] - p[0]) < 1e-9)

    # an untrained reader started from the probe gives the probe's scores
    s73 = _load("s73", "73_token_structure.py")
    Xp = (rng.normal(size=(9, 5)) * 2 + 1).astype(np.float32)
    o9 = np.array([0, 4, 7, 9])
    si, ki, yi = make_items(o9, np.array([3, -1, 1]), np.arange(3))
    yi[0] = 1
    sc, clf = s73.fit_probe(Xp[o9[si] + ki], yi, np.ones(len(yi)), 0)
    allit = np.arange(len(si))
    s0 = train_reader(Xp, o9, si, ki, yi, allit, allit, None, 0, "cpu", init=(sc, clf), epochs=0, hidden=4)
    ref = clf.decision_function(sc.transform(Xp[o9[si] + ki]))
    check("the reader starts exactly at the probe", np.allclose(s0, ref, atol=1e-3), "max diff %.1e" % float(np.abs(s0 - ref).max()))

    # Two worlds. ORDER: every solution has one risky step; when the solution goes wrong, the error is the step
    # right after it. Error steps look like correct steps; only knowing what came JUST before finds them.
    # LEVEL: error steps are shifted on one coordinate; every reader finds them.
    n, F = 600, 4
    n_steps = rng.integers(5, 10, size=n)
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(n_steps, out=off[1:])
    keys = np.array(["p%d" % i for i in range(n)])
    for world in ("order", "level"):
        X = rng.normal(size=(off[-1], F)).astype(np.float32)
        fe = np.full(n, -1)
        for i in range(n):
            r = int(rng.integers(0, n_steps[i] - 1))
            if world == "order":
                X[off[i] + r, 0] = 5.0
                if rng.random() < 0.5:
                    fe[i] = r + 1
            elif rng.random() < 0.5:
                fe[i] = r + 1
                X[off[i] + r + 1, 1] += 2.5
        res, _, _ = evaluate(X, off, fe, keys, np.arange(n), 0, "cpu", 50, lambda m: None, seeds=(0,), n_folds=3,
                             epochs=20, hidden=16, batch=64, lr=5e-3)
        a = {kk: v["auroc"] for kk, v in res.items()}
        if world == "order":
            # shuffled order keeps part of the signal here: a correct step whose solution's risky step comes
            # LATER has no risky step among its earlier steps, which the set alone reveals (expected ~0.8)
            check("order world: real order finds the error, shuffled less, the probe hardly",
                  a["gru_real_order_seed0"] > 0.85 and a["gru_real_order_seed0"] - a["gru_shuffled_order_seed0"] > 0.08
                  and a["probe"] < 0.65, " ".join("%s %.3f" % (kk, v) for kk, v in a.items()))
        else:
            check("level world: every reader finds the error", min(a.values()) > 0.8,
                  " ".join("%s %.3f" % (kk, v) for kk, v in a.items()))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prefetch", action="store_true", help="download ProcessBench into the HF cache and exit")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    if a.self_test:
        print("  self-test")
        raise SystemExit(0 if self_test() else 1)
    if a.prefetch:
        rows = load_rows(pb_cfg()[1])
        print("  ProcessBench: %d solutions, %d with an error, splits %s"
              % (len(rows), sum(r["label"] >= 0 for r in rows), sorted({r["split"] for r in rows})))
        return
    run(a.out_dir, a.seed, a.n_boot, a.device)


if __name__ == "__main__":
    main()
