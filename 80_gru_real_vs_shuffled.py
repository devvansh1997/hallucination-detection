"""
80_gru_real_vs_shuffled.py -- can a TRAINED sequence model use the order of tokens to tell right from wrong?

THE CHECK. A small GRU reads an answer's token states one by one; its outputs are averaged over the answer
and a linear layer gives the score. It is trained to label answers right or wrong, twice:
    real order       the tokens as the model wrote them
    shuffled order   each answer's tokens in a fixed random order
Same model, data, training and evaluation; only the order differs. If real order beats shuffled, a trained
temporal model has something to use. If they tie, the model learns from the set of states, which the
token probe already captures.

TABLE. Same-question AUROC on held-out questions (78's folds), for the first half, second half and whole
answer: the trained layer applied to the average output over that part. In the real-order model an output
depends only on the tokens up to it, so the first-half score is what the model knows by the middle.

Reads the store 77 wrote (9 blocks x 64 directions per token) and labels from the sequences file. The 150
basis questions are left out. GPU if present, else CPU.

  python 80_gru_real_vs_shuffled.py --self-test
  python 80_gru_real_vs_shuffled.py --model_folder qwen-2.5-7b-instruct
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "token_structure")
HIDDEN = 64
EPOCHS = 8
BATCH = 64
LR = 1e-3
N_SCALE = 200000            # training tokens used for the per-coordinate mean and std
PARTS = ("first_half", "second_half", "whole_answer")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def shuffled_orders(offsets, idx, seed):
    """One fixed random order of the tokens of every answer in idx."""
    rng = np.random.default_rng(seed)
    return {int(n): rng.permutation(int(offsets[n + 1] - offsets[n])) for n in idx}


def make_batch(X, offsets, idx, orders, mu, sd):
    """(answers, longest, F) float32 standardized states, zero-padded, and the answer lengths."""
    lens = np.array([offsets[n + 1] - offsets[n] for n in idx], dtype=np.int64)
    B = np.zeros((len(idx), int(lens.max()), X.shape[1]), dtype=np.float32)
    for i, n in enumerate(idx):
        r = np.arange(offsets[n], offsets[n + 1])
        if orders is not None:
            r = r[orders[int(n)]]
        B[i, :len(r)] = (np.asarray(X[r], dtype=np.float32) - mu) / sd
    return B, lens


def part_masks(lens, T):
    """(3, answers, T) masks for the first half, second half and whole answer, split as 73 splits them."""
    pos = np.arange(T)[None, :]
    L = lens[:, None]
    return np.stack([pos < (L + 1) // 2, (pos >= L // 2) & (pos < L), pos < L]).astype(np.float32)


def build_model(F, hidden):
    import torch

    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(F, hidden, batch_first=True)
            self.head = torch.nn.Linear(hidden, 1)

        def forward(self, x, lens, masks):
            """masks: (P, B, T). Returns (P, B) scores: the head on the mean output over each part."""
            packed = torch.nn.utils.rnn.pack_padded_sequence(x, torch.as_tensor(lens), batch_first=True,
                                                             enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
            pooled = (masks[..., None] * out[None]).sum(2) / masks.sum(2, keepdim=True).clamp(min=1)
            return self.head(pooled).squeeze(-1)
    return Reader()


def batches(idx, lengths, size, rng=None):
    """Answers of similar length together (less padding); batch order shuffled when rng is given."""
    order = idx[np.argsort(lengths, kind="stable")]
    out = [order[a:a + size] for a in range(0, len(order), size)]
    if rng is not None:
        out = [out[i] for i in rng.permutation(len(out))]
    return out


def train_and_score(X, offsets, y, tr, te, orders, seed, device, epochs=EPOCHS, hidden=HIDDEN, batch=BATCH, lr=LR):
    """Train on answers tr, return (len(te), 3) scores for te and the last epoch's mean training loss."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lengths = np.diff(offsets)
    rows = np.concatenate([np.arange(offsets[n], offsets[n + 1]) for n in tr])
    rows = rng.choice(rows, size=min(N_SCALE, len(rows)), replace=False)
    S = np.asarray(X[np.sort(rows)], dtype=np.float64)
    mu, sd = S.mean(0).astype(np.float32), (S.std(0) + 1e-6).astype(np.float32)

    model = build_model(X.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor(float((y[tr] == 0).sum()) / max(int(y[tr].sum()), 1), device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        model.train()
        losses = []
        for b in batches(tr, lengths[tr], batch, rng):
            xb, lb = make_batch(X, offsets, b, orders, mu, sd)
            m = torch.from_numpy(part_masks(lb, xb.shape[1])[2:3]).to(device)
            logit = model(torch.from_numpy(xb).to(device), lb, m)[0]
            loss = loss_fn(logit, torch.as_tensor(y[b], dtype=torch.float32, device=device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
    model.eval()
    out = np.empty((len(te), 3))
    where = {int(n): i for i, n in enumerate(te)}
    with torch.no_grad():
        for b in batches(te, lengths[te], batch):
            xb, lb = make_batch(X, offsets, b, orders, mu, sd)
            m = torch.from_numpy(part_masks(lb, xb.shape[1])).to(device)
            s = model(torch.from_numpy(xb).to(device), lb, m).cpu().numpy()
            for j, n in enumerate(b):
                out[where[int(n)]] = s[:, j]
    return out, float(np.mean(losses))


def evaluate(X, offsets, y, pid, keep, fold, seed, device, log=print, **kw):
    """{condition: (len(keep), 3) held-out scores} for real and shuffled order."""
    orders = shuffled_orders(offsets, keep, seed)
    scores = {c: np.full((len(keep), 3), np.nan) for c in ("real_order", "shuffled_order")}
    for f in range(int(fold.max()) + 1):
        tr, te = keep[fold != f], keep[fold == f]
        for c, o in (("real_order", None), ("shuffled_order", orders)):
            t0 = time.time()
            scores[c][fold == f], loss = train_and_score(X, offsets, y, tr, te, o, seed + f, device, **kw)
            log("    fold %d, %-14s trained on %d answers, last-epoch loss %.3f (%.0fs)" % (f, c, len(tr), loss, time.time() - t0))
    return scores


def run(model_folder, dataset, out_dir, seed, n_boot):
    import torch
    import yaml
    s78 = _load("s78", "78_math_same_question.py")
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    root = os.path.normpath(os.path.join(HERE, cfg["math"]["out_dir"], model_folder))
    d = os.path.join(root, "%s_states" % dataset)
    if not os.path.exists(os.path.join(d, "meta.json")):
        raise SystemExit("%s is missing or incomplete -- run 77_extract_math_states.py first" % d)
    seq = torch.load(os.path.join(root, "%s_sequences_v1.pt" % dataset), weights_only=False)
    y = np.asarray(seq["all_hallucination_flag"]).astype(int)            # 1 = wrong final number
    pid = np.asarray(seq["prompt_id"], dtype=np.int64)
    offsets = np.load(os.path.join(d, "offsets.npy")).astype(np.int64)
    if not np.array_equal(pid, np.load(os.path.join(d, "prompt_id.npy")).astype(np.int64)):
        raise SystemExit("the state store and the sequences file disagree on question ids")
    keep = np.flatnonzero(~np.load(os.path.join(d, "basis_question.npy")))
    tokens = np.load(os.path.join(d, "tokens.npy"))
    X = tokens.reshape(len(tokens), -1)
    fold, _ = s78.question_folds(pid[keep], y[keep], s78.N_FOLDS, seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("  [%s/%s] %d answers evaluated, %d coordinates per token, device %s | GRU hidden %d, %d epochs"
          % (model_folder, dataset, len(keep), X.shape[1], device, HIDDEN, EPOCHS), flush=True)

    scores = evaluate(X, offsets, y, pid, keep, fold, seed, device, lambda m: print(m, flush=True))
    out = {"model_folder": model_folder, "dataset": dataset, "seed": seed, "n_boot": n_boot, "hidden": HIDDEN,
           "epochs": EPOCHS, "columns": list(PARTS), "same_question_auroc": {}}
    for c, S in scores.items():
        s = s78.summarize(s78.same_question_stats(S, y[keep], pid[keep]), n_boot, seed)
        out["same_question_auroc"][c] = {"auroc": s["same_question_auroc"], "interval": s["auroc_interval"],
                                         "pairs": s["pairs"], "questions": s["questions"]}
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "gru_order_%s_%s.json" % (model_folder, dataset))
    json.dump(s78.clean(out), open(dst, "w"), indent=1)

    r = out["same_question_auroc"]["real_order"]
    print("\n  SUMMARY: same-question AUROC [95%%], %d questions, %d pairs" % (r["questions"][2], r["pairs"][2]))
    print("    %-30s %-20s %-20s %s" % ("", "first half", "second half", "whole answer"))
    for c, label in (("real_order", "GRU, tokens in real order"), ("shuffled_order", "GRU, tokens shuffled")):
        v = out["same_question_auroc"][c]
        print("    %-30s %s" % (label, "  ".join("%.3f [%.3f, %.3f]" % (v["auroc"][j], v["interval"][0][j], v["interval"][1][j])
                                                  for j in range(3))))
    ref = os.path.join(out_dir, "math_%s_%s.json" % (model_folder, dataset))
    if os.path.exists(ref):
        t = json.load(open(ref))["token_scorers"]["pooled_probe"]["by_tokens_used"]
        a = dict(zip(t["columns"], t["same_question_auroc"]))
        print("    %-30s %-20s %-20s %.3f" % ("token probe (position check)", "%.3f" % a["first_half"], "%.3f" % a["second_half"], a["mean"]))
    print("  wrote %s" % dst)


# ---------------------------------------------------------------------------------------------

def self_test():
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print("    [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
        ok = ok and bool(cond)

    offs = np.array([0, 3, 5])
    orders = shuffled_orders(offs, [0, 1], 0)
    check("shuffled orders are permutations", sorted(orders[0].tolist()) == [0, 1, 2] and sorted(orders[1].tolist()) == [0, 1])
    m = part_masks(np.array([5, 4]), 6)
    check("halves split as 73 splits them", m[0, 0].tolist() == [1, 1, 1, 0, 0, 0] and m[1, 0].tolist() == [0, 0, 1, 1, 1, 0]
          and m[2, 1].tolist() == [1, 1, 1, 1, 0, 0])
    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    B, lens = make_batch(X, offs, [0, 1], {0: np.array([2, 0, 1]), 1: np.array([1, 0])}, 0.0, 1.0)
    check("batches follow the given order and pad with zeros", B[0, :, 0].tolist() == [4, 0, 2] and B[1, :, 0].tolist() == [8, 6, 0])

    # Two synthetic worlds. In the first, right and wrong answers hold the SAME set of values and differ only
    # in order (rising versus falling): only the real-order model can tell them apart. In the second, wrong
    # answers are shifted: the order is irrelevant and both models should succeed.
    s78 = _load("s78", "78_math_same_question.py")
    Q, F = 60, 4
    pid = np.repeat(np.arange(Q), 10)
    y = (rng.random(Q * 10) < 0.4).astype(int)
    lengths = rng.integers(20, 41, size=Q * 10)
    offsets = np.zeros(Q * 10 + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    keep = np.arange(Q * 10)
    fold, _ = s78.question_folds(pid, y, 3, 0)
    for world in ("order", "level"):
        X = rng.normal(size=(offsets[-1], F)).astype(np.float32)
        for n in range(Q * 10):
            v = np.sort(rng.normal(size=lengths[n])) * 2.0
            if world == "order":
                X[offsets[n]:offsets[n + 1], 0] = v[::-1] if y[n] else v
            else:
                X[offsets[n]:offsets[n + 1], 0] = rng.normal(size=lengths[n]) + (0.8 if y[n] else 0.0)
        sc = evaluate(X, offsets, y, pid, keep, fold, 0, "cpu", lambda m: None, epochs=12, hidden=16, batch=32, lr=3e-3)
        a = {c: s78.summarize(s78.same_question_stats(S, y, pid), 100, 0)["same_question_auroc"][2] for c, S in sc.items()}
        if world == "order":
            check("order-only world: real order separates, shuffled does not",
                  a["real_order"] > 0.9 and abs(a["shuffled_order"] - 0.5) < 0.12,
                  "real %.3f, shuffled %.3f" % (a["real_order"], a["shuffled_order"]))
        else:
            check("level world: both separate", a["real_order"] > 0.8 and a["shuffled_order"] > 0.8,
                  "real %.3f, shuffled %.3f" % (a["real_order"], a["shuffled_order"]))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model_folder")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    if a.self_test:
        print("  self-test")
        raise SystemExit(0 if self_test() else 1)
    if not a.model_folder:
        raise SystemExit("--model_folder is required (or --self-test)")
    run(a.model_folder, a.dataset, a.out_dir, a.seed, a.n_boot)


if __name__ == "__main__":
    main()
