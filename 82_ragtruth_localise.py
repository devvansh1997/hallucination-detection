"""
82_ragtruth_localise.py -- does reading tokens IN ORDER help find where a hallucination is? (T-026)
==================================================================================================

THE CHECK. Every token of a RAGTruth response is labeled inside or outside a human-marked hallucinated span.
Four readers score every test token:

    probe                  logistic regression on the token's own state
    probe + position       the same plus where the token sits (share of the response, log of its index)
    GRU, real order        reads the response token by token (state + position); scores each token from
                           its own inputs plus the GRU's summary of the tokens up to it (probe + context)
    GRU, shuffled order    the same model and training, each response's tokens in one fixed random order

"Probe + position" is there because a reader that sees tokens in order can count how far into the response
it is, and hallucinations are not spread evenly. The GRUs get the same position inputs, so the only thing
real order adds over shuffled is which tokens came just before.

SCORE. Within-response AUROC: inside one hallucinated test response, does a token inside a span score above a
token outside it? Pairs are pooled over responses, and the 95% interval resamples responses (78's
same-question code with the response as the group). A response-wide offset cannot help here; this is
localisation only. Pooled token AUROC is printed for reference.

READING. GRU real order above both shuffled order and probe + position: order helps localisation. Otherwise
the token's own state (and where it sits) is all the readers use.

Reads 74's store (RAGTruth train split for fitting, test split for scoring). Each GRU is trained 3 times
(seeds 0-2) and every run is reported. GPU if present.

  python 82_ragtruth_localise.py --self-test
  python 82_ragtruth_localise.py --generator llama-2-7b-chat
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
EPOCHS = 10
BATCH = 32
LR = 1e-3
SEEDS = (0, 1, 2)
N_SCALE = 200000
N_BOOT = 2000


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def position_features(offsets):
    """(total_T, 2) float32: (t + 0.5) / T and log(1 + t) / log(513) for token t of a response of T tokens."""
    lengths = np.diff(offsets)
    t = np.concatenate([np.arange(L) for L in lengths]).astype(np.float64)
    L = np.repeat(lengths, lengths).astype(np.float64)
    return np.stack([(t + 0.5) / L, np.log1p(t) / np.log(513.0)], axis=1).astype(np.float32)


def shuffled_orders(offsets, idx, seed):
    rng = np.random.default_rng(seed)
    return {int(n): rng.permutation(int(offsets[n + 1] - offsets[n])) for n in idx}


def make_batch(X, P, flags, offsets, idx, orders, mu, sd):
    """(B, longest, F + 2) inputs, (B, longest) labels, lengths, and per response the original positions in
    the order they were fed."""
    lens = np.array([offsets[n + 1] - offsets[n] for n in idx], dtype=np.int64)
    Tm = int(lens.max())
    xb = np.zeros((len(idx), Tm, X.shape[1] + P.shape[1]), dtype=np.float32)
    yb = np.zeros((len(idx), Tm), dtype=np.float32)
    fed = []
    for i, n in enumerate(idx):
        o = orders[int(n)] if orders is not None else np.arange(lens[i])
        r = offsets[n] + o
        xb[i, :lens[i], :X.shape[1]] = (np.asarray(X[r], dtype=np.float32) - mu) / sd
        xb[i, :lens[i], X.shape[1]:] = P[r]
        yb[i, :lens[i]] = flags[r]
        fed.append(o)
    return xb, yb, lens, fed


def build_tagger(F, hidden):
    import torch

    class Tagger(torch.nn.Module):
        """Score of token t = linear layer on [GRU output at t, token t's own inputs]. The direct path makes
        the tagger the probe plus context, so it cannot fall below the probe by failing to learn the token."""
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(F, hidden, batch_first=True)
            self.head = torch.nn.Linear(hidden + F, 1)

        def forward(self, x, lens):
            packed = torch.nn.utils.rnn.pack_padded_sequence(x, torch.as_tensor(lens), batch_first=True,
                                                             enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
            return self.head(torch.cat([out, x], dim=-1)).squeeze(-1)
    return Tagger()


def batches(idx, lengths, size, rng=None):
    order = idx[np.argsort(lengths, kind="stable")]
    out = [order[a:a + size] for a in range(0, len(order), size)]
    if rng is not None:
        out = [out[i] for i in rng.permutation(len(out))]
    return out


def probe_head(sc, clf, mu, sd, n_x):
    """Weights and bias that reproduce a probe fitted on [X, P] (StandardScaler sc, logistic clf) on the
    tagger's inputs [(X - mu) / sd, P]."""
    coef, m, s = clf.coef_.ravel(), sc.mean_, sc.scale_
    w = np.r_[coef[:n_x] * sd / s[:n_x], coef[n_x:] / s[n_x:]]
    b = clf.intercept_[0] + np.sum(coef[:n_x] * (mu - m[:n_x]) / s[:n_x]) - np.sum(coef[n_x:] * m[n_x:] / s[n_x:])
    return w.astype(np.float32), float(b)


def train_tagger(X, P, flags, offsets, tr, te, orders, seed, device, init=None, epochs=EPOCHS, hidden=HIDDEN,
                 batch=BATCH, lr=LR):
    """Train on responses tr; return (total_T,) scores filled for the tokens of te (NaN elsewhere) and the
    last epoch's mean loss. init = (scaler, model) of the probe + position: the tagger starts exactly at that
    probe (direct path set to its weights, context path at zero) and training adds what context contributes."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    s73 = _load("s73", "73_token_structure.py")
    rows = s73.token_rows(offsets, tr)
    sub = np.sort(rng.choice(rows, size=min(N_SCALE, len(rows)), replace=False))
    S = np.asarray(X[sub], dtype=np.float64)
    mu, sd = S.mean(0).astype(np.float32), (S.std(0) + 1e-6).astype(np.float32)
    n_pos = float(flags[rows].sum())
    pos_weight = torch.tensor((len(rows) - n_pos) / max(n_pos, 1.0), device=device)

    model = build_tagger(X.shape[1] + P.shape[1], hidden).to(device)
    if init is not None:
        w, b = probe_head(init[0], init[1], mu.astype(np.float64), sd.astype(np.float64), X.shape[1])
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.weight[0, hidden:] = torch.from_numpy(w)
            model.head.bias.fill_(b)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lengths = np.diff(offsets)
    losses = []
    for _ in range(epochs):
        model.train()
        losses = []
        for b in batches(tr, lengths[tr], batch, rng):
            xb, yb, lb, _ = make_batch(X, P, flags, offsets, b, orders, mu, sd)
            mask = torch.from_numpy((np.arange(xb.shape[1])[None, :] < lb[:, None]).astype(np.float32)).to(device)
            logit = model(torch.from_numpy(xb).to(device), lb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, torch.from_numpy(yb).to(device), pos_weight=pos_weight, reduction="none")
            loss = (loss * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
    model.eval()
    losses = losses or [float("nan")]
    out = np.full(int(offsets[-1]), np.nan)
    with torch.no_grad():
        for b in batches(te, lengths[te], batch):
            xb, _, lb, fed = make_batch(X, P, flags, offsets, b, orders, mu, sd)
            s = model(torch.from_numpy(xb).to(device), lb).cpu().numpy()
            for j, n in enumerate(b):
                out[offsets[n] + fed[j]] = s[j, :lb[j]]
    return out, float(np.mean(losses))


def fit_probes(X, P, flags, offsets, tr, seed):
    """Token-label probes on the training responses' tokens, classes weighted equally (75's weighting).
    Returns {name: (scaler, model, uses position)}."""
    s73 = _load("s73", "73_token_structure.py")
    rows = s73.token_rows(offsets, tr)
    yt = flags[rows].astype(int)
    w = np.where(yt == 1, 0.5 / max(yt.sum(), 1), 0.5 / max((yt == 0).sum(), 1))
    w *= len(w) / w.sum()
    out = {}
    for name, use_pos in (("probe", False), ("probe_plus_position", True)):
        Z = np.hstack([X[rows], P[rows]]) if use_pos else np.asarray(X[rows])
        sc, clf = s73.fit_probe(Z, yt, w, seed)
        out[name] = (sc, clf, use_pos)
    return out


def probe_scores(probes, X, P, rows, total):
    out = {}
    for name, (sc, clf, use_pos) in probes.items():
        s = np.full(total, np.nan)
        Z = np.hstack([X[rows], P[rows]]) if use_pos else np.asarray(X[rows])
        s[rows] = clf.decision_function(sc.transform(Z))
        out[name] = s
    return out


def within_response(scores, flags, offsets, te, names, n_boot, seed):
    """Within-response AUROC for each column, via 78's same-question code with the response as the group."""
    from sklearn.metrics import roc_auc_score
    s73 = _load("s73", "73_token_structure.py")
    s78 = _load("s78", "78_math_same_question.py")
    rows = s73.token_rows(offsets, te)
    group = np.repeat(np.asarray(te), np.diff(offsets)[te])
    yt = flags[rows].astype(int)
    S = np.stack([scores[n][rows] for n in names], axis=1)
    st = s78.same_question_stats(S, yt, group)
    s = s78.summarize(st, n_boot, seed)
    return {n: {"auroc": s["same_question_auroc"][j], "interval": [s["auroc_interval"][0][j], s["auroc_interval"][1][j]],
                "pooled_token_auroc": float(roc_auc_score(yt, S[:, j]))}
            for j, n in enumerate(names)}, {"responses": int(s["questions"][0]), "pairs": int(s["pairs"][0])}


def evaluate(X, flags, offsets, tr, te, seed, device, n_boot, log=print, seeds=SEEDS, **kw):
    P = position_features(offsets)
    t0 = time.time()
    probes = fit_probes(X, P, flags, offsets, tr, seed)
    s73 = _load("s73", "73_token_structure.py")
    scores = probe_scores(probes, X, P, s73.token_rows(offsets, te), int(offsets[-1]))
    log("    probes fitted (%.0fs)" % (time.time() - t0))
    orders = shuffled_orders(offsets, np.concatenate([tr, te]), seed)
    names = ["probe", "probe_plus_position"]
    for cond, o in (("gru_real_order", None), ("gru_shuffled_order", orders)):
        for sd in seeds:
            t0 = time.time()
            name = "%s_seed%d" % (cond, sd)
            scores[name], loss = train_tagger(X, P, flags, offsets, tr, te, o, sd, device,
                                              init=probes["probe_plus_position"][:2], **kw)
            names.append(name)
            log("    %-28s last-epoch loss %.3f (%.0fs)" % (name, loss, time.time() - t0))
    return within_response(scores, flags, offsets, te, names, n_boot, seed)


def run(generator, out_dir, seed, n_boot):
    import torch
    s74 = _load("s74", "74_extract_ragtruth.py")
    c = s74.ragtruth_cfg()
    d = os.path.join(s74.resolve(c["out_dir"]), generator)
    if not os.path.exists(os.path.join(d, "meta.json")):
        raise SystemExit("%s is missing or incomplete -- run 74_extract_ragtruth.py first" % d)
    meta = json.load(open(os.path.join(d, "meta.json")))
    offsets = np.load(os.path.join(d, "offsets.npy")).astype(np.int64)
    flags = np.load(os.path.join(d, "token_halluc.npy")).astype(np.int64)
    split = np.load(os.path.join(d, "split.npy"))
    tokens = np.load(os.path.join(d, "tokens.npy"))
    X = tokens.reshape(len(tokens), -1)
    tr, te = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.enabled = False        # cluster cuDNN clash, see 80
    print("  [%s] %d responses (%d train, %d test), %d tokens, %.1f%% inside spans | device %s"
          % (generator, len(split), len(tr), len(te), len(flags), 100 * flags.mean(), device), flush=True)

    res, counts = evaluate(X, flags, offsets, tr, te, seed, device, n_boot, lambda m: print(m, flush=True))
    out = {"generator": generator, "model_id": meta["model_id"], "seed": seed, "n_boot": n_boot, "hidden": HIDDEN,
           "epochs": EPOCHS, "gru_seeds": list(SEEDS), "within_response": res, "counts": counts}
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "ragtruth_localise_%s.json" % generator)
    s78 = _load("s78", "78_math_same_question.py")
    json.dump(s78.clean(out), open(dst, "w"), indent=1)

    print("\n  SUMMARY %s: within-response token AUROC [95%%], %d hallucinated test responses, %d pairs"
          % (generator, counts["responses"], counts["pairs"]))
    print("    %-34s %-22s %s" % ("", "within response", "pooled tokens"))
    for n, label in (("probe", "probe, one token"), ("probe_plus_position", "probe + position")):
        v = res[n]
        print("    %-34s %.3f [%.3f, %.3f]   %.3f" % (label, v["auroc"], v["interval"][0], v["interval"][1], v["pooled_token_auroc"]))
    for cond, label in (("gru_real_order", "GRU + position, real order"), ("gru_shuffled_order", "GRU + position, shuffled")):
        vs = [res["%s_seed%d" % (cond, s)] for s in SEEDS]
        print("    %-34s %.3f  (runs %s)   %.3f" % (label, np.mean([v["auroc"] for v in vs]),
                                                   " ".join("%.3f" % v["auroc"] for v in vs),
                                                   np.mean([v["pooled_token_auroc"] for v in vs])))
    v = res["gru_real_order_seed0"]
    print("    (interval of one GRU run, real order seed 0: [%.3f, %.3f])" % tuple(v["interval"]))
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
    P = position_features(offs)
    check("position features", np.allclose(P[:, 0], [0.5 / 3, 1.5 / 3, 2.5 / 3, 0.25, 0.75]) and P[0, 1] == 0)
    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    fl = np.array([0, 1, 0, 1, 0])
    xb, yb, lb, fed = make_batch(X, P, fl, offs, [0, 1], {0: np.array([2, 0, 1]), 1: np.array([1, 0])}, 0.0, 1.0)
    check("batch follows the order and carries labels with their tokens",
          xb[0, :, 0].tolist() == [4, 0, 2] and yb[0].tolist() == [0, 0, 1] and yb[1, :2].tolist() == [0, 1]
          and lb.tolist() == [3, 2], "%s %s" % (xb[0, :, 0].tolist(), yb.tolist()))

    # an untrained tagger started from the probe + position gives the probe's scores (order does not matter
    # without the context path)
    Xs = rng.normal(size=(offs[-1] + 200, 3)).astype(np.float32) * 3 + 1
    o2 = np.array([0, 3, 5, 90, 205])
    P2 = position_features(o2)
    f2 = (rng.random(len(Xs)) < 0.3).astype(np.int64)
    pr = fit_probes(Xs, P2, f2, o2, np.arange(4), 0)
    ref = probe_scores(pr, Xs, P2, np.arange(len(Xs)), len(Xs))["probe_plus_position"]
    s0, _ = train_tagger(Xs, P2, f2, o2, np.arange(4), np.arange(4), shuffled_orders(o2, np.arange(4), 1), 0, "cpu",
                         init=pr["probe_plus_position"][:2], epochs=0, hidden=8)
    check("the tagger starts exactly at the probe + position", np.allclose(s0, ref, atol=1e-3),
          "max diff %.1e" % float(np.nanmax(np.abs(s0 - ref))))

    # Two worlds. ORDER: a trigger token (feature 0 spike) is followed by a span of 5 tokens that look exactly
    # like other tokens; only a reader that knows what came just before can find them. LEVEL: span tokens are
    # shifted on feature 1; any reader finds them.
    n, F = 400, 4
    lengths = rng.integers(30, 61, size=n)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    tr, te = np.arange(300), np.arange(300, n)
    for world in ("order", "level"):
        X = rng.normal(size=(offsets[-1], F)).astype(np.float32)
        flags = np.zeros(offsets[-1], dtype=np.int64)
        for k in range(n):
            a = offsets[k] + int(rng.integers(2, lengths[k] - 8))
            if world == "order":
                X[a, 0] = 6.0
                flags[a + 1:a + 6] = 1
            else:
                flags[a + 1:a + 6] = 1
                X[a + 1:a + 6, 1] += 2.0
        res, _ = evaluate(X, flags, offsets, tr, te, 0, "cpu", 100, lambda m: None, seeds=(0,), epochs=15, hidden=16,
                          batch=32, lr=5e-3)
        a = {k: v["auroc"] for k, v in res.items()}
        if world == "order":
            check("order world: real-order GRU finds the span, the others do not",
                  a["gru_real_order_seed0"] > 0.85 and a["gru_shuffled_order_seed0"] < 0.7 and a["probe_plus_position"] < 0.7,
                  "real %.3f shuffled %.3f probe+pos %.3f" % (a["gru_real_order_seed0"], a["gru_shuffled_order_seed0"],
                                                               a["probe_plus_position"]))
        else:
            check("level world: every reader finds the span",
                  min(a.values()) > 0.85, " ".join("%s %.3f" % (k, v) for k, v in a.items()))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--generator", default="llama-2-7b-chat")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    if a.self_test:
        print("  self-test")
        raise SystemExit(0 if self_test() else 1)
    run(a.generator, a.out_dir, a.seed, a.n_boot)


if __name__ == "__main__":
    main()
