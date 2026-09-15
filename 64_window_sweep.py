"""
64_window_sweep.py -- does the reported detector depend on its layer window? (T-012)
==================================================================================================

The paper reads the residual stream after blocks 15..23 of BOTH models, the same absolute layers for
a 28-block Qwen and a 32-block LLaMA. This script keeps the reported detector exactly as it is --
triple_concat, ranks (5,64)/(5,64)/(4,64), random forest, the question-level split, five seeds -- and
moves only the window. 58 asks where single layers carry signal; this asks what the full detector
loses or gains if the nine layers are somewhere else.

WINDOWS, PRE-REGISTERED 2026-09-15 before any LLaMA per-layer result existed (TICKETS T-012).
A window is nine consecutive hidden-state indices starting at s, in 57's indexing (index 0 is the
embedding output, index l is block l-1). Peak and range use indices s..s+8; the update uses the eight
differences inside, v[s..s+7]. s = 16 is the reported window, blocks 15..23.

  Qwen   s in {16, 4, 8, 12, 14, 20}        14 = depth-matched to LLaMA's window
  LLaMA  s in {16, 4, 8, 12, 18, 19, 20, 24} 19 = depth-matched to Qwen's window

Depth-matched means the window's CENTER block sits at the same relative depth, (j + 0.5) / n_blocks,
in the other model (depth_matched_start). LLaMA's first registration used the window's START instead
(s = 18); that was corrected before any result existed and 18 stays in the grid, so neither reading
is chosen after the fact.

GATE. At s = 16 the rebuilt features must reproduce results/flatten_control (the reported detector,
same seeds) within GATE_PTS points of pooled AUROC. Not bit-exact, and it should not be: 57 ran its
own bf16 forward pass with different batching, and pooled its quantiles in numpy rather than torch.
If the gate fails, no other window is scored -- a sweep that cannot reproduce its own reference
point measures the extraction, not the window.

The main window does not change, whatever this shows.

  python 64_window_sweep.py --self-test
  python 64_window_sweep.py --dataset tydiqa_gp --model_folder llama-3.1-8b
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.abspath(os.path.join(HERE, "..", "data-alllayers"))
DEFAULT_OUT = os.path.join(HERE, "results", "window_sweep")
CONDITION = "triple_concat"
WIDTH = 9                      # hidden-state indices per window
REPORTED_START = 16            # blocks 15..23
GATE_PTS = 0.5
N_BLOCKS = {"qwen-2.5-7b-instruct": 28, "llama-3.1-8b": 32}
STARTS = {"qwen-2.5-7b-instruct": [16, 4, 8, 12, 14, 20],
          "llama-3.1-8b": [16, 4, 8, 12, 18, 19, 20, 24]}
DEPTH_MATCHED = {"qwen-2.5-7b-instruct": 14, "llama-3.1-8b": 19}


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_mods = {}


def mods():
    if not _mods:
        _mods["s63"] = _load("s63", "63_rank_sweep.py")
        _mods.update(_mods["s63"].mods())          # s61, s43, s26, s44 -- the modules 63 uses
    return _mods


# ---------------------------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------------------------

def depth_matched_start(n_blocks_from, n_blocks_to, start_from=REPORTED_START):
    """Start index in the target model whose window center has the same relative depth.
    A window starting at index s covers blocks s-1..s+7, so its center block is s+3."""
    rel = (start_from + 3 + 0.5) / n_blocks_from
    center_to = int(np.floor(rel * n_blocks_to))    # j with (j + 0.5) / n nearest rel, ties upward
    return center_to - 3


def blocks_of(s):
    return [s - 1, s + WIDTH - 2]


def window_feats(raw, s):
    """The three sub-tensors of triple_concat at window start s, built as
    44_eval_phase3.load_new_dataset builds them from the pinned features: static is q95 then q05 on
    the feature axis, velocity v95 then v05."""
    n_idx = raw["core"].shape[1]
    if not (1 <= s and s + WIDTH <= n_idx):
        raise ValueError("window start %d does not fit %d hidden-state indices" % (s, n_idx))
    c = slice(s, s + WIDTH)
    v = slice(s, s + WIDTH - 1)
    feats = {"core": raw["core"][:, c].astype(np.float32),
             "static": np.concatenate([raw["q95"][:, c], raw["q05"][:, c]], axis=2).astype(np.float32),
             "velocity": np.concatenate([raw["v95"][:, v], raw["v05"][:, v]], axis=2).astype(np.float32)}
    bad = {k: int((~np.isfinite(a)).sum()) for k, a in feats.items()}
    if any(bad.values()):
        raise SystemExit("non-finite entries in window %d: %s. 57 writes NaN for an empty completion "
                         "and inf where float16 overflowed; neither can be fed to the detector."
                         % (s, bad))
    return feats


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

def load_alllayers(in_dir, model_folder, dataset):
    """57's arrays in the ORIGINAL beam order (57 sorts by question and records beam_row). The update
    comes from the same file when it was extracted with everything else (LLaMA), or from the
    _velocity file when core and static already existed (Qwen); the two must list the same beams."""
    base = os.path.join(in_dir, model_folder)
    main = np.load(os.path.join(base, "%s_alllayers.npz" % dataset))
    rows = np.asarray(main["beam_row"])
    raw = {k: main[k] for k in ("core", "q95", "q05")}
    if "v95" in main.files:
        raw["v95"], raw["v05"] = main["v95"], main["v05"]
    else:
        vpath = os.path.join(base, "%s_alllayers_velocity.npz" % dataset)
        if not os.path.exists(vpath):
            raise SystemExit("%s has no update and %s does not exist -- run 57 with --streams velocity"
                             % (os.path.join(base, "%s_alllayers.npz" % dataset), vpath))
        vel = np.load(vpath)
        if not np.array_equal(np.asarray(vel["beam_row"]), rows):
            raise SystemExit("beam_row differs between %s_alllayers.npz and its _velocity file"
                             % dataset)
        raw["v95"], raw["v05"] = vel["v95"], vel["v05"]
    order = np.argsort(rows, kind="stable")
    if not np.array_equal(rows[order], np.arange(len(rows))):
        raise SystemExit("beam_row is not a permutation of 0..N-1: 57 ran with --limit, or the file "
                         "is truncated")
    raw = {k: v[order] for k, v in raw.items()}
    return raw, np.asarray(main["label"])[order], np.asarray(main["prompt_id"])[order]


def feature_agreement(new, pinned, n_sample=200000, seed=0):
    """How close 57's window-16 features are to the pinned ones: correlation and median relative
    difference over a random sample of entries. Reported, not gated -- the gate is the AUROC."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in new:
        a, b = new[k].reshape(-1), pinned[k].reshape(-1)
        assert a.shape == b.shape, (k, new[k].shape, pinned[k].shape)
        idx = rng.integers(0, a.size, size=min(n_sample, a.size))
        x, z = a[idx].astype(np.float64), b[idx].astype(np.float64)
        out[k] = {"corr": float(np.corrcoef(x, z)[0, 1]) if x.std() > 0 and z.std() > 0 else None,
                  "median_rel_diff": float(np.median(np.abs(x - z) / (np.abs(z) + 1e-3)))}
    return out


# ---------------------------------------------------------------------------------------------
# Gate and summary
# ---------------------------------------------------------------------------------------------

def gate(reported, ref_path):
    """reported: the s = 16 result. Passes if its pooled mean is within GATE_PTS of the flatten
    control's hosvd arm, which IS the reported detector on the pinned features."""
    if not os.path.exists(ref_path):
        return {"passed": False, "reason": "no reference at %s" % ref_path}
    with open(ref_path) as f:
        ref = json.load(f)["summary"]["hosvd"]
    diff = 100.0 * (reported["pooled_mean"] - ref["pooled_mean"])
    ref_seed = {r["seed"]: r["pooled_auroc"] for r in ref["per_seed"]}
    per_seed = {str(r["seed"]): round(100.0 * (r["pooled_auroc"] - ref_seed[r["seed"]]), 3)
                for r in reported["per_seed"] if r["seed"] in ref_seed}
    return {"passed": bool(abs(diff) <= GATE_PTS), "diff_pts": round(float(diff), 3),
            "reference_mean": float(ref["pooled_mean"]), "rebuilt_mean": float(reported["pooled_mean"]),
            "per_seed_diff_pts": per_seed, "tolerance_pts": GATE_PTS}


def summarize(res, model_folder):
    by_start = sorted(res.values(), key=lambda r: r["start"])
    rep = res[str(REPORTED_START)]
    best = max(by_start, key=lambda r: r["pooled_mean"])
    dm = res.get(str(DEPTH_MATCHED.get(model_folder)))
    return {"curve": [[r["start"], r["blocks"], float(r["pooled_mean"]), float(r["pooled_std"])]
                      for r in by_start],
            "reported_mean": float(rep["pooled_mean"]), "reported_std": float(rep["pooled_std"]),
            "best_start": int(best["start"]), "best_mean": float(best["pooled_mean"]),
            "reported_minus_best_pts": round(100.0 * (rep["pooled_mean"] - best["pooled_mean"]), 3),
            "depth_matched_start": None if dm is None else int(dm["start"]),
            "depth_matched_minus_reported_pts": (None if dm is None else
                                                 round(100.0 * (dm["pooled_mean"] - rep["pooled_mean"]), 3))}


def report(summ):
    print("\n  start  blocks     AUROC")
    for s, b, mean, std in summ["curve"]:
        tag = " <- reported" if s == REPORTED_START else (
            " <- depth-matched" if s == summ["depth_matched_start"] else "")
        print("  %5d  %2d-%-2d   %.4f +- %.4f%s" % (s, b[0], b[1], mean, std, tag))
    print("  reported minus best: %+.2f pts (best start %d)"
          % (summ["reported_minus_best_pts"], summ["best_start"]))
    if summ["depth_matched_minus_reported_pts"] is not None:
        print("  depth-matched minus reported: %+.2f pts" % summ["depth_matched_minus_reported_pts"])


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------

def run(dataset, model_folder, data_dir, in_dir, out_dir, readout, seeds=None, starts=None,
        use_gate=True, resume=False):
    m = mods()
    import methods.base as B
    c = B.canonical()
    s61, s63 = m["s61"], m["s63"]
    seeds = [int(x) for x in (seeds or c["seeds"])]
    spec = m["s44"].CONDITION_SPECS[CONDITION]
    preregistered = starts is None
    starts = list(starts or STARTS[model_folder])
    if REPORTED_START in starts:                       # the gate first, so a failure costs one window
        starts = [REPORTED_START] + [s for s in starts if s != REPORTED_START]
    dst = os.path.join(out_dir, "window_%s_%s_%s.json" % (model_folder, dataset, readout))
    ref_path = os.path.join(HERE, "results", "flatten_control",
                            "flatten_%s_%s_%s.json" % (model_folder, dataset, readout))
    s61._retry_os(lambda: os.makedirs(out_dir, exist_ok=True), "creating %s" % out_dir)

    pinned, y, pid, is_known = m["s44"].load_new_dataset(dataset, data_dir, model_folder,
                                                         condition=CONDITION)
    y, pid = np.asarray(y, dtype=int), np.asarray(pid)
    raw, y57, pid57 = load_alllayers(in_dir, model_folder, dataset)
    if not (np.array_equal(y57.astype(int), y) and np.array_equal(pid57.astype(np.int64), pid.astype(np.int64))):
        raise SystemExit("labels or question ids in 57's file do not match the pinned features row for "
                         "row -- the two cannot be compared")
    print("  [%s/%s] %d answers | %d hidden-state indices | starts %s%s | readout %s"
          % (model_folder, dataset, len(y), raw["core"].shape[1], starts,
             "" if preregistered else " (NOT the pre-registered grid)", readout), flush=True)

    res, extra = {}, {}
    if resume and os.path.exists(dst):
        with open(dst) as f:
            old = json.load(f)
        res, extra = old.get("results", {}), {k: old[k] for k in ("gate", "feature_agreement") if k in old}
        print("  resuming: %d windows already scored" % len(res), flush=True)

    def payload(complete, more=None):
        d = {"dataset": dataset, "model_folder": model_folder, "readout": readout,
             "condition": CONDITION, "spec": [list(s) for s in spec], "seeds": seeds,
             "protocol": "question-level (paper protocol)", "window_width": WIDTH,
             "indexing": "57: index 0 = embedding output, index l = block l-1; update v[s..s+7]",
             "preregistered_grid": preregistered, "results": res, "complete": complete}
        d.update(extra)
        d.update(more or {})
        return d

    if REPORTED_START in starts and "feature_agreement" not in extra:
        extra["feature_agreement"] = feature_agreement(window_feats(raw, REPORTED_START), pinned)
        print("  window 16 vs pinned features: %s" % json.dumps(extra["feature_agreement"]), flush=True)
    del pinned

    t_start = time.time()
    for i, s in enumerate(starts, 1):
        if str(s) in res:
            continue
        t0 = time.time()
        r = s63.score_setting(m, c, window_feats(raw, s), spec, y, pid, is_known, seeds, readout)
        r.update({"start": s, "blocks": blocks_of(s), "seconds": round(time.time() - t0, 1)})
        res[str(s)] = r
        print("  [%d/%d] start %-2d blocks %2d-%-2d  AUROC %.4f +- %.4f  (%.0fs)"
              % (i, len(starts), s, r["blocks"][0], r["blocks"][1], r["pooled_mean"],
                 r["pooled_std"], r["seconds"]), flush=True)
        if s == REPORTED_START:
            extra["gate"] = gate(r, ref_path)
            print("  GATE: %s" % json.dumps(extra["gate"]), flush=True)
        s61._write_result(dst, payload(False))
        if s == REPORTED_START and use_gate and not extra["gate"]["passed"]:
            print("  GATE FAILED -- no other window scored. The rebuilt features do not reproduce the "
                  "reported detector; check the feature agreement above before anything else.")
            raise SystemExit(1)

    summ = summarize(res, model_folder) if str(REPORTED_START) in res else None
    if summ:
        report(summ)
    s61._write_result(dst, payload(True, {"summary": summ,
                                          "elapsed_seconds": round(time.time() - t_start, 1)}))


# ---------------------------------------------------------------------------------------------

def self_test():
    import tempfile
    print("=" * 78)
    print("  SELF-TEST: 64_window_sweep")
    print("=" * 78)

    # Pre-registered grid: fits both models, depth-matched starts follow the stated rule.
    assert depth_matched_start(32, 28) == DEPTH_MATCHED["qwen-2.5-7b-instruct"] == 14
    assert depth_matched_start(28, 32) == DEPTH_MATCHED["llama-3.1-8b"] == 19
    for mf, grid in STARTS.items():
        assert grid[0] == REPORTED_START and DEPTH_MATCHED[mf] in grid and len(set(grid)) == len(grid)
        assert all(1 <= s and s + WIDTH <= N_BLOCKS[mf] + 1 for s in grid), (mf, grid)
    assert 18 in STARTS["llama-3.1-8b"]
    assert blocks_of(16) == [15, 23] and blocks_of(14) == [13, 21] and blocks_of(19) == [18, 26]
    print("  [PASS] grid fits both models; depth-matched by window center: Qwen 14 (blocks 13-21), "
          "LLaMA 19 (blocks 18-26); 18 kept")

    # window_feats on arrays whose values are their own index: the slices must land where the
    # pinned pipeline's features come from.
    N, L1, D = 3, 29, 2
    lay = np.arange(L1, dtype=np.float16)[None, :, None] * np.ones((N, 1, D), dtype=np.float16)
    raw = {"core": lay, "q95": lay + 0.5, "q05": lay - 0.5,
           "v95": 100 + lay[:, :L1 - 1], "v05": 200 + lay[:, :L1 - 1]}
    f = window_feats(raw, 16)
    assert f["core"].shape == (N, 9, D) and list(f["core"][0, :, 0]) == list(range(16, 25))
    assert f["static"].shape == (N, 9, 2 * D) and f["static"][0, 0, 0] == 16.5 and f["static"][0, 0, D] == 15.5
    assert f["velocity"].shape == (N, 8, 2 * D)
    assert list(f["velocity"][0, :, 0]) == list(range(116, 124)) and f["velocity"][0, 0, D] == 216
    try:
        window_feats(raw, 21)
        raise AssertionError("start 21 must not fit 29 indices")
    except ValueError:
        pass
    bad = dict(raw, core=raw["core"].copy())
    bad["core"][1, 18, 0] = np.nan
    try:
        window_feats(bad, 16)
        raise AssertionError("a NaN in the window must stop the run")
    except SystemExit:
        pass
    print("  [PASS] window 16: peak/range indices 16..24, update v[16..23], q95 before q05; "
          "out-of-range start and NaN refused")

    # load_alllayers: 57 stores rows sorted by question; this must restore beam order, take the
    # update from the _velocity file when needed, and refuse mismatched or partial files.
    rng = np.random.default_rng(0)
    beams = np.arange(12)
    pid_beam = np.repeat(np.array([3, 1, 2]), 4)
    order = np.argsort(pid_beam, kind="stable")
    core = rng.standard_normal((12, L1, D)).astype(np.float16)
    vel = rng.standard_normal((12, L1 - 1, D)).astype(np.float16)
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "m"))
        np.savez(os.path.join(tmp, "m", "ds_alllayers.npz"), core=core[order], q95=core[order],
                 q05=core[order], label=(beams % 2)[order], prompt_id=pid_beam[order], beam_row=beams[order])
        np.savez(os.path.join(tmp, "m", "ds_alllayers_velocity.npz"), v95=vel[order], v05=vel[order],
                 beam_row=beams[order])
        r2, y2, p2 = load_alllayers(tmp, "m", "ds")
        assert np.array_equal(r2["core"], core) and np.array_equal(r2["v95"], vel)
        assert np.array_equal(y2, beams % 2) and np.array_equal(p2, pid_beam)
        np.savez(os.path.join(tmp, "m", "ds_alllayers_velocity.npz"), v95=vel, v05=vel, beam_row=beams)
        try:
            load_alllayers(tmp, "m", "ds")
            raise AssertionError("mismatched beam_row must be refused")
        except SystemExit:
            pass
    print("  [PASS] load_alllayers restores beam order, joins the _velocity file, refuses a mismatch")

    # Feature agreement and gate.
    a = {"core": rng.standard_normal((5, 9, 4)).astype(np.float32)}
    fa = feature_agreement(a, {"core": a["core"].copy()})
    assert abs(fa["core"]["corr"] - 1.0) < 1e-9 and fa["core"]["median_rel_diff"] == 0.0
    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "flatten.json")
        with open(ref, "w") as fh:
            json.dump({"summary": {"hosvd": {"pooled_mean": 0.8782, "per_seed": [
                {"seed": 42, "pooled_auroc": 0.88}, {"seed": 0, "pooled_auroc": 0.8764}]}}}, fh)
        ok = gate({"pooled_mean": 0.8750, "per_seed": [{"seed": 42, "pooled_auroc": 0.877}]}, ref)
        no = gate({"pooled_mean": 0.8700, "per_seed": [{"seed": 42, "pooled_auroc": 0.870}]}, ref)
        assert ok["passed"] and not no["passed"] and abs(ok["diff_pts"] + 0.32) < 1e-9
        assert not gate({"pooled_mean": 0.9, "per_seed": []}, os.path.join(tmp, "missing.json"))["passed"]
        json.dumps(ok)
    print("  [PASS] gate passes at -0.32 pts, fails at -0.82 pts and when the reference is missing")

    # End to end on a planted signal that lives only in hidden-state indices 12..20: the window
    # starting at 12 must find it and the window starting at 1 must not, through 63's scorer.
    m = mods()
    import methods.base as B
    c = B.canonical()
    nq, per_q, F = 40, 5, 12
    n = nq * per_q
    pid = np.repeat(np.arange(nq), per_q)
    y = (np.arange(n) % 2).astype(int)
    L1 = 29
    raw = {k: rng.standard_normal((n, L1 if k in ("core", "q95", "q05") else L1 - 1, F)).astype(np.float16)
           for k in ("core", "q95", "q05", "v95", "v05")}
    uf = rng.standard_normal(F).astype(np.float16)
    uf /= np.linalg.norm(uf)
    for k in raw:
        raw[k][:, 12:21] += (3.0 * y[:, None, None] * uf[None, None, :]).astype(np.float16)
    is_known, _ = m["s44"].derive_is_known(y, pid)
    spec = [(key, r_l, 4) for key, r_l, _ in m["s44"].CONDITION_SPECS[CONDITION]]
    hit = m["s63"].score_setting(m, c, window_feats(raw, 12), spec, y, pid, is_known, [42, 0], "RF")
    miss = m["s63"].score_setting(m, c, window_feats(raw, 1), spec, y, pid, is_known, [42, 0], "RF")
    assert hit["pooled_mean"] > 0.8 and miss["pooled_mean"] < 0.7, (hit["pooled_mean"], miss["pooled_mean"])
    print("  [PASS] planted signal in indices 12..20: window 12 AUROC %.3f, window 1 AUROC %.3f"
          % (hit["pooled_mean"], miss["pooled_mean"]))

    res = {str(s): {"start": s, "blocks": blocks_of(s), "pooled_mean": 0.85 + 0.001 * s,
                    "pooled_std": 0.01, "per_seed": []} for s in STARTS["llama-3.1-8b"]}
    summ = summarize(res, "llama-3.1-8b")
    report(summ)
    json.dumps(summ)
    assert summ["best_start"] == 24 and summ["depth_matched_start"] == 19
    print("  [PASS] summary and end-of-run table run on the full LLaMA grid and serialise to JSON")

    print("\n  ALL PASS")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dataset")
    p.add_argument("--model_folder")
    p.add_argument("--data-dir", default=None, help="pinned features (for labels, split and the gate)")
    p.add_argument("--in-dir", default=DEFAULT_IN, help="57's all-layer files")
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--readout", default="RF", choices=["RF", "LR"])
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--starts", nargs="+", type=int, default=None,
                   help="override the pre-registered grid (recorded as such in the output)")
    p.add_argument("--no-gate", action="store_true", help="score every window even if s=16 fails")
    p.add_argument("--resume", action="store_true", help="skip windows already in the output file")
    a = p.parse_args()

    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not a.dataset or not a.model_folder:
        raise SystemExit("--dataset and --model_folder are required (or use --self-test)")
    if a.starts is None and a.model_folder not in STARTS:
        raise SystemExit("no pre-registered grid for %s; pass --starts" % a.model_folder)
    data_dir = a.data_dir
    if not data_dir:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]
    run(a.dataset, a.model_folder, data_dir, a.in_dir, a.out_dir, a.readout, seeds=a.seeds,
        starts=a.starts, use_gate=not a.no_gate, resume=a.resume)


if __name__ == "__main__":
    main()
