"""
45_analysis_loader.py -- Session 08 shared loader + plot style for the three PI-meeting figures
=====================================================================================================
READ-ONLY with respect to every existing artifact. This module never writes to ../data/ or to
results/; the figure scripts write exclusively under analysis_out/{model_folder}/ (gitignored).

Two dataset layouts exist for LLaMA and the difference is NOT cosmetic:

  triviaqa / nq_open / tydiqa_gp -- one file, {ds}_phase2_features.npz, produced by
    42_extract_phase2.py. Keys: static_max/static_q95/static_q05 [N,9,4096] fp16,
    velocity_q95/velocity_q05 [N,8,4096] fp16, label/prompt_id/beam_idx [N] int64.

  truthfulqa -- TWO files, both inside the raw_state_store/ directory, produced by the OLDER
    Route N path (34_gate_reconstruct_or_regenerate.py), never by 42:
      raw_state_store/truthfulqa_v3_pooled.pt            -> all_emb (list of [9,4096] float32),
                                                             all_hallucination_flag, prompt_indices
      raw_state_store/velocity_kinematic_repooling.npz   -> S95/S05 [N,9,4096], V95/V05 [N,8,4096],
                                                             all float32, plus label/prompt_id
    Note the key names differ (S95 vs static_q95) and the dtype differs (float32 vs fp16).
    raw_state_store/ is therefore a PERMANENT artifact for truthfulqa, not a transient one --
    42-era stores are deleted after pooling, this one predates that pipeline and must be kept.

Provenance asymmetry is recorded in every emitted JSON: truthfulqa's tensors come from Route N
while the other three come from script 42. The pooling math (35_derive_streams.py) and generation
config are shared by construction, so the two are ASSUMED equivalent -- that assumption has not
been empirically verified and any cross-dataset claim inherits it.

Only layers 15-23 were ever retained by either path (raw_state_meta.json's layer_order is
['15'..'23','final_norm']). There is no data outside the extraction window, so no figure here can
speak to full-depth localization.

Usage (imported by 46/47/48; also runnable for its own self-test):
  python 45_analysis_loader.py --self-test
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS = ["truthfulqa", "triviaqa", "nq_open", "tydiqa_gp"]

# Consistent across all three figures AND grayscale-safe: linestyle and marker carry the identity,
# colour is redundant decoration (a PI printout is usually black and white).
DATASET_STYLE = {
    "truthfulqa": {"color": "#0072B2", "ls": "-",  "marker": "o", "label": "TruthfulQA"},
    "triviaqa":   {"color": "#D55E00", "ls": "--", "marker": "s", "label": "TriviaQA"},
    "nq_open":    {"color": "#009E73", "ls": "-.", "marker": "^", "label": "NQ-Open"},
    "tydiqa_gp":  {"color": "#CC79A7", "ls": ":",  "marker": "D", "label": "TyDiQA-GP"},
}
REFERENCE_STYLE = {
    "random_null":   {"color": "#555555", "ls": (0, (1, 1)), "marker": "x", "label": "random 64-dim null"},
    "q95_vs_q05":    {"color": "#000000", "ls": (0, (3, 1, 1, 1)), "marker": "+", "label": "q95 vs q05 (within stream)"},
}

# The taxonomy hypothesis FIG 1 tests. Recorded so the figure can report whether the split holds,
# without the plotting code assuming it does.
DATASET_TYPE = {"truthfulqa": "reasoning", "tydiqa_gp": "reasoning",
                "triviaqa": "retrieval", "nq_open": "retrieval"}

WINDOW_LAYERS = list(range(15, 24))   # the 9 retained layers; index i of a [N,9,D] tensor is layer 15+i
N_SPLITS = 5
SEED = 0

RAW_KEYS_STANDARD = {"core": "static_max", "static_q95": "static_q95", "static_q05": "static_q05",
                     "velocity_q95": "velocity_q95", "velocity_q05": "velocity_q05"}
RAW_KEYS_TRUTHFULQA = {"static_q95": "S95", "static_q05": "S05",
                       "velocity_q95": "V95", "velocity_q05": "V05"}


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_data_dir(explicit=None):
    if explicit:
        return explicit
    import yaml
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    return cfg["output"]["data_dir"]


def output_dir(model_folder="llama-3.1-8b", root=None):
    """All figure outputs live here. Deliberately NOT results/ -- that directory holds Phase 2/3
    artifacts this session must not touch, and analysis_out/ is gitignored so nothing here can be
    committed or clobbered by a git operation."""
    return os.path.join(root or os.path.join(HERE, "analysis_out"), model_folder)


def announce_outputs(paths):
    """Print every path a run will create BEFORE it computes anything (Step 3 requirement)."""
    print("=" * 78)
    print("  This run will WRITE the following paths (and nothing else):")
    for p in paths:
        print(f"    {p}")
    print("  Read-only elsewhere: ../data/** and results/** are never modified.")
    print("=" * 78, flush=True)


# ==============================================================================
# ALIGNMENT -- an assert, not an assumption (Session 08 ruling #3)
# ==============================================================================

def verify_alignment(dataset, y_a, pid_a, source_a, y_b, pid_b, source_b, n_b=None):
    """TruthfulQA's core and its streams come from two SEPARATE files written by different steps.
    Nothing in the pipeline guarantees row correspondence, so require elementwise equality of BOTH
    prompt ids and labels. Returns 'elementwise' or 'N_only'; hard-fails on genuine disagreement.

    n_b: row count of source B when it carries no ids/labels of its own (then only N can be
    compared). Ignored when y_b is present -- its length is authoritative."""
    n_b_eff = len(y_b) if y_b is not None else n_b
    if n_b_eff is None:
        raise ValueError(f"{dataset}: {source_b} supplied neither labels nor an explicit row count; "
                         f"alignment cannot be checked at all.")
    if len(y_a) != n_b_eff:
        raise ValueError(
            f"{dataset}: row-count mismatch between {source_a} (N={len(y_a)}) and {source_b} "
            f"(N={n_b_eff}). These two files do not describe the same beams; refusing to proceed.")
    if y_b is None or pid_b is None:
        print(f"  [WARN] {dataset}: {source_b} lacks ids/labels -- only N could be compared. "
              f"Figures will be stamped alignment_verified='N_only'.", flush=True)
        return "N_only"
    if not np.array_equal(np.asarray(pid_a), np.asarray(pid_b)):
        raise ValueError(
            f"{dataset}: prompt ids differ elementwise between {source_a} and {source_b}. Row "
            f"order is not shared; every downstream number would silently pair the wrong beams.")
    if not np.array_equal(np.asarray(y_a).astype(np.int64), np.asarray(y_b).astype(np.int64)):
        raise ValueError(
            f"{dataset}: labels differ elementwise between {source_a} and {source_b} despite "
            f"matching prompt ids. One of the two sources is stale.")
    return "elementwise"


# ==============================================================================
# LOADING
# ==============================================================================

def _truthfulqa_paths(data_dir, model_folder):
    store = os.path.join(data_dir, model_folder, "raw_state_store")
    return (os.path.join(store, "truthfulqa_v3_pooled.pt"),
            os.path.join(store, "velocity_kinematic_repooling.npz"))


def load_arrays(dataset, data_dir, model_folder, need=("core",), verbose=True):
    """need: any of core / static_q95 / static_q05 / velocity_q95 / velocity_q05.

    Returns dict with:
      arrays          -- {name: ndarray}; kept in their ON-DISK dtype (fp16 for script-42 datasets)
                         so a 99,600-beam tensor stays ~7GB instead of ~15GB. Cast per-slice at use.
      y, prompt_idx   -- int64 [N]
      alignment_verified, provenance, n_beams, n_prompts, channel_dim
    """
    need = tuple(need)
    unknown = [n for n in need if n not in RAW_KEYS_STANDARD]
    if unknown:
        raise ValueError(f"unknown array request {unknown}; valid: {sorted(RAW_KEYS_STANDARD)}")

    arrays = {}
    if dataset == "truthfulqa":
        import torch
        pooled_pt, vel_npz = _truthfulqa_paths(data_dir, model_folder)
        for p in (pooled_pt, vel_npz):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"truthfulqa requires {p}. This dataset never went through 42_extract_phase2.py; "
                    f"its artifacts live in raw_state_store/ (Route N path). See module docstring.")
        pooled = torch.load(pooled_pt, weights_only=False)
        y = np.asarray([int(f) for f in pooled["all_hallucination_flag"]], dtype=np.int64)
        prompt_idx = np.asarray(pooled["prompt_indices"], dtype=np.int64)
        if "core" in need:
            arrays["core"] = torch.stack(pooled["all_emb"]).float().numpy()
        del pooled

        stream_need = [n for n in need if n != "core"]
        z = np.load(vel_npz)   # lazy NpzFile
        align = verify_alignment(dataset, y, prompt_idx, os.path.basename(pooled_pt),
                                 np.asarray(z["label"], dtype=np.int64),
                                 np.asarray(z["prompt_id"], dtype=np.int64),
                                 os.path.basename(vel_npz))
        for n in stream_need:
            arrays[n] = z[RAW_KEYS_TRUTHFULQA[n]]
        provenance = "route_n (34_gate_reconstruct_or_regenerate.py); NOT script 42"
    else:
        npz_path = os.path.join(data_dir, model_folder, f"{dataset}_phase2_features.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"{dataset} requires {npz_path} (run 42_extract_phase2.py).")
        z = np.load(npz_path)   # lazy
        y = np.asarray(z["label"], dtype=np.int64)
        prompt_idx = np.asarray(z["prompt_id"], dtype=np.int64)
        for n in need:
            arrays[n] = z[RAW_KEYS_STANDARD[n]]
        # single-file layout: rows are aligned by construction (42 packs them together)
        align = "elementwise (single-file layout, aligned by construction)"
        provenance = "script 42 (42_extract_phase2.py)"

    any_arr = next(iter(arrays.values())) if arrays else None
    info = {"arrays": arrays, "y": y, "prompt_idx": prompt_idx,
            "alignment_verified": align, "provenance": provenance,
            "n_beams": int(len(y)), "n_prompts": int(len(np.unique(prompt_idx))),
            "channel_dim": int(any_arr.shape[2]) if any_arr is not None else None,
            "dataset": dataset, "dataset_type": DATASET_TYPE[dataset]}
    if verbose:
        shapes = {k: (tuple(v.shape), str(v.dtype)) for k, v in arrays.items()}
        print(f"  [{dataset}] N={info['n_beams']} beams / {info['n_prompts']} prompts  "
              f"halluc={y.mean()*100:.1f}%  alignment={align}", flush=True)
        for k, (s, d) in shapes.items():
            print(f"      {k:14s} {s} {d}", flush=True)
    return info


def concat_stream(info, which):
    """Build the stream exactly as the Phase 3 pipeline does: q95 and q05 concatenated along the
    CHANNEL axis, giving 2*D channels (8192 for LLaMA). This is what core_concat/joint_tensor
    actually combine, so it is the correct object for a 'why does combining help' figure --
    comparing 4096-channel halves instead would measure something the pipeline never uses."""
    a = info["arrays"][f"{which}_q95"].astype(np.float32)
    b = info["arrays"][f"{which}_q05"].astype(np.float32)
    return np.concatenate([a, b], axis=2)


def make_folds(y, prompt_idx, n_splits=N_SPLITS):
    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups=prompt_idx))
    for tr, va in folds:
        assert set(prompt_idx[tr].tolist()).isdisjoint(set(prompt_idx[va].tolist()))
    return folds


# ==============================================================================
# STATS
# ==============================================================================

def cluster_bootstrap_ci(stat_fn, prompt_idx, n_boot=1000, seed=SEED, alpha=0.05):
    """Resample PROMPTS with replacement (not beams) -- beams within a prompt are correlated, so a
    beam-level bootstrap would understate the interval. stat_fn(beam_index_array) -> float or nan."""
    rng = np.random.default_rng(seed)
    unique_prompts = np.unique(prompt_idx)
    idx_by_prompt = {p: np.where(prompt_idx == p)[0] for p in unique_prompts}
    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(unique_prompts, size=len(unique_prompts), replace=True)
        beam_idx = np.concatenate([idx_by_prompt[p] for p in drawn])
        v = stat_fn(beam_idx)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n_boot_valid": 0}
    return {"mean": float(np.mean(vals)),
            "ci95": [float(np.percentile(vals, 100 * alpha / 2)),
                     float(np.percentile(vals, 100 * (1 - alpha / 2)))],
            "n_boot_valid": len(vals)}


def principal_angles(U, V):
    """cos(theta_i) between the column spaces of U (D x k) and V (D x k), descending.
    Re-orthonormalizes both via QR first: the Tucker factors come from randomized SVD and are
    orthonormal to numerical tolerance, but principal angles are sensitive to drift, and QR is
    cheap relative to the SVD that produced them."""
    Uq, _ = np.linalg.qr(U)
    Vq, _ = np.linalg.qr(V)
    s = np.linalg.svd(Uq.T @ Vq, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def random_subspace_pair_angles(dim, k, seed=SEED):
    """Null reference: two independent uniformly-random k-dim subspaces of R^dim. Essential
    context -- in high dimension random subspaces are nearly orthogonal, so 'static and velocity
    are nearly orthogonal' is only interesting relative to THIS curve."""
    rng = np.random.default_rng(seed)
    A, _ = np.linalg.qr(rng.standard_normal((dim, k)))
    B, _ = np.linalg.qr(rng.standard_normal((dim, k)))
    return principal_angles(A, B)


def shared_direction_count(cosines, threshold=0.5):
    return int(np.sum(np.asarray(cosines) > threshold))


# ==============================================================================
# OUTPUT
# ==============================================================================

def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Wrote: {path}", flush=True)


def save_figure(fig, path_stem):
    os.makedirs(os.path.dirname(path_stem), exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        p = f"{path_stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        written.append(p)
        print(f"Wrote: {p}", flush=True)
    return written


def base_provenance(infos):
    """Block embedded in every figure JSON so a number can always be traced to its source."""
    return {
        "model_folder": "llama-3.1-8b-instruct",
        "window_layers": WINDOW_LAYERS,
        "window_note": ("Only layers 15-23 were retained by either extraction path; layers outside "
                        "the window do not exist on disk. Full-depth profiling requires GPU "
                        "re-extraction (deferred, not dropped)."),
        "seed": SEED, "n_splits": N_SPLITS,
        "per_dataset": {d: {"provenance": i["provenance"],
                            "alignment_verified": i["alignment_verified"],
                            "n_beams": i["n_beams"], "n_prompts": i["n_prompts"],
                            "dataset_type": DATASET_TYPE[d]} for d, i in infos.items()},
        "provenance_caveat": ("truthfulqa's tensors originate from the older Route-N path while "
                              "triviaqa/nq_open/tydiqa_gp come from script 42. Pooling math and "
                              "generation config are shared by construction, so the two paths are "
                              "ASSUMED equivalent; this has not been empirically verified."),
    }


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: 45_analysis_loader (synthetic, no cluster files, no GPU)")
    print("=" * 70)

    rng = np.random.default_rng(0)
    n_prompts, per_prompt, D = 20, 10, 64
    n = n_prompts * per_prompt
    prompt_idx = np.repeat(np.arange(n_prompts), per_prompt)
    y = (rng.random(n) < 0.5).astype(np.int64)

    folds = make_folds(y, prompt_idx, n_splits=5)
    assert len(folds) == 5
    for tr, va in folds:
        assert set(prompt_idx[tr]).isdisjoint(set(prompt_idx[va]))
    print("  [PASS] make_folds: 5 prompt-disjoint folds")

    # alignment: identical -> elementwise; permuted ids -> hard fail; flipped label -> hard fail
    assert verify_alignment("t", y, prompt_idx, "A", y.copy(), prompt_idx.copy(), "B") == "elementwise"
    try:
        bad = prompt_idx.copy(); bad[0] = bad[0] + 1
        verify_alignment("t", y, prompt_idx, "A", y, bad, "B")
        raise AssertionError("permuted prompt ids must hard-fail")
    except ValueError as e:
        assert "prompt ids differ" in str(e)
    try:
        bady = y.copy(); bady[0] = 1 - bady[0]
        verify_alignment("t", y, prompt_idx, "A", bady, prompt_idx, "B")
        raise AssertionError("flipped label must hard-fail")
    except ValueError as e:
        assert "labels differ" in str(e)
    try:
        verify_alignment("t", y, prompt_idx, "A", y[:-1], prompt_idx[:-1], "B")
        raise AssertionError("row-count mismatch must hard-fail")
    except ValueError as e:
        assert "row-count mismatch" in str(e)
    assert verify_alignment("t", y, prompt_idx, "A", None, None, "B", n_b=len(y)) == "N_only"
    try:
        verify_alignment("t", y, prompt_idx, "A", None, None, "B", n_b=len(y) - 1)
        raise AssertionError("N-only path must still hard-fail on a row-count mismatch")
    except ValueError as e:
        assert "row-count mismatch" in str(e)
    try:
        verify_alignment("t", y, prompt_idx, "A", None, None, "B")
        raise AssertionError("no labels and no row count must hard-fail, not silently pass")
    except ValueError as e:
        assert "neither labels nor an explicit row count" in str(e)
    print("  [PASS] verify_alignment: elementwise / N_only / 5 distinct hard-fail modes")

    # principal angles: identical subspace -> all cos ~1 ; orthogonal complement -> all ~0
    A, _ = np.linalg.qr(rng.standard_normal((D, 8)))
    assert np.allclose(principal_angles(A, A), 1.0, atol=1e-8)
    B = np.linalg.qr(rng.standard_normal((D, D)))[0]
    left, right = B[:, :8], B[:, 8:16]
    assert np.allclose(principal_angles(left, right), 0.0, atol=1e-8)
    print("  [PASS] principal_angles: identical->1, orthogonal->0")

    # a planted partial overlap must land strictly between, and be order-invariant
    mixed = np.concatenate([left[:, :4], right[:, :4]], axis=1)
    cos_mixed = principal_angles(left, mixed)
    assert np.allclose(cos_mixed[:4], 1.0, atol=1e-8) and np.allclose(cos_mixed[4:], 0.0, atol=1e-8), cos_mixed
    assert np.allclose(cos_mixed, principal_angles(mixed, left), atol=1e-8), "must be symmetric"
    assert shared_direction_count(cos_mixed, 0.5) == 4
    print("  [PASS] principal_angles: 4 planted shared directions recovered exactly, symmetric")

    null = random_subspace_pair_angles(4096, 64, seed=0)
    assert len(null) == 64 and null.max() < 0.5, f"random 64-dim subspaces in R^4096 should be far from parallel: max={null.max():.3f}"
    assert np.allclose(null, random_subspace_pair_angles(4096, 64, seed=0)), "null must be seed-reproducible"
    print(f"  [PASS] random null in R^4096: max cos={null.max():.4f}, shared(>0.5)={shared_direction_count(null)} (reproducible)")

    # cluster bootstrap: a constant statistic has a degenerate interval; a real one is ordered
    const = cluster_bootstrap_ci(lambda idx: 0.7, prompt_idx, n_boot=50)
    assert abs(const["mean"] - 0.7) < 1e-12 and abs(const["ci95"][0] - 0.7) < 1e-12
    var = cluster_bootstrap_ci(lambda idx: float(y[idx].mean()), prompt_idx, n_boot=200)
    assert var["ci95"][0] <= var["mean"] <= var["ci95"][1] and var["n_boot_valid"] == 200
    print(f"  [PASS] cluster_bootstrap_ci: constant->degenerate, real stat ordered "
          f"({var['ci95'][0]:.3f} <= {var['mean']:.3f} <= {var['ci95'][1]:.3f})")

    # style table must be complete and grayscale-distinguishable
    assert set(DATASET_STYLE) == set(DATASETS)
    assert len({s["ls"] for s in DATASET_STYLE.values()}) == 4, "linestyles must be unique (grayscale)"
    assert len({s["marker"] for s in DATASET_STYLE.values()}) == 4, "markers must be unique (grayscale)"
    print("  [PASS] DATASET_STYLE: all 4 datasets, unique linestyle AND marker (grayscale-safe)")

    print("\n[PASS] All self-test assertions passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return
    print("This module is a library for 46/47/48. Run with --self-test to verify it.")


if __name__ == "__main__":
    main()
