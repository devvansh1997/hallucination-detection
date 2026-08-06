"""
44_eval_phase3.py -- Session 06 Phase 3: Combination + Rank Probe + Transfer Matrix (CPU-only)
=====================================================================================================
No GPU, no regeneration -- every tensor already exists from Phase 2 (triviaqa/nq_open/tydiqa_gp's
{dataset}_phase2_features.npz) or TruthfulQA's canonical artifacts (v3 pooled.pt +
velocity_kinematic_repooling.npz, same files used throughout sessions 04/05/06).

Reuses 43_eval_phase2.py's Tucker/eval infrastructure (compute_ul_ud_randomized,
fold_pure_core_randomized, summarize_oof, paired_bootstrap_delta, original_harp_split, fit_eval)
rather than reimplementing it. Part A generalizes Phase 2's single-tensor condition evaluation
into a "core-builder" callable abstraction -- Phase 2 never needed to combine multiple feature
tensors into one classifier input, so there was nothing to fork there; this is genuinely new
logic Part A's combination conditions require.

Re-runs core-max and q-velocity's grouped-CV fresh in Part A (not reusing Phase 2's saved
summaries) -- this is necessary, not redundant: Phase 2's results/session06_phase2_*.json only
persisted summary statistics (mean AUROC, CI), never the per-beam out-of-fold score arrays, and
valid PAIRED deltas require both conditions' OOF scores under the IDENTICAL fold assignment.

CLI stages (mirrors the established per-dataset-then-combine pattern, extended for Part A's
cross-dataset "best condition" dependency that Parts B and C both need):
  --self-test
  --dataset {truthfulqa,triviaqa,nq_open,tydiqa_gp} --part a   [+ --core-pooled-pt/--velocity-meta for truthfulqa]
    [+ --condition {one of 6}                                   (parallel-job mode; writes a summary JSON +
                                                                  companion .npz per condition, not one big file)]
  --dataset {ds} --backfill-deltas                              (optional, re-runnable: once core_max/q_velocity
                                                                  are both on disk, backfills paired_deltas into
                                                                  whichever other --condition results already
                                                                  exist, without waiting for all 6)
  --dataset {ds} --combine-conditions                           (merges a dataset's 6 --condition runs)
  --combine-part-a                                              (determines the best condition)
  --dataset {triviaqa,truthfulqa} --part b                      (rank probe, needs best-condition)
  --part c                                                       (transfer matrix, needs best-condition, loads all 4)
  --combine                                                      (master table, metrics JSON, leaderboard v2, verdict)
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    import resource   # POSIX only (the cluster) -- absent on Windows (local --self-test runs).
except ImportError:
    resource = None


def peak_rss_mb():
    """Process peak resident set size so far, in MB. q_velocity OOM'd silently over 6+ hours on
    triviaqa with no completed fold to show for it (job 747801) -- wiring this into the existing
    per-fold/per-seed progress prints means a future OOM leaves a trail of *where* memory was
    building, instead of just a bare kernel kill with no diagnostic signal. None on Windows/if
    unavailable; callers must handle that."""
    if resource is None:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

HERE = os.path.dirname(os.path.abspath(__file__))

# Wall-clock reference for every "elapsed since process start" print in this script -- lets you
# tail the .out/.err file and see real progress/ETA rather than long silent stretches, per the
# explicit ask to make these long CPU jobs easy to watch.
_PROCESS_START = time.time()


def since_start():
    return time.time() - _PROCESS_START


def fmt_elapsed(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s01 = _load("s01", "26_grouped_baseline.py")
s02_band_eval = _load("s02_band_eval", "28_eval_band.py")
s02_eval = _load("s02_eval", "43_eval_phase2.py")

fit_eval = s01.fit_eval
original_harp_split = s01.original_harp_split
project_core = s01.project_core
summarize_oof = s02_eval.summarize_oof
paired_bootstrap_delta = s02_eval.paired_bootstrap_delta
compute_ul_ud_randomized = s02_eval.compute_ul_ud_randomized
fold_pure_core_randomized = s02_eval.fold_pure_core_randomized
robust_scale_3d = s02_eval.robust_scale_3d
derive_is_known = s02_eval.derive_is_known
composition_line = s02_eval.composition_line
# per-model output isolation -- defined in 43 so both scripts share one implementation
resolve_results_dir = s02_eval.resolve_results_dir
legacy_read_path = s02_eval.legacy_read_path
has_standard_features = s02_eval.has_standard_features

SEED = 0
N_SPLITS = 5
N_BOOTSTRAP = 1000
HARP_SEEDS = [42, 0, 1, 2, 3]
DATASETS = ["truthfulqa", "triviaqa", "nq_open", "tydiqa_gp"]
RANK_PROBE_DATASETS = ["triviaqa", "truthfulqa"]
RANK_PROBE_RDS = [96, 128]

# (sub-tensor keys used, r_l, r_d) per condition -- the pre-registered combination grid.
CONDITION_SPECS = {
    "core_max": [("core", 5, 64)],
    "q_static": [("static", 5, 64)],
    "q_velocity": [("velocity", 4, 64)],
    "core_concat": [("core", 5, 64), ("velocity", 4, 64)],
    "joint_tensor": [("joint", 8, 64)],
    "triple_concat": [("core", 5, 64), ("static", 5, 64), ("velocity", 4, 64)],
}
NEW_CONDITIONS = ["q_static", "core_concat", "joint_tensor", "triple_concat"]

# Which raw tensor(s) each condition actually touches -- used to avoid loading/building the
# other 1-3 tensor types when a --condition job only needs a subset. "joint" needs both static
# and velocity's underlying npz fields to construct (concat), even though the condition itself
# only uses the resulting stacked tensor, not static/velocity individually.
CONDITION_RAW_NEEDS = {
    "core_max": {"core"}, "q_static": {"static"}, "q_velocity": {"velocity"},
    "core_concat": {"core", "velocity"}, "joint_tensor": {"joint"},
    "triple_concat": {"core", "static", "velocity"},
}

# Rough relative Tucker-fitting cost per condition, relative to a single-tensor fit like
# core_max (=1): core_concat/triple_concat do 2/3 separate sub-fits; joint_tensor is one fit
# but over a bigger stacked tensor (L=17 vs 8-9), so roughly ~2x a single condition's cost.
# Used only to print an early, honest-effort estimate of Part A's TOTAL runtime (grouped-CV
# alone runs 6 conditions, not 1) right after the first condition finishes -- per this phase's
# explicit "TriviaQA is the heavy CPU item" instruction, one condition's fold-0 estimate alone
# understates the real commitment by roughly 10x.
CONDITION_COST_WEIGHTS = {"core_max": 1, "q_static": 1, "q_velocity": 1,
                           "core_concat": 2, "joint_tensor": 2, "triple_concat": 3}
TOTAL_COST_WEIGHT = sum(CONDITION_COST_WEIGHTS.values())

# HARP published cross-dataset transfer matrix, LLaMA-3.1-8B (source -> target rows, as given).
HARP_TRANSFER_PUBLISHED = {
    "nq_open":    {"nq_open": 89.4, "truthfulqa": 77.9, "triviaqa": 91.7, "tydiqa_gp": 81.6},
    "truthfulqa": {"nq_open": 76.2, "truthfulqa": 88.5, "triviaqa": 91.9, "tydiqa_gp": 81.2},
    "triviaqa":   {"nq_open": 89.2, "truthfulqa": 84.9, "triviaqa": 92.9, "tydiqa_gp": 80.7},
    "tydiqa_gp":  {"nq_open": 82.0, "truthfulqa": 80.9, "triviaqa": 91.8, "tydiqa_gp": 86.6},
}


# ==============================================================================
# FEATURE TENSORS PER CONDITION -- builds the [N,L,8192-or-4096] tensor(s) a spec needs
# ==============================================================================

def build_feats(core_raw, static_q95, static_q05, velocity_q95, velocity_q05):
    S_concat = np.concatenate([static_q95, static_q05], axis=2).astype(np.float32)
    V_concat = np.concatenate([velocity_q95, velocity_q05], axis=2).astype(np.float32)
    joint = np.concatenate([S_concat, V_concat], axis=1)
    return {"core": core_raw.astype(np.float32), "static": S_concat, "velocity": V_concat, "joint": joint}


def make_grouped_builder(spec, feats):
    """spec: list of (tensor_key, r_l, r_d). Returns core_builder(tr_idx, seed) -> (N, dim)
    array, concatenating sub-Tucker projections in spec order when len(spec) > 1 (the
    'straight concat, no residualization' combination conditions)."""
    def builder(tr, seed):
        parts = [fold_pure_core_randomized(feats[key], tr, r_l, r_d, seed + i)
                 for i, (key, r_l, r_d) in enumerate(spec)]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
    return builder


# ==============================================================================
# GENERIC GROUPED-CV / HARP EVALUATORS -- parameterized by a core-builder callable instead of
# a fixed tensor, since Part A's combination conditions can't be expressed as a single tensor.
# ==============================================================================

def run_grouped_generic(core_builder, y, prompt_idx, folds, seed=SEED, label="condition"):
    n_beams = len(y)
    oof_rf = np.full(n_beams, np.nan); oof_lr = np.full(n_beams, np.nan)
    fold_rf, fold_lr = [], []
    fold_times = []
    n_folds = len(folds)
    for fold_i, (tr, va) in enumerate(folds):
        t0 = time.time()
        rss0 = peak_rss_mb()
        rss0_str = f"  peakRSS={rss0:.0f}MB" if rss0 is not None else ""
        print(f"  [{label}] fold {fold_i+1}/{n_folds} starting  [job elapsed {fmt_elapsed(since_start())}]"
              f"{rss0_str}", flush=True)
        core = core_builder(tr, seed + fold_i)
        rf_scores = fit_eval("RF", core[tr], y[tr], core[va], seed + fold_i)
        oof_rf[va] = rf_scores; fold_rf.append(float(roc_auc_score(y[va], rf_scores)))
        lr_scores = fit_eval("LR", core[tr], y[tr], core[va], seed + fold_i)
        oof_lr[va] = lr_scores; fold_lr.append(float(roc_auc_score(y[va], lr_scores)))
        elapsed = time.time() - t0
        fold_times.append(elapsed)
        avg = sum(fold_times) / len(fold_times)
        eta = avg * (n_folds - fold_i - 1)
        rss = peak_rss_mb()
        rss_str = f"  peakRSS={rss:.0f}MB" if rss is not None else ""
        print(f"  [{label}] fold {fold_i+1}/{n_folds} done in {elapsed:.1f}s (RF={fold_rf[-1]:.4f}, "
              f"LR={fold_lr[-1]:.4f})  -- ETA this condition: {eta:.0f}s  "
              f"[job elapsed {fmt_elapsed(since_start())}]{rss_str}", flush=True)
    return ({"RF": summarize_oof(oof_rf, y, prompt_idx, fold_rf, seed),
             "LR": summarize_oof(oof_lr, y, prompt_idx, fold_lr, seed)},
            {"RF": oof_rf, "LR": oof_lr})


# Which unit the 75/25 split inside the known group is taken over.
#   "question" -- the paper's protocol (S3.1, "the set of all inputs"). Our default; unchanged.
#   "answer"   -- what HARP's RELEASED code actually does (main.py:259 -> utils.split_data ->
#                 torch.randperm over the flat per-answer list).
# The second exists so OUR method can be measured under THEIR protocol, which is the fourth cell
# of the method x protocol grid. Set once from --split-unit before any run; never mutated after.
SPLIT_UNIT = "question"


def answer_level_harp_split(is_known, prompt_idx, N, seed):
    """HARP's released split: pool the known group's ANSWERS into one list, shuffle, cut 75/25.

    Mirrors utils.split_data using numpy, with the same save/seed/restore discipline
    original_harp_split uses so neither disturbs the global RNG. Unknown prompts still go wholly
    to valid, exactly as main.py:264 does -- ONLY the known group's split unit differs.

    A known question lands entirely on one side only 0.75^10 = 5.6% of the time, so ~94% appear on
    both. That is the leakage; the caller asserts it actually happened."""
    known_prompts = set(np.where(is_known)[0].tolist())
    known_rows = np.array([i for i in range(N) if prompt_idx[i] in known_prompts], dtype=int)
    unknown_rows = np.array([i for i in range(N) if prompt_idx[i] not in known_prompts], dtype=int)

    rng_state = np.random.get_state()
    np.random.seed(seed)
    perm = np.random.permutation(len(known_rows))
    np.random.set_state(rng_state)

    s = int(len(known_rows) * 0.75)
    t_idx = np.sort(known_rows[perm[:s]])
    v_idx = np.sort(np.concatenate([known_rows[perm[s:]], unknown_rows]))
    return t_idx, v_idx


def harp_split(is_known, prompt_idx, N, seed):
    """Dispatch on SPLIT_UNIT so every caller of the HARP protocol honours the same choice."""
    if SPLIT_UNIT == "answer":
        return answer_level_harp_split(is_known, prompt_idx, N, seed)
    return original_harp_split(is_known, prompt_idx, N, seed=seed)


def run_harp_generic(builders, y, prompt_idx, is_known, seeds=HARP_SEEDS, label_prefix=""):
    n_beams = len(y)
    per_seed = {name: [] for name in builders}
    n_seeds = len(seeds)
    n_conditions = len(builders)
    for seed_i, seed in enumerate(seeds):
        t_idx, v_idx = harp_split(is_known, prompt_idx, n_beams, seed)
        tq = set(prompt_idx[t_idx].tolist())
        vq = set(prompt_idx[v_idx].tolist())
        overlap = len(tq & vq)
        if SPLIT_UNIT == "question":
            assert overlap == 0, f"question-level split leaked {overlap} prompts across sides"
        else:
            # Positive check, not a formality: if the answer-level arm somehow produced a clean
            # question split we would be measuring the SAME protocol twice and reporting the
            # comparison as if it differed. Expect ~94% of known prompts on both sides.
            assert overlap > 0.8 * len(tq), (
                f"answer-level split put only {overlap}/{len(tq)} train prompts on both sides; "
                f"expected ~94% -- this is not HARP's released protocol")
        print(f"  [{label_prefix}] HARP({SPLIT_UNIT}) seed {seed_i+1}/{n_seeds} (seed={seed}): "
              f"n_train={len(t_idx)}  n_valid={len(v_idx)}  prompts_on_both_sides={overlap}  "
              f"[job elapsed {fmt_elapsed(since_start())}]", flush=True)
        for cond_i, (name, builder) in enumerate(builders.items()):
            t0 = time.time()
            core = builder(t_idx, seed)
            row = {"seed": seed, "n_train": int(len(t_idx)), "n_valid": int(len(v_idx))}
            for clf in ("RF", "LR"):
                scores = fit_eval(clf, core[t_idx], y[t_idx], core[v_idx], seed)
                row[clf] = float(roc_auc_score(y[v_idx], scores))
            per_seed[name].append(row)
            rss = peak_rss_mb()
            rss_str = f"  peakRSS={rss:.0f}MB" if rss is not None else ""
            print(f"    [{label_prefix}] seed {seed_i+1}/{n_seeds}, condition {cond_i+1}/{n_conditions} "
                  f"({name}): RF={row['RF']:.4f} LR={row['LR']:.4f}  ({time.time()-t0:.1f}s){rss_str}", flush=True)
    summary = {}
    for name, rows in per_seed.items():
        summary[name] = {"per_seed": rows,
                          "RF_mean": float(np.mean([r["RF"] for r in rows])),
                          "RF_std": float(np.std([r["RF"] for r in rows])),
                          "LR_mean": float(np.mean([r["LR"] for r in rows])),
                          "LR_std": float(np.std([r["LR"] for r in rows]))}
    return summary


# ==============================================================================
# PART A -- COMBINATION GRID
# ==============================================================================

ALL_CONDITIONS = ["core_max", "q_velocity"] + NEW_CONDITIONS


def compute_folds(feats, y, prompt_idx):
    """GroupKFold's split only depends on `groups` (prompt_idx) and the array LENGTH, never on
    feature values -- so this is byte-identical whether computed once in a sequential run or
    independently in N separate parallel condition-jobs, as long as y/prompt_idx match (which
    they do, since every job loads the same saved dataset). Deliberately does NOT hardcode
    feats["core"]: selective loading (see _needed_raw_types) means "core" is only present when
    a condition actually needs it -- q_velocity/q_static/joint_tensor jobs never load it at all,
    so indexing it here unconditionally is a real bug (this exact KeyError crashed a real
    q_velocity job), not a hypothetical one. Any tensor in feats has the right length; which one
    is arbitrary."""
    any_tensor = next(iter(feats.values()))
    folds = list(GroupKFold(n_splits=N_SPLITS).split(any_tensor, y, groups=prompt_idx))
    for tr, va in folds:
        assert set(prompt_idx[tr].tolist()).isdisjoint(set(prompt_idx[va].tolist()))
    return folds


def run_condition_only(dataset_name, cond_name, feats, y, prompt_idx, is_known, folds):
    """The unit of work for running Part A's 6 conditions as separate parallel jobs instead of
    one long sequential run: grouped-CV + HARP for exactly ONE condition, nothing else. Each
    condition fits its own Tucker basis + classifiers on the same read-only feature tensors, and
    HARP's per-seed splits are a pure function of (is_known, prompt_idx, seed) -- no state is
    shared across conditions, so this is safe to run concurrently in as many separate processes
    as you have allocations for, with results identical to a sequential run."""
    print(f"[{dataset_name}/{cond_name}] grouped-CV (5 folds)  [job elapsed {fmt_elapsed(since_start())}] ...",
          flush=True)
    t0 = time.time()
    builder = make_grouped_builder(CONDITION_SPECS[cond_name], feats)
    grouped_summary, oof = run_grouped_generic(builder, y, prompt_idx, folds, SEED, f"{dataset_name}/{cond_name}")
    print(f"[{dataset_name}/{cond_name}] grouped-CV complete: pooled RF AUROC="
          f"{grouped_summary['RF']['pooled_oof_auroc']:.4f}  ({time.time()-t0:.0f}s)  "
          f"[job elapsed {fmt_elapsed(since_start())}]", flush=True)

    print(f"[{dataset_name}/{cond_name}] HARP protocol ({len(HARP_SEEDS)} seeds)  "
          f"[job elapsed {fmt_elapsed(since_start())}] ...", flush=True)
    t1 = time.time()
    harp_one = run_harp_generic({cond_name: builder}, y, prompt_idx, is_known, seeds=HARP_SEEDS,
                                 label_prefix=f"{dataset_name}/{cond_name}")[cond_name]
    print(f"[{dataset_name}/{cond_name}] HARP complete ({time.time()-t1:.0f}s)  "
          f"[job elapsed {fmt_elapsed(since_start())}]", flush=True)

    return {"dataset": dataset_name, "condition": cond_name, "grouped_summary": grouped_summary,
            "oof_rf": oof["RF"].tolist(), "oof_lr": oof["LR"].tolist(), "harp": harp_one}


def _oof_npz_filename(dataset_name, cond_name):
    return f"session06_phase3_partA_{dataset_name}_{cond_name}_oof.npz"


def write_condition_result(dataset_name, cond_name, result, results_dir):
    """Splits run_condition_only()'s result into a small summary JSON + a companion .npz for the
    per-beam OOF score vectors. A single condition's OOF arrays are one-per-beam (99,600 beams
    for TriviaQA) -- embedded inline as JSON lists this bloated a single condition's result file
    to ~3.3MB/199k lines (session06_phase3_partA_triviaqa_core_concat.json), unreadable as a
    normal text file. Every consumer of these arrays (combine_conditions, backfill_paired_deltas)
    only ever needs them as numpy arrays, never as JSON text, so there's no reason to pay the
    JSON-list serialization cost at all -- .npz is both smaller and faster to load."""
    oof_filename = _oof_npz_filename(dataset_name, cond_name)
    os.makedirs(results_dir, exist_ok=True)   # model-scoped dir may not exist yet on a first run
    oof_path = os.path.join(results_dir, oof_filename)
    np.savez(oof_path, oof_rf=np.asarray(result["oof_rf"], dtype=np.float64),
              oof_lr=np.asarray(result["oof_lr"], dtype=np.float64))
    summary = {k: v for k, v in result.items() if k not in ("oof_rf", "oof_lr")}
    summary["oof_npz"] = oof_filename
    out_path = os.path.join(results_dir, f"session06_phase3_partA_{dataset_name}_{cond_name}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return out_path, oof_path


def load_condition_result(dataset_name, cond_name, results_dir):
    """Inverse of write_condition_result(): reads the summary JSON, reattaches oof_rf/oof_lr from
    the companion .npz as numpy arrays. Returns (result_dict_with_oof_arrays, json_path) or
    (None, json_path) if the condition hasn't been run yet.

    Transparently migrates pre-fix files: real result JSONs already exist on disk (e.g. a real
    TriviaQA core_concat run) written by the OLD code, with oof_rf/oof_lr embedded directly and
    no "oof_npz" key -- 7+ hours of real compute already paid for. Re-splitting them here on
    first load (write the npz, rewrite the JSON summary-only) means the fix applies without
    forcing a wasteful recompute of anything already finished."""
    p = legacy_read_path(f"session06_phase3_partA_{dataset_name}_{cond_name}.json", results_dir)
    if not os.path.exists(p):
        return None, p
    with open(p) as f:
        result = json.load(f)
    # Migrate/resolve relative to the dir the file was actually FOUND in, not results_dir -- for a
    # legacy flat-results file those differ, and writing the .npz to the model-scoped dir while
    # re-reading the legacy JSON would leave the reread still missing "oof_npz" (KeyError below).
    found_dir = os.path.dirname(p) or "."
    if "oof_npz" not in result:
        print(f"  [migrate] {p}: old format (OOF arrays embedded inline) -- splitting into "
              f".npz now, no recompute needed", flush=True)
        write_condition_result(dataset_name, cond_name, result, found_dir)
        with open(p) as f:
            result = json.load(f)
    oof_path = os.path.join(found_dir, result["oof_npz"])
    oof_npz = np.load(oof_path)
    result["oof_rf"] = oof_npz["oof_rf"]
    result["oof_lr"] = oof_npz["oof_lr"]
    return result, p


def backfill_paired_deltas(dataset_name, results_dir, y, prompt_idx):
    """Every non-baseline condition's paired_deltas are measured vs core_max and vs q_velocity --
    but Part A's 6 conditions run as independent parallel jobs with no fixed completion order, so
    a condition (e.g. core_concat) routinely finishes and gets written to disk before one or both
    baselines have. combine_conditions() requires all 6 present before computing any deltas at
    all, which means an already-finished condition's result sits with no paired_deltas for as
    long as its baselines are still running -- exactly what happened to TriviaQA's core_concat.

    This re-scans whichever per-condition files exist, and as soon as BOTH baselines are present,
    backfills vs_core_max/vs_q_velocity into every other available condition's own JSON (in
    place) -- without waiting for all 6. Safe to call repeatedly as more conditions land; picks
    up whatever's newly available each time. Purely an interim convenience for visibility while
    jobs are still in flight -- combine_conditions() still recomputes paired_deltas from scratch
    into the merged dataset-level file once all 6 exist, so this is never the source of truth."""
    core_max, _ = load_condition_result(dataset_name, "core_max", results_dir)
    q_velocity, _ = load_condition_result(dataset_name, "q_velocity", results_dir)
    if core_max is None or q_velocity is None:
        missing = [n for n, r in (("core_max", core_max), ("q_velocity", q_velocity)) if r is None]
        print(f"[{dataset_name}] backfill-deltas: baseline(s) not on disk yet ({missing}) -- "
              f"nothing to backfill.")
        return {}

    core_max_rf = np.asarray(core_max["oof_rf"])
    q_velocity_rf = np.asarray(q_velocity["oof_rf"])
    written = {}
    for name in NEW_CONDITIONS:
        cond_result, cond_path = load_condition_result(dataset_name, name, results_dir)
        if cond_result is None:
            print(f"[{dataset_name}] backfill-deltas: {name} not on disk yet -- skipping.")
            continue
        cond_rf = np.asarray(cond_result["oof_rf"])
        deltas = {
            "vs_core_max": {
                "pooled": paired_bootstrap_delta(cond_rf, core_max_rf, y, prompt_idx, N_BOOTSTRAP, SEED, False),
                "within_prompt": paired_bootstrap_delta(cond_rf, core_max_rf, y, prompt_idx, N_BOOTSTRAP, SEED, True),
            },
            "vs_q_velocity": {
                "pooled": paired_bootstrap_delta(cond_rf, q_velocity_rf, y, prompt_idx, N_BOOTSTRAP, SEED, False),
                "within_prompt": paired_bootstrap_delta(cond_rf, q_velocity_rf, y, prompt_idx, N_BOOTSTRAP, SEED, True),
            },
        }
        cond_summary = {k: v for k, v in cond_result.items() if k not in ("oof_rf", "oof_lr")}
        cond_summary["paired_deltas"] = deltas
        with open(cond_path, "w") as f:
            json.dump(cond_summary, f, indent=2, default=str)
        written[name] = deltas
        print(f"[{dataset_name}] backfill-deltas: wrote paired_deltas into {cond_path}  "
              f"(vs core_max within-p={deltas['vs_core_max']['within_prompt']['mean_delta']:.4f}, "
              f"vs q_velocity within-p={deltas['vs_q_velocity']['within_prompt']['mean_delta']:.4f})")
    return written


def combine_conditions(dataset_name, per_condition, y, prompt_idx):
    """per_condition: dict cond_name -> a run_condition_only() result (freshly computed or
    reloaded from disk -- doesn't matter which). Assembles the exact schema a sequential Part A
    run produces, so --combine-part-a downstream doesn't care whether Part A ran as one job or
    six parallel condition-jobs."""
    comp = composition_line(dataset_name, y, prompt_idx)
    missing = [c for c in ALL_CONDITIONS if c not in per_condition]
    if missing:
        raise ValueError(f"Missing condition results for {dataset_name}: {missing}")

    grouped = {name: per_condition[name]["grouped_summary"] for name in ALL_CONDITIONS}
    oofs = {name: {"RF": np.asarray(per_condition[name]["oof_rf"]),
                    "LR": np.asarray(per_condition[name]["oof_lr"])} for name in ALL_CONDITIONS}
    harp = {name: per_condition[name]["harp"] for name in ALL_CONDITIONS}

    print(f"\n[{dataset_name}] Computing paired bootstrap deltas ({N_BOOTSTRAP} reps x 4 conditions "
          f"x 2 baselines x 2 metrics) ...", flush=True)
    paired_deltas = {}
    for name in NEW_CONDITIONS:
        t_delta0 = time.time()
        paired_deltas[name] = {
            "vs_core_max": {
                "pooled": paired_bootstrap_delta(oofs[name]["RF"], oofs["core_max"]["RF"], y, prompt_idx,
                                                  N_BOOTSTRAP, SEED, False),
                "within_prompt": paired_bootstrap_delta(oofs[name]["RF"], oofs["core_max"]["RF"], y, prompt_idx,
                                                          N_BOOTSTRAP, SEED, True),
            },
            "vs_q_velocity": {
                "pooled": paired_bootstrap_delta(oofs[name]["RF"], oofs["q_velocity"]["RF"], y, prompt_idx,
                                                  N_BOOTSTRAP, SEED, False),
                "within_prompt": paired_bootstrap_delta(oofs[name]["RF"], oofs["q_velocity"]["RF"], y, prompt_idx,
                                                          N_BOOTSTRAP, SEED, True),
            },
        }
        print(f"  [{dataset_name}] {name} vs core-max: pooled={paired_deltas[name]['vs_core_max']['pooled']['mean_delta']:.4f} "
              f"within-p={paired_deltas[name]['vs_core_max']['within_prompt']['mean_delta']:.4f}")
        print(f"  [{dataset_name}] {name} vs q-velocity: pooled={paired_deltas[name]['vs_q_velocity']['pooled']['mean_delta']:.4f} "
              f"within-p={paired_deltas[name]['vs_q_velocity']['within_prompt']['mean_delta']:.4f}  "
              f"({time.time()-t_delta0:.1f}s)", flush=True)

    return {"dataset": dataset_name, "composition": comp, "grouped": grouped,
            "paired_deltas": paired_deltas, "harp": harp}


def run_part_a(dataset_name, feats, y, prompt_idx, is_known):
    """Sequential convenience path (unchanged behavior/output from before this refactor) -- runs
    all 6 conditions one after another in this single process via the same run_condition_only()/
    combine_conditions() building blocks the parallel per-condition CLI mode uses."""
    comp = composition_line(dataset_name, y, prompt_idx)
    print(f"\n[{dataset_name}] composition: {comp['n_prompts']} prompts, {comp['n_beams']} beams, "
          f"{comp['hallucination_rate_pct']:.1f}% hallucinated  [job elapsed {fmt_elapsed(since_start())}]")
    folds = compute_folds(feats, y, prompt_idx)

    per_condition = {}
    for cond_i, name in enumerate(ALL_CONDITIONS):
        print(f"\n[{dataset_name}] Part A grouped: condition {cond_i+1}/{len(ALL_CONDITIONS)} = {name} "
              f"[job elapsed {fmt_elapsed(since_start())}] ...", flush=True)
        t_cond0 = time.time()
        per_condition[name] = run_condition_only(dataset_name, name, feats, y, prompt_idx, is_known, folds)
        if name == "core_max":
            cond_elapsed = time.time() - t_cond0
            est_grouped_total = cond_elapsed * TOTAL_COST_WEIGHT / CONDITION_COST_WEIGHTS["core_max"]
            print(f"\n  [{dataset_name}] core_max (grouped+HARP) took {cond_elapsed:.0f}s -- rough "
                  f"estimate for ALL 6 conditions run sequentially: ~{est_grouped_total:.0f}s "
                  f"({est_grouped_total/3600:.1f}h). Weights are approximate (see "
                  f"CONDITION_COST_WEIGHTS) -- treat as an order-of-magnitude planning number, "
                  f"not a tight bound. Consider running conditions in parallel instead (see "
                  f"--condition in the CLI help) if this is too long.")

    result = combine_conditions(dataset_name, per_condition, y, prompt_idx)
    print(f"\n[{dataset_name}] Part A COMPLETE  [job elapsed {fmt_elapsed(since_start())}]", flush=True)
    return result


def determine_best_condition(part_a_results):
    """Best = highest mean grouped within-prompt RF AUROC averaged across all 4 datasets.
    Within-prompt AUROC is this project's standing primary honest metric (pooled AUROC can be
    inflated by question-difficulty), so it's the natural tie-breaker here -- documented choice,
    not specified verbatim by the phase-3 prompt (which only says 'the best Part A condition')."""
    scores = {name: [] for name in NEW_CONDITIONS}
    for ds, r in part_a_results.items():
        for name in NEW_CONDITIONS:
            scores[name].append(r["grouped"][name]["RF"]["within_prompt"]["within_prompt_auroc"])
    means = {name: float(np.mean(v)) for name, v in scores.items()}
    best = max(means, key=means.get)
    return best, means


# ==============================================================================
# PART B -- RANK PROBE
# ==============================================================================

def run_part_b(dataset_name, feats, y, prompt_idx, best_condition):
    # run_part_b always loads full feats (main() forces condition=None for --part b), so
    # feats["core"] happens to be safe here in practice too -- routed through the same
    # compute_folds() as everywhere else regardless, for one consistent code path rather than
    # a second hardcoded reference that could silently break the same way if that assumption
    # ever changes.
    folds = compute_folds(feats, y, prompt_idx)
    results = {}
    conds = ("core_max", best_condition)
    total_runs = len(conds) * (1 + len(RANK_PROBE_RDS))
    run_i = 0
    for cond_name in conds:
        base_spec = CONDITION_SPECS[cond_name]
        base_builder = make_grouped_builder(base_spec, feats)
        run_i += 1
        print(f"\n[{dataset_name}] Part B run {run_i}/{total_runs}: {cond_name} baseline r_d=64 "
              f"[job elapsed {fmt_elapsed(since_start())}] ...", flush=True)
        base_summary, base_oof = run_grouped_generic(base_builder, y, prompt_idx, folds, SEED,
                                                       f"{dataset_name}/{cond_name}/r_d=64")
        cond_results = {"r_d_64": base_summary}
        for new_rd in RANK_PROBE_RDS:
            scaled_spec = [(key, r_l, new_rd) for (key, r_l, r_d) in base_spec]
            builder = make_grouped_builder(scaled_spec, feats)
            run_i += 1
            print(f"\n[{dataset_name}] Part B run {run_i}/{total_runs}: {cond_name} r_d={new_rd} "
                  f"[job elapsed {fmt_elapsed(since_start())}] ...", flush=True)
            summary, oof = run_grouped_generic(builder, y, prompt_idx, folds, SEED,
                                                f"{dataset_name}/{cond_name}/r_d={new_rd}")
            delta_pooled = paired_bootstrap_delta(oof["RF"], base_oof["RF"], y, prompt_idx, N_BOOTSTRAP, SEED, False)
            delta_wp = paired_bootstrap_delta(oof["RF"], base_oof["RF"], y, prompt_idx, N_BOOTSTRAP, SEED, True)
            cond_results[f"r_d_{new_rd}"] = {"summary": summary,
                                              "delta_vs_r_d_64": {"pooled": delta_pooled, "within_prompt": delta_wp}}
            print(f"  [{dataset_name}] {cond_name} r_d={new_rd} vs r_d=64: pooled delta="
                  f"{delta_pooled['mean_delta']:.4f} excl0={delta_pooled['excludes_zero']}")
        results[cond_name] = cond_results
    print(f"\n[{dataset_name}] Part B COMPLETE  [job elapsed {fmt_elapsed(since_start())}]", flush=True)
    return {"dataset": dataset_name, "results": results}


def rank_probe_verdict(part_b_by_dataset, best_condition):
    """Pre-registered adoption rule: adopt a larger rank only if its delta CI excludes zero
    (positively) on TriviaQA AND shows no significant regression on TruthfulQA."""
    verdicts = {}
    for cond_name in ("core_max", best_condition):
        for rd in RANK_PROBE_RDS:
            tri = part_b_by_dataset["triviaqa"]["results"][cond_name][f"r_d_{rd}"]["delta_vs_r_d_64"]["pooled"]
            tru = part_b_by_dataset["truthfulqa"]["results"][cond_name][f"r_d_{rd}"]["delta_vs_r_d_64"]["pooled"]
            tri_improves = tri["excludes_zero"] and tri["mean_delta"] > 0
            tru_regresses = tru["excludes_zero"] and tru["mean_delta"] < 0
            adopt = tri_improves and not tru_regresses
            verdicts[f"{cond_name}_r_d_{rd}"] = {
                "adopt": adopt, "triviaqa_delta": tri["mean_delta"], "triviaqa_improves": tri_improves,
                "truthfulqa_delta": tru["mean_delta"], "truthfulqa_regresses": tru_regresses,
            }
    return verdicts


# ==============================================================================
# PART C -- CROSS-DATASET TRANSFER MATRIX
# ==============================================================================

def fit_transfer_pipeline(X_raw_source, r_l, r_d, seed):
    """Fits robust-scale + randomized-SVD Tucker on ALL of X_raw_source (no train/val split --
    the 'training set' for a transfer-matrix row is the entire source dataset). Returns a
    transform callable that applies the SAME fitted scale params + basis to any other dataset's
    raw tensor of matching (L, D). This is a fit/apply split that fold_pure_core_randomized
    doesn't provide (it always fits and projects the same array) -- a small, necessary extension
    for cross-dataset transfer, not a fork of its math."""
    N, L, D = X_raw_source.shape
    X_flat = X_raw_source.reshape(N, L * D)
    params = s02_band_eval.fit_robust_scale(X_flat)
    X_scaled = s02_band_eval.apply_robust_scale(X_flat, params).reshape(N, L, D)
    U_L, U_D = compute_ul_ud_randomized(X_scaled, r_l, r_d, seed)

    def transform(X_raw_target):
        Nt, Lt, Dt = X_raw_target.shape
        assert (Lt, Dt) == (L, D), f"shape mismatch: fit on ({L},{D}), got ({Lt},{Dt})"
        X_flat_t = X_raw_target.reshape(Nt, Lt * Dt)
        X_scaled_t = s02_band_eval.apply_robust_scale(X_flat_t, params).reshape(Nt, Lt, Dt)
        return project_core(X_scaled_t, U_L, U_D)

    return transform


def build_transfer_transform(condition_name, feats_source, seed):
    spec = CONDITION_SPECS[condition_name]
    sub_transforms = [(key, fit_transfer_pipeline(feats_source[key], r_l, r_d, seed + i))
                       for i, (key, r_l, r_d) in enumerate(spec)]

    def transform(feats_target):
        parts = [t(feats_target[key]) for key, t in sub_transforms]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
    return transform


def run_part_c(all_feats, all_y, condition_name, seed=SEED):
    """all_feats/all_y: dict dataset_name -> feats/y. Fits on ALL of source, scores pooled AUROC
    on ALL of every target (including source itself -- an in-sample/upper-bound diagonal cell,
    by design, matching HARP's own published convention where the diagonal is each row's best
    entry)."""
    matrix_rf, matrix_lr = {}, {}
    for src_i, src in enumerate(DATASETS):
        t_src0 = time.time()
        print(f"  [Part C] source {src_i+1}/{len(DATASETS)}: fitting {condition_name} on {src} "
              f"(full dataset, no held-out split)  [job elapsed {fmt_elapsed(since_start())}] ...",
              flush=True)
        transform = build_transfer_transform(condition_name, all_feats[src], seed)
        core_src_all = transform(all_feats[src])
        y_src = all_y[src]
        row_rf, row_lr = {}, {}
        # fit classifiers once on the full source core, per spec ("scaler + Tucker + readout ...
        # on dataset A's entire data")
        for tgt_i, tgt in enumerate(DATASETS):
            t_tgt0 = time.time()
            core_tgt = transform(all_feats[tgt]) if tgt != src else core_src_all
            y_tgt = all_y[tgt]
            rf_scores = fit_eval("RF", core_src_all, y_src, core_tgt, seed)
            lr_scores = fit_eval("LR", core_src_all, y_src, core_tgt, seed)
            row_rf[tgt] = float(roc_auc_score(y_tgt, rf_scores))
            row_lr[tgt] = float(roc_auc_score(y_tgt, lr_scores))
            print(f"    [{condition_name}] {src} -> {tgt} ({tgt_i+1}/{len(DATASETS)}): "
                  f"RF={row_rf[tgt]:.4f} LR={row_lr[tgt]:.4f}  ({time.time()-t_tgt0:.1f}s)", flush=True)
        matrix_rf[src] = row_rf
        matrix_lr[src] = row_lr
        print(f"  [{condition_name}] {src} row complete ({time.time()-t_src0:.0f}s)  -> "
              + "  ".join(f"{t}={row_rf[t]:.3f}" for t in DATASETS)
              + f"  [job elapsed {fmt_elapsed(since_start())}]", flush=True)
    return {"RF": matrix_rf, "LR": matrix_lr}


def print_transfer_matrix(name, matrix_rf):
    print(f"\n{name} (RF pooled AUROC, rows=source/fit, cols=target/score):")
    header = f"{'':12s}" + "".join(f"{t:>12s}" for t in DATASETS)
    print(header)
    for src in DATASETS:
        print(f"{src:12s}" + "".join(f"{matrix_rf[src][t]*100:>12.1f}" for t in DATASETS))
    print(f"{'HARP (pub.)':12s}" + "".join(f"{'':>12s}" for _ in DATASETS))
    for src in DATASETS:
        row = HARP_TRANSFER_PUBLISHED[src]
        print(f"  {src:10s}" + "".join(f"{row[t]:>12.1f}" for t in DATASETS))


# ==============================================================================
# LOADING
# ==============================================================================

def _needed_raw_types(condition):
    """None (default) -> everything, for run_part_a()'s sequential all-conditions path, which
    genuinely needs all four. A specific condition name -> only what that condition's spec
    touches -- the memory-saving path for a --condition parallel job. For TriviaQA, loading
    everything is ~35GB+ per process; a lone core_max job only needs ~7GB of that."""
    if condition is None:
        return {"core", "static", "velocity", "joint"}
    return CONDITION_RAW_NEEDS[condition]


def load_new_dataset(dataset, data_dir, model_folder, condition=None):
    out_dir = os.path.join(data_dir, model_folder)
    npz_path = os.path.join(out_dir, f"{dataset}_phase2_features.npz")
    d = np.load(npz_path)   # lazy NpzFile -- does not materialize an array until indexed by key
    need = _needed_raw_types(condition)
    # static/velocity are needed as scratch to build "joint" even when the condition itself
    # (e.g. joint_tensor) never touches them directly -- built as local variables, only kept in
    # the RETURNED feats dict if individually in `need`, so a joint_tensor-only job's resident
    # memory during the actual (many-hour) Tucker/classifier fitting phase is just "joint", not
    # static+velocity+joint all at once (~55GB vs ~111GB at TriviaQA scale).
    need_static_raw = "static" in need or "joint" in need
    need_velocity_raw = "velocity" in need or "joint" in need

    feats = {}
    if "core" in need:
        feats["core"] = d["static_max"].astype(np.float32)
    static_tensor = velocity_tensor = None
    if need_static_raw:
        static_tensor = np.concatenate([d["static_q95"].astype(np.float32),
                                         d["static_q05"].astype(np.float32)], axis=2)
        if "static" in need:
            feats["static"] = static_tensor
    if need_velocity_raw:
        velocity_tensor = np.concatenate([d["velocity_q95"].astype(np.float32),
                                           d["velocity_q05"].astype(np.float32)], axis=2)
        if "velocity" in need:
            feats["velocity"] = velocity_tensor
    if "joint" in need:
        feats["joint"] = np.concatenate([static_tensor, velocity_tensor], axis=1)

    y = d["label"].astype(np.int64)
    prompt_idx = d["prompt_id"].astype(np.int64)
    is_known, _ = derive_is_known(y, prompt_idx)
    return feats, y, prompt_idx, is_known


def load_truthfulqa(core_pooled_pt, velocity_meta, condition=None):
    import torch
    need = _needed_raw_types(condition)
    need_static_raw = "static" in need or "joint" in need
    need_velocity_raw = "velocity" in need or "joint" in need

    pooled = torch.load(core_pooled_pt, weights_only=False)
    y = np.array([int(f) for f in pooled["all_hallucination_flag"]], dtype=np.int64)
    prompt_idx = np.array(pooled["prompt_indices"], dtype=np.int64)

    feats = {}
    if "core" in need:
        feats["core"] = torch.stack(pooled["all_emb"]).float().numpy()
    static_tensor = velocity_tensor = None
    if need_static_raw or need_velocity_raw:
        vel_npz_path = os.path.splitext(velocity_meta)[0].replace("_meta", "") + ".npz"
        vel_data = np.load(vel_npz_path)   # lazy
        if need_static_raw:
            static_tensor = np.concatenate([vel_data["S95"].astype(np.float32),
                                             vel_data["S05"].astype(np.float32)], axis=2)
            if "static" in need:
                feats["static"] = static_tensor
        if need_velocity_raw:
            velocity_tensor = np.concatenate([vel_data["V95"].astype(np.float32),
                                               vel_data["V05"].astype(np.float32)], axis=2)
            if "velocity" in need:
                feats["velocity"] = velocity_tensor
    if "joint" in need:
        feats["joint"] = np.concatenate([static_tensor, velocity_tensor], axis=1)

    is_known, _ = derive_is_known(y, prompt_idx)
    return feats, y, prompt_idx, is_known


# ==============================================================================
# PART D -- MASTER TABLE, VERDICT, LEADERBOARD
# ==============================================================================

def print_master_table(part_a_results, best_condition, means):
    print(f"\n{'='*100}\nPART A MASTER TABLE (RF, grouped pooled / within-prompt AUROC)\n{'='*100}")
    header = f"{'Dataset':12s} {'Condition':14s} {'Pooled':>10s} {'Within-p':>10s}"
    print(header)
    for ds in DATASETS:
        r = part_a_results.get(ds)
        if r is None:
            print(f"{ds:12s}  (missing)"); continue
        for name in ["core_max", "q_velocity"] + NEW_CONDITIONS:
            g = r["grouped"][name]["RF"]
            print(f"{ds:12s} {name:14s} {g['pooled_oof_auroc']:>10.4f} "
                  f"{g['within_prompt']['within_prompt_auroc']:>10.4f}")
    print(f"\nMean within-prompt RF AUROC across datasets, per new condition: "
          + ", ".join(f"{k}={v:.4f}" for k, v in means.items()))
    print(f"Best Part A condition (by mean within-prompt AUROC): {best_condition}")


def compute_verdict(part_a_results, best_condition):
    """[ADOPT-COMBINED]: best condition beats max(core-max, q-velocity) with CI excluding zero
    (positive) on at least one dataset/metric, and never regresses significantly anywhere.
    [FUSION-DILUTES]: best condition is significantly WORSE than the better single feature on
    at least one dataset/metric.
    [DATASET-CONDITIONAL]: neither of the above -- combined ~= max(singles) everywhere."""
    any_significant_win = False
    any_significant_loss = False
    detail = {}
    for ds, r in part_a_results.items():
        pd = r["paired_deltas"][best_condition]
        checks = {
            "vs_core_max_pooled": pd["vs_core_max"]["pooled"],
            "vs_core_max_within_prompt": pd["vs_core_max"]["within_prompt"],
            "vs_q_velocity_pooled": pd["vs_q_velocity"]["pooled"],
            "vs_q_velocity_within_prompt": pd["vs_q_velocity"]["within_prompt"],
        }
        detail[ds] = checks
        for name, c in checks.items():
            if c["excludes_zero"] and c["mean_delta"] > 0:
                any_significant_win = True
            if c["excludes_zero"] and c["mean_delta"] < 0:
                any_significant_loss = True

    if any_significant_loss:
        verdict = "FUSION-DILUTES"
    elif any_significant_win:
        verdict = "ADOPT-COMBINED"
    else:
        verdict = "DATASET-CONDITIONAL"
    return verdict, detail


def update_leaderboard_v2(part_a_results, part_c_results, best_condition, leaderboard_path):
    entries = []
    for ds, r in part_a_results.items():
        for name in ["core_max", "q_velocity"] + NEW_CONDITIONS:
            g = r["grouped"][name]["RF"]
            h = r["harp"][name]
            entries.append({"dataset": ds, "condition": name, "protocol": "GroupKFold(5)",
                             "rank": CONDITION_SPECS[name], "pooled_auroc": g["pooled_oof_auroc"],
                             "pooled_ci95": g["ci95"],
                             "within_prompt_auroc": g["within_prompt"]["within_prompt_auroc"],
                             "within_prompt_ci95": g["within_prompt"]["ci95"]})
            entries.append({"dataset": ds, "condition": name, "protocol": "HARP 5-seed",
                             "rank": CONDITION_SPECS[name], "harp_rf_mean": h["RF_mean"],
                             "harp_rf_std": h["RF_std"], "harp_lr_mean": h["LR_mean"],
                             "harp_lr_std": h["LR_std"]})
    if part_c_results:
        for cond_name, mats in part_c_results.items():
            for src in DATASETS:
                for tgt in DATASETS:
                    entries.append({"dataset": f"{src}->{tgt}", "condition": cond_name,
                                     "protocol": "transfer (fit-all-source, score-all-target)",
                                     "pooled_auroc_rf": mats["RF"][src][tgt],
                                     "pooled_auroc_lr": mats["LR"][src][tgt]})
    leaderboard = {"best_condition": best_condition, "entries": entries}
    os.makedirs(os.path.dirname(leaderboard_path) or ".", exist_ok=True)
    with open(leaderboard_path, "w") as f:
        json.dump(leaderboard, f, indent=2)
    return leaderboard_path


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: combination builders, generic evaluators, transfer fit/apply, verdicts")
    print("=" * 70)

    # ---- split protocols: pure, no features, no model ------------------------------------
    n_q, n_b = 200, 10
    st_prompt_idx = np.repeat(np.arange(n_q), n_b)
    st_N = n_q * n_b
    st_is_known = np.zeros(n_q, dtype=bool)
    st_is_known[:140] = True                       # 140 known / 60 unknown prompts

    t_q, v_q = original_harp_split(st_is_known, st_prompt_idx, st_N, seed=42)
    tq, vq = set(st_prompt_idx[t_q].tolist()), set(st_prompt_idx[v_q].tolist())
    assert not (tq & vq), "question-level split must leave no prompt on both sides"
    assert len(tq) == 105, f"75% of 140 known prompts is 105, got {len(tq)}"
    assert len(t_q) + len(v_q) == st_N and len(set(t_q) | set(v_q)) == st_N
    print("  [PASS] original_harp_split: prompt-disjoint, 105/35 known prompts, all rows covered")

    t_a, v_a = answer_level_harp_split(st_is_known, st_prompt_idx, st_N, seed=42)
    ta, va = set(st_prompt_idx[t_a].tolist()), set(st_prompt_idx[v_a].tolist())
    both = len(ta & va)
    assert len(t_a) + len(v_a) == st_N and len(set(t_a) | set(v_a)) == st_N, \
        "answer-level split must still partition every row exactly once"
    assert len(t_a) == int(140 * n_b * 0.75), \
        f"train should be 75% of the 1400 known ANSWERS, got {len(t_a)}"
    assert not (ta & set(np.where(~st_is_known)[0].tolist())), \
        "no unknown prompt may appear in training (main.py:264)"
    assert both > 0.8 * len(ta), \
        f"answer-level split must leak: only {both}/{len(ta)} prompts on both sides"
    print(f"  [PASS] answer_level_harp_split: partitions all rows, unknown prompts held out, "
          f"{both}/140 known prompts on both sides (expect ~132 = 94%)")

    # both splits must be pure functions of (is_known, prompt_idx, seed) and leave the global
    # RNG untouched -- otherwise re-running a seed gives different folds
    np.random.seed(7)
    before = np.random.rand()
    np.random.seed(7)
    answer_level_harp_split(st_is_known, st_prompt_idx, st_N, seed=42)
    original_harp_split(st_is_known, st_prompt_idx, st_N, seed=42)
    assert np.random.rand() == before, "splits must not disturb the global numpy RNG"
    t_a2, _ = answer_level_harp_split(st_is_known, st_prompt_idx, st_N, seed=42)
    assert np.array_equal(t_a, t_a2), "same seed must give the same answer-level split"
    t_a3, _ = answer_level_harp_split(st_is_known, st_prompt_idx, st_N, seed=43)
    assert not np.array_equal(t_a, t_a3), "different seeds must give different splits"
    print("  [PASS] both splits are deterministic per seed and leave the global RNG untouched")

    # D must comfortably exceed r_d=64 even BEFORE concatenation (core_max/q_static use the raw
    # D alone, not the concatenated 2D) -- D=48 originally used here was too small and silently
    # under-filled r_d, caught by the shape assertion below rather than a crash.
    D = 80
    data = s01.generate_synthetic_data(n_prompts=120, beams_per_prompt=10, L=9, D=D, seed=0)
    core_raw, y, prompt_idx, is_known = data["X"], data["y"], data["prompt_idx"], data["is_known"]
    n_beams = data["n_beams"]
    rng = np.random.default_rng(1)
    static_q95 = core_raw.copy() + rng.normal(0, 0.05, size=core_raw.shape).astype(np.float32)
    static_q05 = core_raw.copy() - rng.normal(0, 0.05, size=core_raw.shape).astype(np.float32)
    velocity_q95 = rng.normal(0, 0.3, size=(n_beams, 8, D)).astype(np.float32)
    for l in range(8):
        velocity_q95[:, l, 5:10] += core_raw[:, 0, 5:10]
    velocity_q05 = rng.normal(0, 0.3, size=(n_beams, 8, D)).astype(np.float32)

    feats = build_feats(core_raw, static_q95, static_q05, velocity_q95, velocity_q05)
    assert feats["static"].shape == (n_beams, 9, 2 * D)
    assert feats["velocity"].shape == (n_beams, 8, 2 * D)
    assert feats["joint"].shape == (n_beams, 17, 2 * D)
    print("  [PASS] build_feats: static/velocity/joint shapes correct")

    # -- selective loading: a --condition job must load ONLY the raw tensors its own condition
    # needs, not all four -- the memory-saving path for running Part A's 6 conditions as
    # separate parallel jobs. Verified against a real (fabricated) npz file, not just the pure
    # CONDITION_RAW_NEEDS mapping, so this actually exercises load_new_dataset()'s branching.
    tmp_dir_load = os.path.join(HERE, "results", "_selftest_phase3")
    os.makedirs(tmp_dir_load, exist_ok=True)
    fake_npz_path = os.path.join(tmp_dir_load, "selftest_phase2_features.npz")
    np.savez_compressed(fake_npz_path, static_max=core_raw.astype(np.float16),
                         static_q95=static_q95.astype(np.float16), static_q05=static_q05.astype(np.float16),
                         velocity_q95=velocity_q95.astype(np.float16), velocity_q05=velocity_q05.astype(np.float16),
                         kinematic=np.zeros((n_beams, 30), dtype=np.float32),
                         prompt_id=prompt_idx, beam_idx=np.arange(n_beams), label=y)
    feats_core_only, y_l, prompt_idx_l, is_known_l = load_new_dataset(
        "selftest", os.path.dirname(tmp_dir_load), os.path.basename(tmp_dir_load), condition="core_max")
    assert set(feats_core_only.keys()) == {"core"}, \
        f"condition='core_max' should load ONLY 'core', got {set(feats_core_only.keys())}"
    feats_joint_only, _, _, _ = load_new_dataset(
        "selftest", os.path.dirname(tmp_dir_load), os.path.basename(tmp_dir_load), condition="joint_tensor")
    assert set(feats_joint_only.keys()) == {"joint"}, \
        (f"condition='joint_tensor' should load ONLY 'joint' -- static/velocity are scratch used "
         f"to build it but shouldn't be retained in the returned dict, got "
         f"{set(feats_joint_only.keys())}")
    feats_all, _, _, _ = load_new_dataset(
        "selftest", os.path.dirname(tmp_dir_load), os.path.basename(tmp_dir_load), condition=None)
    assert set(feats_all.keys()) == {"core", "static", "velocity", "joint"}
    assert np.array_equal(feats_all["core"], feats_core_only["core"]), \
        "selective and full loading must produce identical values for the tensors they share"
    assert np.array_equal(feats_all["joint"], feats_joint_only["joint"]), \
        "joint tensor must be identical whether static/velocity are separately retained or not"
    print("  [PASS] load_new_dataset(condition=...): retains exactly the raw tensors each "
          "condition needs in the returned dict (core_max->{core}, joint_tensor->{joint} only, "
          "static/velocity dropped once used as scratch; None->everything), with identical "
          "values to full loading for shared tensors")

    # -- regression test for the real bug that crashed a real q_velocity job: compute_folds()
    # used to hardcode feats["core"], which KeyErrors for any condition whose selective load
    # doesn't include "core" (q_velocity, q_static, joint_tensor). This runs the EXACT
    # production path -- load_new_dataset(condition=X) followed by compute_folds() on its
    # result -- for every one of the 6 conditions, not just the ones that happen to include
    # "core" (which is what the previous test coverage effectively did, and why this slipped
    # through).
    for cond_name in ALL_CONDITIONS:
        feats_selective, y_sel, prompt_idx_sel, is_known_sel = load_new_dataset(
            "selftest", os.path.dirname(tmp_dir_load), os.path.basename(tmp_dir_load), condition=cond_name)
        folds_selective = compute_folds(feats_selective, y_sel, prompt_idx_sel)
        assert len(folds_selective) == N_SPLITS
        # full run_condition_only() end to end on the selectively-loaded feats -- the actual
        # code path a real --condition slurm job executes, not just the folds computation alone
        one_result = run_condition_only("selftest", cond_name, feats_selective, y_sel, prompt_idx_sel,
                                         is_known_sel, folds_selective)
        assert 0.0 <= one_result["grouped_summary"]["RF"]["pooled_oof_auroc"] <= 1.0
    print(f"  [PASS] compute_folds() + run_condition_only() work end to end on SELECTIVELY-loaded "
          f"feats for all {len(ALL_CONDITIONS)} conditions, not just the ones that happen to "
          f"include 'core' -- regression test for the real q_velocity KeyError")

    for name, spec in CONDITION_SPECS.items():
        builder = make_grouped_builder(spec, feats)
        tr = np.arange(n_beams)
        core = builder(tr, 0)
        expected_dim = sum(r_l * r_d for (_, r_l, r_d) in spec)
        assert core.shape == (n_beams, expected_dim), f"{name}: got {core.shape}, expected dim {expected_dim}"
    print("  [PASS] make_grouped_builder: all 6 condition dims correct "
          "(core_max=320, q_static=320, q_velocity=256, core_concat=576, joint_tensor=512, triple_concat=896)")

    folds = list(GroupKFold(n_splits=N_SPLITS).split(core_raw, y, groups=prompt_idx))
    core_max_builder = make_grouped_builder(CONDITION_SPECS["core_max"], feats)
    summary, oof = run_grouped_generic(core_max_builder, y, prompt_idx, folds, SEED, "core_max")
    assert summary["RF"]["pooled_oof_auroc"] > 0.6
    print(f"  [PASS] run_grouped_generic: core_max RF pooled={summary['RF']['pooled_oof_auroc']:.4f}")

    triple_builder = make_grouped_builder(CONDITION_SPECS["triple_concat"], feats)
    summary_triple, oof_triple = run_grouped_generic(triple_builder, y, prompt_idx, folds, SEED, "triple_concat")
    assert summary_triple["RF"]["pooled_oof_auroc"] > 0.6
    print(f"  [PASS] run_grouped_generic on a multi-tensor concat condition: "
          f"triple_concat RF pooled={summary_triple['RF']['pooled_oof_auroc']:.4f}")

    delta = paired_bootstrap_delta(oof_triple["RF"], oof["RF"], y, prompt_idx, n_boot=200, seed=0)
    assert "mean_delta" in delta and "excludes_zero" in delta
    print(f"  [PASS] paired_bootstrap_delta on generic-builder OOF arrays: delta={delta['mean_delta']:.4f}")

    harp = run_harp_generic({"core_max": core_max_builder, "triple_concat": triple_builder},
                             y, prompt_idx, is_known, seeds=[42, 0], label_prefix="selftest")
    assert harp["core_max"]["per_seed"][0]["n_train"] == harp["triple_concat"]["per_seed"][0]["n_train"], \
        "paired conditions must see identical splits within a seed"
    print("  [PASS] run_harp_generic: paired conditions share identical n_train/n_valid per seed")

    fake_part_a = {"selftest": {"grouped": {n: {"RF": {"within_prompt": {"within_prompt_auroc": 0.7 + 0.01 * i}}}
                                              for i, n in enumerate(NEW_CONDITIONS)}}}
    best, means = determine_best_condition(fake_part_a)
    assert best == NEW_CONDITIONS[-1], f"expected the highest-scoring condition ({NEW_CONDITIONS[-1]}), got {best}"
    print(f"  [PASS] determine_best_condition: picks the condition with highest mean within-prompt AUROC ({best})")

    # -- transfer fit/apply split: fit on dataset A, apply unchanged basis to dataset B --
    core_raw_b = core_raw + rng.normal(0, 0.01, size=core_raw.shape).astype(np.float32)
    transform = fit_transfer_pipeline(core_raw, 5, 10, seed=0)
    core_a_via_transform = transform(core_raw)
    core_b_via_transform = transform(core_raw_b)
    assert core_a_via_transform.shape == (n_beams, 50)
    assert core_b_via_transform.shape == (n_beams, 50)
    assert not np.allclose(core_a_via_transform, core_b_via_transform), \
        "different input data should give different projected cores"
    print("  [PASS] fit_transfer_pipeline: fit-on-A, apply-to-B produces correctly-shaped, "
          "genuinely different projections")

    feats_b = build_feats(core_raw_b, static_q95, static_q05, velocity_q95, velocity_q05)
    transfer_transform = build_transfer_transform("triple_concat", feats, seed=0)
    core_self = transfer_transform(feats)
    core_cross = transfer_transform(feats_b)
    assert core_self.shape == (n_beams, 896) and core_cross.shape == (n_beams, 896)
    print("  [PASS] build_transfer_transform on a multi-tensor condition (triple_concat): "
          f"shapes correct ({core_self.shape})")

    # -- verdict logic: fabricate a clear FUSION-DILUTES case and a clear ADOPT-COMBINED case --
    def fake_delta(mean_delta, excludes_zero):
        return {"mean_delta": mean_delta, "ci95": (mean_delta - 0.01, mean_delta + 0.01), "excludes_zero": excludes_zero}

    dilutes_case = {"ds1": {"paired_deltas": {"joint_tensor": {
        "vs_core_max": {"pooled": fake_delta(-0.05, True), "within_prompt": fake_delta(-0.05, True)},
        "vs_q_velocity": {"pooled": fake_delta(-0.02, True), "within_prompt": fake_delta(-0.02, True)}}}}}
    verdict, _ = compute_verdict(dilutes_case, "joint_tensor")
    assert verdict == "FUSION-DILUTES"

    adopt_case = {"ds1": {"paired_deltas": {"joint_tensor": {
        "vs_core_max": {"pooled": fake_delta(0.03, True), "within_prompt": fake_delta(0.02, True)},
        "vs_q_velocity": {"pooled": fake_delta(0.01, False), "within_prompt": fake_delta(0.01, False)}}}}}
    verdict2, _ = compute_verdict(adopt_case, "joint_tensor")
    assert verdict2 == "ADOPT-COMBINED"

    wash_case = {"ds1": {"paired_deltas": {"joint_tensor": {
        "vs_core_max": {"pooled": fake_delta(0.001, False), "within_prompt": fake_delta(-0.001, False)},
        "vs_q_velocity": {"pooled": fake_delta(0.001, False), "within_prompt": fake_delta(-0.001, False)}}}}}
    verdict3, _ = compute_verdict(wash_case, "joint_tensor")
    assert verdict3 == "DATASET-CONDITIONAL"
    print(f"  [PASS] compute_verdict: correctly classifies FUSION-DILUTES / ADOPT-COMBINED / "
          f"DATASET-CONDITIONAL from fabricated delta patterns")

    # -- rank-probe adoption rule -- needs an entry for every r_d in RANK_PROBE_RDS since
    # rank_probe_verdict() iterates all of them, not just the one under direct test
    fake_part_b = {
        "triviaqa": {"results": {"core_max": {
            "r_d_96": {"delta_vs_r_d_64": {"pooled": fake_delta(0.02, True)}},
            "r_d_128": {"delta_vs_r_d_64": {"pooled": fake_delta(0.03, True)}}}}},
        "truthfulqa": {"results": {"core_max": {
            "r_d_96": {"delta_vs_r_d_64": {"pooled": fake_delta(-0.03, True)}},
            "r_d_128": {"delta_vs_r_d_64": {"pooled": fake_delta(0.001, False)}}}}},
    }
    verdicts = rank_probe_verdict(fake_part_b, "core_max")
    assert verdicts["core_max_r_d_96"]["adopt"] is False, \
        "TruthfulQA regression should block adoption even though TriviaQA improved"
    assert verdicts["core_max_r_d_128"]["adopt"] is True, \
        "TriviaQA improves and TruthfulQA shows no significant regression -- should adopt"
    print("  [PASS] rank_probe_verdict: correctly blocks adoption when TruthfulQA regresses "
          "significantly (r_d=96), and correctly adopts when it doesn't (r_d=128)")

    # -- the real correctness claim behind running Part A's 6 conditions as separate parallel
    # jobs: run_part_a() (sequential) and run_condition_only()+combine_conditions() (split,
    # simulating 6 independent jobs whose results get merged after the fact) must produce
    # BYTE-IDENTICAL numbers on the same data -- not just similar.
    sequential = run_part_a("splitcheck", feats, y, prompt_idx, is_known)
    folds_split = compute_folds(feats, y, prompt_idx)
    per_condition_split = {name: run_condition_only("splitcheck", name, feats, y, prompt_idx, is_known, folds_split)
                            for name in ["core_max", "q_velocity"] + NEW_CONDITIONS}
    combined = combine_conditions("splitcheck", per_condition_split, y, prompt_idx)
    for name in ["core_max", "q_velocity"] + NEW_CONDITIONS:
        assert (sequential["grouped"][name]["RF"]["pooled_oof_auroc"]
                == combined["grouped"][name]["RF"]["pooled_oof_auroc"]), \
            f"{name}: sequential and split-then-combined grouped AUROC diverged"
        assert (sequential["harp"][name]["RF_mean"] == combined["harp"][name]["RF_mean"]), \
            f"{name}: sequential and split-then-combined HARP AUROC diverged"
    for name in NEW_CONDITIONS:
        assert (sequential["paired_deltas"][name]["vs_core_max"]["pooled"]["mean_delta"]
                == combined["paired_deltas"][name]["vs_core_max"]["pooled"]["mean_delta"]), \
            f"{name}: paired delta diverged between sequential and split-then-combined runs"
    print("  [PASS] run_part_a (sequential) and run_condition_only+combine_conditions (split, "
          "as if run as 6 separate parallel jobs) produce byte-identical grouped AUROC, HARP "
          "AUROC, and paired deltas for all 6 conditions")

    # -- write_condition_result/load_condition_result: JSON must NOT embed OOF arrays (that's
    # the whole point of this split -- a single condition's OOF vectors bloated one real
    # TriviaQA result file to ~3.3MB/199k lines) -- and backfill_paired_deltas must produce
    # deltas numerically IDENTICAL to combine_conditions()'s all-at-once computation.
    backfill_dir = os.path.join(HERE, "results", "_selftest_phase3_backfill")
    shutil.rmtree(backfill_dir, ignore_errors=True)
    os.makedirs(backfill_dir, exist_ok=True)
    for name in ["core_max", "q_velocity"] + NEW_CONDITIONS:
        out_path, oof_path = write_condition_result("splitcheck", name, per_condition_split[name], backfill_dir)
        with open(out_path) as f:
            written_json = json.load(f)
        assert "oof_rf" not in written_json and "oof_lr" not in written_json, \
            f"{name}: result JSON must not embed OOF score arrays -- that's what bloated a real file"
        assert written_json["oof_npz"] == os.path.basename(oof_path)
        reloaded, _ = load_condition_result("splitcheck", name, backfill_dir)
        assert np.array_equal(reloaded["oof_rf"], np.asarray(per_condition_split[name]["oof_rf"])), \
            f"{name}: OOF scores must round-trip exactly through the .npz split"
    print("  [PASS] write_condition_result/load_condition_result: JSON holds summaries only, OOF "
          "score vectors round-trip exactly through the companion .npz")

    # -- migration: a real pre-fix result file (old format, OOF arrays embedded, no "oof_npz"
    # key) must load transparently and get split in place, no recompute, matching what
    # write_condition_result would have produced from the same underlying result.
    migrate_dir = os.path.join(HERE, "results", "_selftest_phase3_migrate")
    shutil.rmtree(migrate_dir, ignore_errors=True)
    os.makedirs(migrate_dir, exist_ok=True)
    old_format_result = {"dataset": "splitcheck", "condition": "core_max",
                          "grouped_summary": per_condition_split["core_max"]["grouped_summary"],
                          "oof_rf": per_condition_split["core_max"]["oof_rf"],
                          "oof_lr": per_condition_split["core_max"]["oof_lr"],
                          "harp": per_condition_split["core_max"]["harp"]}
    old_path = os.path.join(migrate_dir, "session06_phase3_partA_splitcheck_core_max.json")
    with open(old_path, "w") as f:
        json.dump(old_format_result, f, indent=2, default=str)
    migrated, _ = load_condition_result("splitcheck", "core_max", migrate_dir)
    assert np.array_equal(migrated["oof_rf"], np.asarray(old_format_result["oof_rf"])), \
        "migrated OOF scores must match the pre-fix embedded values exactly"
    with open(old_path) as f:
        rewritten = json.load(f)
    assert "oof_rf" not in rewritten and "oof_npz" in rewritten, \
        "loading an old-format file must rewrite it in place to the new summary-only format"
    print("  [PASS] load_condition_result: transparently migrates old-format (pre-fix) files "
          "with embedded OOF arrays -- no recompute needed")

    written = backfill_paired_deltas("splitcheck", backfill_dir, y, prompt_idx)
    for name in NEW_CONDITIONS:
        assert (written[name]["vs_core_max"]["pooled"]["mean_delta"]
                == combined["paired_deltas"][name]["vs_core_max"]["pooled"]["mean_delta"]), \
            f"{name}: backfilled delta must match combine_conditions()'s own computation exactly"
        with open(os.path.join(backfill_dir, f"session06_phase3_partA_splitcheck_{name}.json")) as f:
            persisted = json.load(f)
        assert "paired_deltas" in persisted, f"{name}: backfill must persist paired_deltas into its own JSON"
    print("  [PASS] backfill_paired_deltas: matches combine_conditions()'s deltas exactly, "
          "persisted into each condition's own JSON")

    # -- the actual scenario that motivated this: core_concat finishes, its baselines haven't
    # yet (this is exactly what happened to real TriviaQA data). backfill must no-op cleanly
    # (not crash) until both baselines exist, then pick up already-finished conditions.
    partial_dir = os.path.join(HERE, "results", "_selftest_phase3_backfill_partial")
    shutil.rmtree(partial_dir, ignore_errors=True)
    os.makedirs(partial_dir, exist_ok=True)
    write_condition_result("splitcheck", "core_concat", per_condition_split["core_concat"], partial_dir)
    empty = backfill_paired_deltas("splitcheck", partial_dir, y, prompt_idx)
    assert empty == {}, "backfill must no-op (not crash) when neither baseline is on disk yet"
    write_condition_result("splitcheck", "core_max", per_condition_split["core_max"], partial_dir)
    still_empty = backfill_paired_deltas("splitcheck", partial_dir, y, prompt_idx)
    assert still_empty == {}, "backfill must still no-op with only ONE baseline (core_max) present"
    write_condition_result("splitcheck", "q_velocity", per_condition_split["q_velocity"], partial_dir)
    now_written = backfill_paired_deltas("splitcheck", partial_dir, y, prompt_idx)
    assert set(now_written.keys()) == {"core_concat"}, \
        "once both baselines land, backfill should pick up exactly the conditions already on disk"
    print("  [PASS] backfill_paired_deltas: no-ops until both baselines exist, then correctly "
          "picks up whichever conditions already finished (the real core_concat-finishes-first case)")

    tmp_dir = os.path.join(HERE, "results", "_selftest_phase3")
    os.makedirs(tmp_dir, exist_ok=True)
    lb_path = os.path.join(tmp_dir, "leaderboard_v2_selftest.json")
    fake_full_part_a = {"selftest": {"grouped": {n: {"RF": summary["RF"], "LR": summary["LR"]}
                                                  for n in ["core_max", "q_velocity"] + NEW_CONDITIONS},
                                      "harp": {n: harp.get(n, harp["core_max"]) for n in
                                               ["core_max", "q_velocity"] + NEW_CONDITIONS}}}
    update_leaderboard_v2(fake_full_part_a, None, "triple_concat", lb_path)
    with open(lb_path) as f:
        lb = json.load(f)
    assert lb["best_condition"] == "triple_concat"
    assert len(lb["entries"]) == 12   # 6 conditions x 2 protocols
    print(f"  [PASS] update_leaderboard_v2: wrote {len(lb['entries'])} entries, best_condition tagged")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=DATASETS, default=None)
    parser.add_argument("--model_folder", type=str, default="llama-3.1-8b-instruct")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--core-pooled-pt", type=str, default=None, help="truthfulqa only")
    parser.add_argument("--velocity-meta", type=str, default=None, help="truthfulqa only")
    parser.add_argument("--part", type=str, choices=["a", "b"], default=None)
    parser.add_argument("--condition", type=str, choices=ALL_CONDITIONS, default=None,
                         help="With --part a: run ONLY this one condition instead of all 6 -- "
                              "lets Part A's 6 conditions run as separate parallel jobs (they "
                              "are mathematically independent). Follow up with "
                              "--combine-conditions once all 6 per-condition jobs finish.")
    parser.add_argument("--combine-conditions", action="store_true",
                         help="Merge a dataset's 6 separately-run --condition jobs into the "
                              "same session06_phase3_partA_{dataset}.json a sequential --part a "
                              "run would have produced.")
    parser.add_argument("--backfill-deltas", action="store_true",
                         help="Interim, re-runnable: as soon as core_max and q_velocity are both "
                              "on disk, backfill vs_core_max/vs_q_velocity paired_deltas into "
                              "whichever other --condition results already exist, without waiting "
                              "for all 6. Safe to call again each time a new condition finishes.")
    parser.add_argument("--combine-part-a", action="store_true")
    parser.add_argument("--part-c", action="store_true")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None,
                         help="Defaults to results/{model_folder} -- see resolve_results_dir(). "
                              "Pass explicitly only to override that per-model isolation.")
    parser.add_argument("--leaderboard", type=str, default=None,
                         help="Defaults to {results-dir}/leaderboard_v2.json.")
    parser.add_argument("--split-unit", type=str, choices=["question", "answer"],
                         default="question",
                         help="Unit of the 75/25 split inside the known group. 'question' is the "
                              "HARP paper's protocol and the default -- existing results were all "
                              "produced under it. 'answer' reproduces HARP's RELEASED code "
                              "(main.py:259). Use a separate --results-dir with 'answer': the "
                              "output filenames key on dataset+condition only, so the two "
                              "protocols would otherwise overwrite each other.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    global SPLIT_UNIT
    SPLIT_UNIT = args.split_unit
    if SPLIT_UNIT != "question":
        print(f"*** SPLIT_UNIT = {SPLIT_UNIT} -- reproducing HARP's released split, NOT the "
              f"paper's. Results are not comparable with question-level runs.", flush=True)
        if not args.results_dir:
            print("*** WARNING: --results-dir not set. Answer-level results will be written into "
                  "the same per-model directory as the question-level ones and will overwrite "
                  "them. Pass --results-dir explicitly.", flush=True)

    if args.self_test:
        self_test()
        return

    # Per-model output isolation. Every Part A/B/C output filename is keyed on dataset+condition
    # only -- NOT model -- so a Qwen run would have silently overwritten LLaMA's
    # session06_phase3_partA_triviaqa.json, i.e. ~30h of parallel compute plus a 1.5h combine.
    # Resolving the default here (rather than editing all ~13 os.path.join(args.results_dir, ...)
    # sites) makes every write model-scoped in one place. Reads fall back to the legacy flat
    # results/ location via legacy_read_path(), so LLaMA's already-completed files keep working
    # with no migration.
    args.results_dir = resolve_results_dir(args.results_dir, args.model_folder)
    if args.leaderboard is None:
        args.leaderboard = os.path.join(args.results_dir, "leaderboard_v2.json")
    # every branch below writes something here, and the model-scoped dir won't exist on a first run
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"[results-dir] {args.results_dir}", flush=True)

    def get_data_dir():
        if args.data_dir:
            return args.data_dir
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        return cfg["output"]["data_dir"]

    def use_legacy_truthfulqa(ds):
        """TruthfulQA was reachable ONLY via --core-pooled-pt/--velocity-meta, since LLaMA's
        TruthfulQA features predate this pipeline (older Route N artifacts). Now that 39/42
        support truthfulqa for any model, Qwen has a standard {dataset}_phase2_features.npz and
        no canonical .pt to point those flags at. Dispatch on which artifact EXISTS, not on the
        dataset name -- Qwen takes the standard path automatically, LLaMA keeps its legacy flags."""
        return ds == "truthfulqa" and not has_standard_features(ds, get_data_dir(), args.model_folder)

    def load_dataset(ds, condition=None):
        if use_legacy_truthfulqa(ds):
            if not args.core_pooled_pt or not args.velocity_meta:
                print(f"ERROR: truthfulqa for model_folder={args.model_folder} has no "
                      f"truthfulqa_phase2_features.npz, so --core-pooled-pt and --velocity-meta "
                      f"are required (LLaMA's legacy path). Run 42_extract_phase2.py --dataset "
                      f"truthfulqa --model_folder {args.model_folder} to use the standard path "
                      f"instead."); sys.exit(1)
            return load_truthfulqa(args.core_pooled_pt, args.velocity_meta, condition=condition)
        return load_new_dataset(ds, get_data_dir(), args.model_folder, condition=condition)

    if args.backfill_deltas:
        if not args.dataset:
            print("ERROR: --dataset required with --backfill-deltas."); sys.exit(1)
        _, y, prompt_idx, _ = load_dataset(args.dataset, condition="core_max")
        backfill_paired_deltas(args.dataset, args.results_dir, y, prompt_idx)
        return

    if args.combine_conditions:
        if not args.dataset:
            print("ERROR: --dataset required with --combine-conditions."); sys.exit(1)
        # only y/prompt_idx are needed here (for composition_line) -- "core_max" is the cheapest
        # single tensor to load, avoids pulling in static/velocity/joint for no reason.
        feats, y, prompt_idx, is_known = load_dataset(args.dataset, condition="core_max")
        per_condition = {}
        for cond_name in ALL_CONDITIONS:
            cond_result, p = load_condition_result(args.dataset, cond_name, args.results_dir)
            if cond_result is None:
                print(f"ERROR: missing {p} -- run `--dataset {args.dataset} --part a --condition "
                      f"{cond_name}` first."); sys.exit(1)
            per_condition[cond_name] = cond_result
        result = combine_conditions(args.dataset, per_condition, y, prompt_idx)
        out_path = os.path.join(args.results_dir, f"session06_phase3_partA_{args.dataset}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nWrote: {out_path} (merged from {len(ALL_CONDITIONS)} per-condition runs)")
        print("Next: after all 4 datasets, run --combine-part-a")
        return

    if args.combine_part_a:
        part_a_results = {}
        for ds in DATASETS:
            p = legacy_read_path(f"session06_phase3_partA_{ds}.json", args.results_dir)
            if os.path.exists(p):
                with open(p) as f:
                    part_a_results[ds] = json.load(f)
        if len(part_a_results) < len(DATASETS):
            missing = set(DATASETS) - set(part_a_results)
            print(f"ERROR: missing Part A results for: {missing}"); sys.exit(1)
        best_condition, means = determine_best_condition(part_a_results)
        print_master_table(part_a_results, best_condition, means)
        out_path = os.path.join(args.results_dir, "session06_phase3_best_condition.json")
        with open(out_path, "w") as f:
            json.dump({"best_condition": best_condition, "means": means}, f, indent=2)
        print(f"\nWrote: {out_path}")
        return

    if args.part_c:
        best_path = legacy_read_path("session06_phase3_best_condition.json", args.results_dir)
        with open(best_path) as f:
            best_condition = json.load(f)["best_condition"]
        print(f"Part C: transfer matrix for core_max and {best_condition} (best Part A condition)")
        all_feats, all_y = {}, {}
        for ds_i, ds in enumerate(DATASETS):
            t_load0 = time.time()
            print(f"Loading {ds_i+1}/{len(DATASETS)}: {ds} ...  [job elapsed {fmt_elapsed(since_start())}]",
                  flush=True)
            # same existence-based dispatch as load_dataset() -- Part C loads all 4 datasets
            # directly rather than going through it
            feats, y, _, _ = load_dataset(ds)
            all_feats[ds] = feats
            all_y[ds] = y
            print(f"  loaded {ds}: {len(y)} beams  ({time.time()-t_load0:.1f}s)", flush=True)
        part_c_results = {}
        for cond_i, cond_name in enumerate(("core_max", best_condition)):
            print(f"\n[Part C] Transfer matrix {cond_i+1}/2: {cond_name}  "
                  f"[job elapsed {fmt_elapsed(since_start())}]", flush=True)
            part_c_results[cond_name] = run_part_c(all_feats, all_y, cond_name, seed=SEED)
            print_transfer_matrix(cond_name, part_c_results[cond_name]["RF"])
        out_path = os.path.join(args.results_dir, "session06_phase3_partC.json")
        with open(out_path, "w") as f:
            json.dump(part_c_results, f, indent=2, default=str)
        print(f"\nWrote: {out_path}")
        return

    if args.combine:
        part_a_results = {}
        for ds in DATASETS:
            p = legacy_read_path(f"session06_phase3_partA_{ds}.json", args.results_dir)
            with open(p) as f:
                part_a_results[ds] = json.load(f)
        best_path = legacy_read_path("session06_phase3_best_condition.json", args.results_dir)
        with open(best_path) as f:
            best_info = json.load(f)
        best_condition = best_info["best_condition"]

        part_b_results = {}
        for ds in RANK_PROBE_DATASETS:
            p = legacy_read_path(f"session06_phase3_partB_{ds}.json", args.results_dir)
            if os.path.exists(p):
                with open(p) as f:
                    part_b_results[ds] = json.load(f)

        part_c_path = legacy_read_path("session06_phase3_partC.json", args.results_dir)
        part_c_results = None
        if os.path.exists(part_c_path):
            with open(part_c_path) as f:
                part_c_results = json.load(f)

        print_master_table(part_a_results, best_condition, best_info["means"])
        verdict, detail = compute_verdict(part_a_results, best_condition)
        print(f"\n{'='*70}\nVERDICT: [{verdict}]\n{'='*70}")

        rank_verdicts = None
        if len(part_b_results) == len(RANK_PROBE_DATASETS):
            rank_verdicts = rank_probe_verdict(part_b_results, best_condition)
            print("\nRank-probe adoption verdicts:")
            for k, v in rank_verdicts.items():
                print(f"  {k}: adopt={v['adopt']}  (TriviaQA delta={v['triviaqa_delta']:.4f}, "
                      f"TruthfulQA delta={v['truthfulqa_delta']:.4f})")

        out_path = os.path.join(args.results_dir, "session06_phase3_metrics.json")
        with open(out_path, "w") as f:
            json.dump({"part_a": part_a_results, "best_condition": best_condition,
                       "part_b": part_b_results, "rank_verdicts": rank_verdicts,
                       "part_c": part_c_results, "verdict": verdict, "verdict_detail": detail},
                      f, indent=2, default=str)
        print(f"\nWrote: {out_path}")
        lb_path = update_leaderboard_v2(part_a_results, part_c_results, best_condition, args.leaderboard)
        print(f"Wrote: {lb_path}")
        return

    if not args.dataset or not args.part:
        print("ERROR: --dataset and --part required (or --combine-part-a / --part-c / --combine)."); sys.exit(1)

    # --condition (part a only) loads just that condition's raw tensors -- the memory-saving
    # path for running Part A's 6 conditions as separate parallel jobs. Part b's rank probe
    # needs core_max plus whatever the best condition turns out to be, so it always loads
    # everything (condition=None).
    load_condition = args.condition if (args.part == "a" and args.condition) else None
    feats, y, prompt_idx, is_known = load_dataset(args.dataset, condition=load_condition)

    if args.part == "a":
        os.makedirs(args.results_dir, exist_ok=True)
        if args.condition:
            folds = compute_folds(feats, y, prompt_idx)
            result = run_condition_only(args.dataset, args.condition, feats, y, prompt_idx, is_known, folds)
            out_path, oof_path = write_condition_result(args.dataset, args.condition, result, args.results_dir)
            print(f"\nWrote: {out_path}")
            print(f"Wrote: {oof_path}  ({len(result['oof_rf'])}-beam RF/LR OOF score vectors -- "
                  f"kept out of the JSON, which is summaries only)")
            print(f"Next: run the other 5 conditions (in parallel, if you like), then "
                  f"--dataset {args.dataset} --combine-conditions")
            return
        result = run_part_a(args.dataset, feats, y, prompt_idx, is_known)
        out_path = os.path.join(args.results_dir, f"session06_phase3_partA_{args.dataset}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nWrote: {out_path}")
        print("Next: after all 4 datasets, run --combine-part-a")
        return

    if args.part == "b":
        if args.dataset not in RANK_PROBE_DATASETS:
            print(f"ERROR: rank probe only runs on {RANK_PROBE_DATASETS}."); sys.exit(1)
        best_path = legacy_read_path("session06_phase3_best_condition.json", args.results_dir)
        if not os.path.exists(best_path):
            print("ERROR: run --combine-part-a first to determine the best condition."); sys.exit(1)
        with open(best_path) as f:
            best_condition = json.load(f)["best_condition"]
        result = run_part_b(args.dataset, feats, y, prompt_idx, best_condition)
        out_path = os.path.join(args.results_dir, f"session06_phase3_partB_{args.dataset}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nWrote: {out_path}")
        return


if __name__ == "__main__":
    main()
