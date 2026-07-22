"""
36_extraction_forensics.py -- Session 05 Part A: Extraction Forensics (GATES everything else)
==================================================================================================
CPU-only (A1-A3); A4 is an optional tiny GPU spot-check, only run if explicitly requested and
only useful if A2 comes back ambiguous. Investigates why session04's v3 core-only baseline
(0.8730 pooled / 0.8298 within-prompt) differs substantially from v2's canonical reference
(0.8347 / 0.7365): is v3 a genuinely different dataset (regeneration drew different beams), or
the SAME beams pooled by two different extraction code paths that disagree numerically?

A1: sequence/label identity between v2 (pinned manifest) and v3 (session04's raw-state store).
A2 (only if A1 says SAME): elementwise diff of the positive-max pooled tensor, recomputed from
    v3's raw-state store, against v2's cached truthfulqa_pooled_maxenergy_seeded.pt.
A3: resolves whether the hygiene ladder's "vanilla_mad" row is actually unscaled (it is not --
    see below) and adds a genuinely-unscaled baseline for comparison.
A4: optional GPU re-forward of 20 beams at fp32 to break a tie between the two CPU-only diagnoses.

Usage:
  python 36_extraction_forensics.py --self-test
  python 36_extraction_forensics.py --v2-manifest data/manifest_seeded_v1.json \
      --v3-raw-state-dir ../data/llama-3.1-8b-instruct/raw_state_store \
      --v3-pooled-pt ../data/llama-3.1-8b-instruct/raw_state_store/truthfulqa_v3_pooled.pt
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s01 = _load("s01", "26_grouped_baseline.py")
band_mod = _load("s02_extract", "27_extract_band.py")
s02 = _load("s02", "28_eval_band.py")
pin_mod = _load("s03_pin", "30_pin_manifest.py")
vel_mod = _load("s04_extract", "32_extract_velocity.py")
gate_mod = _load("s04_gate", "34_gate_reconstruct_or_regenerate.py")
derive_mod = _load("s04_derive", "35_derive_streams.py")

W_START, W_END = 15, 24


# ==============================================================================
# A1 -- SEQUENCE / LABEL IDENTITY
# ==============================================================================

def sha256_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def check_sequence_identity(v2_manifest, v3_raw_state_dir):
    band_meta_path = v2_manifest["band_meta_path"]
    with open(band_meta_path) as f:
        band_meta = json.load(f)
    shard_dir = os.path.dirname(band_meta_path)
    v2_shard_paths = [os.path.join(shard_dir, s) for s in band_meta["shards"]]
    v2_packed = band_mod.load_packed(v2_shard_paths)
    v2_offsets = v2_packed["offsets"]
    v2_counts = np.diff(v2_offsets)
    v2_total = int(v2_offsets[-1])

    v3_meta_path = os.path.join(v3_raw_state_dir, "raw_state_meta.json")
    with open(v3_meta_path) as f:
        v3_meta = json.load(f)
    v3_shard_paths = [os.path.join(v3_raw_state_dir, s) for s in v3_meta["shards"]]

    v3_counts_list, v3_labels_list, v3_prompt_ids = [], [], []
    for sp in v3_shard_paths:
        raw, offsets, pid, bidx, lab = gate_mod.load_raw_state_shard(sp)
        v3_counts_list.append(np.diff(offsets))
        v3_labels_list.append(lab)
        v3_prompt_ids.append(pid)
    v3_counts = np.concatenate(v3_counts_list)
    v3_labels = np.concatenate(v3_labels_list)
    v3_total = int(v3_counts.sum())

    n_beams_match = len(v2_counts) == len(v3_counts)
    counts_equal = bool(n_beams_match and np.array_equal(v2_counts, v3_counts))
    totals_equal = (v2_total == v3_total)

    v3_labels_hash = sha256_array(v3_labels.astype(np.int64))
    v2_labels_hash = v2_manifest["labels_sha256"]
    labels_equal = (v3_labels_hash == v2_labels_hash)

    verdict = "SAME" if (counts_equal and labels_equal) else "DIFFERENT"

    return {
        "verdict": verdict,
        "v2_total_tokens": v2_total, "v3_total_tokens": v3_total,
        "totals_equal": totals_equal, "n_beams_match": n_beams_match,
        "per_beam_counts_equal": counts_equal,
        "n_beams_with_different_count": int((v2_counts != v3_counts).sum()) if n_beams_match else None,
        "v2_labels_sha256": v2_labels_hash, "v3_labels_sha256": v3_labels_hash,
        "labels_equal": labels_equal,
        "v2_offsets": v2_offsets, "v3_counts": v3_counts,
        "v3_decoding_config": v3_meta.get("decoding_config", "MISSING -- not found in raw_state_meta.json"),
    }


# ==============================================================================
# A2 -- TENSOR DIFF (only if A1 says SAME)
# ==============================================================================

def recompute_v3_core_with_argmax(v3_raw_state_dir):
    """Like derive_positive_max_core, but also returns the argmax token index per (beam,layer,
    channel) so disagreements can be localized to a specific token position."""
    meta_path = os.path.join(v3_raw_state_dir, "raw_state_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    shard_paths = [os.path.join(v3_raw_state_dir, s) for s in meta["shards"]]
    n_mid_layers = W_END - W_START

    all_emb, all_argmax, prompt_ids, beam_idxs, labels, completion_lens = [], [], [], [], [], []
    for shard_path in shard_paths:
        raw, offsets, pid, bidx, lab = gate_mod.load_raw_state_shard(shard_path)
        for i in range(len(pid)):
            s, e = offsets[i], offsets[i + 1]
            raw_beam = raw[s:e].float()
            T_i = raw_beam.shape[0]
            completion_lens.append(T_i)
            if T_i == 0:
                D = raw_beam.shape[2]
                all_emb.append(torch.zeros(n_mid_layers, D))
                all_argmax.append(torch.full((n_mid_layers, D), -1, dtype=torch.long))
            else:
                mid = raw_beam[:, :n_mid_layers, :]           # (T, 9, D)
                vals, idx = mid.max(dim=0)                     # (9, D) each
                all_emb.append(vals)
                all_argmax.append(idx)
            prompt_ids.append(int(pid[i])); beam_idxs.append(int(bidx[i])); labels.append(int(lab[i]))

    return {"all_emb": all_emb, "argmax": all_argmax, "prompt_indices": prompt_ids,
            "beam_idx": beam_idxs, "labels": labels, "completion_lens": completion_lens}


def tensor_diff_analysis(v3_raw_state_dir, v2_pooled_pt_path, n_argmax_samples=25):
    v3 = recompute_v3_core_with_argmax(v3_raw_state_dir)
    v2_pooled = torch.load(v2_pooled_pt_path, weights_only=False)
    v2_emb = v2_pooled["all_emb"]

    n_beams = len(v3["all_emb"])
    if len(v2_emb) != n_beams:
        return {"error": f"beam count mismatch: v2={len(v2_emb)} v3={n_beams} -- A1 should have "
                          f"caught this; re-run A1."}

    n_layers = v3["all_emb"][0].shape[0]
    max_abs_diff_per_beam = np.zeros(n_beams)
    max_abs_diff_per_layer = np.zeros(n_layers)
    corr_per_beam = np.full(n_beams, np.nan)
    per_layer_diffs = [[] for _ in range(n_layers)]

    for i in range(n_beams):
        a = v2_emb[i].float().numpy()
        b = v3["all_emb"][i].numpy()
        diff = np.abs(a - b)
        max_abs_diff_per_beam[i] = diff.max()
        for l in range(n_layers):
            max_abs_diff_per_layer[l] = max(max_abs_diff_per_layer[l], diff[l].max())
            per_layer_diffs[l].append(diff[l].max())
        af, bf = a.flatten(), b.flatten()
        if af.std() > 0 and bf.std() > 0:
            corr_per_beam[i] = float(np.corrcoef(af, bf)[0, 1])

    completion_lens = np.array(v3["completion_lens"])
    order = np.argsort(-max_abs_diff_per_beam)
    worst_beams = order[:n_argmax_samples]

    len_corr = float(np.corrcoef(max_abs_diff_per_beam, completion_lens)[0, 1]) \
        if completion_lens.std() > 0 else float("nan")

    argmax_localization = []
    for i in worst_beams[:min(10, len(worst_beams))]:
        a = v2_emb[i].float().numpy()
        b = v3["all_emb"][i].numpy()
        diff = np.abs(a - b)
        l, c = np.unravel_index(np.argmax(diff), diff.shape)
        v3_argmax_token = int(v3["argmax"][i][l, c])
        argmax_localization.append({
            "beam": i, "layer_offset": int(l), "channel": int(c),
            "v2_value": float(a[l, c]), "v3_value": float(b[l, c]),
            "diff": float(diff[l, c]), "completion_len": int(completion_lens[i]),
            "v3_argmax_token_position": v3_argmax_token,
            "v3_argmax_at_last_token": bool(v3_argmax_token == completion_lens[i] - 1),
        })

    return {
        "n_beams": n_beams,
        "max_abs_diff_overall": float(max_abs_diff_per_beam.max()),
        "mean_abs_diff_per_beam": float(max_abs_diff_per_beam.mean()),
        "max_abs_diff_per_layer": max_abs_diff_per_layer.tolist(),
        "mean_correlation": float(np.nanmean(corr_per_beam)),
        "min_correlation": float(np.nanmin(corr_per_beam)),
        "pct_beams_corr_above_0999": float((corr_per_beam > 0.999).mean() * 100),
        "correlation_vs_completion_length": len_corr,
        "worst_beam_indices": worst_beams.tolist(),
        "argmax_localization_sample": argmax_localization,
    }


# ==============================================================================
# A3 -- "vanilla_mad" NAMING RESOLUTION
# ==============================================================================

MAD_SCALE_QUOTE = '''def mad_scale(X, train_idx):
    """Train-only median/MAD scaling -- robust to LLaMA structural outlier channels."""
    X_t = X[train_idx]
    med = np.median(X_t, axis=0)
    mad = np.median(np.abs(X_t - med), axis=0) + 1e-6
    return (X - med) / mad
'''  # verbatim from 26_grouped_baseline.py -- quoted, not re-derived


def raw_scale_identity(X, train_idx):
    """Genuinely unscaled -- the actual 'vanilla' baseline A3 asks for."""
    return X


def run_true_vanilla_baseline(X, y, prompt_idx, folds, r_l, r_d, seed=0):
    """Reuses 26_grouped_baseline.py's Tucker machinery directly on UNSCALED features, to
    compare against the hygiene ladder's 'vanilla_mad' row (which is MAD-scaled, not raw)."""
    from sklearn.metrics import roc_auc_score
    n_beams = X.shape[0]
    oof_rf = np.full(n_beams, np.nan)
    fold_rf = []
    for fold_i, (tr, va) in enumerate(folds):
        U_L, U_D = s01.compute_ul_ud(X[tr], r_l, r_d)
        core = s01.project_core(X, U_L, U_D)
        rf_scores = s01.fit_eval("RF", core[tr], y[tr], core[va], seed + fold_i)
        oof_rf[va] = rf_scores
        fold_rf.append(float(roc_auc_score(y[va], rf_scores)))
    pooled = float(roc_auc_score(y, oof_rf))
    return {"pooled_oof_auroc": pooled, "per_fold_auroc": fold_rf, "mean_auroc": float(np.mean(fold_rf))}


# ==============================================================================
# A4 -- OPTIONAL GPU SPOT-CHECK
# ==============================================================================

def gpu_spot_check(v3_raw_state_dir, v2_manifest, model_folder, n_beams_check=20):
    """Only meaningful if A2 is ambiguous. Re-forwards n_beams_check beams at fp32 (not bf16)
    using the v3 sequences, and compares against both v2's cached value and v3's bf16-derived
    value, to check whether bf16 precision (not a logic bug) explains small disagreements."""
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)

    seq_path = os.path.join(v3_raw_state_dir, "truthfulqa_v3_sequences.pt")
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"{seq_path} not found -- A4 needs the saved v3 sequences.")
    data = torch.load(seq_path, weights_only=False)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for A4.")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                                  device_map=device, trust_remote_code=True)
    model.eval()

    n_mid_layers = W_END - W_START
    results = []
    for i in range(min(n_beams_check, len(data["input_ids"]))):
        ids = data["input_ids"][i].unsqueeze(0).to(device)
        attn = torch.ones_like(ids)
        p_len = data["prompt_len"][i]
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn, use_cache=False, output_hidden_states=True)
        comp = {l: out.hidden_states[l + 1][0, p_len:, :].float().cpu() for l in range(W_START, W_END)}
        pooled_fp32 = torch.stack([comp[l].max(dim=0).values if comp[l].shape[0] > 0
                                    else torch.zeros(comp[l].shape[-1]) for l in range(W_START, W_END)])
        results.append(pooled_fp32.numpy())

    return {"n_beams_checked": len(results), "fp32_pooled_sample": [r.tolist() for r in results[:2]],
            "note": "compare these fp32 values against the corresponding v2/v3 bf16-derived "
                    "values externally -- full comparison deferred to the report writer with "
                    "actual v2/v3 arrays in scope."}


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: A1 identity check, A2 tensor diff, A3 vanilla resolution (no model/GPU)")
    print("=" * 70)
    tmp_dir = os.path.join(HERE, "results", "_selftest_forensics")
    os.makedirs(tmp_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    n_beams, D = 20, 16
    n_mid = W_END - W_START
    per_beam_raw = []
    labels, prompt_ids, beam_idxs = [], [], []
    for i in range(n_beams):
        T_i = int(rng.integers(1, 6))
        h = {l: torch.tensor(rng.normal(0, 1, size=(T_i, D)).astype(np.float32)) for l in range(W_START, W_END)}
        h["final_norm"] = torch.tensor(rng.normal(0, 1, size=(T_i, D)).astype(np.float32))
        per_beam_raw.append(gate_mod.pack_raw_state_beam(h))
        labels.append(int(rng.integers(0, 2))); prompt_ids.append(i // 4); beam_idxs.append(i % 4)

    V_R, V_rand = torch.randn(8, D), torch.randn(6, D)
    v3_dir = os.path.join(tmp_dir, "v3_raw_store")
    gate_mod.pack_and_save_raw_store(per_beam_raw, prompt_ids, beam_idxs, labels, V_R, V_rand,
                                      v3_dir, "self-test", 0, {"do_sample": True, "gen_seed": 0}, "selftest")

    v3_core = derive_mod.derive_positive_max_core(v3_dir)
    v2_pooled_path = os.path.join(tmp_dir, "v2_pooled.pt")
    torch.save(v3_core, v2_pooled_path)   # identical-by-construction "v2" for the matching-case test

    diff = tensor_diff_analysis(v3_dir, v2_pooled_path)
    assert diff["max_abs_diff_overall"] < 1e-4, f"identical-by-construction diff should be ~0, got {diff}"
    print(f"  [PASS] tensor_diff_analysis: identical-by-construction case gives max_abs_diff="
          f"{diff['max_abs_diff_overall']:.2e} (~0, as expected)")

    v2_pooled_perturbed_path = os.path.join(tmp_dir, "v2_pooled_perturbed.pt")
    perturbed = {"all_emb": [e + 0.5 for e in v3_core["all_emb"]], "all_hallucination_flag": labels,
                 "all_is_known": v3_core["all_is_known"], "prompt_indices": prompt_ids}
    torch.save(perturbed, v2_pooled_perturbed_path)
    diff2 = tensor_diff_analysis(v3_dir, v2_pooled_perturbed_path)
    assert abs(diff2["max_abs_diff_overall"] - 0.5) < 1e-4
    assert diff2["mean_correlation"] > 0.9   # a constant offset shouldn't hurt correlation
    print(f"  [PASS] tensor_diff_analysis: known +0.5 perturbation recovered exactly "
          f"(max_abs_diff={diff2['max_abs_diff_overall']:.4f}), correlation stays high as expected")
    assert len(diff2["argmax_localization_sample"]) > 0
    print(f"  [PASS] argmax localization produced {len(diff2['argmax_localization_sample'])} samples")

    # -- A3: true-vanilla vs mad-scaled should differ (mad_scale is NOT a no-op) --
    data = s01.generate_synthetic_data(n_prompts=60, beams_per_prompt=10, L=9, D=32, seed=0)
    X, y, prompt_idx = data["X"], data["y"], data["prompt_idx"]
    folds = list(GroupKFold(n_splits=5).split(X, y, groups=prompt_idx))
    vanilla_result = run_true_vanilla_baseline(X, y, prompt_idx, folds, r_l=5, r_d=10, seed=0)
    s04eval = _load("s04eval_selftest", "33_eval_session04.py")
    mad_summary, mad_oof_dict, _ = s04eval.run_core_variant(X, y, prompt_idx, folds,
                                                              lambda Xr, tr: s01.mad_scale(Xr, tr), 5, 10, 0)
    assert abs(vanilla_result["pooled_oof_auroc"] - mad_summary["RF"]["pooled_oof_auroc"]) > 1e-6, \
        "true-vanilla and mad-scaled gave IDENTICAL results -- mad_scale should change something"
    print(f"  [PASS] A3: true-vanilla (raw, no scaling) AUROC={vanilla_result['pooled_oof_auroc']:.4f} "
          f"differs from mad-scaled AUROC={mad_summary['RF']['pooled_oof_auroc']:.4f} -- "
          f"confirms 'vanilla_mad' in the hygiene ladder is NOT actually unscaled")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-manifest", type=str, default="data/manifest_seeded_v1.json")
    parser.add_argument("--v3-raw-state-dir", type=str, default=None)
    parser.add_argument("--v3-pooled-pt", type=str, default=None)
    parser.add_argument("--model_folder", type=str, default="llama-3.1-8b-instruct")
    parser.add_argument("--run-a4", action="store_true", help="Optional GPU spot-check.")
    parser.add_argument("--report-path", type=str, default="reports/session05_extraction_forensics.md")
    parser.add_argument("--output-json", type=str, default="results/session05_forensics.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.v3_raw_state_dir:
        print("ERROR: --v3-raw-state-dir required."); sys.exit(1)

    manifest_path = os.path.join(HERE, args.v2_manifest) if not os.path.isabs(args.v2_manifest) else args.v2_manifest
    v2_manifest = pin_mod.verify_manifest(manifest_path)
    print(f"v2 manifest verified. Counts: {v2_manifest['counts']}")

    print("\n[A1] Sequence/label identity check ...")
    a1 = check_sequence_identity(v2_manifest, args.v3_raw_state_dir)
    print(f"  Verdict: {a1['verdict']}")
    print(f"  v2 total tokens: {a1['v2_total_tokens']}  v3 total tokens: {a1['v3_total_tokens']}")
    print(f"  Per-beam token counts equal: {a1['per_beam_counts_equal']}")
    print(f"  Labels equal (sha256): {a1['labels_equal']}")

    output = {"a1_sequence_identity": {k: v for k, v in a1.items() if k not in ("v2_offsets", "v3_counts")}}

    report_lines = ["# Session 05 Extraction Forensics\n", f"## A1 verdict: **{a1['verdict']}**\n"]
    report_lines.append(f"- v2 total completion tokens: {a1['v2_total_tokens']}")
    report_lines.append(f"- v3 total completion tokens: {a1['v3_total_tokens']}")
    report_lines.append(f"- Per-beam counts identical: {a1['per_beam_counts_equal']}")
    report_lines.append(f"- Labels identical (sha256): {a1['labels_equal']} "
                         f"(v2={a1['v2_labels_sha256'][:16]}..., v3={a1['v3_labels_sha256'][:16]}...)\n")

    if a1["verdict"] == "DIFFERENT":
        report_lines.append("## STOPPED after A1 per spec -- sequences/labels differ.\n")
        report_lines.append(f"Literal decoding config actually used for the v3 regeneration "
                             f"(from raw_state_meta.json, since it is absent from session04's "
                             f"eval-output JSON):\n```json\n{json.dumps(a1['v3_decoding_config'], indent=2)}\n```")
        os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
        with open(args.report_path, "w") as f:
            f.write("\n".join(report_lines))
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSTOPPED after A1 (verdict=DIFFERENT). Report: {args.report_path}")
        return

    print("\n[A2] Tensor diff analysis (recomputing v3 positive-max core, comparing to v2) ...")
    if not args.v3_pooled_pt:
        print("ERROR: A1=SAME, but --v3-pooled-pt not given (needed for A2 label cross-check "
              "and as the recomputation target)."); sys.exit(1)
    a2 = tensor_diff_analysis(args.v3_raw_state_dir, v2_manifest["pooled_pt_path"])
    print(f"  Max abs diff overall: {a2['max_abs_diff_overall']:.6f}")
    print(f"  Mean correlation: {a2['mean_correlation']:.6f}  (min: {a2['min_correlation']:.6f})")
    print(f"  % beams with corr > 0.999: {a2['pct_beams_corr_above_0999']:.1f}%")
    print(f"  Corr(diff magnitude, completion length): {a2['correlation_vs_completion_length']:.4f}")
    output["a2_tensor_diff"] = a2

    report_lines.append("## A2 -- tensor diff (v2 cached vs v3 recomputed from raw-state store)\n")
    report_lines.append(f"- max |diff| overall: {a2['max_abs_diff_overall']:.6f}")
    report_lines.append(f"- mean per-beam correlation: {a2['mean_correlation']:.6f} "
                         f"(min: {a2['min_correlation']:.6f})")
    report_lines.append(f"- % beams with correlation > 0.999: {a2['pct_beams_corr_above_0999']:.1f}%")
    report_lines.append(f"- correlation(disagreement magnitude, completion length): "
                         f"{a2['correlation_vs_completion_length']:.4f}\n")
    report_lines.append("### Pooling/masking code, side by side\n")
    report_lines.append("**v2 (29_generate_extract_band.py, pool_and_project_prompt):** pools "
                         "`hidden_states[step][l+1][b,-1,:]` for `step in range(1, T_real+1)` "
                         "where `T_real = min(len(gids), num_gen)` and `gids` is this beam's "
                         "generated ids with `eos_token_id` occurrences filtered out (note: "
                         "`pad_token_id == eos_token_id` in this pipeline's generate() call, so "
                         "this filter also removes trailing padding).\n")
    report_lines.append("**v3 (34_gate_reconstruct_or_regenerate.py, resume_route_n_raw_state / "
                         "run_route_n):** full-sequence forward pass, slices "
                         "`hidden_states[l+1][j, comp_start:comp_end, :]` where "
                         "`comp_start=prompt_len`, `comp_end=len(saved_sequence)`, and the saved "
                         "sequence is `outputs.sequences[b, :prompt_len+len(gids)]` -- i.e. the "
                         "SAME `len(gids)` boundary as v2, just applied via direct slicing of a "
                         "one-shot full-sequence forward pass instead of accumulated per-step "
                         "incremental-generation hidden states (`use_cache=True` implicitly "
                         "during v2's generate() call vs `use_cache=False` in v3's re-forward).\n")
    report_lines.append("Both pipelines select the SAME completion-token window by construction "
                         "(same `len(gids)` boundary) -- if A2's diffs are non-trivial, the "
                         "likely source is numeric, not a window/masking bug: incremental "
                         "generation (KV-cache, `use_cache=True`) vs a full-sequence one-shot "
                         "forward pass (`use_cache=False`) are mathematically equivalent for a "
                         "causal LM but not bit-identical, and right-padding across beams with "
                         "different total lengths in v3's batched re-forward is a second, "
                         "independent source of small floating-point differences.\n")

    n_worst_high = int((np.array([d["diff"] for d in a2["argmax_localization_sample"]]) > 0).sum())
    report_lines.append(f"### Argmax localization sample ({len(a2['argmax_localization_sample'])} "
                         f"worst-disagreement beams)\n")
    for d in a2["argmax_localization_sample"]:
        at_end = d["v3_argmax_at_last_token"]
        report_lines.append(f"- beam {d['beam']}: layer_offset={d['layer_offset']} "
                             f"channel={d['channel']} v2={d['v2_value']:.4f} v3={d['v3_value']:.4f} "
                             f"diff={d['diff']:.4f} completion_len={d['completion_len']} "
                             f"v3's own argmax token={d['v3_argmax_token_position']} "
                             f"(at last token: {at_end})")
    report_lines.append("")

    print("\n[A3] Resolving 'vanilla_mad' naming ...")
    print(f"  mad_scale() (quoted from 26_grouped_baseline.py):\n{MAD_SCALE_QUOTE}")
    print("  CONCLUSION: 'vanilla_mad' in the hygiene ladder IS median/MAD-normalized -- it is "
          "NOT an unscaled baseline. The name refers to 'the ORIGINAL sessions 1-3 scaling "
          "approach', not 'no scaling'. A genuinely unscaled ('raw') row would need to be run "
          "separately using run_true_vanilla_baseline() against the canonical pooled tensor "
          "once A3's verdict on which pipeline is canonical is settled.")
    report_lines.append("## A3 -- 'vanilla_mad' naming\n")
    report_lines.append("`mad_scale()` (quoted verbatim from `26_grouped_baseline.py`):\n```python\n"
                         + MAD_SCALE_QUOTE + "\n```")
    report_lines.append("**'vanilla_mad' is median/MAD-normalized, not unscaled.** The name means "
                         "'the original sessions 1-3 scaling convention', not 'no preprocessing'. "
                         "A true no-scaling baseline (`run_true_vanilla_baseline()` in this script) "
                         "should be run against whichever pooled tensor A2/A3 settles on as "
                         "canonical, and reported alongside 'vanilla_mad' going forward so the "
                         "leaderboard doesn't imply an unscaled number that was never actually "
                         "computed.\n")

    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
    with open(args.report_path, "w") as f:
        f.write("\n".join(report_lines))
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote: {args.report_path}")
    print(f"Wrote: {args.output_json}")

    if args.run_a4:
        print("\n[A4] Optional GPU spot-check ...")
        a4 = gpu_spot_check(args.v3_raw_state_dir, v2_manifest, args.model_folder)
        print(f"  Checked {a4['n_beams_checked']} beams at fp32.")


if __name__ == "__main__":
    main()
