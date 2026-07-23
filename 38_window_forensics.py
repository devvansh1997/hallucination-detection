"""
38_window_forensics.py -- Session 05b: Window Forensics Completion
=======================================================================
CPU-only (loads the tokenizer only -- no model, no GPU). Completes the investigation
36_extraction_forensics.py's A1 gate stopped on: v2 and v3 have identical labels (sha256) but
disagree on per-beam completion-token counts (v3 total 112,693 vs v2's 110,159).

2. Diff anatomy: per-beam token-count delta, positional localization, and classification of
   every disputed token (content / EOS / post-EOS pad) using v3's actual saved token ids
   (truthfulqa_v3_sequences.pt) -- v2 never saved ids, so v3's ids are used as the ground truth
   for what the shared underlying content actually was (justified by A1's label-identity result).
3. The decisive replay: recompute positive-max pooled cores from the SAME raw-state-store
   tensors under three window definitions (v2 count / v3 count / canonical = content through
   first stop-token inclusive), and run paired core-only RF comparisons among them.
4. Explicit verdict logic based on what the disputed tokens turn out to be.
5. Writes canonical per-beam completion lengths so 35_derive_streams.py can re-derive every
   stream (core-max, q-static, q-velocity, kinematic) on the canonical window for Parts B/C.

Usage:
  python 38_window_forensics.py --self-test
  python 38_window_forensics.py --v2-manifest data/manifest_seeded_v1.json \
      --v3-raw-state-dir ../data/llama-3.1-8b-instruct/raw_state_store \
      --v3-sequences ../data/llama-3.1-8b-instruct/raw_state_store/truthfulqa_v3_sequences.pt \
      --model_folder llama-3.1-8b-instruct
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
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
s03 = _load("s03", "31_eval_session03.py")
pin_mod = _load("s03_pin", "30_pin_manifest.py")
gate_mod = _load("s04_gate", "34_gate_reconstruct_or_regenerate.py")

W_START, W_END = 15, 24
N_MID_LAYERS = W_END - W_START
SEED = 0
N_SPLITS = 5

paired_bootstrap_delta = s03.paired_bootstrap_delta
summarize_oof = s03.summarize_oof


# ==============================================================================
# TOKEN CLASSIFICATION
# ==============================================================================

def compute_eos_ids(tokenizer):
    """Exact reconstruction of the stop-token set used during generation
    (34_gate_reconstruct_or_regenerate.py / 29_generate_extract_band.py)."""
    eos_strs = [".", "!", "?", ".\n", "!\n", "?\n", "\n", "\n\n"]
    eos_ids = {tokenizer.eos_token_id}
    for s in eos_strs:
        eos_ids.update(tokenizer.encode(s, add_special_tokens=False))
        eos_ids.update(tokenizer.encode("Yes" + s, add_special_tokens=False)[1:])
    return eos_ids


def classify_token(tok_id, position, canonical_len, eos_token_id, eos_ids_set):
    if tok_id == eos_token_id:
        return "eos_or_pad"
    if tok_id in eos_ids_set:
        return "custom_stop"
    if position < canonical_len:
        return "content"
    return "post_stop_extra"


def find_canonical_length(comp_ids, eos_ids_set):
    """First stop-token (literal EOS or any custom stop id) inclusive; else the full window."""
    for i, tok in enumerate(comp_ids):
        if int(tok) in eos_ids_set:
            return i + 1
    return len(comp_ids)


# ==============================================================================
# ITEM 2 -- DIFF ANATOMY
# ==============================================================================

def diff_anatomy(v2_band_offsets, v3_seq_data, eos_ids_set, eos_token_id):
    v2_counts = np.diff(v2_band_offsets)
    n_beams = len(v2_counts)
    input_ids_list = v3_seq_data["input_ids"]
    prompt_lens = v3_seq_data["prompt_len"]

    v3_counts = np.array([len(input_ids_list[i]) - prompt_lens[i] for i in range(n_beams)])
    delta = v3_counts - v2_counts

    canonical_lengths = np.zeros(n_beams, dtype=np.int64)
    disputed_class_counts = {"content": 0, "eos_or_pad": 0, "custom_stop": 0, "post_stop_extra": 0}
    disputed_position_fractions = []
    n_beams_affected = 0

    for i in range(n_beams):
        comp_ids = input_ids_list[i][prompt_lens[i]:].tolist()
        canon_len = find_canonical_length(comp_ids, eos_ids_set)
        canonical_lengths[i] = canon_len

        d = int(delta[i])
        if d == 0:
            continue
        n_beams_affected += 1
        # disputed tokens = the extra tokens in the LONGER window beyond the SHORTER window's
        # boundary, localized at the end of v3's stored window (v3 count is always >= v2 in the
        # aggregate; check both directions per-beam without assuming sign)
        lo, hi = (v2_counts[i], v3_counts[i]) if d > 0 else (v3_counts[i], v2_counts[i])
        for pos in range(int(lo), int(hi)):
            tok = comp_ids[pos] if pos < len(comp_ids) else None
            if tok is None:
                continue
            cls = classify_token(tok, pos, canon_len, eos_token_id, eos_ids_set)
            disputed_class_counts[cls] = disputed_class_counts.get(cls, 0) + 1
            disputed_position_fractions.append(pos / max(v3_counts[i], 1))

    total_disputed = sum(disputed_class_counts.values())
    return {
        "n_beams": n_beams,
        "n_beams_with_delta": n_beams_affected,
        "pct_beams_with_delta": float(n_beams_affected / n_beams * 100),
        "delta_histogram": {str(k): int(v) for k, v in
                             zip(*np.unique(delta, return_counts=True))},
        "delta_mean": float(delta.mean()), "delta_std": float(delta.std()),
        "disputed_token_class_counts": disputed_class_counts,
        "disputed_token_class_fractions": {k: (v / total_disputed if total_disputed else 0.0)
                                            for k, v in disputed_class_counts.items()},
        "mean_disputed_position_fraction": float(np.mean(disputed_position_fractions))
                                             if disputed_position_fractions else float("nan"),
        "disputed_concentrated_at_end": bool(disputed_position_fractions and
                                              np.mean(disputed_position_fractions) > 0.8),
        "canonical_lengths": canonical_lengths,
        "v2_counts": v2_counts, "v3_counts": v3_counts,
    }


# ==============================================================================
# ITEM 3 -- THREE-WINDOW REPLAY
# ==============================================================================

def repool_under_window(v3_raw_state_dir, window_lengths):
    """window_lengths: array len n_beams, per-beam token count to keep (prefix truncation of
    v3's stored per-token tensor -- both windows start at the same completion-start position)."""
    meta_path = os.path.join(v3_raw_state_dir, "raw_state_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    shard_paths = [os.path.join(v3_raw_state_dir, s) for s in meta["shards"]]

    all_emb, prompt_ids, beam_idxs, labels = [], [], [], []
    global_i = 0
    for shard_path in shard_paths:
        raw, offsets, pid, bidx, lab = gate_mod.load_raw_state_shard(shard_path)
        for i in range(len(pid)):
            s, e = offsets[i], offsets[i + 1]
            T_i = e - s
            keep = min(int(window_lengths[global_i]), T_i)
            raw_beam = raw[s:s + keep].float() if keep > 0 else raw[s:s].float()
            if keep == 0:
                D = raw.shape[-1]
                all_emb.append(torch.zeros(N_MID_LAYERS, D))
            else:
                mid = raw_beam[:, :N_MID_LAYERS, :]
                all_emb.append(mid.max(dim=0).values)
            prompt_ids.append(int(pid[i])); beam_idxs.append(int(bidx[i])); labels.append(int(lab[i]))
            global_i += 1

    return {"all_emb": all_emb, "all_hallucination_flag": labels, "prompt_indices": prompt_ids}


def run_core_only_rf(core_data, folds, r_l=5, r_d=64, seed=SEED):
    X = torch.stack(core_data["all_emb"]).float().numpy()
    y = np.array(core_data["all_hallucination_flag"], dtype=np.int64)
    prompt_idx = np.array(core_data["prompt_indices"], dtype=np.int64)
    n_beams = X.shape[0]
    oof = np.full(n_beams, np.nan)
    fold_auroc = []
    for fold_i, (tr, va) in enumerate(folds):
        Xs = s01.mad_scale(X, tr)
        U_L, U_D = s01.compute_ul_ud(Xs[tr], r_l, r_d)
        core = s01.project_core(Xs, U_L, U_D)
        scores = s01.fit_eval("RF", core[tr], y[tr], core[va], seed + fold_i)
        oof[va] = scores
        fold_auroc.append(float(roc_auc_score(y[va], scores)))
    return {"RF": summarize_oof(oof, y, prompt_idx, fold_auroc, seed)}, {"RF": oof}, y, prompt_idx


def three_window_replay(v3_raw_state_dir, v2_counts, v3_counts, canonical_lengths, seed=SEED):
    windows = {"v2_window": v2_counts, "v3_window": v3_counts, "canonical": canonical_lengths}
    results, oofs, y_ref, prompt_ref = {}, {}, None, None
    for name, lengths in windows.items():
        core_data = repool_under_window(v3_raw_state_dir, lengths)
        n_beams = len(core_data["all_emb"])
        prompt_idx = np.array(core_data["prompt_indices"], dtype=np.int64)
        y = np.array(core_data["all_hallucination_flag"], dtype=np.int64)
        folds = list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(n_beams), y, groups=prompt_idx))
        summary, oof, y_out, prompt_out = run_core_only_rf(core_data, folds, seed=seed)
        results[name] = summary
        oofs[name] = oof
        y_ref, prompt_ref = y_out, prompt_out
        print(f"  {name}: RF pooled={summary['RF']['pooled_oof_auroc']:.4f}  "
              f"within-prompt={summary['RF']['within_prompt']['within_prompt_auroc']:.4f}")

    pairs = [("v3_window", "v2_window"), ("canonical", "v2_window"), ("canonical", "v3_window")]
    deltas = {}
    for a, b in pairs:
        d_pooled = paired_bootstrap_delta(oofs[a]["RF"], oofs[b]["RF"], y_ref, prompt_ref, seed=seed)
        d_wp = paired_bootstrap_delta(oofs[a]["RF"], oofs[b]["RF"], y_ref, prompt_ref, seed=seed,
                                       within_prompt=True)
        deltas[f"{a}_vs_{b}"] = {"pooled": d_pooled, "within_prompt": d_wp}
        print(f"  {a} vs {b}: pooled delta={d_pooled['mean_delta']:.4f} excl0={d_pooled['excludes_zero']}  "
              f"within-prompt delta={d_wp['mean_delta']:.4f} excl0={d_wp['excludes_zero']}")

    return results, deltas


# ==============================================================================
# ITEM 4 -- VERDICT
# ==============================================================================

def print_verdict(diff_result, window_deltas):
    content_frac = (diff_result["disputed_token_class_fractions"].get("content", 0) +
                    diff_result["disputed_token_class_fractions"].get("custom_stop", 0))
    pad_frac = diff_result["disputed_token_class_fractions"].get("eos_or_pad", 0)

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print(f"  Disputed tokens: {content_frac*100:.1f}% content/custom-stop, {pad_frac*100:.1f}% "
          f"eos-or-pad, concentrated at end of window: {diff_result['disputed_concentrated_at_end']}")

    canon_vs_v2 = window_deltas["canonical_vs_v2_window"]["pooled"]
    canon_vs_v3 = window_deltas["canonical_vs_v3_window"]["pooled"]

    if content_frac > pad_frac:
        verdict = ("Disputed tokens are content/custom-stop -> v2 UNDERCOUNTED (off-by-one class "
                    "bug). Canonical should be close to v3's window. v2-era ABSOLUTE numbers get "
                    "an asterisk; sessions 02-03 PAIRED null conclusions stand -- both arms of "
                    "every session02/03 comparison shared v2's (undercounted) window equally, so "
                    "the null findings (band ~= rand, honest fusion ~= neutral) are not invalidated "
                    "by this, only the absolute AUROC magnitudes are suspect.")
        expect = f"canonical ~= v3_window: delta={canon_vs_v3['mean_delta']:.4f} " \
                 f"(expect close to 0 if this diagnosis is right)"
    else:
        verdict = ("Disputed tokens are predominantly eos-or-pad -> canonical = the corrected "
                    "window; v3's extra tokens over v2 are a PADDING-STATE ARTIFACT, not real "
                    "content. Report honestly: this means v3's higher baseline AUROC was partly "
                    "driven by pooling over padding/EOS-repeat states, which happen to be "
                    "predictive (a real but accidental signal). Flag 'terminal summary "
                    "positions' as a candidate DELIBERATE feature for a future session -- "
                    "explicitly engineered, not smuggled in by a window bug.")
        expect = f"canonical should differ from v3_window: delta={canon_vs_v3['mean_delta']:.4f}"

    print(f"\n  {verdict}\n")
    print(f"  {expect}")
    print(f"  canonical vs v2_window: delta={canon_vs_v2['mean_delta']:.4f} "
          f"excludes_zero={canon_vs_v2['excludes_zero']}")
    return verdict


# ==============================================================================
# SELF-TEST
# ==============================================================================

def self_test():
    print("=" * 70)
    print("  SELF-TEST: token classification, three-window replay, verdict logic (no model)")
    print("=" * 70)
    tmp_dir = os.path.join(HERE, "results", "_selftest_window_forensics")
    os.makedirs(tmp_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    class FakeTokenizer:
        eos_token_id = 999

        def encode(self, s, add_special_tokens=False):
            # deterministic fake token ids per string, distinct from content-range ids
            return [500 + (hash(s) % 50)]

    tok = FakeTokenizer()
    eos_ids_set = compute_eos_ids(tok)
    assert tok.eos_token_id in eos_ids_set
    print(f"  [PASS] compute_eos_ids: {len(eos_ids_set)} stop-token ids reconstructed")

    n_beams, D = 40, 16
    per_beam_raw = []
    input_ids_list, prompt_lens = [], []
    labels, prompt_ids, beam_idxs = [], [], []
    v2_counts = []

    for i in range(n_beams):
        p_len = 5
        content_len = int(rng.integers(3, 10))
        # canonical: content tokens, then ONE eos token (custom stop), then some post-stop padding
        n_pad = int(rng.integers(0, 4))
        comp_ids = list(rng.integers(1, 400, size=content_len)) + [tok.eos_token_id] + [tok.eos_token_id] * n_pad
        full_ids = [rng.integers(1, 400) for _ in range(p_len)] + comp_ids
        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        prompt_lens.append(p_len)

        T_i = len(comp_ids)   # v3's stored window = everything captured (content+eos+pad)
        h = {l: torch.tensor(rng.normal(0, 1, size=(T_i, D)).astype(np.float32)) for l in range(W_START, W_END)}
        h["final_norm"] = torch.tensor(rng.normal(0, 1, size=(T_i, D)).astype(np.float32))
        per_beam_raw.append(gate_mod.pack_raw_state_beam(h))

        # v2 "undercounts" by 1 relative to canonical (off-by-one) for half the beams
        v2_counts.append(content_len - 1 if i % 2 == 0 else content_len + 1)  # test both directions

        labels.append(int(rng.integers(0, 2))); prompt_ids.append(i // 4); beam_idxs.append(i % 4)

    v2_counts = np.array(v2_counts)
    v3_offsets = np.zeros(n_beams + 1, dtype=np.int64)
    for i in range(n_beams):
        v3_offsets[i + 1] = v3_offsets[i] + (len(input_ids_list[i]) - prompt_lens[i])
    v2_offsets = np.concatenate([[0], np.cumsum(v2_counts)])

    v3_seq_data = {"input_ids": input_ids_list, "prompt_len": prompt_lens}
    diff_result = diff_anatomy(v2_offsets, v3_seq_data, eos_ids_set, tok.eos_token_id)
    assert diff_result["n_beams"] == n_beams
    assert diff_result["n_beams_with_delta"] > 0
    print(f"  [PASS] diff_anatomy: {diff_result['n_beams_with_delta']}/{n_beams} beams affected, "
          f"classes={diff_result['disputed_token_class_counts']}")
    assert sum(diff_result["disputed_token_class_counts"].values()) > 0

    V_R, V_rand = torch.randn(8, D), torch.randn(6, D)
    v3_dir = os.path.join(tmp_dir, "v3_raw_store")
    gate_mod.pack_and_save_raw_store(per_beam_raw, prompt_ids, beam_idxs, labels, V_R, V_rand,
                                      v3_dir, "self-test", 0, {"do_sample": True}, "selftest")

    v3_counts = diff_result["v3_counts"]
    canonical_lengths = diff_result["canonical_lengths"]

    core_v2 = repool_under_window(v3_dir, v2_counts)
    core_v3 = repool_under_window(v3_dir, v3_counts)
    core_canon = repool_under_window(v3_dir, canonical_lengths)
    assert len(core_v2["all_emb"]) == n_beams == len(core_canon["all_emb"])
    print(f"  [PASS] repool_under_window: produced {len(core_v2['all_emb'])} beams for all 3 windows")

    # canonical should exactly match v3-window pooling for beams where v2 undercounted (i%2==0,
    # since canonical == content_len == v3 stored minus padding, and v2 was content_len-1) --
    # spot check beam 0 specifically
    assert canonical_lengths[0] <= v3_counts[0]
    print(f"  [PASS] canonical_lengths <= v3_counts for beam 0 ({canonical_lengths[0]} <= {v3_counts[0]})")

    folds = list(GroupKFold(n_splits=5).split(np.zeros(n_beams), labels, groups=prompt_ids))
    results, oof_deltas = three_window_replay(v3_dir, v2_counts, v3_counts, canonical_lengths, seed=0)
    assert "v2_window" in results and "v3_window" in results and "canonical" in results
    print(f"  [PASS] three_window_replay ran all 3 windows + pairwise deltas")

    verdict = print_verdict(diff_result, oof_deltas)
    assert isinstance(verdict, str) and len(verdict) > 0
    print(f"  [PASS] print_verdict produced a verdict string")

    out_path = os.path.join(HERE, "results", "session05b_selftest_metrics.json")
    with open(out_path, "w") as f:
        json.dump({
            "diff_anatomy": {k: v for k, v in diff_result.items()
                              if k not in ("canonical_lengths", "v2_counts", "v3_counts")},
            "three_window_results": {name: r["RF"] for name, r in results.items()},
            "window_deltas": oof_deltas,
        }, f, indent=2, default=str)
    assert os.path.exists(out_path)
    print(f"  [PASS] JSON written to {out_path}")

    print("\n[PASS] All self-test assertions passed.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-manifest", type=str, default="data/manifest_seeded_v1.json")
    parser.add_argument("--v3-raw-state-dir", type=str, default=None)
    parser.add_argument("--v3-sequences", type=str, default=None)
    parser.add_argument("--model_folder", type=str, default="llama-3.1-8b-instruct")
    parser.add_argument("--output-json", type=str, default="results/session05b_metrics.json")
    parser.add_argument("--canonical-lengths-out", type=str,
                         default="results/session05b_canonical_lengths.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.v3_raw_state_dir or not args.v3_sequences:
        print("ERROR: --v3-raw-state-dir and --v3-sequences required."); sys.exit(1)

    manifest_path = os.path.join(HERE, args.v2_manifest) if not os.path.isabs(args.v2_manifest) else args.v2_manifest
    v2_manifest = pin_mod.verify_manifest(manifest_path)
    band_meta_path = v2_manifest["band_meta_path"]
    with open(band_meta_path) as f:
        band_meta = json.load(f)
    v2_shard_paths = [os.path.join(os.path.dirname(band_meta_path), s) for s in band_meta["shards"]]
    v2_packed = band_mod.load_packed(v2_shard_paths)
    v2_offsets = v2_packed["offsets"]

    import yaml
    from transformers import AutoTokenizer
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == args.model_folder)
    print(f"Loading tokenizer only (CPU-safe, no model/GPU): {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    eos_ids_set = compute_eos_ids(tokenizer)
    print(f"Reconstructed stop-token set: {len(eos_ids_set)} ids")

    v3_seq_data = torch.load(args.v3_sequences, weights_only=False)
    v3_decoding_config = v3_seq_data.get("decoding_config", {})
    print(f"NOTE: v3's regeneration ran under transformers={v3_decoding_config.get('transformers', '?')} "
          f"torch={v3_decoding_config.get('torch', '?')} cuda={v3_decoding_config.get('cuda', '?')}. "
          f"v2's original extraction's library versions were never logged (a gap from before this "
          f"version-pinning convention existed) -- a transformers-version difference in generate()'s "
          f"stopping-criteria/hidden-states handling around custom eos_token_id lists is a real "
          f"alternative (or contributing) explanation for the token-count discrepancy, independent "
          f"of the content/pad classification below. Cannot be ruled in or out without v2's versions.")

    print("\n[Item 2] Diff anatomy ...")
    diff_result = diff_anatomy(v2_offsets, v3_seq_data, eos_ids_set, tokenizer.eos_token_id)
    print(f"  {diff_result['n_beams_with_delta']}/{diff_result['n_beams']} beams affected "
          f"({diff_result['pct_beams_with_delta']:.1f}%)")
    print(f"  delta mean={diff_result['delta_mean']:.3f} std={diff_result['delta_std']:.3f}")
    print(f"  disputed token classes: {diff_result['disputed_token_class_counts']}")
    print(f"  disputed token fractions: {diff_result['disputed_token_class_fractions']}")
    print(f"  concentrated at end of window: {diff_result['disputed_concentrated_at_end']}")

    print("\n[Item 3] Three-window replay ...")
    results, window_deltas = three_window_replay(args.v3_raw_state_dir, diff_result["v2_counts"],
                                                    diff_result["v3_counts"],
                                                    diff_result["canonical_lengths"], seed=SEED)

    print("\n[Item 4] Verdict ...")
    verdict = print_verdict(diff_result, window_deltas)

    with open(args.canonical_lengths_out, "w") as f:
        json.dump({"canonical_lengths": diff_result["canonical_lengths"].tolist(),
                    "prompt_indices": v3_seq_data.get("prompt_id", None)}, f, indent=2, default=str)
    print(f"\nWrote canonical lengths: {args.canonical_lengths_out}")

    output = {
        "diff_anatomy": {k: v for k, v in diff_result.items()
                          if k not in ("canonical_lengths", "v2_counts", "v3_counts")},
        "three_window_results": {name: r["RF"] for name, r in results.items()},
        "window_deltas": window_deltas,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Wrote: {args.output_json}")


if __name__ == "__main__":
    main()
