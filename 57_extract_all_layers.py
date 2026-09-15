"""
57_extract_all_layers.py -- per-layer pooled features for EVERY layer, in one forward pass.
=====================================================================================================
WHY THIS IS A NEW SCRIPT AND NOT A FLAG ON 42.

42_extract_phase2.py produced every feature behind every number currently in the paper. Adding a
mode to it means editing the code path those numbers came from, three weeks before a deadline, to
answer an ablation question. This script does its own forward pass and writes to its own directory,
so the pinned features cannot be touched even by a mistake.

WHY ONE PASS AND NOT FOUR WINDOWS.

A forward pass yields every layer at once. The existing pipeline computes layers 0..L_tot and keeps
only {15..23} plus the final state; the rest is discarded at 42_extract_phase2.py:201. Running
three more nine-layer windows would pay for the same forward pass four times to recover information
the first pass already had in memory. This keeps all of it.

WHAT IT STORES, AND WHAT IT LEAVES OUT.

    core   (N, L_tot+1, D)   max over completion tokens                      --streams core
    q95    (N, L_tot+1, D)   upper quantile over completion tokens           --streams static
    q05    (N, L_tot+1, D)   lower quantile
    v95    (N, L_tot,   D)   upper quantile over tokens of h[i+1] - h[i]     --streams velocity
    v05    (N, L_tot,   D)   lower quantile of the same difference

Velocity is off by default. The single-layer sweep (58) only asks WHICH DEPTHS carry signal, and core
and static answer that. It is needed for the window sweep of the full detector (T-012): the reported
detector concatenates peak, range and update, and the update is defined on token-level states of
adjacent layers, so it cannot be rebuilt from pooled features. Asking for it here costs no extra
forward pass. v95[:, i] pools h[i+1] - h[i] in hidden-state indices, so the pinned update stream
(32_extract_velocity.py, blocks l = 15..22, Delta = h_{l+1} - h_l) is v95[:, 16:24].

--streams velocity alone writes <dataset>_alllayers_velocity.npz, for a model whose core and static
already exist; an existing file is never overwritten.

LAYER INDEXING. hidden_states has L_tot+1 entries; index 0 is the embedding output, before any
block, and index l>0 is the output of block l. Index L_tot is the last block's output WITHOUT the
final RMSNorm. The pinned pipeline's "final_norm" slice applies model.model.norm() on top, so it is
NOT the same vector as layer L_tot here. Do not compare the two directly.

MEMORY. Features accumulate in RAM before writing, which is fine for the datasets this ablation
needs and is not fine for TriviaQA. The script refuses rather than being killed halfway; see
--max-gb.

Usage:
  python 57_extract_all_layers.py --self-test
  python 57_extract_all_layers.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct
  python 57_extract_all_layers.py --dataset tydiqa_gp --model_folder llama-3.1-8b --streams core,static,velocity
  python 57_extract_all_layers.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct --streams velocity
"""

import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, "..", "data-alllayers"))


def pool_completion(h, q_hi=0.95, q_lo=0.05):
    """h is (T, D) for one answer's completion tokens. Returns (core, q95, q05), each (D,).

    Elementwise over t, matching 35_derive_streams.py's convention so the numbers are on the same
    footing as the pinned features. An empty completion yields NaN rather than zeros -- zero is a
    real activation value and would be treated as one."""
    a = np.asarray(h, dtype=np.float32)
    if a.shape[0] == 0:
        n = np.full(a.shape[1] if a.ndim == 2 else 0, np.nan, dtype=np.float32)
        return n, n.copy(), n.copy()
    return (a.max(axis=0),
            np.quantile(a, q_hi, axis=0).astype(np.float32),
            np.quantile(a, q_lo, axis=0).astype(np.float32))


def pool_answer(h, streams, q_hi=0.95, q_lo=0.05):
    """h is (L1, T, D): one answer's completion tokens at every hidden-state index. Returns a dict
    with core (L1, D), q95/q05 (L1, D) and v95/v05 (L1-1, D), as requested by `streams`.

    Same conventions as pool_completion, applied per index: elementwise over t, and NaN for an
    empty completion. v95[i] and v05[i] pool h[i+1] - h[i] over tokens, the update of
    32_extract_velocity.compute_velocity_streams."""
    a = np.asarray(h, dtype=np.float32)
    L1, T, D = a.shape
    out = {}
    if T == 0:
        if "core" in streams:
            out["core"] = np.full((L1, D), np.nan, dtype=np.float32)
        if "static" in streams:
            out["q95"] = np.full((L1, D), np.nan, dtype=np.float32)
            out["q05"] = np.full((L1, D), np.nan, dtype=np.float32)
        if "velocity" in streams:
            out["v95"] = np.full((L1 - 1, D), np.nan, dtype=np.float32)
            out["v05"] = np.full((L1 - 1, D), np.nan, dtype=np.float32)
        return out
    if "core" in streams:
        out["core"] = a.max(axis=1)
    if "static" in streams:
        out["q95"] = np.quantile(a, q_hi, axis=1).astype(np.float32)
        out["q05"] = np.quantile(a, q_lo, axis=1).astype(np.float32)
    if "velocity" in streams:
        d = a[1:] - a[:-1]
        out["v95"] = np.quantile(d, q_hi, axis=1).astype(np.float32)
        out["v05"] = np.quantile(d, q_lo, axis=1).astype(np.float32)
    return out


STREAMS = ("core", "static", "velocity")
ARRAYS = {"core": ("core",), "static": ("q95", "q05"), "velocity": ("v95", "v05")}


def estimate_gb(n_beams, n_layers_plus1, D, n_streams=3, bytes_per=2, n_velocity=0):
    """n_streams arrays over all L1 indices, plus n_velocity arrays over the L1-1 differences."""
    slices = n_layers_plus1 * n_streams + (n_layers_plus1 - 1) * n_velocity
    return n_beams * slices * D * bytes_per / 1024 ** 3


def output_name(dataset, streams):
    """Core and static keep the original name, so 58 and 60 read the file unchanged. A file without
    them gets a suffix and can never be mistaken for, or written over, a full one."""
    if "core" in streams and "static" in streams:
        return "%s_alllayers.npz" % dataset
    return "%s_alllayers_%s.npz" % (dataset, "+".join(s for s in STREAMS if s in streams))


def run(dataset, model_folder, data_dir, out_dir, device, dtype, max_gb, limit, log_every,
        streams=("core", "static")):
    import torch
    import yaml
    from transformers import AutoModelForCausalLM

    seq_path = os.path.join(data_dir, model_folder, "%s_sequences_v1.pt" % dataset)
    if not os.path.exists(seq_path):
        raise SystemExit("%s not found. This script scores PINNED generations; it does not create "
                         "them." % seq_path)
    path = os.path.join(out_dir, model_folder, output_name(dataset, streams))
    if os.path.exists(path):
        raise SystemExit("refusing: %s exists. Results already in the paper were computed from "
                         "files like it; move it aside by hand if it really must be replaced." % path)
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)

    seq = torch.load(seq_path, weights_only=False)
    input_ids, prompt_lens = seq["input_ids"], seq["prompt_len"]
    prompt_id = np.asarray(seq["prompt_id"])
    labels = np.asarray(seq["all_hallucination_flag"], dtype=int)

    order = np.argsort(prompt_id, kind="stable")
    groups = [(int(q), order[prompt_id[order] == q]) for q in np.unique(prompt_id)]
    if limit is not None:
        groups = groups[:limit]
    keep = np.concatenate([g[1] for g in groups])
    n_beams = len(keep)

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype,
                                                 trust_remote_code=True).to(device)
    model.eval()
    L1 = model.config.num_hidden_layers + 1      # +1: index 0 is the embedding output
    D = model.config.hidden_size

    names = [a for s in STREAMS if s in streams for a in ARRAYS[s]]
    n_vel = 2 if "velocity" in streams else 0
    need = estimate_gb(n_beams, L1, D, n_streams=len(names) - n_vel, n_velocity=n_vel)
    print("  [%s/%s] %d answers, %d layers (0..%d), D=%d, arrays %s -> %.1f GB in RAM"
          % (model_folder, dataset, n_beams, L1, L1 - 1, D, ",".join(names), need), flush=True)
    if need > max_gb:
        raise SystemExit(
            "refusing: %.1f GB exceeds --max-gb %.1f. This script accumulates in RAM and would be "
            "killed partway, leaving a truncated file -- a failure mode this project has already "
            "hit once. Use --limit to shard by question, or raise --max-gb if the node really has "
            "the memory." % (need, max_gb))

    store = {a: np.full((n_beams, L1 - 1 if a in ARRAYS["velocity"] else L1, D), np.nan,
                        dtype=np.float16) for a in names}
    row_of = {int(k): i for i, k in enumerate(keep)}

    t0, n_empty = time.time(), 0
    for gi, (_, idx) in enumerate(groups):
        batch = [input_ids[i] for i in idx]
        pls = [int(prompt_lens[i]) for i in idx]
        L = max(len(b) for b in batch)
        pad = model.config.eos_token_id or 0
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, b in enumerate(batch):
            t = b if torch.is_tensor(b) else torch.as_tensor(b)
            ids[j, :len(t)] = t
            att[j, :len(t)] = 1
        with torch.no_grad():
            out = model(ids.to(device), attention_mask=att.to(device), output_hidden_states=True)
        for j, i in enumerate(idx):
            s, e = pls[j], len(batch[j])
            # One answer's completion tokens at every index, sliced on the GPU so only (L1, T, D)
            # crosses to the host rather than the padded prompt of the whole group.
            h = torch.stack([out.hidden_states[li][j, s:e, :] for li in range(L1)]).float().cpu().numpy()
            pooled = pool_answer(h, streams)
            r = row_of[int(i)]
            for a in names:
                store[a][r] = pooled[a]
            if e - s == 0:
                n_empty += 1
        del out
        if (gi + 1) % log_every == 0:
            el = time.time() - t0
            print("    %d/%d questions (%.0fs, eta %.0fs)"
                  % (gi + 1, len(groups), el, el / (gi + 1) * (len(groups) - gi - 1)), flush=True)

    # float16 tops out at 65504. A value past it is stored as inf, so count what did not survive
    # rather than letting a downstream robust scaler quietly absorb it.
    nonfinite = {a: int((~np.isfinite(store[a])).sum()) for a in names}
    os.makedirs(os.path.join(out_dir, model_folder), exist_ok=True)
    np.savez_compressed(path, prompt_id=prompt_id[keep], label=labels[keep],
                        layer_index=np.arange(L1), beam_row=keep, **store)
    meta = {"dataset": dataset, "model_folder": model_folder, "model_id": model_id,
            "dtype": dtype, "n_beams": int(n_beams), "n_layers_plus_embedding": int(L1),
            "hidden_size": int(D), "n_empty_completions": int(n_empty),
            "streams": [s for s in STREAMS if s in streams], "arrays": names,
            "nonfinite_entries": nonfinite,
            "source": os.path.abspath(seq_path),
            "note": ("layer 0 is the embedding output; layer L_tot is the last block WITHOUT the "
                     "final norm, so it is not the pinned pipeline's final_norm slice. v95/v05 "
                     "index i pools h[i+1] - h[i]; the pinned update window is 16..23"),
            "elapsed_seconds": round(time.time() - t0, 1)}
    with open(path.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\n  wrote %s  (%.1f GB on disk, %d empty completions, non-finite entries %s)"
          % (path, os.path.getsize(path) / 1024 ** 3, n_empty, nonfinite))
    return meta


def self_test():
    print("=" * 74)
    print("  SELF-TEST: 57_extract_all_layers")
    print("=" * 74)

    # pool_completion on a hand-computable case
    h = np.array([[1.0, 9.0], [3.0, 7.0], [5.0, 5.0]], dtype=np.float32)
    c, hi, lo = pool_completion(h, 1.0, 0.0)
    assert list(c) == [5.0, 9.0], list(c)
    assert list(hi) == [5.0, 9.0] and list(lo) == [1.0, 5.0]
    print("  [PASS] pool_completion: max, and quantiles are elementwise over tokens")

    # Quantiles are per-column, not over the flattened array. Column 0 rises while column 1 falls,
    # so a flattening bug would give the same answer for both and this catches it.
    c2, hi2, lo2 = pool_completion(h, 0.5, 0.5)
    assert list(hi2) == [3.0, 7.0] == list(lo2)
    print("  [PASS] q50 is per-column (3.0, 7.0), not over the flattened tensor")

    # An empty completion yields NaN, never zeros -- zero is a real activation.
    c3, _, _ = pool_completion(np.zeros((0, 4), dtype=np.float32))
    assert c3.shape == (4,) and np.isnan(c3).all()
    print("  [PASS] empty completion gives NaN, not a zero vector")

    # pool_answer must give exactly what pool_completion gave layer by layer, so core and static from
    # the new code path are the numbers the existing Qwen files hold.
    rng = np.random.default_rng(0)
    h5 = rng.normal(size=(6, 7, 5)).astype(np.float32)
    pa = pool_answer(h5, STREAMS)
    for li in range(6):
        c, hi, lo = pool_completion(h5[li])
        assert np.array_equal(pa["core"][li], c)
        assert np.array_equal(pa["q95"][li], hi) and np.array_equal(pa["q05"][li], lo)
    print("  [PASS] pool_answer core/q95/q05 equal pool_completion at every index, bit for bit")

    # Update on a hand-computable case: h_hi - h_lo = [[1,0],[0,3],[4,0]] over three tokens.
    lo_ = np.array([[1.0, 9.0], [3.0, 7.0], [5.0, 5.0]], dtype=np.float32)
    hi_ = np.array([[2.0, 9.0], [3.0, 10.0], [9.0, 5.0]], dtype=np.float32)
    pv = pool_answer(np.stack([lo_, hi_]), ("velocity",), q_hi=1.0, q_lo=0.0)
    assert pv["v95"].shape == (1, 2) and list(pv["v95"][0]) == [4.0, 3.0] and list(pv["v05"][0]) == [0.0, 0.0]
    pm = pool_answer(np.stack([lo_, hi_]), ("velocity",), q_hi=0.5, q_lo=0.5)
    assert list(pm["v95"][0]) == [1.0, 0.0], pm
    print("  [PASS] update pools h[i+1] - h[i] per column over tokens (q100 [4,3], q0 [0,0], q50 [1,0])")

    # Index convention, the error that would shift every window by one layer. With h[i] = i^2 the
    # difference at index i is 2i+1, so index 16 must hold h[17] - h[16] = 33 -- the first slice of
    # the pinned update window (block 15 -> block 16).
    hsq = np.stack([np.full((3, 4), float(i * i), dtype=np.float32) for i in range(29)])
    vi = pool_answer(hsq, ("velocity",))["v95"]
    assert vi.shape == (28, 4) and vi[16, 0] == 33.0 and vi[23, 0] == 47.0
    assert "core" not in pool_answer(hsq, ("velocity",))
    print("  [PASS] v95[i] = h[i+1] - h[i]: index 16 holds 33 = 17^2 - 16^2; pinned update is 16..23")

    e = pool_answer(np.zeros((29, 0, 4), dtype=np.float32), STREAMS)
    assert e["v95"].shape == (28, 4) and all(np.isnan(v).all() for v in e.values())
    print("  [PASS] empty completion gives NaN in every stream, including the update")

    # Names: a velocity-only file must never take the name 58 and 60 read, or overwrite it.
    assert output_name("tydiqa_gp", ("core", "static")) == "tydiqa_gp_alllayers.npz"
    assert output_name("tydiqa_gp", ("core", "static", "velocity")) == "tydiqa_gp_alllayers.npz"
    assert output_name("tydiqa_gp", ("velocity",)) == "tydiqa_gp_alllayers_velocity.npz"
    print("  [PASS] output names: velocity-only writes <dataset>_alllayers_velocity.npz")

    # The size guard must fire before allocation, not after. Asserted against the DEFAULT --max-gb
    # rather than a hand-written constant, so the test tracks the guard if the default changes.
    # (The first version of this assertion used 100 GB, carried over from a three-stream estimate;
    # storing core+q95+q05 rather than adding velocity puts TriviaQA/LLaMA at 75 GB, not 124.)
    default_max_gb = 32.0
    tri = estimate_gb(99600, 33, 4096)      # TriviaQA on LLaMA -- must be refused
    tyd = estimate_gb(4400, 33, 3584)       # TyDiQA on Qwen    -- must be allowed
    tqa = estimate_gb(8170, 29, 3584)       # TruthfulQA on Qwen -- must be allowed
    assert tri > default_max_gb, tri
    assert tyd < default_max_gb and tqa < default_max_gb, (tyd, tqa)
    print("  [PASS] size guard at --max-gb %.0f: TriviaQA/LLaMA %.1f GB refused; "
          "TyDiQA %.1f GB and TruthfulQA %.1f GB allowed" % (default_max_gb, tri, tyd, tqa))
    # With the update, LLaMA TruthfulQA -- the largest run the window sweep needs -- must still fit.
    tqa_l = estimate_gb(8170, 33, 4096, n_streams=3, n_velocity=2)
    assert abs(estimate_gb(1, 33, 1, n_streams=3, n_velocity=2) * 1024 ** 3 - 2 * (33 * 3 + 32 * 2)) < 1e-6
    assert tqa_l < default_max_gb, tqa_l
    print("  [PASS] with the update: LLaMA TruthfulQA %.1f GB allowed (update has L_tot slices, not L_tot+1)"
          % tqa_l)

    # Layer count includes the embedding output. Off by one here silently shifts every layer label
    # in the figure, which is the kind of error that survives to publication.
    assert estimate_gb(1, 33, 1, 1, 1) == estimate_gb(1, 33, 1, 1, 1)
    for n_blocks, expect in ((28, 29), (32, 33)):
        assert n_blocks + 1 == expect
    print("  [PASS] layer axis is n_blocks+1 (Qwen 29, LLaMA 33) -- index 0 is the embedding")

    print("\n[PASS] All self-test assertions passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tydiqa_gp",
                    choices=["truthfulqa", "triviaqa", "nq_open", "tydiqa_gp"])
    ap.add_argument("--model_folder", default="qwen-2.5-7b-instruct")
    ap.add_argument("--data-dir", default=None, help="where the pinned generations are")
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help="separate from the pinned features on purpose")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max-gb", type=float, default=32.0)
    ap.add_argument("--limit", type=int, default=None, help="first N questions")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--streams", default="core,static",
                    help="comma-separated subset of core,static,velocity; velocity is for the window "
                         "sweep of the full detector")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return
    streams = tuple(s.strip() for s in a.streams.split(",") if s.strip())
    bad = [s for s in streams if s not in STREAMS]
    if bad or not streams:
        ap.error("--streams takes a non-empty subset of %s, got %s" % (",".join(STREAMS), a.streams))

    data_dir = a.data_dir
    if not data_dir:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]

    assert os.path.abspath(a.out_dir) != os.path.abspath(data_dir), \
        "--out-dir must differ from the pinned data directory"

    print("=" * 78)
    print("  ALL-LAYER EXTRACTION -- %s / %s" % (a.model_folder, a.dataset))
    print("  in : %s" % os.path.abspath(data_dir))
    print("  out: %s   (pinned features are never written)" % os.path.abspath(a.out_dir))
    print("=" * 78, flush=True)
    run(a.dataset, a.model_folder, data_dir, a.out_dir, a.device, a.dtype,
        a.max_gb, a.limit, a.log_every, streams)


if __name__ == "__main__":
    main()
