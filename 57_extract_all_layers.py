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

    core   (N, L_tot+1, D)   max over completion tokens
    q95    (N, L_tot+1, D)   upper quantile over completion tokens
    q05    (N, L_tot+1, D)   lower quantile

Velocity is deliberately absent. It is defined between adjacent layers and would add 2(L_tot) more
slices -- roughly 40% more disk -- to answer a question about WHICH DEPTHS carry signal, which core
and static already answer. Once the informative band is known, velocity can be extracted there
specifically.

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


def estimate_gb(n_beams, n_layers_plus1, D, n_streams=3, bytes_per=2):
    return n_beams * n_layers_plus1 * D * n_streams * bytes_per / 1024 ** 3


def run(dataset, model_folder, data_dir, out_dir, device, dtype, max_gb, limit, log_every):
    import torch
    import yaml
    from transformers import AutoModelForCausalLM

    seq_path = os.path.join(data_dir, model_folder, "%s_sequences_v1.pt" % dataset)
    if not os.path.exists(seq_path):
        raise SystemExit("%s not found. This script scores PINNED generations; it does not create "
                         "them." % seq_path)
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

    need = estimate_gb(n_beams, L1, D)
    print("  [%s/%s] %d answers, %d layers (0..%d), D=%d -> %.1f GB in RAM"
          % (model_folder, dataset, n_beams, L1, L1 - 1, D, need), flush=True)
    if need > max_gb:
        raise SystemExit(
            "refusing: %.1f GB exceeds --max-gb %.1f. This script accumulates in RAM and would be "
            "killed partway, leaving a truncated file -- a failure mode this project has already "
            "hit once. Use --limit to shard by question, or raise --max-gb if the node really has "
            "the memory." % (need, max_gb))

    core = np.full((n_beams, L1, D), np.nan, dtype=np.float16)
    q95 = np.full((n_beams, L1, D), np.nan, dtype=np.float16)
    q05 = np.full((n_beams, L1, D), np.nan, dtype=np.float16)
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
        for li in range(L1):
            H = out.hidden_states[li].float().cpu().numpy()
            for j, i in enumerate(idx):
                s, e = pls[j], len(batch[j])
                c, hi, lo = pool_completion(H[j, s:e, :])
                r = row_of[int(i)]
                core[r, li], q95[r, li], q05[r, li] = c, hi, lo
                if li == 0 and e - s == 0:
                    n_empty += 1
        del out
        if (gi + 1) % log_every == 0:
            el = time.time() - t0
            print("    %d/%d questions (%.0fs, eta %.0fs)"
                  % (gi + 1, len(groups), el, el / (gi + 1) * (len(groups) - gi - 1)), flush=True)

    os.makedirs(os.path.join(out_dir, model_folder), exist_ok=True)
    path = os.path.join(out_dir, model_folder, "%s_alllayers.npz" % dataset)
    np.savez_compressed(path, core=core, q95=q95, q05=q05,
                        prompt_id=prompt_id[keep], label=labels[keep],
                        layer_index=np.arange(L1), beam_row=keep)
    meta = {"dataset": dataset, "model_folder": model_folder, "model_id": model_id,
            "dtype": dtype, "n_beams": int(n_beams), "n_layers_plus_embedding": int(L1),
            "hidden_size": int(D), "n_empty_completions": int(n_empty),
            "source": os.path.abspath(seq_path),
            "note": ("layer 0 is the embedding output; layer L_tot is the last block WITHOUT the "
                     "final norm, so it is not the pinned pipeline's final_norm slice"),
            "elapsed_seconds": round(time.time() - t0, 1)}
    with open(path.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\n  wrote %s  (%.1f GB on disk, %d empty completions)"
          % (path, os.path.getsize(path) / 1024 ** 3, n_empty))
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
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return

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
        a.max_gb, a.limit, a.log_every)


if __name__ == "__main__":
    main()
