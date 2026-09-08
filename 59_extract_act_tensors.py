"""
59_extract_act_tensors.py -- pooled Activation Tensors in ACT-ViT's format, from our generations.
=====================================================================================================
WHY THIS EXISTS. ACT-ViT (Bar-Shalom et al., NeurIPS 2025) consumes an Activation Tensor per
response: A in R^(L x N x D), every layer, every generated token, full hidden width. To put their
method in our results table it has to run on OUR pinned generations under OUR protocol -- the same
treatment HARP got via 49_harp_adapter.py. This script produces the input side of that.

THE BLOCKER IS STORAGE, NOT COMPUTE. Unpooled, Qwen/TriviaQA is 28 x 64 x 3584 x 2 bytes = 12.9 MB
per beam, and 99,600 beams is ~1.3 TB. Their own answer is to pool immediately, so we pool INSIDE
the forward loop and never hold a full-resolution tensor. At (L_p, N_p) = (8, 100) a beam is 5.7 MB:
TyDiQA 25 GB, TruthfulQA 47 GB, TriviaQA 571 GB. The first two are fine; TriviaQA needs a harder
pool and is refused by --max-gb until someone decides that deliberately.

POOLING IS THEIRS, NOT OURS. `act_pool` reimplements their Algorithm 1 exactly:
    permute to [D, L, N] -> pad so L, N are divisible -> max_pool2d(kernel=(ceil(L/L_p),
    ceil(N/N_p))) -> permute back to [L_p, N_p, D]
Two details in that algorithm are easy to get wrong and are pinned by the self-test:
  * the kernel is computed from the PADDED length, and stride equals kernel (non-overlapping), so
    the output is exactly (L_p, N_p) for any input size;
  * padding must not invent maxima. F.max_pool2d pads with -inf, but we pad by EDGE REPLICATION
    before pooling, because a -inf column that survives into an output cell would produce -inf
    features for short answers -- and our answers are short (<= 64 tokens) relative to N_p = 100.

WHAT THIS DOES NOT DO. It does not train ACT-ViT. Their Linear Adapter (D -> 128) and ViT backbone
are supervised and must be fit inside a split, so they belong in the harness as methods/act_vit.py,
which consumes what this writes. Keeping extraction separate is what lets one 6-hour forward pass
serve five seeds x two protocols.

LAYER INDEXING. Index 0 is the embedding output, before any block; index l > 0 is block l's output;
the last index is the final block WITHOUT the final norm. Same convention as 57, and NOT the pinned
pipeline's final_norm slice. ACT-ViT uses all layers, so no window is applied here.

Usage:
  python 59_extract_act_tensors.py --self-test
  python 59_extract_act_tensors.py --dataset tydiqa_gp  --model_folder qwen-2.5-7b-instruct
  python 59_extract_act_tensors.py --dataset truthfulqa --model_folder qwen-2.5-7b-instruct
  python 59_extract_act_tensors.py --dataset triviaqa   --model_folder qwen-2.5-7b-instruct \
      --n-pool 16 --max-gb 100          # TriviaQA only lands under a harder pool
"""

import argparse
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, "..", "data-acttensors"))
DEFAULT_DATA = os.path.abspath(os.path.join(HERE, "..", "data"))

L_POOL = 8               # ACT-ViT's default L_p
N_POOL = 100             # ACT-ViT's default N_p
DEFAULT_MAX_GB = 60.0


def act_pool(A, l_pool=L_POOL, n_pool=N_POOL):
    """ACT-ViT Algorithm 1. A is (L, N, D) -> returns (l_pool, n_pool, D).

    Pure numpy so the self-test runs without torch. Edge-replication padding, then non-overlapping
    max over blocks. See the module docstring for why the padding is replication and not -inf.
    """
    A = np.asarray(A, dtype=np.float32)
    L, N, D = A.shape
    if L == 0 or N == 0 or D == 0:
        raise ValueError("act_pool got an empty tensor with shape %r" % (A.shape,))

    f_l, f_n = int(math.ceil(L / l_pool)), int(math.ceil(N / n_pool))
    L_pad, N_pad = f_l * l_pool, f_n * n_pool
    if L_pad != L or N_pad != N:
        A = np.pad(A, ((0, L_pad - L), (0, N_pad - N), (0, 0)), mode="edge")

    # (l_pool, f_l, n_pool, f_n, D) -> max over the two block axes.
    return A.reshape(l_pool, f_l, n_pool, f_n, D).max(axis=(1, 3))


def estimate_gb(n_beams, l_pool, n_pool, D, bytes_per=2):
    return n_beams * l_pool * n_pool * D * bytes_per / 1024 ** 3


def run(dataset, model_folder, data_dir, out_dir, device, dtype, l_pool, n_pool, max_gb, limit,
        log_every):
    import torch
    import yaml
    from transformers import AutoModelForCausalLM

    out_dir = os.path.abspath(out_dir)
    if out_dir == os.path.abspath(data_dir):
        raise SystemExit("refusing to write into the pinned data directory %s. These are a new "
                         "artifact in a new location, on purpose." % data_dir)

    seq_path = os.path.join(data_dir, model_folder, "%s_sequences_v1.pt" % dataset)
    if not os.path.exists(seq_path):
        raise SystemExit("%s not found. This script reads PINNED generations; it does not create "
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
    L1 = model.config.num_hidden_layers + 1
    D = model.config.hidden_size

    need = estimate_gb(n_beams, l_pool, n_pool, D)
    print("  [%s/%s] %d answers, %d layers (0..%d), D=%d, pool (%d,%d) -> %.1f GB"
          % (model_folder, dataset, n_beams, L1, L1 - 1, D, l_pool, n_pool, need), flush=True)

    # Our completions cap at 64 new tokens, so ACT-ViT's default N_p = 100 pads rather than pools.
    # Not silently corrected: replicated columns become extra ViT patches with distinct positional
    # encodings, so changing N_p changes THEIR model's input, and that is their call to make, not a
    # storage optimisation to slip in. Reported so the choice is deliberate.
    max_new = int(max(len(b) - int(p_) for b, p_ in zip(input_ids, prompt_lens)))
    if n_pool > max_new:
        print("  NOTE: --n-pool %d exceeds the longest completion (%d tokens). Every pooled cell "
              "beyond token %d is edge replication, so the token axis is expanded, not compressed; "
              "only the layer axis (%d -> %d) actually shrinks. Faithful to their default, but "
              "--n-pool %d would store the same information in %.1f GB."
              % (n_pool, max_new, max_new, L1, l_pool, max_new,
                 estimate_gb(n_beams, l_pool, max_new, D)), flush=True)
    if need > max_gb:
        raise SystemExit(
            "refusing: %.1f GB exceeds --max-gb %.1f. Unpooled this dataset would be %.0f GB, which "
            "is why ACT-ViT pools first. Lower --n-pool (their ablation runs (L_p,N_p) down to "
            "(4,20) and still beats the best probe), use --limit to shard, or raise --max-gb if "
            "that is a deliberate decision about disk."
            % (need, max_gb, estimate_gb(n_beams, L1, 64, D)))

    at = np.full((n_beams, l_pool, n_pool, D), np.nan, dtype=np.float16)
    row_of = {int(k): i for i, k in enumerate(keep)}
    tok_len = np.zeros(n_beams, dtype=np.int32)

    t0, n_empty = time.time(), 0
    for gi, (_, idx) in enumerate(groups):
        batch = [input_ids[i] for i in idx]
        pls = [int(prompt_lens[i]) for i in idx]
        Lmax = max(len(b) for b in batch)
        pad = model.config.eos_token_id or 0
        ids = torch.full((len(batch), Lmax), pad, dtype=torch.long)
        att = torch.zeros((len(batch), Lmax), dtype=torch.long)
        for j, b in enumerate(batch):
            t = b if torch.is_tensor(b) else torch.as_tensor(b)
            ids[j, :len(t)] = t
            att[j, :len(t)] = 1
        with torch.no_grad():
            out = model(ids.to(device), attention_mask=att.to(device), output_hidden_states=True)

        # Stack layers for ONE beam at a time. Stacking the whole batch first is what makes the
        # unpooled tensor appear in memory, which is the thing this script exists to avoid.
        for j, i in enumerate(idx):
            s, e = pls[j], len(batch[j])
            r = row_of[int(i)]
            tok_len[r] = e - s
            if e - s == 0:
                n_empty += 1
                continue
            layers = [out.hidden_states[li][j, s:e, :].float().cpu().numpy() for li in range(L1)]
            at[r] = act_pool(np.stack(layers, axis=0), l_pool, n_pool).astype(np.float16)
        del out

        if (gi + 1) % log_every == 0:
            el = time.time() - t0
            print("    %d/%d questions (%.0fs, eta %.0fs)"
                  % (gi + 1, len(groups), el, el / (gi + 1) * (len(groups) - gi - 1)), flush=True)

    os.makedirs(os.path.join(out_dir, model_folder), exist_ok=True)
    path = os.path.join(out_dir, model_folder,
                        "%s_at_L%d_N%d.npz" % (dataset, l_pool, n_pool))
    np.savez(path, at=at, prompt_id=prompt_id[keep], label=labels[keep],
             beam_row=keep, token_len=tok_len)
    meta = {"dataset": dataset, "model_folder": model_folder, "model_id": model_id,
            "dtype": dtype, "n_beams": int(n_beams), "n_layers_plus_embedding": int(L1),
            "hidden_size": int(D), "l_pool": int(l_pool), "n_pool": int(n_pool),
            "n_empty_completions": int(n_empty),
            "median_completion_tokens": int(np.median(tok_len)),
            "max_completion_tokens": int(tok_len.max()),
            "source": os.path.abspath(seq_path),
            "pooling": ("ACT-ViT Algorithm 1, edge-replication padding then non-overlapping max; "
                        "layer 0 is the embedding output, last layer has no final norm"),
            "elapsed_seconds": round(time.time() - t0, 1)}
    with open(path.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\n  wrote %s  (%.1f GB on disk, %d empty completions)"
          % (path, os.path.getsize(path) / 1024 ** 3, n_empty))
    return meta


def self_test():
    print("=" * 74)
    print("  SELF-TEST: 59_extract_act_tensors")
    print("=" * 74)

    # Shape is exact for any input size, which is the whole point of their padding step.
    for L, N in [(28, 64), (33, 7), (8, 100), (1, 1), (29, 250)]:
        o = act_pool(np.random.randn(L, N, 3), 8, 100)
        assert o.shape == (8, 100, 3), (L, N, o.shape)
    print("  [PASS] act_pool returns exactly (8, 100, D) for L in {1,8,28,29,33}, N in {1,7,64,250}")

    # Max, not mean, and taken over the right block. With L=4 -> l_pool=2 the blocks are rows
    # {0,1} and {2,3}; a mean would give 0.5 and 2.5 instead of 1 and 3.
    A = np.arange(4, dtype=np.float32).reshape(4, 1, 1)
    o = act_pool(A, 2, 1)
    assert o.ravel().tolist() == [1.0, 3.0], o.ravel().tolist()
    print("  [PASS] pooling is max over non-overlapping blocks (1, 3), not mean (0.5, 2.5)")

    # Padding must not invent values. Ten identical rows pooled to 8 must stay at that value:
    # -inf padding would leave -inf, and zero padding would cap a negative tensor at 0.
    neg = np.full((10, 3, 2), -5.0, dtype=np.float32)
    o = act_pool(neg, 8, 100)
    assert np.all(o == -5.0), "padding leaked a value that is not in the input"
    assert np.isfinite(o).all()
    print("  [PASS] edge padding on an all-negative tensor stays at -5.0 (no -inf, no zero cap)")

    # The two axes must not be transposed. Column 1 is large and row 3 is large; a swapped
    # permute would put the large value in the wrong output cell.
    B = np.zeros((4, 4, 1), dtype=np.float32)
    B[3, :, 0] = 7.0        # last LAYER is hot
    B[:, 1, 0] = 2.0
    o = act_pool(B, 2, 2)[:, :, 0]
    assert o[1, 0] == 7.0 and o[0, 0] == 2.0, o.tolist()
    print("  [PASS] layer axis and token axis are not transposed (hot layer lands in row 1)")

    # A single token replicates across the whole token axis rather than erroring, so short
    # answers are representable. Our answers are <= 64 tokens against N_p = 100.
    one = np.arange(6, dtype=np.float32).reshape(3, 1, 2)
    o = act_pool(one, 3, 4)
    assert o.shape == (3, 4, 2) and np.allclose(o[:, 0, :], o[:, 3, :])
    print("  [PASS] a 1-token completion replicates across N_p rather than failing")

    # Size guard. TyDiQA and TruthfulQA must pass at the default pool; TriviaQA must be refused,
    # and must become feasible once n_pool drops -- which is the documented escape hatch.
    tyd = estimate_gb(4400, 8, 100, 3584)
    tqa = estimate_gb(8170, 8, 100, 3584)
    tri = estimate_gb(99600, 8, 100, 3584)
    tri16 = estimate_gb(99600, 8, 16, 3584)
    assert tyd < DEFAULT_MAX_GB and tqa < DEFAULT_MAX_GB, (tyd, tqa)
    assert tri > DEFAULT_MAX_GB, tri
    assert tri16 < tri / 5.0, (tri16, tri)
    print("  [PASS] sizes: TyDiQA %.0f GB, TruthfulQA %.0f GB pass; TriviaQA %.0f GB refused, "
          "%.0f GB at N_p=16" % (tyd, tqa, tri, tri16))

    # Pooling at their default buys much less than it looks like it should, and the reason is
    # worth pinning: N_p = 100 is LARGER than our longest completion (64 new tokens), so the token
    # axis is replicated rather than compressed and only the layer axis (29 -> 8) actually shrinks.
    # The saving is ~2.3x, not the ~20x an unexamined reading of "pooling" would suggest.
    raw = estimate_gb(99600, 29, 64, 3584)
    assert 2.0 < raw / tri < 3.0, (raw, tri)
    assert estimate_gb(1, 8, 100, 3584) > estimate_gb(1, 29, 64, 3584) / 3, "layer axis dominates"
    print("  [PASS] unpooled TriviaQA is %.0f GB, only %.1fx the pooled size -- at N_p=100 the "
          "token axis is replicated, not compressed" % (raw, raw / tri))

    # N_p at or below the real completion length is where compression actually happens. This is
    # the knob to turn for TriviaQA, and their own ablation runs (L_p, N_p) down to (4, 20).
    assert estimate_gb(99600, 8, 32, 3584) < tri / 3, "N_p=32 must be a real reduction"
    print("  [PASS] N_p=32 cuts TriviaQA to %.0f GB; N_p is the knob, L_p is already at their "
          "default" % estimate_gb(99600, 8, 32, 3584))

    print("\n  ALL PASS")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dataset")
    p.add_argument("--model_folder")
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--l-pool", type=int, default=L_POOL)
    p.add_argument("--n-pool", type=int, default=N_POOL)
    p.add_argument("--max-gb", type=float, default=DEFAULT_MAX_GB)
    p.add_argument("--limit", type=int, default=None, help="first N questions only")
    p.add_argument("--log-every", type=int, default=50)
    a = p.parse_args()

    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not a.dataset or not a.model_folder:
        raise SystemExit("--dataset and --model_folder are required (or use --self-test)")
    run(a.dataset, a.model_folder, a.data_dir, a.out_dir, a.device, a.dtype,
        a.l_pool, a.n_pool, a.max_gb, a.limit, a.log_every)


if __name__ == "__main__":
    main()
