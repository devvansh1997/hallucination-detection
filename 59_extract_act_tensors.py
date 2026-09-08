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
TyDiQA 23 GB, TruthfulQA 44 GB, TriviaQA 532 GB. The first two are fine; TriviaQA needs a harder
pool and is refused by --max-gb until someone decides that deliberately.

POOLING IS THEIRS, NOT OURS, AND IT IS NOT WHAT THE PAPER'S ALGORITHM 1 SUGGESTS. The released
preprocessing (utils/dataset_preprocess.py, process_file lines 210-221) is TWO padding steps:

    L_for_pad = (int(L / L_eff) + 1) * L_eff      # 29 -> 32; adds a full block even if divisible
    N_for_pad = N_MAX if N_MAX % N_eff == 0 else (int(N_MAX/N_eff)+1)*N_eff
    pad_activations_tensor(..., pad_value=0)      # ZERO pad, and TRUNCATE anything past N_MAX
    patch_down_sample(..., method='max_pool')     # replicate pad to divisibility, then max

The replicate pad in the second step never fires for any configuration we run, because the zero pad
has already made both axes divisible. So the padding that reaches the data is ZEROS. An earlier
version of this file read Algorithm 1 from the paper and used edge replication throughout, which
was wrong twice over:

  * tokens. Our median completion is 16-17 against N_MAX = 100, so 84 of the 100 columns are zero.
    Edge replication filled them with copies of the last real token instead -- a different tensor,
    and a different ViT input.
  * layers. L = 29 zero-pads to 32 and factor_L = 4, so the last of the eight output slots is
    max(layer 28, 0, 0, 0), an elementwise ReLU on the final layer. Edge replication gave layer 28
    unchanged.

The self-test asserts bit-identity against their own pad_activations_tensor and patch_down_sample,
imported from ../ACT-ViT, over 42 combinations of (L, T, N_eff). Nothing here relies on a reading
of their paper.

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
      --n-pool 20 --max-gb 150          # TriviaQA only lands under a harder pool
  python 59_extract_act_tensors.py --repool-from ../data-acttensors/qwen-2.5-7b-instruct/\
      tydiqa_gp_at_L8_N100.npz --n-pool 20        # CPU only, no model load
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
N_POOL = 100             # ACT-ViT's default N_p (their args.N_eff)
N_MAX = 100              # their utils/constants.py N_MAX -- the zero-pad target
DEFAULT_MAX_GB = 60.0


def act_pool(A, l_pool=L_POOL, n_pool=N_POOL, n_max=N_MAX):
    """ACT-ViT preprocessing. A is (L, N, D) -> returns (l_pool, n_pool, D).

    Faithful to utils/dataset_preprocess.py in their repo, which is TWO padding steps, not one.
    `process_file` lines 210-221 do:

        L_for_pad = (int(L / L_eff) + 1) * L_eff          # 29 -> 32; always adds >= 1 full block
        N_for_pad = N_MAX if N_MAX % N_eff == 0 else (int(N_MAX/N_eff)+1)*N_eff
        pad_activations_tensor(..., pad_value=0)          # ZERO pad, and TRUNCATE if longer
        patch_down_sample(..., method='max_pool')         # replicate pad to divisibility, then max

    The replicate pad inside `patch_down_sample` is a no-op for every configuration we run, because
    the zero pad already made both axes divisible. So the padding that actually reaches the data is
    ZEROS. Getting this backwards matters twice over:

      * tokens. Our median completion is 16-17 against N_MAX = 100, so 84 of the 100 columns are
        zero. Edge replication would instead fill them with copies of the last real token, which is
        a different tensor and a different ViT input.
      * layers. L = 29 zero-pads to 32, factor_L = 4, so the last of the eight output layer slots is
        max(layer 28, 0, 0, 0) -- an elementwise ReLU on the final layer. Under edge replication it
        would be layer 28 unchanged.

    The self-test asserts equality against their two functions imported from the cloned repo, so
    this docstring is not the thing being trusted.
    """
    A = np.asarray(A, dtype=np.float32)
    L, N, D = A.shape
    if L == 0 or N == 0 or D == 0:
        raise ValueError("act_pool got an empty tensor with shape %r" % (A.shape,))

    l_for_pad = (int(L / l_pool) + 1) * l_pool
    n_for_pad = n_max if (n_max % n_pool) == 0 else (int(n_max / n_pool) + 1) * n_pool

    # Zero pad, truncating any axis that is already longer (their pad_activations_tensor takes
    # min(N, N_max), so a completion beyond N_MAX is CUT, not pooled).
    P = np.zeros((l_for_pad, n_for_pad, D), dtype=np.float32)
    P[:min(L, l_for_pad), :min(N, n_for_pad), :] = A[:min(L, l_for_pad), :min(N, n_for_pad), :]

    # patch_down_sample: replicate to divisibility (a no-op once the zero pad has run), then
    # non-overlapping max over blocks.
    l_pad = (l_pool - (l_for_pad % l_pool)) % l_pool + l_for_pad
    n_pad = (n_pool - (n_for_pad % n_pool)) % n_pool + n_for_pad
    if l_pad != l_for_pad or n_pad != n_for_pad:
        P = np.pad(P, ((0, l_pad - l_for_pad), (0, n_pad - n_for_pad), (0, 0)), mode="edge")

    f_l, f_n = l_pad // l_pool, n_pad // n_pool
    return P.reshape(l_pool, f_l, n_pool, f_n, D).max(axis=(1, 3))


def estimate_gb(n_beams, l_pool, n_pool, D, bytes_per=2):
    return n_beams * l_pool * n_pool * D * bytes_per / 1024 ** 3


def repool(in_path, n_pool, out_path=None, n_max=N_MAX):
    """Re-derive a smaller N_eff from an existing extraction, with no forward pass.

    NOT act_pool applied twice. Their L_for_pad = (int(L/L_eff) + 1) * L_eff adds a full block even
    when L is already divisible, so act_pool is not idempotent on the layer axis: run it again on an
    8-layer tensor and it zero-pads to 16 and collapses to 4 real slots plus 4 of zeros. This
    function therefore touches the TOKEN axis only and leaves the layer axis exactly as extracted.

    Valid only when the source did no token pooling, i.e. its factor_N was 1, which holds when the
    source N_eff equals N_MAX. Then its first token_len columns are the real per-token states and
    the remainder is their zero pad, so re-blocking is exact: the max over a 5-wide block of
    singleton maxima is the max over the 5-wide block. Checked below, not assumed.

    Why bother. Our median completion is 16-17 tokens against N_MAX = 100, so at N_eff = 100 the
    ViT receives 800 activation pixels of which roughly 670 are zeros. N_eff = 20 sits inside their
    own published ablation grid (Figure 3 sweeps (L_p, N_p) over {4,8} x {20,100}), so it is their
    hyperparameter chosen for our input lengths rather than a deviation from their method.
    """
    z = np.load(in_path)
    at, tok_len = z["at"], z["token_len"]
    n, l_pool, n_src, D = at.shape
    if n_src != n_max:
        raise SystemExit(
            "repool needs a source extracted at N_eff = N_MAX = %d, so that factor_N was 1 and its "
            "columns are per-token. This file has N_eff=%d, whose columns are already maxima over "
            "blocks. Re-extract instead." % (n_max, n_src))
    if n_pool > n_src:
        raise SystemExit("cannot repool upward: source has N_eff=%d, asked for %d" % (n_src, n_pool))

    n_for_pad = n_max if (n_max % n_pool) == 0 else (int(n_max / n_pool) + 1) * n_pool
    f_n = n_for_pad // n_pool

    out = np.empty((n, l_pool, n_pool, D), dtype=at.dtype)
    for i in range(n):
        t = min(int(tok_len[i]), n_src)
        P = np.zeros((l_pool, n_for_pad, D), dtype=np.float32)
        P[:, :t, :] = at[i, :, :t, :]        # real columns; the rest stays zero, as theirs does
        out[i] = P.reshape(l_pool, n_pool, f_n, D).max(axis=2).astype(at.dtype)

    if out_path is None:
        out_path = in_path.replace("_N%d.npz" % n_src, "_N%d.npz" % n_pool)
    np.savez(out_path, at=out, prompt_id=z["prompt_id"], label=z["label"],
             beam_row=z["beam_row"], token_len=tok_len)
    print("  repooled %s (N_eff %d -> %d): %.1f GB -> %.1f GB"
          % (os.path.basename(in_path), n_src, n_pool,
             os.path.getsize(in_path) / 1024 ** 3, os.path.getsize(out_path) / 1024 ** 3))
    return out_path


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
            "pooling": ("ACT-ViT utils/dataset_preprocess.py process_file: zero-pad to "
                        "(L_for_pad, N_for_pad) then non-overlapping max. L_for_pad = "
                        "(int(L/L_eff)+1)*L_eff adds a full block even when divisible, so at L=29 "
                        "the last of the 8 layer slots is max(layer28, 0) = relu(layer28). The "
                        "replicate pad inside patch_down_sample never fires at these settings."),
            "pooling_verified": ("act_pool asserted bit-identical to their pad_activations_tensor "
                                 "+ patch_down_sample over 42 (L, T, N_eff) combinations; run "
                                 "59_extract_act_tensors.py --self-test with ../ACT-ViT present"),
            "layer_indexing": ("layer 0 is the embedding output; the last is the final block "
                               "WITHOUT the final norm, so not the pinned pipeline's final_norm "
                               "slice"),
            "n_max": int(N_MAX),
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

    # DIFFERENTIAL TEST against ACT-ViT's own code. Their preprocessing is two padding steps and
    # a quirky L_for_pad that always adds a full block; reimplementing it from the paper is how the
    # first version of this file got both padding value and layer grouping wrong. If the repo is
    # present we assert exact equality rather than trusting any reading of it.
    ref = os.path.abspath(os.path.join(HERE, "..", "ACT-ViT"))
    if os.path.isdir(ref):
        import sys
        import torch
        sys.path.insert(0, ref)
        from utils.dataset_preprocess import pad_activations_tensor, patch_down_sample

        rng = np.random.default_rng(0)
        checked = 0
        for L in (29, 33, 8):
            for T in (1, 3, 16, 17, 64, 100, 137):
                for n_eff in (100, 20):
                    A = rng.standard_normal((L, T, 3)).astype(np.float32)
                    l_for_pad = (int(L / 8) + 1) * 8
                    n_for_pad = (N_MAX if (N_MAX % n_eff) == 0
                                 else (int(N_MAX / n_eff) + 1) * n_eff)
                    theirs = patch_down_sample(
                        pad_activations_tensor(torch.from_numpy(A), l_for_pad, n_for_pad, 0),
                        L_new=8, N_new=n_eff, method="max_pool").numpy()
                    mine = act_pool(A, 8, n_eff)
                    assert mine.shape == theirs.shape, (L, T, n_eff, mine.shape, theirs.shape)
                    assert np.allclose(mine, theirs, atol=0, rtol=0),                         "act_pool disagrees with ACT-ViT at L=%d T=%d N_eff=%d (max diff %.3e)" % (
                            L, T, n_eff, np.abs(mine - theirs).max())
                    checked += 1
        sys.path.remove(ref)
        print("    [PASS] act_pool is bit-identical to ACT-ViT's own pad+patch_down_sample "
              "on %d configurations (L, T, N_eff), including T > N_MAX" % checked)
    else:
        print("    [SKIP] ../ACT-ViT not cloned -- differential test against their code not run")

    # The two behaviours the differential test exists to pin, stated so they survive the repo
    # going missing. Zeros in the pad, and a ReLU on the final layer slot.
    A = np.full((29, 16, 2), -3.0, dtype=np.float32)
    o = act_pool(A, 8, 100)
    assert np.all(o[:, 16:, :] == 0.0), "token pad must be ZERO, not an edge replica"
    assert np.all(o[7, :16, :] == 0.0), "L=29 pads to 32, so slot 7 is max(layer28, 0) = relu"
    assert np.all(o[6, :16, :] == -3.0), "slot 6 covers real layers only and must pass through"
    print("    [PASS] zero padding, and slot 7 is relu(final layer) at L=29 -- both would be wrong "
          "under edge replication")

    # Completions longer than N_MAX are TRUNCATED by their pad_activations_tensor, not pooled.
    B = np.zeros((29, 137, 2), dtype=np.float32)
    B[:, 120, :] = 9.0                      # a large value beyond N_MAX must NOT survive
    assert act_pool(B, 8, 100).max() == 0.0, "content past N_MAX=100 must be dropped"
    print("    [PASS] tokens beyond N_MAX are truncated, matching pad_activations_tensor")

    # Size guard. TyDiQA and TruthfulQA must pass at the default pool; TriviaQA must be refused.
    tyd = estimate_gb(4400, 8, 100, 3584)
    tqa = estimate_gb(8170, 8, 100, 3584)
    tri = estimate_gb(99600, 8, 100, 3584)
    assert tyd < DEFAULT_MAX_GB and tqa < DEFAULT_MAX_GB, (tyd, tqa)
    assert tri > DEFAULT_MAX_GB, tri
    assert estimate_gb(99600, 8, 20, 3584) < tri / 4, "N_eff=20 must be a real reduction"
    print("    [PASS] sizes: TyDiQA %.0f GB, TruthfulQA %.0f GB pass; TriviaQA %.0f GB refused, "
          "%.0f GB at N_eff=20" % (tyd, tqa, tri, estimate_gb(99600, 8, 20, 3584)))

    # repool must equal a direct extraction at the smaller N_eff. Max is associative, so blocking
    # five singleton columns of the N_eff=100 file equals blocking five raw columns -- but only if
    # repool leaves the layer axis alone, which is the bug this pins.
    import tempfile
    rng = np.random.default_rng(0)
    raws = [rng.standard_normal((29, int(t), 4)).astype(np.float32) for t in (1, 3, 16, 17, 64)]
    src = np.stack([act_pool(r, 8, 100) for r in raws]).astype(np.float16)
    direct = np.stack([act_pool(r, 8, 20) for r in raws]).astype(np.float16)
    with tempfile.TemporaryDirectory() as d:
        pth = os.path.join(d, "x_at_L8_N100.npz")
        np.savez(pth, at=src, prompt_id=np.arange(len(raws)), label=np.zeros(len(raws)),
                 beam_row=np.arange(len(raws)),
                 token_len=np.array([r.shape[1] for r in raws], dtype=np.int32))
        got = np.load(repool(pth, 20))["at"]
    assert got.shape == direct.shape, (got.shape, direct.shape)
    assert np.array_equal(got, direct), "repool(100 -> 20) != direct extraction at N_eff=20"
    print("    [PASS] repool from N_eff=100 to 20 reproduces direct extraction exactly, "
          "T in {1,3,16,17,64}")

    # Applying act_pool a second time would look plausible and be wrong: L_for_pad adds a full
    # block, so an 8-layer input keeps 4 real slots and zeros the rest. Pinned so that nobody
    # "simplifies" repool back into a second act_pool call.
    twice = act_pool(src[2].astype(np.float32), 8, 20)
    assert np.all(twice[4:] == 0.0), "expected act_pool to zero the top half on a second pass"
    assert not np.array_equal(twice, direct[2])
    print("    [PASS] act_pool applied twice zeros the top 4 layer slots -- why repool is separate")

    with tempfile.TemporaryDirectory() as d:
        pth = os.path.join(d, "y_at_L8_N20.npz")
        np.savez(pth, at=np.zeros((1, 8, 20, 2), dtype=np.float16), prompt_id=np.array([0]),
                 label=np.array([0]), beam_row=np.array([0]),
                 token_len=np.array([64], dtype=np.int32))
        try:
            repool(pth, 10)
            raise AssertionError("repool accepted a source that was already token-pooled")
        except SystemExit as e:
            assert "N_eff = N_MAX" in str(e), str(e)
    print("    [PASS] repool refuses a source not extracted at N_eff = N_MAX")

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
    p.add_argument("--repool-from", default=None,
                   help="existing .npz to re-derive a smaller --n-pool from, CPU only, no "
                        "model load. Valid only if the source did no token pooling.")
    p.add_argument("--limit", type=int, default=None, help="first N questions only")
    p.add_argument("--log-every", type=int, default=50)
    a = p.parse_args()

    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if a.repool_from:
        repool(a.repool_from, a.n_pool)
        raise SystemExit(0)
    if not a.dataset or not a.model_folder:
        raise SystemExit("--dataset and --model_folder are required (or use --self-test)")
    run(a.dataset, a.model_folder, a.data_dir, a.out_dir, a.device, a.dtype,
        a.l_pool, a.n_pool, a.max_gb, a.limit, a.log_every)


if __name__ == "__main__":
    main()
