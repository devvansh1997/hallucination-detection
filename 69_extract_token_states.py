"""
69_extract_token_states.py -- token-level hidden states of the layer window, for the 3-mode Tucker ablation (T-015).
==================================================================================================

WHY. The reported detector summarises the token axis away (peak, range, update over tokens) BEFORE
decomposing, so its Tucker-2 step only ever sees a layers x features matrix. The 3-mode ablation (70)
keeps a pooled token axis and decomposes layers x token-bins x features. That needs the hidden states of
every answer token, which the pinned pipeline deleted after deriving its summaries (42_extract_phase2.py
removes the raw-state store). This script recomputes them once; every pooling size is then built on CPU.

WHAT IS STORED (one directory per model and dataset):
    tokens.npy    (total_T, 9, D) float16   hidden-state indices 16..24 (blocks 15..23), completion tokens
                                            only, answers concatenated in the sequences file's beam order
    offsets.npy   (N + 1,)                  answer n occupies tokens[offsets[n]:offsets[n+1]]
    prompt_id.npy, label.npy (N,)           as in the sequences file
    meta.json                               written LAST: a directory without it is incomplete

tokens.npy is written through a memory map, so host RAM does not have to hold it. Its size is known
exactly before the first forward pass (completion lengths are in the sequences file) and the job refuses
above --max-gb.

PREFLIGHT. After the first --check-questions questions, the max over tokens of the stored states must
match the pinned static_max features of those answers (corr >= 0.999), or the job stops.

  python 69_extract_token_states.py --self-test
  python 69_extract_token_states.py --dataset truthfulqa --model_folder llama-3.1-8b
"""

import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, "..", "data-tokenstates"))
WINDOW = list(range(16, 25))          # hidden-state indices of blocks 15..23
CHECK_CORR = 0.999


def plan(input_ids, prompt_lens):
    """Completion lengths and offsets, in beam order."""
    lengths = np.array([len(ids) - int(pl) for ids, pl in zip(input_ids, prompt_lens)], dtype=np.int64)
    if (lengths < 0).any():
        raise SystemExit("a sequence is shorter than its prompt")
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return lengths, offsets


def size_gb(total_tokens, D, n_layers=len(WINDOW), bytes_per=2):
    return total_tokens * n_layers * D * bytes_per / 1024 ** 3


def corr(a, b):
    x, z = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(z)
    return float(np.corrcoef(x[ok], z[ok])[0, 1])


def run(dataset, model_folder, data_dir, out_root, device, max_gb, check_questions, log_every=50):
    import torch
    import yaml
    from transformers import AutoModelForCausalLM

    out_dir = os.path.join(out_root, model_folder, dataset)
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        raise SystemExit("refusing: %s is complete already" % out_dir)
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)
    seq = torch.load(os.path.join(data_dir, model_folder, "%s_sequences_v1.pt" % dataset), weights_only=False)
    input_ids, prompt_lens = seq["input_ids"], seq["prompt_len"]
    prompt_id = np.asarray(seq["prompt_id"], dtype=np.int64)
    label = np.asarray(seq["all_hallucination_flag"], dtype=np.int64)
    lengths, offsets = plan(input_ids, prompt_lens)
    N, total = len(lengths), int(offsets[-1])

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model.eval()
    D = model.config.hidden_size
    need = size_gb(total, D)
    print("  [%s/%s] %d answers, %d completion tokens (mean %.1f, max %d, empty %d) -> tokens.npy %.1f GB"
          % (model_folder, dataset, N, total, lengths.mean(), lengths.max(), int((lengths == 0).sum()), need),
          flush=True)
    if need > max_gb:
        raise SystemExit("refusing: %.1f GB exceeds --max-gb %.1f" % (need, max_gb))

    os.makedirs(out_dir, exist_ok=True)
    tok = np.lib.format.open_memmap(os.path.join(out_dir, "tokens.npy"), mode="w+", dtype=np.float16,
                                    shape=(total, len(WINDOW), D))
    order = np.argsort(prompt_id, kind="stable")
    groups = [order[prompt_id[order] == q] for q in np.unique(prompt_id)]
    pinned = np.load(os.path.join(data_dir, model_folder, "%s_phase2_features.npz" % dataset))
    pad = model.config.eos_token_id if model.config.eos_token_id is not None else 0
    if isinstance(pad, (list, tuple)):
        pad = pad[0]
    t0, checked = time.time(), False
    for gi, idx in enumerate(groups):
        batch = [input_ids[i] for i in idx]
        Lmax = max(len(b) for b in batch)
        ids = torch.full((len(batch), Lmax), int(pad), dtype=torch.long)
        att = torch.zeros((len(batch), Lmax), dtype=torch.long)
        for j, b in enumerate(batch):
            t = b if torch.is_tensor(b) else torch.as_tensor(b)
            ids[j, :len(t)] = t
            att[j, :len(t)] = 1
        with torch.no_grad():
            out = model(ids.to(device), attention_mask=att.to(device), use_cache=False, output_hidden_states=True)
        for j, n in enumerate(idx):
            s = int(prompt_lens[n])
            e = s + int(lengths[n])
            if e == s:
                continue
            h = torch.stack([out.hidden_states[i][j, s:e, :] for i in WINDOW], dim=1)     # (T, 9, D)
            tok[offsets[n]:offsets[n + 1]] = h.float().cpu().numpy().astype(np.float16)
        del out

        if not checked and gi + 1 >= check_questions:
            done = np.concatenate(groups[:gi + 1])
            done = done[lengths[done] > 0]
            mine = np.stack([np.asarray(tok[offsets[n]:offsets[n + 1]], dtype=np.float32).max(axis=0) for n in done])
            ref = np.asarray(pinned["static_max"])[done]
            c = corr(mine, ref)
            print("  PREFLIGHT: max over tokens of %d stored answers vs pinned static_max: corr %.6f -> %s"
                  % (len(done), c, "PASS" if c >= CHECK_CORR else "FAIL"), flush=True)
            if c < CHECK_CORR:
                raise SystemExit("preflight failed: stored token states do not reproduce the pinned peak features")
            checked = True
        if log_every and (gi + 1) % log_every == 0:
            el = time.time() - t0
            print("    %d/%d questions (%.0fs, eta %.0fs)" % (gi + 1, len(groups), el,
                                                              el / (gi + 1) * (len(groups) - gi - 1)), flush=True)
    tok.flush()
    nonfinite = 0
    for a in range(0, total, 200000):
        nonfinite += int((~np.isfinite(np.asarray(tok[a:a + 200000], dtype=np.float32))).sum())
    del tok
    for name, arr in (("offsets", offsets), ("prompt_id", prompt_id), ("label", label), ("lengths", lengths)):
        np.save(os.path.join(out_dir, "%s.npy" % name), arr)
    meta = {"dataset": dataset, "model_folder": model_folder, "model_id": model_id, "n_answers": N,
            "total_tokens": total, "hidden_size": D, "hidden_state_indices": WINDOW, "dtype": "float16",
            "mean_tokens": float(lengths.mean()), "max_tokens": int(lengths.max()),
            "empty_answers": int((lengths == 0).sum()), "nonfinite_entries": nonfinite,
            "preflight_passed": checked, "size_gb": round(need, 2), "elapsed_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\n  wrote %s  (%.1f GB, non-finite entries %d)" % (out_dir, need, nonfinite))


def self_test():
    ids = [list(range(7)), list(range(4)), list(range(10))]
    lengths, offsets = plan(ids, [3, 4, 2])
    assert list(lengths) == [4, 0, 8] and list(offsets) == [0, 4, 4, 12]
    assert abs(size_gb(1024 ** 3, 1, n_layers=1, bytes_per=1) - 1.0) < 1e-12
    assert size_gb(10, 4096) == 10 * 9 * 4096 * 2 / 1024 ** 3
    try:
        plan([[1, 2]], [3])
        raise AssertionError("a sequence shorter than its prompt must be refused")
    except SystemExit:
        pass
    a = np.arange(12.0)
    assert abs(corr(a, 2 * a + 1) - 1.0) < 1e-12
    print("  [PASS] completion lengths, offsets (an empty answer occupies no tokens), size estimate, corr")
    print("\n  ALL PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dataset", choices=["truthfulqa", "tydiqa_gp", "nq_open", "triviaqa"])
    ap.add_argument("--model_folder")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-gb", type=float, default=60.0)
    ap.add_argument("--check-questions", type=int, default=20)
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not (a.dataset and a.model_folder):
        raise SystemExit("--dataset and --model_folder are required (or --self-test)")
    data_dir = a.data_dir
    if not data_dir:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]
    run(a.dataset, a.model_folder, data_dir, a.out_dir, a.device, a.max_gb, a.check_questions)


if __name__ == "__main__":
    main()
