"""
77_extract_math_states.py -- hidden states of the sampled math answers: per token, and LOTUS's summaries (T-025).
==================================================================================================

WHY. 76 produced K sampled answers per question, each right or wrong by its final number. To ask whether
hidden states mark where an answer goes wrong, and what a pooled detector already achieves, two views of the
same forward pass are needed: the state at every answer token, and the token-pooled summaries LOTUS reads.

EACH ANSWER IS READ BY THE MODEL THAT WROTE IT, on exactly the token ids it generated (the sequences file
stores them), so there is no template to reconstruct.

WHAT IS STORED (config.yaml math.out_dir / <model> / <dataset>_states/), answers in the sequences file's order:
    tokens.npy          (total_T, 9, 64) float16   per-token states of blocks 15..23, robust-scaled and
                                                   projected onto 64 directions per block
    offsets.npy         (N + 1,)                   answer n occupies tokens[offsets[n]:offsets[n+1]]
    lotus_core.npy      (N, 9, D) float16          max over the answer's tokens
    lotus_q95.npy, lotus_q05.npy  (N, 9, D) float16   0.95 / 0.05 quantiles over tokens
    lotus_v95.npy, lotus_v05.npy  (N, 8, D) float16   the same quantiles of h[l+1] - h[l]
    prompt_id.npy       (N,)                       copied from the sequences file, for the reader to verify
    basis_question.npy  (N,) bool                  answers whose question the scaler and basis were fitted on
    nll.npy             (N,)                       mean negative log-likelihood of the answer tokens
    meta.json                                      written LAST: a directory without it is incomplete

LABELS ARE NOT STORED. Right or wrong lives only in the sequences file, which can be regraded (76 --regrade);
every reader takes labels from there, so this store can never hold stale ones.

THE BASIS NEVER SEES AN EVALUATED QUESTION. Raw per-token states would be ~220 GB; projected they are ~4 GB.
The scaler and the 64 directions per block are fitted on tokens of 150 questions drawn at random WITHOUT
reference to any label (so a regrade cannot change which ones). Those questions are flagged in
basis_question.npy and every later measurement leaves them out. The basis itself is not saved (AGENTS: no
cached bases); meta.json records how it was fitted.

LOTUS'S SUMMARIES use 57.pool_answer's definitions (max, and linear-interpolated quantiles, elementwise over
tokens; update = h[l+1] - h[l]). They are computed on the GPU for speed, and a check on the first answers
requires them to equal 57.pool_answer's output before the long pass.

  python 77_extract_math_states.py --self-test
  python 77_extract_math_states.py --model_folder qwen-2.5-7b-instruct
"""

import argparse
import importlib.util
import json
import os
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST_BLOCK = 15           # blocks 15..23, the window of the main experiments
N_BLOCKS = 9
R = 64
N_FIT = 40000              # tokens the scaler and basis are fitted on
FIT_POOL = 60000           # tokens read from the basis questions before subsampling to N_FIT
N_BASIS_QUESTIONS = 150
N_SUMMARY_CHECK = 3
LOTUS_KEYS = ("core", "q95", "q05", "v95", "v05")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def choose_basis_questions(prompt_id, n, seed):
    """n question ids drawn at random from the questions present. Uses nothing but the ids."""
    q = np.unique(np.asarray(prompt_id))
    return np.sort(np.random.default_rng(seed).choice(q, size=min(n, len(q)), replace=False))


def lotus_summaries(Hw):
    """Hw: torch (P, T, D) float32, one answer's tokens at P consecutive hidden-state indices. Returns
    core / q95 / q05 (P, D) and v95 / v05 (P - 1, D) as float32 numpy -- 57.pool_answer's definitions.
    Quantiles run per layer: torch.quantile refuses inputs above 16M elements, which nine layers exceed."""
    import torch
    q = torch.tensor([0.95, 0.05], device=Hw.device, dtype=Hw.dtype)
    qs = torch.stack([torch.quantile(Hw[j], q, dim=0) for j in range(Hw.shape[0])])
    d = Hw[1:] - Hw[:-1]
    vs = torch.stack([torch.quantile(d[j], q, dim=0) for j in range(d.shape[0])])
    out = {"core": Hw.amax(dim=1), "q95": qs[:, 0], "q05": qs[:, 1], "v95": vs[:, 0], "v05": vs[:, 1]}
    return {k: v.cpu().numpy().astype(np.float32) for k, v in out.items()}


def read_answer(model, ids, n_prompt, layer_idx, device):
    """(P, T, D) float32 states of the answer tokens on the device, and the answer's mean NLL."""
    import torch
    x = torch.as_tensor(ids, device=device).reshape(1, -1)
    with torch.no_grad():
        out = model(input_ids=x, output_hidden_states=True)
        Hw = torch.stack([out.hidden_states[l][0, n_prompt:, :] for l in layer_idx]).float()
        nll = torch.nn.functional.cross_entropy(out.logits[0, n_prompt - 1:-1, :].float(), x[0, n_prompt:]).item()
    return Hw, nll


def git_head():
    try:
        return subprocess.run(["git", "-C", HERE, "log", "--oneline", "-1"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def run(model_folder, dataset, device, seed, n_fit):
    import torch
    import yaml
    from transformers import AutoModelForCausalLM

    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next((m["id"] for m in cfg["models"] if m["folder"] == model_folder), None)
    if model_id is None:
        raise SystemExit("unknown model folder %r" % model_folder)
    root = os.path.normpath(os.path.join(HERE, cfg["math"]["out_dir"], model_folder))
    seq_path = os.path.join(root, "%s_sequences_v1.pt" % dataset)
    out_dir = os.path.join(root, "%s_states" % dataset)
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        raise SystemExit("refusing: %s is complete already" % out_dir)

    seq = torch.load(seq_path, weights_only=False)
    input_ids, prompt_len = seq["input_ids"], [int(p) for p in seq["prompt_len"]]
    prompt_id = np.asarray(seq["prompt_id"], dtype=np.int64)
    lengths = np.array([len(ids) - pl for ids, pl in zip(input_ids, prompt_len)], dtype=np.int64)
    if (lengths <= 0).any():
        raise SystemExit("refusing: %d answers have no tokens" % int((lengths <= 0).sum()))
    N = len(lengths)
    offsets = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    total_T = int(offsets[-1])

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model.eval()
    D = int(model.config.hidden_size)
    if FIRST_BLOCK + N_BLOCKS > int(model.config.num_hidden_layers):
        raise SystemExit("window blocks %d..%d exceed the model" % (FIRST_BLOCK, FIRST_BLOCK + N_BLOCKS - 1))
    layer_idx = list(range(FIRST_BLOCK + 1, FIRST_BLOCK + 1 + N_BLOCKS))      # hidden-state index of block b is b + 1
    lotus_gb = N * D * (3 * N_BLOCKS + 2 * (N_BLOCKS - 1)) * 2 / 1e9
    print("  [%s/%s] %d answers, %d tokens (median %d) | tokens.npy %.1f GB + LOTUS summaries %.1f GB"
          % (model_folder, dataset, N, total_T, int(np.median(lengths)), total_T * N_BLOCKS * R * 2 / 1e9, lotus_gb),
          flush=True)

    s28 = _load("s28", "28_eval_band.py")
    s57 = _load("s57", "57_extract_all_layers.py")
    s73 = _load("s73", "73_token_structure.py")
    s74 = _load("s74", "74_extract_ragtruth.py")

    # check: GPU summaries equal 57.pool_answer on the first answers
    worst = 0.0
    for n in range(min(N_SUMMARY_CHECK, N)):
        Hw, _ = read_answer(model, input_ids[n], prompt_len[n], layer_idx, device)
        mine = lotus_summaries(Hw)
        ref = s57.pool_answer(Hw.cpu().numpy(), s57.STREAMS)
        for k in LOTUS_KEYS:
            scale = max(1.0, float(np.abs(ref[k]).max()))
            worst = max(worst, float(np.abs(mine[k] - ref[k]).max()) / scale)
    print("  summary check: largest relative difference from 57.pool_answer on %d answers = %.2e"
          % (min(N_SUMMARY_CHECK, N), worst), flush=True)
    if worst > 1e-4:
        raise SystemExit("refusing: GPU summaries differ from 57.pool_answer")

    # fit pass: scaler and basis from label-free basis questions
    basis_q = choose_basis_questions(prompt_id, N_BASIS_QUESTIONS, seed)
    is_basis = np.isin(prompt_id, basis_q)
    rng = np.random.default_rng(seed)
    fit_H, fit_len, n_read = [], [], 0
    t0 = time.time()
    for n in rng.permutation(np.flatnonzero(is_basis)):
        Hw, _ = read_answer(model, input_ids[n], prompt_len[n], layer_idx, device)
        fit_H.append(Hw.permute(1, 0, 2).cpu().numpy().astype(np.float16))
        fit_len.append(int(Hw.shape[1]))
        n_read += int(Hw.shape[1])
        if n_read >= FIT_POOL:
            break
    fit_tokens = np.concatenate(fit_H, axis=0)
    del fit_H
    fit_off = np.zeros(len(fit_len) + 1, dtype=np.int64)
    np.cumsum(fit_len, out=fit_off[1:])
    space, share, used = s73.fit_space(fit_tokens, fit_off, np.arange(len(fit_len)), s28, n_fit, R, seed)
    del fit_tokens
    print("  fit pass: %d basis questions (%d answers flagged), %d answers / %d tokens read, fitted on %d | "
          "variance kept by 64 directions per block %s (%.0fs)"
          % (len(basis_q), int(is_basis.sum()), len(fit_len), n_read, used, " ".join("%.3f" % v for v in share),
             time.time() - t0), flush=True)

    # store pass
    os.makedirs(out_dir, exist_ok=True)
    mm = np.lib.format.open_memmap(os.path.join(out_dir, "tokens.npy"), mode="w+", dtype=np.float16,
                                   shape=(total_T, N_BLOCKS, R))
    lot = {k: np.lib.format.open_memmap(os.path.join(out_dir, "lotus_%s.npy" % k), mode="w+", dtype=np.float16,
                                        shape=(N, N_BLOCKS if k in ("core", "q95", "q05") else N_BLOCKS - 1, D))
           for k in LOTUS_KEYS}
    nll = np.empty(N, dtype=np.float64)
    t0 = time.time()
    for n in range(N):
        Hw, nll[n] = read_answer(model, input_ids[n], prompt_len[n], layer_idx, device)
        for k, v in lotus_summaries(Hw).items():
            lot[k][n] = v
        mm[offsets[n]:offsets[n + 1]] = s74.project_response(Hw.permute(1, 0, 2).cpu().numpy(), space, s28)
        if (n + 1) % 500 == 0 or n + 1 == N:
            el = time.time() - t0
            print("    store pass %d/%d (%.0fs, ~%.0fs left)" % (n + 1, N, el, el / (n + 1) * (N - n - 1)), flush=True)
    mm.flush()
    nonfinite = {"tokens": 0}
    for a in range(0, total_T, 65536):
        nonfinite["tokens"] += int((~np.isfinite(mm[a:a + 65536])).sum())
    for k in LOTUS_KEYS:
        lot[k].flush()
        nonfinite["lotus_" + k] = int(sum((~np.isfinite(lot[k][a:a + 256])).sum() for a in range(0, N, 256)))
    del mm, lot

    np.save(os.path.join(out_dir, "offsets.npy"), offsets)
    np.save(os.path.join(out_dir, "prompt_id.npy"), prompt_id)
    np.save(os.path.join(out_dir, "basis_question.npy"), is_basis)
    np.save(os.path.join(out_dir, "nll.npy"), nll)
    meta = {"model_folder": model_folder, "model_id": model_id, "dataset": dataset, "sequences": seq_path,
            "blocks": [FIRST_BLOCK + j for j in range(N_BLOCKS)], "hidden_state_indices": layer_idx, "r": R,
            "hidden_size": D, "n_answers": N, "total_tokens": total_T,
            "basis": {"questions": [int(q) for q in basis_q], "answers_flagged": int(is_basis.sum()),
                      "answers_read": len(fit_len), "tokens_read": n_read, "tokens_fitted": used, "seed": seed,
                      "chosen_by": "random draw over question ids, no labels",
                      "variance_kept_by_64_directions": share},
            "lotus_summaries": "57.pool_answer definitions; largest relative difference %.2e on %d answers"
                               % (worst, min(N_SUMMARY_CHECK, N)),
            "labels": "not stored; read all_hallucination_flag from the sequences file",
            "answer_nll_mean": float(nll.mean()), "nonfinite_entries": nonfinite,
            "repo_head": git_head(), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("  done: %d answers, %d tokens, mean answer NLL %.3f, non-finite %s -> %s"
          % (N, total_T, nll.mean(), nonfinite, out_dir), flush=True)
    if any(nonfinite.values()):
        raise SystemExit("non-finite values were stored (float16 overflow?) -- do not use this store")


# ---------------------------------------------------------------------------------------------

def self_test():
    import torch
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print("    [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
        ok = ok and bool(cond)

    s57 = _load("s57", "57_extract_all_layers.py")
    g = torch.Generator().manual_seed(0)
    Hw = torch.randn(9, 37, 50, generator=g) * 3.0
    mine = lotus_summaries(Hw)
    ref = s57.pool_answer(Hw.numpy(), s57.STREAMS)
    worst = max(float(np.abs(mine[k] - ref[k]).max()) for k in LOTUS_KEYS)
    check("summaries equal 57.pool_answer", worst < 1e-4, "largest difference %.2e" % worst)
    check("summary shapes", mine["core"].shape == (9, 50) and mine["q95"].shape == (9, 50) and mine["v05"].shape == (8, 50),
          "%s %s %s" % (mine["core"].shape, mine["q95"].shape, mine["v05"].shape))
    one = lotus_summaries(torch.randn(9, 1, 50, generator=g))
    check("a one-token answer is handled", np.isfinite(one["q95"]).all() and np.allclose(one["q95"], one["q05"]))

    pid = np.repeat(np.arange(40), 10)
    a, b = choose_basis_questions(pid, 6, 0), choose_basis_questions(pid, 6, 0)
    check("basis questions: right count, repeatable", len(a) == 6 and (a == b).all(), str(a.tolist()))
    check("basis questions change with the seed", not (a == choose_basis_questions(pid, 6, 1)).all())
    check("more requested than exist: all of them", len(choose_basis_questions(pid, 999, 0)) == 40)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model_folder")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-fit", type=int, default=N_FIT)
    a = ap.parse_args()
    if a.self_test:
        print("  self-test")
        raise SystemExit(0 if self_test() else 1)
    if not a.model_folder:
        raise SystemExit("--model_folder is required (or --self-test)")
    run(a.model_folder, a.dataset, a.device, a.seed, a.n_fit)


if __name__ == "__main__":
    main()
