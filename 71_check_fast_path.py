"""
71_check_fast_path.py -- does Falcon-H1 run on its Mamba kernels, and do they compute the same thing? (T-014)
==================================================================================================

Falcon-H1 mixes attention with Mamba-2 layers. Without the causal_conv1d and mamba_ssm packages,
transformers falls back to a pure-PyTorch path: the first pilot ran at 44 s per TruthfulQA question and
its batched decoding looped (TICKETS T-014). slurm/build_h1_env.slurm builds both packages into a clone
of hal-det (hal-det-h1) and runs this check before any pilot or generation job uses the clone.

CHECKS, in order (exit 0 only if all pass):
  K1 packages   causal_conv1d and mamba_ssm import, and transformers is not in Hub-kernel mode: when the
                `kernels` package is installed, transformers 5.x looks for these kernels ONLY on the Hub
                (integrations/hub_kernels.lazy_load_kernel) and ignores local builds. torch, transformers
                and CUDA must still be the versions 39 pins, so hal-det-h1 matches hal-det.
  K2 fast path  After the model loads, modeling_falcon_h1.is_fast_path_available is True.
  K3 agreement  One batched forward pass over the first TruthfulQA prompts, right-padded as the extraction
                scripts do, on the fast path and then on the PyTorch path in the same process: relative
                difference of the window hidden states (indices 16..24) over real tokens, and agreement of
                the next-token argmax. Both passes are bf16, so the tolerance allows rounding, not bugs.

  python 71_check_fast_path.py --model_folder falcon-h1-7b-base
  python 71_check_fast_path.py --self-test
"""

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = list(range(16, 25))
MAX_REL_DIFF = 0.05          # per window index, ||fast - slow||_F / ||slow||_F over real tokens
MIN_ARGMAX_AGREE = 0.95      # share of real positions whose next-token argmax is the same


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rel_diff(fast, slow, mask):
    """fast, slow: arrays (B, T, D); mask: (B, T) bool, True on real tokens. Relative Frobenius difference."""
    f = fast[mask].astype(np.float64)
    s = slow[mask].astype(np.float64)
    return float(np.linalg.norm(f - s) / max(np.linalg.norm(s), 1e-12))


def argmax_agreement(fast, slow, mask):
    """fast, slow: int arrays (B, T) of next-token argmax; share of real positions where they agree."""
    return float((fast[mask] == slow[mask]).mean())


def package_versions():
    out = {}
    for pkg in ("causal_conv1d", "mamba_ssm", "triton", "kernels"):
        try:
            m = importlib.import_module(pkg)
            out[pkg] = str(getattr(m, "__version__", "installed"))
        except Exception as e:  # noqa: BLE001
            out[pkg] = "missing (%s: %s)" % (type(e).__name__, str(e).splitlines()[0][:160] if str(e) else "")
    return out


def run(model_folder, n_prompts, out_dir):
    import torch
    import transformers
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.integrations import hub_kernels
    from transformers.utils import import_utils

    s39 = _load("s39", "39_generate_dataset.py")
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)
    ds_cfgs = {d["name"]: d for d in cfg["datasets"]}
    report = {"model_folder": model_folder, "model_id": model_id, "checks": {},
              "tolerance": {"max_rel_diff": MAX_REL_DIFF, "min_argmax_agreement": MIN_ARGMAX_AGREE}}
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "fastpath_%s.json" % model_folder)

    def check(name, ok, **info):
        report["checks"][name] = dict(passed=bool(ok), **info)
        print("  [%s] %s %s" % ("PASS" if ok else "FAIL", name, json.dumps(info, default=str)), flush=True)

    def finish():
        report["all_passed"] = bool(report["checks"]) and all(c["passed"] for c in report["checks"].values())
        with open(dst, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print("\n  ALL CHECKS PASSED: %s   (wrote %s)" % (report["all_passed"], dst), flush=True)
        return report

    # K1 --------------------------------------------------------------------------------------------
    versions = dict(torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
                    **package_versions())
    hub_mode = bool(getattr(hub_kernels, "_kernels_available", False))
    pinned = {k: s39.EXPECTED_VERSIONS[k] for k in ("torch", "transformers", "cuda")}
    detected = dict(causal_conv1d=bool(import_utils.is_causal_conv1d_available()),
                    mamba_2_ssm=bool(import_utils.is_mamba_2_ssm_available()))
    ok = (all(detected.values()) and not hub_mode and all(versions[k] == pinned[k] for k in pinned))
    check("K1_packages", ok, versions=versions, transformers_detects=detected,
          hub_kernel_mode=hub_mode, pinned_by_39=pinned)

    # K2 --------------------------------------------------------------------------------------------
    device = torch.device("cuda")
    try:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
        model.eval()
        mod = sys.modules[type(model).__module__]
        fast = bool(getattr(mod, "is_fast_path_available", False))
        check("K2_fast_path", fast, module=mod.__name__, load_seconds=round(time.time() - t0, 1),
              gpu=torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        check("K2_fast_path", False, error=traceback.format_exc(limit=4))
        return finish()
    if not fast:
        return finish()

    # K3 --------------------------------------------------------------------------------------------
    try:
        samples, _ = s39.load_dataset_samples(ds_cfgs["truthfulqa"])
        enc = [tok(s["prompt_text"]).input_ids for s in samples[:n_prompts]]
        L = max(len(x) for x in enc)
        ids = torch.full((len(enc), L), int(tok.pad_token_id), dtype=torch.long)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for j, x in enumerate(enc):
            ids[j, :len(x)] = torch.tensor(x)
            att[j, :len(x)] = 1
        ids, att = ids.to(device), att.to(device)

        def forward():
            torch.cuda.synchronize()
            t = time.time()
            with torch.no_grad():
                out = model(ids, attention_mask=att, use_cache=False, output_hidden_states=True)
            torch.cuda.synchronize()
            hs = [out.hidden_states[i].float().cpu().numpy() for i in WINDOW]
            return hs, out.logits.argmax(-1).cpu().numpy(), time.time() - t

        fast_hs, fast_am, t_fast = forward()
        setattr(mod, "is_fast_path_available", False)
        try:
            slow_hs, slow_am, t_slow = forward()
        finally:
            setattr(mod, "is_fast_path_available", True)
        mask = att.bool().cpu().numpy()
        diffs = {i: round(rel_diff(f, s, mask), 5) for i, f, s in zip(WINDOW, fast_hs, slow_hs)}
        agree = argmax_agreement(fast_am, slow_am, mask)
        ok = max(diffs.values()) <= MAX_REL_DIFF and agree >= MIN_ARGMAX_AGREE
        check("K3_agreement", ok, prompts=len(enc), max_len=L, real_tokens=int(mask.sum()),
              rel_diff_by_hidden_index=diffs, max_rel_diff=max(diffs.values()),
              argmax_agreement=round(agree, 4),
              forward_seconds=dict(fast=round(t_fast, 2), pytorch=round(t_slow, 2)))
    except Exception:  # noqa: BLE001
        check("K3_agreement", False, error=traceback.format_exc(limit=4))
    return finish()


def self_test():
    rng = np.random.default_rng(0)
    s = rng.standard_normal((3, 7, 16))
    mask = np.ones((3, 7), dtype=bool)
    mask[1, 5:] = False
    mask[2, 3:] = False
    assert rel_diff(s, s, mask) == 0.0
    noisy = s + 0.01 * rng.standard_normal(s.shape)
    d = rel_diff(noisy, s, mask)
    assert 0.005 < d < 0.02, d
    junk = s.copy()
    junk[~mask] = 1e6                      # padded positions must not count
    assert rel_diff(junk, s, mask) == 0.0
    print("  [PASS] rel_diff: 0 for identical states, ~0.01 for 1%% noise (%.4f), padding ignored" % d)
    a = rng.integers(0, 50, (3, 7))
    b = a.copy()
    b[0, 0] += 1                           # one real position differs
    b[2, 6] += 1                           # a padded position differs: ignored
    assert abs(argmax_agreement(a, b, mask) - (1 - 1 / mask.sum())) < 1e-12
    print("  [PASS] argmax_agreement counts real positions only")
    print("\n  ALL PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model_folder")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "pilot_kernels"))
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not a.model_folder:
        raise SystemExit("--model_folder is required (it must have an entry in config.yaml)")
    r = run(a.model_folder, a.n_prompts, a.out_dir)
    raise SystemExit(0 if r.get("all_passed") else 1)


if __name__ == "__main__":
    main()
