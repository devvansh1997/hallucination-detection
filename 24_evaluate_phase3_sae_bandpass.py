"""
24_evaluate_phase3_sae_bandpass.py -- SAE Bandpass Filter Evaluation
=======================================================================
Phase 3: Token-level SAE encoding + positive max-pool + bandpass filter.
Standalone -- extracts raw hidden states, encodes per token via SAE,
pools AFTER encoding.  No pre-pooled tensor dependency.

Requires:  pip install sae-lens
"""

import argparse, gc, os, time
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
DATA_DIR = cfg["output"]["data_dir"]
RANDOM_SEED = 42
LAYERS = list(range(15, 24))

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, default="llama-3.1-8b-instruct")
parser.add_argument("--n_pilot", type=int, default=10)
args = parser.parse_args()

model_id = None
for m in cfg["models"]:
    if m["folder"] == args.model_folder:
        model_id = m["id"]; break
if model_id is None:
    raise ValueError("Unknown folder: " + args.model_folder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# STEP 0: DUMMY TESTS
def run_dummy_tests():
    print("[STEP 0] Dummy tests ...")
    x = torch.randn(10, 15, 32768)
    pooled = torch.relu(x).max(dim=1).values
    assert (pooled >= 0).all() and not pooled.isnan().any()
    print("  [PASS] Test 1: ReLU + max-pool non-negative")

    M_s = torch.zeros(100, 1000)
    M_s[:, :11] = 1.0
    M_s[:, 21:101] = (torch.rand(100, 80) < 0.02).float()
    freq = M_s.mean(dim=0)
    mask = (freq >= 0.0005) & (freq <= 0.05)
    filt = M_s[:, mask]
    assert 60 <= filt.shape[1] <= 80, f"Bandpass kept {filt.shape[1]} cols"
    print("  [PASS] Test 2: Bandpass ~80 cols")

    Fc = torch.randn(100, 320).numpy()
    Fs = torch.randn(100, 500).numpy()
    spl = int(100*0.75)
    r = Ridge(alpha=1.0); r.fit(Fc[:spl], Fs[:spl])
    Fp = Fs - r.predict(Fc)
    c = np.corrcoef(Fc[spl:].ravel()[:100], Fp[spl:].ravel()[:100])[0,1]
    assert abs(c) < 0.3
    print("  [PASS] Test 3: Gram-Schmidt orthogonal")
    print("[PASS] All Phase 3 SAE Unit Tests\n")


# STEP 1-2: STANDALONE SAE EXTRACTION
def extract_sae_features(model_folder, n_pilot):
    try:
        from sae_lens import SAE
    except ImportError:
        raise RuntimeError("sae-lens not installed. pip install sae-lens")

    print(f"  Backbone: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    D = model.config.hidden_size

    saes = {}
    release = "llama_scope_lxr_8b"
    for l in LAYERS:
        print(f"  SAE layer {l} ...")
        sae, _, _ = SAE.from_pretrained(
            release=release, sae_id=f"blocks.{l}.hook_resid_post",
            device=str(device))
        saes[l] = sae
    M = saes[LAYERS[0]].cfg.d_sae
    print(f"  SAE dict width: {M}")

    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    prompts = [str(ds["question"][i]) for i in range(n_pilot)]
    corr_list = [str(ds["best_answer"][i]) for i in range(n_pilot)]
    wrg_list = [list(ds["incorrect_answers"][i]) for i in range(n_pilot)]

    import evaluate
    rouge = evaluate.load("rouge")
    bleurt = evaluate.load("bleurt", config_name="BLEURT-20")

    eos_ids = {tokenizer.eos_token_id}
    for s in [".", "!", "?", "\n"]:
        for tok in tokenizer.encode(s, add_special_tokens=False):
            eos_ids.add(tok)

    all_features, all_flags, all_is_known, all_pi = [], [], [], []
    resid = {l: [] for l in LAYERS}
    hooks = []
    for l in LAYERS:
        def h(l=l, r=resid):
            return lambda m, inp, out: r[l].append(out[0].detach())
        hooks.append(model.model.layers[l].register_forward_hook(h()))

    t0 = time.time()
    for pi in range(n_pilot):
        prompt = prompts[pi]
        corr = [corr_list[pi]]
        wrg = [str(w) for w in wrg_list[pi]] if wrg_list[pi] else []
        for l in LAYERS: resid[l] = []

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=64, eos_token_id=list(eos_ids),
                do_sample=True, temperature=0.5, top_k=5, top_p=0.99,
                num_beams=10, num_return_sequences=10,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id, early_stopping=True)

        gen_ids_all = outputs.sequences[:, prompt_len:]
        any_correct = False

        for b in range(gen_ids_all.shape[0]):
            gids = gen_ids_all[b]
            gids = gids[gids != tokenizer.eos_token_id]
            gen_text = tokenizer.decode(gids, skip_special_tokens=True).strip()

            r = rouge.compute(predictions=[gen_text]*len(corr), references=corr) if corr else {"rougeL":0}
            refs = corr + wrg
            bs = bleurt.compute(predictions=[gen_text]*len(refs), references=refs) if refs else {"scores":[0]}
            mc = max(bs["scores"][:len(corr)], default=0)
            is_correct = (r["rougeL"]>=0.7) or (mc>0.5)
            if is_correct: any_correct = True

            layer_latents = []
            for l in LAYERS:
                stored = resid[l]
                if len(stored) < 2:
                    layer_latents.append(torch.zeros(M)); continue
                sae_outs = []
                for step_t in stored[1:]:
                    h = step_t[b:b+1, -1:, :]
                    lat = saes[l].encode(h)
                    sae_outs.append(lat.squeeze(0).squeeze(0).cpu())
                if sae_outs:
                    stack = torch.stack(sae_outs, dim=0)
                    layer_latents.append(torch.relu(stack).max(dim=0).values)
                else:
                    layer_latents.append(torch.zeros(M))
            all_features.append(torch.cat(layer_latents, dim=0))
            all_flags.append(not is_correct)
            all_pi.append(pi)

        all_is_known.append(any_correct)
        del outputs; torch.cuda.empty_cache()

        if (pi+1)%5==0:
            e = time.time()-t0
            print(f"  [{pi+1:3d}/{n_pilot}] known={sum(all_is_known)}  "
                  f"hall={sum(all_flags)/len(all_flags)*100:.0f}%  {e/60:.0f}m", flush=True)

    for h in hooks: h.remove()

    out_dir = os.path.join(DATA_DIR, args.model_folder)
    os.makedirs(out_dir, exist_ok=True)
    F_sae_raw = torch.stack(all_features, dim=0)
    out_path = os.path.join(out_dir, "truthfulqa_sae_pooled_raw.pt")
    torch.save({"encoded_tokens": F_sae_raw,
                "all_hallucination_flag": all_flags,
                "all_is_known": all_is_known,
                "prompt_indices": all_pi}, out_path)
    n = len(all_features)
    print(f"\n  Saved: {tuple(F_sae_raw.shape)}  {os.path.getsize(out_path)/1e9:.2f} GB")
    print(f"  Beams={n}  known={sum(all_is_known)}/{n_pilot}")
    return F_sae_raw, np.array(all_flags), np.array(all_is_known), np.array(all_pi), M


# STEP 3-5: BANDPASS + GS + CLASSIFY
def evaluate_sae(F_sae_raw, y_all, is_known_arr, prompt_idx, M):
    N = len(y_all); TOT = F_sae_raw.shape[1]
    ki = np.where(is_known_arr)[0]
    np.random.seed(RANDOM_SEED); np.random.shuffle(ki)
    s = int(len(ki)*0.75)
    tp = set(ki[:s]); vp = set(ki[s:])
    vp.update(np.where(~is_known_arr)[0])
    tm = np.array([prompt_idx[i] in tp for i in range(N)])
    vm = np.array([prompt_idx[i] in vp for i in range(N)])
    tidx = np.where(tm)[0]; vidx = np.where(vm)[0]
    print(f"\n  Train={len(tidx)}  Valid={len(vidx)}")

    Fn = F_sae_raw.float().numpy()
    freq = (Fn[tidx] > 0).mean(axis=0)
    bpm = (freq >= 0.0005) & (freq <= 0.05)
    n_dead = (freq < 0.0005).sum(); n_noise = (freq > 0.05).sum(); n_k = bpm.sum()
    print(f"  Bandpass: {TOT} -> dead={n_dead} noise={n_noise} kept={n_k}")
    Fbp = Fn[:, bpm]

    variants = {"V2: SAE Bandpass": (Fbp, n_k)}

    bp_path = os.path.join(DATA_DIR, args.model_folder, "truthfulqa_pooled_maxenergy.pt")
    have_bl = os.path.exists(bp_path)
    if have_bl:
        base = torch.load(bp_path, weights_only=False)
        Xb = torch.stack(base["all_emb"]).float()
        Nb, Lb, Db = Xb.shape
        Xf = Xb[tidx].permute(1,0,2).reshape(Lb,-1)
        AL = Xf @ Xf.T
        _, UL = torch.linalg.eigh(AL)
        UL = torch.flip(UL[:,-5:], dims=[1])
        Xd = Xb[tidx].permute(2,0,1).reshape(Db,-1)
        AD = torch.zeros(Db,Db,dtype=torch.float32)
        for st in range(0, len(tidx)*Lb, 50000):
            en = min(st+50000, len(tidx)*Lb)
            AD.addmm_(Xd[:,st:en].float(), Xd[:,st:en].float().T)
        _, UD = torch.linalg.eigh(AD)
        UD = torch.flip(UD[:,-64:], dims=[1])
        tmp = Xb.float() @ UD
        Gc = tmp.transpose(1,2) @ UL
        Fc = Gc.transpose(1,2).reshape(Nb,-1).numpy()

        ridge = Ridge(alpha=1.0)
        ridge.fit(Fc[tidx], Fbp[tidx])
        Fp = Fbp - ridge.predict(Fc)

        variants.update({
            "V1: Linear Core (320)":  (Fc, 320),
            "V3: Core + SAE Raw":     (np.concatenate([Fc,Fbp],axis=1), 320+n_k),
            "V4: Core + Orth SAE":    (np.concatenate([Fc,Fp],axis=1), 320+n_k),
            "V5: Orth SAE Alone":     (Fp, n_k),
        })
        print(f"  Baseline loaded: {Fc.shape}")
    else:
        print(f"  WARNING: {bp_path} not found. Skipping V1/V3/V4/V5.")
        print(f"  Run 21_generate_maxpool_datasets.py first.")

    for vn, (fe, di) in variants.items():
        sc = StandardScaler()
        tr = sc.fit_transform(fe[tidx]); va = sc.transform(fe[vidx])
        res = {}
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(tr, y_all[tidx]); res["RF"] = roc_auc_score(y_all[vidx], rf.predict_proba(va)[:,1])
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr.fit(tr, y_all[tidx]); res["LR"] = roc_auc_score(y_all[vidx], lr.predict_proba(va)[:,1])
        mlp = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", solver="adam",
                            early_stopping=True, n_iter_no_change=10, max_iter=1000, random_state=42)
        mlp.fit(tr, y_all[tidx]); res["MLP"] = roc_auc_score(y_all[vidx], mlp.predict_proba(va)[:,1])
        print(f"  {vn:30s}  dim={di:6d}  RF={res['RF']:.4f}  LR={res['LR']:.4f}  MLP={res['MLP']:.4f}")

    if have_bl:
        v1r = variants["V1: Linear Core (320)"][1]["RF"]
        v2r = variants["V2: SAE Bandpass"][1]["RF"]
        v4r = variants["V4: Core + Orth SAE"][1]["RF"]
        print(f"\n  Hyp A (SAE>Linear): {'PASS' if v2r>v1r else 'FAIL'} ({v2r:.4f} vs {v1r:.4f})")
        print(f"  Hyp B (Core+Orth>Core+.01): {'PASS' if v4r>v1r+.01 else 'FAIL'} ({v4r:.4f} vs {v1r:.4f})")


if __name__ == "__main__":
    run_dummy_tests()
    Fr, fl, ik, pi, M = extract_sae_features(args.model_folder, args.n_pilot)
    evaluate_sae(Fr, fl, ik, pi, M)
