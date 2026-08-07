"""
25_full_head_pipeline.py — Full Head-Resolved Extraction + Evaluation
========================================================================
Hooks all attention layers, 10-beam search, BLEURT/ROUGE judge,
head-resolved Tucker compression, Gram-Schmidt, multi-variant classifiers.

Usage:
  python 25_full_head_pipeline.py --model_folder llama-3.1-8b-instruct --dataset truthfulqa
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
R_L, R_H, R_D, RANDOM_SEED = 5, 8, 16, 42
LAYERS = list(range(15, 24))

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--n_pilot", type=int, default=0)
args = parser.parse_args()

model_id = None
for m in cfg["models"]:
    if m["folder"] == args.model_folder:
        model_id = m["id"]; break
ds_cfg = None
for d in cfg["datasets"]:
    if d["name"] == args.dataset:
        ds_cfg = d; break

device = torch.device("cuda")

# -- Load dataset --
from datasets import load_dataset
ds = load_dataset(ds_cfg["hf_path"], ds_cfg["hf_config"], split="validation")
template = ds_cfg["prompt_template"]

samples = []
if args.dataset == "truthfulqa":
    for ex in ds:
        samples.append({
            "prompt": str(ex["question"]),
            "correct": [str(ex["best_answer"])],
            "wrong": [str(a) for a in ex["incorrect_answers"]],
        })
elif args.dataset == "tydiqa_gp":
    for ex in ds:
        if ex.get("language", "english") != "english":
            continue
        ctx = ex["context"][0] if isinstance(ex["context"], list) else ex["context"]
        samples.append({
            "prompt": template.format(context=str(ctx), question=ex["question"]),
            "correct": [str(a) for a in ex["answers"]["text"] if a],
            "wrong": [],
        })
else:
    raise ValueError("Unsupported: " + args.dataset)

if args.n_pilot > 0:
    import random; random.seed(42); random.shuffle(samples)
    samples = samples[:args.n_pilot]

n_prompts = len(samples)
print(f"Dataset: {args.dataset}, prompts: {n_prompts}")

# -- Load model --
print(f"Model: {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map=device,
    trust_remote_code=True, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model.eval()
n_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // n_heads
print(f"  Layers: {model.config.num_hidden_layers}  Heads: {n_heads}  Head dim: {head_dim}")

# -- Metrics --
import evaluate
rouge = evaluate.load("rouge")
print("  ROUGE ready")
bleurt = evaluate.load("bleurt", config_name="BLEURT-20")
print("  BLEURT ready")

# -- EOS --
eos_ids = {tokenizer.eos_token_id}
for s in [".", "!", "?", "\n"]:
    for tok in tokenizer.encode(s, add_special_tokens=False):
        eos_ids.add(tok)

# -- Extract --
all_head, all_lb, all_flags, all_is_known, all_pi = [], [], [], [], []
t0_total = time.time()

for pi, sample in enumerate(samples):
    prompt_text = sample["prompt"]
    correct = sample["correct"]
    wrong = sample["wrong"]

    head_storage, attn_storage = {l: [] for l in LAYERS}, {l: [] for l in LAYERS}
    hooks = []

    for l in LAYERS:
        def make_head_h(l):
            return lambda m, inp, out: head_storage[l].append(inp[0].detach().cpu())
        def make_attn_h(l):
            return lambda m, inp, out: (
                attn_storage[l].append(out[1].detach().cpu())
                if isinstance(out, tuple) and len(out) > 1 and out[1] is not None else None)
        hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_hook(make_head_h(l)))
        hooks.append(model.model.layers[l].self_attn.register_forward_hook(make_attn_h(l)))

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=64, eos_token_id=list(eos_ids),
            do_sample=True, temperature=0.5, top_k=5, top_p=0.99,
            num_beams=10, num_return_sequences=10,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id, early_stopping=True)

    for h in hooks: h.remove()

    gen_ids_all = outputs.sequences[:, prompt_len:]
    any_correct = False

    for b in range(gen_ids_all.shape[0]):
        gids = gen_ids_all[b]
        gids = gids[gids != tokenizer.eos_token_id]
        gen_text = tokenizer.decode(gids, skip_special_tokens=True).strip()

        r = rouge.compute(predictions=[gen_text]*len(correct), references=correct) if correct else {"rougeL": 0.0}
        all_refs = correct + wrong
        bs = bleurt.compute(predictions=[gen_text]*len(all_refs), references=all_refs) if all_refs else {"scores": [0]}
        max_correct_b = max(bs["scores"][:len(correct)], default=0)
        is_correct = (r["rougeL"] >= 0.7) or (max_correct_b > 0.5)
        if is_correct: any_correct = True

        # Head tensor
        layer_vecs = []
        for l in LAYERS:
            stored = head_storage.get(l, [])
            if len(stored) > 1:
                layer_vecs.append(stored[1][b, 0, :].reshape(n_heads, head_dim))
            else:
                layer_vecs.append(torch.zeros(n_heads, head_dim))
        all_head.append(torch.stack(layer_vecs, dim=0))

        # Lookback
        lookback_vecs = []
        for l in LAYERS:
            a_stored = attn_storage.get(l, [])
            if len(a_stored) < 2:
                lookback_vecs.append(torch.zeros(n_heads))
            else:
                ratios = []
                for s in a_stored[1:]:
                    at = s[b, :, 0, :]
                    ctx_m = at[:, :min(prompt_len, at.shape[-1])].sum(dim=-1)
                    tot_m = at.sum(dim=-1) + 1e-9
                    ratios.append(ctx_m / tot_m)
                lookback_vecs.append(torch.stack(ratios).mean(dim=0))
        all_lb.append(torch.stack(lookback_vecs, dim=0))

        all_flags.append(not is_correct)
        all_pi.append(pi)

    all_is_known.append(any_correct)
    del outputs, head_storage, attn_storage
    torch.cuda.empty_cache()

    if (pi + 1) % 50 == 0 or pi == 0:
        elapsed = time.time() - t0_total
        eta = elapsed / (pi + 1) * n_prompts - elapsed
        k = sum(all_is_known)
        print(f"  [{pi+1:4d}/{n_prompts}] known={k}  "
              f"hall={sum(all_flags)/len(all_flags)*100:.0f}%  "
              f"elapsed={elapsed/60:.0f}m  eta={eta/60:.0f}m", flush=True)

# -- Save --
out_dir = os.path.join(DATA_DIR, args.model_folder)
os.makedirs(out_dir, exist_ok=True)
X_head = torch.stack(all_head, dim=0)
X_lb   = torch.stack(all_lb, dim=0)
out_path = os.path.join(out_dir, f"{args.dataset}_head_resolved.pt")
torch.save({"all_emb": X_head, "all_lookback": X_lb,
            "all_hallucination_flag": all_flags,
            "all_is_known": all_is_known,
            "prompt_indices": all_pi,
            "metadata": {"n_heads": n_heads, "head_dim": head_dim,
                         "layers": LAYERS}}, out_path)
n_beams = len(all_head)
print(f"\nSaved: head={tuple(X_head.shape)}  lookback={tuple(X_lb.shape)}  "
      f"beams={n_beams}  known={sum(all_is_known)}/{n_prompts}  "
      f"{os.path.getsize(out_path)/1e9:.1f} GB")

# -- Tucker helpers (must be before use) --
def compute_ul(X, dim_size, rank, dim):
    X_u = X.permute(*([dim] + [i for i in range(4) if i != dim])).reshape(dim_size, -1).float()
    A = X_u @ X_u.T
    _, U = torch.linalg.eigh(A)
    return torch.flip(U[:, -rank:], dims=[1])

def compute_ul_ud(X_train, L, D, rl, rd):
    X_f = X_train.permute(1, 0, 2).reshape(L, -1).float()
    A_L = X_f @ X_f.T
    _, U_L = torch.linalg.eigh(A_L)
    X_d = X_train.permute(2, 0, 1).reshape(D, -1)
    A_D = torch.zeros(D, D, dtype=torch.float32)
    for start in range(0, X_train.shape[0]*L, 50000):
        end = min(start+50000, X_train.shape[0]*L)
        A_D.addmm_(X_d[:, start:end].float(), X_d[:, start:end].float().T)
    _, U_D = torch.linalg.eigh(A_D)
    return torch.flip(U_L[:, -rl:], dims=[1]), torch.flip(U_D[:, -rd:], dims=[1])

# -- Tucker --
print(f"\n[Eval] Tucker + GS + Classification")
y_all = np.array(all_flags)
is_known_arr = np.array(all_is_known)
prompt_idx = np.array(all_pi)
N_beams, L9, H, HD = X_head.shape
N_PROMPTS = n_prompts

# Tucker ranks
r_l, r_h, r_d = 5, 8, 16

# HARP split
known_idx = np.where(is_known_arr)[0]
np.random.seed(RANDOM_SEED); np.random.shuffle(known_idx)
s = int(len(known_idx) * 0.75)
tp = set(known_idx[:s]); vp = set(known_idx[s:])
vp.update(np.where(~is_known_arr)[0])
t_mask = np.array([prompt_idx[i] in tp for i in range(N_beams)])
v_mask = np.array([prompt_idx[i] in vp for i in range(N_beams)])
t_idx = np.where(t_mask)[0]; v_idx = np.where(v_mask)[0]

# Tucker on train only
X_t = X_head[t_idx]
G = torch.tensordot(X_t.float(), compute_ul(X_t, L9, r_l, dim=1), dims=([1],[0]))
G = G.permute(0,1,3,2)
G = torch.tensordot(G, compute_ul(X_t, H, r_h, dim=2), dims=([1],[0]))
G = G.permute(0,1,3,2)
G = torch.tensordot(G, compute_ul(X_t, HD, r_d, dim=3), dims=([3],[0]))

U_L_full = compute_ul(X_t, L9, r_l, dim=1)
U_H_full = compute_ul(X_t, H, r_h, dim=2)
U_D_full = compute_ul(X_t, HD, r_d, dim=3)

# Project all
def tucker_project(X):
    G = torch.tensordot(X.float(), U_L_full, dims=([1],[0]))
    G = G.permute(0,1,3,2)
    G = torch.tensordot(G, U_H_full, dims=([1],[0]))
    G = G.permute(0,1,3,2)
    G = torch.tensordot(G, U_D_full, dims=([3],[0]))
    return G.reshape(X.shape[0], -1)

F_tucker = tucker_project(X_head).numpy()
F_lookback_flat = X_lb.float().reshape(N_beams, -1).numpy()

# Phase 1 HOSVD baseline
X_pooled = X_head.reshape(N_beams, L9, H*HD).float()
U_L_p1, U_D_p1 = compute_ul_ud(X_pooled[t_idx], L9, H*HD, 5, 64)
tmp = X_pooled @ U_D_p1
G_p1 = tmp.transpose(1,2) @ U_L_p1
F_core = G_p1.transpose(1,2).reshape(N_beams, -1).numpy()

# Gram-Schmidt
F_unmixed = np.concatenate([F_tucker, F_lookback_flat], axis=1)
ridge = Ridge(alpha=1.0)
ridge.fit(F_core[t_idx], F_unmixed[t_idx])
F_perp = F_unmixed - ridge.predict(F_core)

# Variants
variants = {
    "V1: Core (320)":              F_core,
    "V2: Tucker (640)":            F_tucker,
    "V3: Lookback (288)":          F_lookback_flat,
    "V4: Core + Orth (1248)":      np.concatenate([F_core, F_perp], axis=1),
}

print(f"  Train={len(t_idx)}  Valid={len(v_idx)}")
results = {}
for vname, feats in variants.items():
    scaler = StandardScaler()
    tr = scaler.fit_transform(feats[t_idx])
    va = scaler.transform(feats[v_idx])
    res = {}

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(tr, y_all[t_idx]); res["RF"] = roc_auc_score(y_all[v_idx], rf.predict_proba(va)[:, 1])

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(tr, y_all[t_idx]); res["LR"] = roc_auc_score(y_all[v_idx], lr.predict_proba(va)[:, 1])

    mlp = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", solver="adam",
                        early_stopping=True, n_iter_no_change=10, max_iter=1000, random_state=42)
    mlp.fit(tr, y_all[t_idx]); res["MLP"] = roc_auc_score(y_all[v_idx], mlp.predict_proba(va)[:, 1])
    results[vname] = res
    print(f"  {vname:30s}  RF={res['RF']:.4f}  LR={res['LR']:.4f}  MLP={res['MLP']:.4f}")


def compute_ul(X, dim_size, rank, dim):
    X_u = X.permute(*([dim] + [i for i in range(4) if i != dim])).reshape(dim_size, -1).float()
    A = X_u @ X_u.T
    _, U = torch.linalg.eigh(A)
    return torch.flip(U[:, -rank:], dims=[1])

def compute_ul_ud(X_train, L, D, rl, rd):
    X_f = X_train.permute(1, 0, 2).reshape(L, -1).float()
    A_L = X_f @ X_f.T
    _, U_L = torch.linalg.eigh(A_L)
    X_d = X_train.permute(2, 0, 1).reshape(D, -1)
    A_D = torch.zeros(D, D, dtype=torch.float32)
    for start in range(0, X_train.shape[0]*L, 50000):
        end = min(start+50000, X_train.shape[0]*L)
        A_D.addmm_(X_d[:, start:end].float(), X_d[:, start:end].float().T)
    _, U_D = torch.linalg.eigh(A_D)
    return torch.flip(U_L[:, -rl:], dims=[1]), torch.flip(U_D[:, -rd:], dims=[1])

print(f"\n{'='*60}")
print(f"  DONE — {args.dataset} / {args.model_folder}")
print(f"{'='*60}")
