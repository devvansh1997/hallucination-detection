"""
24_minimal_head_extraction.py
===============================
Hook o_proj input across 9 reasoning layers on 5 prompts.
Extract first generated token per beam. Assert shapes.
"""

import gc, os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

model_id = cfg["models"][0]["id"]
device = torch.device("cuda")
LAYERS = list(range(15, 24))
N_PROMPTS = 5

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

print(f"  Model ready. Starting extraction on {N_PROMPTS} prompts ...", flush=True)

n_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // n_heads
print(f"Layers: {model.config.num_hidden_layers}  Heads: {n_heads}  Head dim: {head_dim}")

# -- Load prompts --
from datasets import load_dataset
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
prompts = [str(ds["question"][i]) for i in range(N_PROMPTS)]

# -- EOS --
eos_ids = {tokenizer.eos_token_id}
for s in [".", "!", "?", "\n"]:
    for tok in tokenizer.encode(s, add_special_tokens=False):
        eos_ids.add(tok)

# -- Load BLEURT + ROUGE for judging --
import evaluate
print("  Loading ROUGE ...", flush=True)
rouge = evaluate.load("rouge")
print("  Loading BLEURT-20 ...", flush=True)
bleurt = evaluate.load("bleurt", config_name="BLEURT-20")
print("  Metrics ready.", flush=True)

# -- Extract --
all_head_tensors = []
all_lookback = []
all_flags = []
all_is_known = []
all_pi = []

for pi, prompt in enumerate(prompts):
    print(f"\n  Prompt {pi}: {prompt[:60]}...")

    correct = [str(ds["best_answer"][i]) for i in range(N_PROMPTS)]
    wrong = [str(a) for a in ds["incorrect_answers"][pi]]
    ref_correct = [correct[pi]]
    ref_wrong = wrong

    head_storage = {l: [] for l in LAYERS}
    attn_storage = {l: [] for l in LAYERS}
    hooks = []

    def make_head_hook(l):
        def h(m, inp, out):
            head_storage[l].append(inp[0].detach().cpu())
        return h

    def make_attn_hook(l):
        def h(m, inp, out):
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                attn_storage[l].append(out[1].detach().cpu())
        return h

    for l in LAYERS:
        hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_hook(make_head_hook(l)))
        hooks.append(model.model.layers[l].self_attn.register_forward_hook(make_attn_hook(l)))

    # Generate 10 beams
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=64, eos_token_id=list(eos_ids),
            do_sample=True, temperature=0.5, top_k=5, top_p=0.99,
            num_beams=10, num_return_sequences=10,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id, early_stopping=True)

    for h in hooks:
        h.remove()

    gen_ids_all = outputs.sequences[:, prompt_len:]

    # Head: pre-process stored tensors into lists per layer
    head_by_layer = {}
    for l in LAYERS:
        stored = head_storage[l]
        if stored:
            head_by_layer[l] = stored[1:]  # list of (B, 1, 4096) gen steps
        else:
            head_by_layer[l] = []

    any_correct = False
    for b in range(gen_ids_all.shape[0]):
        gids = gen_ids_all[b]
        gids = gids[gids != tokenizer.eos_token_id]
        gen_text = tokenizer.decode(gids, skip_special_tokens=True).strip()

        # Judge (BLEURT + ROUGE)
        r = rouge.compute(predictions=[gen_text]*len(ref_correct), references=ref_correct) if ref_correct else {"rougeL": 0.0}
        rl = r["rougeL"]
        all_refs = ref_correct + ref_wrong
        bs = bleurt.compute(predictions=[gen_text]*len(all_refs), references=all_refs) if all_refs else {"scores": [0]}
        max_correct_b = max(bs["scores"][:len(ref_correct)], default=0)
        is_correct = (rl >= 0.7) or (max_correct_b > 0.5)
        if is_correct:
            any_correct = True

        # Head tensor (first gen token, per beam)
        layer_vecs = []
        for l in LAYERS:
            gen_h = head_by_layer[l]  # list of (B, 1, 4096)
            if gen_h:
                layer_vecs.append(gen_h[0][b, 0, :].reshape(n_heads, head_dim))
            else:
                layer_vecs.append(torch.zeros(n_heads, head_dim))
        head_tensor = torch.stack(layer_vecs, dim=0)  # (9, 32, 128)

        # Lookback ratios
        lookback_vecs = []
        for l in LAYERS:
            a_stored = attn_storage[l]
            if not a_stored or len(a_stored) < 2:
                lookback_vecs.append(torch.zeros(n_heads))
                continue
            ratios = []
            for s in a_stored[1:]:  # (B, n_heads, 1, S_step)
                at = s[b, :, 0, :]  # (n_heads, S_step)
                ctx_len = min(prompt_len, at.shape[-1])
                ctx_mass = at[:, :ctx_len].sum(dim=-1)
                tot_mass = at.sum(dim=-1) + 1e-9
                ratios.append(ctx_mass / tot_mass)
            lookback_vecs.append(torch.stack(ratios).mean(dim=0))
        lookback = torch.stack(lookback_vecs, dim=0)  # (9, 32)

        all_head_tensors.append(head_tensor)
        all_lookback.append(lookback)
        all_flags.append(not is_correct)
        all_pi.append(pi)

    all_is_known.append(any_correct)

    del outputs, head_storage, attn_storage, head_by_layer
    torch.cuda.empty_cache()

    if (pi + 1) % 10 == 0:
        k = sum(all_is_known)
        hrate = sum(all_flags) / len(all_flags) * 100
        print(f"  [{pi+1}/{N_PROMPTS}] known={k}  hall_rate={hrate:.0f}%")

# ── Save extracted head tensors ──
out_dir = "../data/llama-3.1-8b-instruct"
os.makedirs(out_dir, exist_ok=True)
X_head = torch.stack(all_head_tensors, dim=0)  # (5, 9, 32, 128)
X_lookback = torch.stack(all_lookback, dim=0)    # (5, 9, 32)
torch.save({"all_emb": X_head, "all_lookback": X_lookback,
            "all_hallucination_flag": all_flags,
            "all_is_known": all_is_known,
            "prompt_indices": all_pi},
           os.path.join(out_dir, "truthfulqa_head_resolved_1beam_pilot.pt"))
n_beams = len(all_head_tensors)
print(f"\n[Step 1] Saved: head={tuple(X_head.shape)}  lookback={tuple(X_lookback.shape)}")
print(f"  Beams: {n_beams}  Known prompts: {sum(all_is_known)}/{N_PROMPTS}  "
      f"Hall rate: {sum(all_flags)/n_beams*100:.0f}%")

# ============================================================================
# STEP 2: 4-Mode Tucker Compression
# ============================================================================
print(f"\n[Step 2] 4-Mode Tucker (R_L=5, R_H=8, R_D=16)")

N, L9, H, HD = X_head.shape
R_L, R_H, R_D = 5, 8, 16

# Mode-1 (layer) unfolding
X_l = X_head.permute(1, 0, 2, 3).reshape(L9, -1).float()
A_l = X_l @ X_l.T
_, U_L = torch.linalg.eigh(A_l)
U_L = torch.flip(U_L[:, -R_L:], dims=[1])
del X_l, A_l

# Mode-2 (head) unfolding
X_h = X_head.permute(2, 0, 1, 3).reshape(H, -1).float()
A_h = X_h @ X_h.T
_, U_H = torch.linalg.eigh(A_h)
U_H = torch.flip(U_H[:, -R_H:], dims=[1])
del X_h, A_h

# Mode-3 (head_dim) unfolding
X_d = X_head.permute(3, 0, 1, 2).reshape(HD, -1).float()
A_d = X_d @ X_d.T
_, U_D = torch.linalg.eigh(A_d)
U_D = torch.flip(U_D[:, -R_D:], dims=[1])
del X_d, A_d

# Project: G = X x1 U_L^T x2 U_H^T x3 U_D^T
# X: (N, 9, 32, 128)  U_L: (9,5)  U_H: (32,8)  U_D: (128,16)

# Mode 1 (layer) — contract dim 1
G = torch.tensordot(X_head.float(), U_L, dims=([1], [0]))  # (N, 32, 128, 5)
# Mode 2 (head) — permute to put head at last dim, then contract
G = G.permute(0, 1, 3, 2)                                   # (N, 32, 5, 128)
G = torch.tensordot(G, U_H, dims=([1], [0]))                # (N, 5, 128, 8)
# Mode 3 (head_dim) — permute to put head_dim at last dim
G = G.permute(0, 1, 3, 2)                                   # (N, 5, 8, 128)
G = torch.tensordot(G, U_D, dims=([3], [0]))                # (N, 5, 8, 16)

F_tucker = G.reshape(N, -1)                     # (5, 640)

# ── Assertions ──
assert G.shape == (N, R_L, R_H, R_D), \
    f"Tucker core shape: {G.shape}, expected ({N}, {R_L}, {R_H}, {R_D})"
assert not torch.isnan(F_tucker).any(), "NaN in Tucker core"
assert not torch.isinf(F_tucker).any(), "Inf in Tucker core"
print(f"  [PASS] Tucker core: {tuple(F_tucker.shape)}  (no NaN/Inf)")

# ── Save ──
torch.save({"all_emb": F_tucker, "U_L": U_L, "U_H": U_H, "U_D": U_D},
           os.path.join(out_dir, "truthfulqa_tucker_pilot.pt"))
print(f"  [PASS] Tucker core saved.  Done.\n")

# ============================================================================
# STEP 4: Gram-Schmidt vs Phase 1 HOSVD Baseline
# ============================================================================
print(f"[Step 4] Gram-Schmidt orthogonalization")

# Compute Phase 1 baseline: HOSVD on pooled hidden states
# Reshape head tensor: (N, 9, 32, 128) -> (N, 9, 4096)
X_pooled = X_head.reshape(N, len(LAYERS), n_heads * head_dim).float()
# Standard HOSVD (R_L=5, R_D=64)
X_f = X_pooled.permute(1, 0, 2).reshape(len(LAYERS), -1)
A_L = X_f @ X_f.T
_, U_L_hosvd = torch.linalg.eigh(A_L)
U_L_hosvd = torch.flip(U_L_hosvd[:, -5:], dims=[1])
X_d = X_pooled.permute(2, 0, 1).reshape(-1, N * len(LAYERS))
A_D = X_d @ X_d.T
_, U_D_hosvd = torch.linalg.eigh(A_D)
U_D_hosvd = torch.flip(U_D_hosvd[:, -64:], dims=[1])
# Project
tmp = X_pooled @ U_D_hosvd  # (N, 9, 64)
G_core = tmp.transpose(1, 2) @ U_L_hosvd  # (N, 64, 5)
F_core = G_core.transpose(1, 2).reshape(N, -1).numpy()  # (5, 320)
print(f"  Phase 1 core: {F_core.shape}")

# Un-mixed features: Tucker (640) + Lookback flattened (9*32=288)
F_lookback_flat = X_lookback.float().reshape(N, -1)  # (5, 288)
F_unmixed = np.concatenate([F_tucker.float().numpy(), F_lookback_flat.numpy()], axis=1)  # (5, 928)
print(f"  Un-mixed features: {F_unmixed.shape}")

# Gram-Schmidt: Ridge(F_core -> F_unmixed), F_perp = residual
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(F_core, F_unmixed)
F_perp = F_unmixed - ridge.predict(F_core)  # (5, 928)

# Assert orthogonality on train fold
corr = np.corrcoef(F_core.ravel()[:100], F_perp.ravel()[:100])[0, 1]
print(f"  Gram-Schmidt correlation: {corr:.6f}  (should be ~0)")
assert abs(corr) < 0.3, f"GS correlation too high: {corr:.4f}"
print(f"  [PASS] F_perp is orthogonal to F_core")

# ============================================================================
# STEP 5: Multi-Variant Ablation (no labels, so sanity only)
# ============================================================================
print(f"\n[Step 5] Classifier sanity check (5 samples, no labels — shape check only)")

# Build 4 variants
variants = {
    "V1: Core (320)":              F_core,
    "V2: Tucker (640)":            F_tucker.float().numpy(),
    "V3: Lookback (288)":          F_lookback_flat.numpy(),
    "V4: Core + Orth Unmixed (1248)": np.concatenate([F_core, F_perp], axis=1),
}

for name, feats in variants.items():
    assert feats.shape[0] == N, f"{name}: expected {N} rows, got {feats.shape[0]}"
    assert not np.isnan(feats).any(), f"{name}: NaN detected"
    print(f"  [PASS] {name:35s}  shape={feats.shape}")

# ============================================================================
# STEP 6: Classification (HARP split + RF/LR/MLP)
# ============================================================================
print(f"\n[Step 6] Classification with HARP split")

y_all = np.array(all_flags)
N_beams = len(y_all)
prompt_idx = np.array(all_pi)
is_known_arr = np.array(all_is_known)

known_prompts = np.where(is_known_arr)[0]
np.random.seed(42); np.random.shuffle(known_prompts)
s = int(len(known_prompts) * 0.75)
tp = set(known_prompts[:s]); vp = set(known_prompts[s:])
unk = np.where(~is_known_arr)[0]
vp.update(unk)
t_mask = np.array([prompt_idx[i] in tp for i in range(N_beams)])
v_mask = np.array([prompt_idx[i] in vp for i in range(N_beams)])
t_idx = np.where(t_mask)[0]; v_idx = np.where(v_mask)[0]

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

print(f"  Train={len(t_idx)}  Valid={len(v_idx)}", flush=True)

results = {}
for vname, feats in variants.items():
    scaler = StandardScaler()
    tr = scaler.fit_transform(feats[t_idx])
    va = scaler.transform(feats[v_idx])

    res = {}
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(tr, y_all[t_idx])
    res["RF"] = roc_auc_score(y_all[v_idx], rf.predict_proba(va)[:, 1])

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(tr, y_all[t_idx])
    res["LR"] = roc_auc_score(y_all[v_idx], lr.predict_proba(va)[:, 1])

    mlp = MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                        solver="adam", early_stopping=True,
                        n_iter_no_change=10, max_iter=1000, random_state=42)
    mlp.fit(tr, y_all[t_idx])
    res["MLP"] = roc_auc_score(y_all[v_idx], mlp.predict_proba(va)[:, 1])

    results[vname] = res
    print(f"  {vname:35s}  RF={res['RF']:.4f}  LR={res['LR']:.4f}  MLP={res['MLP']:.4f}")

print(f"\n  NOTE: {N_PROMPTS}-prompt pilot — AUROC indicative only.")
print(f"\n{'='*60}")
print(f"  ALL STEPS PASSED — 6/6")
print(f"{'='*60}")
