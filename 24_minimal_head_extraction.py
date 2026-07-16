"""
24_minimal_head_extraction.py
===============================
Hook o_proj input across 9 reasoning layers on 5 prompts.
Extract first generated token per beam. Assert shapes.
"""

import gc, os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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

# -- Extract --
all_head_tensors = []
all_lookback = []

for pi, prompt in enumerate(prompts):
    print(f"\n  Prompt {pi}: {prompt[:60]}...")

    # Register hooks for head outputs AND attention weights
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

    # Generate 1 beam only (simple)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=32, eos_token_id=list(eos_ids),
            do_sample=True, temperature=0.7,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id)

    for h in hooks:
        h.remove()

    # Decode
    gen_ids = outputs.sequences[0, prompt_len:]
    gen_ids = gen_ids[gen_ids != tokenizer.eos_token_id]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"  Generated: {gen_text[:80]}...  ({len(gen_ids)} tokens)")

    # ── Assertions ──
    for l in LAYERS:
        stored = head_storage[l]
        assert len(stored) >= 2, f"Layer {l}: only {len(stored)} hook firings"
        assert stored[0].shape == (1, prompt_len, 4096), \
            f"Layer {l} prompt step shape: {stored[0].shape}"
        assert stored[1].shape == (1, 1, 4096), \
            f"Layer {l} gen step shape: {stored[1].shape}"
    print(f"  [PASS] All {len(LAYERS)} layers: prompt={stored[0].shape}, gen={stored[1].shape}")

    # Extract first generated token per layer (head tensor)
    layer_vecs = []
    for l in LAYERS:
        h = head_storage[l][1][0, 0, :].reshape(n_heads, head_dim)
        layer_vecs.append(h)
    head_tensor = torch.stack(layer_vecs, dim=0)  # (9, 32, 128)
    assert head_tensor.shape == (len(LAYERS), n_heads, head_dim)
    print(f"  [PASS] Head tensor: {tuple(head_tensor.shape)}")

    # ── Lookback ratios ──
    lookback_vecs = []
    for l in LAYERS:
        a_stored = attn_storage[l]
        if not a_stored or len(a_stored) < 2:
            lookback_vecs.append(torch.zeros(n_heads))
            continue
        # a_stored[1:] = generation steps: (1, n_heads, 1, S_total) where S_total grows
        ratios = []
        for s in a_stored[1:]:  # (1, n_heads, 1, S_step)
            at = s[0, :, 0, :]  # (n_heads, S_step)
            ctx_len = min(prompt_len, at.shape[-1])
            ctx_mass = at[:, :ctx_len].sum(dim=-1)
            tot_mass = at.sum(dim=-1) + 1e-9
            ratios.append(ctx_mass / tot_mass)
        lookback_vecs.append(torch.stack(ratios).mean(dim=0))  # (n_heads,)
    lookback = torch.stack(lookback_vecs, dim=0)  # (9, 32)
    assert lookback.shape == (len(LAYERS), n_heads), f"Lookback shape: {lookback.shape}"
    print(f"  [PASS] Lookback: {tuple(lookback.shape)}  range=[{lookback.min():.2f}, {lookback.max():.2f}]")

    all_head_tensors.append(head_tensor)
    all_lookback.append(lookback)

    del outputs, head_storage, attn_storage
    torch.cuda.empty_cache()

# ── Save extracted head tensors ──
out_dir = "../data/llama-3.1-8b-instruct"
os.makedirs(out_dir, exist_ok=True)
X_head = torch.stack(all_head_tensors, dim=0)  # (5, 9, 32, 128)
X_lookback = torch.stack(all_lookback, dim=0)    # (5, 9, 32)
torch.save({"all_emb": X_head, "all_lookback": X_lookback},
           os.path.join(out_dir, "truthfulqa_head_resolved_1beam_pilot.pt"))
print(f"\n[Step 1] Saved: head={tuple(X_head.shape)}  lookback={tuple(X_lookback.shape)}")

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
print(f"  [PASS] Tucker core saved.  Done.")
