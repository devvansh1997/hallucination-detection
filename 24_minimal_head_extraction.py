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

for pi, prompt in enumerate(prompts):
    print(f"\n  Prompt {pi}: {prompt[:60]}...")

    # Register hooks
    head_storage = {l: [] for l in LAYERS}
    hooks = []

    def make_hook(l):
        def h(m, inp, out):
            head_storage[l].append(inp[0].detach().cpu())
        return h

    for l in LAYERS:
        hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_hook(make_hook(l)))

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
        # stored[0] = prompt step: (1, prompt_len, 4096)
        assert stored[0].shape == (1, prompt_len, 4096), \
            f"Layer {l} prompt step shape: {stored[0].shape}"
        # stored[1] = first gen step: (1, 1, 4096)  [batch=1 since no beam search]
        assert stored[1].shape == (1, 1, 4096), \
            f"Layer {l} gen step shape: {stored[1].shape}"
    print(f"  [PASS] All {len(LAYERS)} layers: prompt={stored[0].shape}, gen={stored[1].shape}")

    # Extract first generated token per layer
    layer_vecs = []
    for l in LAYERS:
        # stored[1][0, 0, :] = (4096,) -> reshape to (32, 128)
        h = head_storage[l][1][0, 0, :].reshape(n_heads, head_dim)
        layer_vecs.append(h)
    head_tensor = torch.stack(layer_vecs, dim=0)  # (9, 32, 128)

    # -- Assert final shape --
    assert head_tensor.shape == (len(LAYERS), n_heads, head_dim), \
        f"Final shape: {head_tensor.shape}"
    print(f"  [PASS] Final tensor: {tuple(head_tensor.shape)}")

    all_head_tensors.append(head_tensor)

    del outputs, head_storage
    torch.cuda.empty_cache()

# ── Save extracted head tensors ──
out_dir = "../data/llama-3.1-8b-instruct"
os.makedirs(out_dir, exist_ok=True)
X_head = torch.stack(all_head_tensors, dim=0)  # (5, 9, 32, 128)
torch.save({"all_emb": X_head},
           os.path.join(out_dir, "truthfulqa_head_resolved_1beam_pilot.pt"))
print(f"\n[Step 1] Saved: {tuple(X_head.shape)}")

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

# Project: G = X ×1 U_L^T ×2 U_H^T ×3 U_D^T
G = X_head.float()
G = torch.tensordot(G, U_L, dims=([1], [0]))   # (N, R_L, H, HD)
G = G.permute(0, 2, 1, 3)                       # (N, H, R_L, HD)
G = torch.tensordot(G, U_H, dims=([1], [0]))   # (N, R_L, R_H, HD)
G = G.permute(0, 2, 1, 3)                       # (N, R_H, R_L, HD)
G = torch.tensordot(G, U_D, dims=([2], [0]))   # (N, R_H, R_L, R_D)
G = G.permute(0, 2, 1, 3)                       # (N, R_L, R_H, R_D)

F_tucker = G.reshape(N, -1)                     # (5, 640)

# ── Assertions ──
assert F_tucker.shape == (N, R_L * R_H * R_D), \
    f"Tucker core shape: {F_tucker.shape}, expected ({N}, {R_L*R_H*R_D})"
assert not torch.isnan(F_tucker).any(), "NaN in Tucker core"
assert not torch.isinf(F_tucker).any(), "Inf in Tucker core"
print(f"  [PASS] Tucker core: {tuple(F_tucker.shape)}  (no NaN/Inf)")

# ── Save ──
torch.save({"all_emb": F_tucker, "U_L": U_L, "U_H": U_H, "U_D": U_D},
           os.path.join(out_dir, "truthfulqa_tucker_pilot.pt"))
print(f"  [PASS] Tucker core saved.  Done.")
