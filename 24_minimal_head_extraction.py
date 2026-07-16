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

# ── Save ──
out_dir = "../data/llama-3.1-8b-instruct"
os.makedirs(out_dir, exist_ok=True)
out = torch.stack(all_head_tensors, dim=0)
print(f"\nSaved: {out.shape}  ({N_PROMPTS} prompts, {len(LAYERS)} layers, {n_heads} heads, {head_dim} dim)")

out_path = os.path.join(out_dir, "truthfulqa_head_resolved_1beam_pilot.pt")
torch.save({"all_emb": out}, out_path)
print(f"  {out_path}")
