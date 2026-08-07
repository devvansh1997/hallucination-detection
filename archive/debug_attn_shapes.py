"""
debug_attn_shapes.py — Inspection Script for Attention Hook Outputs
=====================================================================
Hooks one attention layer, runs a single generation, prints everything.

Run on cluster:  python debug_attn_shapes.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

model_id = cfg["models"][0]["id"]
print(f"Model: {model_id}")

model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="cuda",
    trust_remote_code=True, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

L = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // n_heads
print(f"Layers: {L}  Heads: {n_heads}  Head dim: {head_dim}")

# Hook o_proj to get per-head pre-W_O values
captured = {}

def hook_fn(module, input, output):
    """Hook the o_proj linear layer — input is un-projected head outputs."""
    print("\n=== O_PROJ HOOK FIRED ===")
    if isinstance(input, tuple):
        inp = input[0]
        print(f"  input shape: {tuple(inp.shape)}  dtype: {inp.dtype}")
        print(f"  reshaped to (B, S, {n_heads}, {head_dim}): would be "
              f"{tuple(inp.shape[:2]) + (n_heads, head_dim)}")
    if hasattr(output, 'shape'):
        print(f"  output shape: {tuple(output.shape)}")

hook = model.model.layers[20].self_attn.o_proj.register_forward_hook(hook_fn)

# Single generation
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print(f"\nPrompt tokens: {inputs.input_ids.shape[1]}")

with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=3,
        output_attentions=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id)

print("\nDone.")
hook.remove()
