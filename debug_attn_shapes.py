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

# Hook one layer (layer 20 — middle of reasoning window)
captured = {}

def hook_fn(module, input, output):
    """Capture everything from the attention module."""
    print("\n=== ATTENTION HOOK FIRED ===")
    # input is tuple of (hidden_states, attention_mask, position_ids, ...)
    if isinstance(input, tuple):
        for i, inp in enumerate(input):
            if inp is not None and hasattr(inp, 'shape'):
                print(f"  input[{i}] shape: {tuple(inp.shape)}")
    # output is typically a tuple
    if isinstance(output, tuple):
        print(f"  output is tuple of length {len(output)}")
        for i, out in enumerate(output):
            if out is None:
                print(f"  output[{i}]: None")
            elif hasattr(out, 'shape'):
                print(f"  output[{i}] shape: {tuple(out.shape)}  dtype: {out.dtype}")
            elif isinstance(out, tuple):
                print(f"  output[{i}] is tuple of length {len(out)}")
                for j, o2 in enumerate(out):
                    if o2 is not None and hasattr(o2, 'shape'):
                        print(f"    output[{i}][{j}] shape: {tuple(o2.shape)}")
            else:
                print(f"  output[{i}] type: {type(out).__name__}")

    # Also try to access internal attributes
    if hasattr(module, 'o_proj'):
        print(f"\n  module.o_proj exists")
    captured['done'] = True

hook = model.model.layers[20].self_attn.register_forward_hook(hook_fn)

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
