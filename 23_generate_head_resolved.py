"""
23_generate_head_resolved.py — Head-Resolved Tensor Extraction
===============================================================
Hooks o_proj input (pre-W_O, per-head) and attention weights across
layers 15-23. Applies signed absolute extremum pooling + lookback
ratios. Saves compact dataset for downstream Tucker evaluation.

Usage:
  python 23_generate_head_resolved.py --model_folder llama-3.1-8b-instruct --dataset truthfulqa
"""

import argparse, gc, os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import torch, yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--n_pilot", type=int, default=0,
                    help="Limit prompts (0 = full dataset)")
args = parser.parse_args()

model_id = None
for m in cfg["models"]:
    if m["folder"] == args.model_folder:
        model_id = m["id"]; break
if model_id is None:
    raise ValueError("Unknown model folder: " + args.model_folder)

ds_cfg = None
for d in cfg["datasets"]:
    if d["name"] == args.dataset:
        ds_cfg = d; break
if ds_cfg is None:
    raise ValueError("Unknown dataset: " + args.dataset)

device = torch.device("cuda")

# -- Load dataset --
from datasets import load_dataset
name = ds_cfg["name"]
path = ds_cfg["hf_path"]
config = ds_cfg["hf_config"]
template = ds_cfg["prompt_template"]
ds = load_dataset(path, config, split="validation")

samples = []
if name == "truthfulqa":
    for ex in ds:
        samples.append({
            "prompt_text": template.format(question=ex["question"]),
            "correct_answers": [str(ex["best_answer"])],
            "incorrect_answers": [str(a) for a in ex["incorrect_answers"]],
        })
elif name == "tydiqa_gp":
    for ex in ds:
        if ex.get("language", "english") != "english":
            continue
        ctx = ex["context"][0] if isinstance(ex["context"], list) else ex["context"]
        samples.append({
            "prompt_text": template.format(context=str(ctx), question=ex["question"]),
            "correct_answers": [str(a) for a in ex["answers"]["text"] if a],
            "incorrect_answers": [],
        })
else:
    raise ValueError("Unsupported: " + name)

if args.n_pilot > 0:
    import random; random.seed(42); random.shuffle(samples)
    samples = samples[:args.n_pilot]

n_prompts = len(samples)
print(f"Dataset: {name}, prompts: {n_prompts}")

# -- Load model --
print(f"Loading: {model_id}")
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
W_START, W_END = 0, model.config.num_hidden_layers
print(f"  Heads: {n_heads}  Head dim: {head_dim}  Layers: {W_END}")

# -- Metrics --
import evaluate
rouge = evaluate.load("rouge")
bleurt = evaluate.load("bleurt", config_name="BLEURT-20")

# -- EOS --
eos_ids = {tokenizer.eos_token_id}
for s in [".", "!", "?", ".\n", "!\n", "?\n", "\n", "\n\n"]:
    eos_ids.update(tokenizer.encode(s, add_special_tokens=False))
    eos_ids.update(tokenizer.encode("Yes" + s, add_special_tokens=False)[1:])

gen_cfg = cfg["generation"]

all_head, all_lookback, all_flags, all_is_known, all_pi = [], [], [], [], []

for pi, sample in enumerate(tqdm(samples, desc=f"  {name}")):
    prompt_text = sample["prompt_text"]
    correct = sample["correct_answers"]
    wrong = sample["incorrect_answers"]

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    # Per-prompt storage for hooks
    head_storage = {l: [] for l in range(W_START, W_END)}
    attn_storage = {l: [] for l in range(W_START, W_END)}
    hooks = []

    def make_o_proj_hook(l):
        def h(m, inp, out):
            # inp[0]: (1, S, 4096) — un-projected head outputs
            head_storage[l].append(inp[0].detach().cpu())
        return h

    def make_attn_hook(l):
        def h(m, inp, out):
            # out[1]: (1, 32, S, S_total) — attention weights
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                attn_storage[l].append(out[1].detach().cpu())
        return h

    for l in range(W_START, W_END):
        hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_hook(make_o_proj_hook(l)))
        hooks.append(model.model.layers[l].self_attn.register_forward_hook(make_attn_hook(l)))

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=gen_cfg["max_new_tokens"],
            eos_token_id=list(eos_ids),
            do_sample=gen_cfg["do_sample"], temperature=gen_cfg["temperature"],
            top_k=gen_cfg["top_k"], top_p=gen_cfg["top_p"],
            num_beams=gen_cfg["num_beams"],
            num_return_sequences=gen_cfg["num_return_sequences"],
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id, early_stopping=True)

    for h in hooks:
        h.remove()

    gen_ids_all = outputs.sequences[:, prompt_len:]

    any_correct = False
    for b in range(gen_ids_all.shape[0]):
        gids = gen_ids_all[b]
        gids = gids[gids != tokenizer.eos_token_id]
        gen_text = tokenizer.decode(gids, skip_special_tokens=True).strip()

        # Judge
        r = rouge.compute(predictions=[gen_text] * len(correct), references=correct) if correct else {"rougeL": 0.0}
        rl = r["rougeL"]
        all_refs = correct + wrong
        bs = bleurt.compute(predictions=[gen_text] * len(all_refs), references=all_refs) if all_refs else {"scores": [0]}
        max_correct_b = max(bs["scores"][:len(correct)], default=0)
        is_correct = (rl >= 0.7) or (max_correct_b > 0.5)
        if is_correct:
            any_correct = True

            # -- Head tensor: per-layer o_proj inputs (generated tokens only) --
            layer_tensors = []
            for l in range(W_START, W_END):
                stored = head_storage[l]
                if not stored:
                    layer_tensors.append(torch.zeros(1, n_heads, head_dim))
                    continue
                # stored[0] = prompt step (num_beams, prompt_len, 4096) — discard
                # stored[1:] = generation steps (num_beams, 1, 4096) per step
                gen_steps = stored[1:min(len(gids)+1, len(stored))]
                if not gen_steps:
                    layer_tensors.append(torch.zeros(1, n_heads, head_dim))
                    continue
                step_tensors = []
                for s in gen_steps:
                    sb = s[b:b+1]  # (1, 1, 4096) — extract beam b
                    step_tensors.append(sb.reshape(1, 1, n_heads, head_dim))
                layer_cat = torch.cat(step_tensors, dim=1)  # (1, T_gen, n_heads, head_dim)
                # Signed absolute extremum pooling across time
                abs_vals = layer_cat.abs()
                max_idx = abs_vals.flatten(1).abs().sum(dim=2).argmax(dim=1, keepdim=True)
                flat = layer_cat.reshape(1, -1, n_heads, head_dim)
                pooled = flat.gather(dim=1, index=max_idx.unsqueeze(-1).unsqueeze(-1)
                                     .expand(-1, 1, n_heads, head_dim)).squeeze(1)
                layer_tensors.append(pooled)
        head_tensor = torch.cat(layer_tensors, dim=0)  # (9, n_heads, head_dim)

        # -- Lookback ratios --
        lookback_vecs = []
        for l in range(W_START, W_END):
            stored = attn_storage[l]
            if not stored:
                lookback_vecs.append(torch.zeros(n_heads))
                continue
            gen_steps = stored[1:min(len(gids)+1, len(stored))]
            if not gen_steps:
                lookback_vecs.append(torch.zeros(n_heads))
                continue
            ratios = []
            for s in gen_steps:  # (num_beams, n_heads, 1, S_total)
                sb = s[b]        # (n_heads, 1, S_total)
                total_len = sb.shape[-1]
                ctx_len = min(prompt_len, total_len)
                ctx_mass = sb[:, 0, :ctx_len].sum(dim=-1)  # (n_heads,)
                tot_mass = sb[:, 0, :].sum(dim=-1) + 1e-9
                ratios.append(ctx_mass / tot_mass)
            lookback_vecs.append(torch.stack(ratios).mean(dim=0))  # (n_heads,)
        lookback = torch.stack(lookback_vecs, dim=0)  # (L, n_heads)

        all_head.append(head_tensor)
        all_lookback.append(lookback)
        all_flags.append(not is_correct)

    all_is_known.append(any_correct)
    all_pi.extend([pi] * 10)

    del outputs, head_storage, attn_storage
    torch.cuda.empty_cache()

# -- Save --
out_dir = os.path.join("../data", args.model_folder)
os.makedirs(out_dir, exist_ok=True)

head_path = os.path.join(out_dir, f"{args.dataset}_head_resolved_absmax.pt")
torch.save({
    "all_emb": all_head,
    "all_lookback": all_lookback,
    "all_hallucination_flag": all_flags,
    "all_is_known": all_is_known,
    "prompt_indices": all_pi,
    "metadata": {"n_heads": n_heads, "head_dim": head_dim,
                 "layers": list(range(W_START, W_END))},
}, head_path)
print(f"\nSaved: {head_path}  ({os.path.getsize(head_path)/1e9:.2f} GB)")
print(f"  Beams: {len(all_head)}  Known: {sum(all_is_known)}/{n_prompts}")
