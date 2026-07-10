"""
21_generate_maxpool_datasets.py -- Max-Energy Pooled Extraction
=================================================================
Deprecates mean-pooling. Extracts max-pooled hidden states across
the 9-layer reasoning window (layers 15-23) for full datasets.

Saves compact (L=9, D) tensors — not terabyte-scale raw tokens.

Usage:
  python 21_generate_maxpool_datasets.py --model_folder llama-3.1-8b-instruct --dataset truthfulqa
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

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")
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
    raise ValueError("Unsupported dataset: " + name)

print(f"Dataset: {name}, prompts: {len(samples)}")

# -- Load metrics --
import evaluate
rouge = evaluate.load("rouge")
bleurt = evaluate.load("bleurt", config_name="BLEURT-20")

# -- Load model --
print(f"Loading: {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model.eval()
L = model.config.num_hidden_layers
D = model.config.hidden_size
W_START, W_END = 15, 24
print(f"  Layers: {L}  Hidden: {D}  Window: {W_START}:{W_END}")

# -- EOS --
eos_strs = [".", "!", "?", ".\n", "!\n", "?\n", "\n", "\n\n"]
eos_ids = {tokenizer.eos_token_id}
for s in eos_strs:
    eos_ids.update(tokenizer.encode(s, add_special_tokens=False))
    eos_ids.update(tokenizer.encode("Yes" + s, add_special_tokens=False)[1:])

gen_cfg = cfg["generation"]

all_emb, all_flags, all_is_known, all_prompt_idx = [], [], [], []

for idx, sample in enumerate(tqdm(samples, desc=f"  {name}")):
    prompt_text = sample["prompt_text"]
    correct = sample["correct_answers"]
    wrong = sample["incorrect_answers"]

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=gen_cfg["max_new_tokens"],
            eos_token_id=list(eos_ids),
            do_sample=gen_cfg["do_sample"], temperature=gen_cfg["temperature"],
            top_k=gen_cfg["top_k"], top_p=gen_cfg["top_p"],
            num_beams=gen_cfg["num_beams"],
            num_return_sequences=gen_cfg["num_return_sequences"],
            output_hidden_states=True, return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id, early_stopping=True)

    hidden_states = outputs.hidden_states
    num_gen = len(hidden_states) - 1
    gen_ids = outputs.sequences[:, prompt_len:]

    any_correct = False
    for b in range(gen_ids.shape[0]):
        gids = gen_ids[b]
        gids = gids[gids != tokenizer.eos_token_id]
        gen_text = tokenizer.decode(gids, skip_special_tokens=True).strip()

        # Judge
        r = rouge.compute(predictions=[gen_text], references=correct)
        rl = r["rougeL"]
        all_refs = correct + wrong
        candidates = [gen_text] * len(all_refs)
        bs = bleurt.compute(predictions=candidates, references=all_refs)
        max_correct_b = max(bs["scores"][:len(correct)], default=0)
        is_correct = (rl >= 0.7) or (max_correct_b > 0.5)
        if is_correct:
            any_correct = True

        # Max-pool across generated tokens, layers 15-23 only
        if num_gen == 0 or len(gids) == 0:
            H_pooled = torch.zeros(W_END - W_START, D)
        else:
            layers = []
            for l in range(W_START, W_END):
                tokens = []
                for step in range(1, len(hidden_states)):
                    if step - 1 >= len(gids):
                        break
                    tokens.append(hidden_states[step][l + 1][b].cpu())
                layer_cat = torch.cat(tokens, dim=0) if tokens else torch.zeros(0, D)
                # Max-pool across token dimension
                layers.append(layer_cat.max(dim=0).values if layer_cat.shape[0] > 0
                              else torch.zeros(D))
            H_pooled = torch.stack(layers, dim=0)  # (9, D)

        all_emb.append(H_pooled)
        all_flags.append(not is_correct)

    all_is_known.append(any_correct)
    all_prompt_idx.extend([idx] * 10)

    del outputs, hidden_states
    torch.cuda.empty_cache()

# -- Save --
out_dir = os.path.join("../data", args.model_folder)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{args.dataset}_pooled_maxenergy.pt")
torch.save({
    "all_emb": all_emb,
    "all_hallucination_flag": all_flags,
    "all_is_known": all_is_known,
    "prompt_indices": all_prompt_idx,
}, out_path)

fsize = os.path.getsize(out_path) / 1e9
print(f"\nSaved: {out_path}  ({fsize:.2f} GB)")
print(f"  Beams: {len(all_emb)}  Known prompts: {sum(all_is_known)}/{len(samples)}")
