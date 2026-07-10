"""
20a_pilot_raw_tokens.py -- 100-Sample Raw Token Extraction
===========================================================
Generates 100 prompts from TruthfulQA, saves RAW (L, T, D) per-beam
tensors (no mean-pooling).  Used by 20_pilot_token_extraction.py.

Usage:
  python 20a_pilot_raw_tokens.py --model meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import gc, os, sys, time, argparse
import torch, yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, default="llama-3.1-8b-instruct",
                    help="Matches config.yaml folders")
args = parser.parse_args()

model_id = None
for m in cfg["models"]:
    if m["folder"] == args.model_folder:
        model_id = m["id"]; break
if model_id is None:
    raise ValueError("Unknown model folder: " + args.model_folder)
MODEL_ID = model_id
N_PILOT = 100

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")
device = torch.device("cuda")

# -- Load dataset --
from datasets import load_dataset
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
prompts = ds["question"][:N_PILOT]
refs    = ds["best_answer"][:N_PILOT]
incorrect = ds["incorrect_answers"][:N_PILOT]

import evaluate
rouge = evaluate.load("rouge")
bleurt = evaluate.load("bleurt", config_name="BLEURT-20")

# -- Load model --
print(f"Loading: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model.eval()
L = model.config.num_hidden_layers
D = model.config.hidden_size
print(f"  Layers: {L}  Hidden: {D}")

# -- EOS tokens --
eos_strs = [".", "!", "?", ".\n", "!\n", "?\n", "\n", "\n\n"]
eos_ids = {tokenizer.eos_token_id}
for s in eos_strs:
    eos_ids.update(tokenizer.encode(s, add_special_tokens=False))
    eos_ids.update(tokenizer.encode("Yes" + s, add_special_tokens=False)[1:])

# -- Generate --
all_tensors = []          # per prompt → per beam → (prompt_bottleneck, gen_tokens)
all_flags = []            # per beam
all_is_known = []         # per prompt
all_prompt_idx = []

for idx in tqdm(range(N_PILOT), desc="  Generating"):
    prompt = str(prompts[idx])
    correct = [str(refs[idx])]
    wrong = [str(w) for w in incorrect[idx]] if incorrect[idx] else []

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=64, eos_token_id=list(eos_ids),
            do_sample=True, temperature=0.5, top_k=5, top_p=0.99,
            num_beams=10, num_return_sequences=10,
            output_hidden_states=True, return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id, early_stopping=True)

    hidden_states = outputs.hidden_states
    num_gen = len(hidden_states) - 1
    gen_ids = outputs.sequences[:, prompt_len:]
    num_beams = gen_ids.shape[0]

    prompt_tensors = []
    prompt_flags = []
    any_correct = False

    for b in range(num_beams):
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
        prompt_flags.append(not is_correct)

        # Extract raw tensors
        # A) Last prompt token — from hidden_states[0] (prompt processing step)
        prompt_hs = hidden_states[0]           # tuple of (L+1) tensors, each (1, prompt_len, D)
        if prompt_len == 0:
            prompt_bottleneck = torch.zeros(L, D)
        else:
            p_layers = []
            for l in range(L):
                h = prompt_hs[l + 1][0, -1, :].cpu()     # last token, shape (D,)
                p_layers.append(h)
            prompt_bottleneck = torch.stack(p_layers, dim=0)  # (L, D)

        # B) Generated tokens — from hidden_states[1:]
        if num_gen == 0 or len(gids) == 0:
            gen_tokens = torch.zeros(L, 1, D)
        else:
            layers = []
            for l in range(L):
                tokens = []
                for step in range(1, len(hidden_states)):
                    if step - 1 >= len(gids):
                        break
                    tokens.append(hidden_states[step][l + 1][b].cpu())
                layers.append(torch.cat(tokens, dim=0) if tokens
                              else torch.zeros(0, D))
            gen_tokens = torch.stack(layers, dim=0)  # (L, T, D)

        prompt_tensors.append((prompt_bottleneck, gen_tokens))

    all_tensors.append(prompt_tensors)
    all_flags.extend(prompt_flags)
    all_is_known.append(any_correct)
    all_prompt_idx.extend([idx] * len(prompt_tensors))

    del outputs, hidden_states
    torch.cuda.empty_cache()

# -- Save --
out_dir = os.path.join("../data_unpooled", args.model_folder)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "truthfulqa_pilot_raw_tokens.pt")
torch.save({
    "all_tensors": all_tensors,
    "all_hallucination_flag": all_flags,
    "all_is_known": all_is_known,
    "prompt_indices": all_prompt_idx,
}, out_path)
print(f"\nSaved {N_PILOT} prompts to {out_path}")
