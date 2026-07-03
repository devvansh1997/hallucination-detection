"""
01_generate_and_extract.py — HARP-Compatible Generation & Extraction
=====================================================================
Beam search (10 beams) per prompt.  Contrastive BLEURT/ROUGE judge.
No system prompt — exact prompt templates matching HARP's protocol.

Usage:
  python 01_generate_and_extract.py
  python 01_generate_and_extract.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset triviaqa
"""

# -- MUST be before ANY other imports: prevents TensorFlow seizing all VRAM --
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import argparse
import gc
import sys

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# CONFIG
# ==============================================================================

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

# ==============================================================================
# GPU ENFORCEMENT
# ==============================================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available — aborting")
device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==============================================================================
# DATASET LOADER
# ==============================================================================

def load_dataset(ds_cfg: dict, debug: bool):
    """Return list of dicts: {prompt_text, correct_answers, incorrect_answers}.

    prompt_text is the fully formatted string ready for tokenization.
    """
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

    elif name == "triviaqa":
        # Deduplicate by question_id (HARP)
        seen = set()
        for ex in ds:
            qid = ex["question_id"]
            if qid in seen:
                continue
            seen.add(qid)
            value = ex["answer"]["value"]
            samples.append({
                "prompt_text": template.format(question=ex["question"]),
                "correct_answers": [str(value)],
                "incorrect_answers": [],
            })

    elif name == "nq_open":
        for ex in ds:
            samples.append({
                "prompt_text": template.format(question=ex["question"]),
                "correct_answers": [str(a) for a in ex["answer"]],
                "incorrect_answers": [],
            })

    elif name == "tydiqa_gp":
        # English only (HARP)
        for ex in ds:
            if ex.get("language", "english") != "english":
                continue
            ctx = ex["context"][0] if isinstance(ex["context"], list) else ex["context"]
            ans_texts = ex["answers"]["text"]
            samples.append({
                "prompt_text": template.format(
                    context=str(ctx), question=ex["question"]),
                "correct_answers": [str(a) for a in ans_texts if a],
                "incorrect_answers": [],
            })

    else:
        raise ValueError(f"Unknown dataset: {name}")

    if debug:
        import random
        random.seed(42)
        random.shuffle(samples)
        samples = samples[:min(50, len(samples))]

    ratio = cfg["output"]["dataset_ratio"]
    if ratio < 1.0 and not debug:
        import random
        random.seed(42)
        random.shuffle(samples)
        samples = samples[:int(len(samples) * ratio)]

    return samples


# ==============================================================================
# RUBRIC (HARP contrastive)
# ==============================================================================

def _load_metrics():
    import evaluate
    rouge = evaluate.load("rouge")
    bleurt = evaluate.load("bleurt", config_name=cfg["judge"]["bleurt_model"])
    return rouge, bleurt


def judge_contrastive(generated: str, correct: list[str], incorrect: list[str],
                      rouge, bleurt) -> bool:
    jc = cfg["judge"]

    all_refs = correct + incorrect
    if not all_refs:
        return False
    candidates = [generated] * len(all_refs)
    bleurt_scores = bleurt.compute(predictions=candidates, references=all_refs)["scores"]
    n_corr = len(correct)
    max_correct_bleurt = max(bleurt_scores[:n_corr], default=0.0)
    max_incorrect_bleurt = max(bleurt_scores[n_corr:], default=0.0)

    max_correct_rouge = 0.0
    for ref in correct:
        r = rouge.compute(predictions=[generated], references=[ref])
        max_correct_rouge = max(max_correct_rouge, r["rougeL"])

    advantage_ok = (max_correct_bleurt - max_incorrect_bleurt) > jc["correct_advantage"]
    threshold_ok = (max_correct_bleurt > jc["sen_sim_threshold"]
                    or max_correct_rouge > jc["rouge_threshold"])
    return advantage_ok and threshold_ok


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_model(model_id: str, debug: bool):
    print(f"  Loading: {model_id}")
    load_kwargs = dict(device_map=device, trust_remote_code=True)

    if debug:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        load_kwargs["dtype"] = torch.bfloat16
    else:
        load_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    L = model.config.num_hidden_layers
    D = model.config.hidden_size
    print(f"  Layers: {L}  |  Hidden: {D}  |  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer, L, D


# ==============================================================================
# BEAM SEARCH + JUDGE + EXTRACT  (per prompt)
# ==============================================================================

EOS_STOP_STRINGS = [".", "!", "?", ".\n", "!\n", "?\n", "\n", "\n\n"]


def process_prompt(sample: dict, model, tokenizer, L: int, D: int,
                   rouge, bleurt) -> list[dict]:
    gen_cfg = cfg["generation"]
    prompt_text = sample["prompt_text"]

    # Build stop token IDs
    eos_ids = {tokenizer.eos_token_id}
    for s in EOS_STOP_STRINGS:
        eos_ids.update(tokenizer.encode(s, add_special_tokens=False))
        eos_ids.update(tokenizer.encode("Yes" + s, add_special_tokens=False)[1:])
    eos_ids = list(eos_ids)

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            eos_token_id=eos_ids,
            do_sample=gen_cfg["do_sample"],
            temperature=gen_cfg["temperature"],
            top_k=gen_cfg["top_k"],
            top_p=gen_cfg["top_p"],
            num_beams=gen_cfg["num_beams"],
            num_return_sequences=gen_cfg["num_return_sequences"],
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )

    hidden_states = outputs.hidden_states
    num_generated = len(hidden_states) - 1
    generated_ids = outputs.sequences[:, prompt_len:]

    results = []
    for b in range(generated_ids.shape[0]):
        gen_ids = generated_ids[b]
        gen_ids = gen_ids[gen_ids != tokenizer.eos_token_id]
        generation = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if not generation:
            results.append({
                "pooled_tensor": torch.zeros(L, D),
                "is_hallucination": True,
                "is_correct": False,
                "generation": "",
            })
            continue

        is_correct = judge_contrastive(
            generation, sample["correct_answers"],
            sample["incorrect_answers"], rouge, bleurt)

        if num_generated == 0:
            H_pooled = torch.zeros(L, D)
        else:
            layer_tensors = []
            for l in range(L):
                tokens = []
                for step in range(1, len(hidden_states)):
                    if step - 1 >= len(gen_ids):
                        break
                    h = hidden_states[step][l + 1][b]
                    tokens.append(h.cpu())
                if tokens:
                    layer_tensors.append(torch.cat(tokens, dim=0))
                else:
                    layer_tensors.append(torch.zeros(0, D))
            H_raw = torch.stack(layer_tensors, dim=0)
            H_pooled = H_raw.float().mean(dim=1)

        results.append({
            "pooled_tensor": H_pooled,
            "is_hallucination": not is_correct,
            "is_correct": is_correct,
            "generation": generation,
        })

    del outputs, hidden_states
    torch.cuda.empty_cache()
    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("  HARP-COMPATIBLE BEAM SEARCH & EXTRACTION")
    print("=" * 60)

    models_to_run = cfg["models"]
    if args.model:
        models_to_run = [m for m in models_to_run if m["id"] == args.model]
    datasets_to_run = cfg["datasets"]
    if args.dataset:
        datasets_to_run = [d for d in datasets_to_run if d["name"] == args.dataset]

    print(f"  Models:   {[m['id'] for m in models_to_run]}")
    print(f"  Datasets: {[d['name'] for d in datasets_to_run]}")
    print(f"  Debug:    {args.debug}")

    rouge, bleurt = _load_metrics()

    for model_cfg in models_to_run:
        model_id = model_cfg["id"]
        folder = model_cfg["folder"]
        print(f"\n{'=' * 60}")
        print(f"  MODEL: {model_id}")
        print(f"{'=' * 60}")

        model, tokenizer, L, D = load_model(model_id, args.debug)

        for ds_cfg in datasets_to_run:
            ds_name = ds_cfg["name"]
            print(f"\n  --- {ds_name} ---")

            samples = load_dataset(ds_cfg, args.debug)
            n_prompts = len(samples)
            print(f"  Prompts: {n_prompts}")

            all_emb = []
            all_flags = []
            all_is_known = []
            n_known = 0
            n_unknown = 0

            for idx, sample in enumerate(tqdm(samples, desc=f"  {ds_name}")):
                beam_results = process_prompt(
                    sample, model, tokenizer, L, D, rouge, bleurt)

                any_correct = any(r["is_correct"] for r in beam_results)
                all_is_known.append(any_correct)
                if any_correct:
                    n_known += 1
                else:
                    n_unknown += 1

                for r in beam_results:
                    all_emb.append(r["pooled_tensor"])
                    all_flags.append(r["is_hallucination"])

                if (idx + 1) % 100 == 0:
                    vram = torch.cuda.memory_allocated() / 1e9
                    tqdm.write(
                        f"    [{idx+1:5d}/{n_prompts}]  "
                        f"known: {n_known}  unknown: {n_unknown}  "
                        f"VRAM: {vram:.1f} GB")

            out_dir = os.path.join(cfg["output"]["data_dir"], folder)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{ds_name}_pooled.pt")

            data = {
                "all_emb": all_emb,
                "all_hallucination_flag": all_flags,
                "all_is_known": all_is_known,
                "metadata": {
                    "model": model_id,
                    "dataset": ds_name,
                    "n_prompts": n_prompts,
                    "n_beams_total": len(all_emb),
                    "n_known_prompts": n_known,
                    "n_unknown_prompts": n_unknown,
                },
            }
            torch.save(data, out_path)
            print(f"  Saved: {out_path}")
            print(f"  Beams: {len(all_emb)}  "
                  f"(known prompts: {n_known}, unknown: {n_unknown})")

        del model, tokenizer
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
