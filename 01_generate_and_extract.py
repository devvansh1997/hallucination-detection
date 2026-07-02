"""
01_generate_and_extract.py -- Dual-Mode LLaMA-3.1-8B Generation & Extraction
=============================================================================
Generates responses for QA datasets (TruthfulQA / TriviaQA / TyDiQA),
evaluates hallucination on-the-fly via ROUGE-L + BLEURT, and extracts
mean-pooled hidden states from generated tokens only.

Dual mode:
  --debug    4-bit quant, 5% of data  -> 12 GB VRAM (RTX 5070)
  (default)  bfloat16, full dataset   -> 80 GB VRAM (A100/H100 cluster)

Usage:
  python 01_generate_and_extract.py --dataset truthfulqa --debug
  python 01_generate_and_extract.py --dataset triviaqa
"""

import argparse
import gc
import os
import sys

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)


# ==============================================================================
# SYSTEM PROMPT & FEW-SHOT (forces concise, single-line answers)
# ==============================================================================

SYSTEM_PROMPT = (
    "You are a strict, factual Q&A bot. You must answer the user's question "
    "using the absolute fewest words possible. Provide only the direct answer. "
    "Do not use full sentences. Do not add explanations or pleasantries."
)

FEWSHOT_USER      = "Who wrote Hamlet?"
FEWSHOT_ASSISTANT = "William Shakespeare"


class StopOnNewline(StoppingCriteria):
    """Halts generation on newline, but only AFTER a minimum number of tokens
    (to avoid stopping on chat-template formatting newlines)."""
    def __init__(self, newline_token_id: int, min_tokens: int = 3):
        self.nl_id = newline_token_id
        self.min_tokens = min_tokens
        self._seen = 0

    def __call__(self, input_ids, scores, **kwargs):
        self._seen += 1
        if self._seen <= self.min_tokens:
            return False
        if input_ids[0, -1].item() == self.nl_id:
            return True
        return False


# ==============================================================================
# ARGPARSE
# ==============================================================================

parser = argparse.ArgumentParser(
    description="LLaMA generation + hallucination grading + extraction"
)
parser.add_argument(
    "--model",
    type=str,
    default="meta-llama/Llama-3.2-3B-Instruct",
    help="HuggingFace model ID (used for loading and file naming)",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["truthfulqa", "triviaqa", "tydiqa"],
    help="QA dataset to benchmark on",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="4-bit quantisation, 5% slice (max 50 samples) -- for local GPU testing",
)
args = parser.parse_args()

# ==============================================================================
# CONSTANTS
# ==============================================================================

MODEL_ID  = args.model
MODEL_KEY = MODEL_ID.split("/")[-1].replace("-", "").replace(".", "_").lower()
# e.g. "meta-llama/Llama-3.2-3B-Instruct" -> "llama_3_2_3b_instruct"

MAX_NEW_TOKENS = 50
DEBUG_MAX_SAMPLES = 50
DEBUG_FRAC     = 0.05
RANDOM_SEED    = 42

SUFFIX = "debug" if args.debug else "full"
OUTPUT_PATH = f"../data/{MODEL_KEY}_{args.dataset}_pooled_{SUFFIX}.pt"

torch.manual_seed(RANDOM_SEED)

# ==============================================================================
# DATASET LOADERS
# ==============================================================================


def load_dataset(name: str, debug: bool) -> list[tuple[str, str]]:
    """Return list of (prompt, reference_answer) pairs for the given QA dataset.

    In debug mode, shuffles and slices to 5% (capped at DEBUG_MAX_SAMPLES).
    """
    from datasets import load_dataset

    if name == "truthfulqa":
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        prompts = ds["question"]
        references = ds["best_answer"]
        print(f"  TruthfulQA: {len(prompts)} samples loaded")

    elif name == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        prompts = ds["question"]
        # answer is a dict with 'value', 'aliases', 'normalized_value' etc.
        references = [a["value"] for a in ds["answer"]]
        print(f"  TriviaQA: {len(prompts)} samples loaded")

    elif name == "tydiqa":
        ds = load_dataset(
            "google-research-datasets/tydiqa", "secondary_task", split="validation"
        )
        prompts = ds["question"]
        references = [a["text"][0] if len(a["text"]) > 0 else "" for a in ds["answers"]]
        print(f"  TyDiQA: {len(prompts)} samples loaded")

    else:
        raise ValueError(f"Unknown dataset: {name}")

    pairs = list(zip(prompts, references))

    if debug:
        n_total = len(pairs)
        n_use = min(int(n_total * DEBUG_FRAC), DEBUG_MAX_SAMPLES)
        # Shuffle with fixed seed for reproducibility
        import random

        rng = random.Random(RANDOM_SEED)
        rng.shuffle(pairs)
        pairs = pairs[:n_use]
        print(f"  Debug mode: sliced to {n_use} samples (from {n_total})")

    return pairs


# ==============================================================================
# HALLUCINATION GRADING  (ROUGE-L + BLEURT -- Faizul rubric)
# ==============================================================================


def _load_metrics(debug: bool):
    """Load ROUGE and BLEURT once.  In debug mode, force BLEURT onto CPU."""
    import evaluate

    rouge = evaluate.load("rouge")

    # BLEURT is TensorFlow-based; placing it on CPU in debug mode keeps
    # GPU VRAM exclusively for the LLaMA model.
    bleurt_device = "cpu" if debug else None  # None = auto
    bleurt = evaluate.load("bleurt", config_name="BLEURT-20", device=bleurt_device)
    return rouge, bleurt


def judge_hallucination(prediction: str, reference: str, rouge, bleurt) -> bool:
    """Return True if hallucinated, False if factual.

    Faizul rubric: ROUGE-L >= 0.7  OR  BLEURT >= 0.5  ->  factual.
    """
    # ROUGE-L
    r = rouge.compute(predictions=[prediction], references=[reference])
    rouge_l = r["rougeL"]

    # BLEURT
    b = bleurt.compute(predictions=[prediction], references=[reference])
    bleurt_score = b["scores"][0]

    is_factual = (rouge_l >= 0.7) or (bleurt_score >= 0.5)
    return not is_factual


# ==============================================================================
# MODEL LOADING  (dual-mode)
# ==============================================================================


def load_model(debug: bool):
    """Load LLaMA-3.1-8B.  4-bit for debug, bfloat16 for full cluster."""
    print(f"\nLoading model: {MODEL_ID}")

    if debug:
        print("  Mode: 4-bit quantisation (BitsAndBytes)")
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        print("  Mode: bfloat16 native (cluster)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    num_layers = model.config.num_hidden_layers  # 32
    hidden_dim = model.config.hidden_size  # 4096
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(
        f"  Layers: {num_layers}  |  Hidden dim: {hidden_dim}"
        f"  |  VRAM: {vram_gb:.2f} GB"
    )
    return model, tokenizer, num_layers, hidden_dim


# ==============================================================================
# GENERATION LOOP  (one sample, aggressive memory hygiene)
# ==============================================================================


def process_one_sample(
    prompt: str,
    reference: str,
    model,
    tokenizer,
    num_layers: int,
    hidden_dim: int,
    rouge,
    bleurt,
) -> tuple[torch.Tensor, bool, int]:
    """Generate, grade, extract.  Returns (H_pooled, is_hallucinated, T_new).

    H_pooled in R^{32 x 4096} -- mean-pooled across generated tokens.
    T_new = number of generated tokens.
    """
    # -- Format via chat template (system prompt + few-shot + question) ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEWSHOT_USER},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT},
        {"role": "user", "content": prompt},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    # Stop on first newline -- prevents rambling past the answer
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
    stop_criteria = StoppingCriteriaList([StopOnNewline(newline_id)])

    # -- Generate -----------------------------------------------------------------
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    # -- Decode generated text (prompt excluded) -------------------------
    generated_ids = outputs.sequences[0][prompt_len:]
    generation = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # -- Grade --------------------------------------------------------------------
    is_hallucination = judge_hallucination(generation, reference, rouge, bleurt)

    # -- Extract hidden states: NEW TOKENS ONLY --------------------------
    # hidden_states: tuple of (1 + num_generated) entries.
    #   hidden_states[0]     -> prompt step -> DISCARD
    #   hidden_states[1..T]  -> generation steps -> KEEP
    # Each entry is a tuple of (num_layers + 1) tensors (embedding + layers).
    # We skip index 0 (embedding) and keep indices 1..num_layers.

    hidden_states = outputs.hidden_states
    num_generated = len(hidden_states) - 1

    # Pre-allocate accumulator for mean-pooling  (on CPU to save VRAM)
    H_sum = torch.zeros(num_layers, hidden_dim, device="cpu", dtype=torch.float32)

    if num_generated > 0:
        for l in range(num_layers):
            # Accumulate across generation steps
            for step in range(1, len(hidden_states)):
                # hidden_states[step][l+1] -> skip embedding layer at index 0
                h = hidden_states[step][l + 1]  # (1, 1, hidden_dim)
                H_sum[l] += h.squeeze(0).squeeze(0).cpu().float()

        # Mean-pool across tokens
        H_pooled = H_sum / num_generated  # (32, 4096)
    else:
        # Edge case: model produced nothing -- return zeros
        H_pooled = H_sum  # all zeros

    # -- Memory hygiene --------------------------------------------------
    del outputs, hidden_states
    torch.cuda.empty_cache()

    return H_pooled, is_hallucination, num_generated


# ==============================================================================
# MAIN LOOP
# ==============================================================================


def main():
    print("=" * 60)
    print(
        f"  LLaMA-3.1-8B  |  {args.dataset.upper()}"
        f"  |  {'DEBUG' if args.debug else 'FULL'}"
    )
    print("=" * 60)

    # -- 1. Load dataset -------------------------------------------------
    print("\n[1/4] Loading dataset ...")
    pairs = load_dataset(args.dataset, args.debug)
    print(f"       Total prompts: {len(pairs)}")

    # -- 2. Load metrics (BLEURT on CPU in debug mode) -------------------
    print("\n[2/4] Loading ROUGE-L + BLEURT-20 ...")
    rouge, bleurt = _load_metrics(args.debug)
    print("       Metrics ready.")

    # -- 3. Load model ---------------------------------------------------
    print("\n[3/4] Loading LLaMA-3.1-8B ...")
    model, tokenizer, num_layers, hidden_dim = load_model(args.debug)

    # -- 4. Generation loop ----------------------------------------------
    print(f"\n[4/4] Processing {len(pairs)} prompts ...")
    print(f"       Saving to: {OUTPUT_PATH}\n")

    all_emb = []  # list of (32, 4096) pooled tensors
    all_hallucination_flag = []  # list of bool

    for idx, (prompt, reference) in enumerate(pairs):
        H_pooled, is_hall, t_new = process_one_sample(
            prompt,
            reference,
            model,
            tokenizer,
            num_layers,
            hidden_dim,
            rouge,
            bleurt,
        )

        all_emb.append(H_pooled)
        all_hallucination_flag.append(is_hall)

        # Progress
        if (idx + 1) % 10 == 0 or idx == 0:
            n_hall = sum(all_hallucination_flag)
            rate = n_hall / len(all_hallucination_flag) * 100
            vram = torch.cuda.memory_allocated() / 1e9
            print(
                f"  [{idx + 1:4d}/{len(pairs)}]  "
                f"hall rate: {rate:5.1f}%  |  "
                f"VRAM: {vram:.2f} GB  |  "
                f"last T: {t_new}",
                flush=True,
            )

        # Periodic checkpoint every 200 samples (safety net)
        if (idx + 1) % 200 == 0:
            ckpt = {
                "all_emb": all_emb,
                "all_hallucination_flag": all_hallucination_flag,
            }
            torch.save(ckpt, OUTPUT_PATH + ".ckpt")
            print(f"       [checkpoint saved at sample {idx + 1}]", flush=True)

    # -- Final save ------------------------------------------------------
    data = {
        "all_emb": all_emb,
        "all_hallucination_flag": all_hallucination_flag,
    }
    torch.save(data, OUTPUT_PATH)

    n_total = len(all_hallucination_flag)
    n_hall = sum(all_hallucination_flag)
    print(f"\n{'=' * 60}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  Dataset:   {args.dataset}")
    print(f"  Mode:      {'debug' if args.debug else 'full'}")
    print(f"  Samples:   {n_total}")
    print(f"  Halluc.:   {n_hall}/{n_total}  ({n_hall / n_total * 100:.1f}%)")
    print(f"  Saved to:  {OUTPUT_PATH}")
    print(f"{'=' * 60}")

    # Clean up checkpoint if it exists
    ckpt_path = OUTPUT_PATH + ".ckpt"
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == "__main__":
    main()
