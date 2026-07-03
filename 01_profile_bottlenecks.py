"""
01_generate_and_extract.py -- Distributed Dual-Extraction Pipeline
===================================================================
Generates responses, grades hallucination via BLEURT/ROUGE-L, and extracts
BOTH Faizul's 8-dim thermodynamic features AND HOSVD mean-pooled tensors
from the raw 3D hidden states before they leave VRAM.

Supports SLURM job arrays via --num_shards / --shard_idx.

Usage:
  python 01_generate_and_extract.py --model meta-llama/Llama-3.2-3B-Instruct --dataset triviaqa --num_shards 8 --shard_idx 0
"""

import argparse
import gc
import os
import sys
import time

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
    description="Distributed dual-extraction generation pipeline"
)
parser.add_argument("--model", type=str,
                    default="meta-llama/Llama-3.2-3B-Instruct",
                    help="HuggingFace model ID")
parser.add_argument("--dataset", type=str, required=True,
                    choices=["truthfulqa", "triviaqa", "tydiqa"])
parser.add_argument("--debug", action="store_true", default=False,
                    help="4-bit quant, 5% slice (max 50)")
parser.add_argument("--num_shards", type=int, default=1,
                    help="Total number of shards (for SLURM array)")
parser.add_argument("--shard_idx", type=int, default=0,
                    help="This shard's index (0-based, from SLURM_ARRAY_TASK_ID)")

args = parser.parse_args()

# ==============================================================================
# CONSTANTS
# ==============================================================================

MODEL_ID  = args.model
MODEL_KEY = MODEL_ID.split("/")[-1].replace("-", "").replace(".", "_").lower()

MAX_NEW_TOKENS = 50
DEBUG_MAX_SAMPLES = 50
DEBUG_FRAC     = 0.05
RANDOM_SEED    = 42

torch.manual_seed(RANDOM_SEED)


# ==============================================================================
# DATASET LOADER  (50% slice + shard)
# ==============================================================================

def load_dataset_sharded(name: str, debug: bool,
                         num_shards: int, shard_idx: int
                         ) -> list[tuple[str, str]]:
    """Load dataset, shuffle (seed=42), take 50%, then extract this shard's slice."""
    from datasets import load_dataset

    if name == "truthfulqa":
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        prompts    = ds["question"]
        references = ds["best_answer"]
    elif name == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        prompts    = ds["question"]
        references = [a["value"] for a in ds["answer"]]
    elif name == "tydiqa":
        ds = load_dataset("google-research-datasets/tydiqa", "secondary_task",
                          split="validation")
        prompts    = ds["question"]
        references = [a["text"][0] if len(a["text"]) > 0 else ""
                      for a in ds["answers"]]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    print(f"  {name}: {len(prompts)} samples loaded")

    # -- Convert to HuggingFace Dataset for shuffle/slice/shard ------------
    from datasets import Dataset as HFDataset
    ds = HFDataset.from_dict({"prompt": prompts, "reference": references})

    if debug:
        n_use = min(int(len(ds) * DEBUG_FRAC), DEBUG_MAX_SAMPLES)
        ds = ds.shuffle(seed=RANDOM_SEED).select(range(n_use))
        print(f"  Debug mode: sliced to {n_use} samples")
    else:
        # 50% slice (shuffled) then shard
        total = len(ds)
        ds = ds.shuffle(seed=RANDOM_SEED)
        ds = ds.select(range(total // 2))          # take exactly 50%
        print(f"  50% slice: {len(ds)} samples (from {total})")

    # Shard across job-array tasks
    if num_shards > 1:
        ds = ds.shard(num_shards=num_shards, index=shard_idx)
        print(f"  Shard {shard_idx}/{num_shards}: {len(ds)} samples")

    pairs = [(ds["prompt"][i], ds["reference"][i]) for i in range(len(ds))]
    return pairs


# ==============================================================================
# HALLUCINATION GRADING  (ROUGE-L + BLEURT)
# ==============================================================================

def _load_metrics(debug: bool):
    import evaluate
    rouge = evaluate.load("rouge")
    bleurt_device = "cpu" if debug else None
    bleurt = evaluate.load("bleurt", config_name="BLEURT-20", device=bleurt_device)
    return rouge, bleurt


def judge_hallucination(prediction: str, reference: str, rouge, bleurt) -> bool:
    r = rouge.compute(predictions=[prediction], references=[reference])
    rouge_l = r["rougeL"]
    b = bleurt.compute(predictions=[prediction], references=[reference])
    bleurt_score = b["scores"][0]
    is_factual = (rouge_l >= 0.7) or (bleurt_score >= 0.5)
    return not is_factual


# ==============================================================================
# FAIZUL 8-DIM THERMODYNAMIC FEATURES  (from raw 3D tensor)
# ==============================================================================

def compute_faizul_features(H_raw: torch.Tensor) -> list[float]:
    """Extract 4 scalars from a raw 3D hidden-state tensor (L, T, D):

      1. Frobenius norm of the full tensor
      2. Entropy of singular values from layer-mode (mode-0) unfolding
      3. Effective rank: exp(entropy)
      4. Top-3 concentration: sum of top 3 normalised SVs

    Returns [core_norm, entropy, eff_rank, top_k_concentration].
    """
    L, T, D = H_raw.shape

    # 1. Frobenius norm
    core_norm = float(torch.linalg.norm(H_raw.float()))

    # 2-4. Layer-mode SVD (unfold mode 0 -> (L, T*D))
    unfolded = H_raw.float().reshape(L, -1)           # (L, T*D)
    S = torch.linalg.svdvals(unfolded)                 # singular values
    p = S / (S.sum() + 1e-9)
    p = torch.clamp(p, min=1e-9)

    entropy  = float(-torch.sum(p * torch.log(p)))
    eff_rank = float(torch.exp(torch.tensor(entropy)))
    top_k    = float(p[:3].sum())

    return [core_norm, entropy, eff_rank, top_k]


# ==============================================================================
# MODEL LOADING  (dual-mode)
# ==============================================================================

def load_model(debug: bool):
    print(f"\nLoading model: {MODEL_ID}")
    if debug:
        print("  Mode: 4-bit quantisation (BitsAndBytes)")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map="auto",
            dtype=torch.bfloat16, trust_remote_code=True)
    else:
        print("  Mode: bfloat16 native (cluster)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Layers: {num_layers}  |  Hidden dim: {hidden_dim}"
          f"  |  VRAM: {vram_gb:.2f} GB")
    return model, tokenizer, num_layers, hidden_dim


# ==============================================================================
# PER-SAMPLE PROCESSING  (dual extraction)
# ==============================================================================

def process_one_sample(
    prompt: str, reference: str,
    model, tokenizer, num_layers: int, hidden_dim: int,
    rouge, bleurt,
) -> tuple[list[float], torch.Tensor, bool, int, dict]:
    """Generate, grade, dual-extract.  Returns results + per-block timing dict."""

    timings = {}

    # -- Chat template ---------------------------------------------------
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEWSHOT_USER},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT},
        {"role": "user", "content": prompt},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
    stop_criteria = StoppingCriteriaList([StopOnNewline(newline_id)])

    # -- BLOCK A: Generate + hidden state extraction ----------------------
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stop_criteria,
            output_hidden_states=True, return_dict_in_generate=True,
            do_sample=True, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id)

    generated_ids = outputs.sequences[0][prompt_len:]
    generation = tokenizer.decode(generated_ids, skip_special_tokens=True)

    hidden_states = outputs.hidden_states
    num_generated = len(hidden_states) - 1

    if num_generated == 0:
        del outputs, hidden_states
        torch.cuda.empty_cache()
        timings["gen"] = time.time() - t0
        timings["bleurt"] = 0.0
        timings["faizul"] = 0.0
        timings["hosvd"] = 0.0
        return [0.0]*8, torch.zeros(num_layers, hidden_dim), False, 0, timings

    layer_tensors = []
    for l in range(num_layers):
        tokens = []
        for step in range(1, len(hidden_states)):
            h = hidden_states[step][l + 1]
            tokens.append(h.squeeze(0))
        layer_tensors.append(torch.cat(tokens, dim=0))

    H_raw = torch.stack(layer_tensors, dim=0).cpu()
    timings["gen"] = time.time() - t0

    # -- BLOCK B: BLEURT + ROUGE -----------------------------------------
    t0 = time.time()
    is_hallucination = judge_hallucination(generation, reference, rouge, bleurt)
    timings["bleurt"] = time.time() - t0

    # -- BLOCK C: Faizul 8-dim SVD features ------------------------------
    t0 = time.time()
    H_delta = H_raw[1:] - H_raw[:-1]
    faizul_H       = compute_faizul_features(H_raw)
    faizul_H_delta = compute_faizul_features(H_delta)
    faizul_vec = faizul_H + faizul_H_delta
    timings["faizul"] = time.time() - t0

    # -- BLOCK D: HOSVD mean-pooling -------------------------------------
    t0 = time.time()
    H_pooled = H_raw.float().mean(dim=1)
    timings["hosvd"] = time.time() - t0

    del outputs, hidden_states, layer_tensors, H_raw, H_delta
    torch.cuda.empty_cache()

    return faizul_vec, H_pooled, is_hallucination, num_generated, timings


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print(f"  {MODEL_KEY}  |  {args.dataset.upper()}"
          f"  |  {'DEBUG' if args.debug else 'FULL'}"
          f"  |  shard {args.shard_idx}/{args.num_shards}")
    print("=" * 60)

    # -- 1. Load dataset --------------------------------------------------
    print("\n[1/4] Loading dataset ...")
    pairs = load_dataset_sharded(args.dataset, args.debug,
                                 args.num_shards, args.shard_idx)
    print(f"       Total prompts: {len(pairs)}")

    # -- 2. Load metrics --------------------------------------------------
    print("\n[2/4] Loading ROUGE-L + BLEURT-20 ...")
    rouge, bleurt = _load_metrics(args.debug)
    print("       Metrics ready.")

    # -- 3. Load model ----------------------------------------------------
    print("\n[3/4] Loading model ...")
    model, tokenizer, num_layers, hidden_dim = load_model(args.debug)

    # -- 4. Generation loop (profiling: 100 samples max) ------------------
    PROFILE_SAMPLES = 100
    n_profile = min(len(pairs), PROFILE_SAMPLES)
    print(f"\n[4/4] Processing {n_profile} samples (PROFILING MODE) ...\n")

    # Cumulative timers
    t_gen    = 0.0
    t_bleurt = 0.0
    t_faizul = 0.0
    t_hosvd  = 0.0

    for idx, (prompt, reference) in enumerate(pairs[:n_profile]):
        faizul_vec, H_pooled, is_hall, t_new, timings = process_one_sample(
            prompt, reference, model, tokenizer,
            num_layers, hidden_dim, rouge, bleurt)

        t_gen    += timings["gen"]
        t_bleurt += timings["bleurt"]
        t_faizul += timings["faizul"]
        t_hosvd  += timings["hosvd"]

        if (idx + 1) % 10 == 0 or idx == 0:
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"  [{idx+1:4d}/{n_profile}]  "
                  f"gen: {timings['gen']:.2f}s  bleurt: {timings['bleurt']:.2f}s  "
                  f"faizul: {timings['faizul']:.2f}s  hosvd: {timings['hosvd']:.3f}s  "
                  f"VRAM: {vram:.2f} GB", flush=True)

    # -- Diagnostic report ------------------------------------------------
    t_total = t_gen + t_bleurt + t_faizul + t_hosvd
    m = n_profile

    print(f"\n{'=' * 60}")
    print(f"  PROFILING REPORT  ({m} samples)")
    print(f"{'=' * 60}")
    print(f"  {'Block':30s}  {'Cumul.':>8s}  {'Avg/sample':>10s}  {'%':>6s}")
    print(f"  {'─' * 30}  {'─' * 8}  {'─' * 10}  {'─' * 6}")
    for name, t in [("A: Generation + token extraction", t_gen),
                     ("B: BLEURT + ROUGE evaluation",    t_bleurt),
                     ("C: Faizul SVD features",          t_faizul),
                     ("D: HOSVD mean-pooling",           t_hosvd)]:
        print(f"  {name:30s}  {t:8.1f}s  {t/m:10.3f}s  {t/t_total*100:5.1f}%")
    print(f"  {'─' * 30}  {'─' * 8}  {'─' * 10}  {'─' * 6}")
    print(f"  {'TOTAL':30s}  {t_total:8.1f}s  {t_total/m:10.3f}s  100.0%")
    print(f"")

    # Projected for full run
    ESTIMATED_SAMPLES = 18000
    proj = t_total / m * ESTIMATED_SAMPLES
    print(f"  Projected for {ESTIMATED_SAMPLES:,} samples:")
    print(f"    Generation:      {t_gen/m * ESTIMATED_SAMPLES / 3600:6.1f} hours")
    print(f"    BLEURT/ROUGE:    {t_bleurt/m * ESTIMATED_SAMPLES / 3600:6.1f} hours")
    print(f"    Faizul SVD:      {t_faizul/m * ESTIMATED_SAMPLES / 3600:6.1f} hours")
    print(f"    HOSVD pooling:   {t_hosvd/m * ESTIMATED_SAMPLES / 3600:6.1f} hours")
    print(f"    ─────────────────────────────")
    print(f"    TOTAL projected: {proj / 3600:.1f} hours")
    print(f"{'=' * 60}\n")

    print("  Profiling complete — exiting before full dataset generation.")


if __name__ == "__main__":
    main()
