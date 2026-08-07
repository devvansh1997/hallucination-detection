"""
Colab Notebook -- Full-Beam TriviaQA Generation
================================================
Copy each cell block into a separate Colab cell and run sequentially.
Expected runtime: ~6-8 hours total on T4 (4-bit).
"""

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1: Mount Drive & Install                                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

from google.colab import drive
drive.mount('/content/drive')

!pip install -q torch --index-url https://download.pytorch.org/whl/cu126
!pip install -q transformers>=4.48.2 datasets evaluate rouge_score tensorflow pyyaml tqdm scikit-learn bitsandbytes
!pip install -q git+https://github.com/google-research/bleurt.git


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2: Clone Repo & Set Up                                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["HF_METRICS_CACHE"] = "/tmp/rouge_cache"

!rm -rf /content/hallucination-detection
!git clone https://github.com/devvansh1997/hallucination-detection.git
%cd /content/hallucination-detection

DRIVE_OUT = "/content/drive/MyDrive/hosvd_data"
os.makedirs(DRIVE_OUT, exist_ok=True)

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3: Run — LLaMA-3.1-8B on TriviaQA (bfloat16, ~8-10h on A100)      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

!python 01_generate_full_beams.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset triviaqa

!cp "../data/llama-3.1-8b-instruct/triviaqa_pooled_fullbeams.pt" "{DRIVE_OUT}/"


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4: Run — Qwen-2.5-7B on TriviaQA (bfloat16, ~8-10h on A100)       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

!python 01_generate_full_beams.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset triviaqa

!cp "../data/qwen-2.5-7b-instruct/triviaqa_pooled_fullbeams.pt" "{DRIVE_OUT}/"

print("Done. Files saved to Google Drive.")
