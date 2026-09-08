#!/bin/bash
#SBATCH --job-name=act-ext
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#
# Pooled Activation Tensors in ACT-ViT's format, from our pinned generations.
#   sbatch -p highgpu --gres=gpu:1 --mem=80G --time=02:00:00 \
#          --job-name=act-tydiqa slurm/act_extract.slurm qwen-2.5-7b-instruct tydiqa_gp
#
# Args: <model_folder> <dataset> [n_pool] [max_gb]
#
# GPU memory is modest -- this is a forward pass with output_hidden_states, one question (10 beams)
# at a time. Host RAM is what matters: the output array is held in RAM and written once, so --mem
# must exceed the size 59 prints before it starts. It refuses rather than dying half-written.
#
# Log path uses %x_%j, not a bare directory. A previous job in this project wrote to
# slurm_logs/<name>_%j/ and the trailing slash made SLURM ignore --job-name entirely.

set -euo pipefail

MODEL="${1:?usage: act_extract.slurm <model_folder> <dataset> [n_pool] [max_gb]}"
DS="${2:?usage: act_extract.slurm <model_folder> <dataset> [n_pool] [max_gb]}"
NPOOL="${3:-100}"
MAXGB="${4:-60}"

REPO="${HD_REPO:-$HOME/Hallucination-Detection/hallucination-detection}"
cd "$REPO"
mkdir -p slurm_logs

source /share/apps/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate hal-det
module load cuda/12.6 2>/dev/null || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "host    : $(hostname)"
echo "model   : $MODEL"
echo "dataset : $DS"
echo "n_pool  : $NPOOL   max_gb: $MAXGB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python 59_extract_act_tensors.py --self-test

python 59_extract_act_tensors.py \
    --dataset "$DS" \
    --model_folder "$MODEL" \
    --n-pool "$NPOOL" \
    --max-gb "$MAXGB"

echo
echo "done. Output: ../data-acttensors/$MODEL/${DS}_at_L8_N${NPOOL}.npz"
