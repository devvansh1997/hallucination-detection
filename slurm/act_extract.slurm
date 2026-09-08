#!/bin/bash
#SBATCH --job-name=act-ext
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Pooled Activation Tensors in ACT-ViT's format, from our pinned generations.
#
# usage (via submit_act.sh, not directly):
#   sbatch slurm/act_extract.slurm <model_folder> <dataset> [n_pool] [max_gb]
#
# GPU memory is modest -- a forward pass with output_hidden_states, one question (10 beams) at a
# time. HOST RAM is the binding constraint: the output array is held in RAM and written once, so
# --mem must exceed the size 59 prints before it starts. It refuses rather than dying half-written.
#
# SHELL OPTIONS: `set -u`, deliberately NOT `set -euo pipefail`. Under `set -e` a single failing
# line in the module/conda preamble kills the job in under a second, so sbatch returns a job id for
# a job that is already dead and never appears in squeue. That exact failure is why the first
# version of this file queued nothing. Every python call carries its own `|| exit 1` instead.
#
# MODULE NAMES come from hgreb_stage.slurm, which is known to run on Newton. AGENTS.md's CLUSTER
# section is stale on both the anaconda path and the repo path -- do not copy from it.

set -u

MODEL="${1:?usage: act_extract.slurm <model_folder> <dataset> [n_pool] [max_gb]}"
DS="${2:?usage: act_extract.slurm <model_folder> <dataset> [n_pool] [max_gb]}"
NPOOL="${3:-100}"
MAXGB="${4:-60}"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPO="${HD_REPO:-/home/de807845/Hallucination-Detection/hallucination-detection}"

echo "${TIMESTAMP}: act extract ${MODEL}/${DS}  pool=(8,${NPOOL})  max_gb=${MAXGB}"

module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0
conda activate hal-det

export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$MPLCONFIGDIR"

cd "$REPO" || { echo "FATAL: cannot cd to $REPO -- set HD_REPO"; exit 1; }
echo "repo HEAD: $(git -C $REPO log --oneline -1 2>/dev/null)"
echo "python   : $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

python 59_extract_act_tensors.py --self-test || exit 1

python 59_extract_act_tensors.py \
    --dataset "$DS" \
    --model_folder "$MODEL" \
    --n-pool "$NPOOL" \
    --max-gb "$MAXGB" || exit 1

echo
echo "done. Output: ../data-acttensors/${MODEL}/${DS}_at_L8_N${NPOOL}.npz"
