#!/bin/bash
#SBATCH --job-name=harp-qwen-nq_open
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/harp-qwen-nq_open_%j/harp-qwen-nq_open_%j.out
#SBATCH --error=slurm_logs/harp-qwen-nq_open_%j/harp-qwen-nq_open_%j.err

# FIRE ONLY AFTER harp_qwen_tydiqa HAS FINISHED (model-keyed SVD cache, main.py:109).
# Safe to run alongside harp_qwen_truthfulqa.
#
# Biggest of the three small runs: 36,100 beams, prompt+answer stored.
#   ~28GB into hidden_state/  -- check the filesystem has room before firing.
#   ~150G host RAM because data_device is "cpu" after the patch, so the whole thing
#   accumulates in RAM before torch.save (main.py:206-224).
#   GPU: ~30GB fp32 model + ~8GB projected data at proj_dim=1024 -> fits 80GB.
# Longest of the three: 36,100 sequential forward passes (main.py:206 leaves
# get_hidden_state's batch_size at its default of 1), then 7 proj_dims x 50 epochs.
#
# This is the dataset where the split finding predicts the SMALLEST inflation (~3% of
# test answers affected), so it is the control end of that prediction.

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPO=/home/de807845/Hallucination-Detection/hallucination-detection
HARP=/home/de807845/Hallucination-Detection/HARP-Code

echo "${TIMESTAMP}: Loading modules.. "
module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0

conda activate hal-det
# HARP targets the transformers 4.5x API (main.py:105 uses model.model.config, and
# bleurt_pytorch needs pytorch_utils.find_pruneable_heads_and_indices -- both gone in
# 5.13). This venv was built with --system-site-packages, so its transformers 4.51
# shadows hal-det's 5.13 while torch/CUDA come through from the conda env unchanged.
source $HOME/envs/harp-rerun/bin/activate || exit 1
echo "${TIMESTAMP}: Env activated.."
python -c "import torch,transformers;print('  transformers',transformers.__version__,'| torch',torch.__version__)"
python -c "import transformers,sys;sys.exit(0 if transformers.__version__.startswith('4.') else 1)" || { echo "ERROR: transformers is not 4.x -- venv not active, aborting"; exit 1; }

export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$MPLCONFIGDIR"

cd $REPO
git stash
git pull
echo "${TIMESTAMP}: GPU Connected:"
nvidia-smi
echo "Free space where hidden states will land:"
df -h "$HARP"

bash harp_patch_devices.sh "$HARP" || exit 1

mkdir -p "${HARP}/dataset"
cp "${REPO}/harp_input/qwen-2.5-7b-instruct/[known]nq_open.jsonl"   "${HARP}/dataset/" || exit 1
cp "${REPO}/harp_input/qwen-2.5-7b-instruct/[unknown]nq_open.jsonl" "${HARP}/dataset/" || exit 1
echo "Line counts (known / unknown prompts):"
wc -l "${HARP}/dataset/[known]nq_open.jsonl" "${HARP}/dataset/[unknown]nq_open.jsonl"

echo "${TIMESTAMP}: Running HARP main.py.. "
cd $HARP
python ./main.py \
    --train_ratio=0.75 \
    --train_batch_size=128 \
    --train_epochs=50 \
    --model_name="Qwen2.5-7B-Instruct" \
    --dataset_name="nq_open"

echo "Exit status: $?"
du -sh "${HARP}/hidden_state" "${HARP}/svd" "${HARP}/mlp_result" 2>/dev/null

deactivate 2>/dev/null || true
conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
