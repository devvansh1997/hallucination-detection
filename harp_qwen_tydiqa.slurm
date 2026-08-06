#!/bin/bash
#SBATCH --job-name=harp-qwen-tydiqa
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/harp-qwen-tydiqa_%j/harp-qwen-tydiqa_%j.out
#SBATCH --error=slurm_logs/harp-qwen-tydiqa_%j/harp-qwen-tydiqa_%j.err

# RUN THIS ONE FIRST, ALONE. Two reasons:
#   1. main.py:109 keys the SVD cache on MODEL ONLY, not dataset:
#          svd_path = f"./{svd_dir}/[{model_quantity}]unembedding_svd.pt"
#      Concurrent Qwen jobs would all find it missing, all compute it, and all write the
#      same ~2GB file. Running one job first materialises it; later jobs just load it.
#   2. main.py:64-68 mkdir() is check-then-create, so concurrent first-starts can race
#      into FileExistsError. One job first creates the whole tree.
# TyDiQA is the cheapest smoke test: only 4,400 beams, and it is the ONE dataset that
# stores answer tokens only (need_prompt=not have_context, main.py:167+213) -> ~1GB.

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

export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}      # matplotlib font-cache race (mlp_trainer.py:9)
export TOKENIZERS_PARALLELISM=false
mkdir -p "$MPLCONFIGDIR"

cd $REPO
git stash
git pull
echo "${TIMESTAMP}: GPU Connected:"
nvidia-smi

# ---- the only edit to their code: two device strings (see script for justification) ----
bash harp_patch_devices.sh "$HARP" || exit 1

# ---- hand our generations over via their own generation cache (main.py:120-123) --------
# Filenames contain [square brackets], which are glob characters -- quote every path.
mkdir -p "${HARP}/dataset"
cp "${REPO}/harp_input/qwen-2.5-7b-instruct/[known]tydiqa.jsonl"   "${HARP}/dataset/" || exit 1
cp "${REPO}/harp_input/qwen-2.5-7b-instruct/[unknown]tydiqa.jsonl" "${HARP}/dataset/" || exit 1
echo "Staged jsonl into ${HARP}/dataset:"
ls -la "${HARP}/dataset/"
echo "Line counts (known / unknown prompts):"
wc -l "${HARP}/dataset/[known]tydiqa.jsonl" "${HARP}/dataset/[unknown]tydiqa.jsonl"

# ---- their code, their command line, straight from their README ------------------------
echo "${TIMESTAMP}: Running HARP main.py.. "
cd $HARP
python ./main.py \
    --train_ratio=0.75 \
    --train_batch_size=128 \
    --train_epochs=50 \
    --model_name="Qwen2.5-7B-Instruct" \
    --dataset_name="tydiqa"

echo "Exit status: $?"
echo "Hidden state artifacts written:"
du -sh "${HARP}/hidden_state" "${HARP}/svd" "${HARP}/mlp_result" 2>/dev/null

deactivate 2>/dev/null || true
conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
