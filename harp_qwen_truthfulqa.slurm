#!/bin/bash
#SBATCH --job-name=harp-qwen-truthfulqa
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/harp-qwen-truthfulqa_%j/harp-qwen-truthfulqa_%j.out
#SBATCH --error=slurm_logs/harp-qwen-truthfulqa_%j/harp-qwen-truthfulqa_%j.err

# FIRE ONLY AFTER harp_qwen_tydiqa HAS FINISHED -- it materialises the model-keyed SVD
# cache (main.py:109) that this job will otherwise race against. Safe to run alongside
# harp_qwen_nq_open once that has happened; their dataset/, hidden_state/ and mlp_result/
# filenames are all keyed on dataset name, so the two do not collide.
#
# ~8,170 beams, prompt+answer hidden states stored -> expect ~6GB in hidden_state/.
# This is the dataset where the split finding predicts the LARGEST inflation: roughly half
# of its test answers come from questions that also appear in training.

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPO=/home/de807845/Hallucination-Detection/hallucination-detection
HARP=/home/de807845/Hallucination-Detection/HARP-Code

echo "${TIMESTAMP}: Loading modules.. "
module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0

conda activate hal-det
echo "${TIMESTAMP}: Conda Env Activated.. "

export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$MPLCONFIGDIR"

cd $REPO
git stash
git pull
echo "${TIMESTAMP}: GPU Connected:"
nvidia-smi

bash harp_patch_devices.sh "$HARP" || exit 1

mkdir -p "${HARP}/dataset"
cp "${REPO}/harp_input/qwen-2.5-7b-instruct/[known]truthful_qa.jsonl"   "${HARP}/dataset/" || exit 1
cp "${REPO}/harp_input/qwen-2.5-7b-instruct/[unknown]truthful_qa.jsonl" "${HARP}/dataset/" || exit 1
echo "Line counts (known / unknown prompts):"
wc -l "${HARP}/dataset/[known]truthful_qa.jsonl" "${HARP}/dataset/[unknown]truthful_qa.jsonl"

echo "${TIMESTAMP}: Running HARP main.py.. "
cd $HARP
python ./main.py \
    --train_ratio=0.75 \
    --train_batch_size=128 \
    --train_epochs=50 \
    --model_name="Qwen2.5-7B-Instruct" \
    --dataset_name="truthful_qa"

echo "Exit status: $?"
du -sh "${HARP}/hidden_state" "${HARP}/svd" "${HARP}/mlp_result" 2>/dev/null

conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
