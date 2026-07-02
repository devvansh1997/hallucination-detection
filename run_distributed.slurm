#!/bin/bash
# ==============================================================================
# run_distributed.slurm
# ==============================================================================
# Distributed dual-extraction across 8-way SLURM job array.
# Each task processes 1/8 of the 50% dataset slice for all 3 QA datasets.
#
# Submit with:  sbatch run_distributed.slurm
#
# After completion, merge with:
#   python 01b_merge_shards.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset truthfulqa --num_shards 8
#   python 01b_merge_shards.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset triviaqa   --num_shards 8
#   python 01b_merge_shards.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset tydiqa     --num_shards 8
# ==============================================================================

#SBATCH --job-name=dual_extract
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=preempt

echo "=============================================="
echo " SLURM Job ID:    $SLURM_JOB_ID"
echo " Array Task ID:   $SLURM_ARRAY_TASK_ID"
echo " Node:            $SLURMD_NODENAME"
echo "=============================================="

# ---- Environment setup (adjust to your cluster) ----
module load cuda/12.6
source /path/to/your/conda/bin/activate holo
cd /path/to/hallucination-detection

echo "Virtual environment activated"
echo "Starting distributed dual-extraction"

MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
SHARDS=8
IDX=$SLURM_ARRAY_TASK_ID

# ---- TruthfulQA ----
python 01_generate_and_extract.py \
    --model $MODEL --dataset truthfulqa \
    --num_shards $SHARDS --shard_idx $IDX

# ---- TriviaQA ----
python 01_generate_and_extract.py \
    --model $MODEL --dataset triviaqa \
    --num_shards $SHARDS --shard_idx $IDX

# ---- TyDiQA ----
python 01_generate_and_extract.py \
    --model $MODEL --dataset tydiqa \
    --num_shards $SHARDS --shard_idx $IDX

echo ""
echo "=============================================="
echo " Task $IDX complete"
echo "=============================================="
