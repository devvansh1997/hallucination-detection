#!/bin/bash
#SBATCH --job-name=qwen42-nq_open
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/qwen42-nq_open_%j/qwen42-nq_open_%j.out
#SBATCH --error=slurm_logs/qwen42-nq_open_%j/qwen42-nq_open_%j.err

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

echo "${TIMESTAMP}: Loading modules.. "
module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0

conda activate hal-det
echo "${TIMESTAMP}: Conda Env Activated.. "
cd /home/de807845/Hallucination-Detection/hallucination-detection
git stash
git pull
echo "${TIMESTAMP}: GPU Connected:"
nvidia-smi

echo "${TIMESTAMP}: Running python code.. "
python 42_extract_phase2.py --dataset nq_open --model_folder qwen-2.5-7b-instruct
conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
