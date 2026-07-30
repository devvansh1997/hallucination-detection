#!/bin/bash
#SBATCH --job-name=qwen42-tydiqa_gp
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/qwen42-tydiqa_gp_%j/qwen42-tydiqa_gp_%j.out
#SBATCH --error=slurm_logs/qwen42-tydiqa_gp_%j/qwen42-tydiqa_gp_%j.err

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
python 42_extract_phase2.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct
conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
