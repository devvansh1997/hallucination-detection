#!/bin/bash
#SBATCH --job-name=qwen39-triviaqa
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --output=slurm_logs/qwen39-triviaqa_%j/qwen39-triviaqa_%j.out
#SBATCH --error=slurm_logs/qwen39-triviaqa_%j/qwen39-triviaqa_%j.err

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
python 39_generate_dataset.py --dataset triviaqa --model_folder qwen-2.5-7b-instruct
python 40_validate_dataset.py --manifest ../data/qwen-2.5-7b-instruct/manifest_triviaqa_v1.json \
    --model_folder qwen-2.5-7b-instruct --dataset triviaqa
conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
