#!/bin/bash
#SBATCH --job-name=tri44-core_concat
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/tri44-core_concat_%j/tri44-core_concat_%j.out
#SBATCH --error=slurm_logs/tri44-core_concat_%j/tri44-core_concat_%j.err

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
python 44_eval_phase3.py --dataset triviaqa --part a --condition core_concat
conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
