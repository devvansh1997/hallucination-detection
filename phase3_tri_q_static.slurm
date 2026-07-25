#!/bin/bash
#SBATCH --job-name=tri44-q_static
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/tri44-q_static_%j/tri44-q_static_%j.out
#SBATCH --error=slurm_logs/tri44-q_static_%j/tri44-q_static_%j.err

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
python 44_eval_phase3.py --dataset triviaqa --part a --condition q_static
conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
