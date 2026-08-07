#!/bin/bash
#SBATCH --job-name=debug-attn
#SBATCH -c 2
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH -C gmem80
#SBATCH --time=00:30:00
#SBATCH --output=slurm_debug_%j.out
#SBATCH --error=slurm_debug_%j.err

module load anaconda3/2022.05
module load cuda/12.6
source /share/apps/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate hal-det
cd /home/devansh/Hallucination-Detection/hallucination-detection

python debug_attn_shapes.py
