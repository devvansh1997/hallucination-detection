#!/bin/bash
#SBATCH --job-name=halDet-triv
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH -C gmem80
#SBATCH --time=20:00:00
#SBATCH --output=slurm_triviaqa_%j.out
#SBATCH --error=slurm_triviaqa_%j.err

echo "Loading modules"
module load anaconda3/2022.05
module load cuda/12.6
source /share/apps/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate hal-det
cd /home/devansh/Hallucination-Detection/hallucination-detection

python 01_generate_and_extract.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset triviaqa

conda deactivate
