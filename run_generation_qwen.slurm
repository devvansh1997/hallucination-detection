#!/bin/bash
#SBATCH --job-name=halDet-qwen
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH -C gmem80
#SBATCH --time=28:00:00
#SBATCH --output=slurm_qwen_%j.out
#SBATCH --error=slurm_qwen_%j.err

echo "Loading modules"
module load anaconda3/2022.05
module load cuda/12.6

source /share/apps/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate hal-det
echo "Virtual environment activated"

cd /home/devansh/Hallucination-Detection/hallucination-detection

# Run all 4 datasets for Qwen-2.5-7B
echo "Starting: Qwen-2.5-7B-Instruct"
python 01_generate_and_extract.py --model Qwen/Qwen2.5-7B-Instruct --dataset truthfulqa
python 01_generate_and_extract.py --model Qwen/Qwen2.5-7B-Instruct --dataset triviaqa
python 01_generate_and_extract.py --model Qwen/Qwen2.5-7B-Instruct --dataset nq_open
python 01_generate_and_extract.py --model Qwen/Qwen2.5-7B-Instruct --dataset tydiqa_gp

conda deactivate
echo "Done."
