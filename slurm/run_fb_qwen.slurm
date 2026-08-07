#!/bin/bash
#SBATCH --job-name=fullbeam-qwen
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH -C gmem80
#SBATCH --time=24:00:00
#SBATCH --output=slurm_fb_qwen_%j.out
#SBATCH --error=slurm_fb_qwen_%j.err

echo "Loading modules"
module load anaconda3/2022.05
module load cuda/12.6
source /share/apps/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate hal-det
cd /home/devansh/Hallucination-Detection/hallucination-detection

echo "Starting: Qwen-2.5-7B Full-Beam TriviaQA"
python 01_generate_full_beams.py --model Qwen/Qwen2.5-7B-Instruct --dataset triviaqa

conda deactivate
echo "Done."
