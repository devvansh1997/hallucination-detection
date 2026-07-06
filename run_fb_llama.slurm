#!/bin/bash
#SBATCH --job-name=fullbeam-llama
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH -C gmem80
#SBATCH --time=24:00:00
#SBATCH --output=slurm_fb_llama_%j.out
#SBATCH --error=slurm_fb_llama_%j.err

echo "Loading modules"
module load anaconda3/2022.05
module load cuda/12.6
source /share/apps/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate hal-det
cd /home/devansh/Hallucination-Detection/hallucination-detection

echo "Starting: LLaMA-3.1-8B Full-Beam TriviaQA"
python 01_generate_full_beams.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset triviaqa

conda deactivate
echo "Done."
