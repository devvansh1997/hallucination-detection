#!/bin/bash
#SBATCH --job-name=ansplit-tri
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/ansplit-tri_%j/ansplit-tri_%j.out
#SBATCH --error=slurm_logs/ansplit-tri_%j/ansplit-tri_%j.err

# TriviaQA half of the fourth cell: OUR method under HARP's RELEASED (answer-level) split.
# Same mem/time as the original phase3_tri_*.slurm single-condition jobs, since the work is the
# same -- one condition, 99,600 beams, 5 seeds; only the row selection per seed differs.
#
# q_static is the condition that wins TriviaQA under the question-level split (92.7, and TriviaQA
# is the one dataset where LR beats RF for every condition -- read the LR column here too).
#
# --results-dir is MANDATORY: output filenames key on dataset+condition only, so without it this
# overwrites the question-level TriviaQA result the comparison depends on.

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPO=/home/de807845/Hallucination-Detection/hallucination-detection
OUT=results/qwen-2.5-7b-instruct-answersplit

echo "${TIMESTAMP}: Loading modules.. "
module load anaconda/anaconda-2024.10

conda activate hal-det
echo "${TIMESTAMP}: Conda Env Activated.. "
cd $REPO

echo "${TIMESTAMP}: Running python code.. "
python 44_eval_phase3.py \
    --dataset triviaqa \
    --model_folder qwen-2.5-7b-instruct \
    --condition q_static \
    --split-unit answer \
    --results-dir "$OUT"
echo "Exit status: $?"

ls -la "${REPO}/${OUT}" 2>/dev/null

conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
