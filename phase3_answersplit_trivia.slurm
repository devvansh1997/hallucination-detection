#!/bin/bash
#SBATCH --job-name=ansplit-tri
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/ansplit-tri_%j/ansplit-tri_%j.out
#SBATCH --error=slurm_logs/ansplit-tri_%j/ansplit-tri_%j.err

# TriviaQA half of the fourth cell: OUR method under HARP's RELEASED (answer-level) split.
#
# ONE CONDITION PER JOB, taking the condition as $1 -- same pattern as the original
# phase3_tri_*.slurm set, because a single TriviaQA condition runs for hours and six of them
# sequentially would not fit any time limit. Fire all six so this dataset's answer-level cell is
# a best-of-six, chosen the same way as its question-level cell:
#
#   for C in core_max q_velocity q_static core_concat joint_tensor triple_concat; do
#       sbatch phase3_answersplit_trivia.slurm $C
#   done
#
# Cluster contention measured at ~4x slowdown around 7 concurrent jobs, so consider firing these
# in two batches of three rather than all six at once.
#
# Then merge them exactly as the question-level run did:
#   python 44_eval_phase3.py --dataset triviaqa --model_folder qwen-2.5-7b-instruct \
#       --combine-conditions --split-unit answer --results-dir results/qwen-2.5-7b-instruct-answersplit
#
# TriviaQA is also the one dataset where logistic regression beats the random forest for every
# condition -- read the LR column here, not just RF.
#
# --results-dir is MANDATORY: output filenames key on dataset+condition only, so without it this
# overwrites the question-level TriviaQA results the comparison depends on.

COND=${1:-core_max}

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPO=/home/de807845/Hallucination-Detection/hallucination-detection
OUT=results/qwen-2.5-7b-instruct-answersplit

echo "${TIMESTAMP}: Loading modules.. "
module load anaconda/anaconda-2024.10

conda activate hal-det
echo "${TIMESTAMP}: Conda Env Activated.. "
cd $REPO

# --- provenance: makes a stale copy of this script impossible to miss -------------------------
# This file can be submitted from anywhere (it cd's to an absolute $REPO), which also means a
# forgotten copy outside the repo will run happily and silently ignore every git pull. Print
# which file SLURM actually launched and what the repo is at, so the log answers "am I running
# the version I think I am?" without anyone having to guess.
echo "--- provenance ---"
echo "  submitted script : ${SLURM_SUBMIT_DIR}/$(basename "$0")"
echo "  submit dir       : ${SLURM_SUBMIT_DIR}"
echo "  repo HEAD        : $(git -C $REPO log --oneline -1 2>/dev/null)"
echo "  repo dirty?      : $(git -C $REPO status --porcelain 2>/dev/null | wc -l) modified file(s)"
echo "  condition        : ${COND}"
echo "------------------"

echo "${TIMESTAMP}: triviaqa / ${COND} / answer-level split"
# --part a is REQUIRED alongside --condition (44_eval_phase3.py:1424 rejects the call without
# it); --condition narrows that part to a single condition rather than running all six. This
# matches the original phase3_tri_*.slurm invocation exactly.
python 44_eval_phase3.py \
    --dataset triviaqa \
    --model_folder qwen-2.5-7b-instruct \
    --part a \
    --condition "$COND" \
    --split-unit answer \
    --results-dir "$OUT"
echo "Exit status: $?"

ls -la "${REPO}/${OUT}" 2>/dev/null

conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
