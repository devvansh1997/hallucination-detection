#!/bin/bash
#SBATCH --job-name=ansplit-small
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/ansplit-small_%j/ansplit-small_%j.out
#SBATCH --error=slurm_logs/ansplit-small_%j/ansplit-small_%j.err

# THE FOURTH CELL of the method x protocol grid: OUR method under HARP's RELEASED split
# (answers pooled and shuffled) rather than the paper's (questions taken whole).
#
# Runs ONE condition per dataset -- the one that already wins under the question-level split --
# so our two rows in the report differ in NOTHING but the split unit, exactly parallel to the two
# HARP rows. Best-of-six on one side against a single condition on the other would not be a
# controlled contrast.
#     tydiqa_gp   core_max      (88.3 question-level)
#     truthfulqa  triple_concat (87.8)
#     nq_open     core_concat   (79.4)
# TriviaQA runs separately (q_static, 10h, 128G) -- see phase3_answersplit_trivia.slurm.
#
# --results-dir is MANDATORY here. Output filenames key on dataset+condition only, so without it
# these would silently overwrite the question-level results this whole comparison rests on.
# CPU only: 44 does no GPU work.

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPO=/home/de807845/Hallucination-Detection/hallucination-detection
OUT=results/qwen-2.5-7b-instruct-answersplit

echo "${TIMESTAMP}: Loading modules.. "
module load anaconda/anaconda-2024.10

conda activate hal-det
echo "${TIMESTAMP}: Conda Env Activated.. "
cd $REPO

for PAIR in "tydiqa_gp core_max" "truthfulqa triple_concat" "nq_open core_concat"; do
    set -- $PAIR
    DS=$1
    COND=$2
    echo ""
    echo "############################################################"
    echo "###  $DS / $COND / answer-level split"
    echo "############################################################"
    python 44_eval_phase3.py \
        --dataset "$DS" \
        --model_folder qwen-2.5-7b-instruct \
        --condition "$COND" \
        --split-unit answer \
        --results-dir "$OUT"
    echo "### $DS exit status: $?"
done

echo ""
echo "Answer-split results written to ${REPO}/${OUT}:"
ls -la "${REPO}/${OUT}" 2>/dev/null

conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
