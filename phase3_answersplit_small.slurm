#!/bin/bash
#SBATCH --job-name=ansplit-small
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/ansplit-small_%j/ansplit-small_%j.out
#SBATCH --error=slurm_logs/ansplit-small_%j/ansplit-small_%j.err

# THE FOURTH CELL of the method x protocol grid: OUR method under HARP's RELEASED split
# (answers pooled and shuffled) rather than the paper's (questions taken whole).
#
# ALL SIX CONDITIONS per dataset, via --part a. An earlier version of this job ran only the
# single condition that wins under the QUESTION-level split, on the theory that holding the
# condition fixed made the contrast cleaner. That was wrong: it selects the condition on one arm
# and denies the other arm the same choice, so the question-level cell is a best-of-six maximum
# while the answer-level cell is a pre-committed single. The bias runs in our favour -- if some
# other condition exploits the leaky split better, pinning to the question-level winner
# understates our answer-level score and makes our method look more leakage-resistant than it is.
# Both cells are now best-of-six, chosen the same way.
#
# --results-dir is MANDATORY. Output filenames key on dataset+condition only, so without it these
# silently overwrite the question-level results the whole comparison rests on.
# CPU only: 44 does no GPU work. Datasets run smallest-first; 44 persists each condition's result
# as it finishes, so a time-limit kill loses only the condition in flight.

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPO=/home/de807845/Hallucination-Detection/hallucination-detection
OUT=results/qwen-2.5-7b-instruct-answersplit

echo "${TIMESTAMP}: Loading modules.. "
module load anaconda/anaconda-2024.10

conda activate hal-det
echo "${TIMESTAMP}: Conda Env Activated.. "
cd $REPO

for DS in tydiqa_gp truthfulqa nq_open; do
    echo ""
    echo "############################################################"
    echo "###  $DS -- all 6 conditions, answer-level split"
    echo "############################################################"
    python 44_eval_phase3.py \
        --dataset "$DS" \
        --model_folder qwen-2.5-7b-instruct \
        --part a \
        --split-unit answer \
        --results-dir "$OUT"
    echo "### $DS exit status: $?"
done

echo ""
echo "Answer-split results written to ${REPO}/${OUT}:"
ls -la "${REPO}/${OUT}" 2>/dev/null

conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
