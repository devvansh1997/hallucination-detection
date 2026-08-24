#!/bin/bash
#SBATCH --job-name=hg-pilot
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/hg-pilot_%j/hg-pilot_%j.out
#SBATCH --error=slurm_logs/hg-pilot_%j/hg-pilot_%j.err

# PILOT ONLY -- deliberately does not attempt a full dataset.
#
# HalluGuard takes one autograd.grad call PER GENERATED TOKEN (halluguard_true.py:103-113), so
# cost scales with beams x tokens, not beams. Our completions average 8-14 tokens, which puts the
# full four-dataset sweep somewhere between 3 and 20 hours per model depending on per-backward
# cost -- a range too wide to budget from. This measures it on 200 beams and stops.
#
# Three things this pilot answers, all of which decide whether the full run is worth firing:
#   1. seconds/beam, so the real sweep can be budgeted instead of guessed
#   2. how many scores come back non-finite -- their models/_load_model.py defaults to float16
#      and this code path takes GRADIENTS, where fp16 can overflow in ways a forward-only
#      pipeline never shows. If many are non-finite we rerun with --dtype float32.
#   3. whether the AUROC is even in a plausible range on our data
#
# PREREQUISITE. The repo must be cloned ON THE CLUSTER, next to HARP-Code:
#   cd ~/Hallucination-Detection && git clone https://github.com/Susan571/HalluGuard-ICLR2026.git
# Only Score/halluguard_true.py is imported -- it depends on torch alone, so none of their
# requirements.txt, dataloaders or model wrappers are needed.

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPO=/home/de807845/Hallucination-Detection/hallucination-detection
HG=/home/de807845/Hallucination-Detection/HalluGuard-ICLR2026/Score

echo "${TIMESTAMP}: Loading modules.. "
module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0
conda activate hal-det

export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$MPLCONFIGDIR"

cd $REPO
echo "repo HEAD: $(git -C $REPO log --oneline -1 2>/dev/null)"
[ -f "$HG/halluguard_true.py" ] || {
    echo "ERROR: $HG/halluguard_true.py not found."
    echo "  cd ~/Hallucination-Detection && git clone https://github.com/Susan571/HalluGuard-ICLR2026.git"
    exit 1; }
nvidia-smi

# TyDiQA is the smallest dataset, so 200 beams is a real sample of it rather than a corner.
echo ""
echo "### PILOT 1/2: fp16 (their default), 200 beams"
python 53_halluguard_score.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct \
    --hg-dir "$HG" --dtype float16 --limit 200 --log-every 50 || exit 1

# Same 200 beams in fp32. If fp16 lost scores to overflow, this says so directly and tells us
# what the full run should use -- measured, not assumed.
echo ""
echo "### PILOT 2/2: fp32, same 200 beams, for comparison"
python 53_halluguard_score.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct \
    --hg-dir "$HG" --dtype float32 --limit 200 --log-every 50 \
    --out-dir "$REPO/results/halluguard_fp32" || exit 1

echo ""
echo "Compare the two 'finite' lines and the seconds/beam above before firing the full sweep."
conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
