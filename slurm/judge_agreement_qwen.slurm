#!/bin/bash
#SBATCH --job-name=judge-agree
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/judge-agree_%j/judge-agree_%j.out
#SBATCH --error=slurm_logs/judge-agree_%j/judge-agree_%j.err

# AUDIT #1 -- does OUR correctness label match what HARP's own DatasetJudge produces?
#
# This is the audit the EXTERNAL claim rests on. "Changing only the split unit costs 3.8-13.3
# points" holds no matter what the labels are, because both arms share them. "HARP's honest
# TyDiQA number is 79.9, not 88.4" needs our known/unknown partition to be the one their paper
# was computed over -- which is exactly what this measures.
#
# NO REGENERATION. 49_harp_adapter.py already wrote gen_text for every beam plus the correct /
# incorrect answer lists into harp_input/. That is precisely DatasetJudge.judge()'s input. No LLM
# is loaded, nothing is generated, HARP-Code is read-only.
#
# Needs the harp-rerun venv: DatasetJudge imports bleurt_pytorch, which needs transformers 4.5x.
# First run downloads BLEURT-20 (~2GB) from HF into the default cache.
#
# ORDER is smallest-first so a failure surfaces in minutes:
#   tydiqa 4,400 beams | truthful_qa 8,170 | nq_open 36,100 | trivia_qa 99,600
# TriviaQA is the slow one -- its answer lists are long (value + aliases), and the judge scores
# each beam against every one of them.
#
# DEVICE. Their judge never moves BLEURT off the CPU, so cpu is the faithful setting. We use cuda
# for throughput and then MEASURE whether that mattered: --check-device 300 re-judges the first
# 300 beams on cpu and reports how many labels flip. A BLEURT score shifting by 1e-4 across the
# 0.5 threshold flips a label, and that is the very quantity under test -- so it is measured
# rather than assumed. If flips are non-zero, rerun the affected dataset with --device cpu.

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPO=/home/de807845/Hallucination-Detection/hallucination-detection

echo "${TIMESTAMP}: Loading modules.. "
module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0

conda activate hal-det
source $HOME/envs/harp-rerun/bin/activate || exit 1
python -c "import torch,transformers;print('  transformers',transformers.__version__,'| torch',torch.__version__)"
python -c "import transformers,sys;sys.exit(0 if transformers.__version__.startswith('4.') else 1)" || { echo "ERROR: transformers is not 4.x -- venv not active, aborting"; exit 1; }

export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$MPLCONFIGDIR"

cd $REPO
echo "--- provenance ---"
echo "  submitted script : ${SLURM_SUBMIT_DIR}/$(basename "$0")"
echo "  repo HEAD        : $(git -C $REPO log --oneline -1 2>/dev/null)"
echo "------------------"
nvidia-smi

# Pilot first: 50 prompts of the smallest dataset, plus the cpu-vs-gpu label-flip check.
# If BLEURT-20 fails to download, or the device check shows flips, we find out in ~2 minutes
# rather than after TriviaQA has run for hours.
echo ""
echo "### PILOT: tydiqa, 50 prompts, with cpu-vs-gpu device check"
python 51_judge_agreement.py --dataset tydiqa --model_folder qwen-2.5-7b-instruct \
    --device cuda --limit-prompts 50 --check-device 300 || exit 1

for DS in tydiqa truthful_qa nq_open trivia_qa; do
    echo ""
    echo "############################################################"
    echo "###  $DS"
    echo "############################################################"
    python 51_judge_agreement.py --dataset "$DS" --model_folder qwen-2.5-7b-instruct --device cuda
    echo "### $DS exit status: $?"
done

echo ""
ls -la "${REPO}/results/judge_agreement" 2>/dev/null

deactivate 2>/dev/null || true
conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
