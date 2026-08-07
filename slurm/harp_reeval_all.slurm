#!/bin/bash
#SBATCH --job-name=harp-reeval-qwen
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/harp-reeval-qwen_%j/harp-reeval-qwen_%j.out
#SBATCH --error=slurm_logs/harp-reeval-qwen_%j/harp-reeval-qwen_%j.err

# THE EXPERIMENT: HARP's method, their hidden states, their SVD, their projection, their MLP,
# their hyperparameters -- with ONLY the unit of the 75/25 split changed.
#
#   arm "answer"    their utils.split_data verbatim. CONTROL. Must land near the numbers their
#                   own main.py printed (truthful_qa 86.27, trivia_qa 92.89, nq_open 83.12,
#                   tydiqa 91.87). If it does not, the harness is wrong and the other arm is
#                   meaningless -- read the two together or not at all.
#   arm "question"  the paper's protocol: 75/25 over known QUESTIONS, taken whole.
#
# proj_dim 256 ONLY. Their S5.3 fixes the reasoning subspace dimension at 256 globally and every
# Table 1 value is reported at it, so it is the only cell we quote. The full 7-dim sweep x 5
# seeds x 2 arms would be 70 MLP trainings on TriviaQA's 47,370 training answers for numbers
# that never appear in the report.
#
# Datasets run smallest-first so a failure surfaces in minutes, not after the 52G load.
# mem=200G is sized for TriviaQA: ~52G of fp32 per-token states in host RAM, projected copy
# (~3.6G at proj_dim 256) on the GPU.
# Reads HARP-Code read-only. Writes only to results/harp_reeval/ in our repo.

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPO=/home/de807845/Hallucination-Detection/hallucination-detection
HARP=/home/de807845/Hallucination-Detection/HARP-Code

echo "${TIMESTAMP}: Loading modules.. "
module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0

conda activate hal-det
source $HOME/envs/harp-rerun/bin/activate || exit 1
echo "${TIMESTAMP}: Env activated.."
python -c "import torch,transformers;print('  transformers',transformers.__version__,'| torch',torch.__version__)"

export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$MPLCONFIGDIR"

cd $REPO
echo "${TIMESTAMP}: GPU Connected:"
nvidia-smi

for DS in tydiqa truthful_qa nq_open trivia_qa; do
    echo ""
    echo "############################################################"
    echo "###  $DS"
    echo "############################################################"
    python 50_harp_reeval_prompt_level.py \
        --dataset "$DS" \
        --model_name "Qwen2.5-7B-Instruct" \
        --harp-dir "$HARP" \
        --arms answer,question \
        --proj-dims 256 \
        --seeds 5
    echo "### $DS exit status: $?"
done

deactivate 2>/dev/null || true
conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
