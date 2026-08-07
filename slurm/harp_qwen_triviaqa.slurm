#!/bin/bash
#SBATCH --job-name=harp-qwen-triviaqa
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/harp-qwen-triviaqa_%j/harp-qwen-triviaqa_%j.out
#SBATCH --error=slurm_logs/harp-qwen-triviaqa_%j/harp-qwen-triviaqa_%j.err

# Largest of the four: 99,600 beams, 6,316 known / 3,644 unknown questions.
# Sized from three completed runs rather than a guess:
#     tydiqa      4,400 beams -> 1.2G   (answer tokens only)
#     truthfulqa  8,170 beams -> 5.4G   (661 KB/beam)
#     nq_open    36,100 beams -> 18.4G  (510 KB/beam)
#   => triviaqa  99,600 beams -> ~50-66G of hidden_state/. Quota checked: 290G of 1000G used.
#
# mem=200G because the patch puts data_device on cpu, so the whole ~60G of fp32 per-token
# states accumulates in host RAM before torch.save (main.py:206-224) and stays resident
# through the proj_dim loop. GPU side is the fp32 model (~30G) plus the projected copy
# (~14G at proj_dim 1024) -> comfortable on 80G.
#
# time=06:00:00 against an estimate of 2-3h. Extraction scales with beams (~60 min), but here
# the MLP sweep is the bigger term: 47,370 training answers x 7 proj_dims x 50 epochs, versus
# 2,700 training answers for nq_open.
#
# CAVEAT ON THE COMPARISON, not on the run: TriviaQA is the ONE dataset where our correctness
# labels differ from theirs by construction. Their DatasetInit.py:103 folds normalized_aliases
# into the correct-answer list; our 39_generate_dataset.py uses value + aliases only. Their
# accepted-answer set is broader, so their published 92.8 was computed over a different
# known/unknown partition than ours. Read this dataset's gap with that in mind.

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPO=/home/de807845/Hallucination-Detection/hallucination-detection
HARP=/home/de807845/Hallucination-Detection/HARP-Code

echo "${TIMESTAMP}: Loading modules.. "
module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0

conda activate hal-det
# HARP targets the transformers 4.5x API (main.py:105 uses model.model.config, and
# bleurt_pytorch needs pytorch_utils.find_pruneable_heads_and_indices -- both gone in
# 5.13). This venv was built with --system-site-packages, so its transformers 4.51
# shadows hal-det's 5.13 while torch/CUDA come through from the conda env unchanged.
source $HOME/envs/harp-rerun/bin/activate || exit 1
echo "${TIMESTAMP}: Env activated.."
python -c "import torch,transformers;print('  transformers',transformers.__version__,'| torch',torch.__version__)"
python -c "import transformers,sys;sys.exit(0 if transformers.__version__.startswith('4.') else 1)" || { echo "ERROR: transformers is not 4.x -- venv not active, aborting"; exit 1; }

export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$MPLCONFIGDIR"

cd $REPO
echo "${TIMESTAMP}: GPU Connected:"
nvidia-smi
echo "Quota and free space before starting:"
lfs quota -h -u $USER /lustre/fs1 2>/dev/null || df -h "$HARP"

bash harp_patch_devices.sh "$HARP" || exit 1

mkdir -p "${HARP}/dataset"
cp "${REPO}/harp_input/qwen-2.5-7b-instruct/[known]trivia_qa.jsonl"   "${HARP}/dataset/" || exit 1
cp "${REPO}/harp_input/qwen-2.5-7b-instruct/[unknown]trivia_qa.jsonl" "${HARP}/dataset/" || exit 1
echo "Line counts (known / unknown prompts) -- expect 6316 / 3644:"
wc -l "${HARP}/dataset/[known]trivia_qa.jsonl" "${HARP}/dataset/[unknown]trivia_qa.jsonl"

echo "${TIMESTAMP}: Running HARP main.py.. "
cd $HARP
python ./main.py \
    --train_ratio=0.75 \
    --train_batch_size=128 \
    --train_epochs=50 \
    --model_name="Qwen2.5-7B-Instruct" \
    --dataset_name="trivia_qa"

echo "Exit status: $?"
du -sh "${HARP}/hidden_state" "${HARP}/svd" "${HARP}/mlp_result" 2>/dev/null
lfs quota -h -u $USER /lustre/fs1 2>/dev/null

deactivate 2>/dev/null || true
conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
