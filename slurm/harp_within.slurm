#!/bin/bash
#SBATCH --job-name=harp-within
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# HARP's WITHIN-QUESTION AUROC (T-009), for the three-detector table Dr. Ibrahim asked for
# (LOTUS, HARP, ACT-ViT; the other baselines score once per question, so the metric is 50 for them
# by construction).
#
#   sbatch --export=ALL slurm/harp_within.slurm
#
# WHAT IT RUNS. 50_harp_reeval_prompt_level.py --within, question arm only, proj_dim 256, five
# seeds, on the two datasets where ACT-ViT also exists, for both models. Their code is untouched:
# their validate() already reduces each answer to one max-over-tokens score, and --within wraps the
# function it hands that vector to. Each seed asserts that the AUROC recomputed from the captured
# scores equals the one their trainer reported, so a misalignment stops the job instead of
# producing a plausible wrong number.
#
# WRITES results/harp_within/ ONLY -- not results/harp_reeval/, whose files hold the answer-arm
# numbers the split appendix already cites and which a question-only rerun would overwrite.
#
# Preamble is harp_reeval_all.slurm's: HARP needs the 4.5x venv layered over hal-det (conda first,
# venv second, or conda's PATH wins), and we assert the version rather than trusting activation.

set -u
REPO=/home/de807845/Hallucination-Detection/hallucination-detection
HARP=/home/de807845/Hallucination-Detection/HARP-Code
OUT="$REPO/results/harp_within"

for _v in $(compgen -v | grep '^CONDA' || true); do unset "$_v"; done
unset _CE_M _CE_CONDA || true
module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0
conda activate hal-det
source $HOME/envs/harp-rerun/bin/activate || exit 1
python -c "import transformers,sys;sys.exit(0 if transformers.__version__.startswith('4.') else 1)" \
    || { echo "ERROR: transformers is not 4.x -- the venv is not active"; exit 1; }

export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$MPLCONFIGDIR" "$OUT"
cd "$REPO" || exit 1
echo "repo HEAD: $(git -C $REPO log --oneline -1 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

status=0
MODELS="${MODELS:-Qwen2.5-7B-Instruct Llama-3.1-8B}"
DATASETS="${DATASETS:-tydiqa truthful_qa}"
for MODEL in $MODELS; do
    # HARP-Code/dataset/ holds ONE model's jsonl at a time, and 50 reads it to recover the prompt
    # ids for their flat hidden-state list. On 2026-09-23 the Qwen runs found LLaMA's jsonl there
    # and stopped on the length assertion (4,040 answers implied against 3,020 states). Copy this
    # model's own files first, exactly as the harp stage of pipe_stage.slurm does.
    case "$MODEL" in
        Qwen2.5-7B-Instruct) OURS=qwen-2.5-7b-instruct ;;
        Llama-3.1-8B)        OURS=llama-3.1-8b ;;
        *) echo "ERROR: no harp_input folder known for $MODEL"; exit 1 ;;
    esac
    for DS in $DATASETS; do
        echo ""
        echo "############################################################"
        echo "###  $MODEL / $DS"
        echo "############################################################"
        cp "$REPO/harp_input/$OURS/[known]$DS.jsonl"   "$HARP/dataset/" || { status=1; continue; }
        cp "$REPO/harp_input/$OURS/[unknown]$DS.jsonl" "$HARP/dataset/" || { status=1; continue; }
        echo "    dataset jsonl in place from harp_input/$OURS"
        python 50_harp_reeval_prompt_level.py \
            --dataset "$DS" \
            --model_name "$MODEL" \
            --harp-dir "$HARP" \
            --arms question \
            --within \
            --proj-dims 256 \
            --seeds 5 \
            --out-dir "$OUT" || status=1
        echo "### $MODEL/$DS exit status: $?"
    done
done

deactivate 2>/dev/null || true
conda deactivate
echo "done (exit $status). Results: $OUT/<model>_<dataset>_reeval.json -> arms.question.256.within_mean"
exit $status
