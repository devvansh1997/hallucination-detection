#!/bin/bash
#SBATCH --job-name=pipe
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/pipe_%j/pipe_%j.out
#SBATCH --error=slurm_logs/pipe_%j/pipe_%j.err
#
# One parameterised stage of the full per-model pipeline. submit_pipeline.sh submits this
# repeatedly with different args and overrides --mem / --time / --gres / -p per stage, so the
# resource profile lives in the driver and the logic lives here. Do not sbatch this directly
# unless you also pass those overrides -- there are no useful defaults for them.
#
#   $1  stage    prefetch | gen | extract | eval | adapter | harp | summary
#   $2  model_folder     e.g. llama-3.1-8b
#   $3  dataset          our name (tydiqa_gp|truthfulqa|nq_open|triviaqa); unused by
#                        prefetch/adapter/summary
#
# ENVIRONMENTS. Our code runs under hal-det (transformers 5.13). HARP's code cannot: it needs
# the 4.5x API (main.py:105 reads model.model.config) and bleurt_pytorch needs
# pytorch_utils.find_pruneable_heads_and_indices, removed in 5.x. Only the 'harp' stage switches
# to the venv, and it asserts the version rather than trusting the activation.

set -u
STAGE="${1:?stage required}"
MODEL="${2:?model_folder required}"
DS="${3:-}"

REPO=/home/de807845/Hallucination-Detection/hallucination-detection
HARP=/home/de807845/Hallucination-Detection/HARP-Code
ANSWERSPLIT_DIR="results/${MODEL}-answersplit"

# our dataset name -> HARP's
case "$DS" in
    tydiqa_gp)  HDS=tydiqa ;;
    truthfulqa) HDS=truthful_qa ;;
    nq_open)    HDS=nq_open ;;
    triviaqa)   HDS=trivia_qa ;;
    *)          HDS="" ;;
esac

echo "=============================================================="
echo "  stage=${STAGE}  model=${MODEL}  dataset=${DS:-<n/a>}"
echo "  job=${SLURM_JOB_ID}  submitted from ${SLURM_SUBMIT_DIR}"
echo "=============================================================="

module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0
conda activate hal-det
export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$MPLCONFIGDIR"
cd "$REPO" || exit 1
echo "repo HEAD: $(git -C $REPO log --oneline -1 2>/dev/null)"

case "$STAGE" in

  prefetch)
    # Serialised on purpose. meta-llama/Llama-3.1-8B is gated and ~16GB; three generation jobs
    # starting cold would all pull it into the same cache at once. Everything else fans out
    # only after this succeeds, so the download happens exactly once.
    MODEL_ID=$(python -c "import yaml;c=yaml.safe_load(open('config.yaml'));print(next(m['id'] for m in c['models'] if m['folder']=='${MODEL}'))") || exit 1
    echo "prefetching ${MODEL_ID} ..."
    python - <<PYEOF || exit 1
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = "${MODEL_ID}"
AutoTokenizer.from_pretrained(mid)
AutoModelForCausalLM.from_pretrained(mid, dtype="auto", device_map="cpu")
print("  cached OK:", mid)
PYEOF
    ;;

  gen)
    python 39_generate_dataset.py --dataset "$DS" --model_folder "$MODEL" || exit 1
    python 40_validate_dataset.py --dataset "$DS" --model_folder "$MODEL" || exit 1
    ;;

  extract)
    python 42_extract_phase2.py --dataset "$DS" --model_folder "$MODEL" || exit 1
    ;;

  eval)
    echo "--- question-level (the paper's protocol; our headline numbers) ---"
    python 44_eval_phase3.py --dataset "$DS" --model_folder "$MODEL" --part a || exit 1
    echo "--- answer-level (HARP's released protocol; the leakage arm) ---"
    python 44_eval_phase3.py --dataset "$DS" --model_folder "$MODEL" --part a \
        --split-unit answer --results-dir "$ANSWERSPLIT_DIR" || exit 1
    ;;

  adapter)
    # ONCE per model, never per dataset: 49 writes a single adapter_summary.json per model
    # (49_harp_adapter.py:287), so parallel per-dataset runs would overwrite each other.
    # --dataset all skips any dataset whose sequences file is absent, so a partial run is fine.
    python 49_harp_adapter.py --dataset all --model_folder "$MODEL" || exit 1
    ;;

  harp)
    # HARP's own code. Needs the 4.5x venv layered over hal-det -- conda first, venv second,
    # or conda's PATH wins and we silently get transformers 5.13.
    source $HOME/envs/harp-rerun/bin/activate || exit 1
    python -c "import transformers,sys;sys.exit(0 if transformers.__version__.startswith('4.') else 1)" \
        || { echo "ERROR: transformers is not 4.x -- venv not active"; exit 1; }
    HKEY=$(python -c "import importlib.util,sys;s=importlib.util.spec_from_file_location('a','49_harp_adapter.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print(m.MODEL_OURS_TO_HARP.get('${MODEL}') or '')")
    [ -n "$HKEY" ] || { echo "ERROR: ${MODEL} has no HARP model key -- see MODEL_OURS_TO_HARP"; exit 1; }
    bash harp_patch_devices.sh "$HARP" || exit 1
    mkdir -p "${HARP}/dataset"
    cp "${REPO}/harp_input/${MODEL}/[known]${HDS}.jsonl"   "${HARP}/dataset/" || exit 1
    cp "${REPO}/harp_input/${MODEL}/[unknown]${HDS}.jsonl" "${HARP}/dataset/" || exit 1
    ( cd "$HARP" && python ./main.py --train_ratio=0.75 --train_batch_size=128 \
        --train_epochs=50 --model_name="$HKEY" --dataset_name="$HDS" ) || exit 1
    echo "--- re-scoring their method under both split protocols ---"
    python 50_harp_reeval_prompt_level.py --dataset "$HDS" --model_name "$HKEY" \
        --harp-dir "$HARP" --arms answer,question --proj-dims 256 --seeds 5 || exit 1
    deactivate 2>/dev/null || true
    ;;

  summary)
    python 52_pipeline_summary.py --model_folder "$MODEL" || exit 1
    ;;

  *)
    echo "ERROR: unknown stage '${STAGE}'"; exit 1 ;;
esac

echo "stage ${STAGE} (${DS:-n/a}) exit 0"
conda deactivate
