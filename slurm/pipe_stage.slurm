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

DATA_DIR=$(python -c "import yaml;print(yaml.safe_load(open('config.yaml'))['output']['data_dir'])") || exit 1
FORCE="${FORCE:-0}"

# Every stage is resumable. Resubmitting the whole graph after ANY failure must redo only what
# is actually missing -- otherwise a one-line bug late in the chain costs GPU-hours of finished
# work all over again, which is exactly what happened twice on this model.
#
# The guard tests each stage's LAST artifact, not its first: a partially-written stage leaves
# earlier files behind but never the final one, so "final artifact exists" means "this stage ran
# to completion". FORCE=1 overrides everything, for when the inputs changed rather than failed.
skip_if_done() {   # skip_if_done <artifact-path> <label>
    [ "$FORCE" = "1" ] && return 1
    if [ -e "$1" ]; then
        echo "SKIP: $2 already complete -> $1"
        echo "      (FORCE=1 to redo it anyway)"
        return 0
    fi
    return 1
}

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
    # 40_validate_dataset.py requires BOTH --manifest and --dataset; passing only --dataset
    # exits 1 with "ERROR: --manifest and --dataset required". 39 prints the manifest path it
    # wrote, and it is derived from config's output.data_dir, so build it the same way.
    SEQ="${DATA_DIR}/${MODEL}/${DS}_sequences_v1.pt"
    MANIFEST="${DATA_DIR}/${MODEL}/manifest_${DS}_v1.json"
    # Generation is the expensive half (GPU-hours) and is pinned + determinism-checked once
    # written, so it is skipped independently of validation -- validation is seconds and worth
    # re-running as a cheap integrity recheck.
    if [ -f "$SEQ" ] && [ -f "$MANIFEST" ] && [ "$FORCE" != "1" ]; then
        echo "SKIP: generation already complete -> $SEQ  (revalidating)"
    else
        python 39_generate_dataset.py --dataset "$DS" --model_folder "$MODEL" || exit 1
    fi
    python 40_validate_dataset.py --manifest "$MANIFEST" --dataset "$DS" \
        --model_folder "$MODEL" || exit 1
    ;;

  extract)
    skip_if_done "${DATA_DIR}/${MODEL}/${DS}_phase2_features.npz" "extraction" && exit 0
    python 42_extract_phase2.py --dataset "$DS" --model_folder "$MODEL" || exit 1
    ;;

  eval)
    # The two protocols are guarded separately: they write to different results dirs and either
    # can fail on its own, so a rerun should not redo whichever one already landed.
    echo "--- question-level (the paper's protocol; our headline numbers) ---"
    if skip_if_done "results/${MODEL}/session06_phase3_partA_${DS}.json" "eval (question-level)"; then
        :
    else
        python 44_eval_phase3.py --dataset "$DS" --model_folder "$MODEL" --part a || exit 1
    fi
    echo "--- answer-level (HARP's released protocol; the leakage arm) ---"
    if skip_if_done "${ANSWERSPLIT_DIR}/session06_phase3_partA_${DS}.json" "eval (answer-level)"; then
        :
    else
        python 44_eval_phase3.py --dataset "$DS" --model_folder "$MODEL" --part a \
            --split-unit answer --results-dir "$ANSWERSPLIT_DIR" || exit 1
    fi
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
    # The reeval JSON is the LAST thing this stage writes -- after their main.py and after our
    # A/B rescoring -- so its presence means the whole stage finished, not just the HARP half.
    skip_if_done "results/harp_reeval/${HKEY}_${HDS}_reeval.json" "harp + reeval" && exit 0
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
