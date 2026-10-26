#!/bin/bash
# submit_temporal_batch.sh -- the four order checks of T-026, one job each.
#
#   DRY=1 bash submit_temporal_batch.sh       # check inputs, queue nothing
#   bash submit_temporal_batch.sh             # queue all four
#   ONLY="81 84" bash submit_temporal_batch.sh
#
# RUN FROM A NEWTON TERMINAL with hal-det active (the ProcessBench download runs here, on the login node).
#
#   81  do hallucinated sentences cluster?           RAGTruth labels, all generators   CPU, minutes
#   82  does order help find where a span is?        RAGTruth Llama-2-7B token store   GPU, ~1 h
#   83  does order help spot the first wrong step?   ProcessBench, read by Qwen        GPU, ~1-2 h (reads once)
#   84  what is the late signal in GSM8K answers?    GSM8K Qwen store                  GPU, ~1-2 h
#
# NEW ON DISK: ../data-processbench/qwen-2.5-7b-instruct/ (~100 MB of step states) and
# ../data-math/qwen-2.5-7b-instruct/gsm8k_token_nll.npy (~14 MB). Results go to results/token_structure/.

set -euo pipefail

HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs" "$HD_REPO/results/token_structure"

PART="${PART:-highgpu}"
EXCLUDE="${EXCLUDE:-evc103}"
ONLY="${ONLY:-81 82 83 84}"
MATH_MODEL="qwen-2.5-7b-instruct"
RT_GEN="llama-2-7b-chat"
PARENT="$(cd "$HD_REPO/.." && pwd)"

wants() { [[ " $ONLY " == *" $1 "* ]]; }

queue() {   # name gpu(0/1) mem time script args...
    local name="$1" gpu="$2" mem="$3" time="$4"; shift 4
    if [ -n "$(squeue -h -u "$USER" -n "$name" -o %i 2>/dev/null | head -1 || true)" ]; then
        echo "  $name: already queued or running"; return 0
    fi
    local gres=""
    [ "$gpu" = "1" ] && gres="--gres=gpu:1"
    if [ "${DRY:-0}" = "1" ]; then
        echo "  $name: would queue ($PART ${gres:-CPU}, mem $mem, $time): $*"; return 0
    fi
    local out
    if ! out=$(sbatch --parsable -p "$PART" $gres --mem="$mem" --cpus-per-task=8 --time="$time" --job-name="$name" \
            ${EXCLUDE:+--exclude=$EXCLUDE} --export=ALL,HD_REPO="$HD_REPO" \
            "$HD_REPO/slurm/analysis_stage.slurm" "$@" 2>&1); then
        echo "ERROR: sbatch rejected $name:" >&2; printf '  %s\n' "$out" >&2; return 1
    fi
    echo "  $name -> $out   (log slurm_logs/${name}_${out}.out)"
}

if wants 81; then
    [ -f "$PARENT/data-ragtruth/RAGTruth/dataset/response.jsonl" ] || { echo "missing RAGTruth response.jsonl"; exit 1; }
    queue t81-sent 0 16G 01:00:00 81_sentence_neighbours.py
fi
if wants 82; then
    [ -f "$PARENT/data-ragtruth/$RT_GEN/meta.json" ] || { echo "missing the RAGTruth store for $RT_GEN (74)"; exit 1; }
    queue t82-loc 1 64G 04:00:00 82_ragtruth_localise.py --generator "$RT_GEN"
fi
if wants 83; then
    if [ ! -f "$PARENT/data-processbench/$MATH_MODEL/meta.json" ] && [ "${DRY:-0}" != "1" ]; then
        echo "  prefetching ProcessBench on the login node ..."
        python 83_processbench_steps.py --prefetch || {
            echo "prefetch failed: activate hal-det here (module load anaconda/anaconda-2024.10; conda activate hal-det) and rerun"; exit 1; }
    fi
    queue t83-pb 1 64G 05:00:00 83_processbench_steps.py
fi
if wants 84; then
    [ -f "$PARENT/data-math/$MATH_MODEL/gsm8k_states/meta.json" ] || { echo "missing the GSM8K state store (77)"; exit 1; }
    queue t84-last 1 64G 05:00:00 84_math_last_part.py --model_folder "$MATH_MODEL"
fi

echo
echo "Watch: squeue -u \$USER. Each log ends with a SUMMARY block; send those four blocks back."
