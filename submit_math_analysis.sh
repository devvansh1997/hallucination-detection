#!/bin/bash
# submit_math_analysis.sh -- same-question analysis of the sampled math answers (78, T-025). One CPU job per model.
#
#   DRY=1 bash submit_math_analysis.sh        # check inputs, queue nothing
#   bash submit_math_analysis.sh              # default: qwen-2.5-7b-instruct, gsm8k
#   EXTRA="--skip-lotus" bash submit_math_analysis.sh    # token scorers only (minutes)
#
# RUN FROM A NEWTON TERMINAL, after 77 has written ../data-math/<model>/<dataset>_states/meta.json.
#
# WHAT IT ANSWERS. Can a right and a wrong answer to the SAME question be told apart from hidden states, by
# LOTUS and by two token-level scorers, and where along the answer does the separation sit.
#
# ISOLATION. 78 only reads ../data-math and writes results/token_structure/math_<model>_<dataset>.json.
# No GPU, nothing new on disk beyond that JSON. Nothing reported in the paper is touched.

set -euo pipefail

HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUT_ROOT="$(cd "$HD_REPO/.." && pwd)/data-math"
cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs" "$HD_REPO/results/token_structure"

MODEL="${MODEL:-qwen-2.5-7b-instruct}"
DATASET="${DATASET:-gsm8k}"
PART="${PART:-highgpu}"
EXCLUDE="${EXCLUDE:-evc103}"
EXTRA="${EXTRA:-}"

STATES="$OUT_ROOT/$MODEL/${DATASET}_states"
[ -f "$STATES/meta.json" ] || { echo "missing $STATES/meta.json -- run submit_math_states.sh first"; exit 1; }
[ -f "$OUT_ROOT/$MODEL/${DATASET}_sequences_v1.pt" ] || { echo "missing the sequences file under $OUT_ROOT/$MODEL"; exit 1; }
OUT="$HD_REPO/results/token_structure/math_${MODEL}_${DATASET}.json"
if [ -f "$OUT" ] && grep -q '"complete": true' "$OUT"; then
    echo "  $MODEL $DATASET: SKIP (results complete; delete $OUT to rerun)"; exit 0
fi
NAME="man-${MODEL:0:4}-${DATASET}"
if [ -n "$(squeue -h -u "$USER" -n "$NAME" -o %i 2>/dev/null | head -1 || true)" ]; then
    echo "  $MODEL $DATASET: already queued or running ($NAME)"; exit 0
fi
if [ "${DRY:-0}" = "1" ]; then
    echo "  $MODEL $DATASET: would queue (CPU, mem 64G, 04:00:00): 78_math_same_question.py --model_folder $MODEL --dataset $DATASET $EXTRA"
    exit 0
fi

if ! out=$(sbatch --parsable -p "$PART" --mem=64G --cpus-per-task=8 --time=04:00:00 --job-name="$NAME" \
        ${EXCLUDE:+--exclude=$EXCLUDE} --export=ALL,HD_REPO="$HD_REPO" \
        "$HD_REPO/slurm/analysis_stage.slurm" 78_math_same_question.py --model_folder "$MODEL" --dataset "$DATASET" $EXTRA 2>&1); then
    echo "ERROR: sbatch rejected the job:" >&2; printf '  %s\n' "$out" >&2; exit 1
fi
echo "  $MODEL $DATASET -> $out"
echo
echo "Watch: squeue -u \$USER     Log: slurm_logs/${NAME}_${out}.out"
echo "The log ends with a SUMMARY block. Its first table is the answer: same-question AUROC per detector."
echo "Output: results/token_structure/math_${MODEL}_${DATASET}.json (written after the token scorers, again at the end)"
