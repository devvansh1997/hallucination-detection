#!/bin/bash
# submit_math.sh -- sampled chain-of-thought answers to GSM8K, graded by final number (76, T-025).
#
#   bash submit_math.sh pilot        # first 200 questions x 10 answers: answer lengths and mixed questions
#   bash submit_math.sh full         # all 1,319 questions
#   MODEL=llama-3.1-8b-instruct bash submit_math.sh pilot
#
# RUN FROM A NEWTON TERMINAL, after caching the dataset once on a node with internet (hal-det env):
#   python 76_generate_math.py --prefetch
#
# Read the pilot's SUMMARY before running `full`: the number of MIXED questions (both a correct and a wrong
# answer among the 10) decides whether GSM8K is enough for this model or MATH has to be added.
#
# ISOLATION. 76 writes only ../data-math/<model>/. Nothing reported in the paper is touched.
# The pilot and the full run draw the same answers for the same question (seeds are per question).

set -euo pipefail

MODE="${1:?usage: submit_math.sh pilot|full}"
HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUT_ROOT="$(cd "$HD_REPO/.." && pwd)/data-math"
cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs"

MODEL="${MODEL:-qwen-2.5-7b-instruct}"
DATASET="${DATASET:-gsm8k}"
PILOT_N="${PILOT_N:-200}"
PART="${PART:-highgpu}"
EXCLUDE="${EXCLUDE:-evc103}"

case "$MODE" in
    pilot) STEM="${DATASET}_pilot${PILOT_N}"; ARGS="--dataset $DATASET --limit-questions $PILOT_N"; TIME=01:30:00;;
    full)  STEM="${DATASET}";                 ARGS="--dataset $DATASET";                            TIME=08:00:00;;
    *) echo "usage: submit_math.sh pilot|full"; exit 1;;
esac

if [ -f "$OUT_ROOT/$MODEL/${STEM}_sequences_v1.pt" ]; then
    echo "  $MODEL $STEM: SKIP (already generated: $OUT_ROOT/$MODEL/${STEM}_sequences_v1.pt)"; exit 0
fi
NAME="mgen-${MODE}-${MODEL:0:4}"
if [ -n "$(squeue -h -u "$USER" -n "$NAME" -o %i 2>/dev/null | head -1 || true)" ]; then
    echo "  $MODEL $STEM: already queued or running ($NAME)"; exit 0
fi
if [ "${DRY:-0}" = "1" ]; then
    echo "  $MODEL $STEM: would queue (GPU, mem 48G, $TIME): 76_generate_math.py --model_folder $MODEL $ARGS"; exit 0
fi

if ! out=$(sbatch --parsable -p "$PART" --gres=gpu:1 --mem=48G --time="$TIME" --job-name="$NAME" \
        ${EXCLUDE:+--exclude=$EXCLUDE} --export=ALL,HD_REPO="$HD_REPO" \
        "$HD_REPO/slurm/math_generate.slurm" "$MODEL" $ARGS 2>&1); then
    echo "ERROR: sbatch rejected the job:" >&2; printf '  %s\n' "$out" >&2; exit 1
fi
echo "  $MODEL $STEM -> $out"
echo
echo "Watch: squeue -u \$USER     Log: slurm_logs/${NAME}_${out}.out (progress every 25 questions, SUMMARY at the end)"
echo "Output: ../data-math/$MODEL/${STEM}_sequences_v1.pt and ${STEM}_summary.json"
