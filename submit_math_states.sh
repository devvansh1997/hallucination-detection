#!/bin/bash
# submit_math_states.sh -- hidden states of the sampled math answers (77, T-025). One GPU job per model.
#
#   DRY=1 bash submit_math_states.sh          # check inputs, queue nothing
#   bash submit_math_states.sh                # default: qwen-2.5-7b-instruct, gsm8k
#   MODEL=llama-3.1-8b-instruct bash submit_math_states.sh
#
# RUN FROM A NEWTON TERMINAL, after 76 has written ../data-math/<model>/<dataset>_sequences_v1.pt.
# Regrade first if the grading rules changed (`python 76_generate_math.py --regrade --model_folder <model>`);
# the order does not matter for correctness -- 77 stores no labels -- but it keeps the record straight.
#
# ISOLATION. 77 writes only ../data-math/<model>/<dataset>_states/. Nothing reported in the paper is touched.
# DISK. About 4 GB of per-token states plus 4 GB of LOTUS summaries for GSM8K on Qwen2.5-7B-Instruct;
# the job prints the exact sizes before its first forward pass.

set -euo pipefail

HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUT_ROOT="$(cd "$HD_REPO/.." && pwd)/data-math"
cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs"

MODEL="${MODEL:-qwen-2.5-7b-instruct}"
DATASET="${DATASET:-gsm8k}"
PART="${PART:-highgpu}"
EXCLUDE="${EXCLUDE:-evc103}"

SEQ="$OUT_ROOT/$MODEL/${DATASET}_sequences_v1.pt"
[ -f "$SEQ" ] || { echo "missing $SEQ -- run submit_math.sh full first"; exit 1; }
if [ -f "$OUT_ROOT/$MODEL/${DATASET}_states/meta.json" ]; then
    echo "  $MODEL $DATASET: SKIP (states complete)"; exit 0
fi
NAME="mst-${MODEL:0:4}-${DATASET}"
if [ -n "$(squeue -h -u "$USER" -n "$NAME" -o %i 2>/dev/null | head -1 || true)" ]; then
    echo "  $MODEL $DATASET: already queued or running ($NAME)"; exit 0
fi
if [ "${DRY:-0}" = "1" ]; then
    echo "  $MODEL $DATASET: would queue (GPU, mem 64G, 04:00:00): 77_extract_math_states.py --model_folder $MODEL --dataset $DATASET"; exit 0
fi

if ! out=$(sbatch --parsable -p "$PART" --gres=gpu:1 --mem=64G --time=04:00:00 --job-name="$NAME" \
        ${EXCLUDE:+--exclude=$EXCLUDE} --export=ALL,HD_REPO="$HD_REPO" \
        "$HD_REPO/slurm/math_states.slurm" "$MODEL" --dataset "$DATASET" 2>&1); then
    echo "ERROR: sbatch rejected the job:" >&2; printf '  %s\n' "$out" >&2; exit 1
fi
echo "  $MODEL $DATASET -> $out"
echo
echo "Watch: squeue -u \$USER     Log: slurm_logs/${NAME}_${out}.out"
echo "Read first in the log: the sizes line, the summary check (must be ~1e-6), then the fit pass line."
echo "Output: ../data-math/$MODEL/${DATASET}_states/ (meta.json is written last)"
