#!/bin/bash
# submit_crossmodel.sh -- cross-model transfer (T-013): train the detector on one model, score the other.
#
#   bash submit_crossmodel.sh extract        # GPU: each model reads the other's answers (4 jobs)
#   bash submit_crossmodel.sh eval           # CPU: the transfer evaluation (4 jobs), once extract is done
#   DATASETS=tydiqa_gp bash submit_crossmodel.sh extract
#   DRY=1 bash submit_crossmodel.sh extract  # print the plan, queue nothing
#
# RUN FROM A NEWTON TERMINAL (extract needs GPUs).
#
# PAIRS. Qwen -> LLaMA and LLaMA -> Qwen, on TruthfulQA and TyDiQA-GP. For the direction S -> T the
# extraction is "S reads T's answers" (65), and the evaluation (66) trains on S and scores T.
#
# DISK. Window features only, float16: ~2.5-2.9 GB per TruthfulQA pair, ~1.4-1.6 GB per TyDiQA-GP pair,
# ~8.5 GB in total, under ../data-crossread.

set -euo pipefail

STAGE_NAME="${1:?usage: submit_crossmodel.sh extract|eval}"
HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CROSS_ROOT="$(cd "$HD_REPO/.." && pwd)/data-crossread"
cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs" "$HD_REPO/results/cross_model"

QWEN=qwen-2.5-7b-instruct
LLAMA=llama-3.1-8b
PAIRS="${PAIRS:-$QWEN:$LLAMA $LLAMA:$QWEN}"          # source:target
DATASETS="${DATASETS:-tydiqa_gp truthfulqa}"
PART="${PART:-highgpu}"
# A node to avoid, e.g. EXCLUDE=evc103: on 2026-09-16 every job there spent ~1 h importing torch before
# doing any work, and the two TyDiQA-GP reads were killed at their 1 h limit with nothing written.
EXCLUDE="${EXCLUDE:-}"

JOBID=""
sub() {
    local stage="$1" opts="$2"; shift 2
    local out
    if ! out=$(sbatch --parsable $opts "$stage" "$@" 2>&1); then
        echo "" >&2
        echo "ERROR: sbatch rejected this job -- nothing further has been queued." >&2
        printf '  %s\n' "$out" >&2
        echo "  opts: $opts" >&2; echo "  args: $*" >&2
        exit 1
    fi
    [ -n "$out" ] || { echo "ERROR: sbatch returned an empty job id" >&2; exit 1; }
    JOBID="$out"
}

N=0
for P in $PAIRS; do
  IFS=: read -r SRC TGT <<< "$P"
  for DS in $DATASETS; do
    META="$CROSS_ROOT/${SRC}_reads_${TGT}/${DS}_window.json"     # written after the .npz is complete
    tag="${SRC:0:4}->${TGT:0:4} ${DS}"
    case "$STAGE_NAME" in
      extract)
        if [ -f "$META" ] || [ -f "${META%.json}.npz" ]; then
          printf "  %-26s SKIP (exists)\n" "$tag"; continue
        fi
        mem=$([ "$DS" = truthfulqa ] && echo 80G || echo 64G)
        tim=03:00:00      # the work takes 5-10 min; the rest is headroom for a slow node (see EXCLUDE)
        if [ "${DRY:-0}" = "1" ]; then
          printf "  %-26s would queue: %s reads %s (mem %s, %s)\n" "$tag" "$SRC" "$TGT" "$mem" "$tim"; continue
        fi
        sub "$HD_REPO/slurm/crossread_extract.slurm" \
            "-p $PART --gres=gpu:1 --mem=$mem --time=$tim --job-name=xread-${SRC:0:4}-${DS:0:4} ${EXCLUDE:+--exclude=$EXCLUDE} --export=ALL,HD_REPO=$HD_REPO" \
            "$SRC" "$TGT" "$DS"
        printf "  %-26s -> %s  (%s reads %s, mem %s, %s)\n" "$tag" "$JOBID" "$SRC" "$TGT" "$mem" "$tim"
        N=$((N+1)) ;;
      eval)
        OUT="$HD_REPO/results/cross_model/cross_${SRC}_to_${TGT}_${DS}_RF.json"
        if [ "${FORCE:-0}" != "1" ] && [ -f "$OUT" ] && grep -q '"complete": true' "$OUT"; then
          printf "  %-26s SKIP (complete)\n" "$tag"; continue
        fi
        if [ ! -f "$META" ]; then
          printf "  %-26s WAITING (extraction not finished)\n" "$tag"; continue
        fi
        mem=$([ "$DS" = truthfulqa ] && echo 96G || echo 64G)
        if [ "${DRY:-0}" = "1" ]; then
          printf "  %-26s would queue (mem %s)\n" "$tag" "$mem"; continue
        fi
        sub "$HD_REPO/slurm/analysis_stage.slurm" \
            "-p $PART --mem=$mem --cpus-per-task=8 --time=04:00:00 --job-name=xmod-${SRC:0:4}-${DS:0:4} ${EXCLUDE:+--exclude=$EXCLUDE} --export=ALL,HD_REPO=$HD_REPO" \
            66_cross_model.py --source "$SRC" --target "$TGT" --dataset "$DS"
        printf "  %-26s -> %s  (mem %s)\n" "$tag" "$JOBID" "$mem"
        N=$((N+1)) ;;
      *) echo "unknown stage '$STAGE_NAME' (extract|eval)" >&2; exit 1 ;;
    esac
  done
done

echo
echo "Queued $N jobs.  Watch: squeue -u \$USER"
case "$STAGE_NAME" in
  extract) echo "Each log prints a PREFLIGHT line first (the reader re-reading its own answers must match its"
           echo "pinned features, corr >= 0.999). A FAIL stops the job before the real work."
           echo "When all four .json files exist: bash submit_crossmodel.sh eval" ;;
  eval)    echo "Each log ends with the four rows (in_model, aligned, naive, proxy) and a line saying whether"
           echo "in_model reproduces the flatten control -- it should read EXACT." ;;
esac
