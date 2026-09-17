#!/bin/bash
# submit_tucker3.sh -- the 3-mode Tucker ablation (T-015), extraction and evaluation submitted together.
#
#   bash submit_tucker3.sh                       # 2 models x 2 datasets: 4 GPU extractions + 16 CPU evaluations
#   DRY=1 bash submit_tucker3.sh                 # print the plan, queue nothing
#   MODELS=llama-3.1-8b DATASETS=tydiqa_gp bash submit_tucker3.sh
#   POOLS="mean max" bash submit_tucker3.sh      # also max-pooling (default: mean only)
#
# RUN FROM A NEWTON TERMINAL.
#
# ISOLATION. Nothing here touches the reported detector: 69 writes only ../data-tokenstates, 70 writes only
# results/tensor_tucker, and both only READ the pinned sequences and features. No existing script changes.
#
# WHAT IS QUEUED, per model and dataset:
#   1 GPU job    69: token-level window states (skipped if ../data-tokenstates/<model>/<dataset>/meta.json exists)
#   4 CPU jobs   70: one per token-pool size T', started automatically when the extraction succeeds
#                    (--dependency=afterok); each runs both token ranks for its T':
#                    T'=1: R_T 1 | T'=2: R_T 1,2 | T'=4: R_T 2,4 | T'=8: R_T 4,8
# If an extraction fails, its four evaluations stay PENDING with reason DependencyNeverSatisfied: scancel them,
# fix, and rerun this script (finished pieces are skipped).
#
# DISK. tokens.npy is float16, (total answer tokens) x 9 x D: roughly 10-15 GB per TruthfulQA cell and 2-4 GB
# per TyDiQA-GP cell. 69 prints the exact size before its first forward pass and refuses above 60 GB.

set -euo pipefail

HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TOK_ROOT="$(cd "$HD_REPO/.." && pwd)/data-tokenstates"
cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs" "$HD_REPO/results/tensor_tucker"

MODELS="${MODELS:-qwen-2.5-7b-instruct llama-3.1-8b}"
DATASETS="${DATASETS:-tydiqa_gp truthfulqa}"
POOLS="${POOLS:-mean}"
PART="${PART:-highgpu}"
EXCLUDE="${EXCLUDE:-evc103}"
TGROUPS="1:1 2:1,2:2 4:2,4:4 8:4,8:8"         # one evaluation job per T' (not GROUPS: that name is a bash builtin)

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
for MO in $MODELS; do
  for DS in $DATASETS; do
    tag="${MO:0:4} ${DS}"
    EXT=""
    if [ -f "$TOK_ROOT/$MO/$DS/meta.json" ]; then
      printf "  %-16s extraction SKIP (complete)\n" "$tag"
    else
      mem=$([ "$DS" = truthfulqa ] && echo 96G || echo 64G)
      if [ "${DRY:-0}" = "1" ]; then
        printf "  %-16s extraction would queue (GPU, mem %s, 03:00:00)\n" "$tag" "$mem"; EXT="DRY"
      else
        sub "$HD_REPO/slurm/tokenstates_extract.slurm" \
            "-p $PART --gres=gpu:1 --mem=$mem --time=03:00:00 --job-name=tok-${MO:0:4}-${DS:0:4} ${EXCLUDE:+--exclude=$EXCLUDE} --export=ALL,HD_REPO=$HD_REPO" \
            "$MO" "$DS"
        EXT="$JOBID"; N=$((N+1))
        printf "  %-16s extraction -> %s\n" "$tag" "$EXT"
      fi
    fi
    for POOL in $POOLS; do
      for G in $TGROUPS; do
        T="${G%%:*}"
        OUT="$HD_REPO/results/tensor_tucker/tucker3_${MO}_${DS}_${POOL}_RF_T${T}.json"
        if [ -f "$OUT" ] && grep -q '"complete": true' "$OUT"; then
          printf "  %-16s %-4s T'=%s SKIP (complete)\n" "$tag" "$POOL" "$T"; continue
        fi
        if [ "$DS" = truthfulqa ]; then mem=$([ "$T" = 8 ] && echo 160G || echo 96G)
        else                            mem=$([ "$T" = 8 ] && echo 96G || echo 64G); fi
        dep=""
        [ -n "$EXT" ] && [ "$EXT" != "DRY" ] && dep="--dependency=afterok:$EXT"
        if [ "${DRY:-0}" = "1" ]; then
          printf "  %-16s %-4s T'=%s settings %-8s would queue (mem %s, 12:00:00%s)\n" "$tag" "$POOL" "$T" "$G" "$mem" \
              "$([ -n "$EXT" ] && echo ', after extraction')"
          continue
        fi
        sub "$HD_REPO/slurm/analysis_stage.slurm" \
            "-p $PART --mem=$mem --cpus-per-task=8 --time=12:00:00 --job-name=tk3-${MO:0:4}-${DS:0:4}-T${T} ${dep} ${EXCLUDE:+--exclude=$EXCLUDE} --export=ALL,HD_REPO=$HD_REPO" \
            70_tensor_tucker.py --dataset "$DS" --model_folder "$MO" --pool "$POOL" --settings "$G"
        printf "  %-16s %-4s T'=%s -> %s  (mem %s%s)\n" "$tag" "$POOL" "$T" "$JOBID" "$mem" "$([ -n "$dep" ] && echo ", after $EXT")"
        N=$((N+1))
      done
    done
  done
done

echo
echo "Queued $N jobs.  Watch: squeue -u \$USER   (evaluations show PENDING (Dependency) until their extraction ends)"
echo "Extraction logs print a PREFLIGHT line (stored states reproduce the pinned peak features, corr >= 0.999)."
echo "Results: results/tensor_tucker/tucker3_<model>_<dataset>_<pool>_RF_T<T'>.json"
