#!/bin/bash
# submit_tokstruct.sh -- token-mode structure probe (73), the feasibility step for the functional token model.
#
#   DRY=1 bash submit_tokstruct.sh                # print the plan, queue nothing -- run this first
#   bash submit_tokstruct.sh                      # 2 models x 2 datasets
#   MODELS=qwen-2.5-7b-instruct DATASETS=tydiqa_gp bash submit_tokstruct.sh   # one cell, as a pilot
#
# RUN FROM A NEWTON TERMINAL.
#
# REUSE. 73 reads ../data-tokenstates, which 69 wrote for the 3-mode Tucker ablation (submit_tucker3.sh).
# Where that store is complete it is reused and only a CPU job is queued. Where it is missing (deleted to
# free quota, say), the GPU extraction is queued first and the analysis waits on it (--dependency=afterok).
#
# ISOLATION. 73 only reads the token store and the pinned features, and writes only
# results/token_structure/. Nothing reported in the paper is touched.
#
# DISK. A re-extraction writes float16 (answer tokens x 9 x D): roughly 10-15 GB per TruthfulQA cell and
# 2-4 GB per TyDiQA-GP cell. The plan below prints the size of any store already on disk.

set -euo pipefail

HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TOK_ROOT="$(cd "$HD_REPO/.." && pwd)/data-tokenstates"
cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs" "$HD_REPO/results/token_structure"

MODELS="${MODELS:-qwen-2.5-7b-instruct llama-3.1-8b}"
DATASETS="${DATASETS:-tydiqa_gp truthfulqa}"
PART="${PART:-highgpu}"
EXCLUDE="${EXCLUDE:-evc103}"

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
    OUT="$HD_REPO/results/token_structure/tokstruct_${MO}_${DS}.json"
    if [ -f "$OUT" ] && grep -q '"complete": true' "$OUT"; then
      printf "  %-16s SKIP (results complete)\n" "$tag"; continue
    fi

    EXT=""
    if [ -f "$TOK_ROOT/$MO/$DS/meta.json" ]; then
      printf "  %-16s token store REUSED (%s)\n" "$tag" "$(du -sh "$TOK_ROOT/$MO/$DS" 2>/dev/null | cut -f1)"
    else
      mem=$([ "$DS" = truthfulqa ] && echo 96G || echo 64G)
      if [ "${DRY:-0}" = "1" ]; then
        printf "  %-16s token store MISSING -- extraction would queue (GPU, mem %s, 03:00:00)\n" "$tag" "$mem"
        EXT="DRY"
      else
        sub "$HD_REPO/slurm/tokenstates_extract.slurm" \
            "-p $PART --gres=gpu:1 --mem=$mem --time=03:00:00 --job-name=tok-${MO:0:4}-${DS:0:4} ${EXCLUDE:+--exclude=$EXCLUDE} --export=ALL,HD_REPO=$HD_REPO" \
            "$MO" "$DS"
        EXT="$JOBID"; N=$((N+1))
        printf "  %-16s token store MISSING -- extraction -> %s\n" "$tag" "$EXT"
      fi
    fi

    mem=$([ "$DS" = truthfulqa ] && echo 64G || echo 32G)
    dep=""
    [ -n "$EXT" ] && [ "$EXT" != "DRY" ] && dep="--dependency=afterok:$EXT"
    if [ "${DRY:-0}" = "1" ]; then
      printf "  %-16s analysis would queue (CPU, mem %s, 04:00:00%s)\n" "$tag" "$mem" "$([ -n "$EXT" ] && echo ', after extraction')"
      continue
    fi
    sub "$HD_REPO/slurm/analysis_stage.slurm" \
        "-p $PART --mem=$mem --cpus-per-task=8 --time=04:00:00 --job-name=tks-${MO:0:4}-${DS:0:4} ${dep} ${EXCLUDE:+--exclude=$EXCLUDE} --export=ALL,HD_REPO=$HD_REPO" \
        73_token_structure.py --dataset "$DS" --model_folder "$MO"
    printf "  %-16s analysis -> %s  (mem %s%s)\n" "$tag" "$JOBID" "$mem" "$([ -n "$dep" ] && echo ", after $EXT")"
    N=$((N+1))
  done
done

echo
echo "Queued $N jobs.  Watch: squeue -u \$USER"
echo "Each log ends with a SUMMARY block (M0 share, M1 index by lag, M2 AUROC by aggregate, M2 gap by position)."
echo "Results: results/token_structure/tokstruct_<model>_<dataset>.json"
