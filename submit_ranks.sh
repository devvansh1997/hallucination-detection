#!/bin/bash
# submit_ranks.sh -- the rank check: does the reported detector depend on its ranks?
#
#   bash submit_ranks.sh                                   # 2 models x 2 datasets, random forest
#   MODELS=llama-3.1-8b DATASETS=tydiqa_gp bash submit_ranks.sh
#   FORCE=1 bash submit_ranks.sh                           # rerun even if a complete result exists
#
# WHAT IT IS FOR. The paper reports layer rank 5 and feature rank 64. This varies one rank at a time
# (layer rank 1..9 at 64; feature rank 8..128 at 5), everything else as in the main results, to show
# whether those values sit on a plateau. 13 settings x 5 seeds per cell. See 63_rank_sweep.py.
#
# CPU ONLY. Each setting costs about one flatten-control arm, so expect roughly 1-3 h per cell.

set -euo pipefail

HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
STAGE="$HD_REPO/slurm/analysis_stage.slurm"
[ -f "$STAGE" ] || { echo "ERROR: $STAGE not found -- is HD_REPO right?" >&2; exit 1; }

# Slurm resolves --output relative to the submission directory: stand in the repo and make the log dir.
cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs" "$HD_REPO/results/rank_sweep"

MODELS="${MODELS:-qwen-2.5-7b-instruct llama-3.1-8b}"
DATASETS="${DATASETS:-tydiqa_gp truthfulqa}"
READOUT="${READOUT:-RF}"
PART="${PART:-highgpu}"
JOB_TIME="${JOB_TIME:-12:00:00}"
job_mem() { [ -n "${MEM:-}" ] && { echo "$MEM"; return; }
            case "$1" in tydiqa_gp) echo "64G";; truthfulqa) echo "96G";; *) echo "128G";; esac; }

# Sets a global rather than echoing: called as J=$(sub ...) this runs in a subshell where `exit` kills
# only the subshell and the driver reports success for jobs it never queued.
JOBID=""
sub() {
    local opts="$1"; shift
    local out
    if ! out=$(sbatch --parsable $opts "$STAGE" "$@" 2>&1); then
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
    OUT="$HD_REPO/results/rank_sweep/ranks_${MO}_${DS}_${READOUT}.json"
    if [ "${FORCE:-0}" != "1" ] && [ -f "$OUT" ] && grep -q '"complete": true' "$OUT"; then
      printf "  %-24s %-12s SKIP (complete)\n" "$MO" "$DS"
      continue
    fi
    ARGS="63_rank_sweep.py --dataset $DS --model_folder $MO --readout $READOUT"
    [ -n "${HD_DATA:-}" ] && ARGS="$ARGS --data-dir $HD_DATA"
    sub "-p $PART --mem=$(job_mem $DS) --cpus-per-task=8 --time=$JOB_TIME \
         --job-name=rank-${DS:0:4} --export=ALL,HD_REPO=$HD_REPO" $ARGS
    printf "  %-24s %-12s -> %s  (mem %s, %s)\n" "$MO" "$DS" "$JOBID" "$(job_mem $DS)" "$JOB_TIME"
    N=$((N+1))
  done
done

echo
echo "Queued $N jobs.  Watch: squeue -u \$USER   Logs: slurm_logs/rank-<dataset>_<jobid>.out"
echo "Results: results/rank_sweep/ranks_<model>_<dataset>_${READOUT}.json"
echo "The last line of each log must read: reproduction of the flatten control at (5, 64): EXACT"
