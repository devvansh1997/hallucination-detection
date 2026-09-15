#!/bin/bash
# submit_windows.sh -- CPU stage of the layer-window ablation (T-012): the reported detector at
# other layer windows. See 64_window_sweep.py for the pre-registered grid and the gate.
#
#   bash submit_windows.sh                                  # 2 models x 2 datasets, random forest
#   MODELS=llama-3.1-8b DATASETS=tydiqa_gp bash submit_windows.sh
#   FORCE=1 bash submit_windows.sh                          # rerun even if a complete result exists
#
# NEEDS submit_alllayers.sh FIRST. A cell is queued only when its all-layer file and its update exist;
# otherwise it is listed as WAITING and nothing is submitted for it.
#
# CPU ONLY. Each window costs about one rank-sweep setting (5 seeds, RF), and the grid has 6 (Qwen)
# or 8 (LLaMA) windows, so expect roughly 1-2 h per cell.

set -euo pipefail

HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
STAGE="$HD_REPO/slurm/analysis_stage.slurm"
[ -f "$STAGE" ] || { echo "ERROR: $STAGE not found -- is HD_REPO right?" >&2; exit 1; }
IN_ROOT="$(cd "$HD_REPO/.." && pwd)/data-alllayers"

cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs" "$HD_REPO/results/window_sweep"

MODELS="${MODELS:-qwen-2.5-7b-instruct llama-3.1-8b}"
DATASETS="${DATASETS:-tydiqa_gp truthfulqa}"
READOUT="${READOUT:-RF}"
PART="${PART:-highgpu}"
JOB_TIME="${JOB_TIME:-08:00:00}"
job_mem() { [ -n "${MEM:-}" ] && { echo "$MEM"; return; }
            case "$1" in tydiqa_gp) echo "64G";; truthfulqa) echo "96G";; *) echo "128G";; esac; }

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
    OUT="$HD_REPO/results/window_sweep/window_${MO}_${DS}_${READOUT}.json"
    if [ "${FORCE:-0}" != "1" ] && [ -f "$OUT" ] && grep -q '"complete": true' "$OUT"; then
      printf "  %-24s %-12s SKIP (complete)\n" "$MO" "$DS"
      continue
    fi
    # The .json is written after the .npz is complete; a running job leaves a partial .npz behind.
    MAIN="$IN_ROOT/$MO/${DS}_alllayers.json"
    VEL="$IN_ROOT/$MO/${DS}_alllayers_velocity.json"
    # LLaMA carries the update inside the main file; Qwen has it in the _velocity file.
    if [ ! -f "$MAIN" ] || { [ "$MO" = "qwen-2.5-7b-instruct" ] && [ ! -f "$VEL" ]; }; then
      printf "  %-24s %-12s WAITING (extraction not finished)\n" "$MO" "$DS"
      continue
    fi
    ARGS="64_window_sweep.py --dataset $DS --model_folder $MO --readout $READOUT --resume"
    [ -n "${HD_DATA:-}" ] && ARGS="$ARGS --data-dir $HD_DATA"
    sub "-p $PART --mem=$(job_mem $DS) --cpus-per-task=8 --time=$JOB_TIME \
         --job-name=win-${MO:0:4}-${DS:0:4} --export=ALL,HD_REPO=$HD_REPO" $ARGS
    printf "  %-24s %-12s -> %s  (mem %s, %s)\n" "$MO" "$DS" "$JOBID" "$(job_mem $DS)" "$JOB_TIME"
    N=$((N+1))
  done
done

echo
echo "Queued $N jobs.  Watch: squeue -u \$USER   Logs: slurm_logs/win-<model>-<dataset>_<jobid>.out"
echo "Results: results/window_sweep/window_<model>_<dataset>_${READOUT}.json"
echo "Read the GATE line first: the reported window must reproduce the flatten control within 0.5 pts,"
echo "or the job stops after one window. --resume is on, so a timed-out job restarts where it stopped."
