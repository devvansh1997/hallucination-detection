#!/bin/bash
# submit_alllayers.sh -- GPU stage of the layer-window ablation (T-012).
#
#   bash submit_alllayers.sh                                   # the four jobs below
#   JOBS="llama-3.1-8b:tydiqa_gp:core,static,velocity" bash submit_alllayers.sh
#   DRY=1 bash submit_alllayers.sh                             # print the plan and disk, queue nothing
#
# RUN FROM A NEWTON TERMINAL. These are GPU jobs.
#
# WHAT THE FOUR JOBS ARE FOR.
#   LLaMA, core+static+velocity   LLaMA has no per-layer data. core+static give its single-layer
#                                 curve (58, Appendix A for both models); velocity lets the full
#                                 detector be rebuilt at any window without a second forward pass.
#   Qwen, velocity only           Qwen's core and static already exist in ../data-alllayers and are
#                                 behind Appendix A and the spectrum figure, so they are not
#                                 re-extracted or overwritten. Only the update is missing.
# TyDiQA-GP and TruthfulQA only, matching the existing Qwen curves: extractive vs. closed-book, and
# the two cheapest datasets. NQ-Open is 97% hallucinated; TriviaQA does not fit in RAM.
#
# DISK. float16, uncompressed upper bound (savez_compressed saves a little on activations):
#   llama tydiqa_gp   core,static,velocity   4,400 x (33*3 + 32*2) x 4096 x 2 B   ~ 5.5 GB
#   llama truthfulqa  core,static,velocity   8,170 x (33*3 + 32*2) x 4096 x 2 B   ~10.2 GB
#   qwen  tydiqa_gp   velocity               4,400 x (28*2)        x 3584 x 2 B   ~ 1.6 GB
#   qwen  truthfulqa  velocity               8,170 x (28*2)        x 3584 x 2 B   ~ 3.1 GB
#                                                                          total ~20 GB
# The plan below prints current usage first. Check the quota before queueing.

set -euo pipefail

HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
STAGE="$HD_REPO/slurm/alllayers_extract.slurm"
[ -f "$STAGE" ] || { echo "ERROR: $STAGE not found -- is HD_REPO right?" >&2; exit 1; }
OUT_ROOT="$(cd "$HD_REPO/.." && pwd)/data-alllayers"

cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs"

JOBS="${JOBS:-llama-3.1-8b:tydiqa_gp:core,static,velocity llama-3.1-8b:truthfulqa:core,static,velocity qwen-2.5-7b-instruct:tydiqa_gp:velocity qwen-2.5-7b-instruct:truthfulqa:velocity}"
PART="${PART:-highgpu}"
MAXGB="${MAXGB:-32}"

# Host RAM holds the pooled arrays (<= 10.2 GB) plus the model while it loads on the CPU (~16 GB).
mem_for()  { case "$1" in tydiqa_gp) echo "64G";; truthfulqa) echo "80G";; *) echo "96G";; esac; }
# Not measured for 57 on this cluster yet. 59 took 11-12 min for the same answers, and 57 does more
# CPU work per answer (quantiles at every layer), so roughly 3-5x that. The Qwen line printed below
# shows 57's own time from its earlier run, if the file has it -- adjust TIME_* from it.
time_for() { case "$1" in tydiqa_gp) echo "${TIME_TYDI:-01:30:00}";; truthfulqa) echo "${TIME_TQA:-02:30:00}";;
                          *) echo "04:00:00";; esac; }

name_for() {  # must match 57_extract_all_layers.output_name
    case "$2" in *core*static*) echo "$1_alllayers.npz";;
                 velocity)      echo "$1_alllayers_velocity.npz";;
                 *)             echo "$1_alllayers_${2//,/+}.npz";; esac
}

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

echo "output root: $OUT_ROOT"
[ -d "$OUT_ROOT" ] && du -sh "$OUT_ROOT" 2>/dev/null | sed 's/^/  currently: /'
df -h "$(dirname "$OUT_ROOT")" 2>/dev/null | tail -1 | sed 's/^/  filesystem: /'
for f in "$OUT_ROOT"/qwen-2.5-7b-instruct/*_alllayers.json; do
    [ -f "$f" ] || continue
    printf "  earlier 57 run: %-24s elapsed_seconds=%s\n" "$(basename "$f")" \
        "$(grep -o '"elapsed_seconds": *[0-9.]*' "$f" | grep -o '[0-9.]*$' || echo '?')"
done
echo

N=0
for J in $JOBS; do
    IFS=: read -r MO DS ST <<< "$J"
    DST="$OUT_ROOT/$MO/$(name_for "$DS" "$ST")"
    if [ -f "$DST" ]; then
        printf "  %-22s %-11s %-22s SKIP (exists: %s)\n" "$MO" "$DS" "$ST" "$DST"
        continue
    fi
    if [ "${DRY:-0}" = "1" ]; then
        printf "  %-22s %-11s %-22s would queue (mem %s, %s) -> %s\n" "$MO" "$DS" "$ST" \
            "$(mem_for "$DS")" "$(time_for "$DS")" "$DST"
        continue
    fi
    sub "-p $PART --gres=gpu:1 --mem=$(mem_for "$DS") --time=$(time_for "$DS") \
         --job-name=alll-${MO:0:4}-${DS:0:4} --export=ALL,HD_REPO=$HD_REPO" "$MO" "$DS" "$ST" "$MAXGB"
    printf "  %-22s %-11s %-22s -> %s  (mem %s, %s)\n" "$MO" "$DS" "$ST" "$JOBID" \
        "$(mem_for "$DS")" "$(time_for "$DS")"
    N=$((N+1))
done

echo
echo "Queued $N jobs.  Watch: squeue -u \$USER"
echo "If squeue is EMPTY, check: sacct -u \$USER --starttime today --format=JobID,JobName%18,State,Elapsed,ExitCode"
echo "Read the .err first when a job dies (.out is discarded on TIMEOUT)."
echo "Each log ends with 'non-finite entries {...}': anything but zeros means float16 overflowed."
echo "Then, CPU (after the two LLaMA jobs):"
echo "  sbatch -p highgpu --time=04:00:00 --job-name=sweep-ll-tydi --export=ALL,HD_REPO=$HD_REPO slurm/analysis_stage.slurm 58_layer_sweep.py --dataset tydiqa_gp --model_folder llama-3.1-8b"
echo "  sbatch -p highgpu --time=04:00:00 --job-name=sweep-ll-tqa  --export=ALL,HD_REPO=$HD_REPO slurm/analysis_stage.slurm 58_layer_sweep.py --dataset truthfulqa --model_folder llama-3.1-8b"
