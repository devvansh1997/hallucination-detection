#!/bin/bash
# submit_trivia_eval.sh -- TriviaQA phase-3 evaluation, fanned out per condition.
#
#   bash submit_trivia_eval.sh llama-3.1-8b
#
# RUN THIS FROM A STOKES TERMINAL. Stokes is the high-RAM cluster; Newton is for GPU. They share
# the filesystem, so no data moves, but the schedulers are separate and a job lands on whichever
# cluster you submit from. These jobs are pure CPU and want 250-400GB, so they belong on Stokes.
#
# WHY A FAN-OUT AND NOT THE PIPELINE'S 'eval' STAGE
#   Measured per-condition times on TriviaQA (Qwen): core_max 2h39m, q_velocity 3h48m,
#   q_static 4h35m, core_concat 5h52m, triple_concat 7h13m -- and joint_tensor OOM-killed at
#   200GB. Six conditions x two protocols is 40h+ of wall clock in one job, against a 24h limit.
#   So: 12 jobs (6 conditions x 2 protocols), then one merge per protocol.
#
# Every job carries its own skip guard, so re-running this after any failure redoes only what is
# actually missing. That matters here more than anywhere else in the project -- a single
# condition can be seven hours.
#
# PARTITION -- must be 'highmem', and this is not a preference. Measured on Stokes:
#     highmem      ~3,094,585 MB/node  (~3 TB)   <- the only one that fits
#     normal*      ~191,377 MB/node    (~187 GB) <- every request here exceeds it
#     preemptable  ~191,377 MB/node              <- same ceiling, and preemptible
# The smallest request below is 256G, so submitting to normal gets the job rejected outright at
# submit time. Override only if the partition is renamed:
#   CPU_PART=<name> bash submit_trivia_eval.sh llama-3.1-8b

set -euo pipefail

MODEL="${1:?usage: submit_trivia_eval.sh <model_folder>   e.g. llama-3.1-8b}"
DS="${DS:-triviaqa}"
CPU_PART="${CPU_PART:-highmem}"
STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm/pipe_stage.slurm"

CONDITIONS="core_max q_velocity q_static core_concat joint_tensor triple_concat"
TIME_LIMIT="${TIME_LIMIT:-20:00:00}"

# joint_tensor fits ONE decomposition over a stacked 17 x 2D tensor rather than several smaller
# ones, so it peaks well above the rest -- it is the condition that OOM'd at 200GB on Qwen, and
# LLaMA is 14% wider again (D=4096 vs 3584).
mem_for() {
    case "$1" in
        joint_tensor)  echo "${MEM_JOINT:-400G}" ;;
        triple_concat) echo "${MEM_BIG:-320G}" ;;
        *)             echo "${MEM_STD:-256G}" ;;
    esac
}

sub() {
    local opts="$1"; shift
    local out
    out=$(sbatch --parsable $opts "$STAGE" "$@")
    echo "$out"
}

echo "model=$MODEL  dataset=$DS  partition=$CPU_PART  time=$TIME_LIMIT"
echo "stage: $STAGE"
echo

for UNIT in question answer; do
    echo "--- ${UNIT}-level protocol ---"
    IDS=()
    for C in $CONDITIONS; do
        M=$(mem_for "$C")
        J=$(sub "-p $CPU_PART --mem=$M --time=$TIME_LIMIT \
                 --job-name=ev-${UNIT:0:1}-$C" eval_cond "$MODEL" "$DS" "$C" "$UNIT")
        IDS+=("$J")
        printf "  %-14s %-5s -> %s\n" "$C" "$M" "$J"
    done
    DEP=$(IFS=:; echo "${IDS[*]}")
    JC=$(sub "-p $CPU_PART --mem=128G --time=04:00:00 --dependency=afterok:$DEP \
              --job-name=cmb-${UNIT:0:1}" combine "$MODEL" "$DS" "-" "$UNIT")
    echo "  combine -> $JC"
    echo
done

echo "Queued 14 jobs (6 conditions x 2 protocols, plus a merge each)."
echo "Watch:  squeue -u \$USER"
echo
echo "Each condition guards on its own results file, so re-running this script after a failure"
echo "reruns only the missing ones -- important when a single condition can take seven hours."
echo "The two protocols write to separate results dirs and never collide:"
echo "  question -> results/$MODEL"
echo "  answer   -> results/$MODEL-answersplit"
