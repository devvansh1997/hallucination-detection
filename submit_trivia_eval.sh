#!/bin/bash
# submit_trivia_eval.sh -- TriviaQA phase-3 evaluation, fanned out per condition.
#
#   bash submit_trivia_eval.sh llama-3.1-8b
#
# RUN THIS FROM A STOKES TERMINAL. Stokes is the high-RAM cluster; Newton is for GPU. They share
# the filesystem, so no data moves, but the schedulers are separate and a job lands on whichever
# cluster you submit from. These jobs are pure CPU and memory-hungry, so they belong on Stokes.
#
# PARTITION -- 'normal', NOT 'highmem'. highmem has ~3TB/node and looks like the natural home,
# but it is access-gated:
#     AllowAccounts = arcc, gmartin, asavage, pwiegand, szhang, tazarian, course.bsc4445c
# and our association is account 'sibrahim', which is not on that list. Submissions to it are
# rejected with "Invalid account or account/partition combination", and no -A flag fixes an
# access list. 'normal' is what the Qwen/TriviaQA runs used and is what we use here.
#
# MEMORY, sized from those Qwen runs -- same partition, same account, same code:
#   slurm/phase3_tri_*.slurm and phase3_answersplit_trivia.slurm both ran -p normal --mem=128G.
#   Measured peak RSS: core_max 49GB | q_velocity 86GB | q_static 97GB | core_concat 98GB |
#   triple_concat 128GB (completed, just under the 131,072MB ceiling) | joint_tensor killed in
#   fold 2 having already reached 130,813MB, so it needs somewhat more than 128G -- call it
#   ~147GB. LLaMA is 14% wider again (D=4096 vs 3584), putting triple_concat near 149GB and
#   joint_tensor near 167GB. 180G covers both and stays under the ~187GB node ceiling.
#
#   If joint_tensor still OOMs it is no loss relative to Qwen, where we also lack it, and it has
#   never been the best condition on any dataset (3rd, 4th, 5th elsewhere). The combine step is
#   gated on afterok of all six, so it simply will not run and the per-condition files are read
#   directly instead.
#
# Every job carries its own skip guard, so re-running this after a failure redoes only what is
# missing. That matters more here than anywhere else -- a single condition can be seven hours.

set -euo pipefail

MODEL="${1:?usage: submit_trivia_eval.sh <model_folder>   e.g. llama-3.1-8b}"
DS="${DS:-triviaqa}"
CPU_PART="${CPU_PART:-normal}"
MEM="${MEM:-180G}"
TIME_LIMIT="${TIME_LIMIT:-20:00:00}"
CONDITIONS="${CONDITIONS:-core_max q_velocity q_static core_concat joint_tensor triple_concat}"
STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm/pipe_stage.slurm"

# Only needed if a partition is account-gated; 'normal' is not, but keep the hook.
ACCT=""
[ -n "${ACCOUNT:-}" ] && ACCT="-A ${ACCOUNT}"

# Sets the global JOBID rather than echoing, deliberately. Called as J=$(sub ...) the function
# runs in a subshell, so an `exit` on a rejected sbatch kills only that subshell and the caller
# carries on -- which is exactly what happened: all 14 submissions were rejected and the script
# still printed "Queued 14 jobs". A driver must never report success for work it did not queue.
JOBID=""
sub() {
    local opts="$1"; shift
    local out
    if ! out=$(sbatch --parsable $opts "$STAGE" "$@" 2>&1); then
        echo "" >&2
        echo "ERROR: sbatch rejected this job -- nothing further has been queued." >&2
        printf '  %s\n' "$out" >&2
        echo "  opts: $opts" >&2
        echo "  args: $*" >&2
        exit 1
    fi
    [ -n "$out" ] || { echo "ERROR: sbatch returned an empty job id" >&2; exit 1; }
    JOBID="$out"
}

N=$(echo $CONDITIONS | wc -w)
echo "model=$MODEL  dataset=$DS  partition=$CPU_PART  mem=$MEM  time=$TIME_LIMIT"
echo "conditions ($N): $CONDITIONS"
echo "stage: $STAGE"
echo

for UNIT in question answer; do
    echo "--- ${UNIT}-level protocol ---"
    IDS=()
    for C in $CONDITIONS; do
        sub "-p $CPU_PART $ACCT --mem=$MEM --time=$TIME_LIMIT --job-name=ev-${UNIT:0:1}-$C" \
            eval_cond "$MODEL" "$DS" "$C" "$UNIT"
        IDS+=("$JOBID")
        printf "  %-14s -> %s\n" "$C" "$JOBID"
    done
    # afterok on ALL conditions: 44's --combine-conditions errors if any per-condition file is
    # missing, so if one OOMs the merge correctly never runs, rather than half-merging.
    DEP=$(IFS=:; echo "${IDS[*]}")
    sub "-p $CPU_PART $ACCT --mem=64G --time=04:00:00 --dependency=afterok:$DEP --job-name=cmb-${UNIT:0:1}" \
        combine "$MODEL" "$DS" "-" "$UNIT"
    printf "  %-14s -> %s\n" "combine" "$JOBID"
    echo
done

echo "Queued $((N * 2 + 2)) jobs ($N conditions x 2 protocols, plus a merge each)."
echo "Watch:  squeue -u \$USER"
echo
echo "Results:"
echo "  question -> results/$MODEL/session06_phase3_partA_${DS}[_<condition>].json"
echo "  answer   -> results/$MODEL-answersplit/session06_phase3_partA_${DS}[_<condition>].json"
