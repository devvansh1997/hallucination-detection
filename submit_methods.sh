#!/bin/bash
# submit_methods.sh -- run detection methods over the models x datasets grid.
#
#   export HD_REPO=$HOME/Hallucination-Detection/hallucination-detection
#   export HD_DATA=/home/de807845/Hallucination-Detection/data
#
#   bash submit_methods.sh                                   # all registered methods, all cells
#   METHODS="perplexity eigenscore" bash submit_methods.sh    # a subset
#   DATASETS=tydiqa_gp MODELS=qwen-2.5-7b-instruct bash submit_methods.sh   # one cell, to smoke-test
#   FORCE=1 bash submit_methods.sh                            # ignore skip guards
#
# RUN FROM A NEWTON TERMINAL. These are GPU jobs.
#
# HD_DATA may point at somebody else's directory. The harness only ever reads it, so a second
# person can score the pinned generations without copying 91 GB. That is the intended workflow.
#
# Jobs are independent: no dependencies, one per (method, model, dataset), so a single failure
# costs only its own cell and the skip guard means a resubmit redoes just that one.

set -euo pipefail

: "${HD_REPO:?export HD_REPO=/path/to/your/hallucination-detection}"
STAGE="${HD_REPO}/slurm/method_stage.slurm"
[ -f "$STAGE" ] || { echo "ERROR: $STAGE not found -- is HD_REPO right?" >&2; exit 1; }

METHODS="${METHODS:-$(cd "$HD_REPO" && python 56_run_method.py --list | tail -n +2 | awk '{print $1}' | tr '\n' ' ')}"
MODELS="${MODELS:-qwen-2.5-7b-instruct llama-3.1-8b}"
DATASETS="${DATASETS:-tydiqa_gp truthfulqa nq_open triviaqa}"
PART="${PART:-highgpu}"
TAG="${TAG:-}"

# Sized from measured runs: scoring is one forward pass per question and is dominated by model
# load on the small datasets. TriviaQA is the only one that needs real time.
job_time() { [ -n "${JOB_TIME:-}" ] && { echo "$JOB_TIME"; return; }
             case "$1" in tydiqa_gp) echo "00:40:00";; truthfulqa) echo "01:00:00";;
                          nq_open) echo "02:00:00";; triviaqa) echo "05:00:00";;
                          *) echo "02:00:00";; esac; }

# Sets a global rather than echoing: called as J=$(sub ...) this runs in a subshell where `exit`
# kills only the subshell and the driver reports success for jobs it never queued. That has
# happened on this project before.
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

echo "repo    : $HD_REPO"
echo "data    : ${HD_DATA:-<config.yaml default>}"
echo "methods : $METHODS"
echo "models  : $MODELS"
echo "datasets: $DATASETS"
echo

N=0
for ME in $METHODS; do
  for MO in $MODELS; do
    for DS in $DATASETS; do
      sub "-p $PART --mem=80G --gres=gpu:1 --time=$(job_time $DS) \
           --job-name=m-${ME:0:6}-${DS:0:4} \
           --export=ALL,HD_REPO=$HD_REPO,HD_DATA=${HD_DATA:-},FORCE=${FORCE:-0}" \
          "$ME" "$MO" "$DS" "$TAG"
      printf "  %-20s %-24s %-12s -> %s\n" "$ME" "$MO" "$DS" "$JOBID"
      N=$((N+1))
    done
  done
done

echo
echo "Queued $N jobs."
echo "Watch:   squeue -u \$USER"
echo "Results: \$HD_REPO/results/methods/<method>_<model>_<dataset>.json"
echo
echo "Read the two protocol lines in each result, not just one number. A method's AUROC under the"
echo "question-level split and under the answer-level split are different quantities, and reporting"
echo "one without saying which is how this whole investigation started."
