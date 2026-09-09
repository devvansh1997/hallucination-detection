#!/bin/bash
# submit_methods.sh -- run detection methods over the models x datasets grid.
#
#   export HD_REPO=$HOME/Hallucination-Detection/hallucination-detection
#   export HD_DATA=/home/de807845/Hallucination-Detection/data
#
#   bash submit_methods.sh                                   # all registered methods, all cells
#   METHODS="perplexity eigenscore" bash submit_methods.sh    # a subset
#   METHODS=act_vit DATASETS="tydiqa_gp truthfulqa" bash submit_methods.sh
#   METHODS=act_vit TAG=n100 EXTRA="--m-n-eff 100" bash submit_methods.sh   # their default arm
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

# HD_REPO defaults to the directory this script lives in, which IS the repo root. Requiring it as
# an environment variable meant a fresh login shell died here with a message that reads like usage
# text rather than an error, having queued nothing -- three times now, once misdiagnosed as a
# scheduler problem. An explicit export still wins, so a second person can drive their own clone.
HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
STAGE="${HD_REPO}/slurm/method_stage.slurm"
[ -f "$STAGE" ] || { echo "ERROR: $STAGE not found -- is HD_REPO right?" >&2; exit 1; }

# Slurm resolves --output and --error RELATIVE TO THE SUBMISSION DIRECTORY, not to HD_REPO, and the
# job inherits that directory as its cwd. Submitting from $HOME therefore scatters logs into
# $HOME/slurm_logs and hides them when a job dies before its first line -- which reads exactly like
# "the job never fired". Stand in the repo and create the directories first. submit_act.sh and
# submit_flatten.sh already do this; this was the one submitter that did neither.
cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs" "$HD_REPO/results/methods"

METHODS="${METHODS:-$(cd "$HD_REPO" && python 56_run_method.py --list | tail -n +2 | awk '{print $1}' | tr '\n' ' ')}"
MODELS="${MODELS:-qwen-2.5-7b-instruct llama-3.1-8b}"
DATASETS="${DATASETS:-tydiqa_gp truthfulqa nq_open triviaqa}"
PART="${PART:-highgpu}"
TAG="${TAG:-}"
# Anything here is appended to 56_run_method.py's command line, e.g. EXTRA="--m-n-eff 100".
EXTRA="${EXTRA:-}"

# Sized from measured runs. Training-free methods are one forward pass per question, dominated by
# model load on the small datasets. act_vit TRAINS ten times (five seeds x two protocols) and is
# far heavier, and heavier again at N_eff=100 where every batch moves five times the bytes.
#
# MEASURED, 2026-09-08, Qwen, H100:
#   truthfulqa N_eff=20    1111s     tydiqa_gp N_eff=100   2291s
#   truthfulqa N_eff=100   >3600s -- killed on a 1h wall having produced no epoch line
# Scaling tydiqa N=100 by the row ratio (8170/4400 = 1.86) puts truthfulqa N=100 near 71 minutes,
# which is why it did not fit. The act_vit walls below carry roughly 2x on the measurements.
job_time() { [ -n "${JOB_TIME:-}" ] && { echo "$JOB_TIME"; return; }
             if [ "${1:-}" = "act_vit" ]; then
                 case "${2:-}" in tydiqa_gp) echo "02:00:00";; truthfulqa) echo "03:00:00";;
                                  nq_open) echo "08:00:00";; triviaqa) echo "20:00:00";;
                                  *) echo "04:00:00";; esac
             else
                 case "${2:-}" in tydiqa_gp) echo "00:40:00";; truthfulqa) echo "01:00:00";;
                                  nq_open) echo "02:00:00";; triviaqa) echo "05:00:00";;
                                  *) echo "02:00:00";; esac
             fi; }

# Host RAM. Most methods stream from the pinned generations and 80G is ample. act_vit is the
# exception: it holds a whole activation tensor in RAM, 4.7 GB per dataset at N_eff=20 but
# 23.5 GB (TyDiQA) and 43.6 GB (TruthfulQA) at N_eff=100. Override with MEM= for anything unusual.
job_mem() { [ -n "${MEM:-}" ] && { echo "$MEM"; return; }
            if [ "$1" = "act_vit" ]; then
                case "$2" in tydiqa_gp) echo "64G";; truthfulqa) echo "110G";;
                             nq_open) echo "220G";; triviaqa) echo "600G";; *) echo "120G";; esac
            else echo "80G"; fi; }

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
echo "logs    : $HD_REPO/slurm_logs/<job-name>_<jobid>/"
echo

N=0
for ME in $METHODS; do
  for MO in $MODELS; do
    for DS in $DATASETS; do
      sub "-p $PART --mem=$(job_mem $ME $DS) --gres=gpu:1 --time=$(job_time $ME $DS) \
           --job-name=m-${ME:0:6}-${DS:0:4} \
           --export=ALL,HD_REPO=$HD_REPO,HD_DATA=${HD_DATA:-},FORCE=${FORCE:-0}" \
          "$ME" "$MO" "$DS" "$TAG" ${EXTRA:-}
      printf "  %-20s %-24s %-12s -> %s  (mem %s, %s)\n" "$ME" "$MO" "$DS" "$JOBID" \
             "$(job_mem $ME $DS)" "$(job_time $ME $DS)"
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
