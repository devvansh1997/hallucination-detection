#!/bin/bash
# submit_hgproxy.sh -- HalluGuard's default (gradient-free) path over every dataset and both models.
#
#   bash submit_hgproxy.sh                       # both models, all four datasets
#   bash submit_hgproxy.sh qwen-2.5-7b-instruct  # one model
#   DATASETS="tydiqa_gp truthfulqa" bash submit_hgproxy.sh
#   FORCE=1 bash submit_hgproxy.sh               # ignore the skip guards and redo everything
#
# RUN THIS FROM A NEWTON TERMINAL -- these are GPU jobs. Stokes is for the high-RAM CPU work.
#
# All independent: one job per (model, dataset), no dependencies, so they fill whatever the queue
# gives us and a single failure costs only its own cell. Times are sized from beam count at the
# ~25 prompts/s a no-grad forward pass sustains on an H100, doubled for headroom, then rounded up:
#
#   tydiqa_gp     440 prompts /  4,400 beams  ->  00:40:00
#   truthfulqa    817 prompts /  8,170 beams  ->  01:00:00
#   nq_open     3,610 prompts / 36,100 beams  ->  03:00:00
#   triviaqa    9,960 prompts / 99,600 beams  ->  08:00:00
#
# TriviaQA is the one to watch: its prompts are the longest, so its per-prompt cost is above the
# rate the others set. If it times out, the skip guard means a resubmit redoes only TriviaQA.

set -euo pipefail

MODELS="${1:-qwen-2.5-7b-instruct llama-3.1-8b}"
DATASETS="${DATASETS:-tydiqa_gp truthfulqa nq_open triviaqa}"
PART="${PART:-highgpu}"
STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm/hgproxy_stage.slurm"

[ -f "$STAGE" ] || { echo "ERROR: $STAGE not found" >&2; exit 1; }

time_for() {
    case "$1" in
        tydiqa_gp)  echo "00:40:00" ;;
        truthfulqa) echo "01:00:00" ;;
        nq_open)    echo "03:00:00" ;;
        triviaqa)   echo "08:00:00" ;;
        *)          echo "04:00:00" ;;
    esac
}

# Sets a global rather than echoing: called as J=$(sub ...) the function runs in a subshell, where
# `exit` kills only the subshell and the driver carries on reporting success for jobs it never
# queued. That exact failure has already happened once on this project.
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

echo "models  : $MODELS"
echo "datasets: $DATASETS"
echo "stage   : $STAGE"
[ "${FORCE:-0}" = "1" ] && echo "FORCE=1  -- skip guards disabled, finished cells will be redone"
echo

N=0
for M in $MODELS; do
    echo "--- $M ---"
    for DS in $DATASETS; do
        T=$(time_for "$DS")
        sub "-p $PART --mem=80G --gres=gpu:1 --time=$T --job-name=hgp-${DS:0:4}-${M:0:4} \
             --export=ALL,FORCE=${FORCE:-0}" "$M" "$DS"
        printf "  %-12s %-9s -> %s\n" "$DS" "$T" "$JOBID"
        N=$((N + 1))
    done
    echo
done

echo "Queued $N jobs."
echo "Watch:   squeue -u \$USER"
echo "Results: results/halluguard_proxy/hgproxy_<model>_<dataset>.json"
echo
echo "Read the PER BEAM block against D_length_alone before anything else -- if the method does not"
echo "clear token count, the AUROC is not the headline, the comparison is."
