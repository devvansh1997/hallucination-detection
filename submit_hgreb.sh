#!/bin/bash
# submit_hgreb.sh -- HalluGuard as its authors describe it, on beam-search AND nucleus generations.
#
#   bash submit_hgreb.sh                    # the recommended first run: TyDiQA, both decodings
#   bash submit_hgreb.sh --all              # all four datasets, both models, both decodings
#   DATASETS="tydiqa_gp truthfulqa" bash submit_hgreb.sh
#   FORCE=1 bash submit_hgreb.sh            # ignore skip guards
#
# RUN FROM A NEWTON TERMINAL. These are GPU jobs.
#
# WHY BOTH DECODINGS. Their method is a measurement of how spread out the K sampled trajectories
# are: K = H H^T over the final-layer states, then log-det / largest eigenvalue / condition number.
# Our pinned data comes from sampled BEAM SEARCH with 10 beams, whose purpose is to return ten
# SIMILAR sequences. In the self-test, ten identical trajectories score -719 against +43 for ten
# spread ones -- a 762-point swing caused entirely by the decoder. So a beam-search number is not
# evidence about their method until the nucleus number sits beside it.
#
# The nucleus generations go to ../data-nucleus, never into the pinned data directory. 39 refuses
# decoding overrides without --output-dir for exactly that reason.
#
# TIMES. Scoring is forward-pass-only and cheap. Generation is the expensive half and is sized from
# the measured beam-search runs; nucleus sampling with num_beams=1 is FASTER than 10-way beam
# search, so these are upper bounds.
#
#   dataset      questions   gen (nucleus)   score
#   tydiqa_gp          440       00:45:00    00:30:00
#   truthfulqa         817       01:15:00    00:40:00
#   nq_open          3,610       03:00:00    01:30:00
#   triviaqa         9,960       09:00:00    04:00:00

set -euo pipefail

ALL=0
[ "${1:-}" = "--all" ] && ALL=1

if [ "$ALL" = "1" ]; then
    MODELS="${MODELS:-qwen-2.5-7b-instruct llama-3.1-8b}"
    DATASETS="${DATASETS:-tydiqa_gp truthfulqa nq_open triviaqa}"
else
    MODELS="${MODELS:-qwen-2.5-7b-instruct}"
    DATASETS="${DATASETS:-tydiqa_gp}"
fi
PART="${PART:-highgpu}"
STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm/hgreb_stage.slurm"
[ -f "$STAGE" ] || { echo "ERROR: $STAGE not found" >&2; exit 1; }

gen_time()   { case "$1" in tydiqa_gp) echo "00:45:00";; truthfulqa) echo "01:15:00";;
                            nq_open) echo "03:00:00";; triviaqa) echo "09:00:00";;
                            *) echo "04:00:00";; esac; }
score_time() { case "$1" in tydiqa_gp) echo "00:30:00";; truthfulqa) echo "00:40:00";;
                            nq_open) echo "01:30:00";; triviaqa) echo "04:00:00";;
                            *) echo "02:00:00";; esac; }

# Sets a global rather than echoing: called as J=$(sub ...) this runs in a subshell where `exit`
# kills only the subshell, and the driver goes on to report success for jobs it never queued.
# That exact failure has already happened once on this project.
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

echo "models  : $MODELS"
echo "datasets: $DATASETS"
[ "$ALL" = "0" ] && echo "(default scope: TyDiQA on Qwen only. Use --all for the full grid.)"
echo

N=0
for M in $MODELS; do
  for DS in $DATASETS; do
    echo "--- $M / $DS ---"

    # (a) score the EXISTING beam-search generations -- no dependency, starts immediately
    sub "-p $PART --mem=80G --gres=gpu:1 --time=$(score_time $DS) \
         --job-name=hgr-beam-${DS:0:4} --export=ALL,FORCE=${FORCE:-0}" score "$M" "$DS"
    printf "  %-22s -> %s\n" "score (beam search)" "$JOBID"; N=$((N+1))

    # (b) regenerate with nucleus sampling, then (c) score it -- chained
    sub "-p $PART --mem=80G --gres=gpu:1 --time=$(gen_time $DS) \
         --job-name=hgr-ngen-${DS:0:4} --export=ALL,FORCE=${FORCE:-0}" nucleus_gen "$M" "$DS"
    GEN=$JOBID
    printf "  %-22s -> %s\n" "nucleus generation" "$GEN"; N=$((N+1))

    sub "-p $PART --mem=80G --gres=gpu:1 --time=$(score_time $DS) \
         --dependency=afterok:$GEN --job-name=hgr-nucl-${DS:0:4} \
         --export=ALL,FORCE=${FORCE:-0}" score "$M" "$DS" nucleus
    printf "  %-22s -> %s (after %s)\n" "score (nucleus)" "$JOBID" "$GEN"; N=$((N+1))
    echo
  done
done

echo "Queued $N jobs."
echo "Watch:   squeue -u \$USER"
echo "Results: results/halluguard_rebuttal/hgreb_<model>_<dataset>[_nucleus][_last|_mid].json"
echo
echo "READ 'mean numerical rank of K' FIRST, in both runs. If the beam-search run shows a rank well"
echo "below 10 and the nucleus run does not, the decoder was the story and the beam number is void."
