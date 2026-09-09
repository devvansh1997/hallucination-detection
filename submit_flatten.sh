#!/bin/bash
# submit_flatten.sh -- the matched-width control for the multilinear restriction.
#
#   export HD_REPO=$PWD
#   bash submit_flatten.sh                                   # 2 models x 2 datasets x 2 readouts
#   MODELS=llama-3.1-8b bash submit_flatten.sh                # one model
#   DATASETS="tydiqa_gp truthfulqa nq_open triviaqa" bash submit_flatten.sh
#   DATASETS=truthfulqa bash submit_flatten.sh               # just the one that has not run
#   READOUTS=RF bash submit_flatten.sh                       # just the readout the paper reports
#   FORCE=1 bash submit_flatten.sh                           # ignore the skip guard
#
#   # the sample-efficiency curve: same four arms, five training sizes, test set held fixed
#   READOUTS=RF TAG=sampeff JOB_TIME=08:00:00 \
#     EXTRA="--train-frac 0.05 0.1 0.25 0.5 1.0" bash submit_flatten.sh
#
# BOTH READOUTS, DELIBERATELY. The first run used LR only, because that was 61's default. The paper
# reports RF, and on Qwen/TyDiQA the two differ by about five points for our arm -- triple_concat is
# 87.9 with RF against 84.1 with LR. A control against a configuration we do not report answers a
# question nobody asked, so RF is the one that decides this and LR is kept for completeness.
#
# CPU ONLY. 61 is sklearn and numpy; asking for a GPU only lengthens the queue.
#
# MEASURED: tydiqa_gp / LR took 1129s for four arms x five seeds. TruthfulQA has 1.86x the rows and
# the randomized SVD scales with rows x width, so ~35-45 min. Walls below carry roughly 3x.

set -euo pipefail

: "${HD_REPO:?export HD_REPO=/path/to/your/hallucination-detection}"
STAGE="${HD_REPO}/slurm/flatten_stage.slurm"
[ -f "$STAGE" ] || { echo "ERROR: $STAGE not found -- is HD_REPO right?" >&2; exit 1; }

# BOTH MODELS. A control that rewrites a contribution cannot rest on one model. The first run
# was Qwen-only because that was this script's default, which was an oversight rather than a
# decision -- llama-3.1-8b base has the same phase-2 features and is the checkpoint the
# comparison in the main table uses.
MODELS="${MODELS:-qwen-2.5-7b-instruct llama-3.1-8b}"
DATASETS="${DATASETS:-tydiqa_gp truthfulqa}"
READOUTS="${READOUTS:-RF LR}"
PART="${PART:-highgpu}"
TAG="${TAG:-}"
# Appended to 61's command line. --train-frac switches it to the sample-efficiency curve, which
# costs roughly 2-3x the single-size control (the fractions sum to 1.9x the training rows, and the
# test-side transform is full size at every point), so raise JOB_TIME with it.
EXTRA="${EXTRA:-}"

mkdir -p "$HD_REPO/slurm_logs" "$HD_REPO/results/flatten_control"

job_time() { [ -n "${JOB_TIME:-}" ] && { echo "$JOB_TIME"; return; }
             case "$1" in tydiqa_gp) echo "01:30:00";; truthfulqa) echo "03:00:00";;
                          nq_open) echo "08:00:00";; triviaqa) echo "20:00:00";;
                          *) echo "04:00:00";; esac; }
job_mem()  { [ -n "${MEM:-}" ] && { echo "$MEM"; return; }
             case "$1" in tydiqa_gp) echo "64G";; truthfulqa) echo "96G";;
                          nq_open) echo "220G";; triviaqa) echo "500G";; *) echo "96G";; esac; }

# Sets a global rather than echoing: called as J=$(sub ...) this runs in a subshell where `exit`
# kills only the subshell and the driver reports success for jobs it never queued.
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
echo "readouts: $READOUTS"
echo

N=0
for MO in $MODELS; do
  for DS in $DATASETS; do
    for RO in $READOUTS; do
      sub "-p $PART --mem=$(job_mem $DS) --cpus-per-task=8 --time=$(job_time $DS) \
           --job-name=flat-${DS:0:4}-${RO} \
           --export=ALL,HD_REPO=$HD_REPO,HD_DATA=${HD_DATA:-},FORCE=${FORCE:-0}" \
          "$MO" "$DS" "$RO" "$TAG" ${EXTRA:-}
      printf "  %-24s %-12s %-3s -> %s  (mem %s, %s)\n" "$MO" "$DS" "$RO" "$JOBID" \
             "$(job_mem $DS)" "$(job_time $DS)"
      N=$((N+1))
    done
  done
done

echo
echo "Queued $N jobs."
echo "Watch:   squeue -u \$USER"
echo "Results: results/flatten_control/flatten_<model>_<dataset>_<readout>.json"
echo
echo "READ THE RF FILES FIRST. RF is the readout the paper reports; the LR runs are context only."
echo "The number that decides this is delta_vs_hosvd_pts for flat_pca: negative means the"
echo "unrestricted projection wins and the Kronecker restriction is costing accuracy."
