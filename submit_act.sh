#!/bin/bash
# submit_act.sh -- extract pooled Activation Tensors for the ACT-ViT comparison row.
#
#   bash submit_act.sh                       # TyDiQA + TruthfulQA on Qwen (the recommended start)
#   DATASETS="tydiqa_gp" bash submit_act.sh
#   NPOOL=32 MAXGB=200 DATASETS="triviaqa" bash submit_act.sh    # TriviaQA needs a harder pool
#
# RUN FROM A NEWTON TERMINAL. These are GPU jobs.
#
# WHY ONLY TWO DATASETS BY DEFAULT. Storage, not compute. At ACT-ViT's default (L_p, N_p) = (8, 100)
# a Qwen beam is 5.5 MiB:
#
#   dataset      beams     size @ (8,100)   time (est)
#   tydiqa_gp    4,400          23 GB        00:40:00
#   truthfulqa   8,170          44 GB        01:10:00
#   nq_open     36,100         194 GB        04:00:00     <- needs a decision about disk
#   triviaqa    99,600         532 GB        09:00:00     <- refused at default --max-gb
#
# N_p IS THE KNOB. Our completions cap at 64 new tokens, so N_p = 100 pads by replication and only
# the layer axis (29 -> 8) is actually compressed. N_p = 32 cuts TriviaQA to 170 GB. That changes
# ACT-ViT's input, so it is a deliberate choice to record in the results, not a silent optimisation
# -- their own ablation runs (L_p, N_p) down to (4, 20) and still beats the best probe by ~6 points,
# so a smaller N_p is defensible and cheap to justify.
#
# WHAT THIS DOES NOT DO. It does not train ACT-ViT. Their Linear Adapter and ViT backbone are
# supervised and must be fit inside a split, so they live in the harness (methods/act_vit.py) and
# consume what this writes. One forward pass here serves five seeds x two protocols there.

set -euo pipefail

MODELS="${MODELS:-qwen-2.5-7b-instruct}"
DATASETS="${DATASETS:-tydiqa_gp truthfulqa}"
PART="${PART:-highgpu}"
NPOOL="${NPOOL:-100}"
MAXGB="${MAXGB:-60}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$HERE/slurm/act_extract.slurm"
[ -f "$STAGE" ] || { echo "ERROR: $STAGE not found" >&2; exit 1; }

# SLURM does not create the directory for --output, and a job whose log file cannot be opened dies
# before it runs. Make it here, on the login node, rather than inside the job where it is too late.
mkdir -p "$HERE/slurm_logs"

# Host RAM must exceed the array 59 allocates, with headroom for the model. Sized from the table
# above rather than one flat value -- a flat 80G silently fails on nq_open.
mem_for()  { case "$1" in tydiqa_gp) echo "80G";; truthfulqa) echo "110G";;
                          nq_open) echo "260G";; triviaqa) echo "600G";; *) echo "120G";; esac; }
time_for() { case "$1" in tydiqa_gp) echo "00:40:00";; truthfulqa) echo "01:10:00";;
                          nq_open) echo "04:00:00";; triviaqa) echo "09:00:00";;
                          *) echo "03:00:00";; esac; }

# Sets a global rather than echoing: called as J=$(sub ...) this would run in a subshell where
# `exit` kills only the subshell, and the driver would report success for jobs it never queued.
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
echo "pool    : (8, $NPOOL)   max_gb: $MAXGB"
echo

N=0
for M in $MODELS; do
  for DS in $DATASETS; do
    sub "-p $PART --gres=gpu:1 --mem=$(mem_for $DS) --time=$(time_for $DS) \
         --job-name=act-${DS:0:6}" "$M" "$DS" "$NPOOL" "$MAXGB"
    printf "  %-28s -> %s  (mem %s, %s)\n" "$M / $DS" "$JOBID" "$(mem_for $DS)" "$(time_for $DS)"
    N=$((N+1))
  done
done

echo
echo "Queued $N jobs."
echo "Watch:   squeue -u \$USER"
echo "If squeue is EMPTY, the jobs did not fail to submit -- they failed to START. Check:"
echo "    sacct -u \$USER --starttime today --format=JobID,JobName%14,State,Elapsed,ExitCode"
echo "An Elapsed of 00:00:00 or 00:00:01 means the module/conda preamble died; read the .err."
echo "Output:  ../data-acttensors/<model>/<dataset>_at_L8_N${NPOOL}.npz"
echo
echo "READ THE 'NOTE:' LINE in each log. If it says n-pool exceeds the longest completion, the token"
echo "axis is being replicated rather than pooled -- that is faithful to their default but it is the"
echo "reason TriviaQA does not fit, and the log prints the size it would take at the honest N_p."
