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
#   dataset      beams     size @ (8,100)   wall
#   tydiqa_gp    4,400          23 GB        00:25:00    measured 11:16
#   truthfulqa   8,170          44 GB        00:30:00    measured 12:22
#   nq_open     36,100         194 GB        01:30:00    <- needs a decision about disk
#   triviaqa    99,600         532 GB        04:00:00    <- refused at default --max-gb
#
# N_eff IS THE KNOB. Their preprocessing ZERO-pads a response to N_MAX = 100 before pooling, and our
# median completion is 16-17 tokens. So at N_eff = 100 roughly 84 of the 100 columns are zeros and
# the ViT is handed 800 activation pixels of which most carry nothing. N_eff = 20 sits inside their
# own published ablation grid (Figure 3 sweeps (L_p, N_p) over {4,8} x {20,100}), so it is their
# hyperparameter chosen for our input lengths, not a deviation from their method. Extract at 100 and
# derive 20 on CPU -- no second forward pass:
#
#   python 59_extract_act_tensors.py --repool-from <the N100 file> --n-pool 20
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
# Measured, not guessed: 798638 tydiqa_gp 11:16 and 798639 truthfulqa 12:22, both
# including model load. Roughly 2x headroom. A short wall is a feature -- a misconfigured
# job should fail fast, not burn an hour proving it.
time_for() { case "$1" in tydiqa_gp) echo "00:25:00";; truthfulqa) echo "00:30:00";;
                          nq_open) echo "01:30:00";; triviaqa) echo "04:00:00";;
                          *) echo "01:00:00";; esac; }

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
         --job-name=act-${DS:0:6} --export=ALL" "$M" "$DS" "$NPOOL" "$MAXGB"
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
echo ""
echo "NOTE: .out is BUFFERED and is discarded if a job is killed on TIMEOUT. The job traces"
echo "to stderr for that reason, so read the .err first when a job dies."
echo "The self-test is not run inside the job (it needs the ../ACT-ViT clone). Run it here:"
echo "    python 59_extract_act_tensors.py --self-test"
echo "Output:  ../data-acttensors/<model>/<dataset>_at_L8_N${NPOOL}.npz"
echo
echo "READ THE 'NOTE:' LINE in each log. If it says n-pool exceeds the longest completion, the token"
echo "axis is being replicated rather than pooled -- that is faithful to their default but it is the"
echo "reason TriviaQA does not fit, and the log prints the size it would take at the honest N_p."
