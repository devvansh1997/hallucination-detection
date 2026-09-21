#!/bin/bash
# submit_h1.sh -- Falcon-H1 on its Mamba kernels (T-014): build hal-det-h1, then pilot and diagnose on it.
#
#   bash submit_h1.sh                 # 3 GPU jobs: build + check, then pilot and diagnostic (both afterok)
#   DRY=1 bash submit_h1.sh           # print the plan, queue nothing
#   SKIP_BUILD=1 bash submit_h1.sh    # hal-det-h1 already built and checked: pilot + diagnostic only
#
# RUN FROM A NEWTON TERMINAL, after git pull.
#
# ISOLATION. hal-det is never modified: the build clones it to hal-det-h1 and installs only causal-conv1d and
# mamba-ssm (plus einops/ninja if missing) there, with --no-deps, so torch and transformers stay the versions
# 39 pins. Reports go to results/pilot_kernels/; the first pilot's results/pilot/ files stay as the baseline.
#
# DISK. The clone copies hal-det's pip-installed files (torch and its CUDA libraries, TensorFlow for BLEURT),
# so expect roughly the size of hal-det; the build log prints both sizes. If Falcon-H1 is dropped:
#   conda env remove -n hal-det-h1
#
# WHAT IS QUEUED:
#   h1env    slurm/build_h1_env.slurm: clone, build, 71_check_fast_path.py   (GPU, 16 CPUs, 80G, 04:00:00;
#            compiling mamba-ssm is most of it if no prebuilt wheel matches torch 2.13 / CUDA 12.6)
#   h1pilot  67 on 20 TruthfulQA + 10 TyDiQA-GP questions: gates, and generation hours for all four datasets
#   h1diag   68 on 5 + 3 questions: seconds per question and looping share for each decoding variant
# If h1env fails, the other two stay PENDING (DependencyNeverSatisfied): scancel them, fix, rerun.

set -euo pipefail

HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs" "$HD_REPO/results/pilot_kernels"

MODEL="${MODEL:-falcon-h1-7b-base}"
NEW_ENV="${NEW_ENV:-hal-det-h1}"
OUT_DIR="${OUT_DIR:-results/pilot_kernels}"
PART="${PART:-highgpu}"
EXCLUDE="${EXCLUDE:-evc103}"

JOBID=""
submit() {   # submit <label> <sbatch args...>; sets JOBID
    local label="$1"; shift
    if [ "${DRY:-0}" = "1" ]; then
        printf "  %-8s would queue: sbatch %s\n" "$label" "$*"; JOBID="DRY"; return
    fi
    local out
    if ! out=$(sbatch --parsable "$@" 2>&1); then
        echo "ERROR: sbatch rejected $label -- nothing further has been queued." >&2
        printf '  %s\n' "$out" >&2
        exit 1
    fi
    [ -n "$out" ] || { echo "ERROR: sbatch returned an empty job id for $label" >&2; exit 1; }
    JOBID="$out"
    printf "  %-8s -> %s\n" "$label" "$JOBID"
}

DEP=""
if [ -z "${SKIP_BUILD:-}" ]; then
    submit h1env -p "$PART" --gres=gpu:1 --cpus-per-task=16 --mem=80G --time=04:00:00 --job-name=h1env \
        ${EXCLUDE:+--exclude=$EXCLUDE} \
        --export=ALL,HD_REPO=$HD_REPO,NEW_ENV=$NEW_ENV,MODEL=$MODEL,OUT_DIR=$OUT_DIR \
        "$HD_REPO/slurm/build_h1_env.slurm"
    if [ "$JOBID" != "DRY" ]; then DEP="--dependency=afterok:$JOBID"; fi
fi

COMMON="HD_REPO=$HD_REPO,ENV_NAME=$NEW_ENV,OUT_DIR=$OUT_DIR"
submit h1pilot -p "$PART" --gres=gpu:1 --mem=64G --time=02:00:00 --job-name=h1pilot $DEP \
    ${EXCLUDE:+--exclude=$EXCLUDE} \
    --export=ALL,$COMMON,SCRIPT=67_pilot_new_model.py,N_TQA=20,N_TYDI=10 \
    "$HD_REPO/slurm/pilot_model.slurm" "$MODEL"
submit h1diag -p "$PART" --gres=gpu:1 --mem=64G --time=02:00:00 --job-name=h1diag $DEP \
    ${EXCLUDE:+--exclude=$EXCLUDE} \
    --export=ALL,$COMMON,SCRIPT=68_diagnose_generation.py \
    "$HD_REPO/slurm/pilot_model.slurm" "$MODEL"

echo
echo "Watch: squeue -u \$USER   (h1pilot and h1diag show PENDING (Dependency) until h1env ends)"
echo "Read first: the STEP lines in slurm_logs/h1env_<id>.err, then the K1-K3 lines in slurm_logs/h1env_<id>.out."
echo "Reports: $OUT_DIR/fastpath_$MODEL.json, pilot_$MODEL.json (generation_hours_estimate), diagnose_$MODEL.json"
