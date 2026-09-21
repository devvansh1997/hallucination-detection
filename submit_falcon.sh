#!/bin/bash
# submit_falcon.sh -- Falcon-H1-7B-Base through our method (T-014): generate + label, extract, evaluate.
#
#   bash submit_falcon.sh                                 # TruthfulQA and TyDiQA-GP (the default)
#   DRY=1 bash submit_falcon.sh                           # print the plan, queue nothing
#   DATASETS="nq_open" bash submit_falcon.sh              # later: the larger datasets
#
# RUN FROM A NEWTON TERMINAL, after git pull. Needs hal-det-h1 (submit_h1.sh; K1-K3 passed 2026-09-21).
#
# Same stages and scripts as submit_pipeline.sh -- slurm/pipe_stage.slurm with gen = 39 + 40,
# extract = 42, eval = 44 --part a under both splits -- run in hal-det-h1 (ENV_NAME), so Falcon's Mamba
# kernels and their gcc module are loaded. Differences from submit_pipeline.sh, on purpose:
#   * no prefetch: the model and the datasets are already cached (the pilot ran offline);
#   * no adapter, HARP or summary yet: our method first. HARP needs a Falcon key in HARP-Code's
#     main.py and in 49_harp_adapter.MODEL_OURS_TO_HARP before its stage can run;
#   * logs go to slurm_logs/falcon_<stage>-<dataset>_<jobid>.out / .err.
# Every stage is resumable (pipe_stage.slurm skips finished artifacts), so a rerun after a failure
# redoes only what is missing.
#
# PRE-REGISTERED before any Falcon result (reports/TICKETS.md, T-014): the paper's layer window,
# blocks 15-23 (hidden-state indices 16..24), unchanged -- absolute, as for Qwen and LLaMA; the reported
# number is triple_concat with the random forest under the question-level split, seeds {42,0,1,2,3}.
#
# DISK, scaled from Qwen's ~50 GB for three datasets: about 7 GB kept for TruthfulQA and 4 GB for
# TyDiQA-GP under ../data/falcon-h1-7b-base/, plus a raw-state store 2-3x that size while each
# extraction runs, deleted when it finishes.
#
# OUTPUT: results/falcon-h1-7b-base/session06_phase3_partA_<dataset>.json (question-level; the main
# table reads harp.triple_concat.RF_mean) and results/falcon-h1-7b-base-answersplit/ (answer-level).

set -euo pipefail

HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs"

MODEL="falcon-h1-7b-base"
ENV_NAME="${ENV_NAME:-hal-det-h1}"
DATASETS="${DATASETS:-tydiqa_gp truthfulqa}"
GPU_PART="${GPU_PART:-highgpu}"
CPU_PART="${CPU_PART:-normal}"
EXCLUDE="${EXCLUDE:-evc103}"
STAGE_SCRIPT="$HD_REPO/slurm/pipe_stage.slurm"

JOBID=""
submit() {   # submit <label> <sbatch args...> -- <stage args...>; sets JOBID
    local label="$1"; shift
    local opts=() args=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done
    shift
    args=("$@")
    if [ "${DRY:-0}" = "1" ]; then
        printf "  %-22s would queue: sbatch %s %s %s\n" "$label" "${opts[*]}" "$STAGE_SCRIPT" "${args[*]}"
        JOBID="DRY"; return
    fi
    local out
    if ! out=$(sbatch --parsable "${opts[@]}" "$STAGE_SCRIPT" "${args[@]}" 2>&1); then
        echo "ERROR: sbatch rejected $label -- nothing further has been queued." >&2
        printf '  %s\n' "$out" >&2
        exit 1
    fi
    [ -n "$out" ] || { echo "ERROR: sbatch returned an empty job id for $label" >&2; exit 1; }
    JOBID="$out"
    printf "  %-22s -> %s\n" "$label" "$JOBID"
}

common() {   # common <job name>: options every stage shares
    echo "--job-name=$1 --output=slurm_logs/falcon_%x_%j.out --error=slurm_logs/falcon_%x_%j.err" \
         "--export=ALL,ENV_NAME=$ENV_NAME${EXCLUDE:+ --exclude=$EXCLUDE}"
}

echo "model=$MODEL  env=$ENV_NAME  datasets=$DATASETS"
for DS in $DATASETS; do
    # gen: 39 generates and labels one question at a time (pilot: ~1 s/question before labelling),
    # then 40 validates. extract: 42 re-forwards every answer. eval: 44, both split protocols, CPU.
    submit "gen-$DS" -p "$GPU_PART" --gres=gpu:1 --mem=80G --time=06:00:00 $(common "gen-$DS") \
        -- gen "$MODEL" "$DS"
    J_GEN=$JOBID
    DEP=""; if [ "$J_GEN" != "DRY" ]; then DEP="--dependency=afterok:$J_GEN"; fi
    submit "ext-$DS" -p "$GPU_PART" --gres=gpu:1 --mem=100G --time=06:00:00 $DEP $(common "ext-$DS") \
        -- extract "$MODEL" "$DS"
    J_EXT=$JOBID
    DEP=""; if [ "$J_EXT" != "DRY" ]; then DEP="--dependency=afterok:$J_EXT"; fi
    submit "evl-$DS" -p "$CPU_PART" --mem=100G --time=24:00:00 $DEP $(common "evl-$DS") \
        -- eval "$MODEL" "$DS"
done

echo
echo "Watch: squeue -u \$USER   (ext-* and evl-* wait on their dataset's previous stage)"
echo "Logs:  slurm_logs/falcon_<stage>-<dataset>_<jobid>.out/.err"
echo "Read first: gen's 'Version check PASSED' and 40's [PASS] lines, then 42's '[PASS] zero empty windows'."
echo "Result: results/$MODEL/session06_phase3_partA_<dataset>.json -> harp.triple_concat.RF_mean"
