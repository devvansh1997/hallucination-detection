#!/bin/bash
# submit_falcon.sh -- Falcon-H1-7B-Base through our method (T-014): generate + label, extract, evaluate.
#
#   bash submit_falcon.sh                                 # TruthfulQA and TyDiQA-GP (the default)
#   DATASETS="nq_open triviaqa" bash submit_falcon.sh     # the two larger datasets
#   DRY=1 bash submit_falcon.sh                           # print the plan, queue nothing
#
# RUN FROM A NEWTON TERMINAL, after git pull. Needs hal-det-h1 (submit_h1.sh; K1-K3 passed 2026-09-21).
#
# Same stages and scripts as submit_pipeline.sh -- slurm/pipe_stage.slurm with gen = 39 + 40,
# extract = 42, eval = 44 --part a under both splits -- run in hal-det-h1 (ENV_NAME), so Falcon's Mamba
# kernels and their gcc module are loaded. Differences from submit_pipeline.sh, on purpose:
#   * no prefetch: the model and the datasets are already cached (the pilot ran offline);
#   * no adapter, HARP or summary yet: our method first. HARP needs a Falcon key in HARP-Code's
#     main.py and in 49_harp_adapter.MODEL_OURS_TO_HARP before its stage can run;
#   * TriviaQA's evaluation is NOT queued here. As for Qwen and LLaMA it runs fanned out per condition
#     on Stokes (high RAM, separate scheduler, so it cannot wait on a Newton job): once ext-triviaqa has
#     succeeded, from a STOKES terminal run   bash submit_trivia_eval.sh falcon-h1-7b-base
#     (CPU only, so it runs in hal-det; the Mamba kernels are not needed to evaluate);
#   * each generation waits on a CPU prefetch (39 --prefetch-only): dataset and BLEURT-20 checkpoint are
#     cached before the GPU job starts (a per-job BLEURT download killed two generations, 2026-09-22);
#   * after this submission's generations, 72_empty_answers.py counts empty answers for every model
#     and dataset (results/empty_answers.json);
#   * logs go to slurm_logs/falcon_<stage>-<dataset>_<jobid>.out / .err.
# Every stage is resumable (pipe_stage.slurm skips finished artifacts), so a rerun after a failure
# redoes only what is missing.
#
# PRE-REGISTERED before any Falcon result (reports/TICKETS.md, T-014): the paper's layer window,
# blocks 15-23 (hidden-state indices 16..24), unchanged -- absolute, as for Qwen and LLaMA; the reported
# number is triple_concat with the random forest under the question-level split, seeds {42,0,1,2,3}.
#
# BUDGETS. TruthfulQA, TyDiQA-GP and NQ-Open use the pipeline's defaults. TriviaQA uses the budgets
# submit_pipeline.sh learned the hard way: 16 h to generate (39 has no checkpoint, so a timeout loses
# the whole run), 12 h and 256G to extract (every beam's raw states sit in RAM before sharding; 100G was
# OOM-killed, LLaMA needed 256G, Falcon's hidden size 3072 is below both Qwen's and LLaMA's).
#
# DISK, scaled from LLaMA's TriviaQA features (~35 GB for 99,600 answers at hidden size 4096) to
# Falcon's 3072: roughly 26 GB kept for TriviaQA, 10 GB for NQ-Open, 2 GB for TruthfulQA and 1 GB for
# TyDiQA-GP under ../data/falcon-h1-7b-base/, plus a raw-state store 2-3x that size while each
# extraction runs (up to ~80 GB for TriviaQA), deleted when it finishes. Check the quota and
# `du -sh ../data/falcon-h1-7b-base` (the measured size of the first two) before submitting TriviaQA.
#
# OUTPUT: results/falcon-h1-7b-base/session06_phase3_partA_<dataset>.json (question-level; the main
# table reads harp.triple_concat.RF_mean) and results/falcon-h1-7b-base-answersplit/ (answer-level).
# Both have the same file name: copy the FOLDERS, and read "split_unit" inside each file (44, ad48ff2).

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
ANALYSIS_SCRIPT="$HD_REPO/slurm/analysis_stage.slurm"

JOBID=""
submit() {   # submit <label> <sbatch args...> -- <batch script> <script args...>; sets JOBID
    local label="$1"; shift
    local opts=() args=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done
    shift
    args=("$@")
    if [ "${DRY:-0}" = "1" ]; then
        printf "  %-22s would queue: sbatch %s %s\n" "$label" "${opts[*]}" "${args[*]}"
        JOBID="DRY"; return
    fi
    local out
    if ! out=$(sbatch --parsable "${opts[@]}" "${args[@]}" 2>&1); then
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
GEN_IDS=()
TRIVIA=0
PREV_PRE=""
for DS in $DATASETS; do
    case "$DS" in
        triviaqa) GEN_TIME=16:00:00; EXT_TIME=12:00:00; EXT_MEM=256G ;;
        *)        GEN_TIME=06:00:00; EXT_TIME=06:00:00; EXT_MEM=100G ;;
    esac
    # pre: 39 --prefetch-only on a CPU node caches the dataset and the ~2.1 GB BLEURT-20 checkpoint
    # (persistent JUDGE_DOWNLOAD_CACHE), so generation never downloads on a GPU node -- the cause of
    # both failed generations on 2026-09-22. Seconds once cached.
    # SERIALISED (afterany on the previous prefetch): two prefetch jobs downloading the same checkpoint
    # into the same cache raced, and one moved the finished temp file while the other still wanted it --
    # FileNotFoundError on a .incomplete file killed pre-triviaqa on 2026-09-22.
    PRE_DEP=""
    if [ -n "$PREV_PRE" ]; then PRE_DEP="--dependency=afterany:$PREV_PRE"; fi
    submit "pre-$DS" -p "$CPU_PART" --mem=32G --cpus-per-task=2 --time=02:00:00 $PRE_DEP \
        --job-name="pre-$DS" --output=slurm_logs/falcon_%x_%j.out --error=slurm_logs/falcon_%x_%j.err \
        --export=ALL,HD_REPO=$HD_REPO \
        -- "$ANALYSIS_SCRIPT" 39_generate_dataset.py --dataset "$DS" --model_folder "$MODEL" --prefetch-only
    J_PRE=$JOBID
    if [ "$J_PRE" != "DRY" ]; then PREV_PRE="$J_PRE"; fi
    DEP=""; if [ "$J_PRE" != "DRY" ]; then DEP="--dependency=afterok:$J_PRE"; fi
    # gen: 39 generates and labels one question at a time, then 40 validates. extract: 42 re-forwards
    # every answer. eval: 44, both split protocols, CPU.
    submit "gen-$DS" -p "$GPU_PART" --gres=gpu:1 --mem=80G --time=$GEN_TIME $DEP $(common "gen-$DS") \
        -- "$STAGE_SCRIPT" gen "$MODEL" "$DS"
    J_GEN=$JOBID
    DEP=""
    if [ "$J_GEN" != "DRY" ]; then GEN_IDS+=("$J_GEN"); DEP="--dependency=afterok:$J_GEN"; fi
    submit "ext-$DS" -p "$GPU_PART" --gres=gpu:1 --mem=$EXT_MEM --time=$EXT_TIME $DEP $(common "ext-$DS") \
        -- "$STAGE_SCRIPT" extract "$MODEL" "$DS"
    J_EXT=$JOBID
    if [ "$DS" = "triviaqa" ]; then
        TRIVIA=1
        printf "  %-22s NOT queued here -- from a Stokes terminal, after ext-triviaqa succeeds\n" "evl-$DS"
        continue
    fi
    DEP=""; if [ "$J_EXT" != "DRY" ]; then DEP="--dependency=afterok:$J_EXT"; fi
    submit "evl-$DS" -p "$CPU_PART" --mem=100G --time=24:00:00 $DEP $(common "evl-$DS") \
        -- "$STAGE_SCRIPT" eval "$MODEL" "$DS"
done

# Empty answers for every model and dataset whose sequences exist, once this submission's generations end
# (afterany: a failed generation should not stop the count for the others).
DEP=""
if [ "${#GEN_IDS[@]}" -gt 0 ]; then DEP="--dependency=afterany:$(IFS=:; echo "${GEN_IDS[*]}")"; fi
submit "empty-answers" -p "$CPU_PART" --mem=32G --cpus-per-task=2 --time=01:00:00 $DEP \
    --job-name=empty-answers --export=ALL,HD_REPO=$HD_REPO \
    -- "$ANALYSIS_SCRIPT" 72_empty_answers.py

echo
echo "Watch: squeue -u \$USER   (ext-* and evl-* wait on their dataset's previous stage)"
echo "Logs:  slurm_logs/falcon_<stage>-<dataset>_<jobid>.out/.err; empty answers: slurm_logs/empty-answers_<jobid>.out"
echo "Read first: gen's 'Version check PASSED', 'Empty completions' and 40's [PASS] lines, then 42's '[PASS] zero empty windows'."
echo "Result: results/$MODEL/session06_phase3_partA_<dataset>.json -> harp.triple_concat.RF_mean"
if [ "$TRIVIA" = "1" ]; then
    echo
    echo "TriviaQA: when ext-triviaqa has finished, from a STOKES terminal:  bash submit_trivia_eval.sh $MODEL"
fi
