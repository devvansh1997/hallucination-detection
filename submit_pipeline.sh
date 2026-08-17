#!/bin/bash
# submit_pipeline.sh -- queue an entire model's pipeline as one dependency graph.
#
#   bash submit_pipeline.sh llama-3.1-8b
#
# Submits and returns immediately. SLURM runs the rest: nothing else to babysit, and the last
# job prints a consolidated data + inference summary.
#
# WHY A GRAPH AND NOT ONE BIG JOB
#   Stages want different machines -- generation and extraction need a GPU, the phase-3 evals
#   are pure CPU. One monolithic job would hold a GPU idle through hours of CPU work, need a
#   single ~30h wall limit, and lose everything if it died near the end. As a graph the three
#   datasets run side by side and the whole thing lands in roughly 15h instead of ~40h.
#
# WHAT IS DELIBERATELY *NOT* PARALLEL, and why (each of these is a real race we hit or found):
#   1. prefetch  the model is gated and ~16GB; three cold generation jobs would pull it into the
#                same HF cache simultaneously. Everything waits on this one.
#   2. adapter   49_harp_adapter.py:287 writes ONE adapter_summary.json per model, so it runs
#                once with --dataset all, after every generation has finished.
#   3. harp      main.py:109 keys its lm_head SVD cache on the MODEL, not the dataset, and
#                main.py:64-68 mkdir() is check-then-create. The first dataset runs alone to
#                materialise both; the rest then fan out safely.
#
# TriviaQA is excluded by default -- it is ~10x the others and belongs in its own submission.
# Add it with:  DATASETS="tydiqa_gp truthfulqa nq_open triviaqa" bash submit_pipeline.sh <model>

set -euo pipefail

MODEL="${1:?usage: submit_pipeline.sh <model_folder>   e.g. llama-3.1-8b}"
DATASETS="${DATASETS:-tydiqa_gp truthfulqa nq_open}"
REPO_SLURM="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm/pipe_stage.slurm"

GPU_PART="${GPU_PART:-highgpu}"
CPU_PART="${CPU_PART:-normal}"

sub() {  # sub "<sbatch opts>" <stage> [dataset] ; echoes the new job id
    local opts="$1"; shift
    local out
    out=$(sbatch --parsable $opts "$REPO_SLURM" "$@")
    echo "$out"
}

echo "model=$MODEL   datasets=$DATASETS"
echo "stage script: $REPO_SLURM"
echo

# --- 1. prefetch (serialised: gated ~16GB download, once) ----------------------------------
J_PRE=$(sub "-p $CPU_PART --mem=32G --time=02:00:00 --job-name=pre-$MODEL" prefetch "$MODEL")
echo "prefetch            -> $J_PRE"

GEN_IDS=(); EVAL_IDS=()
for DS in $DATASETS; do
    # --- 2. generate + validate (GPU) ------------------------------------------------------
    J_GEN=$(sub "-p $GPU_PART --gres=gpu:1 --mem=80G --time=06:00:00 \
                 --dependency=afterok:$J_PRE --job-name=gen-$DS" gen "$MODEL" "$DS")
    # --- 3. extract features (GPU) ---------------------------------------------------------
    J_EXT=$(sub "-p $GPU_PART --gres=gpu:1 --mem=100G --time=06:00:00 \
                 --dependency=afterok:$J_GEN --job-name=ext-$DS" extract "$MODEL" "$DS")
    # --- 4. evaluate, BOTH split protocols (CPU) -------------------------------------------
    J_EVL=$(sub "-p $CPU_PART --mem=100G --time=24:00:00 \
                 --dependency=afterok:$J_EXT --job-name=evl-$DS" eval "$MODEL" "$DS")
    GEN_IDS+=("$J_GEN"); EVAL_IDS+=("$J_EVL")
    printf "  %-11s gen->%s  extract->%s  eval->%s\n" "$DS" "$J_GEN" "$J_EXT" "$J_EVL"
done

# --- 5. adapter, once, after ALL generations ------------------------------------------------
DEP_GEN=$(IFS=:; echo "${GEN_IDS[*]}")
J_ADP=$(sub "-p $CPU_PART --mem=32G --time=04:00:00 \
             --dependency=afterok:$DEP_GEN --job-name=adp-$MODEL" adapter "$MODEL")
echo "adapter (all ds)    -> $J_ADP"

# --- 6. HARP: first dataset alone to build the model-keyed SVD, then the rest ---------------
FIRST=$(echo $DATASETS | cut -d' ' -f1)
REST=$(echo $DATASETS | cut -d' ' -f2-)
J_H1=$(sub "-p $GPU_PART --gres=gpu:1 --mem=200G --time=12:00:00 \
            --dependency=afterok:$J_ADP --job-name=harp-$FIRST" harp "$MODEL" "$FIRST")
echo "harp $FIRST (svd)   -> $J_H1"
HARP_IDS=("$J_H1")
for DS in $REST; do
    J_H=$(sub "-p $GPU_PART --gres=gpu:1 --mem=200G --time=12:00:00 \
               --dependency=afterok:$J_H1 --job-name=harp-$DS" harp "$MODEL" "$DS")
    HARP_IDS+=("$J_H")
    echo "harp $DS           -> $J_H"
done

# --- 7. summary, after everything ------------------------------------------------------------
DEP_ALL=$(IFS=:; echo "${EVAL_IDS[*]}:${HARP_IDS[*]}")
J_SUM=$(sub "-p $CPU_PART --mem=32G --time=01:00:00 \
             --dependency=afterany:$DEP_ALL --job-name=sum-$MODEL" summary "$MODEL")
echo "summary             -> $J_SUM"
echo
echo "Queued. Watch with:  squeue -u \$USER"
echo "Final report lands in the log for job $J_SUM, and in"
echo "  results/$MODEL/pipeline_summary.json"
echo
echo "NOTE: summary uses afterany, so it still runs (and reports what is missing) even if a"
echo "      branch fails. Every other dependency is afterok -- a failed stage stops its own"
echo "      branch rather than feeding a later stage half-written inputs."
