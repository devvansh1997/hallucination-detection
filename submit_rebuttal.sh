#!/bin/bash
# submit_rebuttal.sh -- every experiment for the ICLR 2027 rebuttal (submission 42795).
#
#   DRY=1 bash submit_rebuttal.sh                  # print the plan, queue nothing
#   bash submit_rebuttal.sh                        # queue everything
#   ONLY="kept responses cost" bash submit_rebuttal.sh
#   NO_TRIVIA=1 bash submit_rebuttal.sh            # skip the TriviaQA matched-storage cell (~61 GB)
#
# RUN FROM A PLAIN LOGIN-NODE SHELL after `git pull`: no conda env active, not inside srun/salloc.
# This script runs no Python. Each job activates its own env (hal-det; hal-det-h1 where Falcon-H1
# runs a forward pass). Safe to rerun: finished outputs are skipped and queued/running jobs are reused
# as dependencies, so a rerun after a failure queues only what is missing.
#
# STAGE            REVIEW POINT                           SCRIPTS          JOBS   EST. WALL TIME
# kept             zjPm Q3/W3, pjMe Q1: kept vs discarded  85               4 CPU  ~1 h
# responses        zjPm Q2: answers per question (K)       86               4 CPU  ~1-1.5 h
# cost             zjPm W2, pjMe Q3: time + peak memory    89               6      ~45 min
# fewer_labels     pjMe Q4/W1: ACT-ViT on 5-100% labels    87               4 GPU  ~1.5-2 h
# storage          fgja W2: ACT-ViT at LOTUS's storage     88 / 59 -> 56   20     ~1.5 h; TriviaQA ~6-9 h
# falcon_transfer  pjMe W2: transfer with a 3rd model      65 -> 66        16     ~2 h
# falcon_actvit    fgja W1, pjMe W2: ACT-ViT on Falcon-H1  59 -> 56         4     ~1-1.5 h
# Times assume jobs start promptly; a busy cluster can double them (TICKETS: contention).
#
# DISK (new files): storage ~15 GB + TriviaQA N=5 ~61 GB, falcon_transfer ~10 GB, falcon_actvit
# ~12 GB. `lfs quota` showed 893 / 1000 GB soft, 1010 GB hard on 2026-11-12.
#
# RESULTS: results/rebuttal/*.json, results/methods/act_vit_*_{L8N5,L4N10}.json,
# results/methods/act_vit_falcon-h1-7b-base_*.json, results/cross_model/cross_*falcon*.json.
# One summary of everything (inside hal-det):  python 90_rebuttal_summary.py

set -euo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "refusing: this shell is inside Slurm job $SLURM_JOB_ID (srun/salloc). Exit it and submit from the login node."; exit 1
fi
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    echo "refusing: conda env '$CONDA_DEFAULT_ENV' is active. Run 'conda deactivate' and submit from a plain shell."; exit 1
fi

HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$HD_REPO"
PARENT="$(cd "$HD_REPO/.." && pwd)"
mkdir -p "$HD_REPO/slurm_logs" "$HD_REPO/results/rebuttal"

GPU_PART="${GPU_PART:-highgpu}"
CPU_PART="${CPU_PART:-normal}"
EXCLUDE="${EXCLUDE:-evc103}"
CPUS="${CPUS:-8}"
ONLY="${ONLY:-kept responses cost fewer_labels storage falcon_transfer falcon_actvit}"

QWEN=qwen-2.5-7b-instruct
LLAMA=llama-3.1-8b
FALCON=falcon-h1-7b-base
MODELS="$QWEN $LLAMA"
SMALL="truthfulqa tydiqa_gp"
AT="$PARENT/data-acttensors"
XR="$PARENT/data-crossread"
RB="$HD_REPO/results/rebuttal"
MR="$HD_REPO/results/methods"

wants() { [[ " $ONLY " == *" $1 "* ]]; }
short() { case "$1" in "$QWEN") echo qwen ;; "$LLAMA") echo llam ;; "$FALCON") echo falc ;;
                       truthfulqa) echo tqa ;; tydiqa_gp) echo tydi ;; triviaqa) echo triv ;; *) echo "$1" ;; esac; }
mem_for() { if [ "$1" = truthfulqa ]; then echo "$2"; else echo "$3"; fi; }   # <ds> <tqa-mem> <other>
need() { [ -e "$1" ] || { echo "MISSING INPUT: $1"; MISSING=$((MISSING+1)); }; }

N=0; MISSING=0; JOBID=""
# queue <name> <gpu 0|1> <mem> <time> <env> <after-jobid|""> <done-test|""> <script> [args...]
#   done-test  "json:<path>"  finished when that JSON contains "complete": true
#              "file:<path>"  finished when that path exists (written only at the very end)
#              ""             always queue
queue() {
    local name="$1" gpu="$2" mem="$3" tim="$4" env="$5" after="$6" done_test="$7"; shift 7
    JOBID=""
    if [ -n "$done_test" ] && [ "${FORCE:-0}" != "1" ]; then
        local kind="${done_test%%:*}" path="${done_test#*:}"
        if { [ "$kind" = json ] && [ -f "$path" ] && grep -q '"complete": true' "$path"; } || \
           { [ "$kind" = file ] && [ -e "$path" ]; }; then
            printf "  %-30s SKIP (done)\n" "$name"; return 0
        fi
    fi
    local running
    running="$(squeue -h -u "$USER" -n "$name" -o %i 2>/dev/null | head -1 || true)"
    if [ -n "$running" ]; then
        JOBID="$running"; printf "  %-30s already queued/running -> %s\n" "$name" "$JOBID"; return 0
    fi
    local part="$CPU_PART" gres="" dep="" kindname="CPU"
    if [ "$gpu" = "1" ]; then part="$GPU_PART"; gres="--gres=gpu:1"; kindname="GPU"; fi
    if [ -n "$after" ]; then dep="--dependency=afterok:$after"; fi
    if [ "${DRY:-0}" = "1" ]; then
        printf "  %-30s would queue: %-7s %s mem %-5s %s %-10s %s\n" "$name" "$part" "$kindname" \
            "$mem" "$tim" "$env" "${after:+after $after}"
        JOBID="dry:$name"; N=$((N+1)); return 0
    fi
    local out
    if ! out=$(sbatch --parsable -p "$part" $gres --mem="$mem" --cpus-per-task="$CPUS" --time="$tim" \
            --job-name="$name" ${EXCLUDE:+--exclude=$EXCLUDE} $dep \
            --export=ALL,HD_REPO="$HD_REPO",ENV_NAME="$env" \
            "$HD_REPO/slurm/rebuttal_stage.slurm" "$@" 2>&1); then
        echo "ERROR: sbatch rejected $name:" >&2; printf '  %s\n' "$out" >&2; exit 1
    fi
    JOBID="$out"; N=$((N+1))
    printf "  %-30s -> %s\n" "$name" "$JOBID"
}

# ---- inputs, checked before anything is queued -------------------------------------------------
if [ "${SKIP_CHECKS:-0}" != "1" ]; then
    for M in $MODELS; do
        for D in $SMALL; do
            need "$PARENT/data/$M/${D}_phase2_features.npz"
            need "$AT/$M/${D}_at_L8_N20.npz"
        done
        need "$AT/$M/triviaqa_at_L8_N20.npz"
    done
    for D in $SMALL; do
        need "$PARENT/data/$FALCON/${D}_phase2_features.npz"
        need "$PARENT/data/$FALCON/${D}_sequences_v1.pt"
    done
    need "$XR/${QWEN}_reads_${LLAMA}/truthfulqa_window.json"
    need "$XR/${LLAMA}_reads_${QWEN}/truthfulqa_window.json"
    if [ "$MISSING" != 0 ]; then echo "stopping: $MISSING input(s) missing (SKIP_CHECKS=1 to override)"; exit 1; fi
fi

echo "Rebuttal batch from $HD_REPO   (stages: $ONLY)"

# ---- CPU checks first: the fastest results -----------------------------------------------------
if wants kept; then
    echo "[kept] 85: kept subspace vs discarded residual   (est. ~30-60 min per job)"
    for M in $MODELS; do for D in $SMALL; do
        queue "rb-kept-$(short $M)-$(short $D)" 0 "$(mem_for $D 96G 64G)" 08:00:00 hal-det "" \
            "json:$RB/kept_vs_discarded_${M}_${D}_RF.json" \
            85_kept_vs_discarded.py --dataset "$D" --model_folder "$M"
    done; done
fi

if wants responses; then
    echo "[responses] 86: answers per question, K = 10 / 5 / 3   (est. ~40-70 min per job)"
    for M in $MODELS; do for D in $SMALL; do
        queue "rb-resp-$(short $M)-$(short $D)" 0 "$(mem_for $D 96G 64G)" 08:00:00 hal-det "" \
            "json:$RB/responses_per_question_${M}_${D}_RF.json" \
            86_responses_per_question.py --dataset "$D" --model_folder "$M"
    done; done
fi

if wants cost; then
    echo "[cost] 89: time and peak memory on TruthfulQA   (est. ~10-30 min per job)"
    for M in $MODELS; do
        queue "rb-cost-fit-$(short $M)" 0 96G 03:00:00 hal-det "" \
            "json:$RB/cost_fit_flatpca_${M}_truthfulqa.json" \
            89_cost_profile.py --dataset truthfulqa --model_folder "$M" --stages fit_lotus fit_flatpca
        queue "rb-cost-gpu-$(short $M)" 1 80G 04:00:00 hal-det "" \
            "json:$RB/cost_act_vit_${M}_truthfulqa.json" \
            89_cost_profile.py --dataset truthfulqa --model_folder "$M" --stages extract act_vit
    done
    queue "rb-cost-map-l2q" 0 96G 03:00:00 hal-det "" \
        "json:$RB/cost_mapping_${QWEN}_truthfulqa_from_${LLAMA}.json" \
        89_cost_profile.py --dataset truthfulqa --model_folder "$QWEN" --source "$LLAMA" --stages mapping
    queue "rb-cost-map-q2l" 0 96G 03:00:00 hal-det "" \
        "json:$RB/cost_mapping_${LLAMA}_truthfulqa_from_${QWEN}.json" \
        89_cost_profile.py --dataset truthfulqa --model_folder "$LLAMA" --source "$QWEN" --stages mapping
fi

# ---- GPU work ----------------------------------------------------------------------------------
if wants fewer_labels; then
    echo "[fewer_labels] 87: ACT-ViT at 5/10/25/50/100% of training questions   (est. TyDi ~45 min, TQA ~1.5 h)"
    for M in $MODELS; do for D in $SMALL; do
        queue "rb-fewlab-$(short $M)-$(short $D)" 1 64G 12:00:00 hal-det "" \
            "json:$RB/actvit_fewer_labels_${M}_${D}.json" \
            87_actvit_fewer_labels.py --dataset "$D" --model_folder "$M"
    done; done
fi

storage_n5() {   # storage_n5 <model> <dataset> <shrink-mem> <train-mem> <train-time>
    local M="$1" D="$2"
    queue "rb-shrink-$(short $M)-$(short $D)" 0 "$3" 03:00:00 hal-det "" \
        "file:$AT/$M/${D}_at_L8_N5.npz" \
        88_shrink_act_tensors.py --dataset "$D" --model_folder "$M" --n-to 5
    queue "rb-avit5-$(short $M)-$(short $D)" 1 "$4" "$5" hal-det "$JOBID" \
        "file:$MR/act_vit_${M}_${D}_L8N5.json" \
        56_run_method.py --method act_vit --dataset "$D" --model_folder "$M" --m-l-eff 8 --m-n-eff 5 --tag L8N5
}

if wants storage; then
    echo "[storage] ACT-ViT at LOTUS's storage: L8 N5 (exact re-pool, 88) and L4 N10 (fresh extraction, 59)"
    echo "          (TriviaQA cells are queued last, at the end of this script)"
    for M in $MODELS; do for D in $SMALL; do
        storage_n5 "$M" "$D" 32G 64G 08:00:00
    done; done
    for M in $MODELS; do for D in $SMALL; do
        queue "rb-ext410-$(short $M)-$(short $D)" 1 80G 03:00:00 hal-det "" \
            "file:$AT/$M/${D}_at_L4_N10.npz" \
            59_extract_act_tensors.py --dataset "$D" --model_folder "$M" --l-pool 4 --n-pool 10
        queue "rb-avit410-$(short $M)-$(short $D)" 1 64G 08:00:00 hal-det "$JOBID" \
            "file:$MR/act_vit_${M}_${D}_L4N10.json" \
            56_run_method.py --method act_vit --dataset "$D" --model_folder "$M" --m-l-eff 4 --m-n-eff 10 --tag L4N10
    done; done
fi

if wants falcon_transfer; then
    echo "[falcon_transfer] 65 reads (GPU) -> 66 transfer (CPU): Falcon-H1 with Qwen and with LLaMA"
    for M in $MODELS; do for D in $SMALL; do
        # Falcon's detector on M's answers: Falcon reads M's answers (Falcon forward pass -> hal-det-h1).
        queue "rb-xr-falc-$(short $M)-$(short $D)" 1 "$(mem_for $D 80G 64G)" 03:00:00 hal-det-h1 "" \
            "file:$XR/${FALCON}_reads_${M}/${D}_window.json" \
            65_extract_cross_read.py --reader "$FALCON" --generator "$M" --dataset "$D"
        queue "rb-xf-falc-$(short $M)-$(short $D)" 0 "$(mem_for $D 96G 64G)" 06:00:00 hal-det "$JOBID" \
            "json:$HD_REPO/results/cross_model/cross_${FALCON}_to_${M}_${D}_RF.json" \
            66_cross_model.py --source "$FALCON" --target "$M" --dataset "$D"
        # M's detector on Falcon's answers: M reads Falcon's answers (M forward pass -> hal-det).
        queue "rb-xr-$(short $M)-falc-$(short $D)" 1 "$(mem_for $D 80G 64G)" 03:00:00 hal-det "" \
            "file:$XR/${M}_reads_${FALCON}/${D}_window.json" \
            65_extract_cross_read.py --reader "$M" --generator "$FALCON" --dataset "$D"
        queue "rb-xf-$(short $M)-falc-$(short $D)" 0 "$(mem_for $D 96G 64G)" 06:00:00 hal-det "$JOBID" \
            "json:$HD_REPO/results/cross_model/cross_${M}_to_${FALCON}_${D}_RF.json" \
            66_cross_model.py --source "$M" --target "$FALCON" --dataset "$D"
    done; done
fi

if wants falcon_actvit; then
    echo "[falcon_actvit] 59 extraction (GPU, hal-det-h1) -> ACT-ViT (GPU)"
    for D in $SMALL; do
        queue "rb-extf-$(short $D)" 1 80G 03:00:00 hal-det-h1 "" \
            "file:$AT/$FALCON/${D}_at_L8_N20.npz" \
            59_extract_act_tensors.py --dataset "$D" --model_folder "$FALCON"
        queue "rb-avitf-$(short $D)" 1 64G 08:00:00 hal-det "$JOBID" \
            "file:$MR/act_vit_${FALCON}_${D}.json" \
            56_run_method.py --method act_vit --dataset "$D" --model_folder "$FALCON"
    done
fi

# Last, because it is the largest request (100 GB of host RAM on a GPU node, ~61 GB of new disk): if
# sbatch refuses it, everything above is already queued. NO_TRIVIA=1 skips it.
if wants storage && [ "${NO_TRIVIA:-0}" != "1" ]; then
    echo "[storage, TriviaQA] the cell fgja named: re-pool ~20-40 min, then ACT-ViT ~4-8 h per model"
    for M in $MODELS; do
        storage_n5 "$M" triviaqa 96G 100G 20:00:00
    done
fi

echo
echo "Queued $N job(s).  Watch: squeue -u \$USER     Logs: slurm_logs/rb-*_<jobid>.out"
echo "Summary of everything finished so far (login node):"
echo "  module load anaconda/anaconda-2024.10 && conda activate hal-det && python 90_rebuttal_summary.py && conda deactivate"
