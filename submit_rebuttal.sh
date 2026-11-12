#!/bin/bash
# submit_rebuttal.sh -- the ICLR 2027 rebuttal experiments (submission 42795), in three waves.
#
#   DRY=1 bash submit_rebuttal.sh 1        # print wave 1, queue nothing
#   bash submit_rebuttal.sh 1              # wave 1: CPU analyses + cost profile          14 jobs  ~1-1.5 h
#   bash submit_rebuttal.sh 2              # wave 2: ACT-ViT fewer labels + every input
#                                          #         wave 3 needs                         24 jobs  ~2 h
#   bash submit_rebuttal.sh 3              # wave 3: everything that uses wave 2's files  20 jobs  ~1-2 h
#                                          #         (the two TriviaQA ACT-ViT runs: 4-8 h)
#   NO_TRIVIA=1 bash submit_rebuttal.sh 2  # leave out the TriviaQA matched-storage cell (~61 GB)
#
# NO JOB WAITS ON ANOTHER. There are no Slurm dependencies anywhere: each wave queues only jobs whose
# inputs already exist, so a failure stays inside its own job and its own log. A wave-3 job whose input
# is missing (its wave-2 job failed or has not finished) is reported NOT READY, not queued. Rerunning a
# wave is safe: finished outputs are skipped and queued/running jobs are not duplicated.
#
# RUN FROM A PLAIN LOGIN-NODE SHELL after `git pull`: no conda env active, not inside srun/salloc. This
# script runs no Python; each job activates its own env (hal-det, or hal-det-h1 where Falcon-H1 runs).
#
# WAVE 1  kept       85  zjPm Q3/W3, pjMe Q1: kept subspace vs discarded residual   4 CPU  30-60 min
#         responses  86  zjPm Q2: 10 / 5 / 3 answers per question                   4 CPU  40-70 min
#         cost       89  zjPm W2, pjMe Q3: time and peak memory per stage          4 CPU + 2 GPU  10-30 min
# WAVE 2  fewer      87  pjMe Q4/W1: ACT-ViT on 5-100% of training questions       4 GPU  45 min-1.5 h
#         shrink     88  ACT-ViT tensors re-pooled to N=5 (exact, CPU)             6 CPU  5 min (TriviaQA 20-40 min)
#         ext410     59  ACT-ViT tensors at 4 layers x 10 tokens                   4 GPU  ~15 min
#         xread      65  Falcon-H1 <-> Qwen / LLaMA reading each other's answers   8 GPU  10-20 min
#         extf       59  ACT-ViT tensors for Falcon-H1                             2 GPU  ~15 min
# WAVE 3  avit5      56  fgja W2: ACT-ViT at LOTUS's storage (L8 N5)               6 GPU  30-60 min (TriviaQA 4-8 h)
#         avit410    56  fgja W2: ACT-ViT with fewer layers (L4 N10)               4 GPU  30-60 min
#         xfer       66  pjMe W2: transfer with Falcon-H1 as a third model         8 CPU  30-60 min
#         avitf      56  ACT-ViT on Falcon-H1                                      2 GPU  30-45 min
# Times assume jobs start promptly; a busy cluster can double them (TICKETS: contention).
#
# DISK (new files): wave 2 writes ~15 GB + ~61 GB for TriviaQA N=5 + ~10 GB of Falcon reads + ~12 GB
# of Falcon tensors. `lfs quota` showed 893 / 1000 GB soft, 1010 GB hard on 2026-11-12.
#
# After each wave, one printout of every result so far (login node; deactivate afterwards):
#   module load anaconda/anaconda-2024.10 && conda activate hal-det && python 90_rebuttal_summary.py; conda deactivate

set -euo pipefail

WAVE="${1:-}"
case "$WAVE" in 1|2|3) ;; *) echo "usage: bash submit_rebuttal.sh 1|2|3   (DRY=1 to print the plan)"; exit 1 ;; esac

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

QWEN=qwen-2.5-7b-instruct
LLAMA=llama-3.1-8b
FALCON=falcon-h1-7b-base
MODELS="$QWEN $LLAMA"
SMALL="truthfulqa tydiqa_gp"
AT="$PARENT/data-acttensors"
XR="$PARENT/data-crossread"
RB="$HD_REPO/results/rebuttal"
MR="$HD_REPO/results/methods"
XM="$HD_REPO/results/cross_model"

short() { case "$1" in "$QWEN") echo qwen ;; "$LLAMA") echo llam ;; "$FALCON") echo falc ;;
                       truthfulqa) echo tqa ;; tydiqa_gp) echo tydi ;; triviaqa) echo triv ;; *) echo "$1" ;; esac; }
mem_for() { if [ "$1" = truthfulqa ]; then echo "$2"; else echo "$3"; fi; }   # <ds> <tqa-mem> <other>
need() { [ -e "$1" ] || { echo "MISSING INPUT: $1"; MISSING=$((MISSING+1)); }; }

N=0; MISSING=0; NOT_READY=0
# queue <name> <gpu 0|1> <mem> <time> <env> <done-test> <input|""> <script> [args...]
#   done-test  "json:<path>"  finished when that JSON contains "complete": true
#              "file:<path>"  finished when that path exists (written only at the very end)
#   input      a file an EARLIER wave writes LAST (a .json written after its .npz, or 88s atomic
#              .npz); if absent the job is NOT queued
queue() {
    local name="$1" gpu="$2" mem="$3" tim="$4" env="$5" done_test="$6" input="$7"; shift 7
    if [ "${FORCE:-0}" != "1" ]; then
        local kind="${done_test%%:*}" path="${done_test#*:}"
        if { [ "$kind" = json ] && [ -f "$path" ] && grep -q '"complete": true' "$path"; } || \
           { [ "$kind" = file ] && [ -e "$path" ]; }; then
            printf "  %-26s SKIP (done)\n" "$name"; return 0
        fi
    fi
    if [ -n "$input" ] && [ ! -e "$input" ]; then
        printf "  %-26s NOT READY (missing %s)\n" "$name" "${input#$PARENT/}"; NOT_READY=$((NOT_READY+1)); return 0
    fi
    local running
    running="$(squeue -h -u "$USER" -n "$name" -o %i 2>/dev/null | head -1 || true)"
    if [ -n "$running" ]; then
        printf "  %-26s already queued/running (%s)\n" "$name" "$running"; return 0
    fi
    local part="$CPU_PART" gres="" kindname="CPU"
    if [ "$gpu" = "1" ]; then part="$GPU_PART"; gres="--gres=gpu:1"; kindname="GPU"; fi
    if [ "${DRY:-0}" = "1" ]; then
        printf "  %-26s would queue: %-7s %s mem %-5s %s %s\n" "$name" "$part" "$kindname" "$mem" "$tim" "$env"
        N=$((N+1)); return 0
    fi
    local out
    if ! out=$(sbatch --parsable -p "$part" $gres --mem="$mem" --cpus-per-task="$CPUS" --time="$tim" \
            --job-name="$name" ${EXCLUDE:+--exclude=$EXCLUDE} \
            --export=ALL,HD_REPO="$HD_REPO",ENV_NAME="$env" \
            "$HD_REPO/slurm/rebuttal_stage.slurm" "$@" 2>&1); then
        echo "ERROR: sbatch rejected $name:" >&2; printf '  %s\n' "$out" >&2; exit 1
    fi
    N=$((N+1))
    printf "  %-26s -> %s\n" "$name" "$out"
}

# ---- base inputs (exist before any wave) ---------------------------------------------------------
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

echo "Rebuttal wave $WAVE from $HD_REPO"

# ================================================================================================
if [ "$WAVE" = 1 ]; then
    echo "[kept] 85: kept subspace vs discarded residual   (~30-60 min per job)"
    for M in $MODELS; do for D in $SMALL; do
        queue "rb-kept-$(short $M)-$(short $D)" 0 "$(mem_for $D 96G 64G)" 08:00:00 hal-det \
            "json:$RB/kept_vs_discarded_${M}_${D}_RF.json" "" \
            85_kept_vs_discarded.py --dataset "$D" --model_folder "$M"
    done; done
    echo "[responses] 86: 10 / 5 / 3 answers per question   (~40-70 min per job)"
    for M in $MODELS; do for D in $SMALL; do
        queue "rb-resp-$(short $M)-$(short $D)" 0 "$(mem_for $D 96G 64G)" 08:00:00 hal-det \
            "json:$RB/responses_per_question_${M}_${D}_RF.json" "" \
            86_responses_per_question.py --dataset "$D" --model_folder "$M"
    done; done
    echo "[cost] 89: time and peak memory on TruthfulQA   (~10-30 min per job)"
    for M in $MODELS; do
        queue "rb-cost-fit-$(short $M)" 0 96G 03:00:00 hal-det \
            "json:$RB/cost_fit_flatpca_${M}_truthfulqa.json" "" \
            89_cost_profile.py --dataset truthfulqa --model_folder "$M" --stages fit_lotus fit_flatpca
        queue "rb-cost-gpu-$(short $M)" 1 80G 04:00:00 hal-det \
            "json:$RB/cost_act_vit_${M}_truthfulqa.json" "" \
            89_cost_profile.py --dataset truthfulqa --model_folder "$M" --stages extract act_vit
    done
    queue "rb-cost-map-l2q" 0 96G 03:00:00 hal-det \
        "json:$RB/cost_mapping_${QWEN}_truthfulqa_from_${LLAMA}.json" "" \
        89_cost_profile.py --dataset truthfulqa --model_folder "$QWEN" --source "$LLAMA" --stages mapping
    queue "rb-cost-map-q2l" 0 96G 03:00:00 hal-det \
        "json:$RB/cost_mapping_${LLAMA}_truthfulqa_from_${QWEN}.json" "" \
        89_cost_profile.py --dataset truthfulqa --model_folder "$LLAMA" --source "$QWEN" --stages mapping
fi

# ================================================================================================
if [ "$WAVE" = 2 ]; then
    echo "[fewer] 87: ACT-ViT at 5/10/25/50/100% of training questions   (TyDi ~45 min, TQA ~1.5 h)"
    for M in $MODELS; do for D in $SMALL; do
        queue "rb-fewer-$(short $M)-$(short $D)" 1 64G 12:00:00 hal-det \
            "json:$RB/actvit_fewer_labels_${M}_${D}.json" "" \
            87_actvit_fewer_labels.py --dataset "$D" --model_folder "$M"
    done; done
    STORAGE_DS="$SMALL"
    if [ "${NO_TRIVIA:-0}" != "1" ]; then STORAGE_DS="$SMALL triviaqa"; fi
    echo "[shrink] 88: exact re-pool of the stored N=20 tensors to N=5   (~5 min; TriviaQA 20-40 min)"
    for M in $MODELS; do for D in $STORAGE_DS; do
        queue "rb-shrink-$(short $M)-$(short $D)" 0 "$([ "$D" = triviaqa ] && echo 96G || echo 32G)" 03:00:00 hal-det \
            "file:$AT/$M/${D}_at_L8_N5.npz" "" \
            88_shrink_act_tensors.py --dataset "$D" --model_folder "$M" --n-to 5
    done; done
    echo "[ext410] 59: ACT-ViT tensors at 4 layer x 10 token slots   (~15 min)"
    for M in $MODELS; do for D in $SMALL; do
        queue "rb-ext410-$(short $M)-$(short $D)" 1 80G 03:00:00 hal-det \
            "file:$AT/$M/${D}_at_L4_N10.json" "" \
            59_extract_act_tensors.py --dataset "$D" --model_folder "$M" --l-pool 4 --n-pool 10
    done; done
    echo "[xread] 65: Falcon-H1 and Qwen / LLaMA read each other's answers   (~10-20 min)"
    for M in $MODELS; do for D in $SMALL; do
        queue "rb-xread-falc-$(short $M)-$(short $D)" 1 "$(mem_for $D 80G 64G)" 03:00:00 hal-det-h1 \
            "file:$XR/${FALCON}_reads_${M}/${D}_window.json" "" \
            65_extract_cross_read.py --reader "$FALCON" --generator "$M" --dataset "$D"
        queue "rb-xread-$(short $M)-falc-$(short $D)" 1 "$(mem_for $D 80G 64G)" 03:00:00 hal-det \
            "file:$XR/${M}_reads_${FALCON}/${D}_window.json" "" \
            65_extract_cross_read.py --reader "$M" --generator "$FALCON" --dataset "$D"
    done; done
    echo "[extf] 59: ACT-ViT tensors for Falcon-H1   (~15 min)"
    for D in $SMALL; do
        queue "rb-extf-$(short $D)" 1 80G 03:00:00 hal-det-h1 \
            "file:$AT/$FALCON/${D}_at_L8_N20.json" "" \
            59_extract_act_tensors.py --dataset "$D" --model_folder "$FALCON"
    done
fi

# ================================================================================================
if [ "$WAVE" = 3 ]; then
    echo "[avit410] 56: ACT-ViT with 4 layer x 10 token slots   (~30-60 min)"
    for M in $MODELS; do for D in $SMALL; do
        queue "rb-avit410-$(short $M)-$(short $D)" 1 64G 08:00:00 hal-det \
            "file:$MR/act_vit_${M}_${D}_L4N10.json" "$AT/$M/${D}_at_L4_N10.json" \
            56_run_method.py --method act_vit --dataset "$D" --model_folder "$M" --m-l-eff 4 --m-n-eff 10 --tag L4N10
    done; done
    echo "[xfer] 66: transfer with Falcon-H1 as a third model   (~30-60 min)"
    for M in $MODELS; do for D in $SMALL; do
        queue "rb-xfer-falc-$(short $M)-$(short $D)" 0 "$(mem_for $D 96G 64G)" 06:00:00 hal-det \
            "json:$XM/cross_${FALCON}_to_${M}_${D}_RF.json" "$XR/${FALCON}_reads_${M}/${D}_window.json" \
            66_cross_model.py --source "$FALCON" --target "$M" --dataset "$D"
        queue "rb-xfer-$(short $M)-falc-$(short $D)" 0 "$(mem_for $D 96G 64G)" 06:00:00 hal-det \
            "json:$XM/cross_${M}_to_${FALCON}_${D}_RF.json" "$XR/${M}_reads_${FALCON}/${D}_window.json" \
            66_cross_model.py --source "$M" --target "$FALCON" --dataset "$D"
    done; done
    echo "[avitf] 56: ACT-ViT on Falcon-H1   (~30-45 min)"
    for D in $SMALL; do
        queue "rb-avitf-$(short $D)" 1 64G 08:00:00 hal-det \
            "file:$MR/act_vit_${FALCON}_${D}.json" "$AT/$FALCON/${D}_at_L8_N20.json" \
            56_run_method.py --method act_vit --dataset "$D" --model_folder "$FALCON"
    done
    echo "[avit5] 56: ACT-ViT at LOTUS's storage, L8 N5   (~30-60 min; TriviaQA last, 4-8 h, 100 GB RAM)"
    STORAGE_DS="$SMALL"
    if [ "${NO_TRIVIA:-0}" != "1" ]; then STORAGE_DS="$SMALL triviaqa"; fi
    for D in $STORAGE_DS; do for M in $MODELS; do
        if [ "$D" = triviaqa ]; then mem=100G; tim=20:00:00; else mem=64G; tim=08:00:00; fi
        queue "rb-avit5-$(short $M)-$(short $D)" 1 "$mem" "$tim" hal-det \
            "file:$MR/act_vit_${M}_${D}_L8N5.json" "$AT/$M/${D}_at_L8_N5.npz" \
            56_run_method.py --method act_vit --dataset "$D" --model_folder "$M" --m-l-eff 8 --m-n-eff 5 --tag L8N5
    done; done
fi

echo
echo "Wave $WAVE: queued $N job(s); $NOT_READY not ready.   Watch: squeue -u \$USER   Logs: slurm_logs/rb-*_<jobid>.out"
echo "When they land:  module load anaconda/anaconda-2024.10 && conda activate hal-det && python 90_rebuttal_summary.py; conda deactivate"
