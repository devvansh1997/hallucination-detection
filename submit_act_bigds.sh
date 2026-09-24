#!/bin/bash
# submit_act_bigds.sh -- the two missing ACT-ViT columns of Table 2: NQ-Open, and TriviaQA if we decide to.
#
#   bash submit_act_bigds.sh                  # NQ-Open, both models: extract then train, chained
#   DATASETS=triviaqa bash submit_act_bigds.sh
#   DRY=1 bash submit_act_bigds.sh            # print what would be queued and stop
#
# RUN FROM A NEWTON TERMINAL. Four GPU jobs: one extraction and one training run per model, the
# training held on afterok so a failed extraction does not start a run that would die reading a
# half-written npz.
#
# WHY N_eff = 20 AND NOT 100. At their released N=100 a beam is 5.5 MiB and NQ-Open is 194 GB per
# model, TriviaQA 532 GB -- that is what the cost appendix says makes these columns unaffordable,
# and it stays true. N=20 is the setting we already report for TruthfulQA and TyDiQA-GP and sits
# inside their own published ablation grid, so these cells are comparable with the two we have.
# 59 pools inside the forward loop, so extracting at 20 directly never materializes the N=100 array.
#
# SIZES (storage measured from D and the beam counts; times scaled from measured runs):
#
#   dataset     beams    N=20, per model   extract    train (5 seeds x 2 protocols)
#   nq_open     36,100   38.6 / 44.1 GB    ~55 min    ~1.5 h
#   triviaqa    99,600   106  / 122  GB    ~2.5 h     ~4 h
#
# Extraction scales from job 798638/798639 (4,400 beams in 11:16, 8,170 in 12:22, both including
# model load); training from truthfulqa at N=20, 1,111 s. Walls below carry roughly 2x.
#
# HOST RAM IS THE REAL CONSTRAINT ON TRIVIAQA, NOT THE GPU. 59 holds the whole output array while it
# writes, and methods/act_vit.py loads the whole npz before it trains, so both sides need RAM above
# the file size: ~110G for NQ-Open, ~200G for TriviaQA. If highgpu has no node that large, TriviaQA
# needs a smaller N_eff, not a longer wall.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="${MODELS:-qwen-2.5-7b-instruct llama-3.1-8b}"
DATASETS="${DATASETS:-nq_open}"
PART="${PART:-highgpu}"
NPOOL="${NPOOL:-20}"
DRY="${DRY:-}"

mkdir -p "$HERE/slurm_logs"

# Sized for N_eff=20, not for the N=100 defaults in submit_act.sh / submit_methods.sh.
mem_for()  { case "$1" in nq_open) echo "110G";; triviaqa) echo "200G";; *) echo "110G";; esac; }
maxgb_for() { case "$1" in nq_open) echo "60";; triviaqa) echo "150";; *) echo "60";; esac; }
ext_time() { case "$1" in nq_open) echo "02:00:00";; triviaqa) echo "05:00:00";; *) echo "02:00:00";; esac; }
trn_time() { case "$1" in nq_open) echo "06:00:00";; triviaqa) echo "14:00:00";; *) echo "06:00:00";; esac; }

# Disk, before anything is queued: 83 GB for NQ-Open across both models, 228 GB for TriviaQA, on a
# filesystem that also holds Falcon's TriviaQA artifacts. A job that fills the quota half way
# through an extraction leaves a truncated npz that the training side will happily try to read.
echo "filesystem:"
df -h "$HOME" | tail -n +1
if [ -d "$HOME/Hallucination-Detection/data-acttensors" ]; then
    echo "existing activation tensors: $(du -sh "$HOME/Hallucination-Detection/data-acttensors" | cut -f1)"
fi
echo

for DS in $DATASETS; do
    for M in $MODELS; do
        EXT_OPTS="-p $PART --gres=gpu:1 --mem=$(mem_for "$DS") --time=$(ext_time "$DS") \
                  --job-name=actext-${DS:0:6}-${M:0:4} --export=ALL"
        TRN_OPTS="-p $PART --gres=gpu:1 --mem=$(mem_for "$DS") --time=$(trn_time "$DS") \
                  --job-name=actrun-${DS:0:6}-${M:0:4} --export=ALL"
        if [ -n "$DRY" ]; then
            echo "would queue: act_extract.slurm $M $DS $NPOOL $(maxgb_for "$DS")   [$(mem_for "$DS"), $(ext_time "$DS")]"
            echo "would queue: method_stage.slurm act_vit $M $DS  (afterok)         [$(mem_for "$DS"), $(trn_time "$DS")]"
            continue
        fi
        EXT=$(sbatch --parsable $EXT_OPTS "$HERE/slurm/act_extract.slurm" \
                     "$M" "$DS" "$NPOOL" "$(maxgb_for "$DS")")
        TRN=$(sbatch --parsable --dependency=afterok:"$EXT" $TRN_OPTS \
                     "$HERE/slurm/method_stage.slurm" act_vit "$M" "$DS")
        printf "  %-24s %-9s extract %s -> train %s\n" "$M" "$DS" "$EXT" "$TRN"
    done
done

echo
echo "watch:   squeue -u \$USER -o '%.10i %.22j %.8T %.10M %R'"
echo "results: results/methods/act_vit_<model>_<dataset>.json  ->  protocols.question.pooled_auroc_mean"
