#!/bin/bash
#SBATCH --job-name=hgreb
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x_%j/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j/%x_%j.err
#
# usage (via submit_hgreb.sh, not directly):
#   sbatch hgreb_stage.slurm <stage> <model_folder> <dataset> [tag]
#     stage = nucleus_gen | score
#
# HalluGuard as its AUTHORS describe it in the OpenReview rebuttal: no gradients, no
# back-propagation. K = H H^T / d over the final-layer states of the K sampled trajectories,
# then slogdet / svdvals / eigvalsh on a 10x10 matrix. Forward passes only, so this is cheap --
# their own runtime table puts HalluGuard level with Inside, a forward-only method.
#
# TWO STAGES, AND THE FIRST ONE IS THE POINT.
#   nucleus_gen  Regenerate with plain nucleus sampling (num_beams=1, top-p 0.95, top-k 10),
#                matching their Appendix C.1, into a SEPARATE directory. Our pinned data uses
#                sampled beam search, and this method measures how spread out the trajectories
#                are -- beam search makes them alike by construction. Scoring beam-search data
#                measures our decoder, not their method.
#   score        Run 55 on whichever generations are pointed at.
#
# The pinned beam-search data is never written to: 39 now refuses decoding overrides unless
# --output-dir is given, precisely so a stray flag cannot overwrite what every other result
# depends on.

set -u
STAGE="${1:?usage: hgreb_stage.slurm <stage> <model> <dataset> [tag]}"
MODEL="${2:?model_folder required}"
DS="${3:?dataset required}"
TAG="${4:-}"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPO=/home/de807845/Hallucination-Detection/hallucination-detection
NUCLEUS_DIR=/home/de807845/Hallucination-Detection/data-nucleus

echo "${TIMESTAMP}: ${STAGE} ${MODEL}/${DS} ${TAG}"
module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0
conda activate hal-det

export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=1800
# BLEURT-20 is fetched from Google Cloud Storage, not the HF hub, so HF_HUB_DOWNLOAD_TIMEOUT does
# not cover it -- this one does. Compute-node egress is ~2 MB/s and the archive is 2.14 GB, so the
# default 300s fsspec timeout cannot possibly complete it if a download is ever actually needed.
export HF_DATASETS_DOWNLOAD_TIMEOUT=1800
mkdir -p "$MPLCONFIGDIR"

cd "$REPO"
echo "repo HEAD: $(git -C $REPO log --oneline -1 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

case "$STAGE" in

  nucleus_gen)
    OUT="${NUCLEUS_DIR}/${MODEL}"
    SEQ="${OUT}/${DS}_sequences_v1.pt"
    if [ "${FORCE:-0}" != "1" ] && [ -f "$SEQ" ]; then
        if python - "$SEQ" <<'PYEOF'
import sys, torch
try:
    d = torch.load(sys.argv[1], weights_only=False)
except Exception as e:
    print("  existing sequences unreadable (%s) -- regenerating" % e); sys.exit(1)
need = ("input_ids", "prompt_len", "prompt_id", "all_hallucination_flag", "all_is_known")
missing = [k for k in need if k not in d]
if missing:
    print("  incomplete (missing %s) -- regenerating" % missing); sys.exit(1)
nb = (d.get("decoding_config") or {}).get("num_beams")
if nb != 1:
    print("  WRONG DECODING: num_beams=%s, expected 1 -- regenerating" % nb); sys.exit(1)
print("  verified: %d beams, num_beams=1" % len(d["input_ids"]))
sys.exit(0)
PYEOF
        then echo "SKIP: nucleus generations already present -> $SEQ"; exit 0; fi
    fi
    mkdir -p "$OUT"
    # Appendix C.1: nucleus sampling, temperature 0.5, top-p 0.95, top-k 10, K=10 candidates.
    # Temperature already matches config.yaml, so it is not overridden.
    python 39_generate_dataset.py --dataset "$DS" --model_folder "$MODEL" \
        --output-dir "$OUT" --num-beams 1 --top-p 0.95 --top-k 10 || exit 1
    ;;

  score)
    if [ "$TAG" = "nucleus" ]; then DATA="$NUCLEUS_DIR"; else DATA=""; fi
    ARGS="--dataset $DS --model_folder $MODEL --dtype bfloat16 --log-every 200"
    [ -n "$DATA" ] && ARGS="$ARGS --data-dir $DATA --tag nucleus"
    # The two source documents disagree on the layer (rebuttal: final; Appendix C.1: middle) and on
    # pooling (rebuttal: "hidden states"; C.1: "final token"). We run the rebuttal setting as the
    # headline and the appendix setting as the ablation, rather than picking one and hoping.
    echo ""; echo "### rebuttal spec: final layer, mean pooling, no ridge"
    python 55_halluguard_rebuttal.py $ARGS --layer final --pool mean --ridge 0 || exit 1
    echo ""; echo "### ablation: final layer, LAST-token pooling (Appendix C.1 wording)"
    python 55_halluguard_rebuttal.py $ARGS --layer final --pool last --ridge 0 \
        --tag "${TAG:+${TAG}_}last" || exit 1
    echo ""; echo "### ablation: MIDDLE layer (Appendix C.1 wording)"
    python 55_halluguard_rebuttal.py $ARGS --layer middle --pool mean --ridge 0 \
        --tag "${TAG:+${TAG}_}mid" || exit 1
    ;;

  *) echo "unknown stage: $STAGE"; exit 1 ;;
esac

conda deactivate
echo "${TIMESTAMP}: done ${STAGE} ${MODEL}/${DS}"
