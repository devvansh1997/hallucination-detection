#!/bin/bash
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x_%j/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j/%x_%j.err
#
# usage (normally via submit_methods.sh):
#   sbatch method_stage.slurm <method> <model_folder> <dataset> [tag] [extra args...]
#
# One stage for every detection method. 56_run_method.py owns the data, both split protocols, the
# AUROC and the output schema, so this script does nothing but set the environment and hand over.
#
# PATHS come from the environment, not from this file. Set them once in your shell or in
# submit_methods.sh:
#     export HD_REPO=$HOME/Hallucination-Detection/hallucination-detection
#     export HD_DATA=/home/de807845/Hallucination-Detection/data
# HD_DATA can point at somebody else's directory -- the harness only reads it. That is how a second
# person runs methods against the pinned generations without copying 91 GB.

set -u
METHOD="${1:?usage: method_stage.slurm <method> <model_folder> <dataset> [tag] [extra args...]}"
MODEL="${2:?model_folder required}"
DS="${3:?dataset required}"
TAG="${4:-}"
shift 4 2>/dev/null || shift $#
EXTRA="$*"

REPO="${HD_REPO:?set HD_REPO to your clone of hallucination-detection}"
DATA="${HD_DATA:-}"

echo "$(date +%F_%H-%M-%S): method=$METHOD model=$MODEL dataset=$DS tag=${TAG:-none}"

# STRIP INHERITED CONDA STATE BEFORE `module load`. sbatch exports the submitting shell's
# environment, so submitting from a "(hal-det)" prompt hands the job a conda stack it cannot
# unwind: Lmod's anaconda module retries `conda deactivate` forever, writing thousands of
# "CondaError: Run 'conda init' before 'conda deactivate'" until the wall clock kills it. That
# cost four extraction jobs on 2026-09-08 and was misdiagnosed twice. It is INTERMITTENT -- it
# depends only on whether the submitter had an env active -- so this file appearing to work is
# not evidence that it is safe.
for _v in $(compgen -v | grep '^CONDA' || true); do unset "$_v"; done
unset _CE_M _CE_CONDA || true

module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0
conda activate hal-det

if [ "${CONDA_DEFAULT_ENV:-}" != "hal-det" ]; then
    echo "FATAL: conda env is '${CONDA_DEFAULT_ENV:-<none>}', expected hal-det." >&2
    exit 1
fi

export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT=1800
export HF_DATASETS_DOWNLOAD_TIMEOUT=1800
mkdir -p "$MPLCONFIGDIR"

cd "$REPO"
echo "repo HEAD: $(git -C "$REPO" log --oneline -1 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU)"

ARGS="--method $METHOD --dataset $DS --model_folder $MODEL"
[ -n "$DATA" ] && ARGS="$ARGS --data-dir $DATA"
[ -n "$TAG" ] && ARGS="$ARGS --tag $TAG"

# Skip guard. Re-running the driver after a partial failure redoes only what is missing.
OUT="${REPO}/results/methods/${METHOD}_${MODEL}_${DS}${TAG:+_${TAG}}.json"
if [ "${FORCE:-0}" != "1" ] && [ -f "$OUT" ]; then
    if python - "$OUT" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("  existing result unreadable (%s) -- recomputing" % e); sys.exit(1)
if not d.get("all_rows") or not d.get("protocols"):
    print("  existing result incomplete -- recomputing"); sys.exit(1)
if not all(d["protocols"].get(a, {}).get("per_seed") for a in ("question", "answer")):
    print("  existing result has an empty protocol arm -- recomputing"); sys.exit(1)
print("  verified: %s on %d rows already scored" % (d.get("method"), d.get("n_beams", 0)))
sys.exit(0)
PYEOF
    then echo "SKIP: already complete -> $OUT"; exit 0; fi
fi

python 56_run_method.py $ARGS $EXTRA || exit 1

conda deactivate
echo "$(date +%F_%H-%M-%S): done $METHOD $MODEL/$DS"
