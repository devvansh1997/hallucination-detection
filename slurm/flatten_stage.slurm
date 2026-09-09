#!/bin/bash
#SBATCH --job-name=flat
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# 61_flatten_control.py: does the multilinear restriction earn its place at matched width?
#
# usage (via submit_flatten.sh, not directly):
#   sbatch slurm/flatten_stage.slurm <model_folder> <dataset> <readout>
#
# CPU ONLY. No --gres=gpu here: 61 is sklearn and numpy, and asking for a GPU only lengthens the
# queue. The heavy step is a randomized SVD of an (n_train x L*F) matrix, which is BLAS-bound, so
# --cpus-per-task matters and the GPU does not.
#
# STRIP INHERITED CONDA STATE BEFORE `module load`. sbatch exports the submitting shell's
# environment, so submitting from a "(hal-det)" prompt hands the job a conda stack it cannot unwind:
# Lmod's anaconda module retries `conda deactivate` forever until the wall clock kills it. That cost
# four jobs on 2026-09-08. It is intermittent -- it depends only on whether the submitter had an env
# active -- so this file appearing to work is not evidence that it is safe.
#
# stdout is BUFFERED and discarded if a job is killed on TIMEOUT, hence PYTHONUNBUFFERED.

set -u

MODEL="${1:?usage: flatten_stage.slurm <model_folder> <dataset> <readout>}"
DS="${2:?dataset required}"
READOUT="${3:-RF}"

REPO="${HD_REPO:?set HD_REPO to your clone of hallucination-detection}"
DATA="${HD_DATA:-}"

export PYTHONUNBUFFERED=1
echo "$(date +%F_%H-%M-%S): flatten control  $MODEL/$DS  readout=$READOUT"

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
mkdir -p "$MPLCONFIGDIR"

cd "$REPO" || { echo "FATAL: cannot cd to $REPO" >&2; exit 1; }
echo "repo HEAD: $(git -C "$REPO" log --oneline -1 2>/dev/null)"

# Skip guard. Re-running the driver after a partial failure redoes only what is missing.
OUT="${REPO}/results/flatten_control/flatten_${MODEL}_${DS}_${READOUT}.json"
if [ "${FORCE:-0}" != "1" ] && [ -f "$OUT" ]; then
    if python - "$OUT" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("  existing result unreadable (%s) -- recomputing" % e); sys.exit(1)
arms = ("hosvd", "flat_pca", "layer_mean", "random_proj")
if not all(d.get("summary", {}).get(a, {}).get("per_seed") for a in arms):
    print("  existing result is missing an arm -- recomputing"); sys.exit(1)
print("  verified: %s / %s / %s already complete" % (d["dataset"], d["condition"], d["readout"]))
sys.exit(0)
PYEOF
    then echo "SKIP: already complete -> $OUT"; exit 0; fi
fi

ARGS="--dataset $DS --model_folder $MODEL --readout $READOUT"
[ -n "$DATA" ] && ARGS="$ARGS --data-dir $DATA"

python 61_flatten_control.py $ARGS || exit 1

echo "$(date +%F_%H-%M-%S): done $MODEL/$DS $READOUT"
