#!/bin/bash
#SBATCH --job-name=xread
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# One model reads the other model's answers (cross-model transfer, T-013), via 65_extract_cross_read.py.
#
# usage (via submit_crossmodel.sh, not directly):
#   sbatch slurm/crossread_extract.slurm <reader> <generator> <dataset>
#
# The preamble is act_extract.slurm's, for the same reasons -- read that file for the history:
#   * inherited CONDA_* variables are stripped before `module load`, or Lmod loops on
#     `conda deactivate` until the wall clock kills the job;
#   * the environment is verified before the GPU does any work;
#   * HF_HUB_OFFLINE=1, so a missing model fails in seconds instead of stalling on a ~15 GB fetch;
#   * PYTHONUNBUFFERED and `set -x` to stderr, because .out is discarded on a TIMEOUT kill.
#
# `set -u`, deliberately not `set -euo pipefail`: under -e one failing line in the module preamble
# kills the job in under a second and it never shows in squeue.

set -u

READER="${1:?usage: crossread_extract.slurm <reader> <generator> <dataset>}"
GENERATOR="${2:?usage: crossread_extract.slurm <reader> <generator> <dataset>}"
DS="${3:?usage: crossread_extract.slurm <reader> <generator> <dataset>}"

REPO="${HD_REPO:?set HD_REPO to your clone of hallucination-detection}"

export PYTHONUNBUFFERED=1
set -x

echo "$(date +%F_%T): cross read ${READER} reads ${GENERATOR} / ${DS}" >&2

for _v in $(compgen -v | grep '^CONDA' || true); do unset "$_v"; done
unset _CE_M _CE_CONDA || true

module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0
conda activate hal-det

if [ "${CONDA_DEFAULT_ENV:-}" != "hal-det" ]; then
    echo "FATAL: conda env is '${CONDA_DEFAULT_ENV:-<none>}', expected hal-det." >&2
    exit 1
fi
cd "$REPO" || { echo "FATAL: cannot cd to $REPO -- set HD_REPO" >&2; exit 1; }

python - <<'PYCHK' >&2 || exit 1
import sys
import numpy, torch, transformers, yaml           # noqa: F401
print("  python      : %s" % sys.executable)
print("  torch %s  cuda available: %s" % (torch.__version__, torch.cuda.is_available()))
assert torch.cuda.is_available(), "no GPU visible to torch"
PYCHK

export HF_HUB_OFFLINE="${ALLOW_DOWNLOAD:+0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
echo "  HF_HOME     : ${HF_HOME:-<unset, using ~/.cache/huggingface>}" >&2
echo "  offline     : HF_HUB_OFFLINE=$HF_HUB_OFFLINE (ALLOW_DOWNLOAD=1 to permit a fetch)" >&2

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >&2
echo "  repo HEAD   : $(git -C "$REPO" log --oneline -1 2>/dev/null)" >&2
set +x

python 65_extract_cross_read.py \
    --reader "$READER" \
    --generator "$GENERATOR" \
    --dataset "$DS" || exit 1

echo "$(date +%F_%T): done. Output under ../data-crossread/${READER}_reads_${GENERATOR}/" >&2
