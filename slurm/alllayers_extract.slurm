#!/bin/bash
#SBATCH --job-name=alll-ext
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# All-layer pooled features for the layer-window ablation (T-012), via 57_extract_all_layers.py.
#
# usage (via submit_alllayers.sh, not directly):
#   sbatch slurm/alllayers_extract.slurm <model_folder> <dataset> <streams> [max_gb]
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

MODEL="${1:?usage: alllayers_extract.slurm <model_folder> <dataset> <streams> [max_gb]}"
DS="${2:?usage: alllayers_extract.slurm <model_folder> <dataset> <streams> [max_gb]}"
STREAMS="${3:?usage: alllayers_extract.slurm <model_folder> <dataset> <streams> [max_gb]}"
MAXGB="${4:-32}"

REPO="${HD_REPO:?set HD_REPO to your clone of hallucination-detection}"

export PYTHONUNBUFFERED=1
set -x

echo "$(date +%F_%T): all-layer extract ${MODEL}/${DS}  streams=${STREAMS}  max_gb=${MAXGB}" >&2

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

python 57_extract_all_layers.py \
    --dataset "$DS" \
    --model_folder "$MODEL" \
    --streams "$STREAMS" \
    --max-gb "$MAXGB" || exit 1

echo "$(date +%F_%T): done. Output under ../data-alllayers/${MODEL}/" >&2
