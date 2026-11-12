#!/bin/bash
#SBATCH --job-name=rebut
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# One job of the ICLR rebuttal batch. Queued by submit_rebuttal.sh, which sets the partition, GPU,
# memory, time and the environment; do not sbatch this directly.
#
#   sbatch ... --export=ALL,HD_REPO=<repo>,ENV_NAME=<env> slurm/rebuttal_stage.slurm <script.py> [args...]
#
# ENV_NAME is hal-det for everything except jobs where Falcon-H1 runs a forward pass (it reads answers
# or has its activations extracted): those use hal-det-h1, whose Mamba kernels need the gcc module
# recorded in $CONDA_PREFIX/.h1_modules (as slurm/pipe_stage.slurm does).
#
# Same preamble rules as every other .slurm here (AGENTS.md, CLUSTER): inherited CONDA_* stripped
# before `module load`, `set -u` not `set -e`, `|| exit 1` on real commands, unbuffered output.

set -u

SCRIPT="${1:?usage: rebuttal_stage.slurm <script.py> [args...]}"
shift

REPO="${HD_REPO:?set HD_REPO to your clone of hallucination-detection}"
ENV_NAME="${ENV_NAME:-hal-det}"

export PYTHONUNBUFFERED=1
echo "$(date +%F_%T): $SCRIPT $*"

for _v in $(compgen -v | grep '^CONDA' || true); do unset "$_v"; done
unset _CE_M _CE_CONDA || true

module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0
conda activate "$ENV_NAME"
if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
    echo "FATAL: conda env is '${CONDA_DEFAULT_ENV:-<none>}', expected $ENV_NAME." >&2
    exit 1
fi
if [ -s "$CONDA_PREFIX/.h1_modules" ]; then
    module load $(cat "$CONDA_PREFIX/.h1_modules") || exit 1
fi

export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
# Every model is already in the HF cache; offline makes a missing one fail in seconds instead of
# stalling on a 15 GB download. ALLOW_DOWNLOAD=1 at submit time lifts it.
if [ "${ALLOW_DOWNLOAD:-0}" = "1" ]; then export HF_HUB_OFFLINE=0; else export HF_HUB_OFFLINE=1; fi
mkdir -p "$MPLCONFIGDIR"

cd "$REPO" || { echo "FATAL: cannot cd to $REPO" >&2; exit 1; }
echo "repo HEAD: $(git -C "$REPO" log --oneline -1 2>/dev/null) | env $ENV_NAME | host $(hostname)"
[ -f "$SCRIPT" ] || { echo "FATAL: $SCRIPT not found in $REPO" >&2; exit 1; }
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

python "$SCRIPT" "$@" || exit 1

echo "$(date +%F_%T): done $SCRIPT"
