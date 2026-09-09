#!/bin/bash
#SBATCH --job-name=anly
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Generic stage for the CPU-only analysis scripts. One file rather than one per script, because
# they all need the same four things and differ only in their arguments.
#
#   sbatch --job-name=sweep-tydi slurm/analysis_stage.slurm \
#          58_layer_sweep.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct
#
#   sbatch --job-name=spec-trut slurm/analysis_stage.slurm \
#          60_class_spectrum.py --dataset truthfulqa --model_folder qwen-2.5-7b-instruct --paired
#
# NO GPU. These read ../data-alllayers and run numpy and sklearn. Asking for a GPU only lengthens
# the queue. --cpus-per-task matters because the work is BLAS-bound.
#
# STRIP INHERITED CONDA STATE BEFORE `module load`. sbatch exports the submitting shell's
# environment, so submitting from a "(hal-det)" prompt hands the job a conda stack it cannot unwind
# and Lmod retries `conda deactivate` until the wall clock kills it. That cost four jobs on
# 2026-09-08 and is intermittent, so a script appearing to work is not evidence that it is safe.
#
# stdout is buffered and discarded when a job is killed on TIMEOUT, hence PYTHONUNBUFFERED. 58 and
# 60 both write their results incrementally, so a job that dies keeps what it had -- rerun with the
# script's own --resume where it has one.

set -u

SCRIPT="${1:?usage: analysis_stage.slurm <script.py> [args...]}"
shift

REPO="${HD_REPO:?set HD_REPO to your clone of hallucination-detection}"

export PYTHONUNBUFFERED=1
echo "$(date +%F_%H-%M-%S): $SCRIPT $*"

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
[ -f "$SCRIPT" ] || { echo "FATAL: $SCRIPT not found in $REPO" >&2; exit 1; }

python "$SCRIPT" "$@" || exit 1

echo "$(date +%F_%H-%M-%S): done $SCRIPT"
