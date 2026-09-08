#!/bin/bash
#SBATCH --job-name=act-ext
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Pooled Activation Tensors in ACT-ViT's format, from our pinned generations.
#
# usage (via submit_act.sh, not directly):
#   sbatch slurm/act_extract.slurm <model_folder> <dataset> [n_pool] [max_gb]
#
# WHY THIS FILE IS FULL OF GUARDS. Jobs 798664/798665 burned their entire 40 and 70 minute walls
# and left an EMPTY .out, because SLURM buffers stdout and discards it when a job is killed on
# TIMEOUT. Only stderr survived, showing repeated "CondaError: Run 'conda init' before 'conda
# deactivate'". Three fixes, in order of what actually goes wrong:
#
#   1. --export=ALL is passed by submit_act.sh. submit_hgreb.sh, which works, passes it on every
#      job; this file's driver did not. Without it the module-system shell functions and the HF
#      cache variables do not reach the job.
#   2. Everything is verified BEFORE the GPU does any work, and each check exits non-zero with a
#      message. A wrong environment is now a five-second failure, not a forty-minute one.
#   3. PYTHONUNBUFFERED and `set -x` on stderr, so a killed job still leaves a trail. stdout can be
#      lost; stderr cannot.
#
# The model-cache check matters most. Compute-node egress is ~2 MB/s and Qwen2.5-7B is ~15 GB, so a
# cache miss is a silent multi-hour stall that looks exactly like a hang. Same failure mode as the
# BLEURT-20 incident; this refuses instead.
#
# THE SELF-TEST IS NOT RUN HERE. It asserts bit-identity against ../ACT-ViT, which is a development
# check needing a clone that may not exist on the cluster. It does not belong between a GPU
# allocation and its work. Run it on the login node: python 59_extract_act_tensors.py --self-test
#
# SHELL OPTIONS: `set -u`, deliberately NOT `set -euo pipefail`. Under `set -e` one failing line in
# the module preamble kills the job in under a second, so sbatch returns a job id for a job that is
# already dead and never appears in squeue -- the first version of this file did exactly that.

set -u

MODEL="${1:?usage: act_extract.slurm <model_folder> <dataset> [n_pool] [max_gb]}"
DS="${2:?usage: act_extract.slurm <model_folder> <dataset> [n_pool] [max_gb]}"
NPOOL="${3:-100}"
MAXGB="${4:-60}"

REPO="${HD_REPO:-/home/de807845/Hallucination-Detection/hallucination-detection}"

# stderr is the only stream guaranteed to survive a TIMEOUT kill, so send the trace there.
export PYTHONUNBUFFERED=1
set -x

echo "$(date +%F_%T): act extract ${MODEL}/${DS}  pool=(8,${NPOOL})  max_gb=${MAXGB}" >&2

module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0
conda activate hal-det

# -- verify, loudly and early ------------------------------------------------------------------
if [ "${CONDA_DEFAULT_ENV:-}" != "hal-det" ]; then
    echo "FATAL: conda env is '${CONDA_DEFAULT_ENV:-<none>}', expected hal-det." >&2
    echo "  Was the job submitted with --export=ALL? Without it the module functions do not" >&2
    echo "  reach the job and conda activate silently fails." >&2
    exit 1
fi
cd "$REPO" || { echo "FATAL: cannot cd to $REPO -- set HD_REPO" >&2; exit 1; }

python - <<'PYCHK' >&2 || exit 1
import sys
import numpy, torch, transformers, yaml           # noqa: F401
print("  python      : %s" % sys.executable)
print("  torch %s  cuda available: %s" % (torch.__version__, torch.cuda.is_available()))
PYCHK

# A cache miss here means a ~15 GB download at ~2 MB/s. Refuse rather than stall.
HF_ROOT="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_ID=$(python -c "
import yaml
c = yaml.safe_load(open('config.yaml'))
print(next(m['id'] for m in c['models'] if m['folder'] == '$MODEL'))
")
CACHE_DIR="$HF_ROOT/hub/models--$(echo "$MODEL_ID" | tr '/' '-')"
if [ ! -d "$CACHE_DIR" ]; then
    echo "FATAL: $MODEL_ID is not in the HF cache at $CACHE_DIR." >&2
    echo "  Downloading it here would run at ~2 MB/s on a compute node and stall the job for" >&2
    echo "  hours. Pre-fetch it on a login node, or check that HF_HOME reached the job." >&2
    exit 1
fi
echo "  model cache : $CACHE_DIR" >&2

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >&2
echo "  repo HEAD   : $(git -C "$REPO" log --oneline -1 2>/dev/null)" >&2
set +x

# -- the work ------------------------------------------------------------------------------------
python 59_extract_act_tensors.py \
    --dataset "$DS" \
    --model_folder "$MODEL" \
    --n-pool "$NPOOL" \
    --max-gb "$MAXGB" || exit 1

echo "done. Output: ../data-acttensors/${MODEL}/${DS}_at_L8_N${NPOOL}.npz"
