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
# The model-cache guard matters most, and it is HF_HUB_OFFLINE=1 rather than a path check. Egress on
# a compute node is ~2 MB/s and Qwen2.5-7B is ~15 GB, so a cache miss is a multi-hour stall that
# looks exactly like a hang (the BLEURT-20 failure mode). Offline mode makes that impossible without
# guessing where the cache lives -- an earlier version guessed with `tr '/' '-'`, missed HF's double
# hyphen, and refused two jobs whose model was cached all along. Set ALLOW_DOWNLOAD=1 to permit a
# fetch deliberately.
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

# A cache miss means a ~15 GB download at ~2 MB/s compute-node egress: a multi-hour stall that
# looks exactly like a hang. HF_HUB_OFFLINE makes that structurally impossible -- from_pretrained
# raises immediately instead of fetching -- with no path to guess.
#
# The previous version of this guard DID guess, building the cache path with `tr '/' '-'`. HF uses a
# DOUBLE hyphen for the separator (models--Qwen--Qwen2.5-7B-Instruct), so it looked for a directory
# that never exists and refused two jobs whose model was cached all along. Deleting the guess is the
# fix; asking the library is the guard.
export HF_HUB_OFFLINE="${ALLOW_DOWNLOAD:+0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

python - "$MODEL" <<'PYCACHE' >&2 || exit 1
import os
import sys
import yaml
cfg = yaml.safe_load(open("config.yaml"))
model_id = next(m["id"] for m in cfg["models"] if m["folder"] == sys.argv[1])
print("  model id    : %s" % model_id)
print("  HF_HOME     : %s" % os.environ.get("HF_HOME", "<unset, using ~/.cache/huggingface>"))
print("  offline     : HF_HUB_OFFLINE=%s" % os.environ.get("HF_HUB_OFFLINE"))
try:
    from huggingface_hub import snapshot_download
    print("  resolved to : %s" % snapshot_download(model_id, local_files_only=True))
except Exception as e:
    print("FATAL: %s is not fully cached (%s: %s)." % (model_id, type(e).__name__, e))
    print("  Pre-fetch it on a login node, or re-run with ALLOW_DOWNLOAD=1 if a download here")
    print("  really is intended -- expect hours at ~2 MB/s.")
    sys.exit(1)
PYCACHE

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
