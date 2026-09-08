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
# The download guard is HF_HUB_OFFLINE=1 and nothing else. Egress on a compute node is ~2 MB/s and
# Qwen2.5-7B is ~15 GB, so a fetch here is a multi-hour stall that looks exactly like a hang (the
# BLEURT-20 failure mode). Offline mode makes it impossible; 59's own from_pretrained is then the
# only check needed. Two richer pre-checks were tried and both refused jobs whose model was fine --
# see the note at that block. ALLOW_DOWNLOAD=1 permits a fetch deliberately.
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

# NO DOWNLOAD CAN HAPPEN, so no stall can happen. That is the whole guard, and HF_HUB_OFFLINE is
# the whole implementation: from_pretrained raises in seconds instead of fetching ~15 GB at the
# ~2 MB/s a compute node gets. ALLOW_DOWNLOAD=1 permits a fetch deliberately.
#
# TWO PREVIOUS ATTEMPTS AT A SEPARATE PRE-CHECK BOTH BLOCKED JOBS WHOSE MODEL WAS FINE:
#   * building the cache path with `tr '/' '-'` -- HF uses a DOUBLE hyphen
#     (models--Qwen--Qwen2.5-7B-Instruct), so it looked for a directory that cannot exist;
#   * snapshot_download(local_files_only=True) -- which demands the ENTIRE repo and failed on
#     .gitattributes, LICENSE and README.md, none of which from_pretrained ever reads.
# The lesson is that a pre-check strictly stronger than the operation it guards is not a guard, it
# is a second thing to get wrong. 59 loading the model IS the check.
export HF_HUB_OFFLINE="${ALLOW_DOWNLOAD:+0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
echo "  HF_HOME     : ${HF_HOME:-<unset, using ~/.cache/huggingface>}" >&2
echo "  offline     : HF_HUB_OFFLINE=$HF_HUB_OFFLINE (ALLOW_DOWNLOAD=1 to permit a fetch)" >&2

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
