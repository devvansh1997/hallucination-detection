#!/bin/bash
#SBATCH --job-name=h1env
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# T-014: build hal-det-h1 = a clone of hal-det plus causal-conv1d and mamba-ssm compiled against its torch,
# then check Falcon-H1's fast path with 71_check_fast_path.py. hal-det itself is never modified.
#
# usage (normally via submit_h1.sh):
#   sbatch -p highgpu --gres=gpu:1 --cpus-per-task=16 --mem=80G --time=04:00:00 \
#          --export=ALL,HD_REPO=$PWD slurm/build_h1_env.slurm
#
# Knobs: SRC_ENV (hal-det), NEW_ENV (hal-det-h1), MODEL (falcon-h1-7b-base), MAX_JOBS (10 parallel nvcc),
#        OUT_DIR (results/pilot_kernels), EXTRA_MODULES (e.g. a newer gcc module if a build fails on the
#        compiler; this log lists the gcc modules available).
#
# Steps, each logged as a STEP line in .err so a failure is easy to place. A rerun resumes: an existing
# clone is reused and an importable package is not rebuilt.
#   1 network      pip needs PyPI from this node
#   2 clone        conda create --clone
#   3 toolchain    torch's CUDA and nvcc must share a major version; gcc printed
#   4 kernels pkg  with `kernels` installed, transformers ignores local builds -> uninstall (clone only)
#   5 build        causal-conv1d, then mamba-ssm, --no-build-isolation --no-deps: torch and transformers
#                  are never touched. Each setup.py first tries a prebuilt wheel for this torch/CUDA and
#                  compiles only if there is none. Full pip output: slurm_logs/h1build_<pkg>_<jobid>.log
#   6 check        71_check_fast_path.py: imports, fast path on, fast vs PyTorch path agree
#
# Preamble as in pilot_model.slurm (read act_extract.slurm for the history): inherited CONDA_* stripped
# before `module load`, `set -u` rather than `set -e`, PYTHONUNBUFFERED, commands traced to stderr.

set -u

REPO="${HD_REPO:?set HD_REPO to your clone of hallucination-detection}"
SRC_ENV="${SRC_ENV:-hal-det}"
NEW_ENV="${NEW_ENV:-hal-det-h1}"
MODEL="${MODEL:-falcon-h1-7b-base}"
OUT_DIR="${OUT_DIR:-results/pilot_kernels}"
export MAX_JOBS="${MAX_JOBS:-10}"
export PYTHONUNBUFFERED=1
export PIP_NO_CACHE_DIR=1                 # no second copy of sources and wheels under ~/.cache/pip
export PIP_DISABLE_PIP_VERSION_CHECK=1
JOB="${SLURM_JOB_ID:-local}"

step() { echo "$(date +%F_%T) STEP $*" >&2; }
fail() { echo "FATAL: $*" >&2; exit 1; }

echo "$(date +%F_%T): build $NEW_ENV from $SRC_ENV on $(hostname)" >&2

set -x
for _v in $(compgen -v | grep '^CONDA' || true); do unset "$_v"; done
unset _CE_M _CE_CONDA || true
module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0
if [ -n "${EXTRA_MODULES:-}" ]; then module load $EXTRA_MODULES; fi
set +x
cd "$REPO" || fail "cannot cd to $REPO -- set HD_REPO"
mkdir -p slurm_logs "$OUT_DIR"

step "1 network"
code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' https://pypi.org/simple/causal-conv1d/ || true)
[ "$code" = "200" ] || fail "PyPI not reachable from $(hostname) (HTTP '${code}'). The build needs PyPI; see TICKETS T-014."
gh=$(curl -s -m 20 -o /dev/null -w '%{http_code}' https://github.com/ || true)
echo "  PyPI ok; github.com HTTP ${gh} (prebuilt wheels are fetched from GitHub; without it both packages compile)" >&2

step "2 clone $SRC_ENV -> $NEW_ENV"
if conda env list | awk '{print $1}' | grep -qx "$NEW_ENV"; then
    echo "  $NEW_ENV exists -- reusing it (conda env remove -n $NEW_ENV to start over)" >&2
else
    SRC_PREFIX=$(conda env list | awk -v e="$SRC_ENV" '$1 == e {print $NF}')
    [ -n "$SRC_PREFIX" ] || fail "source env $SRC_ENV not found"
    echo "  $SRC_ENV size: $(du -sh "$SRC_PREFIX" 2>/dev/null | cut -f1)  ($SRC_PREFIX)" >&2
    conda create -y -q -n "$NEW_ENV" --clone "$SRC_ENV" >&2 || fail "conda clone failed"
fi
conda activate "$NEW_ENV"
[ "${CONDA_DEFAULT_ENV:-}" = "$NEW_ENV" ] || fail "conda env is '${CONDA_DEFAULT_ENV:-<none>}', expected $NEW_ENV"
echo "  $NEW_ENV size: $(du -sh "$CONDA_PREFIX" 2>/dev/null | cut -f1)  ($CONDA_PREFIX)" >&2
# A clone copies pip's launcher scripts as they are, so bin/pip can still start hal-det's python and install
# THERE. Every pip call below is `python -m pip`, and python itself must resolve inside the clone.
PIP="python -m pip"
PY_PREFIX=$(python -c "import sys; print(sys.prefix)")
TORCH_FILE=$(python -c "import torch; print(torch.__file__)")
case "$TORCH_FILE" in
    "$CONDA_PREFIX"/*) ;;
    *) fail "torch resolves outside $NEW_ENV ($TORCH_FILE)";;
esac
[ "$PY_PREFIX" = "$CONDA_PREFIX" ] || fail "python prefix is $PY_PREFIX, expected $CONDA_PREFIX"
echo "  python and torch resolve inside $CONDA_PREFIX" >&2

step "3 toolchain"
NVCC_BIN=$(command -v nvcc) || fail "nvcc not on PATH after module load"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC_BIN")")}"
python - <<'PYCHK' >&2 || fail "torch/transformers check failed"
import torch, transformers
print("  torch %s (CUDA %s) | transformers %s | GPU visible: %s"
      % (torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.is_available()))
assert torch.cuda.is_available(), "no GPU visible to torch"
PYCHK
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
NVCC_CUDA=$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\),.*/\1/p')
echo "  nvcc $NVCC_CUDA at $CUDA_HOME | $(gcc --version | head -1)" >&2
[ "${TORCH_CUDA%%.*}" = "${NVCC_CUDA%%.*}" ] || fail "torch was built with CUDA $TORCH_CUDA but nvcc is $NVCC_CUDA; the major versions must match"
echo "  gcc modules available (for EXTRA_MODULES if a build fails on the compiler):" >&2
module -t avail gcc 2>&1 | head -20 >&2 || true

step "4 kernels package"
if python -c "import kernels" 2>/dev/null; then
    echo "  'kernels' is installed: transformers would look for the kernels only on the Hub. Uninstalling it from $NEW_ENV." >&2
    $PIP uninstall -y kernels >&2 || fail "could not uninstall kernels"
else
    echo "  'kernels' not installed -- transformers will use local builds" >&2
fi

step "5 build (MAX_JOBS=$MAX_JOBS)"
python -c "import einops" 2>/dev/null || $PIP install --no-deps einops >&2 || fail "einops install failed"
python -c "import ninja" 2>/dev/null || $PIP install --no-deps ninja >&2 || fail "ninja install failed"
export TMPDIR="/tmp/h1build_${JOB}"
mkdir -p "$TMPDIR"
for pkg in causal-conv1d mamba-ssm; do
    mod="${pkg//-/_}"
    if python -c "import $mod" 2>/dev/null; then
        echo "  $pkg already importable -- not rebuilt" >&2
        continue
    fi
    log="slurm_logs/h1build_${pkg}_${JOB}.log"
    t0=$(date +%s)
    $PIP install --no-build-isolation --no-deps -v "$pkg" > "$log" 2>&1
    rc=$?
    echo "  $pkg: exit $rc after $(( ($(date +%s) - t0) / 60 )) min (log $log)" >&2
    grep -iE "precompiled wheel|building from source|guessing wheel url|successfully installed|error" "$log" | tail -15 >&2
    [ "$rc" -eq 0 ] || fail "$pkg build failed -- read $log"
done
rm -rf "$TMPDIR"
$PIP list 2>/dev/null | grep -iE '^(torch|transformers|triton|causal.conv1d|mamba.ssm|einops|ninja) ' >&2

step "6 check"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
echo "  HF_HOME: ${HF_HOME:-<unset, using ~/.cache/huggingface>} | offline: $HF_HUB_OFFLINE" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >&2
python 71_check_fast_path.py --model_folder "$MODEL" --out-dir "$OUT_DIR"
status=$?

echo "$(date +%F_%T): done (exit $status). Report: $OUT_DIR/fastpath_${MODEL}.json" >&2
exit $status
