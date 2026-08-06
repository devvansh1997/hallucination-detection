#!/bin/bash
# harp_patch_devices.sh -- the ONLY modification we make to HARP's released code.
#
# WHY THIS EXISTS
#   HARP's main.py hardcodes three GPUs:
#       main.py:77   data_device  = "cuda:1"     <- pure storage, no compute ever runs here
#       main.py:78   model_device = "cuda:2"     <- all compute
#   As shipped it therefore requires >=3 visible devices (and never uses cuda:0 at all).
#   On a single-GPU node both strings are invalid and the run dies at model load.
#
#   data_device is only ever used as `save_device=` (main.py:212) and `map_location=`
#   (main.py:174). main.py:238 does emb.to(device) to pull each tensor back onto the MODEL's
#   GPU before projecting, so moving storage to CPU costs one PCIe copy per beam (~0.06 ms
#   against a ~50 ms forward pass) and changes no arithmetic.
#
#   Nothing else in their code is touched: not the SVD, not the projection, not the MLP,
#   not the AUROC, not the split. Two device strings.
#
# The pristine file is kept as main.py.orig so `diff` can prove that at any time.
# Idempotent: safe to call from every job.

set -euo pipefail

HARP_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../HARP-Code" && pwd)}"
MAIN="${HARP_DIR}/main.py"

[ -f "$MAIN" ] || { echo "ERROR: no main.py at ${MAIN}" >&2; exit 1; }

echo "=================================================================="
echo "  HARP device patch -- ${MAIN}"
echo "=================================================================="

# keep one pristine copy, made before the first edit ever happens
if [ ! -f "${MAIN}.orig" ]; then
    cp "$MAIN" "${MAIN}.orig"
    echo "  saved pristine copy: ${MAIN}.orig"
fi

echo "  before:"
grep -n '^\s*\(data_device\|model_device\)\s*=' "$MAIN" | sed 's/^/    /'

sed -i 's|^\(\s*\)data_device\s*=\s*"cuda:1"|\1data_device = "cpu"|'      "$MAIN"
sed -i 's|^\(\s*\)model_device\s*=\s*"cuda:2"|\1model_device = "cuda:0"|' "$MAIN"

echo "  after:"
grep -n '^\s*\(data_device\|model_device\)\s*=' "$MAIN" | sed 's/^/    /'

# hard-fail rather than run against a file we did not actually patch
grep -q 'data_device = "cpu"'      "$MAIN" || { echo "ERROR: data_device not patched"  >&2; exit 1; }
grep -q 'model_device = "cuda:0"'  "$MAIN" || { echo "ERROR: model_device not patched" >&2; exit 1; }

echo
echo "  full diff against pristine (this is the complete set of our changes):"
diff "${MAIN}.orig" "$MAIN" | sed 's/^/    /' || true

# ---- disk check: their outputs land inside HARP-Code, which is usually in $HOME ----
echo
echo "  disk available at ${HARP_DIR}:"
df -h "$HARP_DIR" | sed 's/^/    /'
if command -v quota >/dev/null 2>&1; then
    echo "  home quota:"
    quota -s 2>/dev/null | sed 's/^/    /' || echo "    (quota reported nothing)"
fi
echo
echo "  NOTE: hidden_state/ needs roughly  tydiqa 1G | truthfulqa 6G | nq_open 28G |"
echo "        triviaqa 70-120G.  If this filesystem is short, symlink it to scratch"
echo "        BEFORE running (no code change, their code just follows the link):"
echo "            mkdir -p /scratch/\$USER/harp_hidden_state"
echo "            ln -s /scratch/\$USER/harp_hidden_state ${HARP_DIR}/hidden_state"
echo "=================================================================="
