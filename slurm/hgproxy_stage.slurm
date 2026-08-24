#!/bin/bash
#SBATCH --job-name=hgproxy
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/hgproxy_%j/hgproxy_%j.out
#SBATCH --error=slurm_logs/hgproxy_%j/hgproxy_%j.err
#
# usage (via submit_hgproxy.sh, not directly):
#   sbatch hgproxy_stage.slurm <model_folder> <dataset>
#
# HalluGuard's DEFAULT path -- Score/func/metric.py::getNTKS3Score. No gradients: one no-grad
# forward pass per prompt group, mean-pooled hidden states at layer L/2. This is the path that can
# run Llama2-70B, and the one their published numbers most plausibly came from. 53's gradient
# version needs 0.93 GB per generated token and dies past T=27; this one needs a forward pass.
#
# MEMORY. output_hidden_states=True materialises all n_layers+1 tensors even though we read one:
# 10 beams x seq_len x hidden x (n_layers+1) x 2 bytes. TriviaQA prompts are the long ones, so at
# ~600 tokens that is 10 x 600 x 4096 x 33 x 2 = 1.6 GB per batch on top of a 16 GB bf16 model.
# 80G is comfortable; the ceiling here is nothing like 53's.
#
# IDEMPOTENT. Re-running skips any dataset whose JSON already parses and carries a per-beam block,
# so a partial fan-out can be re-fired without redoing finished work.

set -u
MODEL="${1:?usage: hgproxy_stage.slurm <model_folder> <dataset>}"
DS="${2:?usage: hgproxy_stage.slurm <model_folder> <dataset>}"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPO=/home/de807845/Hallucination-Detection/hallucination-detection
OUT="${REPO}/results/halluguard_proxy"
JSON="${OUT}/hgproxy_${MODEL}_${DS}.json"

echo "${TIMESTAMP}: ${MODEL} / ${DS}"
module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0
conda activate hal-det

export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$MPLCONFIGDIR" "$OUT"

cd "$REPO"
echo "repo HEAD: $(git -C $REPO log --oneline -1 2>/dev/null)"

if [ "${FORCE:-0}" != "1" ] && [ -f "$JSON" ]; then
    if python - "$JSON" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("  existing JSON unreadable (%s) -- recomputing" % e); sys.exit(1)
if not d.get("per_beam") or not d.get("per_prompt") or not d.get("n_beams"):
    print("  existing JSON incomplete -- recomputing"); sys.exit(1)
print("  verified: %d beams / %d prompts already scored" % (d["n_beams"], d["n_prompts"]))
sys.exit(0)
PYEOF
    then echo "SKIP: ${MODEL}/${DS} already complete -> $JSON"; exit 0; fi
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python 54_halluguard_proxy.py --dataset "$DS" --model_folder "$MODEL" \
    --dtype bfloat16 --log-every 200 || exit 1

conda deactivate
echo "${TIMESTAMP}: done ${MODEL}/${DS}"
