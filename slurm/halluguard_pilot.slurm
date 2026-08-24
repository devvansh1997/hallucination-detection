#!/bin/bash
#SBATCH --job-name=hg-pilot2
#SBATCH -p highgpu
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/hg-pilot2_%j/hg-pilot2_%j.out
#SBATCH --error=slurm_logs/hg-pilot2_%j/hg-pilot2_%j.err

# PILOT 2 -- diagnostic. Pilot 1 (job 774134) answered the sizing question and raised three
# larger ones; this run measures those instead of guessing at them.
#
# WHAT PILOT 1 SETTLED, so we do not pay for it again:
#
#   fp16 is impossible, not merely lossy. Their line 126 calls torch.linalg.eigvalsh, which has no
#   half-precision kernel on EITHER device -- "linalg_eigh_cuda"/"linalg_eigh_cpu" not implemented
#   for 'Half', and the same for 'BFloat16'. All 200/200 beams raised NotImplementedError. This is
#   the dtype their own and only call site uses (ntk_reward.py:26 loads fp16 whenever device ==
#   "cuda", then calls with param_subset="last_block"), so the released path cannot run as shipped.
#   The fp16 arm is therefore dropped -- it is a proven crash, not a measurement.
#
#   fp32 runs but lost 64 of 200 beams to CUDA OOM. Their line 124 stacks T per-token gradients
#   into a [T, P] tensor while the list of T separate vectors is still alive, so peak is ~2*T*P*4
#   bytes on top of the 30GB fp32 model. The allocator log gives P directly: single-gradient
#   requests of 933,232,640 bytes => P ~ 233M (Qwen's last block), i.e. ~0.93GB PER GENERATED
#   TOKEN. Failed requests ran up to 48.5GB, so a ~52-token completion needs ~97GB of headroom
#   against the ~49GB left after the model. Length decides whether a beam scores at all.
#
# WHAT THIS RUN MEASURES -- all three now emitted by 53 for every beam, including failed ones:
#   1. SELECTION. Failure is a deterministic function of completion length, so the survivors are
#      not a random sample. The run prints mean T and hallucination rate for scored vs failed
#      beams. If those rates differ, pilot 1's 0.3162 describes a different population than the
#      one HARP and our method are evaluated on, and is not comparable to either.
#   2. WHICH TERM RANKS. K = G G^T has unit-norm rows, so trace(K) = T and det(K) <= 1; collinear
#      gradients push eigenvalues onto the 1e-8 clamp and det_k = exp(log_det) underflows to 0. If
#      std(det_k)/std(score) is ~0 then the determinant -- the reason for T autograd passes at
#      0.93GB each -- does no ranking, and the score is just log(sigma_max) - 2log(kappa).
#   3. LENGTH ALONE. AUROC using nothing but the token count. If that matches the method, it is
#      the baseline a reviewer reaches for first.
#
# 400 beams, not 200: pilot 1 left only 77 within-prompt pairs, too few to read anything from.
# Still TyDiQA, still the smallest dataset, so this stays a real sample rather than a corner.
#
# PREREQUISITE (already satisfied on the cluster): the repo cloned next to HARP-Code at
#   ~/Hallucination-Detection/HalluGuard-ICLR2026
# Only Score/halluguard_true.py is imported, and it is imported UNMODIFIED.

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPO=/home/de807845/Hallucination-Detection/hallucination-detection
HG=/home/de807845/Hallucination-Detection/HalluGuard-ICLR2026/Score

echo "${TIMESTAMP}: Loading modules.. "
module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0
conda activate hal-det

export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
# Their peak is one huge contiguous [T, P] block. Fragmentation, not total free memory, is what
# turns a 19GB request into an OOM on a card with 49GB free -- see the pilot-1 allocator log, where
# requests failed against 12GB free while the run continued. Expandable segments let the allocator
# grow a segment instead of demanding one contiguous slab.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$MPLCONFIGDIR"

cd $REPO
echo "repo HEAD: $(git -C $REPO log --oneline -1 2>/dev/null)"
[ -f "$HG/halluguard_true.py" ] || {
    echo "ERROR: $HG/halluguard_true.py not found."
    echo "  cd ~/Hallucination-Detection && git clone https://github.com/Susan571/HalluGuard-ICLR2026.git"
    exit 1; }
nvidia-smi

echo ""
echo "### fp32, 400 beams, full diagnostics"
python 53_halluguard_score.py --dataset tydiqa_gp --model_folder qwen-2.5-7b-instruct \
    --hg-dir "$HG" --dtype float32 --limit 400 --log-every 50 \
    --out-dir "$REPO/results/halluguard_fp32" || exit 1

echo ""
echo "Read the SELECTION and TERMS blocks before the AUROC. If the hallucination rate differs"
echo "between scored and failed beams, the AUROC is not yet comparable to HARP's or ours."
conda deactivate
echo "${TIMESTAMP}: Env Deactivated.. "
