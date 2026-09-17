#!/bin/bash
# submit_tucker3.sh -- the 3-mode Tucker ablation (T-015), extraction and evaluation submitted together.
#
#   bash submit_tucker3.sh                        # 2 models x 2 datasets: 4 GPU extractions + 32 CPU evaluations
#   DRY=1 bash submit_tucker3.sh                  # print the plan, queue nothing
#   FAMILIES=orderstats bash submit_tucker3.sh    # one family only (default: orderstats mean)
#   MODELS=llama-3.1-8b DATASETS=tydiqa_gp bash submit_tucker3.sh
#
# RUN FROM A NEWTON TERMINAL.
#
# ISOLATION. Nothing here touches the reported detector: 69 writes only ../data-tokenstates, 70 writes only
# results/tensor_tucker, and both only READ the pinned sequences and features. No existing script changes.
#
# WHAT IS QUEUED, per model and dataset:
#   1 GPU job    69: token-level window states (skipped if ../data-tokenstates/<model>/<dataset>/meta.json exists)
#   8 CPU jobs   70: one per (family, T'), started when the extraction succeeds (--dependency=afterok);
#                    each runs both token ranks for its T':  T'=1: R_T 1 | 2: 1,2 | 4: 2,4 | 8: 4,8
#                    orderstats = the reported peak/range/update computed within each token bin; its T'=1 job
#                                 reproduces the reported detector and prints an ANCHOR line -- read it first
#                    mean       = average pooling within each bin (state + update)
# If an extraction fails, its evaluations stay PENDING with reason DependencyNeverSatisfied: scancel them,
# fix, and rerun this script (finished pieces are skipped).
#
# DISK. tokens.npy is float16, (total answer tokens) x 9 x D: roughly 10-15 GB per TruthfulQA cell and 2-4 GB
# per TyDiQA-GP cell. 69 prints the exact size before its first forward pass and refuses above 60 GB.

set -euo pipefail

HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TOK_ROOT="$(cd "$HD_REPO/.." && pwd)/data-tokenstates"
cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs" "$HD_REPO/results/tensor_tucker"

MODELS="${MODELS:-qwen-2.5-7b-instruct llama-3.1-8b}"
DATASETS="${DATASETS:-tydiqa_gp truthfulqa}"
FAMILIES="${FAMILIES:-orderstats mean}"
PART="${PART:-highgpu}"
EXCLUDE="${EXCLUDE:-evc103}"
TGROUPS="1:1 2:1,2:2 4:2,4:4 8:4,8:8"         # one evaluation job per T' (not GROUPS: that name is a bash builtin)

# Memory and wall time per (dataset, family, T'). orderstats carries range and update at twice the hidden size,
# so its T'=8 tensors are the largest (~23 GB at float16 for LLaMA TruthfulQA, plus scaling copies).
eval_mem() {
    case "$1:$2:$3" in
        truthfulqa:orderstats:8) echo 192G;; truthfulqa:orderstats:4) echo 128G;; truthfulqa:mean:8) echo 128G;;
        truthfulqa:*)            echo 96G;;
        tydiqa_gp:orderstats:8)  echo 128G;; tydiqa_gp:orderstats:4)  echo 96G;;  tydiqa_gp:mean:8)  echo 96G;;
        *)                       echo 64G;;
    esac
}
eval_time() { case "$1:$2:$3" in truthfulqa:orderstats:8) echo 16:00:00;; *) echo 12:00:00;; esac; }

JOBID=""
sub() {
    local stage="$1" opts="$2"; shift 2
    local out
    if ! out=$(sbatch --parsable $opts "$stage" "$@" 2>&1); then
        echo "" >&2
        echo "ERROR: sbatch rejected this job -- nothing further has been queued." >&2
        printf '  %s\n' "$out" >&2
        echo "  opts: $opts" >&2; echo "  args: $*" >&2
        exit 1
    fi
    [ -n "$out" ] || { echo "ERROR: sbatch returned an empty job id" >&2; exit 1; }
    JOBID="$out"
}

N=0
for MO in $MODELS; do
  for DS in $DATASETS; do
    tag="${MO:0:4} ${DS}"
    EXT=""
    if [ -f "$TOK_ROOT/$MO/$DS/meta.json" ]; then
      printf "  %-16s extraction SKIP (complete)\n" "$tag"
    else
      mem=$([ "$DS" = truthfulqa ] && echo 96G || echo 64G)
      if [ "${DRY:-0}" = "1" ]; then
        printf "  %-16s extraction would queue (GPU, mem %s, 03:00:00)\n" "$tag" "$mem"; EXT="DRY"
      else
        sub "$HD_REPO/slurm/tokenstates_extract.slurm" \
            "-p $PART --gres=gpu:1 --mem=$mem --time=03:00:00 --job-name=tok-${MO:0:4}-${DS:0:4} ${EXCLUDE:+--exclude=$EXCLUDE} --export=ALL,HD_REPO=$HD_REPO" \
            "$MO" "$DS"
        EXT="$JOBID"; N=$((N+1))
        printf "  %-16s extraction -> %s\n" "$tag" "$EXT"
      fi
    fi
    for FAM in $FAMILIES; do
      for G in $TGROUPS; do
        T="${G%%:*}"
        OUT="$HD_REPO/results/tensor_tucker/tucker3_${MO}_${DS}_${FAM}_RF_T${T}.json"
        if [ -f "$OUT" ] && grep -q '"complete": true' "$OUT"; then
          printf "  %-16s %-10s T'=%s SKIP (complete)\n" "$tag" "$FAM" "$T"; continue
        fi
        mem=$(eval_mem "$DS" "$FAM" "$T"); tim=$(eval_time "$DS" "$FAM" "$T")
        dep=""
        [ -n "$EXT" ] && [ "$EXT" != "DRY" ] && dep="--dependency=afterok:$EXT"
        if [ "${DRY:-0}" = "1" ]; then
          printf "  %-16s %-10s T'=%s settings %-8s would queue (mem %s, %s%s)\n" "$tag" "$FAM" "$T" "$G" "$mem" "$tim" \
              "$([ -n "$EXT" ] && echo ', after extraction')"
          continue
        fi
        sub "$HD_REPO/slurm/analysis_stage.slurm" \
            "-p $PART --mem=$mem --cpus-per-task=8 --time=$tim --job-name=tk3-${MO:0:4}-${DS:0:4}-${FAM:0:3}-T${T} ${dep} ${EXCLUDE:+--exclude=$EXCLUDE} --export=ALL,HD_REPO=$HD_REPO" \
            70_tensor_tucker.py --dataset "$DS" --model_folder "$MO" --family "$FAM" --settings "$G"
        printf "  %-16s %-10s T'=%s -> %s  (mem %s, %s%s)\n" "$tag" "$FAM" "$T" "$JOBID" "$mem" "$tim" "$([ -n "$dep" ] && echo ", after $EXT")"
        N=$((N+1))
      done
    done
  done
done

echo
echo "Queued $N jobs.  Watch: squeue -u \$USER   (evaluations show PENDING (Dependency) until their extraction ends)"
echo "Read first: the extraction PREFLIGHT line, then the ANCHOR line in each orderstats T'=1 log (it must be within 0.5 pts)."
echo "Results: results/tensor_tucker/tucker3_<model>_<dataset>_<family>_RF_T<T'>.json"
