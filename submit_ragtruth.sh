#!/bin/bash
# submit_ragtruth.sh -- RAGTruth read by its own generators (74), then the temporal analysis (75). T-024.
#
#   DRY=1 bash submit_ragtruth.sh                          # check data and weights, queue nothing -- run first
#   bash submit_ragtruth.sh                                # default: llama-2-7b-chat
#   GENERATORS="llama-2-7b-chat llama-2-13b-chat" bash submit_ragtruth.sh
#
# Per generator: one GPU extraction (74), then one CPU analysis (75) that waits on it (--dependency=afterok).
# Safe to rerun: a finished extraction or analysis is skipped, and an extraction already in the queue is not
# queued again -- the analysis is chained onto it instead.
#
# RUN FROM A NEWTON TERMINAL. Two things must exist first; this script checks both and prints the exact
# command for whichever is missing:
#   1. the RAGTruth release at ../data-ragtruth/RAGTruth/dataset/{response,source_info}.jsonl
#   2. the generator's weights in the Hugging Face cache (jobs run offline; the licence is per account)
#
# ISOLATION. 74 writes only ../data-ragtruth/<generator>/; 75 writes only results/token_structure/.
# DISK. Weights ~13.5 GB (7B) / ~26 GB (13B) in the HF cache; the stored states are under 2 GB per generator.

set -euo pipefail

HD_REPO="${HD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RT_ROOT="$(cd "$HD_REPO/.." && pwd)/data-ragtruth"
RAW="$RT_ROOT/RAGTruth/dataset"
cd "$HD_REPO"
mkdir -p "$HD_REPO/slurm_logs" "$HD_REPO/results/token_structure"

GENERATORS="${GENERATORS:-llama-2-7b-chat}"
PART="${PART:-highgpu}"
EXCLUDE="${EXCLUDE:-evc103}"
HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

# must match config.yaml ragtruth.generators
model_id() {
    case "$1" in
        llama-2-7b-chat)  echo "meta-llama/Llama-2-7b-chat-hf";;
        llama-2-13b-chat) echo "meta-llama/Llama-2-13b-chat-hf";;
        *) echo "";;
    esac
}

if [ ! -f "$RAW/response.jsonl" ] || [ ! -f "$RAW/source_info.jsonl" ]; then
    echo "RAGTruth release missing at $RAW. Fetch it (public, a few tens of MB):"
    echo "  git clone --depth 1 https://github.com/ParticleMedia/RAGTruth $RT_ROOT/RAGTruth"
    echo "then check it:  python 74_extract_ragtruth.py --inspect"
    exit 1
fi

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
for GEN in $GENERATORS; do
    ID="$(model_id "$GEN")"
    [ -n "$ID" ] || { echo "unknown generator $GEN (see config.yaml ragtruth.generators)"; exit 1; }

    EXT=""
    if [ -f "$RT_ROOT/$GEN/meta.json" ]; then
        printf "  %-18s extraction SKIP (complete)\n" "$GEN"
    else
        QUEUED="$(squeue -h -u "$USER" -n "rgt-${GEN}" -o %i 2>/dev/null | head -1 || true)"
        if [ -n "$QUEUED" ]; then
            printf "  %-18s extraction already queued or running (%s)\n" "$GEN" "$QUEUED"
            EXT="$QUEUED"
        else
            CACHE="$HUB/models--${ID//\//--}"
            if ! ls "$CACHE"/snapshots/*/*.safetensors >/dev/null 2>&1 && ! ls "$CACHE"/snapshots/*/*.bin >/dev/null 2>&1 \
               && [ "${ALLOW_DOWNLOAD:-0}" != "1" ]; then
                printf "  %-18s weights MISSING in %s\n" "$GEN" "$HUB"
                echo "    once the licence is granted, download them on a node with internet (hal-det env):"
                echo "    python -c \"from huggingface_hub import snapshot_download as s; print(s('$ID', allow_patterns=['*.safetensors','*.json','tokenizer.model']))\""
                continue
            fi
            if [ "${DRY:-0}" = "1" ]; then
                printf "  %-18s extraction would queue (GPU, mem 64G, 03:00:00) -- %s\n" "$GEN" "$ID"
                EXT="DRY"
            else
                sub "$HD_REPO/slurm/ragtruth_extract.slurm" \
                    "-p $PART --gres=gpu:1 --mem=64G --time=03:00:00 --job-name=rgt-${GEN} ${EXCLUDE:+--exclude=$EXCLUDE} --export=ALL,HD_REPO=$HD_REPO" \
                    "$GEN"
                EXT="$JOBID"; N=$((N+1))
                printf "  %-18s extraction -> %s\n" "$GEN" "$EXT"
            fi
        fi
    fi

    OUT="$HD_REPO/results/token_structure/ragtruth_${GEN}.json"
    if [ -f "$OUT" ] && grep -q '"complete": true' "$OUT"; then
        printf "  %-18s analysis SKIP (complete)\n" "$GEN"; continue
    fi
    if [ -n "$(squeue -h -u "$USER" -n "rga-${GEN}" -o %i 2>/dev/null | head -1 || true)" ]; then
        printf "  %-18s analysis already queued\n" "$GEN"; continue
    fi
    dep=""
    [ -n "$EXT" ] && [ "$EXT" != "DRY" ] && dep="--dependency=afterok:$EXT"
    if [ "${DRY:-0}" = "1" ]; then
        printf "  %-18s analysis would queue (CPU, mem 32G, 02:00:00%s)\n" "$GEN" "$([ -n "$EXT" ] && echo ', after extraction')"
        continue
    fi
    sub "$HD_REPO/slurm/analysis_stage.slurm" \
        "-p $PART --mem=32G --cpus-per-task=8 --time=02:00:00 --job-name=rga-${GEN} ${dep} ${EXCLUDE:+--exclude=$EXCLUDE} --export=ALL,HD_REPO=$HD_REPO" \
        75_ragtruth_temporal.py --generator "$GEN"
    printf "  %-18s analysis -> %s%s\n" "$GEN" "$JOBID" "$([ -n "$dep" ] && echo " (after $EXT)")"
    N=$((N+1))
done

echo
echo "Queued $N jobs.  Watch: squeue -u \$USER"
echo "Extraction log (rgt-*): read the span check, the three alignment examples, then the template check line."
echo "Analysis log (rga-*): ends with a SUMMARY block; the ONSET jump line is the answer."
echo "Output: ../data-ragtruth/<generator>/ and results/token_structure/ragtruth_<generator>.json"
