#!/usr/bin/env bash
# Launch the Idea G reorganization-probes experiment on a RunPod pod.
#
# Usage:
#   bash scripts/interventions/run_reorg_probes_runpod.sh <MODEL_ID> <OUTPUT_DIR> [REVISION]
#
# Example:
#   bash scripts/interventions/run_reorg_probes_runpod.sh \
#       allenai/OLMo-7B-hf /workspace/runs/reorg_probes/olmo-7b main
set -euo pipefail

MODEL="${1:?usage: $0 <MODEL_ID> <OUTPUT_DIR> [REVISION]}"
OUTPUT_DIR="${2:?usage: $0 <MODEL_ID> <OUTPUT_DIR> [REVISION]}"
REVISION="${3:-main}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Activate venv (tolerate either bash-style activate or a missing file for CI)
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/filtered}"

mkdir -p "$OUTPUT_DIR"

echo "[reorg-probes] model=$MODEL  revision=$REVISION"
echo "[reorg-probes] dataset_dir=$DATASET_DIR"
echo "[reorg-probes] output_dir=$OUTPUT_DIR"

python -u -m scripts.interventions.reorganization_probes \
    --model "$MODEL" \
    --revision "$REVISION" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --n-imdb-per-class 500 \
    --device cuda \
    --dtype auto \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/run.log"
