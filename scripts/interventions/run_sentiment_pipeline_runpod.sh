#!/usr/bin/env bash
# Run the sentiment-control pipeline (Stages A/B/C) on a single model on the
# RunPod H100. Activates the pre-built venv and writes results under
# /workspace/runs/sentiment_<model>/.
#
# Usage:
#   bash scripts/interventions/run_sentiment_pipeline_runpod.sh \
#       --model allenai/OLMo-7B-hf --output-dir /workspace/runs/sentiment_olmo-7b
#
# Optional args passed through to the python module:
#   --n-per-class N       (default 1000)
#   --max-tokens N        (default 256)
#   --max-samples-grad N  (default 200)
#   --device cuda|cpu     (default cuda)
#   --dtype auto|bfloat16|float32  (default auto → bf16 for 7B+)
#   --skip-attribution / --skip-recovery / --skip-qkv
set -euo pipefail

MODEL=""
OUTPUT_DIR=""
REVISION="main"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"; shift 2 ;;
        --revision)
            REVISION="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    echo "ERROR: --model required" >&2
    exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    # derive sanitized model dir
    SAFE=$(echo "${MODEL}" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
    OUTPUT_DIR="/workspace/runs/sentiment_${SAFE}"
fi

mkdir -p "${OUTPUT_DIR}"
echo "Model:       ${MODEL}"
echo "Revision:    ${REVISION}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Extra args:  ${EXTRA_ARGS[*]:-}"

# Environment
REPO_ROOT="${REPO_ROOT:-/workspace/code/from-tokens-to-semantics}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

if [[ ! -d "${VENV_PATH}" ]]; then
    echo "ERROR: venv not found at ${VENV_PATH}" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "Starting sentiment pipeline..."
python -m scripts.interventions.sentiment_pipeline \
    --model "${MODEL}" \
    --revision "${REVISION}" \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/run.log"

echo "Finished. Results in ${OUTPUT_DIR}"
