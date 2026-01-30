#!/bin/bash
# eval_batch_parallel.sh
# Parallel evaluation for DeltaNet models via Ray

# Resolve repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Ensure all submodules and package code are in PYTHONPATH
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$REPO_ROOT/flame:$REPO_ROOT/flash-linear-attention:$PYTHONPATH"

set -e

# ============================================================================
# Configuration
# ============================================================================

# Usage: bash scripts/eval/eval_batch_parallel.sh <CHECKPOINT_BASE_DIR> [OUTPUT_DIR]

COMPRESSED_BASE=$1
if [[ -z "$COMPRESSED_BASE" ]]; then
    echo "Usage: bash $0 <CHECKPOINT_BASE_DIR> [OUTPUT_DIR]"
    echo "Example: bash $0 ./exp/checkpoints"
    exit 1
fi

# Resolve absolute path for COMPRESSED_BASE
COMPRESSED_BASE=$(cd "$COMPRESSED_BASE" && pwd)

# Default output directory inside the checkpoint base
OUTPUT_BASE_DIR=${2:-"${COMPRESSED_BASE}/eval_results"}
METHOD=${METHOD:-"all"}
MODEL_NAME=${MODEL_NAME:-"none"}

TOKENIZER_PATH="fla-hub/transformer-1.3B-100B"

BATCH_SIZE=8
MAX_LENGTH="10000"
DTYPE="bfloat16"
STEP=-1
TASKS="arc_easy,arc_challenge,hellaswag,winogrande,piqa,wikitext,lambada"

echo "=========================================="
echo "Parallel Evaluation via Ray"
echo "=========================================="
echo "Root:             $REPO_ROOT"
echo "Searching in:     $COMPRESSED_BASE"
echo "Output to:        $OUTPUT_BASE_DIR"
echo "=========================================="

python -u "$REPO_ROOT/scripts/eval/eval_batch_parallel.py" \
    --tasks "${TASKS}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --output_dir "${OUTPUT_BASE_DIR}" \
    --compressed_base "${COMPRESSED_BASE}" \
    --model_prefix "${MODEL_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --dtype "${DTYPE}" \
    --step "${STEP}" \
    --method "${METHOD}"