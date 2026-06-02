#!/bin/bash
# ============================================================
# Example Evaluation Script for TAO-RL
# ============================================================
# Usage:
#   1. Set the PATH variables below to match your environment
#   2. bash scripts/example_test.sh
# ============================================================

set -e

# -------------------- Path Configuration --------------------
# Replace these paths with your actual directories
export MODEL_PATH="/path/to/your/merged_model"       # Path to the merged HF model
export DATA_PATH="/path/to/your/datasets"             # Dataset directory
export LOG_PATH="/path/to/logs"                       # Log directory

# -------------------- Model Configuration --------------------
export MODEL_NAME="Qwen2.5-7B-Step-80"                # Model display name (for logging)

# -------------------- Hardware Configuration --------------------
export CUDA_VISIBLE_DEVICES="0,1,2,3"                 # GPU IDs
export NNODES=1                                        # Number of nodes
export GPUS_PER_NODE=4                                 # GPUs per node
export ROLLOUT_TP=4                                    # Tensor parallel size

# -------------------- Evaluation Parameters --------------------
export MAX_RESPONSE_LENGTH=12000                       # Max response tokens (long for eval)
export MAX_PROMPT_LENGTH=36000                         # Max prompt tokens
export MAX_TURNS=10                                    # Max tool-use turns
export N_VAL=16                                        # N for pass@k
export VAL_SAMPLE_SIZE=500                             # Number of test samples per benchmark
export SP_SIZE=2                                       # Sequence parallel size

# -------------------- Sandbox --------------------
export SANDBOX_ENDPOINT="http://127.0.0.1:12345/faas/sandbox/"

# -------------------- Benchmarks to Evaluate --------------------
# Format: "benchmark_name|dataset_path"
benchmarks=(
    "deepscaler_aime|deepscaler/aime"
    "deepscaler_aime25|deepscaler/aime25"
    "MATH-500|MATH-500/test"
    "OlympiadBench|OlympiadBench/OE_TO_maths_en_COMP"
    "AMC23|amc23/test"
    "HMMT25|hmmt25/test"
    "minerva|minerva/test"
)

# -------------------- Create Log Directory --------------------
mkdir -p "${LOG_PATH}"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Timestamp: ${TIMESTAMP}"
echo "Model: ${MODEL_PATH}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo ""

# -------------------- Run Evaluation --------------------
for benchmark in "${benchmarks[@]}"; do
    # Extract benchmark name and dataset path
    IFS="|" read -r benchmark_name dataset_path <<< "${benchmark}"

    # Create log file for this benchmark
    LOG_FILE="${LOG_PATH}/${benchmark_name}.log"

    echo "============================================================"
    echo "Testing benchmark: ${benchmark_name}"
    echo "Dataset: ${dataset_path}"
    echo "Log file: ${LOG_FILE}"
    echo "============================================================"

    # Run test
    bash train.sh \
        --max_response_length "${MAX_RESPONSE_LENGTH}" \
        --max_prompt_length "${MAX_PROMPT_LENGTH}" \
        --model_name "${MODEL_NAME}" \
        --max_turns "${MAX_TURNS}" \
        --valid_dataset "${dataset_path}" \
        --val_only True \
        --n_val "${N_VAL}" \
        --output_acc_to_file True \
        --val_sample_size "${VAL_SAMPLE_SIZE}" \
        --rollout_tp "${ROLLOUT_TP}" \
        --sp_size "${SP_SIZE}" \
        > "${LOG_FILE}" 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ Benchmark ${benchmark_name} completed successfully"
    else
        echo "✗ Benchmark ${benchmark_name} failed (see log)"
    fi

    sleep 5
    echo ""
done

echo "============================================================"
echo "All benchmarks completed!"
echo "Check logs in: ${LOG_PATH}"
echo "============================================================"
