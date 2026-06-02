#!/bin/bash
# ============================================================
# Example Training Script for TAO-RL
# ============================================================
# Usage:
#   1. Set the PATH variables below to match your environment
#   2. bash scripts/example_train.sh
# ============================================================

set -e

# -------------------- Path Configuration --------------------
# Replace these paths with your actual directories
export MODEL_PATH="/path/to/your/models"           # Base model directory
export DATA_PATH="/path/to/your/datasets"           # Dataset directory
export CHECKPOINT_PATH="/path/to/checkpoints"       # Output checkpoint directory
export LOG_PATH="/path/to/logs"                     # Log directory

# -------------------- Model Configuration --------------------
export MODEL_NAME="Qwen2.5-7B"                      # Model name (subdirectory under MODEL_PATH)
export CKPT_NAME="example_experiment"                # Checkpoint subfolder name

# -------------------- Hardware Configuration --------------------
export CUDA_VISIBLE_DEVICES="0,1,2,3"               # GPU IDs
export NNODES=1                                      # Number of nodes
export GPUS_PER_NODE=4                               # GPUs per node
export ROLLOUT_TP=4                                  # Tensor parallel size for rollout

# -------------------- Training Hyperparameters --------------------
export TRAIN_BATCH_SIZE=256                          # Total training batch size
export MAX_RESPONSE_LENGTH=4000                      # Max response tokens
export MAX_PROMPT_LENGTH=8000                        # Max prompt tokens
export MAX_TURNS=5                                   # Max tool-use turns
export TOTAL_EPOCHS=5                                # Number of epochs
export OVERSAMPLE=2                                  # Oversampling multiplier
export VAL_SAMPLE_SIZE=50                            # Validation samples
export N_VAL=16                                      # N for validation pass@k

# -------------------- Dataset Configuration --------------------
export TRAIN_DATASET="DAPO-Math-17k/train"

# -------------------- Algorithm Flags --------------------
export USE_AEPO_SHAPING=true                         # Enable AEPO shaping
export AEPO_SHAPING_COEFF=0.01                       # AEPO shaping coefficient
export USE_AEPO_ANTI_CLIP=false                      # Disable anti-clipping
export ENTROPY_PERCENTILE=0.8                        # Entropy percentile threshold

# -------------------- Resume / Logging --------------------
export RESUME=false                                  # Resume from latest checkpoint
export WANDB_API_KEY="your_wandb_api_key"            # WandB API key (optional)
export SANDBOX_ENDPOINT="http://127.0.0.1:12345/faas/sandbox/"

# -------------------- Derived Paths --------------------
LOGFILE="${LOG_PATH}/train/${CKPT_NAME}.log"

# -------------------- Launch Training --------------------
echo "Starting training: ${CKPT_NAME}"
echo "Model: ${MODEL_PATH}/${MODEL_NAME}"
echo "Log: ${LOGFILE}"
echo ""

nohup bash train.sh \
    --config_name tao_rl_trainer \
    --model_name "${MODEL_NAME}" \
    --max_response_length "${MAX_RESPONSE_LENGTH}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --max_turns "${MAX_TURNS}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --val_sample_size "${VAL_SAMPLE_SIZE}" \
    --n_val "${N_VAL}" \
    --train_dataset "${TRAIN_DATASET}" \
    --rollout_tp "${ROLLOUT_TP}" \
    --oversample "${OVERSAMPLE}" \
    --total_epochs "${TOTAL_EPOCHS}" \
    --val_before_train True \
    --use_causal_entropy "${USE_AEPO_SHAPING}" \
    --use_aepo_shaping "${USE_AEPO_SHAPING}" \
    --aepo_shaping_coeff "${AEPO_SHAPING_COEFF}" \
    --use_aepo_anti_clip "${USE_AEPO_ANTI_CLIP}" \
    --entropy_coeff "${ENTROPY_COEFF}" \
    --entropy_percentile "${ENTROPY_PERCENTILE}" \
    > "${LOGFILE}" 2>&1 &

echo ""
echo "Task is running in background, PID: $!"
echo "Log file: ${LOGFILE}"
echo "Follow logs: tail -f ${LOGFILE}"
