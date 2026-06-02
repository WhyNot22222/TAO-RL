#!/bin/bash
# ============================================================
# Checkpoint Merger Script for TAO-RL
# ============================================================
# Usage: bash scripts/model_merger.sh
# ============================================================

set -e

# -------------------- Path Configuration --------------------
# 请将以下路径替换为你的实际路径
export HF_MODEL_DIR="/path/to/your/models/Qwen2.5-7B"         # 原始基础模型的路径
export CHECKPOINT_DIR="/path/to/checkpoints/example_experiment" # 训练输出的 checkpoint 目录
export STEP=800                                                 # 你想要合并的 global step 步数
export TARGET_DIR="/path/to/merged/model_output"              # 合并后模型的保存路径

echo "Starting model merge for step ${STEP}..."
echo "Source HF Model: ${HF_MODEL_DIR}"
echo "Checkpoint: ${CHECKPOINT_DIR}/global_step_${STEP}/actor"
echo "Target Dir: ${TARGET_DIR}"

python scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path $HF_MODEL_DIR \
    --local_dir $CHECKPOINT_DIR/global_step_$STEP/actor \
    --target_dir $TARGET_DIR

# 拷贝 tokenizer 相关文件以确保模型可直接用于推理评估
cp $HF_MODEL_DIR/tokenizer* $TARGET_DIR

echo "Model merge completed successfully!"