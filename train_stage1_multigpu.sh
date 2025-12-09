#!/bin/bash

# ForgeLens Stage 1 Multi-GPU Training Script (4 GPUs)
# Fine-tune from pretrained weights with 4x batch size
# Hyperparameters adjusted for larger effective batch size

EXP_NAME="20251208_nanobanana-pro"
TRAIN_JSON="/data/data/deepfake_finetune_dataset/jsons_v2/train_v2.json"
VAL_JSON="/data/data/deepfake_finetune_dataset/jsons_v2/val_v2.json"

# ForgeLens pretrained weight path
PRETRAINED_PATH="/data/weights/forgelens/training_setting_1.pth"

# Multi-GPU Configuration
GPU_IDS="0,1,2,3"  # Change to your available GPU IDs
NUM_GPUS=4

# Hyperparameters adjusted for 4x batch size
# Original: batch_size=32, lr=1e-5 (single GPU)
# New: batch_size=32 per GPU (128 total), lr=2e-5 (sqrt(4) scaling)
BATCH_SIZE=32  # Per GPU
LEARNING_RATE=0.00002  # 2e-5 (2x of original 1e-5)
LR_DECAY_STEP=3  # Slightly increased
LR_DECAY_FACTOR=0.8
EPOCHS=30

echo "=========================================="
echo "ForgeLens Stage 1 Multi-GPU Fine-tuning"
echo "=========================================="
echo "Experiment: ${EXP_NAME}"
echo "Train data: ${TRAIN_JSON}"
echo "Val data: ${VAL_JSON}"
echo "Pretrained: ${PRETRAINED_PATH}"
echo ""
echo "Multi-GPU Configuration:"
echo "  GPUs: ${GPU_IDS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "  Learning rate: ${LEARNING_RATE} (scaled for larger batch)"
echo "=========================================="
echo ""

# Check if pretrained weight exists
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "ERROR: Pretrained weight not found at: $PRETRAINED_PATH"
    echo ""
    echo "Please verify the pretrained weight path."
    echo "Expected: /data/weights/forgelens/training_setting_1.pth"
    exit 1
fi

# Check GPU availability
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

python train.py \
    --experiment_name ${EXP_NAME} \
    --use_json_dataset \
    --train_json ${TRAIN_JSON} \
    --val_json ${VAL_JSON} \
    --training_stage 1 \
    --pretrained_stage1_path ${PRETRAINED_PATH} \
    --use_multi_gpu \
    --gpu_ids ${GPU_IDS} \
    --stage1_batch_size ${BATCH_SIZE} \
    --stage1_epochs ${EPOCHS} \
    --stage1_learning_rate ${LEARNING_RATE} \
    --stage1_lr_decay_step ${LR_DECAY_STEP} \
    --stage1_lr_decay_factor ${LR_DECAY_FACTOR} \
    --WSGM_count 12 \
    --WSGM_reduction_factor 4 \
    --num_workers 8 \
    --seed 3407

echo ""
echo "=========================================="
echo "Stage 1 Multi-GPU Training Complete!"
echo "Model saved to: ./check_points/${EXP_NAME}/train_stage_1/model/"
echo ""
echo "Training time comparison:"
echo "  Single GPU (batch=32): ~15 min/epoch × 30 epochs = 7.5 hours"
echo "  4 GPUs (batch=128):    ~4-5 min/epoch × 30 epochs = 2-2.5 hours"
echo ""
echo "Next step: Run Stage 2 training"
echo "  bash train_stage2_multigpu.sh"
echo "=========================================="
