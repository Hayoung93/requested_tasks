#!/bin/bash

# ForgeLens Stage 2 Multi-GPU Training Script (4 GPUs)
# Train FAFormer after multi-GPU Stage 1

EXP_NAME="20251208_nanobanana-pro"
TRAIN_JSON="/data/data/deepfake_finetune_dataset/jsons_v2/train_v2.json"
VAL_JSON="/data/data/deepfake_finetune_dataset/jsons_v2/val_v2.json"

# Multi-GPU Configuration
GPU_IDS="0,1,2,3"
NUM_GPUS=4

# Hyperparameters for Stage 2 (adjusted for 4x batch size)
# Original (single GPU from train_stage2_from_pretrained.sh):
#   batch_size=16, lr=2e-6
# Multi-GPU (4 GPUs):
#   batch_size=16 per GPU (64 total effective batch)
#   lr=4e-6 (sqrt(4) = 2x scaling)
BATCH_SIZE=16  # Per GPU
LEARNING_RATE=0.000004  # 4e-6 (2x of original 2e-6)
LR_DECAY_STEP=2
LR_DECAY_FACTOR=0.7
EPOCHS=10

echo "=========================================="
echo "ForgeLens Stage 2 Multi-GPU Training"
echo "=========================================="
echo "Experiment: ${EXP_NAME}"
echo "Train data: ${TRAIN_JSON}"
echo "Val data: ${VAL_JSON}"
echo ""
echo "Multi-GPU Configuration:"
echo "  GPUs: ${GPU_IDS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "  Learning rate: ${LEARNING_RATE} (scaled for larger batch)"
echo "=========================================="
echo ""

# Check if Stage 1 model exists
STAGE1_MODEL="./check_points/${EXP_NAME}/train_stage_1/model/intermediate_model_best.pth"
if [ ! -f "$STAGE1_MODEL" ]; then
    echo "ERROR: Stage 1 model not found at: $STAGE1_MODEL"
    echo "Please run train_stage1_multigpu.sh first!"
    exit 1
fi

echo "Using Stage 1 model: $STAGE1_MODEL"
echo ""

# Check GPU availability
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

python train.py \
    --experiment_name ${EXP_NAME} \
    --use_json_dataset \
    --train_json ${TRAIN_JSON} \
    --val_json ${VAL_JSON} \
    --training_stage 2 \
    --use_multi_gpu \
    --gpu_ids ${GPU_IDS} \
    --stage1_batch_size 32 \
    --stage1_epochs 30 \
    --stage1_learning_rate 0.00002 \
    --stage1_lr_decay_step 3 \
    --stage1_lr_decay_factor 0.8 \
    --WSGM_count 12 \
    --WSGM_reduction_factor 4 \
    --stage2_batch_size ${BATCH_SIZE} \
    --stage2_epochs ${EPOCHS} \
    --stage2_learning_rate ${LEARNING_RATE} \
    --stage2_lr_decay_step ${LR_DECAY_STEP} \
    --stage2_lr_decay_factor ${LR_DECAY_FACTOR} \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 8 \
    --seed 3407

echo ""
echo "=========================================="
echo "Stage 2 Multi-GPU Training Complete!"
echo "Model saved to: ./check_points/${EXP_NAME}/train_stage_2/model/"
echo ""
echo "Total training time (estimated):"
echo "  Stage 1: 2-2.5 hours"
echo "  Stage 2: 0.5-1 hour"
echo "  Total: ~3-3.5 hours (vs 13.5 hours on single GPU)"
echo "=========================================="
