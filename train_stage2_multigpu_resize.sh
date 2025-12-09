#!/bin/bash

# ForgeLens Stage 2 Training Script
# After fine-tuning Stage 1 from pretrained weights

EXP_NAME="deepfake_finetune_5k_from_pretrained"
TRAIN_JSON="/data/data/deepfake_finetune_dataset/jsons/train_small_5000.json"
VAL_JSON="/data/data/deepfake_finetune_dataset/jsons/val.json"

echo "=========================================="
echo "ForgeLens Stage 2 Training"
echo "=========================================="
echo "Experiment: ${EXP_NAME}"
echo "Train data: ${TRAIN_JSON}"
echo "Val data: ${VAL_JSON}"
echo "=========================================="
echo ""

# Check if Stage 1 model exists
STAGE1_MODEL="./check_points/${EXP_NAME}/train_stage_1/model/intermediate_model_best.pth"
if [ ! -f "$STAGE1_MODEL" ]; then
    echo "ERROR: Stage 1 model not found at: $STAGE1_MODEL"
    echo "Please run train_stage1_from_pretrained.sh first!"
    exit 1
fi

echo "Using Stage 1 model: $STAGE1_MODEL"
echo ""

python train.py \
    --experiment_name ${EXP_NAME} \
    --use_json_dataset \
    --train_json ${TRAIN_JSON} \
    --val_json ${VAL_JSON} \
    --training_stage 2 \
    --stage1_batch_size 32 \
    --stage1_epochs 30 \
    --stage1_learning_rate 0.00001 \
    --stage1_lr_decay_step 3 \
    --stage1_lr_decay_factor 0.8 \
    --WSGM_count 12 \
    --WSGM_reduction_factor 4 \
    --stage2_batch_size 16 \
    --stage2_epochs 10 \
    --stage2_learning_rate 0.000002 \
    --stage2_lr_decay_step 2 \
    --stage2_lr_decay_factor 0.7 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407 \
    --use_resize_only

echo ""
echo "=========================================="
echo "Stage 2 Training Complete!"
echo "Model saved to: ./check_points/${EXP_NAME}/train_stage_2/model/"
echo "Best model: model_best_val_loss.pth"
echo "=========================================="
