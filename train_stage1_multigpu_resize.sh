#!/bin/bash

# ForgeLens Stage 1 Fine-tuning Script
# Fine-tune from ForgeLens pretrained weights (ProGAN trained)
# Using 5K custom deepfake dataset

EXP_NAME="deepfake_finetune_5k_from_pretrained"
TRAIN_JSON="/data/data/deepfake_finetune_dataset/jsons/train_small_5000.json"
VAL_JSON="/data/data/deepfake_finetune_dataset/jsons/val.json"

# ForgeLens pretrained weight path
# Download from: https://drive.google.com/file/d/1JxfFqVrX50U5FFR_Wm1BGYVtIi-IX_sH/view?usp=sharing
PRETRAINED_PATH="./pretrained_weights/training_setting_1/train_stage_1/model/intermediate_model_best.pth"

echo "=========================================="
echo "ForgeLens Stage 1 Fine-tuning"
echo "=========================================="
echo "Experiment: ${EXP_NAME}"
echo "Train data: ${TRAIN_JSON}"
echo "Val data: ${VAL_JSON}"
echo "Pretrained: ${PRETRAINED_PATH}"
echo "=========================================="
echo ""

# Check if pretrained weight exists
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "ERROR: Pretrained weight not found at: $PRETRAINED_PATH"
    echo ""
    echo "Please download ForgeLens pretrained weights:"
    echo "  Google Drive: https://drive.google.com/file/d/1JxfFqVrX50U5FFR_Wm1BGYVtIi-IX_sH/view?usp=sharing"
    echo ""
    echo "Extract and place at: ./pretrained_weights/training_setting_1/"
    echo ""
    echo "Or train from scratch using: bash train_stage1.sh"
    exit 1
fi

python train.py \
    --experiment_name ${EXP_NAME} \
    --use_json_dataset \
    --train_json ${TRAIN_JSON} \
    --val_json ${VAL_JSON} \
    --training_stage 1 \
    --pretrained_stage1_path ${PRETRAINED_PATH} \
    --stage1_batch_size 32 \
    --stage1_epochs 30 \
    --stage1_learning_rate 0.00001 \
    --stage1_lr_decay_step 3 \
    --stage1_lr_decay_factor 0.8 \
    --WSGM_count 12 \
    --WSGM_reduction_factor 4 \
    --num_workers 4 \
    --seed 3407 \
    --use_resize_only

echo ""
echo "=========================================="
echo "Stage 1 Fine-tuning Complete!"
echo "Model saved to: ./check_points/${EXP_NAME}/train_stage_1/model/"
echo "Best model: intermediate_model_best.pth"
echo ""
echo "Next step: Run Stage 2 training"
echo "  bash train_stage2_from_pretrained.sh"
echo "=========================================="
