#!/bin/bash

# ForgeLens Multi-GPU Evaluation Script
# Evaluate trained model on test.json and test_nfs.json

EXP_NAME="deepfake_finetune_5k_multigpu"
TEST_JSON="/data/data/deepfake_finetune_dataset/jsons/test.json"
TEST_NFS_JSON="/data/data/deepfake_finetune_dataset/jsons/test_nfs.json"

# Model checkpoint (Stage 2 best model)
MODEL_PATH="./check_points/${EXP_NAME}/train_stage_2/model/model_best_val_loss.pth"

# Multi-GPU Configuration (optional for evaluation, usually single GPU is enough)
USE_MULTI_GPU=false  # Set to true if you want multi-GPU evaluation
GPU_IDS="0,1,2,3"

# Evaluation settings
BATCH_SIZE=32  # Can be larger for evaluation (no gradient computation)
EVAL_STAGE=2   # Evaluate Stage 2 model

echo "=========================================="
echo "ForgeLens Multi-GPU Evaluation"
echo "=========================================="
echo "Experiment: ${EXP_NAME}"
echo "Model: ${MODEL_PATH}"
echo "=========================================="
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at: $MODEL_PATH"
    echo ""
    echo "Please train the model first:"
    echo "  bash train_stage1_multigpu.sh"
    echo "  bash train_stage2_multigpu.sh"
    exit 1
fi

echo "Model found: $(du -h $MODEL_PATH | cut -f1)"
echo ""

# Function to run evaluation on a test set
evaluate_testset() {
    local TEST_FILE=$1
    local TEST_NAME=$2

    echo "=========================================="
    echo "Evaluating on: ${TEST_NAME}"
    echo "Test file: ${TEST_FILE}"
    echo "=========================================="

    if [ ! -f "$TEST_FILE" ]; then
        echo "⚠ Test file not found: $TEST_FILE"
        echo "Skipping..."
        echo ""
        return
    fi

    # Count samples in test file
    NUM_SAMPLES=$(python3 -c "import json; print(json.load(open('$TEST_FILE'))['metadata']['total_samples'])")
    echo "Total samples: ${NUM_SAMPLES}"
    echo ""

    # Build command
    CMD="python evaluate_json.py \
        --experiment_name ${EXP_NAME}_${TEST_NAME} \
        --test_json ${TEST_FILE} \
        --weights ${MODEL_PATH} \
        --eval_stage ${EVAL_STAGE} \
        --batch_size ${BATCH_SIZE} \
        --WSGM_count 12 \
        --WSGM_reduction_factor 4 \
        --FAFormer_layers 2 \
        --FAFormer_head 2 \
        --num_workers 8 \
        --seed 3407"

    # Add multi-GPU flag if enabled
    if [ "$USE_MULTI_GPU" = true ]; then
        CMD="$CMD --use_multi_gpu --gpu_ids ${GPU_IDS}"
        echo "Using Multi-GPU: ${GPU_IDS}"
    else
        echo "Using Single GPU: 0"
    fi

    echo ""
    echo "Running evaluation..."
    eval $CMD

    echo ""
    echo "✓ Evaluation complete for ${TEST_NAME}"
    echo ""
}

# Evaluate on test.json
evaluate_testset "${TEST_JSON}" "test"

# Evaluate on test_nfs.json
evaluate_testset "${TEST_NFS_JSON}" "test_nfs"

echo "=========================================="
echo "All Evaluations Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  ./check_points/${EXP_NAME}_test/evaluation_log.log"
echo "  ./check_points/${EXP_NAME}_test_nfs/evaluation_log.log"
echo ""
echo "To view results:"
echo "  cat ./check_points/${EXP_NAME}_test/evaluation_log.log"
echo "  cat ./check_points/${EXP_NAME}_test_nfs/evaluation_log.log"
echo "=========================================="
