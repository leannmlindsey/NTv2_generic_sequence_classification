#!/bin/bash

# Optimized Training Wrapper for NT-v2
#
# This wrapper provides recommended configurations for different
# sequence lengths and hardware.
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_training_optimized.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Dataset Directory ===
# Directory containing train.csv, dev.csv, test.csv
DATASET_DIR="/path/to/your/data"

# === REQUIRED: Output Directory ===
OUTPUT_DIR="/path/to/output"

# === Sequence Length Configuration ===
# Options: 2048 (2k), 4096 (4k), or 8192 (8k)
MAX_LENGTH="2048"

# === Hardware Configuration ===
# Set based on your GPU:
#   - "A100" or "H100": Use bf16, larger batches
#   - "V100" or "older": Use fp16, smaller batches
GPU_TYPE="A100"

# === Training Configuration ===
LEARNING_RATE="3e-5"
NUM_EPOCHS="10"  # Use more epochs with early stopping
EARLY_STOPPING_PATIENCE="3"
SEED="42"

# === Evaluation Configuration ===
# "steps" enables early stopping within epochs
# "epoch" only evaluates at end of each epoch
EVAL_STRATEGY="steps"
EVAL_STEPS="500"

#####################################################################
# AUTO-CONFIGURATION (based on settings above)
#####################################################################

SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/NTv2/NTv2_generic_sequence_classification"

# Set precision based on GPU type
if [ "${GPU_TYPE}" == "A100" ] || [ "${GPU_TYPE}" == "H100" ]; then
    PRECISION_FLAGS="--bf16 --tf32"
    OPTIMIZER="adamw_torch_fused"
else
    PRECISION_FLAGS="--fp16"
    OPTIMIZER="adamw_torch"
fi

# Set batch size based on sequence length and GPU
case "${MAX_LENGTH}" in
    2048)
        if [ "${GPU_TYPE}" == "A100" ] || [ "${GPU_TYPE}" == "H100" ]; then
            BATCH_SIZE="16"
            GRAD_ACCUM="1"
            GRAD_CHECKPOINT=""
        else
            BATCH_SIZE="8"
            GRAD_ACCUM="2"
            GRAD_CHECKPOINT=""
        fi
        ;;
    4096)
        if [ "${GPU_TYPE}" == "A100" ] || [ "${GPU_TYPE}" == "H100" ]; then
            BATCH_SIZE="8"
            GRAD_ACCUM="2"
            GRAD_CHECKPOINT="--gradient_checkpointing"
        else
            BATCH_SIZE="4"
            GRAD_ACCUM="4"
            GRAD_CHECKPOINT="--gradient_checkpointing"
        fi
        ;;
    8192)
        if [ "${GPU_TYPE}" == "A100" ] || [ "${GPU_TYPE}" == "H100" ]; then
            BATCH_SIZE="4"
            GRAD_ACCUM="4"
            GRAD_CHECKPOINT="--gradient_checkpointing"
        else
            BATCH_SIZE="2"
            GRAD_ACCUM="8"
            GRAD_CHECKPOINT="--gradient_checkpointing"
        fi
        ;;
    *)
        echo "ERROR: Unsupported MAX_LENGTH: ${MAX_LENGTH}"
        echo "Supported values: 2048, 4096, 8192"
        exit 1
        ;;
esac

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))

#####################################################################
# VALIDATION
#####################################################################

if [ ! -d "${DATASET_DIR}" ]; then
    echo "ERROR: Dataset directory not found: ${DATASET_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

#####################################################################
# PRINT CONFIGURATION
#####################################################################

echo "=========================================="
echo "NT-v2 Optimized Training"
echo "=========================================="
echo ""
echo "Dataset: ${DATASET_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Sequence length: ${MAX_LENGTH}"
echo "GPU type: ${GPU_TYPE}"
echo ""
echo "Training configuration:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRAD_ACCUM}"
echo "  Effective batch size: ${EFFECTIVE_BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Max epochs: ${NUM_EPOCHS}"
echo "  Early stopping patience: ${EARLY_STOPPING_PATIENCE}"
echo ""
echo "Optimizations:"
echo "  Precision: ${PRECISION_FLAGS}"
echo "  Optimizer: ${OPTIMIZER}"
echo "  Gradient checkpointing: ${GRAD_CHECKPOINT:-disabled}"
echo ""
echo "Evaluation:"
echo "  Strategy: ${EVAL_STRATEGY}"
if [ "${EVAL_STRATEGY}" == "steps" ]; then
    echo "  Eval steps: ${EVAL_STEPS}"
fi
echo "=========================================="
echo ""

#####################################################################
# RUN TRAINING
#####################################################################

cd "${SCRIPT_DIR}"

python finetune_nt_phage.py \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length ${MAX_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --eval_strategy ${EVAL_STRATEGY} \
    --eval_steps ${EVAL_STEPS} \
    --save_strategy ${EVAL_STRATEGY} \
    --save_steps ${EVAL_STEPS} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --optim ${OPTIMIZER} \
    --seed ${SEED} \
    ${PRECISION_FLAGS} \
    ${GRAD_CHECKPOINT}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Training completed successfully"
    echo "Model saved to: ${OUTPUT_DIR}"
else
    echo "Training failed with exit code: ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
