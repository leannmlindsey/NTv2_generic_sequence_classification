#!/bin/bash

# Wrapper script for running NT-v2 embedding analysis on Biowulf
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_embedding_analysis.sh
#
# Or submit directly with environment variables:
#   sbatch --export=ALL,CSV_DIR=/path/to/data,MODEL_PATH=/path/to/model run_embedding_analysis.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Dataset Configuration ===
# Path to directory containing train.csv, dev.csv (or val.csv), test.csv
export CSV_DIR="/home/lindseylm/lindseylm/lambda_final/merged_datasets_filtered/2k"

# === OPTIONAL: Model Configuration ===
# Path to fine-tuned model or HuggingFace model name
# Available NT-v2 models:
#   - InstaDeepAI/nucleotide-transformer-v2-50m-multi-species   (smallest)
#   - InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
#   - InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
#   - InstaDeepAI/nucleotide-transformer-v2-500m-multi-species  (largest)
export MODEL_PATH="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

# === OPTIONAL: Output Directory ===
# Leave empty to use default: ./results/embedding_analysis/$(basename $CSV_DIR)
export OUTPUT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/FINAL_RESULTS/NTv2/embedding_analysis/2k"

# === OPTIONAL: Hyperparameters ===
export BATCH_SIZE="16"
export MAX_LENGTH="2048"           # NT-v2 supports up to 2048 tokens (~12kb)
export POOLING="mean"              # Options: mean, cls, last
export SEED="42"

# === OPTIONAL: 3-Layer NN Parameters ===
export NN_EPOCHS="10"
export NN_HIDDEN_DIM="256"
export NN_LR="0.001"

# === OPTIONAL: Include Random Baseline ===
# Set to "true" to also run analysis on randomly initialized model for comparison
export INCLUDE_RANDOM_BASELINE="true"

#####################################################################
# END CONFIGURATION
#####################################################################

# Validate configuration
if [ "${CSV_DIR}" == "/path/to/your/csv/data" ]; then
    echo "ERROR: Please set CSV_DIR to your actual data directory"
    exit 1
fi

# Verify files exist
if [ ! -d "${CSV_DIR}" ]; then
    echo "ERROR: CSV_DIR does not exist: ${CSV_DIR}"
    exit 1
fi

if [ ! -f "${CSV_DIR}/train.csv" ]; then
    echo "ERROR: train.csv not found in ${CSV_DIR}"
    exit 1
fi

if [ ! -f "${CSV_DIR}/test.csv" ]; then
    echo "ERROR: test.csv not found in ${CSV_DIR}"
    exit 1
fi

# Check for dev.csv or val.csv
if [ ! -f "${CSV_DIR}/dev.csv" ] && [ ! -f "${CSV_DIR}/val.csv" ]; then
    echo "ERROR: Neither dev.csv nor val.csv found in ${CSV_DIR}"
    exit 1
fi

# Get dataset name for job naming
DATASET_NAME=$(basename "${CSV_DIR}")

# Set default output directory if not specified
if [ -z "${OUTPUT_DIR}" ]; then
    export OUTPUT_DIR="./results/embedding_analysis/${DATASET_NAME}"
fi

echo "=========================================="
echo "Submitting NT-v2 Embedding Analysis Job"
echo "=========================================="
echo "Dataset: ${DATASET_NAME}"
echo "CSV dir: ${CSV_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""
echo "Parameters:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Pooling: ${POOLING}"
echo "  Seed: ${SEED}"
echo ""
echo "3-Layer NN:"
echo "  Epochs: ${NN_EPOCHS}"
echo "  Hidden dim: ${NN_HIDDEN_DIM}"
echo "  Learning rate: ${NN_LR}"
echo ""
echo "Include random baseline: ${INCLUDE_RANDOM_BASELINE}"
echo "=========================================="

# Get script directory
#SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/NTv2/NTv2_generic_sequence_classification/slurm_scripts"
cd $SCRIPT_DIR
# Submit job
echo "Submitting job..."
sbatch --export=ALL \
    --job-name="nt_emb_${DATASET_NAME}" \
    "${SCRIPT_DIR}/run_embedding_analysis.sh"

echo ""
echo "Job submitted. Monitor with: squeue -u \$USER"
