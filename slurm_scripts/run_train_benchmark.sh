#!/bin/bash
#SBATCH --job-name=nt_benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=nt_benchmark_%j.out
#SBATCH --error=nt_benchmark_%j.err

# Benchmark script to compare baseline (fp32) vs bf16 training
#
# Usage: sbatch run_train_benchmark.sh
#
# This runs two training configurations on a small dataset:
#   1. Baseline (fp32) - no mixed precision
#   2. bf16 - recommended for A100 GPUs
#
# Results are saved to output/benchmark/ with timing information

echo "============================================================"
echo "Nucleotide Transformer v2 Training Benchmark"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load conda
module load CUDA/12.8

# Activate conda environment
source activate nt

# Ignore user site-packages to avoid conflicts
export PYTHONNOUSERSITE=1

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi

echo ""
echo "Python environment:"
which python
python --version

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

# Small benchmark dataset
DATASET_DIR="/home/lindseylm/lindseylm/lambda_final/merged_datasets_filtered/small"

# Training parameters (same for both runs for fair comparison)
SEED=42
LEARNING_RATE=3e-5
BATCH_SIZE=8
EPOCHS=1  # Just 1 epoch for benchmarking
MAX_LENGTH=2048

# Base output directory
OUTPUT_BASE="/data/lindseylm/GLM_EVALUATIONS/MODELS/NTv2/NTv2_generic_sequence_classification/output/benchmark"
SCRIPT_DIR="/data/lindseylm/GLM_EVALUATIONS/MODELS/NTv2/NTv2_generic_sequence_classification"

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Results file
RESULTS_FILE="${OUTPUT_BASE}/benchmark_results_$(date +%Y%m%d_%H%M%S).txt"

echo ""
echo "============================================================"
echo "Benchmark Configuration"
echo "============================================================"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_DIR"
echo "  Output base: $OUTPUT_BASE"
echo "  Max length: $MAX_LENGTH"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Seed: $SEED"
echo "============================================================"
echo ""

# ============================================================
# TEST 1: Baseline (fp32)
# ============================================================

echo "============================================================" | tee -a "${RESULTS_FILE}"
echo "TEST 1: Baseline (fp32)" | tee -a "${RESULTS_FILE}"
echo "============================================================" | tee -a "${RESULTS_FILE}"

OUTPUT_DIR_BASELINE="${OUTPUT_BASE}/baseline_fp32_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR_BASELINE}"

echo "Output: ${OUTPUT_DIR_BASELINE}" | tee -a "${RESULTS_FILE}"
echo "Started at: $(date)" | tee -a "${RESULTS_FILE}"

START_TIME=$(date +%s)

python "${SCRIPT_DIR}/finetune_nt_phage.py" \
    --model_name "$MODEL_NAME" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR_BASELINE" \
    --max_length $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 16 \
    --num_train_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --early_stopping_patience 0 \
    --seed $SEED

EXIT_CODE_BASELINE=$?
END_TIME=$(date +%s)
ELAPSED_BASELINE=$((END_TIME - START_TIME))

echo "Finished at: $(date)" | tee -a "${RESULTS_FILE}"
echo "Exit code: ${EXIT_CODE_BASELINE}" | tee -a "${RESULTS_FILE}"
echo "Time: ${ELAPSED_BASELINE} seconds ($(echo "scale=1; ${ELAPSED_BASELINE}/60" | bc) minutes)" | tee -a "${RESULTS_FILE}"

# Extract peak memory from training summary if available
if [ -f "${OUTPUT_DIR_BASELINE}/training_summary.json" ]; then
    MEMORY_BASELINE=$(python3 -c "import json; d=json.load(open('${OUTPUT_DIR_BASELINE}/training_summary.json')); print(f\"{d.get('peak_gpu_memory_mb', 0):.0f}\")")
    echo "Peak GPU memory: ${MEMORY_BASELINE} MB" | tee -a "${RESULTS_FILE}"
fi

echo "" | tee -a "${RESULTS_FILE}"

# ============================================================
# TEST 2: bf16 (recommended for A100)
# ============================================================

echo "============================================================" | tee -a "${RESULTS_FILE}"
echo "TEST 2: bf16 (recommended for A100)" | tee -a "${RESULTS_FILE}"
echo "============================================================" | tee -a "${RESULTS_FILE}"

OUTPUT_DIR_BF16="${OUTPUT_BASE}/bf16_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR_BF16}"

echo "Output: ${OUTPUT_DIR_BF16}" | tee -a "${RESULTS_FILE}"
echo "Started at: $(date)" | tee -a "${RESULTS_FILE}"

START_TIME=$(date +%s)

python "${SCRIPT_DIR}/finetune_nt_phage.py" \
    --model_name "$MODEL_NAME" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR_BF16" \
    --max_length $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 16 \
    --num_train_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --early_stopping_patience 0 \
    --bf16 \
    --seed $SEED

EXIT_CODE_BF16=$?
END_TIME=$(date +%s)
ELAPSED_BF16=$((END_TIME - START_TIME))

echo "Finished at: $(date)" | tee -a "${RESULTS_FILE}"
echo "Exit code: ${EXIT_CODE_BF16}" | tee -a "${RESULTS_FILE}"
echo "Time: ${ELAPSED_BF16} seconds ($(echo "scale=1; ${ELAPSED_BF16}/60" | bc) minutes)" | tee -a "${RESULTS_FILE}"

# Extract peak memory from training summary if available
if [ -f "${OUTPUT_DIR_BF16}/training_summary.json" ]; then
    MEMORY_BF16=$(python3 -c "import json; d=json.load(open('${OUTPUT_DIR_BF16}/training_summary.json')); print(f\"{d.get('peak_gpu_memory_mb', 0):.0f}\")")
    echo "Peak GPU memory: ${MEMORY_BF16} MB" | tee -a "${RESULTS_FILE}"
fi

echo "" | tee -a "${RESULTS_FILE}"

# ============================================================
# TEST 3: bf16 + step-based eval + early stopping
# ============================================================

echo "============================================================" | tee -a "${RESULTS_FILE}"
echo "TEST 3: bf16 + step-based eval + early stopping" | tee -a "${RESULTS_FILE}"
echo "============================================================" | tee -a "${RESULTS_FILE}"

OUTPUT_DIR_EARLY="${OUTPUT_BASE}/bf16_early_stop_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR_EARLY}"

echo "Output: ${OUTPUT_DIR_EARLY}" | tee -a "${RESULTS_FILE}"
echo "Started at: $(date)" | tee -a "${RESULTS_FILE}"
echo "Config: 10 epochs max, eval every 100 steps, patience=3" | tee -a "${RESULTS_FILE}"

START_TIME=$(date +%s)

python "${SCRIPT_DIR}/finetune_nt_phage.py" \
    --model_name "$MODEL_NAME" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR_EARLY" \
    --max_length $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 10 \
    --learning_rate $LEARNING_RATE \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --early_stopping_patience 3 \
    --bf16 \
    --seed $SEED

EXIT_CODE_EARLY=$?
END_TIME=$(date +%s)
ELAPSED_EARLY=$((END_TIME - START_TIME))

echo "Finished at: $(date)" | tee -a "${RESULTS_FILE}"
echo "Exit code: ${EXIT_CODE_EARLY}" | tee -a "${RESULTS_FILE}"
echo "Time: ${ELAPSED_EARLY} seconds ($(echo "scale=1; ${ELAPSED_EARLY}/60" | bc) minutes)" | tee -a "${RESULTS_FILE}"

# Check how many epochs actually ran (early stopping test)
if [ -f "${OUTPUT_DIR_EARLY}/training_summary.json" ]; then
    ACTUAL_EPOCHS=$(python3 -c "import json; d=json.load(open('${OUTPUT_DIR_EARLY}/training_summary.json')); print(f\"{d.get('actual_epochs', 'N/A')}\")")
    MEMORY_EARLY=$(python3 -c "import json; d=json.load(open('${OUTPUT_DIR_EARLY}/training_summary.json')); print(f\"{d.get('peak_gpu_memory_mb', 0):.0f}\")")
    echo "Actual epochs completed: ${ACTUAL_EPOCHS} (max was 10)" | tee -a "${RESULTS_FILE}"
    echo "Peak GPU memory: ${MEMORY_EARLY} MB" | tee -a "${RESULTS_FILE}"

    if [ "${ACTUAL_EPOCHS}" != "10" ] && [ "${ACTUAL_EPOCHS}" != "N/A" ]; then
        echo "*** EARLY STOPPING TRIGGERED! ***" | tee -a "${RESULTS_FILE}"
    fi
fi

echo "" | tee -a "${RESULTS_FILE}"

# ============================================================
# SUMMARY
# ============================================================

echo "============================================================" | tee -a "${RESULTS_FILE}"
echo "BENCHMARK SUMMARY" | tee -a "${RESULTS_FILE}"
echo "============================================================" | tee -a "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"
echo "Test 1 & 2: ${EPOCHS} epoch, batch_size=${BATCH_SIZE}, max_length=${MAX_LENGTH}" | tee -a "${RESULTS_FILE}"
echo "Test 3: 10 epochs max, eval every 100 steps, early_stopping_patience=3" | tee -a "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"
echo "Timing Results:" | tee -a "${RESULTS_FILE}"
echo "  1. Baseline (fp32):           ${ELAPSED_BASELINE} seconds ($(echo "scale=1; ${ELAPSED_BASELINE}/60" | bc) min)" | tee -a "${RESULTS_FILE}"
echo "  2. bf16:                      ${ELAPSED_BF16} seconds ($(echo "scale=1; ${ELAPSED_BF16}/60" | bc) min)" | tee -a "${RESULTS_FILE}"
echo "  3. bf16 + early stopping:     ${ELAPSED_EARLY} seconds ($(echo "scale=1; ${ELAPSED_EARLY}/60" | bc) min)" | tee -a "${RESULTS_FILE}"

# Calculate speedup
if [ ${ELAPSED_BF16} -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; ${ELAPSED_BASELINE} / ${ELAPSED_BF16}" | bc)
    echo "" | tee -a "${RESULTS_FILE}"
    echo "Speedup (bf16 vs fp32): ${SPEEDUP}x" | tee -a "${RESULTS_FILE}"
fi

echo "" | tee -a "${RESULTS_FILE}"
echo "Early Stopping Test:" | tee -a "${RESULTS_FILE}"
if [ -f "${OUTPUT_DIR_EARLY}/training_summary.json" ]; then
    echo "  Epochs completed: ${ACTUAL_EPOCHS} / 10 max" | tee -a "${RESULTS_FILE}"
    if [ "${ACTUAL_EPOCHS}" != "10" ] && [ "${ACTUAL_EPOCHS}" != "N/A" ]; then
        echo "  Status: WORKING - stopped early!" | tee -a "${RESULTS_FILE}"
    else
        echo "  Status: Did not trigger (model may need more epochs to converge)" | tee -a "${RESULTS_FILE}"
    fi
else
    echo "  Status: Could not determine (training_summary.json not found)" | tee -a "${RESULTS_FILE}"
fi

echo "" | tee -a "${RESULTS_FILE}"
echo "Results saved to: ${RESULTS_FILE}" | tee -a "${RESULTS_FILE}"
echo "============================================================" | tee -a "${RESULTS_FILE}"

echo ""
echo "Job finished at: $(date)"

exit 0
