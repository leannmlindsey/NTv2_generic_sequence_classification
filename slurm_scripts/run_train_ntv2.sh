#!/bin/bash
#SBATCH --job-name=nt_phage
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=nt_phage_%j.out
#SBATCH --error=nt_phage_%j.err

# Biowulf batch script for Nucleotide Transformer v2 finetuning on phage detection
# Usage: sbatch run_nt_finetune.sh [SEED]
# Example: sbatch run_nt_finetune.sh 42

echo "============================================================"
echo "Nucleotide Transformer v2 Fine-tuning"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load conda
module load CUDA/12.8

# Activate conda environment
source activate nt

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
# CONFIGURATION - MODIFY THESE AS NEEDED
# ============================================================

# Model - NT-v2 variants available:
# - InstaDeepAI/nucleotide-transformer-v2-50m-multi-species   (smallest, fastest)
# - InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
# - InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
# - InstaDeepAI/nucleotide-transformer-v2-500m-multi-species  (largest NT-v2)
MODEL_NAME="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

# Dataset directory (should contain train.csv, dev.csv, test.csv)
# Each CSV should have columns: sequence, label
DATASET_DIR="/home/lindseylm/lindseylm/lambda_final/merged_datasets_filtered/2k"

# Training parameters
SEED=${1:-42}  # Use first argument as seed, default to 42
LEARNING_RATE=3e-5
BATCH_SIZE=4  # Adjust based on GPU memory and sequence length
EPOCHS=3
MAX_LENGTH=2048  # In tokens (~6kb of sequence for NT-v2)
                 # NT-v2 supports up to 2048 tokens (~12kb)
                 # Reduce if you get OOM errors
f="filtered"
len="2k"
# Output directory
OUTPUT_DIR="./output/${f}/${len}/nt_lambda_${f}_${len}_${SEED}_${LEARNING_RATE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# ============================================================
# Print configuration
# ============================================================
echo ""
echo "Training configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Max length (tokens): $MAX_LENGTH"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Seed: $SEED"
echo ""

# ============================================================
# Run training
# ============================================================
# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

python finetune_nt_phage.py \
    --model_name "$MODEL_NAME" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_length $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --logging_steps 100 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model eval_mcc \
    --early_stopping_patience 3 \
    --save_total_limit 2 \
    --fp16 \
    --seed $SEED

EXIT_CODE=$?

# ============================================================
# Finish
# ============================================================
echo ""
echo "============================================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
