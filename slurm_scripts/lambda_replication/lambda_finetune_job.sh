#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 1 of NT-v2 LAMBDA replication: finetune ONE (variant, seed).
# Submitted by run_lambda_training.sh. All paths/resources come via --export.
# This is the orchestration job body — it calls the EXISTING finetune_nt_phage.py
# entry point (it does NOT modify any model code).
#
# Required env:
#   REPO_ROOT          repo root (holds finetune_nt_phage.py)
#   REPL_OUTPUT_DIR    per-length replication output dir (outputs/<LEN>)
#   LAMBDA_DIR         train/val/test CSV directory (LAMBDA_v1 train_val_test/<LEN>)
#   VARIANT            nt_500m
#   SEED               integer
#   MAX_LENGTH         max token length for this window
#   BATCH_SIZE         per-device train batch size for this window
#   GRAD_ACCUM         gradient accumulation steps for this window
# Optional env (with defaults):
#   BASE_MODEL, LR, EVAL_BATCH_SIZE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
#   WEIGHT_DECAY, WARMUP_RATIO, EVAL_STRATEGY, EVAL_STEPS, METRIC_FOR_BEST_MODEL,
#   PRECISION_FLAGS, OPTIMIZER, CONDA_ENV (nt)


echo "=== finetune ${VARIANT} seed=${SEED} len=${LEN:-?} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

# Activate conda (bare style — no set -e; see PORTING_GUIDE gotcha #5).
module load CUDA/12.8
source /data/lindseylm/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-nt}"
if [ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV:-nt}" ]; then
    echo "ERROR: could not activate conda env '${CONDA_ENV:-nt}' (active: '${CONDA_DEFAULT_ENV:-none}'). Aborting." >&2
    exit 1
fi
echo "  conda env: ${CONDA_DEFAULT_ENV}   python: $(command -v python)"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Stay offline so the Biowulf HTTPS proxy can't 503 us mid-run. Cache must be
# pre-warmed from a login node (see lambda_replication/README.md).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-/data/lindseylm/.cache/huggingface}

# REPO_ROOT is supplied by the launcher via --export (the batch script is staged
# to the SLURM spool dir, so its own path can't be used to find the repo).
if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

BASE_MODEL=${BASE_MODEL:-InstaDeepAI/nucleotide-transformer-v2-500m-multi-species}
LR=${LR:-3e-5}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-10}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
EVAL_STRATEGY=${EVAL_STRATEGY:-steps}
EVAL_STEPS=${EVAL_STEPS:-500}
METRIC_FOR_BEST_MODEL=${METRIC_FOR_BEST_MODEL:-eval_mcc}
PRECISION_FLAGS=${PRECISION_FLAGS:---bf16 --tf32}
OPTIMIZER=${OPTIMIZER:-adamw_torch_fused}
MAX_LENGTH=${MAX_LENGTH:-512}

# finetune_nt_phage.py's loader reads {train,dev,test}.csv but LAMBDA_v1 ships
# val.csv. Stage a per-seed input dir that symlinks the CSVs and provides a
# dev.csv alias for val.csv. This avoids modifying finetune_nt_phage.py.
STAGE_DIR="${REPL_OUTPUT_DIR}/finetune/${VARIANT}/seed-${SEED}/_data"
mkdir -p "${STAGE_DIR}"
ln -sf "${LAMBDA_DIR}/train.csv" "${STAGE_DIR}/train.csv"
ln -sf "${LAMBDA_DIR}/test.csv"  "${STAGE_DIR}/test.csv"
if [ -f "${LAMBDA_DIR}/dev.csv" ]; then
    ln -sf "${LAMBDA_DIR}/dev.csv" "${STAGE_DIR}/dev.csv"
elif [ -f "${LAMBDA_DIR}/val.csv" ]; then
    ln -sf "${LAMBDA_DIR}/val.csv" "${STAGE_DIR}/dev.csv"
else
    echo "ERROR: neither dev.csv nor val.csv in ${LAMBDA_DIR}"; exit 1
fi
DATASET_DIR="${STAGE_DIR}"

OUTPUT_DIR="${REPL_OUTPUT_DIR}/finetune/${VARIANT}/seed-${SEED}"
mkdir -p "${OUTPUT_DIR}"

echo "  base model:   ${BASE_MODEL}"
echo "  dataset dir:  ${DATASET_DIR}  (from ${LAMBDA_DIR})"
echo "  output:       ${OUTPUT_DIR}"
echo "  lr=${LR}  batch=${BATCH_SIZE}  grad_accum=${GRAD_ACCUM}  max_length=${MAX_LENGTH}  epochs=${NUM_EPOCHS}"
echo "  precision=${PRECISION_FLAGS}  optim=${OPTIMIZER}  metric_for_best=${METRIC_FOR_BEST_MODEL}"

# finetune_nt_phage.py writes test-set metrics to ${OUTPUT_DIR}/test_results.json
# (key eval_mcc), which select_best_model.py reads directly — no copy needed.
python finetune_nt_phage.py \
    --model_name "${BASE_MODEL}" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length ${MAX_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --num_train_epochs ${NUM_EPOCHS} \
    --eval_strategy ${EVAL_STRATEGY} \
    --eval_steps ${EVAL_STEPS} \
    --save_strategy ${EVAL_STRATEGY} \
    --save_steps ${EVAL_STEPS} \
    --load_best_model_at_end \
    --metric_for_best_model ${METRIC_FOR_BEST_MODEL} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --save_total_limit 2 \
    --optim ${OPTIMIZER} \
    ${PRECISION_FLAGS} \
    --seed ${SEED}

if [ -f "${OUTPUT_DIR}/test_results.json" ]; then
    echo "  wrote ${OUTPUT_DIR}/test_results.json"
else
    echo "  WARNING: ${OUTPUT_DIR}/test_results.json not found — training/eval may have failed"
fi

echo "Done: $(date)"
