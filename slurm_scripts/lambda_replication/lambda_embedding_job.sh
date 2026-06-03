#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Pretrained-embedding analysis (Surface D) for one (length, variant). Runs the
# EXISTING embedding_analysis_nt.py entry point on the BASE_MODEL (pretrained,
# not the finetuned checkpoint) — no model code is modified.
#
# Required env:
#   REPO_ROOT
#   REPL_OUTPUT_DIR    per-length replication output dir (outputs/<LEN>)
#   LAMBDA_DIR         train/val/test CSV directory (staged dev.csv alias)
#   VARIANT            nt_500m
#   MAX_LENGTH         max token length for this window
# Optional env:
#   BASE_MODEL, POOLING, EMB_SEED, NN_EPOCHS, NN_HIDDEN_DIM, NN_LR, BATCH_SIZE,
#   INCLUDE_RANDOM_BASELINE, CONDA_ENV (nt)


echo "=== embedding ${VARIANT} len=${LEN:-?} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

module load CUDA/12.8
source /data/lindseylm/conda/etc/profile.d/conda.sh
if [ -z "${CUDA_HOME}" ]; then
    NVCC_PATH=$(which nvcc)
    if [ -n "${NVCC_PATH}" ]; then
        export CUDA_HOME=$(dirname $(dirname "${NVCC_PATH}"))
    fi
fi
conda activate "${CONDA_ENV:-nt}"
if [ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV:-nt}" ]; then
    echo "ERROR: could not activate conda env '${CONDA_ENV:-nt}' (active: '${CONDA_DEFAULT_ENV:-none}'). Aborting." >&2
    exit 1
fi
echo "  conda env: ${CONDA_DEFAULT_ENV}   python: $(command -v python)"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-/data/lindseylm/.cache/huggingface}

if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

BASE_MODEL=${BASE_MODEL:-InstaDeepAI/nucleotide-transformer-v2-500m-multi-species}
POOLING=${POOLING:-mean}
EMB_SEED=${EMB_SEED:-42}
NN_EPOCHS=${NN_EPOCHS:-100}
NN_HIDDEN_DIM=${NN_HIDDEN_DIM:-256}
NN_LR=${NN_LR:-0.001}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-512}
INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-false}

# embedding_analysis_nt.py reads {train,dev,test}.csv from --csv_dir. Stage a
# dir with a dev.csv alias for LAMBDA_v1's val.csv (no model code changed).
STAGE_DIR="${REPL_OUTPUT_DIR}/embedding/${VARIANT}/_data"
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
CSV_DIR="${STAGE_DIR}"

OUTPUT_DIR="${REPL_OUTPUT_DIR}/embedding/${VARIANT}"
mkdir -p "${OUTPUT_DIR}"

echo "  base model:   ${BASE_MODEL}"
echo "  csv dir:      ${CSV_DIR}  (from ${LAMBDA_DIR})"
echo "  output:       ${OUTPUT_DIR}"
echo "  pooling=${POOLING}  max_length=${MAX_LENGTH}  nn_epochs=${NN_EPOCHS}  random_baseline=${INCLUDE_RANDOM_BASELINE}"

RANDOM_BASELINE_FLAG=""
if [ "${INCLUDE_RANDOM_BASELINE}" == "true" ]; then
    RANDOM_BASELINE_FLAG="--include_random_baseline"
fi

# Writes embedding_analysis_results.json to OUTPUT_DIR.
python embedding_analysis_nt.py \
    --csv_dir "${CSV_DIR}" \
    --model_path "${BASE_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --pooling "${POOLING}" \
    --seed ${EMB_SEED} \
    --nn_epochs ${NN_EPOCHS} \
    --nn_hidden_dim ${NN_HIDDEN_DIM} \
    --nn_lr ${NN_LR} \
    ${RANDOM_BASELINE_FLAG}

echo "Done: $(date)"
