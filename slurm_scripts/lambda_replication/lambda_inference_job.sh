#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Inference for the winning seed of VARIANT on one CSV. Reads
# <REPL_OUTPUT_DIR>/winners.json to find the winning finetune checkpoint dir,
# then runs the EXISTING inference_nt.py entry point (no model code is modified).
#
# Required env:
#   REPO_ROOT
#   REPL_OUTPUT_DIR
#   VARIANT
#   INPUT_CSV          path to the CSV to predict on
#   OUTPUT_FILENAME    name for the predictions CSV (e.g. test_predictions.csv)
#   MAX_LENGTH         max token length for this window
# Optional env:
#   BATCH_SIZE (16), THRESHOLD (0.5), CONDA_ENV (nt)


echo "=== inference ${VARIANT}  input=${INPUT_CSV}  output=${OUTPUT_FILENAME} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

# --- conda env setup: BIOWULF ONLY, disabled for Delta ---------------------
# Delta-AI inherits the submitting shell's environment (sbatch --export=ALL), so
# `conda activate nt` happens on the LOGIN node BEFORE the driver runs.
# module load CUDA/12.8
# source /data/lindseylm/conda/etc/profile.d/conda.sh
# conda activate "${CONDA_ENV:-nt}"
# if [ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV:-nt}" ]; then
#     echo "ERROR: could not activate conda env '${CONDA_ENV:-nt}' (active: '${CONDA_DEFAULT_ENV:-none}'). Aborting." >&2
#     exit 1
# fi
echo "  conda env: ${CONDA_DEFAULT_ENV}   python: $(command -v python)"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-/work/hdd/bfzj/llindsey1/hf_cache}

if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-512}
THRESHOLD=${THRESHOLD:-0.5}

# Locate the lambda_replication dir via the exported REPO_ROOT.
SCRIPT_DIR="$(dirname "$(find "${REPO_ROOT}" -path '*lambda_replication/print_winner_exports.py' 2>/dev/null | head -1)")"
WINNERS_JSON="${REPL_OUTPUT_DIR}/winners.json"
if [ ! -f "${WINNERS_JSON}" ]; then
    echo "ERROR: ${WINNERS_JSON} not found (select_best_model must run first)"; exit 1
fi

# print_winner_exports.py emits shlex-quoted exports (WINNER_PATH, WINNER_SEED,
# BASE_MODEL, WINNER_TYPE) read from winners.json[VARIANT].
eval "$(python "${SCRIPT_DIR}/print_winner_exports.py" "${WINNERS_JSON}" "${VARIANT}")"

echo "  winner seed:   ${WINNER_SEED}"
echo "  winner path:   ${WINNER_PATH}"
echo "  base model:    ${BASE_MODEL}"

OUTPUT_DIR="${REPL_OUTPUT_DIR}/inference/${VARIANT}"
mkdir -p "${OUTPUT_DIR}"

# inference_nt.py writes <output_csv stem>_metrics.json next to the predictions
# CSV when --save_metrics and labels are present.
python inference_nt.py \
    --input_csv "${INPUT_CSV}" \
    --model_path "${WINNER_PATH}" \
    --output_csv "${OUTPUT_DIR}/${OUTPUT_FILENAME}" \
    --max_length ${MAX_LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --threshold ${THRESHOLD} \
    --save_metrics

echo "Done: $(date)"
