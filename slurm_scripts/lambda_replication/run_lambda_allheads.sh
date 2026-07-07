#!/bin/bash
#
# NTv2 — genome-wide predictions for BOTH frozen-embedding probe heads (linear
# probe + 3-layer NN) across all genome-wide CSVs. Fills the missing LP + NN
# heads (FT genome-wide already exists) in ONE embedding pass per CSV, for the
# best-of-{LP,NN,FT} genome-wide MCC in the main table + per-head appendix.
#
# Reuses lambda_replication.conf for paths/env/resources. Submits one
# lambda_allheads_job.sh per (LEN, variant, genome CSV). Requires saved probe
# artifacts in OUTPUT_DIR/<LEN>/embedding/<variant>.
#
# Usage (login node, `conda activate nt` first, repo pulled):
#   bash slurm_scripts/lambda_replication/run_lambda_allheads.sh [LEN ...]
# LEN defaults to SEGMENT_LENGTHS from the conf.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"
JOB="${SCRIPT_DIR}/lambda_allheads_job.sh"
[ -f "${CONFIG}" ] || { echo "ERROR: missing ${CONFIG}"; exit 1; }
[ -f "${JOB}" ]    || { echo "ERROR: missing ${JOB}"; exit 1; }
# shellcheck disable=SC1090
source "${CONFIG}"

LENS=("$@"); [ "${#LENS[@]}" -gt 0 ] || read -ra LENS <<< "${SEGMENT_LENGTHS}"
BATCH="${INF_BATCH_SIZE:-${BATCH_SIZE:-16}}"

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"
FLAGS=(--account=bfzj-dtai-gh --partition=ghx4 --gpus-per-node=1 --mem="${INF_MEM}" --time="${INF_TIME}" --cpus-per-task=8)

echo "============================================================"
echo "NTv2 — all-heads (LP + NN) genome-wide"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}   LENGTHS: ${LENS[*]}   VARIANTS: ${VARIANTS}"
echo "============================================================"

NUM=0
for LEN in "${LENS[@]}"; do
    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"
    ml_var="MAX_LENGTH_${LEN}"; MAX_LENGTH="${!ml_var:-2048}"
    gw_var="GENOME_WIDE_${LEN}"; GW_PATH="${!gw_var:-}"
    if [ -z "${GW_PATH}" ] || [ ! -d "${GW_PATH}" ]; then
        echo "WARNING: no genome-wide dir for ${LEN} (${GW_PATH:-unset}) — skipping"; continue
    fi
    for VARIANT in ${VARIANTS}; do
        EMB_DIR="${REPL_LEN_DIR}/embedding/${VARIANT}"
        if [ ! -f "${EMB_DIR}/linear_probe_pretrained.pkl" ]; then
            echo "WARNING: no saved LP probe in ${EMB_DIR} — run embedding analysis first; skipping ${LEN}/${VARIANT}"; continue
        fi
        shopt -s nullglob; gw_csvs=("${GW_PATH}"/*.csv); shopt -u nullglob
        [ "${#gw_csvs[@]}" -gt 0 ] || { echo "WARNING: ${GW_PATH} has no *.csv — skipping ${LEN}/${VARIANT}"; continue; }
        echo "--- ${LEN}/${VARIANT}: ${#gw_csvs[@]} genome CSV(s)  max_length=${MAX_LENGTH} ---"
        for csv in "${gw_csvs[@]}"; do
            stem="$(basename "${csv}" .csv)"; J="gwheads_${LEN}_${VARIANT}_${stem}"
            sbatch --job-name="${J}" \
                --output="${LOGDIR}/${J}_%j.out" --error="${LOGDIR}/${J}_%j.err" \
                "${FLAGS[@]}" \
                --export="ALL,REPO_ROOT=${REPO_ROOT},HF_HOME=${HF_HOME},REPL_OUTPUT_DIR=${REPL_LEN_DIR},VARIANT=${VARIANT},BASE_MODEL=${BASE_MODEL},INPUT_CSV=${csv},MAX_LENGTH=${MAX_LENGTH},BATCH_SIZE=${BATCH},POOLING=${POOLING:-mean},THRESHOLD=${THRESHOLD:-0.5}" \
                "${JOB}"
            NUM=$((NUM+1))
        done
    done
done
echo ""
echo "Submitted ${NUM} all-heads genome-wide jobs. Monitor: squeue -u \$USER"
echo "Output: ${OUTPUT_DIR}/<LEN>/genome_wide_heads/<variant>/{lp,nn}/"
