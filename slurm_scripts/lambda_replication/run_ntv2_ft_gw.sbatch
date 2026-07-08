#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=ntv2_ft_gw
#SBATCH --output=ntv2_ft_gw_%j.log
#SBATCH --error=ntv2_ft_gw_%j.log
#
# NT v2 FT genome-wide (2k, 4k, 8k) — the FT head was never deployed genome-wide
# (a probe won). Deploys the BEST fine-tuned seed's checkpoint across the
# genome-wide CSVs via the EXISTING inference_nt.py, into a dedicated dir (no
# overwrite). Best seeds + test-MCC fingerprints (verify on Delta vs each seed's
# test_results.json before trusting):
#   2k -> seed 5 (0.8636)   4k -> seed 1 (0.8919)   8k -> seed 5 (0.9341)
#
# Submit FROM the NTv2 repo root on Delta, with the nt env active:
#   conda activate nt
#   sbatch slurm_scripts/lambda_replication/run_ntv2_ft_gw.sbatch
set -uo pipefail

echo "conda env: ${CONDA_DEFAULT_ENV:-none}   python: $(command -v python)  node $(hostname)  $(date)"
python -c "import torch, transformers" 2>/dev/null || { echo "ERROR: torch/transformers not importable — activate the nt env before sbatch"; exit 1; }

export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/work/hdd/bfzj/llindsey1/hf_cache}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"; export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

OUTROOT=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/NTv2_generic_sequence_classification/outputs
GWROOT=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/LAMBDA_v1/genome_wide
VARIANT=nt_500m

seed_for () { case "$1" in 2k) echo 5;; 4k) echo 1;; 8k) echo 5;; esac; }
maxlen_for () { case "$1" in 2k) echo 512;; 4k) echo 1024;; 8k) echo 2048;; esac; }

for w in 2k 4k 8k; do
  seed=$(seed_for "$w"); ML=$(maxlen_for "$w")
  CKPT="$OUTROOT/$w/finetune/$VARIANT/seed-$seed"
  OUT="$OUTROOT/$w/genome_wide_ft/$VARIANT"; mkdir -p "$OUT"
  [ -d "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
  echo "===== $w  seed $seed  max_length $ML  ckpt $CKPT ====="
  n=0
  for csv in "$GWROOT/$w"/*.csv; do
    stem=$(basename "$csv" .csv)
    python inference_nt.py \
      --input_csv "$csv" \
      --model_path "$CKPT" \
      --output_csv "$OUT/genome_wide_${stem}_predictions.csv" \
      --max_length $ML --batch_size 16 --threshold 0.5 --save_metrics
    n=$((n+1))
  done
  echo "  $w done: $n genome(s) -> $OUT"
done
echo "DONE $(date)"
