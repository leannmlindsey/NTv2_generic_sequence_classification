#!/bin/bash
#
# Prefetch the HuggingFace model into the local cache so the OFFLINE SLURM jobs
# can find it. RUN THIS ON A LOGIN NODE — Biowulf compute nodes have no internet,
# and the jobs run with HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1.
#
# Reads BASE_MODEL, HF_HOME, CONDA_ENV from lambda_replication.conf, downloads the
# model into HF_HOME, then verifies it loads with offline mode on.
#
# Usage:
#   bash slurm_scripts/lambda_replication/prefetch_hf_cache.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/lambda_replication.conf"

source /data/lindseylm/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export HF_HOME="${HF_HOME:-/data/lindseylm/.cache/huggingface}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE     # must be ONLINE to download

echo "Prefetching '${BASE_MODEL}'"
echo "  into HF_HOME=${HF_HOME}"
echo "  conda env=${CONDA_DEFAULT_ENV}   python=$(command -v python)"

python -c "import sys; from transformers import AutoConfig, AutoTokenizer, AutoModel; m=sys.argv[1]; AutoConfig.from_pretrained(m, trust_remote_code=True); AutoTokenizer.from_pretrained(m, trust_remote_code=True); AutoModel.from_pretrained(m, trust_remote_code=True); print('PREFETCH OK:', m)" "${BASE_MODEL}"

echo "Verifying offline load (simulates the compute node)..."
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "import sys; from transformers import AutoTokenizer, AutoModel; m=sys.argv[1]; AutoTokenizer.from_pretrained(m, trust_remote_code=True); AutoModel.from_pretrained(m, trust_remote_code=True); print('OFFLINE LOAD OK:', m)" "${BASE_MODEL}"
