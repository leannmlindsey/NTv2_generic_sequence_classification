#!/bin/bash
# Setup conda environment for Nucleotide Transformer fine-tuning on NCSA Delta-AI
# (GH200 Grace Hopper nodes, linux-aarch64 / ARM).
#
# This is the Delta-AI counterpart to setup.sh (which targets Biowulf).
# Run this once on a compute node, e.g. after:
#   srun --account=<acct> --partition=ghx4 --nodes=1 --gpus-per-node=1 \
#        --cpus-per-task=8 --mem=16g --time 4:00:00 --pty bash
#
# Run with: bash setup_deltaai.sh
set -euo pipefail

echo "Setting up NT fine-tuning environment (Delta-AI / aarch64)..."

# --- Modules ---------------------------------------------------------------
# Delta-AI module names differ from Biowulf. We do NOT hard-require them:
#  - conda: you already have a personal miniconda3, so a conda module is optional.
#  - cuda:  the PyTorch wheels below bundle their own CUDA runtime, so a system
#           CUDA module is not needed just to run PyTorch.
# Try to load CUDA if present (helps if you later compile CUDA extensions),
# but never abort the script if the module name is missing.
echo "Attempting to load CUDA module (optional)..."
module load cuda 2>/dev/null \
  || module load cuda/12.8 2>/dev/null \
  || echo "  No CUDA module loaded (fine; PyTorch wheels bundle their own CUDA)."

# --- Conda env -------------------------------------------------------------
# Use the conda that is already on PATH (your miniconda3 base). If 'conda'
# is not found, point CONDA_EXE / source conda.sh before running this script.
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not on PATH. Activate your base conda first, e.g.:"
  echo "  source \$HOME/miniconda3/etc/profile.d/conda.sh"
  exit 1
fi

conda create -n nt python=3.11 -y
# 'conda activate' needs the shell hook; fall back to 'source activate'.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nt

# --- PyTorch (CUDA, aarch64) ----------------------------------------------
# Delta-AI GH200 is ARM (linux-aarch64). CRITICAL: a plain `pip install torch`
# (or the cu121 index) gives a CPU-ONLY aarch64 wheel -> torch.cuda.is_available()
# is False. Install from the cu124 index and PIN 2.5.1 — the version proven
# working on Delta GH200 (matches the generanno/dnabert2 envs: torch 2.5.1, CUDA
# 12.4). torch is therefore NOT in requirements.txt; install it here first.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# --- HuggingFace + everything else (transformers PINNED to 4.49.0) ---------
pip install -r "$(dirname "${BASH_SOURCE[0]}")/requirements.txt"

# --- Verify ----------------------------------------------------------------
echo ""
echo "Verifying GPU visibility..."
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| cuda build:', torch.version.cuda)"

echo ""
echo "Environment setup complete!"
echo "Activate with: conda activate nt   (or: source activate nt)"
echo ""
echo "To test the tokenizer download, run:"
echo "  python -c \"from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('InstaDeepAI/nucleotide-transformer-v2-500m-multi-species', trust_remote_code=True); print('Success!')\""
