# Nucleotide Transformer v2 Generic Sequence Classification

> Fine-tune and analyze Nucleotide Transformer v2 for any **binary classification** task using simple CSV files.

---

## Overview

This repository provides tools for:
1. **Fine-tuning** Nucleotide Transformer v2 on custom DNA sequence classification tasks
2. **Embedding analysis** to evaluate embedding quality with linear probes, silhouette scores, PCA visualization, and 3-layer NNs
3. **Random baseline comparison** to measure the "embedding power" gained from pretraining

## Supported Models

| Model Name | Parameters | Context Length |
|------------|:----------:|:--------------:|
| `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` | 50M | 2048 tokens (~12kb) |
| `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species` | 100M | 2048 tokens (~12kb) |
| `InstaDeepAI/nucleotide-transformer-v2-250m-multi-species` | 250M | 2048 tokens (~12kb) |
| `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species` | 500M | 2048 tokens (~12kb) |

---

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repo-url>
cd NTv2_generic_sequence_classification

# Create conda environment
bash setup.sh

# Or manually:
conda create -n nt python=3.10 -y
conda activate nt
pip install -r requirements.txt
```

### 2. Prepare Your Data

Create a directory containing three CSV files with `sequence` and `label` columns:

```
my_dataset/
├── train.csv
├── dev.csv    # (or val.csv)
└── test.csv
```

Each CSV should have this format:
```csv
sequence,label
ACGTACGTACGT...,0
TGCATGCATGCA...,1
GGCCAATTGGCC...,0
```

- `sequence`: DNA sequence (A, C, G, T, N characters)
- `label`: Integer class label (0 or 1 for binary classification)

**Note:** Only `sequence` and `label` columns are used. All other columns are dropped to prevent data leakage.

---

## Fine-tuning

### Run Fine-tuning

```bash
python finetune_nt_phage.py \
    --model_name="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species" \
    --dataset_dir="/path/to/my_dataset" \
    --output_dir="./results/my_task" \
    --per_device_train_batch_size=8 \
    --max_length=2048 \
    --learning_rate=3e-5 \
    --num_train_epochs=3 \
    --seed=42
```

### SLURM Script (for HPC)

```bash
# Edit run_train_ntv2.sh with your paths, then submit:
sbatch run_train_ntv2.sh
```

### Test Results

After training, comprehensive test metrics are saved to `test_results.json`:
- `eval_accuracy`, `eval_precision`, `eval_recall`, `eval_f1`
- `eval_mcc`: Matthews Correlation Coefficient
- `eval_sensitivity`, `eval_specificity`, `eval_auc`

---

## Embedding Analysis

Extract embeddings and evaluate their quality with linear probes, silhouette scores, PCA visualization, and a 3-layer NN.

### Run Embedding Analysis

```bash
python embedding_analysis_nt.py \
    --csv_dir="/path/to/csv/data" \
    --model_path="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species" \
    --output_dir="./results/embedding_analysis" \
    --pooling="mean"
```

### With Finetuned Model

```bash
python embedding_analysis_nt.py \
    --csv_dir="/path/to/csv/data" \
    --model_path="/path/to/finetuned/model" \
    --output_dir="./results/embedding_analysis_finetuned" \
    --pooling="mean"
```

### Outputs

- `embeddings_pretrained.npz`: Extracted embeddings for train/val/test sets
- `pca_visualization_pretrained.png`: PCA plot showing class separation
- `test_predictions_pretrained.csv`: Predictions with probabilities
- `three_layer_nn_pretrained.pt`: Trained 3-layer NN model
- `embedding_analysis_results.json`: All metrics in JSON format

### Metrics Generated

**Linear Probe (Logistic Regression):**
- Accuracy, Precision, Recall, F1
- MCC (Matthews Correlation Coefficient)
- AUC (Area Under ROC Curve)
- Sensitivity, Specificity

**3-Layer Neural Network:**
- Same metrics as linear probe

**Embedding Quality:**
- Silhouette Score: [-1, 1] range, measures cluster separation
- PCA Variance Explained: How much variance PC1 and PC2 capture

---

## Random Baseline Comparison

To measure the contribution of pretraining, compare against a randomly initialized model:

```bash
python embedding_analysis_nt.py \
    --csv_dir="/path/to/csv/data" \
    --model_path="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species" \
    --output_dir="./results/embedding_analysis" \
    --include_random_baseline
```

**Additional Outputs with Random Baseline:**
- `embeddings_random.npz`: Random baseline embeddings
- `pca_visualization_random.png`: PCA plot for random model
- `test_predictions_random.csv`: Random model predictions

**Embedding Power Metrics:**
The JSON output will include `embedding_power_*` metrics showing the difference (pretrained - random):
```json
{
  "embedding_power_linear_probe_accuracy": 0.15,
  "embedding_power_nn_mcc": 0.20,
  "embedding_power_silhouette_score": 0.30
}
```

---

## SLURM Scripts (for HPC)

SLURM scripts are provided in `slurm_scripts/` for running on HPC clusters (configured for NIH Biowulf):

### Fine-tuning
```bash
# Edit configuration in run_train_ntv2.sh, then:
sbatch run_train_ntv2.sh
```

### Embedding Analysis
```bash
# 1. Edit configuration in wrapper_run_embedding_analysis.sh
# 2. Submit job:
bash slurm_scripts/wrapper_run_embedding_analysis.sh

# For interactive testing (no sbatch):
bash slurm_scripts/run_embedding_analysis_interactive.sh
```

---

## Parameters Reference

### Fine-tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `nucleotide-transformer-v2-500m-multi-species` | HuggingFace model name |
| `--dataset_dir` | (required) | Directory with train/dev/test CSVs |
| `--max_length` | 2048 | Max sequence length in tokens |
| `--per_device_train_batch_size` | 8 | Training batch size |
| `--learning_rate` | 3e-5 | Learning rate |
| `--num_train_epochs` | 3 | Number of training epochs |
| `--early_stopping_patience` | 3 | Early stopping patience |
| `--seed` | 42 | Random seed |

### Embedding Analysis Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv_dir` | (required) | Directory with train/dev/test CSVs |
| `--model_path` | `nucleotide-transformer-v2-500m-multi-species` | Model path or HF name |
| `--output_dir` | `./results/embedding_analysis` | Output directory |
| `--batch_size` | 16 | Batch size for embedding extraction |
| `--max_length` | 2048 | Max sequence length in tokens |
| `--pooling` | `mean` | Pooling strategy: mean, cls, last |
| `--seed` | 42 | Random seed |
| `--nn_epochs` | 100 | Epochs for 3-layer NN training |
| `--nn_hidden_dim` | 256 | Hidden dimension for 3-layer NN |
| `--nn_lr` | 0.001 | Learning rate for 3-layer NN |
| `--include_random_baseline` | false | Include random baseline comparison |

---

## Requirements

```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
einops>=0.7.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## Citation

If you use Nucleotide Transformer in your research, please cite:

```bibtex
@article{dalla2024nucleotide,
  title={The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics},
  author={Dalla-Torre, Hugo and Gonzalez, Liam and Mendoza-Revilla, Javier and others},
  journal={Nature Methods},
  year={2024},
  publisher={Nature Publishing Group}
}
```
