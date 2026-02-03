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
# Edit slurm_scripts/run_train_ntv2.sh with your paths, then submit:
sbatch slurm_scripts/run_train_ntv2.sh
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

## Inference

Run inference on a CSV file using a fine-tuned model to get predictions with probability scores.

### Run Inference

```bash
python inference_nt.py \
    --input_csv="/path/to/test.csv" \
    --model_path="/path/to/finetuned/model" \
    --output_csv="/path/to/predictions.csv" \
    --threshold=0.5 \
    --save_metrics
```

### Input Format

CSV file with at least a `sequence` column. If `label` column is present, metrics will be computed.

```csv
sequence,label
ACGTACGTACGT...,0
TGCATGCATGCA...,1
```

### Output Format

CSV file with predictions and probabilities:
- `sequence`: Original sequence
- `label`: Original label (if present)
- `prob_0`: Probability of class 0
- `prob_1`: Probability of class 1
- `pred_label`: Predicted label

If `--save_metrics` is specified and labels are present, a `_metrics.json` file is also saved.

### Threshold Analysis

Use custom threshold for classification:

```bash
python inference_nt.py \
    --input_csv="/path/to/test.csv" \
    --model_path="/path/to/model" \
    --threshold=0.7 \
    --save_metrics
```

### Inference Optimization

Speed up inference with mixed precision (recommended for GPU):

```bash
# Float16 mixed precision (~2.6x faster on A100)
python inference_nt.py \
    --input_csv="/path/to/test.csv" \
    --model_path="/path/to/model" \
    --fp16

# Bfloat16 mixed precision (recommended for A100 GPUs)
python inference_nt.py \
    --input_csv="/path/to/test.csv" \
    --model_path="/path/to/model" \
    --bf16
```

**Performance comparison (500M model, A100 GPU):**

| Precision | Throughput | Memory | Speedup |
|-----------|------------|--------|---------|
| fp32 (default) | 32.7 seq/s | 2322 MB | 1x |
| fp16 | 86.2 seq/s | 1893 MB | **2.6x** |
| bf16 | 76.2 seq/s | 3242 MB | 2.3x |

### torch.compile() Optimization

Use `torch.compile()` for potential additional speedup through kernel fusion and optimization:

```bash
# Combine fp16 with torch.compile
python inference_nt.py \
    --input_csv="/path/to/test.csv" \
    --model_path="/path/to/model" \
    --fp16 \
    --compile

# With max-autotune mode (slower compilation, potentially faster inference)
python inference_nt.py \
    --input_csv="/path/to/test.csv" \
    --model_path="/path/to/model" \
    --fp16 \
    --compile \
    --compile_mode max-autotune
```

**Compile modes:**
- `default` (recommended): Balanced compilation, works with all models
- `reduce-overhead`: Reduces Python overhead using CUDA graphs (may fail with rotary embeddings)
- `max-autotune`: Spends more time compiling for best performance

**Notes:**
- First inference run will be slow due to compilation. Subsequent runs will be faster.
- Use `default` mode for NT-v2 models (reduce-overhead may fail due to rotary embedding caching).

**Important: torch.compile() may not help NT-v2 models**

Testing showed that `torch.compile()` actually *decreased* performance for Nucleotide Transformer v2:

| Configuration | Throughput | Result |
|---------------|------------|--------|
| fp16 only | 86.2 seq/s | **Recommended** |
| fp16 + compile | 37.9 seq/s | 2.3x slower |

This is due to the model's custom implementation with rotary embeddings causing graph breaks and compilation overhead. **For NT-v2, use `--fp16` without `--compile` for best performance.**

### Directory-based Inference

Process all CSV files in a directory with a single model load (much faster than separate jobs):

```bash
python inference_nt_dir.py \
    --input_dir="/path/to/csv_directory" \
    --output_dir="/path/to/output_directory" \
    --model_path="/path/to/finetuned/model" \
    --fp16 \
    --save_metrics
```

This loads the model once and processes all CSV files sequentially, saving predictions to `{basename}_predictions.csv` in the output directory.

---

## Profiling

Profile inference to analyze performance bottlenecks and generate data for roofline analysis.

**Two profiling options:**
- `--profile_torch`: Uses PyTorch's built-in profiler (no special permissions needed)
- `ncu` (Nsight Compute): Kernel-level analysis (requires GPU performance counter access)

### Run Profiling with torch.profiler

```bash
python inference_nt.py \
    --input_csv="/path/to/test.csv" \
    --model_path="/path/to/model" \
    --fp16 \
    --profile_torch \
    --profile_batches=10 \
    --profile_output="./profile_results"
```

### Profiling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--profile_torch` | false | Enable torch.profiler (no special permissions needed) |
| `--profile_warmup` | 3 | Number of warmup batches before profiling |
| `--profile_batches` | 10 | Number of batches to profile |
| `--profile_output` | `./profile_traces` | Directory to save profiling traces |

### Profiling Output

The profiler generates:

1. **Operation breakdown** - Top operations by CUDA time, CPU time, and memory
2. **Chrome trace** - `trace_bs{batch_size}_{precision}_chrome.json` for visualization in `chrome://tracing`
3. **Stats JSON** - `trace_bs{batch_size}_{precision}_stats.json` with:
   - Model size (parameters, GB)
   - Throughput (sequences/second)
   - Memory bandwidth analysis (achieved vs peak GB/s)
   - Compute analysis (achieved vs peak TFLOPS)
   - Roofline analysis (arithmetic intensity, memory/compute bound status)

### Roofline Analysis

The profiler calculates:
- **Arithmetic Intensity**: FLOPs per byte transferred
- **Ridge Point**: Where memory-bound transitions to compute-bound (A100: ~153 FLOPs/Byte)
- **Bound Status**: Whether the workload is memory-bound or compute-bound

### Detailed Profiling with NVIDIA Nsight

For detailed cache analysis and kernel-level metrics:

```bash
# Nsight Systems (timeline and overview)
nsys profile -o inference_profile python inference_nt.py \
    --input_csv="/path/to/test.csv" \
    --model_path="/path/to/model" \
    --fp16

# Nsight Compute (kernel-level roofline)
ncu --set full -o kernel_profile python inference_nt.py \
    --input_csv="/path/to/test.csv" \
    --model_path="/path/to/model" \
    --fp16
```

Key Nsight Compute metrics:
- `l2_tex_read_hit_rate`: L2 cache hit rate
- `dram_read_throughput`: HBM read bandwidth
- `sm_efficiency`: Streaming multiprocessor utilization

---

## SLURM Scripts (for HPC)

SLURM scripts are provided in `slurm_scripts/` for running on HPC clusters (configured for NIH Biowulf):

### Fine-tuning
```bash
# Edit configuration in slurm_scripts/run_train_ntv2.sh, then:
sbatch slurm_scripts/run_train_ntv2.sh
```

### Embedding Analysis
```bash
# 1. Edit configuration in slurm_scripts/wrapper_run_embedding_analysis.sh
# 2. Submit job:
bash slurm_scripts/wrapper_run_embedding_analysis.sh

# For interactive testing (no sbatch):
bash slurm_scripts/run_embedding_analysis_interactive.sh
```

### Inference
```bash
# 1. Edit configuration in slurm_scripts/wrapper_run_inference.sh
# 2. Submit job:
bash slurm_scripts/wrapper_run_inference.sh

# For interactive testing (no sbatch):
bash slurm_scripts/run_inference_interactive.sh
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

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_csv` | (required) | CSV file with 'sequence' column |
| `--model_path` | (required) | Path to fine-tuned model directory |
| `--output_csv` | auto | Output CSV path (default: input with _predictions suffix) |
| `--batch_size` | 16 | Batch size for inference |
| `--max_length` | 2048 | Max sequence length in tokens |
| `--threshold` | 0.5 | Classification threshold for prob_1 |
| `--save_metrics` | false | Save metrics JSON if labels present |
| `--fp16` | false | Use float16 mixed precision (~2.6x faster) |
| `--bf16` | false | Use bfloat16 mixed precision (for A100 GPUs) |
| `--compile` | false | Use torch.compile() for kernel optimization |
| `--compile_mode` | `reduce-overhead` | Compile mode: default, reduce-overhead, max-autotune |
| `--profile_torch` | false | Enable torch.profiler (no special permissions needed) |
| `--profile_batches` | 10 | Number of batches to profile |
| `--profile_output` | `./profile_traces` | Directory for profiling output |

### Directory Inference Parameters (inference_nt_dir.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_dir` | (required) | Directory containing CSV files |
| `--output_dir` | (required) | Directory to save predictions |
| `--model_path` | (required) | Path to fine-tuned model directory |
| `--batch_size` | 16 | Batch size for inference |
| `--max_length` | 2048 | Max sequence length in tokens |
| `--threshold` | 0.5 | Classification threshold for prob_1 |
| `--pattern` | `*.csv` | Glob pattern for input files |
| `--save_metrics` | false | Save metrics JSON for each file |
| `--fp16` | false | Use float16 mixed precision |
| `--bf16` | false | Use bfloat16 mixed precision |

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
