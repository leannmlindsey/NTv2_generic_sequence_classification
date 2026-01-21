#!/usr/bin/env python3
"""
Fine-tuning Nucleotide Transformer v2 for binary sequence classification.
Based on official HuggingFace notebooks for NT fine-tuning.

Usage:
    python finetune_nt_phage.py \
        --model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
        --dataset_dir /path/to/data \
        --output_dir ./output \
        --max_length 2048 \
        --per_device_train_batch_size 8 \
        --num_train_epochs 3
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    roc_auc_score,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Nucleotide Transformer v2 for binary classification"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        help="HuggingFace model name or path",
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing train.csv, dev.csv, test.csv",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length (in tokens). NT-v2 supports up to 2048 tokens (~12kb).",
    )
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./nt_output")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--load_best_model_at_end", action="store_true", default=True)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_mcc")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    
    return parser.parse_args()


def load_data(dataset_dir: str) -> DatasetDict:
    """
    Load train/dev/test CSV files.
    Expected format: CSV with columns 'sequence' and 'label'
    
    IMPORTANT: Only 'sequence' and 'label' columns are used.
    All other columns are explicitly dropped to prevent data leakage.
    """
    dataset_dir = Path(dataset_dir)
    
    datasets = {}
    for split in ["train", "dev", "test"]:
        filepath = dataset_dir / f"{split}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            
            # Log all columns found
            print(f"\n{split}.csv columns found: {list(df.columns)}")
            
            # Ensure column names are standardized
            if "sequence" not in df.columns:
                # Try common alternatives
                for col in ["seq", "dna", "Sequence", "DNA"]:
                    if col in df.columns:
                        df = df.rename(columns={col: "sequence"})
                        print(f"  Renamed '{col}' -> 'sequence'")
                        break
            if "label" not in df.columns:
                for col in ["Label", "labels", "Labels", "class", "Class"]:
                    if col in df.columns:
                        df = df.rename(columns={col: "label"})
                        print(f"  Renamed '{col}' -> 'label'")
                        break
            
            # Verify required columns exist
            if "sequence" not in df.columns:
                raise ValueError(f"Could not find 'sequence' column in {filepath}")
            if "label" not in df.columns:
                raise ValueError(f"Could not find 'label' column in {filepath}")
            
            # EXPLICITLY select only sequence and label columns
            # This prevents any data leakage from other columns
            used_columns = ["sequence", "label"]
            dropped_columns = [col for col in df.columns if col not in used_columns]
            
            if dropped_columns:
                print(f"  DROPPING columns (not used): {dropped_columns}")
            
            # Create dataset with ONLY the two required columns
            df_clean = df[used_columns].copy()
            
            print(f"  USING columns: {list(df_clean.columns)}")
            print(f"  Loaded {split}: {len(df_clean)} samples")
            
            # Verify label distribution
            label_counts = df_clean["label"].value_counts().to_dict()
            print(f"  Label distribution: {label_counts}")
            
            datasets[split if split != "dev" else "validation"] = Dataset.from_pandas(
                df_clean, preserve_index=False
            )
        else:
            print(f"Warning: {filepath} not found")
    
    return DatasetDict(datasets)


def compute_metrics(eval_pred):
    """Compute classification metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # For binary classification, get probabilities for ROC-AUC
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    
    # Calculate sensitivity (recall) and specificity
    # For binary: label 1 = positive, label 0 = negative
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }
    
    # Add ROC-AUC if binary
    try:
        auc = roc_auc_score(labels, probs[:, 1])
        metrics["auc"] = auc
    except Exception:
        pass
    
    return metrics


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    print("=" * 60)
    print("Nucleotide Transformer v2 Fine-tuning")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Max length: {args.max_length}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected!")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    
    # Load model for sequence classification
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,  # Binary classification
        trust_remote_code=True,
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load data
    print("\nLoading dataset...")
    dataset = load_data(args.dataset_dir)
    
    # Tokenize function
    def tokenize_function(examples):
        """Tokenize DNA sequences."""
        return tokenizer(
            examples["sequence"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors=None,  # Return lists for dataset mapping
        )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["sequence"],
        desc="Tokenizing",
    )
    
    # Set format for PyTorch
    tokenized_datasets.set_format("torch")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        report_to="none",  # Disable wandb/tensorboard by default
    )
    
    # Callbacks
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    # Train
    print("\nStarting training...")
    train_result = trainer.train()
    
    # Save model
    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Log training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate on validation set
    if "validation" in tokenized_datasets:
        print("\nEvaluating on validation set...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    # Evaluate on test set
    if "test" in tokenized_datasets:
        print("\nEvaluating on test set...")
        # Use metric_key_prefix="eval" to match the requested output format
        test_metrics = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="eval")
        
        # Save test results to test_results.json in the requested format
        test_results_path = os.path.join(args.output_dir, "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Test results saved to: {test_results_path}")
        
        # Print final test results
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        for key, value in test_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("\nTraining complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
