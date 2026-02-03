#!/usr/bin/env python3
"""
Summarize Inference Results from NT-v2 Directory Inference

This script extracts metrics from a summary.json file (produced by inference_nt_dir.py)
and creates:
1. A CSV file with one row per genome and columns for each metric
2. Summary statistics:
   - Metrics averaged by genome (macro-average)
   - Metrics computed from aggregated TP/TN/FP/FN (micro-average, weighted by samples)

Usage:
    python summarize_inference_results.py --input summary.json --output metrics_summary.csv
    python summarize_inference_results.py --input /path/to/output_dir  # auto-finds summary.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Summarize inference results from NT-v2 directory inference"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to summary.json or directory containing it",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: metrics_summary.csv in same directory as input)",
    )
    return parser.parse_args()


def load_summary(input_path: str) -> dict:
    """Load summary.json from path or directory."""
    if os.path.isdir(input_path):
        json_path = os.path.join(input_path, "summary.json")
    else:
        json_path = input_path

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Summary file not found: {json_path}")

    with open(json_path, "r") as f:
        return json.load(f), json_path


def extract_metrics_dataframe(summary: dict) -> pd.DataFrame:
    """Extract per-genome metrics into a DataFrame."""
    rows = []

    for result in summary.get("results", []):
        if "metrics" not in result:
            continue

        row = {
            "genome": result["file"],
            "samples": result["samples"],
            "time_seconds": result["time_seconds"],
            "throughput": result["throughput"],
        }

        # Add all metrics
        metrics = result["metrics"]
        for key, value in metrics.items():
            row[key] = value

        rows.append(row)

    if not rows:
        raise ValueError("No results with metrics found in summary")

    df = pd.DataFrame(rows)

    # Reorder columns for readability
    metric_cols = [
        "accuracy", "precision", "recall", "f1", "mcc", "auc",
        "sensitivity", "specificity", "fpr", "fnr",
        "true_positives", "false_positives", "true_negatives", "false_negatives"
    ]

    # Build column order
    first_cols = ["genome", "samples", "time_seconds", "throughput"]
    ordered_cols = first_cols + [c for c in metric_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + other_cols]

    return df


def calculate_genome_averaged_metrics(df: pd.DataFrame) -> dict:
    """Calculate metrics as simple average across genomes (macro-average)."""
    metric_cols = [
        "accuracy", "precision", "recall", "f1", "mcc", "auc",
        "sensitivity", "specificity", "fpr", "fnr"
    ]

    metrics = {"method": "genome_averaged (macro)"}
    for col in metric_cols:
        if col in df.columns:
            metrics[col] = float(df[col].mean())
            metrics[f"{col}_std"] = float(df[col].std())

    # Also report totals (convert to native Python int for JSON serialization)
    for col in ["true_positives", "false_positives", "true_negatives", "false_negatives", "samples"]:
        if col in df.columns:
            metrics[f"total_{col}"] = int(df[col].sum())

    return metrics


def calculate_aggregate_metrics(df: pd.DataFrame) -> dict:
    """Calculate metrics from aggregated confusion matrix (micro-average)."""
    # Sum up all TP, TN, FP, FN across genomes (convert to Python int)
    tp = int(df["true_positives"].sum())
    tn = int(df["true_negatives"].sum())
    fp = int(df["false_positives"].sum())
    fn = int(df["false_negatives"].sum())

    total = tp + tn + fp + fn

    metrics = {
        "method": "aggregate (micro)",
        "total_samples": total,
        "total_true_positives": tp,
        "total_false_positives": fp,
        "total_true_negatives": tn,
        "total_false_negatives": fn,
    }

    # Calculate metrics from aggregated counts
    metrics["accuracy"] = float((tp + tn) / total) if total > 0 else 0.0
    metrics["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["sensitivity"] = metrics["recall"]  # Same as recall
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # F1 score
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = float(2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"]))
    else:
        metrics["f1"] = 0.0

    # MCC from confusion matrix
    numerator = (tp * tn) - (fp * fn)
    denom_product = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    denominator = float(denom_product) ** 0.5 if denom_product > 0 else 0.0
    metrics["mcc"] = float(numerator / denominator) if denominator > 0 else 0.0

    # AUC cannot be computed from aggregated counts (needs probabilities)
    # We can report the average AUC as a weighted average
    if "auc" in df.columns:
        total_samples = int(df["samples"].sum())
        weighted_auc = float((df["auc"] * df["samples"]).sum() / total_samples)
        metrics["auc_weighted"] = weighted_auc

    return metrics


def print_summary(genome_metrics: dict, aggregate_metrics: dict):
    """Print formatted summary of both metric calculations."""
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)

    print("\n1. GENOME-AVERAGED METRICS (Macro-Average)")
    print("-" * 50)
    print("   Each genome weighted equally, regardless of sample count")
    print()

    metric_order = ["accuracy", "precision", "recall", "f1", "mcc", "auc",
                    "sensitivity", "specificity", "fpr", "fnr"]

    for metric in metric_order:
        if metric in genome_metrics:
            std_key = f"{metric}_std"
            std_val = genome_metrics.get(std_key, 0.0)
            print(f"   {metric:15s}: {genome_metrics[metric]:.6f} +/- {std_val:.6f}")

    print()
    print(f"   Total samples:  {genome_metrics.get('total_samples', 'N/A'):,}")
    print(f"   Total TP:       {genome_metrics.get('total_true_positives', 'N/A'):,}")
    print(f"   Total FP:       {genome_metrics.get('total_false_positives', 'N/A'):,}")
    print(f"   Total TN:       {genome_metrics.get('total_true_negatives', 'N/A'):,}")
    print(f"   Total FN:       {genome_metrics.get('total_false_negatives', 'N/A'):,}")

    print("\n2. AGGREGATE METRICS (Micro-Average)")
    print("-" * 50)
    print("   Computed from summed TP/TN/FP/FN (weighted by sample count)")
    print()

    for metric in metric_order:
        if metric in aggregate_metrics:
            print(f"   {metric:15s}: {aggregate_metrics[metric]:.6f}")

    if "auc_weighted" in aggregate_metrics:
        print(f"   {'auc_weighted':15s}: {aggregate_metrics['auc_weighted']:.6f}")

    print()
    print(f"   Total samples:  {aggregate_metrics.get('total_samples', 'N/A'):,}")
    print(f"   Total TP:       {aggregate_metrics.get('total_true_positives', 'N/A'):,}")
    print(f"   Total FP:       {aggregate_metrics.get('total_false_positives', 'N/A'):,}")
    print(f"   Total TN:       {aggregate_metrics.get('total_true_negatives', 'N/A'):,}")
    print(f"   Total FN:       {aggregate_metrics.get('total_false_negatives', 'N/A'):,}")

    print("\n" + "=" * 80)


def main():
    args = parse_arguments()

    # Load summary
    print(f"Loading summary from: {args.input}")
    summary, json_path = load_summary(args.input)

    # Determine output path
    if args.output:
        output_csv = args.output
    else:
        output_dir = os.path.dirname(json_path)
        output_csv = os.path.join(output_dir, "metrics_summary.csv")

    # Extract metrics to DataFrame
    df = extract_metrics_dataframe(summary)
    print(f"Extracted metrics for {len(df)} genomes")

    # Save per-genome CSV
    df.to_csv(output_csv, index=False)
    print(f"Per-genome metrics saved to: {output_csv}")

    # Calculate summary metrics
    genome_metrics = calculate_genome_averaged_metrics(df)
    aggregate_metrics = calculate_aggregate_metrics(df)

    # Print summary
    print_summary(genome_metrics, aggregate_metrics)

    # Save summary metrics to separate JSON
    summary_metrics_path = output_csv.replace(".csv", "_summary.json")
    summary_output = {
        "genome_averaged_macro": genome_metrics,
        "aggregate_micro": aggregate_metrics,
        "num_genomes": len(df),
        "source_file": json_path,
    }
    with open(summary_metrics_path, "w") as f:
        json.dump(summary_output, f, indent=2)
    print(f"\nSummary metrics saved to: {summary_metrics_path}")


if __name__ == "__main__":
    main()
