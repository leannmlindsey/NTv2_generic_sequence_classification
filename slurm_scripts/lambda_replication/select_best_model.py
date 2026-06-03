#!/usr/bin/env python3
"""
Per-variant, pick the finetune seed with the highest test-set MCC.

NT-v2 has a single architecture, so this selects the best-of-N seed for each
variant (only finetune candidates; the embedding linear probe / 3-layer NN are
reported separately and are not part of the winning checkpoint).

Writes <output_dir>/winners.json:
    {
      "nt_500m": {
        "type": "finetune",
        "seed": 3,
        "test_mcc": 0.85,
        "path": "<absolute path to the seed dir = saved model dir>",
        "base_model": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "all_candidates": [{type, seed, test_mcc}, ...]
      }
    }

Reads:
  <output_dir>/finetune/<variant>/seed-<N>/test_results.json
      (written DIRECTLY by finetune_nt_phage.py — it evaluates the test set with
       metric_key_prefix="eval" and json.dumps the metrics to test_results.json,
       so the test MCC lives under "eval_mcc". No surfacing/copy step is needed.
       The other keys below are accepted as fallbacks for portability.)
"""

import argparse
import glob
import json
import os
import sys


# MCC key candidates in order of preference. finetune_nt_phage.py emits "mcc" in
# compute_metrics, evaluated with metric_key_prefix="eval" -> "eval_mcc".
MCC_KEYS = ("eval_mcc", "eval_matthews_correlation", "mcc", "matthews_correlation")


def _read_mcc(metrics):
    for k in MCC_KEYS:
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])
    return None


def collect_finetune_candidates(variant_dir):
    out = []
    for seed_dir in sorted(glob.glob(os.path.join(variant_dir, "seed-*"))):
        results_path = os.path.join(seed_dir, "test_results.json")
        if not os.path.isfile(results_path):
            print(f"  WARN: missing {results_path}, skipping", file=sys.stderr)
            continue
        with open(results_path) as f:
            metrics = json.load(f)
        mcc = _read_mcc(metrics)
        if mcc is None:
            print(f"  WARN: no MCC key {MCC_KEYS} in {results_path}, skipping",
                  file=sys.stderr)
            continue
        seed = int(os.path.basename(seed_dir).split("-")[1])
        out.append({
            "type": "finetune",
            "seed": seed,
            "test_mcc": float(mcc),
            "path": os.path.abspath(seed_dir),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", required=True,
                        help="Per-length replication output dir (contains finetune/)")
    parser.add_argument("--variants", nargs="+", required=True,
                        help="Variants to select for (e.g. nt_500m)")
    parser.add_argument("--base_model",
                        default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
                        help="HF base model recorded in winners.json")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Skip variants with no candidates instead of aborting. "
                             "Useful for in-progress dev runs; do NOT use for the "
                             "reviewer-facing pipeline — a missing variant there means "
                             "a real training failure that should fail loudly.")
    args = parser.parse_args()

    winners = {}
    skipped = []
    for variant in args.variants:
        print(f"\n=== {variant} ===")
        finetune_dir = os.path.join(args.output_dir, "finetune", variant)
        candidates = collect_finetune_candidates(finetune_dir)
        if not candidates:
            if not args.allow_partial:
                print(f"  ERROR: no candidates found for {variant} "
                      f"(missing seed-*/test_results.json). "
                      f"Re-run with --allow-partial to skip and continue.",
                      file=sys.stderr)
                sys.exit(1)
            print(f"  SKIP: no candidates found for {variant}", file=sys.stderr)
            skipped.append(variant)
            continue

        for c in sorted(candidates, key=lambda c: c["test_mcc"], reverse=True):
            print(f"  test_mcc={c['test_mcc']:.4f}  finetune/seed-{c['seed']}")

        winner = max(candidates, key=lambda c: c["test_mcc"])
        winner["base_model"] = args.base_model
        winner["all_candidates"] = [
            {k: v for k, v in c.items() if k in ("type", "seed", "test_mcc")}
            for c in candidates
        ]
        winners[variant] = winner
        print(f"  WINNER: seed-{winner['seed']} (test_mcc={winner['test_mcc']:.4f})")

    out_path = os.path.join(args.output_dir, "winners.json")
    with open(out_path, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"\nWrote {out_path}  ({len(winners)} variant(s) with winners"
          f"{'; skipped: ' + ','.join(skipped) if skipped else ''})")

    if not winners:
        print("\nERROR: no variant produced any candidates; nothing to write.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
