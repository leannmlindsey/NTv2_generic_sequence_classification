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


def read_embedding_scores(embedding_dir):
    """Linear-probe / 3-layer-NN candidates from embedding_analysis_results.json.

    NTv2 uses the FLAT schema (pretrained_linear_probe_mcc / pretrained_nn_mcc),
    like GENERanno. Deployable probe artifacts are saved alongside by
    embedding_analysis_nt.py (linear_probe_pretrained.pkl, three_layer_nn_pretrained.pt
    + three_layer_nn_pretrained_scaler.pkl).
    """
    results_path = os.path.join(embedding_dir, "embedding_analysis_results.json")
    if not os.path.isfile(results_path):
        print(f"  WARN: missing {results_path} (no probe candidates)", file=sys.stderr)
        return []
    with open(results_path) as f:
        r = json.load(f)
    out = []
    lp = r.get("pretrained_linear_probe_mcc")
    if lp is not None:
        out.append({"type": "linear_probe", "seed": None, "test_mcc": float(lp),
                    "head_path": os.path.abspath(
                        os.path.join(embedding_dir, "linear_probe_pretrained.pkl"))})
    nn = r.get("pretrained_nn_mcc")
    if nn is not None:
        out.append({"type": "three_layer_nn", "seed": None, "test_mcc": float(nn),
                    "head_path": os.path.abspath(
                        os.path.join(embedding_dir, "three_layer_nn_pretrained.pt")),
                    "scaler_path": os.path.abspath(
                        os.path.join(embedding_dir, "three_layer_nn_pretrained_scaler.pkl"))})
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
        embedding_dir = os.path.join(args.output_dir, "embedding", variant)
        ft = collect_finetune_candidates(finetune_dir)
        emb = read_embedding_scores(embedding_dir)

        # Fine-tuning scored by the MEAN test MCC of its 5 seeds (deployed via the
        # single best seed); each probe scored by its test MCC.
        sel = []
        if ft:
            ft_avg = sum(c["test_mcc"] for c in ft) / len(ft)
            best_seed = max(ft, key=lambda c: c["test_mcc"])
            sel.append({"type": "finetune", "score": float(ft_avg), "seed": best_seed["seed"],
                        "test_mcc": float(ft_avg), "best_seed_test_mcc": best_seed["test_mcc"],
                        "path": best_seed["path"]})
        for cand in emb:
            c = dict(cand); c["score"] = c["test_mcc"]
            sel.append(c)

        if not sel:
            if not args.allow_partial:
                print(f"  ERROR: no candidates for {variant} (no finetune seeds AND no "
                      f"embedding_analysis_results.json). Re-run with --allow-partial to skip.",
                      file=sys.stderr)
                sys.exit(1)
            print(f"  SKIP: no candidates for {variant}", file=sys.stderr)
            skipped.append(variant)
            continue

        def _tag(c):
            return c["type"] + (f"/seed-{c['seed']}" if c.get("seed") is not None else "")
        for c in sorted(sel, key=lambda c: c["score"], reverse=True):
            note = "  (mean of 5 seeds)" if c["type"] == "finetune" else ""
            print(f"  score={c['score']:.4f}  {_tag(c)}{note}")

        # Highest score wins; ties prefer finetune (deploy FT only when its 5-seed
        # average is >= both probes), per the LAMBDA design.
        winner = max(sel, key=lambda c: (c["score"], c["type"] == "finetune"))
        winner["base_model"] = args.base_model
        winner["all_candidates"] = [
            {k: v for k, v in c.items() if k in ("type", "seed", "test_mcc", "score")}
            for c in sel
        ]
        winners[variant] = winner
        print(f"  WINNER: {_tag(winner)} (score={winner['score']:.4f})")

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
