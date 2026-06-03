# NT-v2 LAMBDA_v1 replication

Orchestration layer that fans out the **existing** NT-v2 job scripts
(`finetune_nt_phage.py`, `inference_nt.py`, `embedding_analysis_nt.py`) across
the LAMBDA_v1 windows and seeds, then picks the best seed per window by test-set
MCC and runs all diagnostic + genome-wide inference. The model/experiment code
is unchanged — these scripts only submit it with the right env.

## Two-step workflow

```bash
# 0. (one time) pre-warm the HF cache from a LOGIN node so jobs can run offline:
#    module load conda && source activate nt
#    python -c "from transformers import AutoModel, AutoTokenizer; \
#      m='InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'; \
#      AutoTokenizer.from_pretrained(m, trust_remote_code=True); \
#      AutoModel.from_pretrained(m, trust_remote_code=True)"
#    (set HF_HOME=/data/lindseylm/.cache/huggingface first)

# 1. Edit lambda_replication.conf — confirm LAMBDA_BASE + OUTPUT_DIR.
bash slurm_scripts/lambda_replication/run_lambda_training.sh   # finetune × seeds × windows
# 2. wait — squeue -u $USER
bash slurm_scripts/lambda_replication/check_training.sh        # confirm all seeds healthy
bash slurm_scripts/lambda_replication/run_lambda_inference.sh  # pick winner + all inference
# 3. wait — squeue -u $USER
bash slurm_scripts/lambda_replication/check_inference.sh       # confirm all outputs landed
```

## Files

| File | Role |
|------|------|
| `lambda_replication.conf` | the only file you normally edit — all paths + hyperparameters |
| `run_lambda_training.sh` | submit one finetune job per (window × variant × seed) |
| `lambda_finetune_job.sh` | sbatch body: one finetune run (calls `finetune_nt_phage.py`) |
| `select_best_model.py` | pick best-of-N seed per (window × variant) by **test-set MCC** → `winners.json` |
| `run_lambda_inference.sh` | run winner selection + embedding + all diagnostic/genome-wide inference |
| `lambda_inference_job.sh` | sbatch body: one inference run (calls `inference_nt.py`) |
| `lambda_embedding_job.sh` | sbatch body: pretrained-embedding analysis (calls `embedding_analysis_nt.py`) |
| `print_winner_exports.py` | emit shell exports for the winning checkpoint |
| `check_training.sh` / `check_inference.sh` | post-hoc verification helpers |

## Model-selection logic (two levels)

1. **Per-seed checkpoint** (inside one finetune run): HF Trainer with
   `metric_for_best_model=eval_mcc` + `load_best_model_at_end` (NT-v2's own
   default). Left alone.
2. **Cross-seed winner** (across the N seeds): `select_best_model.py` reads each
   `seed-<N>/test_results.json` and picks the max **test-set MCC** (key
   `eval_mcc`), writing `winners.json`.

`finetune_nt_phage.py` already evaluates the test set with
`metric_key_prefix="eval"` and writes the metrics to
`<output_dir>/test_results.json` (so the test MCC is `eval_mcc`). **No surfacing
/ copy step is needed** — `select_best_model.py` reads that file directly.

## NT-v2-specific notes

- **Single variant.** Only `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`
  (`VARIANTS="nt_500m"`).
- **Hard token limit (~2048).** Per-window token lengths come from the existing
  scripts (`run_optimized_train.sh` case block, `run_memory_test.sh`):
  `2k → 512`, `4k → 1024`, `8k → 2048`. Per-window train batch sizes
  (`8 / 1 / 1`) likewise come from `run_optimized_train.sh`. **8k is at the
  token cap**; do not raise `MAX_LENGTH_8k`.
- **8k diagnostics.** `FNR_8k` and `GENOME_WIDE_8k` exist in LAMBDA_v1 and are
  wired in. The inference driver still warns-and-skips any diagnostic file/dir
  that happens to be missing (defensive; not fatal).
- **`val.csv` → `dev.csv`.** LAMBDA_v1 ships `val.csv`, but `finetune_nt_phage.py`
  and `embedding_analysis_nt.py` read `dev.csv`. The finetune/embedding job
  scripts stage a per-run input dir with a `dev.csv` symlink to `val.csv` (and
  symlinks for `train.csv`/`test.csv`) so no model code is modified.
- **No genome-wide aggregate analysis.** NT-v2 has no genome-wide
  clustering/threshold entry point, so genome-wide inference produces per-CSV
  prediction files under `inference/<variant>/` but no separate analysis job is
  chained (unlike the DNABERT-2 reference).
- **Bare job scripts.** No `set -e` and no `2>/dev/null` masking in the job
  bodies — `source activate` under `set -e` silently kills SLURM jobs.
- Outputs go under `/data/lindseylm/...`, never `/gpfs/gsfs12/...`.
