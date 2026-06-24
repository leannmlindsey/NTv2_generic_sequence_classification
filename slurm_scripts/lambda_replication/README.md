# NT-v2 LAMBDA_v1 replication

Orchestration layer that fans out the **existing** NT-v2 job scripts
(`finetune_nt_phage.py`, `inference_nt.py`, `embedding_analysis_nt.py`) across
the LAMBDA_v1 windows and seeds, then picks the best seed per window by test-set
MCC and runs all diagnostic + genome-wide inference. The model/experiment code
is unchanged — these scripts only submit it with the right env.

## Environment (Delta-AI / GH200, aarch64)

Conda base is `/u/llindsey1/miniconda3`. The env is named `nt` and lives in the
home miniconda3. Build it once (also scripted in the repo's `setup_deltaai.sh`):

```bash
source /u/llindsey1/miniconda3/etc/profile.d/conda.sh
conda create -y -n nt python=3.11
conda activate nt
# 1) torch FIRST from the CUDA index — a plain pip install gives a CPU-only
#    aarch64 wheel. 2.5.1 + cu124 is the version proven working on Delta GH200
#    (matches the generanno/dnabert2 envs).
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# 2) everything else (transformers PINNED 4.49.0; torch is NOT in this file):
pip install -r requirements.txt
# verify on a GPU node (srun ... --partition=ghx4 --gpus-per-node=1):
python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda, transformers.__version__)"
```

`torch.cuda.is_available()` must be `True` before running the pipeline. Pitfalls
that bite on aarch64: a plain `pip install torch` → **CPU-only** wheel
(`cuda False`), and an unpinned `transformers` → **5.x**, which requires
torch ≥ 2.4 and breaks the 4.x-era NT-v2 scripts. Both are pinned now. No
flash-attn / Transformer-Engine compile is needed on GH200.

## Two-step workflow (Delta-AI / GH200)

On Delta the job bodies do **not** self-activate conda — they inherit the login
shell via `sbatch --export=ALL`. So **activate `nt` on the login node first**,
then run the drivers from there (the drivers only `sbatch`; they don't need a
GPU themselves).

```bash
# 0a. (every session) activate the env on the LOGIN node so jobs inherit it:
source /u/llindsey1/miniconda3/etc/profile.d/conda.sh
conda activate nt

# 0b. (one time) pre-warm the HF cache so the (possibly offline) compute nodes
#     can find the model. HF_HOME is read from lambda_replication.conf
#     (=/work/hdd/bfzj/llindsey1/hf_cache):
bash slurm_scripts/lambda_replication/prefetch_hf_cache.sh

# 1. Edit lambda_replication.conf — confirm LAMBDA_BASE + OUTPUT_DIR (Delta /work paths).
bash slurm_scripts/lambda_replication/run_lambda_training.sh   # finetune × seeds × windows
# 2. wait — squeue -u $USER
bash slurm_scripts/lambda_replication/check_training.sh        # confirm all seeds healthy
bash slurm_scripts/lambda_replication/run_lambda_inference.sh  # pick winner + all inference (+ PHROG)
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
- **PHROG annotated set.** `run_lambda_inference.sh` also runs the 2k winner on
  `PHROG_2k` (`fnr_test/2k/phage_annotated_segments_2k.csv`) and writes
  `NTv2_phage_annotated_segments_2k_predictions.csv` (canonical, model-prefixed)
  under `inference/<variant>/`. `inference_nt.py` does `output_df = df.copy()`,
  so the input's `phrog_category` / `phrog_db_category` columns pass through.
  This is distinct from the FNR sliding-window file.
- **Delta-AI paths.** Everything lives on `/work/hdd/bfzj/llindsey1/...`
  (LAMBDA_BASE, OUTPUT_DIR, HF_HOME). Home `/u` is tiny/inode-limited — nothing
  big goes there. The old Biowulf `/data/lindseylm`, `/vf/users`, `/gpfs` paths
  no longer exist for this account.
- **SLURM.** Drivers submit with `--account=bfzj-dtai-gh --partition=ghx4
  --gpus-per-node=1` (GH200). `SCRIPT_DIR` is resolved from each script's own
  location, so the repo can be cloned to any path on Delta.
