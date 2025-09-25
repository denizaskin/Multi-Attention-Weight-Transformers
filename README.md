# MAW Reranker

This project provides a reproduction-ready reranking pipeline that compares a Multi-Attention Window (MAW) variant of a cross-encoder against a non-MAW baseline on standard IR benchmarks (MS MARCO dev, TREC DL, and selected BEIR tasks). Candidate pools are generated using BM25, re-ranked with identical pools across variants, and evaluated with official trec-style metrics via `ir_measures`. The codebase also logs MAW depth statistics, performs multi-seed runs, calculates significance tests, and emits experiment artifacts for paper-grade reporting.

## Key Features
- **True reranking setup**: Top-1000 candidate pools built with Pyserini/ir\_datasets and cached to `runs/<dataset>/bm25_top1000.trec`.
- **Variant framework**: Toggle MAW injection depth, gating modes, LoRA fine-tuning, and hyperparameter sweeps through `VariantConfig` definitions.
- **Official metrics**: Uses `ir_measures` for MRR@10, nDCG@10, and recall@1000 depending on dataset.
- **Multi-seed evaluation**: Runs each variant over ≥3 seeds, aggregates mean±std, and applies paired t-tests (SciPy) on per-query gains.
- **MAW introspection**: Logs depth-weight distributions to JSONL for later analysis.
- **Experiment artifacts**: Saves runfiles, per-query metrics, depth logs, configs, environment metadata, git commit, and system info to `experiments/`.
- **One-click scripts**: `build_bm25.sh`, `gen_pools.sh`, `rerank_eval.sh`, and `suite_runner.sh` orchestrate the common workflows.

## Environment Setup
1. **Create virtual environment** (optional):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, use Conda:
   ```bash
   conda env create -f environment.yml
   conda activate maw-reranker
   ```
3. **Optional services**:
   - Set `ENABLE_WANDB=1` and export `WANDB_API_KEY` if you want Weights & Biases logging.

## Preparing Candidate Pools
All reranking requires BM25 pools per dataset:
```bash
./build_bm25.sh                          # primary benchmarks
./gen_pools.sh                           # build for all registered datasets
python MAW_reranker.py --mode build-pools --datasets MSMARCO/dev-small
```
Use `--force-pools` to refresh cached runs.

## Running Experiments
### Full Suite (defaults + ablations + dev sweep)
```bash
./suite_runner.sh
```
This expands the dataset list with secondary BEIR tasks, runs all registered variants, and performs a dev-only hyperparameter sweep on MS MARCO.

### Baseline vs MAW (minimal)
```bash
./rerank_eval.sh MSMARCO/dev-small TREC-DL-2019-passage
```
Runs only the default variants (`non_maw`, `maw_default`).

### CLI Options
`python MAW_reranker.py --help` summarizes available flags. Common patterns:
- `--mode suite|build-pools|dev-sweep`
- `--datasets <space separated dataset keys>`
- `--variants default|ablations|all`
- `--include-secondary` to append BEIR sanity datasets
- `--with-dev-sweep` to enable the MS MARCO hyperparameter grid
- `--seeds 13 17 29` to override seed list

## Outputs and Artifacts
Each run is stored under `experiments/<dataset>/<variant>/seed_<seed>/` and contains:
- `config.json`: configuration, seed, git commit hash, system info.
- `per_query_metrics.json`: per-query metric values from `ir_measures`.
- `per_query_scores.jsonl`: reranked document scores for each query.
- `<variant>_seed<seed>.trec`: TREC-format runfile.
- `maw_depth.jsonl` (MAW variants): depth routing distributions.

Suite summaries land in `experiments/summary/`, e.g. `suite_<timestamp>.json`.

## Variant Catalogue
Key variants defined in `MAW_reranker.py`:
- `non_maw`: baseline without MAW injection.
- `maw_default`: MAW injected into the final attention layer (`inject_last_k=1`).
- Ablations: depth sizes (`maw_depth1`, `maw_depth4`, `maw_depth8`), gating modes (`uniform`, `random`, `argmax`), multiple layer injection (`maw_inject_last2`), and LoRA fine-tuning (`maw_lora_finetune`).
- Dev sweep (MS MARCO): grid over `maw_strength ∈ {0.02,0.03,0.05}` × `depth_dim ∈ {4,8}`.

## Troubleshooting
- **`ir_measures` missing**: install via `pip install ir-measures` (already listed in requirements).
- **`pyserini` index errors**: ensure Java is installed and that prebuilt indexes are accessible; rerun with `--force-pools` if runs are corrupt.
- **`scipy` missing**: install for significance tests (`pip install scipy`).
- **`git` errors when logging commit hash**: run inside a git repo or ignore the warning.
- **WANDB credential prompts**: set `WANDB_API_KEY` env var or leave `ENABLE_WANDB` unset to skip logging.

## Citation
If you build on this work, please cite the original MAW reranker paper (add once published) and the Pyserini/ir\_datasets tooling that powers candidate retrieval.

