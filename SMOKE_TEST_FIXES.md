# Smoke Test Fixes

## Issues Fixed

### 1. Dataset Loading Issues
**Problem**: MS MARCO dataset was reporting "missing a required evaluation split" error.

**Root Causes**:
- **Nested Directory Structure**: The BEIR download created `msmarco/msmarco/` nested structure that wasn't being flattened properly
- **GenericDataLoader State Pollution**: The BEIR `GenericDataLoader` filters its internal `queries` dict after loading the first split, causing KeyError when trying to load subsequent splits with the same instance

**Fixes**:
- Improved `_ensure_beir_dataset()` to properly flatten nested directories and verify all required files exist
- Modified `_try_load_split()` to create a fresh `GenericDataLoader` instance for each split to avoid state pollution
- Changed `_ensure_beir_dataset()` to return the dataset path string instead of a cached loader instance

### 2. BM25 Implementation
**Problem**: BEIR's `BM25Search` requires Elasticsearch and doesn't support the `index_dir` parameter.

**Fix**: Replaced with `rank_bm25` (pure Python implementation):
- Uses `BM25Okapi` with k1=0.9, b=0.4 parameters
- Simple tokenization (lowercase + split on whitespace)
- No external dependencies beyond the rank_bm25 package

### 3. PyTorch autocast API Deprecation
**Problem**: `torch.cuda.amp.autocast(device_type=...)` is deprecated and causes `TypeError`.

**Fix**: Updated all autocast calls to use the simpler `autocast(dtype=...)` signature:
- Line 1063: Evaluation encoding
- Line 1000: Training context manager

### 4. GradScaler Deprecation Warning
**Issue**: `torch.cuda.amp.GradScaler(args...)` shows deprecation warning (non-breaking).

**Note**: Could be updated to `torch.amp.GradScaler('cuda', args...)` in future revisions.

## Running the Smoke Test

### Single Process (CPU/Single GPU)
```bash
python tier1_fixed.py --quick-smoke-test --msmarco
```

### Multi-Process (Multi-GPU via torchrun)
```bash
python run_smoke_test.py --nproc 4 --master-port 29511 --extra-args --msmarco --beir nq hotpotqa
```

### With Multiple Datasets
```bash
python tier1_fixed.py --quick-smoke-test --msmarco --beir nq hotpotqa scifact fiqa
```

## Smoke Test Scope
- **Queries**: 64 per split (train/dev) or all if fewer
- **Documents**: 2000 from corpus (includes all relevant docs for sampled queries)
- **Variants Tested**:
  - BM25 baseline
  - Dense zero-shot (no training)
  - Dense + LoRA (fine-tuned)
  - Dense + LoRA + MAW (fine-tuned with multi-attention)
  - Dense + MAW full fine-tuning

## Expected Runtime
- Single dataset (MS MARCO smoke): ~5-10 minutes
- Multiple datasets: ~15-30 minutes depending on number of datasets

## Files Modified
- `tier1_fixed.py`: Main benchmark script with all fixes applied

## Dependencies Required
- `rank-bm25`: For BM25 baseline (already in requirements.txt)
- `beir`: For dataset loading
- `torch`: Version compatible with amp API
- All other dependencies from `requirements.txt`
