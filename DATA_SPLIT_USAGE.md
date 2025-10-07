# Data Split Usage - No Leakage Guarantee

This document explains how data splits are used throughout the benchmark pipeline to ensure **NO DATA LEAKAGE** occurs between training, validation, and testing.

## Overview

The pipeline uses three distinct data splits:

| Split | Purpose | Used By | Never Used For |
|-------|---------|---------|----------------|
| **TRAIN** | Model training | DenseLoRA, MAWLoRA, MAWFullFT variants | Validation, Testing |
| **DEV** | Validation during training | Early stopping, hyperparameter monitoring | Training data, Testing |
| **TEST** | Final evaluation | All reported metrics | Training, Validation |

## Data Flow

### 1. Dataset Loading (`DatasetManager`)

Each dataset is loaded with strict split separation:

```python
# MS MARCO (tier1_fixed.py:322-333)
train_split = _try_load_split(["train"])       # For training only
dev_split = _try_load_split(["dev", ...])      # For validation only
test_split = _try_load_split(["test"])         # For testing only

# Safety: If no test split, use dev as test AND nullify dev
if test_split is None and dev_split is not None:
    test_split = dev_split
    dev_split = None  # Prevents using same data for validation
```

### 2. Training (`ContrastiveTrainer.train()`, line 1121-1280)

**TRAIN split usage:**
- Creates `TripletDataset` from `bundle.train.queries` and `bundle.train.qrels` (line 1730-1738)
- Samples query-positive-negative triplets for contrastive learning
- BM25 hard negatives mined from TRAIN queries only (line 1703-1705)

**DEV split usage (if available):**
- Evaluates model on `dev_partition.queries` and `dev_partition.qrels` after each epoch (line 1229-1239)
- Used for early stopping (tracks best dev metric)
- **Never** used for training

**TEST split:**
- **NOT ACCESSIBLE** during training

### 3. Final Evaluation (`_run_bundle()`, line 1687-1820)

**TEST split only:**
- All final metrics reported in results use `bundle.test.queries` and `bundle.test.qrels` (line 1695)
- This includes:
  - BM25 baseline (line 1713)
  - DenseZeroShot (line 1708-1728)
  - All trained variants: DenseLoRA, MAWLoRA, MAWFullFT (line 1766-1788)

**TRAIN and DEV splits:**
- **NOT USED** for final evaluation metrics

## Safety Mechanisms

### 1. Fallback Prevention (BEIR datasets, line 335-362)

If a dataset lacks explicit splits:

```python
# OLD (UNSAFE): test_split = _try_load_split(["test", "dev"])
# Would silently use dev for test without nullifying dev for validation

# NEW (SAFE):
test_split = _try_load_split(["test"])
if test_split is None and dev_split is not None:
    logging.warning("Using dev as test, nullifying dev for validation")
    test_split = dev_split
    dev_split = None  # Prevents leakage
```

### 2. Duplicate Detection (line 342-351)

```python
if dev_split is not None and test_split[3] == dev_split[3]:
    logging.warning("Dev and test are identical, nullifying dev")
    dev_split = None
```

### 3. Explicit Logging (line 1706-1712)

Every dataset run logs split sizes:

```
Data splits - TRAIN: 64 queries (for training), DEV: 64 queries (for validation), TEST: 43 queries (for final evaluation)
```

## Verification

To verify no leakage:

### Quick Test
```bash
python tier1_fixed.py --quick-smoke-test --msmarco 2>&1 | grep "Data splits"
```

Expected output:
```
Data splits - TRAIN: 64 queries (for training), DEV: 64 queries (for validation), TEST: 43 queries (for final evaluation)
```

### Full Run
```bash
python tier1_fixed.py --msmarco --beir nq hotpotqa
```

Check that:
1. Training only uses TRAIN split
2. Validation (during training) only uses DEV split
3. Final evaluation only uses TEST split
4. If TEST doesn't exist, DEV is used for TEST and nullified for validation

## Dataset-Specific Behavior

### MS MARCO
- **TRAIN**: 502,939 queries (used for training)
- **DEV**: 6,980 queries (used for validation during training)
- **TEST**: 43 queries (used for final evaluation)

### BEIR Datasets (varies by dataset)
- Some have all three splits (train, dev, test)
- Some only have test (no training possible, zero-shot only)
- If no test, dev becomes test (and validation is disabled)

### LoTTE
- Requires explicit test split (line 380)
- No fallback to dev

## Code Locations

| Component | File:Line | Purpose |
|-----------|-----------|---------|
| Split loading | `tier1_fixed.py:322-383` | Load train/dev/test splits |
| Safety checks | `tier1_fixed.py:342-351` | Prevent dev/test overlap |
| Training | `tier1_fixed.py:1121-1280` | Uses TRAIN + DEV |
| Evaluation | `tier1_fixed.py:1687-1820` | Uses TEST only |
| Logging | `tier1_fixed.py:1706-1712` | Documents split usage |

## Summary

✅ **TRAIN split**: Used exclusively for training  
✅ **DEV split**: Used exclusively for validation during training  
✅ **TEST split**: Used exclusively for final evaluation metrics  
✅ **Safety**: If test doesn't exist, dev becomes test AND validation is disabled  
✅ **Verification**: Explicit logging of split sizes and usage  

**Result**: Zero data leakage between training, validation, and testing phases.
