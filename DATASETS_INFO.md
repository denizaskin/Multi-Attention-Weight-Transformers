# Supported Datasets (Training Sets Only)

## Overview

This codebase now **ONLY supports datasets with training sets** for fine-tuning MAW models.

## ✅ Supported Datasets

### 1. **MS MARCO** (Default)
- **Train**: 502,939 queries
- **Dev**: 6,980 queries  
- **Test**: 43 queries
- **Usage**: `--msmarco`
- **Description**: Microsoft Machine Reading Comprehension dataset with web passages

### 2. **LoTTE** (Long-Tail Topic-stratified Evaluation)
- **Splits**: `search` and `forum`
- **Each split includes**: train/dev/test partitions
- **Usage**: `--lotte search forum`
- **Description**: Long-tail queries across different domains

## ❌ Removed Datasets

### BEIR Datasets (Removed)
The following datasets were **REMOVED** because they **lack training sets**:
- `nq` (Natural Questions)
- `hotpotqa` (HotpotQA)
- `scifact` (SciFact)
- `fiqa` (FiQA)
- `trec-covid` (TREC-COVID)
- `arguana` (ArguAna)
- And other BEIR datasets

**Reason**: BEIR datasets are designed for zero-shot evaluation only and do not provide training data.

## Model Variants

All variants are evaluated on the supported datasets:

| Variant | Training Required | Description |
|---------|-------------------|-------------|
| **BM25** | ❌ No | Traditional keyword-based baseline |
| **DenseZeroShot** | ❌ No | Pre-trained encoder (no fine-tuning) |
| **DenseLoRA** | ✅ Yes | LoRA fine-tuning on dense encoder |
| **MAWLoRA** | ✅ Yes | LoRA + MAW on last layer |
| **MAWFullFT** | ✅ Yes | Full fine-tuning + MAW on last layer |

## Usage Examples

### Quick Smoke Test (~5-10 minutes)
```bash
python tier1_fixed.py --quick-smoke-test --msmarco
```

### Full MS MARCO Evaluation
```bash
python tier1_fixed.py --msmarco
```

### LoTTE Evaluation
```bash
python tier1_fixed.py --lotte search forum
```

### Both Datasets
```bash
python tier1_fixed.py --msmarco --lotte search
```

### Default (MS MARCO if no dataset specified)
```bash
python tier1_fixed.py
```

## Data Split Usage

All datasets use **strict train/dev/test separation**:
- **TRAIN**: Used for model fine-tuning
- **DEV**: Used for validation during training (early stopping)
- **TEST**: Used for final evaluation (reported metrics)

**No data leakage** between splits is guaranteed by the code.

## MAW Configuration

By default, MAW is applied to the **last encoder layer only**:
- `--maw-depth 64` (depth dimension)
- `--maw-heads 8` (attention heads)
- `--maw-layer-indices "-1"` (last layer only)

The 5D attention tensor has shape: `(batch, heads, seq_q, seq_k, depth)` and is scaled by √depth_dim for numerical stability.

## Summary

**Current Status**:
- ✅ 2 dataset families supported (MS MARCO, LoTTE)
- ✅ All datasets have training sets
- ✅ All 5 model variants can be trained and evaluated
- ❌ BEIR datasets removed (zero-shot only, no training data)

This ensures the benchmark focuses on **trainable retrieval models** rather than zero-shot evaluation.
