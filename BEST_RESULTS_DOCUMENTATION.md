# BEST_RESULTS.json Documentation

## Overview

The `BEST_RESULTS.json` file is automatically generated at the end of the Tier-1 benchmark evaluation. It contains the **best results for each dataset and method** along with complete hyperparameters for reproducibility.

**Location**: `logs/tier1/BEST_RESULTS.json`

---

## File Structure

```json
{
  "metadata": {
    "benchmark_name": "TIER-1 Multi-Attention-Weight Transformers Evaluation",
    "generated_at": "2025-10-06T14:35:22.123456",
    "timestamp": "20251006_143522",
    "total_datasets": 4,
    "methods_per_dataset": 3,
    "description": "Best results for each dataset and method with hyperparameters"
  },
  "global_configuration": {
    "seed": 42,
    "num_layers": 6,
    "hidden_dim": 768,
    "num_heads": 12,
    "depth_dim": 32
  },
  "datasets": {
    "MS MARCO": {
      "dataset_info": {...},
      "methods": {
        "zero_shot": {...},
        "lora_fine_tuned": {...},
        "maw_fine_tuned": {...}
      }
    },
    "BEIR Natural Questions": {...},
    "BEIR HotpotQA": {...},
    "BEIR TriviaQA": {...}
  }
}
```

---

## Metadata Section

```json
"metadata": {
  "benchmark_name": "TIER-1 Multi-Attention-Weight Transformers Evaluation",
  "generated_at": "2025-10-06T14:35:22.123456",  // ISO timestamp
  "timestamp": "20251006_143522",                 // Filesystem-safe timestamp
  "total_datasets": 4,                            // Number of datasets evaluated
  "methods_per_dataset": 3,                       // Methods per dataset (zero-shot, LoRA, MAW)
  "description": "Best results for each dataset and method with hyperparameters"
}
```

---

## Global Configuration

Common settings across all evaluations:

```json
"global_configuration": {
  "seed": 42,              // Random seed for reproducibility
  "num_layers": 6,         // Transformer layers
  "hidden_dim": 768,       // Hidden dimension (BERT-base standard)
  "num_heads": 12,         // Attention heads
  "depth_dim": 32          // MAW depth dimension
}
```

---

## Dataset Structure

Each dataset contains:

### Dataset Info
```json
"dataset_info": {
  "name": "MS MARCO",
  "type": "msmarco",
  "venue": "MSFT/TREC",
  "train_size": 2000,      // Training queries
  "val_size": 500,         // Validation queries
  "test_size": 1000        // Test queries
}
```

### Methods

Each dataset has **3 methods**:

---

## Method 1: Zero-Shot

```json
"zero_shot": {
  "method_name": "Zero-shot Retrieval",
  "description": "Off-the-shelf retriever without any fine-tuning",
  "training": "None - No training performed",
  "hyperparameters": {
    "note": "No training - using pre-initialized weights"
  },
  "best_metrics": {
    "nDCG@10": 0.6123,
    "MAP": 0.5234,
    "Recall@100": 0.7456,
    "MRR@10": 0.6789,
    // ... all 36 TIER-1 metrics
  },
  "runtime": {
    "seconds": 45.23,
    "minutes": 0.75
  }
}
```

**Key Points:**
- No training performed
- Baseline performance
- All 36 metrics included

---

## Method 2: LoRA Fine-Tuned

```json
"lora_fine_tuned": {
  "method_name": "LoRA Fine-tuned Retrieval",
  "description": "Parameter-efficient fine-tuning using LoRA adapters",
  "training": "LoRA adapters trained on last layer",
  "hyperparameters": {
    "use_lora": true,
    "lora_rank": 8,
    "lora_alpha": 16,
    "num_epochs": 10,
    "batch_size": 32,
    "eval_batch_size": 128,
    "learning_rate": 1e-5,
    "warmup_steps": 1000,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "best_epoch": 8,                      // Epoch with best validation score
    "best_validation_ndcg10": 0.7234,     // Best validation nDCG@10
    "trainable_parameters": "LoRA adapters only (~thousands of parameters)"
  },
  "training_history": {
    "train_loss": [0.3, 0.25, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.16],
    "val_mrr": [0.65, 0.68, 0.70, 0.71, 0.72, 0.72, 0.73, 0.73, 0.73, 0.73],
    "val_ndcg": [0.64, 0.67, 0.69, 0.70, 0.71, 0.71, 0.72, 0.72, 0.72, 0.72]
  },
  "best_metrics": {
    "nDCG@10": 0.7234,
    "MAP": 0.6345,
    // ... all 36 TIER-1 metrics from best epoch
  },
  "runtime": {
    "seconds": 152.67,
    "minutes": 2.54
  },
  "improvements_vs_zeroshot": {
    "nDCG@10": {
      "absolute": 0.1111,      // Absolute improvement
      "relative_percent": 18.14 // Percentage improvement
    },
    "MAP": {
      "absolute": 0.1111,
      "relative_percent": 21.23
    }
    // ... improvements for all 36 metrics
  }
}
```

**Key Points:**
- Complete hyperparameters for reproducibility
- Training history (losses, validation scores per epoch)
- Best epoch identified
- All 36 metrics from best checkpoint
- Improvements vs zero-shot for all metrics

---

## Method 3: MAW Fine-Tuned

```json
"maw_fine_tuned": {
  "method_name": "MAW Fine-tuned Retrieval",
  "description": "Multi-Attention-Weight architecture with GRPO router and selective layer fine-tuning",
  "training": "Last layer + GRPO router fine-tuned",
  "hyperparameters": {
    "architecture": "MAW (Multi-Attention-Weight)",
    "maw_layers": [6],                    // Layers with MAW enabled
    "finetune_layers": [6],               // Layers fine-tuned
    "depth_dim": 32,
    "grpo_enabled": true,
    "num_epochs": 10,
    "batch_size": 32,
    "eval_batch_size": 128,
    "learning_rate": 1e-5,
    "warmup_steps": 1000,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "best_epoch": 7,
    "best_validation_ndcg10": 0.7456,
    "trainable_parameters": "Full last layer + GRPO router (~millions of parameters)"
  },
  "training_history": {
    "train_loss": [0.28, 0.23, 0.21, 0.20, 0.19, 0.18, 0.17, 0.17, 0.16, 0.16],
    "val_mrr": [0.67, 0.70, 0.72, 0.73, 0.73, 0.74, 0.74, 0.74, 0.74, 0.74],
    "val_ndcg": [0.66, 0.69, 0.71, 0.72, 0.73, 0.74, 0.75, 0.75, 0.74, 0.74]
  },
  "best_metrics": {
    "nDCG@10": 0.7456,
    "MAP": 0.6567,
    // ... all 36 TIER-1 metrics from best epoch
  },
  "runtime": {
    "seconds": 185.32,
    "minutes": 3.09
  },
  "improvements_vs_zeroshot": {
    "nDCG@10": {
      "absolute": 0.1333,
      "relative_percent": 21.76
    }
    // ... improvements for all 36 metrics
  },
  "improvements_vs_lora": {
    "nDCG@10": {
      "absolute": 0.0222,
      "relative_percent": 3.07
    }
    // ... improvements for all 36 metrics
  }
}
```

**Key Points:**
- Complete MAW architecture settings
- GRPO router configuration
- Training history with validation scores
- Best epoch identified
- All 36 metrics from best checkpoint
- Improvements vs both zero-shot AND LoRA

---

## Usage Examples

### Load and Parse

```python
import json

# Load BEST_RESULTS.json
with open('logs/tier1/BEST_RESULTS.json', 'r') as f:
    best_results = json.load(f)

# Access metadata
print(f"Generated at: {best_results['metadata']['generated_at']}")
print(f"Total datasets: {best_results['metadata']['total_datasets']}")

# Access a specific dataset
ms_marco = best_results['datasets']['MS MARCO']
print(f"Dataset: {ms_marco['dataset_info']['name']}")
print(f"Venue: {ms_marco['dataset_info']['venue']}")

# Access zero-shot results
zero_shot = ms_marco['methods']['zero_shot']
print(f"Zero-shot nDCG@10: {zero_shot['best_metrics']['nDCG@10']:.4f}")

# Access LoRA results and hyperparameters
lora = ms_marco['methods']['lora_fine_tuned']
print(f"LoRA nDCG@10: {lora['best_metrics']['nDCG@10']:.4f}")
print(f"LoRA best epoch: {lora['hyperparameters']['best_epoch']}")
print(f"LoRA learning rate: {lora['hyperparameters']['learning_rate']}")

# Access MAW results and hyperparameters
maw = ms_marco['methods']['maw_fine_tuned']
print(f"MAW nDCG@10: {maw['best_metrics']['nDCG@10']:.4f}")
print(f"MAW best epoch: {maw['hyperparameters']['best_epoch']}")
print(f"MAW layers: {maw['hyperparameters']['maw_layers']}")

# Get improvements
improvement = maw['improvements_vs_lora']['nDCG@10']
print(f"MAW vs LoRA improvement: {improvement['absolute']:+.4f} ({improvement['relative_percent']:+.2f}%)")
```

### Compare Methods Across Datasets

```python
import json
import pandas as pd

with open('logs/tier1/BEST_RESULTS.json', 'r') as f:
    best_results = json.load(f)

# Extract nDCG@10 for all methods and datasets
data = []
for dataset_name, dataset_data in best_results['datasets'].items():
    methods = dataset_data['methods']
    data.append({
        'Dataset': dataset_name,
        'Zero-shot': methods['zero_shot']['best_metrics']['nDCG@10'],
        'LoRA': methods['lora_fine_tuned']['best_metrics']['nDCG@10'],
        'MAW': methods['maw_fine_tuned']['best_metrics']['nDCG@10']
    })

df = pd.DataFrame(data)
print(df)
```

### Extract Hyperparameters for Reproduction

```python
import json

with open('logs/tier1/BEST_RESULTS.json', 'r') as f:
    best_results = json.load(f)

# Get LoRA hyperparameters for MS MARCO
lora_params = best_results['datasets']['MS MARCO']['methods']['lora_fine_tuned']['hyperparameters']

print("Reproduce LoRA training with:")
print(f"  lora_rank: {lora_params['lora_rank']}")
print(f"  lora_alpha: {lora_params['lora_alpha']}")
print(f"  learning_rate: {lora_params['learning_rate']}")
print(f"  batch_size: {lora_params['batch_size']}")
print(f"  num_epochs: {lora_params['num_epochs']}")
print(f"  Stop at epoch: {lora_params['best_epoch']}")
```

---

## Key Features

✅ **All Results in One Place**: Single file with best results from all methods and datasets  
✅ **Complete Hyperparameters**: Full training configuration for reproducibility  
✅ **Training History**: Loss curves and validation metrics per epoch  
✅ **Best Epoch Tracking**: Identifies which epoch achieved best performance  
✅ **Improvement Analysis**: Absolute and relative improvements for all metrics  
✅ **Runtime Information**: Execution time for each method  
✅ **36 TIER-1 Metrics**: All comprehensive metrics included  

---

## Benefits

1. **Quick Overview**: See best results at a glance
2. **Reproducibility**: All hyperparameters documented
3. **Analysis**: Compare methods across datasets
4. **Publication**: Ready for tables/figures in papers
5. **Debugging**: Training history shows convergence
6. **Efficiency**: No need to parse multiple files

---

## File Size

Typical size: **~100-200 KB** (uncompressed)

Contains:
- 4 datasets × 3 methods = 12 evaluations
- Each with 36 metrics + hyperparameters + training history
- All improvements calculated

---

## When is it Created?

- **Timing**: At the very end of the complete benchmark run
- **Location**: `logs/tier1/BEST_RESULTS.json`
- **Always Created**: Yes, regardless of compression settings
- **Overwritten**: No, unique timestamp per run

---

## Related Files

- **BEST_RESULTS.json** ← Start here (best results + hyperparameters)
- `tier1_complete_benchmark_*.json` - Full detailed results
- `*_results.json.gz` - Per-dataset compressed results
- `checkpoints/tier1/` - Model weights for best epochs

---

## Summary

The `BEST_RESULTS.json` file is your **go-to file** for:
- 📊 Comparing methods across datasets
- 🔬 Reproducing experiments (all hyperparameters included)
- 📈 Analyzing training dynamics (history included)
- 📝 Creating publication tables/figures
- 🎯 Understanding improvements (vs zero-shot and vs LoRA)

**It contains everything you need in one clean, organized file!** 🏆
