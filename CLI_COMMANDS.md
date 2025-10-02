# CLI Command Reference

## ✅ Successfully Tested

Both scripts now support:
- ✅ GPU with CPU fallback
- ✅ Detailed device information printing  
- ✅ **Automatic log file generation** (JSON + TXT)
- ✅ Timestamped results saved to `logs/` directory

---

## 💾 Automatic Logging

**Every run automatically saves:**
- 📄 **JSON file**: Machine-readable results with full metrics (`logs/benchmark_grpo_YYYYMMDD_HHMMSS.json`)
- 📝 **Text file**: Human-readable summary (`logs/benchmark_grpo_YYYYMMDD_HHMMSS.txt`)

**Log contents include:**
- Timestamp and device info (GPU/CPU)
- All configuration parameters
- Complete results for all datasets
- Metrics for all K values
- Train/test split information

**Example log files:**
```
logs/
├── benchmark_grpo_20251002_160254.json          # GRPO run JSON
├── benchmark_grpo_20251002_160254.txt           # GRPO run summary
├── benchmark_supervised_20251002_160352.json    # Supervised run JSON
└── benchmark_supervised_20251002_160352.txt     # Supervised run summary
```

---

## 🚀 Quick Test Commands

### Minimal Test (Fastest - ~30 seconds)
```bash
# GRPO with 5 samples, 2 epochs
python benchmark_evaluation_GRPO.py --dataset MS_MARCO --samples 5 --epochs 2 --k-values 1 5 10

# Supervised Classification with 5 samples, 2 epochs
python benchmark_evaluation_Supervised_Classification.py --dataset MS_MARCO --samples 5 --epochs 2 --k-values 1 5 10
```

### Small Test (~2-3 minutes)
```bash
# GRPO with 10 samples, 5 epochs
python benchmark_evaluation_GRPO.py --dataset MS_MARCO --samples 10 --epochs 5

# Supervised Classification with 10 samples, 5 epochs
python benchmark_evaluation_Supervised_Classification.py --dataset TREC_DL --samples 10 --epochs 5
```

### Medium Test (~5-10 minutes)
```bash
# GRPO with 20 samples, 10 epochs
python benchmark_evaluation_GRPO.py --dataset MS_MARCO --samples 20 --epochs 10

# Supervised Classification with 20 samples, 10 epochs
python benchmark_evaluation_Supervised_Classification.py --dataset Natural_Questions --samples 20 --epochs 10
```

---

## 📚 Dataset Selection

### Single Dataset
```bash
python benchmark_evaluation_GRPO.py --dataset MS_MARCO --samples 15

python benchmark_evaluation_Supervised_Classification.py --dataset TREC_DL --samples 15
```

### Multiple Datasets
```bash
python benchmark_evaluation_GRPO.py --datasets MS_MARCO TREC_DL --samples 20

python benchmark_evaluation_Supervised_Classification.py --datasets MS_MARCO TREC_DL Natural_Questions --samples 15
```

### All Datasets (Full Benchmark)
```bash
# This will use full dataset sizes (takes hours)
python benchmark_evaluation_GRPO.py

python benchmark_evaluation_Supervised_Classification.py
```

---

## 🎮 Device Selection

### Auto-Detect (Default - Recommended)
```bash
# Uses GPU if available, otherwise CPU
python benchmark_evaluation_GRPO.py --dataset MS_MARCO --samples 10
```

### Force GPU
```bash
# Will fall back to CPU if GPU not available
python benchmark_evaluation_GRPO.py --dataset MS_MARCO --samples 20 --device cuda
```

### Force CPU
```bash
# Useful for testing or when GPU is busy
python benchmark_evaluation_GRPO.py --dataset MS_MARCO --samples 10 --device cpu
```

---

## ⚙️ Custom Configuration

### Custom Train/Test Split
```bash
# 80% train, 20% test
python benchmark_evaluation_GRPO.py --dataset MS_MARCO --samples 20 --train-ratio 0.8
```

### Custom K Values
```bash
# Only compute metrics for K=1,5,10,20
python benchmark_evaluation_GRPO.py --dataset MS_MARCO --samples 15 --k-values 1 5 10 20
```

### Custom Epochs
```bash
# More training
python benchmark_evaluation_Supervised_Classification.py --dataset TREC_DL --samples 20 --epochs 20

# Less training (faster)
python benchmark_evaluation_GRPO.py --dataset MS_MARCO --samples 10 --epochs 3
```

### Combined Custom Options
```bash
python benchmark_evaluation_GRPO.py \
    --dataset Natural_Questions \
    --samples 25 \
    --epochs 15 \
    --device cuda \
    --train-ratio 0.75 \
    --k-values 1 5 10 50 100
```

---

## 📊 Available Datasets

| Dataset Code | Full Name | Domain | Queries | Docs/Query |
|--------------|-----------|--------|---------|------------|
| `MS_MARCO` | MS MARCO Passage Ranking | Web Search | 50 | 50 |
| `TREC_DL` | TREC Deep Learning | DL Track | 40 | 50 |
| `Natural_Questions` | Natural Questions | Open QA | 35 | 40 |
| `SciDocs` | SciDocs Citation | Scientific | 30 | 45 |
| `FiQA` | FiQA Financial QA | Finance | 25 | 35 |

---

## 📋 Device Information Printed

Both scripts now print detailed device information:

### GPU Available:
```
🎮 Device: GPU (CUDA) - NVIDIA A40
   GPU Memory: 47.73 GB
📋 Configuration: hidden_dim=256, num_heads=8, depth_dim=32
...
   🔧 Data creation device: CUDA
   🔧 Training device: CUDA
   🔧 Evaluation device: CUDA
   🧹 GPU memory cleared
```

### CPU Only:
```
🖥️  Device: CPU
📋 Configuration: hidden_dim=256, num_heads=8, depth_dim=32
...
   🔧 Data creation device: CPU
   🔧 Training device: CPU
   🔧 Evaluation device: CPU
```

---

## 🆘 Help Commands

```bash
# View all options for GRPO
python benchmark_evaluation_GRPO.py --help

# View all options for Supervised Classification
python benchmark_evaluation_Supervised_Classification.py --help
```

---

## 💡 Recommended Workflow

1. **First Time:** Run minimal test to verify setup
   ```bash
   python benchmark_evaluation_GRPO.py --dataset MS_MARCO --samples 5 --epochs 2 --k-values 1 5 10
   ```

2. **Quick Exploration:** Test with ~10-20 samples
   ```bash
   python benchmark_evaluation_GRPO.py --dataset MS_MARCO --samples 15 --epochs 8
   python benchmark_evaluation_Supervised_Classification.py --dataset TREC_DL --samples 15 --epochs 8
   ```

3. **Compare Approaches:** Run both on same dataset
   ```bash
   python benchmark_evaluation_Supervised_Classification.py --dataset MS_MARCO --samples 20 --epochs 10
   python benchmark_evaluation_GRPO.py --dataset MS_MARCO --samples 20 --epochs 10
   ```

4. **Full Benchmark:** Run on all datasets (takes hours)
   ```bash
   python benchmark_evaluation_GRPO.py
   python benchmark_evaluation_Supervised_Classification.py
   ```

---

## 🎯 Example Output

```
🚀 MAW vs NON-MAW Evaluation with Real GRPO RL Algorithm
====================================================================================================
🎮 Device: GPU (CUDA) - NVIDIA A40
   GPU Memory: 47.73 GB
📋 Configuration: hidden_dim=256, num_heads=8, depth_dim=32
🔧 Training: 2 epochs | Train/Test Split: 70%/30%
📊 Evaluation metrics: Hit Rate, MRR, NDCG @ K=[1, 5, 10]
📚 Datasets to evaluate: MS_MARCO
🔢 Sample size: 5 queries per dataset

====================================================================================================
DATASET 1/1: MS MARCO Passage Ranking
====================================================================================================
   🔧 Data creation device: CUDA
   Split: 3 train, 2 test queries

🔨 Creating models on CUDA...
   ✅ NON-MAW model: 262912 parameters
   ✅ MAW+GRPO model: 759746 parameters

🔍 Evaluating NON-MAW baseline (zero-shot on test set)...
   🔧 Evaluation device: CUDA

🎯 Training MAW+GRPO RL on training set (3 queries, 2 epochs)...
   🔧 Training device: CUDA

📊 Evaluating MAW+GRPO RL on test set (2 queries)...
   🔧 Evaluation device: CUDA

📊 Results Table...
   🧹 GPU memory cleared
```

---

## ✅ Summary

- ✅ **GPU Support:** Auto-detects and uses GPU if available
- ✅ **CPU Fallback:** Automatically falls back to CPU if GPU unavailable
- ✅ **Device Info:** Prints device for all operations (data creation, training, evaluation)
- ✅ **Memory Management:** Clears GPU memory after each dataset
- ✅ **Flexible CLI:** Control samples, epochs, datasets, device, metrics
- ✅ **No Separate CLI Tool:** All functionality in main scripts
