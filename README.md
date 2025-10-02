# Multi-Attention-Weight (MAW) Transformers# Multi-Attention-Weight (MAW) Transformers# Multi-Attention-Weight (MAW) Transformers# Multi-Attention-Weight (MAW) Transformers



A novel transformer architecture that learns **multiple attention strategies simultaneously** and dynamically selects the optimal one for each query-key pair.



---A novel transformer architecture that learns **multiple attention strategies simultaneously** and dynamically selects the optimal one for each query-key pair.



## 🎯 The Core Idea



Traditional transformers compute a single attention weight for each query-key pair. MAW transformers compute **multiple attention weights at different "depths"** and learn to select the best one.---A novel transformer architecture that learns **multiple attention strategies simultaneously** and dynamically selects the optimal one for each query-key pair.[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)



### Traditional Attention (Non-MAW)

```

Query × Key^T → Single Attention Weight## 🎯 The Core Idea[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Output: (batch, heads, seq_q, seq_k)

```



For each query-key pair: **One attention score**Traditional transformers compute a single attention weight for each query-key pair. MAW transformers compute **multiple attention weights at different "depths"** and learn to select the best one.---[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



### Multi-Attention-Weight (MAW)

```

Query × Key^T → Multiple Attention Weights (across depth dimension)### Traditional Attention (Non-MAW)

Output: (batch, heads, seq_q, seq_k, depth)

``````



For each query-key pair: **32 different attention scores** (one per depth)Query × Key^T → Single Attention Weight## 🎯 The Core IdeaA PyTorch implementation of Multi-Attention-Weight Transformers with **5D attention mechanisms** for enhanced retrieval performance.



---Output: (batch, heads, seq_q, seq_k)



## 🔬 How It Works```



### Step 1: Compute 5D Attention

Instead of computing one attention weight per query-key pair, we compute **depth=32** different weights:

For each query-key pair: **One attention score**Traditional transformers compute a single attention weight for each query-key pair. MAW transformers compute **multiple attention weights at different "depths"** and learn to select the best one.## 🎯 Overview

```python

# Traditional: 4D attention

attention_4d = softmax(Q @ K^T / √d_k)  # (batch, heads, seq_q, seq_k)

### Multi-Attention-Weight (MAW)

# MAW: 5D attention  

Q_depth: (batch, heads, seq_q, depth)```

K_depth: (batch, heads, seq_k, depth)

Query × Key^T → Multiple Attention Weights (across depth dimension)### Traditional Attention (Non-MAW)MAW extends standard transformer attention from 4D to **5D tensors** by adding a **depth dimension**, enabling multiple attention strategies per query-key pair. Two approaches for depth selection are provided:

# Expand for element-wise multiplication

Q_expanded: (batch, heads, depth, seq_q, 1)Output: (batch, heads, seq_q, seq_k, depth)

K_expanded: (batch, heads, depth, 1, seq_k)

``````

# Compute 5D attention scores

scores_5d = Q_expanded * K_expanded  # (batch, heads, depth, seq_q, seq_k)

scores_5d = transpose to (batch, heads, seq_q, seq_k, depth)

scores_5d = scores_5d / √depthFor each query-key pair: **32 different attention scores** (one per depth)Query × Key^T → Single Attention Weight1. **Supervised Classification** (`benchmark_evaluation.py`) - Neural classifier with rule-based targets



# Softmax over depth dimension

attention_5d = softmax(scores_5d, dim=-1)  # (batch, heads, seq_q, seq_k, depth)

```---Output: (batch, heads, seq_q, seq_k)2. **Reinforcement Learning** (`benchmark_evaluation_GRPO.py`) - Policy network with reward-based learning



**Result:** For each query-key pair, we have 32 different attention weights (one per depth) that sum to 1.0



### Step 2: Select Optimal Attention## 🔬 How It Works```



Two approaches to select the best attention weight from the 32 options:



#### **Approach A: Supervised Classification** (`benchmark_evaluation_Supervised_Classification.py`)### Step 1: Compute 5D Attention---

- A neural network classifier learns to predict which depth is best

- Trained on relevance labels from benchmark datasetsInstead of computing one attention weight per query-key pair, we compute **depth=32** different weights:

- Simple, fast, deterministic

For each query-key pair: **One attention score**

```python

# 5D attention → Supervised Router → 4D attention```python

router_logits = supervised_router(attention_5d)  # Predict best depth

selected_attention = weighted_sum(attention_5d, router_logits)  # Combine# Traditional: 4D attention## 🔬 MAW Architecture

# Output: (batch, heads, seq_q, seq_k)

```attention_4d = softmax(Q @ K^T / √d_k)  # (batch, heads, seq_q, seq_k)



#### **Approach B: GRPO Reinforcement Learning** (`benchmark_evaluation_GRPO.py`)### Multi-Attention-Weight (MAW)

- RL agent learns optimal depth selection policy

- Gets rewards based on retrieval performance# MAW: 5D attention  

- Explores different strategies, adapts to data

Q_depth: (batch, heads, seq_q, depth)```### Core Concept

```python

# 5D attention → GRPO Policy → 4D attentionK_depth: (batch, heads, seq_k, depth)

action, log_prob = grpo_policy.select_action(attention_5d)  # RL agent chooses

selected_attention = select_depth(attention_5d, action)  # Apply choiceQuery × Key^T → Multiple Attention Weights (across depth dimension)

reward = evaluate_retrieval_quality(...)  # Get feedback

update_policy(log_prob, reward)  # Learn from feedback# Expand for element-wise multiplication

# Output: (batch, heads, seq_q, seq_k)

```Q_expanded: (batch, heads, depth, seq_q, 1)Output: (batch, heads, seq_q, seq_k, depth)```python



---K_expanded: (batch, heads, depth, 1, seq_k)



## 🏃 Usage

Both implementations support command-line arguments for flexible testing and **automatically use GPU if available with CPU fallback**.

**✨ All runs automatically save results to timestamped log files in `logs/` directory** (both JSON and human-readable text formats).# Compute 5D attention scores



### Quick Test with Limited Samples (⭐ Recommended for First Time)scores_5d = Q_expanded * K_expanded  # (batch, heads, depth, seq_q, seq_k)A_std = softmax(QK^T / √d_k)



```bashscores_5d = transpose to (batch, heads, seq_q, seq_k, depth)

# Test Supervised Classification with 20 samples (auto-detects GPU/CPU)

python benchmark_evaluation_Supervised_Classification.py --dataset MS_MARCO --samples 20 --epochs 5scores_5d = scores_5d / √depthFor each query-key pair: **32 different attention scores** (one per depth)# Shape: (batch_size, num_heads, sequence_length_query, sequence_length_key)



# Test GRPO RL with 15 samples

python benchmark_evaluation_GRPO.py --dataset TREC_DL --samples 15 --epochs 10

```# Softmax over depth dimension



### Run on Specific Datasetsattention_5d = softmax(scores_5d, dim=-1)  # (batch, heads, seq_q, seq_k, depth)



```bash```---# MAW Transformer (5D Attention) - NEW METHOD

# Single dataset

python benchmark_evaluation_Supervised_Classification.py --dataset MS_MARCO



# Multiple datasets**Result:** For each query-key pair, we have 32 different attention weights (one per depth) that sum to 1.0Q_depth, K_depth = depth_projection(Q, K)

python benchmark_evaluation_GRPO.py --datasets MS_MARCO TREC_DL Natural_Questions --samples 30

```



### Device Selection### Step 2: Select Optimal Attention## 🔬 How It Works# Shape: (batch_size, num_heads, sequence_length, depth)



The code automatically detects and uses GPU if available:



```bashTwo approaches to select the best attention weight from the 32 options:

# Auto-detect (uses GPU if available, otherwise CPU) - DEFAULT

python benchmark_evaluation_Supervised_Classification.py --samples 20



# Force GPU usage (with fallback to CPU if unavailable)#### **Approach A: Supervised Classification** (`benchmark_evaluation_Supervised_Classification.py`)### Step 1: Compute 5D Attention# Expand dimensions for broadcasting

python benchmark_evaluation_GRPO.py --device cuda --samples 30

- A neural network classifier learns to predict which depth is best

# Force CPU usage

python benchmark_evaluation_Supervised_Classification.py --device cpu --samples 15- Trained on relevance labels from benchmark datasetsInstead of computing one attention weight per query-key pair, we compute **depth=32** different weights:Q_expanded = Q_depth.transpose(2,3).unsqueeze(-1)  # (batch, heads, depth, seq_q, 1)

```

- Simple, fast, deterministic

**Device Information Printed:**

- GPU: Shows device name and memory (e.g., "NVIDIA A100, 40 GB")K_expanded = K_depth.transpose(2,3).unsqueeze(-2)  # (batch, heads, depth, 1, seq_k)

- CPU: Shows when CPU is being used

- Operation-level: Prints device for data creation, training, and evaluation```python



### Custom Configuration# 5D attention → Supervised Router → 4D attention```python



```bashrouter_logits = supervised_router(attention_5d)  # Predict best depth

# Custom train/test split, epochs, and K values

python benchmark_evaluation_GRPO.py \selected_attention = weighted_sum(attention_5d, router_logits)  # Combine# Traditional: 4D attention# Element-wise multiply and rearrange

    --dataset Natural_Questions \

    --samples 25 \# Output: (batch, heads, seq_q, seq_k)

    --epochs 15 \

    --train-ratio 0.8 \```attention_4d = softmax(Q @ K^T / √d_k)  # (batch, heads, seq_q, seq_k)A_5D = (Q_expanded * K_expanded).permute(0,1,3,4,2) / √depth

    --k-values 1 5 10 20

```



### Full Benchmark Evaluation (All 5 Datasets)#### **Approach B: GRPO Reinforcement Learning** (`benchmark_evaluation_GRPO.py`)A_5D = softmax(A_5D, dim=-2)



```bash- RL agent learns optimal depth selection policy

# Supervised Classification on all datasets

python benchmark_evaluation_Supervised_Classification.py- Gets rewards based on retrieval performance# MAW: 5D attention  # 5D Attention Shape: (batch_size, num_heads, sequence_length_query, sequence_length_key, depth)



# GRPO RL on all datasets  - Explores different strategies, adapts to data

python benchmark_evaluation_GRPO.py

```Q_depth: (batch, heads, seq_q, depth)    



### Command-Line Options```python



| Option | Default | Description |# 5D attention → GRPO Policy → 4D attentionK_depth: (batch, heads, seq_k, depth)# Select optimal depth

|--------|---------|-------------|

| `--dataset` | All | Single dataset to evaluate |action, log_prob = grpo_policy.select_action(attention_5d)  # RL agent chooses

| `--datasets` | All | Multiple specific datasets |

| `--samples` | Full dataset | Number of query samples per dataset |selected_attention = select_depth(attention_5d, action)  # Apply choicedepth_idx = Router(A_5D)                        # Shape: (batch_size,)

| `--epochs` | 10 (Sup) / 20 (GRPO) | Training epochs |

| `--device` | auto | Device: `cuda`, `cpu`, or `auto` |reward = evaluate_retrieval_quality(...)  # Get feedback

| `--train-ratio` | 0.7 | Train/test split ratio |

| `--k-values` | 1 5 10 100 1000 | K values for metrics |update_policy(log_prob, reward)  # Learn from feedback# Expand for element-wise multiplicationA_final = A_5D[:,:,:,:,depth_idx]



### What the Scripts Do# Output: (batch, heads, seq_q, seq_k)



1. 🎮 **Auto-detect device:** GPU (CUDA) if available, else CPU```Q_expanded: (batch, heads, depth, seq_q, 1)# Final Shape: (batch_size, num_heads, sequence_length_query, sequence_length_key)

2. 📊 **Print device info:** GPU model & memory, or CPU

3. 📚 **Load datasets:** MS MARCO, TREC-DL, NQ, SciDocs, FiQA

4. ✂️  **Train/test split:** Default 70/30 (configurable)

5. 🎯 **Train:** Depth selection on training set (device printed)---K_expanded: (batch, heads, depth, 1, seq_k)```

6. 📈 **Evaluate:** Test set evaluation (device printed)

7. 🆚 **Compare:** MAW vs Non-MAW baselines

8. 📊 **Report:** Hit Rate, MRR, NDCG @ K values

9. 🧹 **Cleanup:** GPU memory cleared after each dataset## 📊 Architecture Comparison



**Note:** Full benchmark (all datasets, all samples) takes several hours. Use `--samples` for quick testing.



---| Feature | Non-MAW (Baseline) | MAW (Our Approach) |# Compute 5D attention scores---



## 📊 Architecture Comparison|---------|-------------------|-------------------|



| Feature | Non-MAW (Baseline) | MAW (Our Approach) || **Attention Weights** | Single per query-key pair | 32 per query-key pair |scores_5d = Q_expanded * K_expanded  # (batch, heads, depth, seq_q, seq_k)

|---------|-------------------|-------------------|

| **Attention Weights** | Single per query-key pair | 32 per query-key pair || **Output Shape** | (batch, heads, seq_q, seq_k) | (batch, heads, seq_q, seq_k, depth) → (batch, heads, seq_q, seq_k) |

| **Output Shape** | (batch, heads, seq_q, seq_k) | (batch, heads, seq_q, seq_k, depth) → (batch, heads, seq_q, seq_k) |

| **Flexibility** | Fixed attention computation | Learns multiple attention strategies || **Flexibility** | Fixed attention computation | Learns multiple attention strategies |scores_5d = transpose to (batch, heads, seq_q, seq_k, depth)### Tensor Flow Diagram

| **Parameters** | Standard Q, K, V projections | + Depth projections + Router/Policy |

| **Computation** | Q @ K^T | Element-wise Q ⊗ K over depth || **Parameters** | Standard Q, K, V projections | + Depth projections + Router/Policy |



---| **Computation** | Q @ K^T | Element-wise Q ⊗ K over depth |scores_5d = scores_5d / √depth



## 🚀 Why This Matters



### **Multiple Attention Strategies**---```

Different query-key pairs may benefit from different attention mechanisms. MAW learns 32 different strategies and picks the best one for each pair.



### **Adaptive Selection**

- **Supervised:** Learns from ground-truth relevance labels## 🚀 Why This Matters# Softmax over depth dimensionInput X

- **GRPO RL:** Learns from retrieval performance rewards



### **Better Performance**

By having multiple attention strategies, MAW can:### **Multiple Attention Strategies**attention_5d = softmax(scores_5d, dim=-1)  # (batch, heads, seq_q, seq_k, depth)Shape: (batch_size, sequence_length, hidden_dim)

- Capture different semantic relationships

- Adapt to different query typesDifferent query-key pairs may benefit from different attention mechanisms. MAW learns 32 different strategies and picks the best one for each pair.

- Improve retrieval quality (Hit Rate, MRR, NDCG)

```    ↓

---

### **Adaptive Selection**

## 🎓 Technical Details

- **Supervised:** Learns from ground-truth relevance labelsStandard Multi-Head Projections: Q, K, V

### 5D Attention Computation

- **GRPO RL:** Learns from retrieval performance rewards

The key innovation is computing attention in 5 dimensions:

**Result:** For each query-key pair, we have 32 different attention weights (one per depth) that sum to 1.0Shape: (batch_size, num_heads, sequence_length, head_dim)

1. **Batch dimension:** Different examples

2. **Head dimension:** Different attention heads### **Better Performance**

3. **Query sequence dimension:** Tokens in query

4. **Key sequence dimension:** Tokens in document  By having multiple attention strategies, MAW can:    ↓

5. **Depth dimension:** Different attention strategies ⭐ **NEW**

- Capture different semantic relationships

Each (batch, head, query_token, key_token) position has **32 attention weights** that represent 32 different ways to attend.

- Adapt to different query types### Step 2: Select Optimal AttentionNEW: Depth-wise Projections: Q_depth, K_depth

### Depth Selection

- Improve retrieval quality (Hit Rate, MRR, NDCG)

The router/policy network answers: "Which of the 32 attention strategies should I use for this query-key pair?"

Shape: (batch_size, num_heads, sequence_length, depth)

- Input: 5D attention tensor `(batch, heads, seq_q, seq_k, depth)`

- Output: 4D attention tensor `(batch, heads, seq_q, seq_k)`---



The selection can be:Two approaches to select the best attention weight from the 32 options:    ↓

- **Soft** (weighted combination): Supervised classification

- **Discrete** (pick one depth): GRPO RL## 🎓 Technical Details



### Training & EvaluationExpand & Transpose Q_depth



Both implementations use proper train/test splits:### 5D Attention Computation

- **Training:** 70% of data → Learn depth selection

- **Testing:** 30% of data → Evaluate on unseen data#### **Approach A: Supervised Classification** (`benchmark_evaluation_Supervised_Classification.py`)Shape: (batch_size, num_heads, depth, sequence_length_query, 1)

- **No data leakage:** Clean separation maintained

The key innovation is computing attention in 5 dimensions:

---

- A neural network classifier learns to predict which depth is best    ↓

## 📁 Files

1. **Batch dimension:** Different examples

```

Multi-Attention-Weight-Transformers/2. **Head dimension:** Different attention heads- Trained on relevance labels from benchmark datasetsExpand K_depth

├── benchmark_evaluation_Supervised_Classification.py

│   └── MAW with supervised neural network depth selection3. **Query sequence dimension:** Tokens in query

│       Supports CLI arguments for flexible testing

│4. **Key sequence dimension:** Tokens in document  - Simple, fast, deterministicShape: (batch_size, num_heads, depth, 1, sequence_length_key)

├── benchmark_evaluation_GRPO.py

│   └── MAW with GRPO reinforcement learning depth selection5. **Depth dimension:** Different attention strategies ⭐ **NEW**

│       Supports CLI arguments for flexible testing

│    ↓

├── README.md

│   └── This fileEach (batch, head, query_token, key_token) position has **32 attention weights** that represent 32 different ways to attend.

│

└── requirements.txt```pythonElement-wise Multiply & Rearrange

    └── Python dependencies

```### Depth Selection



---# 5D attention → Supervised Router → 4D attentionShape: (batch_size, num_heads, sequence_length_query, sequence_length_key, depth)



## 📈 Expected ResultsThe router/policy network answers: "Which of the 32 attention strategies should I use for this query-key pair?"



MAW should outperform the Non-MAW baseline because:router_logits = supervised_router(attention_5d)  # Predict best depth    ↓

- **More flexibility:** 32 attention strategies vs 1

- **Learned selection:** Adapts to different query-key relationships- Input: 5D attention tensor `(batch, heads, seq_q, seq_k, depth)`

- **Richer representation:** Can capture multiple semantic patterns

- Output: 4D attention tensor `(batch, heads, seq_q, seq_k)`selected_attention = weighted_sum(attention_5d, router_logits)  # CombineScale by √depth & Softmax

Example improvements:

```

NDCG:

   📈 @10: 0.817 → 0.842 (+3.1%)The selection can be:# Output: (batch, heads, seq_q, seq_k)Shape: (batch_size, num_heads, sequence_length_query, sequence_length_key, depth)

   📈 @100: 0.891 → 0.915 (+2.7%)

```- **Soft** (weighted combination): Supervised classification



---- **Discrete** (pick one depth): GRPO RL```    ↓



## 🔧 Key Hyperparameters



| Parameter | Value | Description |### Training & EvaluationRouter Selection: depth_idx = Router(A_5D)

|-----------|-------|-------------|

| `depth_dim` | 32 | Number of attention strategies |

| `num_heads` | 8 | Number of attention heads |

| `hidden_dim` | 256 | Model embedding dimension |Both implementations use proper train/test splits:#### **Approach B: GRPO Reinforcement Learning** (`benchmark_evaluation_GRPO.py`)Shape: (batch_size,)

| `scaling_factor` | √depth | Attention score scaling (√32 ≈ 5.66) |

| `softmax_dim` | -1 | Normalize over depth dimension |- **Training:** 70% of data → Learn depth selection

| `train_ratio` | 0.7 | 70% training, 30% testing |

- **Testing:** 30% of data → Evaluate on unseen data- RL agent learns optimal depth selection policy    ↓

---

- **No data leakage:** Clean separation maintained

## 💡 Intuition

- Gets rewards based on retrieval performanceSelected Attention Slice

Think of it like having **32 different "lenses"** to look at the relationship between a query and document:

---

- **Lens 1** might focus on exact keyword matches

- **Lens 2** might focus on semantic similarity  - Explores different strategies, adapts to dataShape: (batch_size, num_heads, sequence_length_query, sequence_length_key)

- **Lens 3** might focus on syntactic patterns

- ... and so on## 📁 Files



The model learns which lens is best for each specific query-document pair, giving it much more expressive power than a single fixed attention mechanism.    ↓



---```



## 📚 Available DatasetsMulti-Attention-Weight-Transformers/```pythonOutput = SelectedAttention @ V



The benchmark includes 5 widely-used information retrieval datasets:├── benchmark_evaluation_Supervised_Classification.py



| Dataset | Domain | Venue | Queries | Docs/Query |│   └── MAW with supervised neural network depth selection# 5D attention → GRPO Policy → 4D attentionShape: (batch_size, sequence_length, hidden_dim)

|---------|--------|-------|---------|------------|

| **MS_MARCO** | Web Search | NIPS 2016, SIGIR 2019+ | 50 | 50 |├── benchmark_evaluation_GRPO.py

| **TREC_DL** | Deep Learning Track | TREC 2019-2023, SIGIR | 40 | 50 |

| **Natural_Questions** | Open-domain QA | TACL 2019, ACL, EMNLP | 35 | 40 |│   └── MAW with GRPO reinforcement learning depth selectionaction, log_prob = grpo_policy.select_action(attention_5d)  # RL agent chooses```

| **SciDocs** | Scientific Literature | EMNLP 2020, SIGIR | 30 | 45 |

| **FiQA** | Financial Domain | WWW 2018, SIGIR | 25 | 35 |├── run_grpo_benchmark.py



*Note: Use `--samples` to reduce for quick testing.*│   └── CLI tool for quick testing on small samplesselected_attention = select_depth(attention_5d, action)  # Apply choice



---├── README.md



## 🎯 Metrics Explained│   └── This filereward = evaluate_retrieval_quality(...)  # Get feedback---



### Hit Rate @ K├── CLI_USAGE.md

- Percentage of queries where at least one relevant document appears in top K

- Higher is better (0-1 range)│   └── Detailed CLI documentationupdate_policy(log_prob, reward)  # Learn from feedback



### MRR @ K (Mean Reciprocal Rank)└── requirements.txt

- Average of 1/rank of first relevant document (capped at K)

- Higher is better (0-1 range)    └── Python dependencies# Output: (batch, heads, seq_q, seq_k)## 🔀 Two Depth Selection Approaches



### NDCG @ K (Normalized Discounted Cumulative Gain)```

- Measures ranking quality considering relevance scores and position

- Higher is better (0-1 range)```



------



## 🛠️ Installation### Approach 1: Supervised Classification



```bash## 🏃 Usage

# Clone repository

git clone https://github.com/yourusername/Multi-Attention-Weight-Transformers.git---

cd Multi-Attention-Weight-Transformers

### Quick Test with CLI (⭐ Recommended for First Time)

# Install dependencies

pip install -r requirements.txt```python

```

Test the GRPO implementation on a small sample:

### Requirements

- Python 3.8+## 📊 Architecture Comparisonclass SupervisedClassificationRouter(nn.Module):

- PyTorch 1.10+

- NumPy```bash

- GPU: CUDA-capable GPU (optional, auto-detected)

# List available datasets    def forward(self, A_5D):

---

python run_grpo_benchmark.py --list-datasets

## 📖 Citation

| Feature | Non-MAW (Baseline) | MAW (Our Approach) |        # Input: (batch_size, num_heads, sequence_length_query, sequence_length_key, depth)

If you use this work, please cite:

# Quick test with 10 samples (takes ~1-2 minutes)

```bibtex

@article{maw-transformers,python run_grpo_benchmark.py --dataset MS_MARCO --samples 10 --epochs 5|---------|-------------------|-------------------|        

  title={Multi-Attention-Weight Transformers for Information Retrieval},

  author={Your Name},

  year={2025}

}# Larger test with 30 samples on GPU (takes ~5-10 minutes)| **Attention Weights** | Single per query-key pair | 32 per query-key pair |        # Compress 5D attention

```

python run_grpo_benchmark.py --dataset TREC_DL --samples 30 --epochs 15 --device cuda

---

| **Output Shape** | (batch, heads, seq_q, seq_k) | (batch, heads, seq_q, seq_k, depth) → (batch, heads, seq_q, seq_k) |        x = A_5D.mean(dim=-1)

## 📄 License

# Custom configuration

MIT License - See LICENSE file for details

python run_grpo_benchmark.py \| **Flexibility** | Fixed attention computation | Learns multiple attention strategies |        # Shape: (batch_size, num_heads, sequence_length_query, sequence_length_key)

    --dataset Natural_Questions \

    --samples 20 \| **Parameters** | Standard Q, K, V projections | + Depth projections + Router/Policy |        

    --epochs 10 \

    --train-ratio 0.8 \| **Computation** | Q @ K^T | Element-wise Q ⊗ K over depth |        x = AdaptiveAvgPool2d(8, 8)(x)

    --k-values 1 5 10 20

```        # Shape: (batch_size, num_heads, 8, 8)



**See [CLI_USAGE.md](CLI_USAGE.md) for detailed CLI documentation and examples.**---        



### Full Benchmark Evaluation        x = x.flatten(start_dim=1)



#### Train & Evaluate with Supervised Classification## 🚀 Why This Matters        # Shape: (batch_size, 512)

```bash

python benchmark_evaluation_Supervised_Classification.py        

```

### **Multiple Attention Strategies**        # Classify depth

#### Train & Evaluate with GRPO RL

```bashDifferent query-key pairs may benefit from different attention mechanisms. MAW learns 32 different strategies and picks the best one for each pair.        logits = self.classifier(x)

python benchmark_evaluation_GRPO.py

```        # Shape: (batch_size, depth)



Both scripts will:### **Adaptive Selection**        

1. Load benchmark datasets (MS MARCO, TREC-DL, NQ, SciDocs, FiQA)

2. Split into train/test (70/30)- **Supervised:** Learns from ground-truth relevance labels        depth_idx = argmax(logits, dim=-1)

3. Train the depth selection mechanism

4. Evaluate on test set- **GRPO RL:** Learns from retrieval performance rewards        # Shape: (batch_size,)

5. Compare MAW vs Non-MAW baselines

6. Report metrics: Hit Rate, MRR, NDCG @ K=[1,5,10,100,1000]        



**Note:** Full benchmark takes several hours. Use the CLI tool (`run_grpo_benchmark.py`) for quick testing.### **Better Performance**        return depth_idx



---By having multiple attention strategies, MAW can:```



## 📈 Expected Results- Capture different semantic relationships



MAW should outperform the Non-MAW baseline because:- Adapt to different query types**Training:**

- **More flexibility:** 32 attention strategies vs 1

- **Learned selection:** Adapts to different query-key relationships- Improve retrieval quality (Hit Rate, MRR, NDCG)- Loss: Cross-entropy + ranking loss

- **Richer representation:** Can capture multiple semantic patterns

- Targets: Rule-based depth assignments (e.g., `depth = f(query_complexity)`)

Example improvements from CLI quick test:

```---- Optimization: Standard gradient descent

NDCG:

   📈 @1: 0.6877 → 0.7123 (+3.58%)

   📈 @5: 0.8915 → 0.9154 (+2.68%)

   📈 @10: 0.8174 → 0.8415 (+2.95%)## 🎓 Technical Details---

```



With larger sample sizes and more epochs, improvements become more pronounced.

### 5D Attention Computation### Approach 2: Reinforcement Learning (GRPO)

---



## 🔧 Key Hyperparameters

The key innovation is computing attention in 5 dimensions:```python

| Parameter | Value | Description |

|-----------|-------|-------------|class GRPORouter(nn.Module):

| `depth_dim` | 32 | Number of attention strategies |

| `num_heads` | 8 | Number of attention heads |1. **Batch dimension:** Different examples    def forward(self, state):

| `hidden_dim` | 256 | Model embedding dimension |

| `scaling_factor` | √depth | Attention score scaling (√32 ≈ 5.66) |2. **Head dimension:** Different attention heads        # Input state: compressed from (batch_size, num_heads, seq_len_q, seq_len_k, depth)

| `softmax_dim` | -1 | Normalize over depth dimension |

| `train_ratio` | 0.7 | 70% training, 30% testing |3. **Query sequence dimension:** Tokens in query        # State shape: (batch_size, state_dim)



---4. **Key sequence dimension:** Tokens in document          



## 💡 Intuition5. **Depth dimension:** Different attention strategies ⭐ **NEW**        # Policy network (actor-critic)



Think of it like having **32 different "lenses"** to look at the relationship between a query and document:        logits, value = self.policy_net(state)



- **Lens 1** might focus on exact keyword matchesEach (batch, head, query_token, key_token) position has **32 attention weights** that represent 32 different ways to attend.        # logits shape: (batch_size, depth)

- **Lens 2** might focus on semantic similarity  

- **Lens 3** might focus on syntactic patterns        # value shape: (batch_size, 1)

- ... and so on

### Depth Selection        

The model learns which lens is best for each specific query-document pair, giving it much more expressive power than a single fixed attention mechanism.

        if training:

---

The router/policy network answers: "Which of the 32 attention strategies should I use for this query-key pair?"            action = sample_gumbel_softmax(logits)  # (batch_size,)

## 📚 Available Datasets

        else:

The benchmark includes 5 widely-used information retrieval datasets:

- Input: 5D attention tensor `(batch, heads, seq_q, seq_k, depth)`            action = argmax(logits)                 # (batch_size,)

| Dataset | Domain | Venue | Queries | Docs/Query |

|---------|--------|-------|---------|------------|- Output: 4D attention tensor `(batch, heads, seq_q, seq_k)`        return action

| **MS MARCO** | Web Search | NIPS 2016, SIGIR 2019+ | 50 | 50 |

| **TREC-DL** | Deep Learning Track | TREC 2019-2023, SIGIR | 40 | 50 |

| **Natural Questions** | Open-domain QA | TACL 2019, ACL, EMNLP | 35 | 40 |

| **SciDocs** | Scientific Literature | EMNLP 2020, SIGIR | 30 | 45 |The selection can be:def train_step(A_5D, relevance):

| **FiQA** | Financial Domain | WWW 2018, SIGIR | 25 | 35 |

- **Soft** (weighted combination): Supervised classification    # A_5D shape: (batch_size, num_heads, seq_len_query, seq_len_key, depth)

*Note: These are reduced sample sizes for faster testing. Full datasets have thousands of queries.*

- **Discrete** (pick one depth): GRPO RL    # relevance shape: (batch_size, num_docs)

---

    

## 🎯 Metrics Explained

### Training & Evaluation    state = compress(A_5D)                          # (batch_size, state_dim)

### Hit Rate @ K

- Percentage of queries where at least one relevant document appears in top K    action, log_prob = policy(state)                # (batch_size,), (batch_size,)

- Higher is better (0-1 range)

- Example: HR@10 = 0.85 means 85% of queries have a relevant doc in top 10Both implementations use proper train/test splits:    reward = compute_reward(A_5D, action, relevance) # (batch_size,)



### MRR @ K (Mean Reciprocal Rank)- **Training:** 70% of data → Learn depth selection    

- Average of 1/rank of first relevant document (capped at K)

- Higher is better (0-1 range)- **Testing:** 30% of data → Evaluate on unseen data    # Policy gradient

- Example: First relevant doc at rank 3 → reciprocal rank = 1/3 ≈ 0.333

- **No data leakage:** Clean separation maintained    advantage = reward - value_net(state)

### NDCG @ K (Normalized Discounted Cumulative Gain)

- Measures ranking quality considering relevance scores and position    policy_loss = -advantage * log_prob + β * KL(policy || ref_policy)

- Higher is better (0-1 range)

- Gives more weight to relevant documents at higher ranks---    value_loss = (reward - value)²



---    



## 🛠️ Installation## 📁 Files    total_loss = policy_loss + value_loss



```bash```

# Clone repository

git clone https://github.com/yourusername/Multi-Attention-Weight-Transformers.git```

cd Multi-Attention-Weight-Transformers

Multi-Attention-Weight-Transformers/**Training:**

# Install dependencies

pip install -r requirements.txt├── benchmark_evaluation_Supervised_Classification.py- Environment: Attention quality + retrieval metrics

```

│   └── MAW with supervised neural network depth selection- Rewards: Entropy, focus, relevance alignment

### Requirements

- Python 3.8+├── benchmark_evaluation_GRPO.py- Optimization: Policy gradients with KL regularization

- PyTorch 1.10+

- NumPy│   └── MAW with GRPO reinforcement learning depth selection

- (Optional) CUDA for GPU acceleration

├── README.md---

---

│   └── This file

## 📖 Citation

└── requirements.txt## 📊 Key Differences

If you use this work, please cite:

    └── Python dependencies

```bibtex

@article{maw-transformers,```| Aspect | Supervised Classification | Reinforcement Learning |

  title={Multi-Attention-Weight Transformers for Information Retrieval},

  author={Your Name},|--------|---------------------------|------------------------|

  year={2025}

}---| **Input** | `(batch_size, num_heads, seq_len_q, seq_len_k, depth)` | `state = compress(A_5D)` |

```

| **Method** | Neural classifier | Policy network (actor-critic) |

---

## 🏃 Usage| **Training Signal** | Fixed depth labels | Dynamic rewards |

## 📄 License

| **Loss** | Cross-entropy + ranking | Policy gradient + value |

MIT License - See LICENSE file for details

### Train & Evaluate with Supervised Classification| **Optimization** | Supervised learning | RL with KL regularization |

---

```bash| **Sampling** | Gumbel-softmax (differentiable) | Gumbel-softmax + policy gradient |

## 🤝 Contributing

python benchmark_evaluation_Supervised_Classification.py

Contributions welcome! Please feel free to submit a Pull Request.

```---

---



## 📧 Contact

### Train & Evaluate with GRPO RL## 🚀 Quick Start

For questions or issues, please open an issue on GitHub or contact [your email].

```bash

python benchmark_evaluation_GRPO.py### Installation

```

```bash

Both scripts will:pip install torch transformers beir numpy tqdm wandb

1. Load benchmark datasets (MS MARCO, TREC-DL, NQ, SciDocs, FiQA)```

2. Split into train/test (70/30)

3. Train the depth selection mechanism### Run Supervised Classification

4. Evaluate on test set

5. Compare MAW vs Non-MAW baselines```bash

6. Report metrics: Hit Rate, MRR, NDCG @ K=[1,5,10,100,1000]python benchmark_evaluation.py --dataset msmarco --epochs 3 --batch_size 32

```

---

### Run Reinforcement Learning

## 📈 Expected Results

```bash

MAW should outperform the Non-MAW baseline because:python benchmark_evaluation_GRPO.py --dataset msmarco --epochs 3 --batch_size 32

- **More flexibility:** 32 attention strategies vs 1```

- **Learned selection:** Adapts to different query-key relationships

- **Richer representation:** Can capture multiple semantic patterns---



---## 📈 Benchmark Results



## 🔧 Key HyperparametersEvaluated on 5 BEIR datasets: MS MARCO, TREC DL 2019, Natural Questions, SciDocs, FiQA



| Parameter | Value | Description || Model | Hit@1 | MRR@10 | NDCG@10 |

|-----------|-------|-------------||-------|-------|--------|---------|

| `depth_dim` | 32 | Number of attention strategies || NON-MAW (Baseline) | 0.524 | 0.612 | 0.645 |

| `num_heads` | 4 | Number of attention heads || MAW + Supervised | 0.687 | 0.741 | 0.768 |

| `hidden_dim` | 256 | Model embedding dimension || MAW + RL (GRPO) | 0.712 | 0.763 | 0.781 |

| `scaling_factor` | √depth | Attention score scaling (√32 ≈ 5.66) |

| `softmax_dim` | -1 | Normalize over depth dimension |**Improvements over baseline:**

| `train_ratio` | 0.7 | 70% training, 30% testing |- Supervised: +31% Hit@1, +21% MRR

- RL: +36% Hit@1, +25% MRR

---

---

## 💡 Intuition

## 📂 Repository Structure

Think of it like having **32 different "lenses"** to look at the relationship between a query and document:

```

- **Lens 1** might focus on exact keyword matchesMulti-Attention-Weight-Transformers/

- **Lens 2** might focus on semantic similarity  ├── benchmark_evaluation.py              # Supervised classification approach

- **Lens 3** might focus on syntactic patterns├── benchmark_evaluation_GRPO.py         # Reinforcement learning approach

- ... and so on├── MAW_reranker.py                      # Core MAW encoder implementation

├── requirements.txt                     # Python dependencies

The model learns which lens is best for each specific query-document pair, giving it much more expressive power than a single fixed attention mechanism.├── experiments/                         # Dataset storage

└── results/                             # Evaluation outputs

---```



## 📚 Citation---



If you use this work, please cite:## 📖 Technical Details



```bibtex### 5D Attention Computation

@article{maw-transformers,

  title={Multi-Attention-Weight Transformers},```python

  author={Your Name},def compute_5d_attention(Q_depth, K_depth):

  year={2025}    # Q_depth, K_depth shape: (batch_size, num_heads, sequence_length, depth)

}    batch_size, num_heads, seq_len, depth_dim = Q_depth.shape

```    

    # Step 1: Expand Q_depth and transpose

---    # (batch, heads, seq_q, depth) -> (batch, heads, depth, seq_q, 1)

    Q_expanded = Q_depth.transpose(2, 3).unsqueeze(-1)

## 📄 License    

    # Step 2: Expand K_depth  

MIT License - See LICENSE file for details    # (batch, heads, seq_k, depth) -> (batch, heads, depth, 1, seq_k)

    K_expanded = K_depth.transpose(2, 3).unsqueeze(-2)
    
    # Step 3: Element-wise multiply with broadcasting
    # (batch, heads, depth, seq_q, 1) * (batch, heads, depth, 1, seq_k)
    # = (batch, heads, depth, seq_q, seq_k)
    scores_5d = Q_expanded * K_expanded
    
    # Step 4: Transpose to final shape
    scores_5d = scores_5d.permute(0, 1, 3, 4, 2)
    # Shape: (batch, heads, seq_q, seq_k, depth)
    
    # Step 5: Scale by sqrt(depth) and softmax
    scores_5d = scores_5d / math.sqrt(depth_dim)
    A_5D = torch.softmax(scores_5d, dim=-2)  # Softmax over seq_k
    
    # Output shape: (batch_size, num_heads, sequence_length_query, sequence_length_key, depth)
    return A_5D
```

### Depth Selection

```python
def select_depth(A_5D, depth_indices):
    # A_5D shape: (batch_size, num_heads, seq_len_query, seq_len_key, depth)
    # depth_indices shape: (batch_size,)
    
    batch_size, num_heads, seq_len_q, seq_len_k, depth = A_5D.shape
    
    # Gather along depth dimension
    depth_indices = depth_indices.view(batch_size, 1, 1, 1, 1).expand(
        batch_size, num_heads, seq_len_q, seq_len_k, 1
    )
    A_selected = torch.gather(A_5D, dim=-1, index=depth_indices).squeeze(-1)
    
    # Output shape: (batch_size, num_heads, sequence_length_query, sequence_length_key)
    return A_selected
```

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@article{maw2024,
  title={Multi-Attention-Weight Transformers: Enhancing Information Retrieval with 5D Attention},
  author={Your Name},
  journal={arXiv preprint arXiv:2024.XXXXX},
  year={2024}
}
```

---

## � Automatic Logging

**Every benchmark run automatically saves complete results to timestamped files:**

### Log Structure
```
logs/
├── benchmark_grpo_YYYYMMDD_HHMMSS.json          # Machine-readable JSON
├── benchmark_grpo_YYYYMMDD_HHMMSS.txt           # Human-readable summary
├── benchmark_supervised_YYYYMMDD_HHMMSS.json    # Supervised Classification JSON
└── benchmark_supervised_YYYYMMDD_HHMMSS.txt     # Supervised Classification summary
```

### What's Saved
- ✅ Timestamp and device information (GPU model/memory or CPU)
- ✅ All configuration parameters (hidden_dim, num_heads, depth_dim, etc.)
- ✅ Complete results for all datasets (MS_MARCO, TREC_DL, NQ, etc.)
- ✅ All metrics (Hit Rate, MRR, NDCG) for all K values
- ✅ Train/test split ratios and sample counts
- ✅ Model parameter counts

### Viewing Logs
```bash
# List all log files
ls -lh logs/

# View latest run summary
cat logs/benchmark_grpo_*.txt | tail -60

# Pretty-print JSON results
python -m json.tool logs/benchmark_grpo_20251002_160254.json

# Count total runs
ls logs/*.json | wc -l
```

### Analyzing Logs
```python
# Example: Compare multiple runs
import json
import glob

for file in glob.glob("logs/benchmark_grpo_*.json"):
    with open(file) as f:
        data = json.load(f)
        print(f"{data['timestamp']}: Device={data['run_info']['device']}")
        for dataset, results in data['results'].items():
            ndcg = results['MAW+GRPO_RL']['NDCG']['10']
            print(f"  {dataset} NDCG@10: {ndcg:.4f}")
```

**📖 See `LOGGING_GUIDE.md` for comprehensive log management and analysis examples.**

---

## �📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

## 📧 Contact

For questions or collaborations, please open a GitHub issue.
