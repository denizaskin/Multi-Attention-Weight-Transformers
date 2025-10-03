# Quick Implementation Roadmap for Additional Baselines

## 🎯 Goal
Add 4 key baselines to strengthen the paper for Tier-1 journal submission.

---

## 📋 Implementation Priority Order

### 1. Random Depth Selection (1 hour) 🟢 EASY
**Purpose:** Shows that learning is necessary, not just selection

**Implementation:**
```python
class RandomDepthEncoder(nn.Module):
    """Baseline: Randomly select depth instead of learning"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.base_attention = MultiHeadAttention5D(config)
    
    def forward(self, query, key, value):
        # Compute 5D attention
        attention_5d = self.base_attention(query, key, value)
        # (batch, heads, seq_q, seq_k, depth)
        
        batch_size = attention_5d.size(0)
        
        # RANDOM selection (no learning!)
        random_depth = torch.randint(
            0, self.config.depth_dim, 
            (batch_size,), 
            device=attention_5d.device
        )
        
        # Select depth
        attention_4d = attention_5d[:, :, :, :, random_depth]
        
        # Apply to values
        output = torch.matmul(attention_4d, value)
        return output
```

**Changes Needed:**
- Add class to both files
- Add to evaluation loop
- Add to results comparison

**Expected Result:** Should perform worse than MAW but better than Non-MAW

---

### 2. Fixed Depth Selection (1 hour) 🟢 EASY
**Purpose:** Shows adaptive selection is better than fixed strategy

**Implementation:**
```python
class FixedDepthEncoder(nn.Module):
    """Baseline: Always use middle depth (no adaptation)"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.base_attention = MultiHeadAttention5D(config)
        self.fixed_depth = config.depth_dim // 2  # Always use middle
    
    def forward(self, query, key, value):
        # Compute 5D attention
        attention_5d = self.base_attention(query, key, value)
        
        # FIXED selection (same for all queries!)
        attention_4d = attention_5d[:, :, :, :, self.fixed_depth]
        
        # Apply to values
        output = torch.matmul(attention_4d, value)
        return output
```

**Changes Needed:**
- Add class to both files
- Add to evaluation loop
- Add to results comparison

**Expected Result:** Should perform between Random and MAW

---

### 3. Multi-Head 32 Heads (30 minutes) 🟢 VERY EASY
**Purpose:** Shows depth diversity different from head diversity

**Implementation:**
```python
# Just modify config for baseline
config_32_heads = Config(
    hidden_dim=256,
    num_heads=32,  # ← Changed from 8
    seq_len=128,
    vocab_size=30000,
    dropout=0.1
)

# Use standard Non-MAW encoder with 32 heads
baseline_32heads = NonMAWEncoder(config_32_heads)
```

**Changes Needed:**
- Add separate config variant
- Add to evaluation loop
- Track as separate baseline

**Expected Result:** Should perform better than Non-MAW but worse than MAW

---

### 4. Fine-tuned Projections (2 hours) 🟡 MEDIUM
**Purpose:** Controls for parameter count - shows architecture matters

**Implementation:**
```python
class FineTunedProjectionsEncoder(nn.Module):
    """Baseline: Add learnable projections (same param count as MAW router)"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Base attention
        self.base_attention = MultiHeadAttention(config)
        
        # Add learnable transformations (similar params to MAW router)
        self.query_transform = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)
        )
        
        self.key_transform = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)
        )
    
    def forward(self, query, key, value):
        # Transform query and key
        query_ft = self.query_transform(query)
        key_ft = self.key_transform(key)
        
        # Standard attention with transformed inputs
        output = self.base_attention(query_ft, key_ft, value)
        return output
```

**Changes Needed:**
- Add class to both files
- Ensure similar parameter count to MAW
- Add to evaluation loop
- Compare parameters explicitly in results

**Expected Result:** Should perform better than Non-MAW but worse than MAW (shows architecture > just parameters)

---

## 🔧 Integration Steps

### Step 1: Add Baseline Classes (Both Files)
Add all 4 baseline classes to both:
- `benchmark_evaluation_GRPO.py`
- `benchmark_evaluation_Supervised_Classification.py`

### Step 2: Update Evaluation Loop
```python
# Current: Only Non-MAW vs MAW
# New: Non-MAW vs Random vs Fixed vs 32-heads vs Fine-tuned vs MAW

baselines = {
    'NON-MAW': NonMAWEncoder(config),
    'Random-Depth': RandomDepthEncoder(config),
    'Fixed-Depth': FixedDepthEncoder(config),
    '32-Heads': NonMAWEncoder(config_32_heads),
    'Fine-tuned-Proj': FineTunedProjectionsEncoder(config)
}

# Evaluate each baseline
for name, model in baselines.items():
    results[name] = evaluate_model_on_dataset(
        model, name, test_queries, test_documents, 
        test_relevance, device, k_values
    )

# Train and evaluate MAW
maw_results = train_and_evaluate_maw(...)
results['MAW'] = maw_results
```

### Step 3: Update Results Table
```python
def print_results_with_all_baselines(results, k_values):
    print("=" * 120)
    print(f"{'Model':<25} {'Params':<10}", end="")
    for k in k_values:
        print(f" NDCG@{k:<5}", end="")
    print()
    print("-" * 120)
    
    # Print each baseline
    for model_name in ['NON-MAW', 'Random-Depth', 'Fixed-Depth', 
                       '32-Heads', 'Fine-tuned-Proj', 'MAW']:
        print(f"{model_name:<25} {get_param_count(model):<10}", end="")
        for k in k_values:
            print(f" {results[model_name]['NDCG'][k]:<6.3f}", end="")
        print()
```

### Step 4: Update Logging
Ensure all baselines are saved to log files:
```json
{
  "results": {
    "MS_MARCO": {
      "NON-MAW": {...},
      "Random-Depth": {...},
      "Fixed-Depth": {...},
      "32-Heads": {...},
      "Fine-tuned-Proj": {...},
      "MAW": {...}
    }
  }
}
```

---

## ⏱️ Time Estimate

| Task | Time | Cumulative |
|------|------|------------|
| 1. Random Depth | 1h | 1h |
| 2. Fixed Depth | 1h | 2h |
| 3. 32 Heads | 0.5h | 2.5h |
| 4. Fine-tuned Proj | 2h | 4.5h |
| 5. Integration | 1h | 5.5h |
| 6. Testing | 1h | 6.5h |
| 7. Documentation | 0.5h | 7h |

**Total: ~7 hours of development work**

---

## 📊 Expected Results Pattern

```
Model                 | Params | NDCG@10 | Pattern
---------------------|--------|---------|----------------------------------
NON-MAW              | 262K   | 0.645   | Baseline (worst)
Random-Depth         | 470K   | 0.658   | Slightly better (random helps a bit)
Fixed-Depth          | 470K   | 0.661   | Slightly better than random
32-Heads             | 520K   | 0.668   | More heads help
Fine-tuned-Proj      | 475K   | 0.671   | Similar params to MAW, decent
---------------------|--------|---------|----------------------------------
MAW (Supervised)     | 470K   | 0.703   | ← Best (shows architecture wins)
MAW (GRPO)           | 760K   | 0.718   | ← Best overall
```

**Key Insights from Pattern:**
1. Learning > Random > None (validates learning)
2. Adaptive > Fixed (validates adaptation)
3. MAW > Fine-tuned (architecture > just parameters)
4. MAW > 32 Heads (depth diversity > head diversity)

---

## 📝 Paper Writing Impact

### Abstract:
"We compare MAW against multiple baselines including random/fixed depth selection, 
increased attention heads, and fine-tuned projections, demonstrating consistent 
improvements across all configurations."

### Experiments Section:
```markdown
## 4. Experiments

### 4.1 Baselines
We compare against five baselines:
1. **Non-MAW**: Standard attention without depth selection
2. **Random Depth**: Random depth selection (no learning)
3. **Fixed Depth**: Always selects middle depth
4. **32-Head Attention**: Increased head count (more capacity)
5. **Fine-tuned Projections**: Learnable Q/K transforms (matched params)

### 4.2 Results
Table 1 shows MAW outperforms all baselines across datasets...

### 4.3 Ablation Studies
Random/Fixed selection shows that learning is critical...
```

---

## 🎯 Deliverables

After implementation:
1. ✅ Both files support 6 models (5 baselines + MAW)
2. ✅ Consistent evaluation across all models
3. ✅ Logs capture all baseline results
4. ✅ Results tables show all comparisons
5. ✅ Parameter counts clearly documented
6. ✅ Ready for paper writing

---

## 🚀 Next Steps

**Option A: Implement All Now**
- Takes ~7 hours
- Complete baseline comparison
- Strong paper foundation

**Option B: Implement Tier 1 First**
- Takes ~2.5 hours (Random + Fixed + 32 Heads)
- Quick win for acceptable paper
- Add more later if needed

**Option C: Wait for User Decision**
- Discuss which baselines are priority
- Implement based on submission timeline
- Can always add more later

---

## 💡 My Recommendation

**Implement in this order:**

### Week 1: Core Baselines (3 hours)
1. Random Depth (validates learning)
2. Fixed Depth (validates adaptation)
3. 32 Heads (validates depth vs head diversity)

**Goal:** Acceptable paper with solid ablations

### Week 2: Competitive Baselines (3 hours)
4. Fine-tuned Projections (controls parameters)

**Goal:** Strong paper that addresses obvious alternatives

### Later: Advanced Baselines (Optional)
5. Mixture of Experts (if reviewers ask)
6. Neural Architecture Search (if going for top venue)

**Goal:** Exceptional paper for NeurIPS/ICLR

---

Would you like me to start implementing these baselines? I can begin with the easy ones (Random, Fixed, 32 Heads) which would take ~2-3 hours and significantly strengthen your paper.
