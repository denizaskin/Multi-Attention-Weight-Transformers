# What Gets Trained and Tested in Each File

## Answer to Your Question

**YES** - Both files train and test exactly what you described:

---

## `benchmark_evaluation.py`

### **Trains:**
```python
SupervisedClassificationRouter:
    Input:  (batch_size, num_heads, sequence_length_query, sequence_length_key, depth)
    Output: (batch_size, num_heads, sequence_length_query, sequence_length_key)
```

### **How:**
1. **Training Phase** (on training set):
   ```python
   # Line 549-650: train_supervised_classification_on_dataset()
   
   # Step 1: Compute 5D attention weights from MAW encoder
   attention_weights_5d = compute_5d_attention(Q_depth, K_depth)
   # Shape: (1, num_heads, seq_len, seq_len, depth_dim)
   
   # Step 2: Router predicts depth logits
   router_logits = model.supervised_router(attention_weights_5d)
   # Shape: (1, depth_dim)
   
   # Step 3: Rule-based target depth (based on number of relevant docs)
   if num_relevant >= 3:
       target_depth = 0
   elif num_relevant >= 2:
       target_depth = 1
   # ... etc
   
   # Step 4: Classification loss
   loss = CrossEntropyLoss(router_logits, target_depth) + ranking_loss
   
   # Step 5: Backprop and update router weights
   loss.backward()
   optimizer.step()
   ```

2. **Testing Phase** (on test set):
   ```python
   # Line 776: Evaluate on unseen test queries
   maw_results = evaluate_model_on_dataset(
       maw_model, "MAW+SupervisedClassification", 
       test_queries, test_documents, test_relevance
   )
   
   # During evaluation, the trained router selects depths:
   depth_idx = model.supervised_router.get_depth_selection(attention_5d)
   selected_attention = attention_5d[:, :, :, :, depth_idx]
   ```

### **What is Trained:**
- Only the `SupervisedClassificationRouter` parameters
- The MAW encoder (5D attention computation) is NOT trained
- Trains to classify: 5D tensor → optimal depth index

---

## `benchmark_evaluation_GRPO.py`

### **Trains:**
```python
GRPORouter (with Policy Network + Environment):
    Input:  (batch_size, num_heads, sequence_length_query, sequence_length_key, depth)
    Output: (batch_size, num_heads, sequence_length_query, sequence_length_key)
```

### **How:**
1. **Training Phase** (on training set):
   ```python
   # Line 719-818: train_grpo_rl_on_dataset()
   
   # Step 1: Compute 5D attention weights from MAW encoder (same as supervised)
   attention_weights_5d = compute_5d_attention(Q_depth, K_depth)
   # Shape: (1, num_heads, seq_len, seq_len, depth_dim)
   
   # Step 2: GRPO RL computes loss (NOT classification!)
   grpo_loss = model.grpo_router.compute_grpo_loss(
       attention_weights_5d, query_rel, baseline_scores
   )
   
   # Inside compute_grpo_loss():
   # a) Reset environment with 5D attention
   state = environment.reset(attention_weights_5d, relevance, baseline)
   
   # b) Policy network samples action (depth selection)
   actions, log_probs, values = policy.get_action(state)
   
   # c) Environment computes reward
   rewards = environment.step(actions)
   
   # d) Compute GRPO loss
   policy_loss = -mean(ratio * advantages) + β*KL(π||π_ref)
   value_loss = MSE(values, rewards)
   total_loss = policy_loss + value_loss + entropy_loss
   
   # Step 3: Backprop and update policy weights
   grpo_loss.backward()
   optimizer.step()
   ```

2. **Testing Phase** (on test set):
   ```python
   # Line 980: Evaluate on unseen test queries
   maw_results = evaluate_model_on_dataset(
       maw_model, "MAW+GRPO_RL",
       test_queries, test_documents, test_relevance
   )
   
   # During evaluation, the trained RL policy selects depths:
   depth_idx = model.grpo_router.get_depth_selection(attention_5d)
   selected_attention = attention_5d[:, :, :, :, depth_idx]
   ```

### **What is Trained:**
- Only the `GRPORouter.policy` (policy network) parameters
- The MAW encoder (5D attention computation) is NOT trained
- Trains to learn: 5D tensor → optimal depth index (via RL rewards)

---

## Key Differences in Training

| Aspect | Supervised Classification | GRPO RL |
|--------|---------------------------|---------|
| **What trains** | `SupervisedClassificationRouter` | `GRPOPolicyNetwork` |
| **Input to router** | 5D attention tensor directly | Compressed state from 5D tensor |
| **Training signal** | Fixed depth labels (rule-based) | Dynamic rewards (performance) |
| **Loss function** | Cross-entropy + ranking | Policy gradient + value + entropy |
| **Training iterations** | 10 epochs | 20 epochs |
| **Updates per epoch** | Router classification weights | Policy network weights |

---

## Common Aspects

Both files:

1. ✅ Create **train/test splits** (70/30) to prevent data leakage
2. ✅ Train **only the router/policy** (not the MAW encoder)
3. ✅ Compute the same **5D attention tensor** from MAW encoder
4. ✅ Select **one depth** per batch item to get back to 4D attention
5. ✅ Evaluate on **unseen test data** (proper ML evaluation)

---

## Execution Flow

### `benchmark_evaluation.py`:
```
For each dataset:
  1. Create train/test split
  2. Evaluate NON-MAW baseline (zero-shot on test set)
  3. Train SupervisedClassificationRouter on training set
  4. Test trained router on test set
  5. Compare: NON-MAW vs MAW+SupervisedClassification
```

### `benchmark_evaluation_GRPO.py`:
```
For each dataset:
  1. Create train/test split
  2. Evaluate NON-MAW baseline (zero-shot on test set)
  3. Train GRPOPolicyNetwork on training set (using RL)
  4. Test trained policy on test set
  5. Compare: NON-MAW vs MAW+GRPO_RL
```

---

## Summary

**Your understanding is CORRECT!**

Both files:
- **Input to router:** `(batch_size, num_heads, sequence_length_query, sequence_length_key, depth)`
- **Output from router:** `(batch_size, num_heads, sequence_length_query, sequence_length_key)`
- **Training:** Learn to select optimal depth from 5D tensor
- **Testing:** Apply learned selection on unseen data

The only difference is **HOW** they learn to select the depth:
- `benchmark_evaluation.py`: Supervised classification with rule-based targets
- `benchmark_evaluation_GRPO.py`: Reinforcement learning with performance rewards
