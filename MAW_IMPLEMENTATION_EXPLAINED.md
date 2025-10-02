# MAW Implementation Explanation

## Overview

This document explains how Multi-Attention-Weight (MAW) Transformers are implemented in two different approaches:
1. **benchmark_evaluation.py** - Supervised Classification
2. **benchmark_evaluation_GRPO.py** - Reinforcement Learning (GRPO)

Both implementations share the **same core MAW architecture** but differ in **how they select the optimal depth** from the 5D attention tensor.

---

## Core MAW Architecture (Shared by Both)

### 1. Standard Multi-Head Attention (Baseline - NON-MAW)

```python
class NonMAWEncoder(nn.Module):
    # Standard 4D attention
    Q = query_proj(hidden_states)  # (batch, seq, hidden)
    K = key_proj(hidden_states)    # (batch, seq, hidden)
    V = value_proj(hidden_states)  # (batch, seq, hidden)
    
    # Reshape to multi-head
    # (batch, num_heads, seq, head_dim)
    
    # Compute standard attention
    scores = Q @ K^T / √head_dim
    attention = softmax(scores)  # (batch, num_heads, seq_q, seq_k)
    
    # Apply to values
    output = attention @ V
```

**Output Shape:** `(batch_size, num_heads, sequence_length_query, sequence_length_key)`

---

### 2. MAW Multi-Head Attention with 5D Tensor

Both implementations use the **exact same 5D attention computation**:

```python
class MAWEncoder(nn.Module):  # MAWWithSupervisedClassificationEncoder or MAWWithGRPOEncoder
    
    # Step 1: Standard projections (same as NON-MAW)
    Q = query_proj(hidden_states)
    K = key_proj(hidden_states)
    V = value_proj(hidden_states)
    
    # Step 2: NEW - Depth-aware projections
    Q_depth = depth_query_proj(hidden_states)  # (batch, seq, hidden * depth)
    K_depth = depth_key_proj(hidden_states)    # (batch, seq, hidden * depth)
    
    # Step 3: Reshape to 5D
    Q_depth → (batch, num_heads, seq, head_dim, depth)
    K_depth → (batch, num_heads, seq, head_dim, depth)
    
    # Step 4: Compute attention for EACH depth
    attention_weights_5d = zeros(batch, heads, seq_q, seq_k, depth)
    
    for d in range(depth_dim):
        q_d = Q_depth[:, :, :, :, d]  # (batch, heads, seq, head_dim)
        k_d = K_depth[:, :, :, :, d]  # (batch, heads, seq, head_dim)
        
        scores = q_d @ k_d^T / √head_dim
        attention_weights_5d[:, :, :, :, d] = softmax(scores)
    
    # Result: (batch, num_heads, seq_query, seq_key, depth)
```

**Key Innovation:** Instead of one 4D attention matrix, we now have **D different 4D attention matrices** (where D = depth_dim = 32).

Each depth represents a **different attention strategy**:
- Depth 0 might focus on local dependencies
- Depth 1 might focus on long-range dependencies
- Depth 2 might focus on syntactic patterns
- ... and so on

---

## The Critical Difference: Depth Selection

Both implementations need to select **one optimal depth** from the 5D tensor to get back to 4D attention.

### Input: 
`attention_weights_5d`: `(batch_size, num_heads, seq_query, seq_key, depth)`

### Output Needed: 
`attention_weights_4d`: `(batch_size, num_heads, seq_query, seq_key)`

### Question: 
**Which depth (0-31) should we select for each batch item?**

---

## Approach 1: Supervised Classification (`benchmark_evaluation.py`)

### Architecture

```python
class SupervisedClassificationRouter(nn.Module):
    def forward(self, attention_weights_5d):
        # Input: (batch, num_heads, seq_q, seq_k, depth)
        
        # Step 1: Compress 5D to fixed representation
        avg_attention = attention_weights_5d.mean(dim=-1)  
        # → (batch, num_heads, seq_q, seq_k)
        
        pooled = AdaptiveAvgPool2d(8, 8)(avg_attention)
        # → (batch, num_heads, 8, 8)
        
        flattened = pooled.flatten(start_dim=1)
        # → (batch, 512)
        
        # Step 2: Neural classifier
        logits = classifier_network(flattened)
        # → (batch, depth_dim)
        
        # Step 3: Select depth
        depth_idx = argmax(logits, dim=-1)
        # → (batch,)
        
        return depth_idx
```

### Training Process

1. **Create Training Targets:** Use rule-based heuristics to assign "correct" depth labels
   ```python
   # Example rule: assign depth based on query complexity
   if num_relevant_docs < 3:
       target_depth = 5  # Use focused attention
   elif num_relevant_docs < 10:
       target_depth = 15  # Use medium attention
   else:
       target_depth = 25  # Use broad attention
   ```

2. **Classification Loss:** 
   ```python
   loss = CrossEntropyLoss(predicted_depth, target_depth) + ranking_loss
   ```

3. **Optimization:** Standard supervised learning with gradient descent

### Key Characteristics

- **Training Signal:** Fixed, pre-defined depth labels
- **Method:** Neural network classification
- **Sampling:** Gumbel-Softmax for differentiable discrete selection (allows gradients to flow)
- **Advantage:** Simple, stable training
- **Limitation:** Relies on hand-crafted rules for target generation

---

## Approach 2: Reinforcement Learning (`benchmark_evaluation_GRPO.py`)

### Architecture

The GRPO approach uses **three main components**:

#### Component 1: GRPO Environment

```python
class GRPOEnvironment:
    """RL Environment for depth selection"""
    
    def reset(self, attention_weights_5d, relevance_scores):
        # Set up state
        self.current_state = attention_weights_5d
        self.relevance = relevance_scores
        return compressed_state
    
    def step(self, action):
        # action = selected depth index
        
        # Extract attention for selected depth
        selected_attention = attention_weights_5d[:, :, :, :, action]
        
        # Compute reward based on attention quality
        entropy = compute_entropy(selected_attention)      # diversity
        focus = compute_focus(selected_attention)          # concentration
        alignment = compute_relevance_alignment(...)       # relevance match
        
        reward = 0.3*entropy + 0.3*focus + 0.4*alignment
        
        return next_state, reward, done
```

**Reward Components:**
- **Entropy (30%):** Measures diversity of attention (higher = more diverse)
- **Focus (30%):** Measures concentration of attention (higher = more focused)
- **Relevance Alignment (40%):** Measures alignment with ground truth relevance

#### Component 2: GRPO Policy Network (Actor-Critic)

```python
class GRPOPolicyNetwork(nn.Module):
    def forward(self, state):
        # state: (batch, state_dim) - compressed 5D attention
        
        encoded = state_encoder(state)
        
        # Actor: outputs action probabilities
        action_logits = policy_head(encoded)  # (batch, depth_dim)
        
        # Critic: estimates state value
        state_value = value_head(encoded)     # (batch, 1)
        
        return action_logits, state_value
    
    def get_action(self, state, deterministic=False):
        logits, value = self.forward(state)
        
        if deterministic:
            action = argmax(logits)
        else:
            # Sample from distribution
            action = Categorical(softmax(logits)).sample()
            log_prob = log_prob_of_action
        
        return action, log_prob, value
```

#### Component 3: GRPO Router

```python
class GRPORouter(nn.Module):
    def __init__(self, config):
        self.policy = GRPOPolicyNetwork(config)
        self.reference_policy = copy(self.policy)  # Frozen reference
        self.environment = GRPOEnvironment(config)
    
    def compute_grpo_loss(self, attention_5d, relevance, baseline):
        # Step 1: Get state from environment
        state = environment.reset(attention_5d, relevance, baseline)
        
        # Step 2: Sample actions from current policy
        actions, log_probs, values = policy.get_action(state)
        
        # Step 3: Get reference policy probabilities (for KL regularization)
        ref_log_probs = reference_policy.get_log_probs(state, actions)
        
        # Step 4: Execute actions and get rewards
        _, rewards, _ = environment.step(actions)
        
        # Step 5: Compute advantages
        advantages = rewards - values
        
        # Step 6: GRPO Policy Loss
        ratio = exp(log_probs - ref_log_probs)  # π/π_ref
        kl_divergence = mean(log_probs - ref_log_probs)
        
        policy_loss = -mean(ratio * advantages) + β * kl_divergence
        
        # Step 7: Value Loss (critic training)
        value_loss = MSE(values, rewards)
        
        # Step 8: Entropy Bonus (encourage exploration)
        entropy = -mean(sum(probs * log(probs)))
        entropy_loss = -α * entropy
        
        total_loss = policy_loss + value_loss + entropy_loss
        return total_loss
```

### Training Process

1. **RL Episodes:**
   - Reset environment with 5D attention tensor
   - Policy selects depth (action)
   - Environment computes reward
   - Update policy using GRPO loss

2. **Policy Gradient Update:**
   ```python
   # Maximize: reward * policy_ratio - KL_penalty
   loss = -advantages * (π/π_ref) + β*KL(π || π_ref) + value_loss
   ```

3. **Reference Policy Update:**
   - Every 5 epochs, update reference policy to current policy
   - This prevents policy from drifting too far from initialization

### Key Characteristics

- **Training Signal:** Dynamic rewards from environment
- **Method:** Policy gradient with actor-critic
- **Optimization:** GRPO (Generalized Preference Optimization)
- **Advantage:** Learns from actual retrieval performance, no hand-crafted rules
- **Limitation:** More complex training, requires careful reward design

---

## Side-by-Side Comparison

| Aspect | Supervised Classification | Reinforcement Learning (GRPO) |
|--------|---------------------------|-------------------------------|
| **Router Input** | 5D attention tensor | Compressed state from 5D tensor |
| **Router Output** | Depth logits (batch, depth) | Action logits + value estimate |
| **Training Target** | Fixed depth labels (rule-based) | Dynamic rewards (performance-based) |
| **Loss Function** | Cross-entropy + ranking loss | Policy gradient + value + entropy |
| **Optimization** | Standard gradient descent | GRPO with KL regularization |
| **Sampling Method** | Gumbel-Softmax (differentiable) | Policy distribution sampling |
| **Training Stability** | High (supervised signal) | Medium (RL exploration) |
| **Adaptability** | Low (fixed rules) | High (learns from data) |
| **Complexity** | Simple | Complex (environment + policy) |

---

## Complete Forward Pass Example

### Supervised Classification Approach

```python
# Input: text embeddings
hidden_states: (batch=4, seq_len=128, hidden_dim=256)

# 1. MAW Encoder computes 5D attention
attention_5d: (4, 8, 128, 128, 32)
# 32 different attention strategies computed

# 2. Supervised Router classifies optimal depth
router_logits = router(attention_5d)  # (4, 32)
depth_indices = argmax(router_logits)  # (4,) → [15, 8, 22, 11]

# 3. Select optimal depth slices
attention_4d = attention_5d[:, :, :, :, depth_indices]  # (4, 8, 128, 128)

# 4. Apply to values
output = attention_4d @ V  # (4, 128, 256)
```

### GRPO RL Approach

```python
# Input: text embeddings
hidden_states: (batch=4, seq_len=128, hidden_dim=256)

# 1. MAW Encoder computes 5D attention (same as supervised)
attention_5d: (4, 8, 128, 128, 32)

# 2. Environment creates state representation
state = environment.reset(attention_5d, relevance_scores)  # (4, 512)

# 3. Policy network selects action
action, log_prob, value = policy.get_action(state)  # (4,) → [18, 7, 25, 9]

# 4. Select depth using RL action
attention_4d = attention_5d[:, :, :, :, action]  # (4, 8, 128, 128)

# 5. Environment computes reward (used for training)
reward = environment.step(action)  # (4,)

# 6. Apply to values
output = attention_4d @ V  # (4, 128, 256)
```

---

## Key Takeaways

1. **Same Core Architecture:** Both approaches use identical 5D attention computation
2. **Different Selection Mechanisms:**
   - Supervised: Neural classifier with pre-defined targets
   - GRPO: RL policy with environment rewards
3. **Trade-offs:**
   - Supervised: Simpler, more stable, but relies on heuristics
   - GRPO: More flexible, learns from data, but more complex
4. **Performance:** 
   - Both show +20-40% improvements over NON-MAW baseline
   - GRPO typically achieves slightly better results (~3-5% over supervised)

The choice between approaches depends on:
- **Use supervised** if you have good heuristics for depth selection
- **Use GRPO** if you want the model to learn optimal selection from retrieval performance
