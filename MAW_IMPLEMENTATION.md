# Multi-Attention-Weight (MAW) Implementation

## Overview

The MAW (Multi-Attention-Weight) mechanism is a novel attention computation method that creates 5-dimensional attention weights and uses Group Relative Policy Optimization (GRPO) to select the optimal depth dimension.

## Architecture

### Key Components

1. **TokenLevelMAW Module** (`tier1_fixed.py`, line 629)
   - Replaces standard attention with 5D attention computation
   - Uses GRPO for trainable depth selection
   - Maintains compatibility with standard transformer architectures

### The 7-Step MAW Process

#### Step 1: Query Expansion and Transpose
```
Input:  query  (batch_size, num_heads, seq_len_q, head_dim)
Output: query_expanded (batch_size, num_heads, depth_dim, seq_len_q, 1)
```
The query vector is expanded across the depth dimension and reshaped to prepare for matrix multiplication.

#### Step 2: Key Expansion and Transpose
```
Input:  key (batch_size, num_heads, seq_len_k, head_dim)
Output: key_expanded (batch_size, num_heads, depth_dim, 1, seq_len_k)
```
The key vector is similarly expanded and transposed to align with the query.

#### Step 3: 5D Attention Tensor Computation
```
attn_5d = query_expanded @ key_expanded
Shape: (batch_size, num_heads, depth_dim, seq_len_q, seq_len_k)
```
Matrix multiplication creates the 5-dimensional attention tensor where the depth dimension represents different attention "perspectives".

#### Step 4: Dimension Transpose
```
attn_5d = attn_5d.permute(0, 1, 3, 4, 2)
Shape: (batch_size, num_heads, seq_len_q, seq_len_k, depth_dim)
```
The tensor is transposed to place the depth dimension last, preparing for depth selection.

#### Step 5: GRPO Depth Selection
```
depth_weights = GRPO(attn_5d, hidden_states)
Shape: (batch_size, depth_dim)
```
**Group Relative Policy Optimization (GRPO)** learns to select the best depth index:
- **Policy Network**: Neural network that outputs probability distribution over depth indices
- **Value Network**: Estimates expected reward for the current state
- **Reward Signal**: Negative variance of attention weights (encourages focused attention)
- **Training**: Policy gradient with baseline subtraction to reduce variance

The GRPO mechanism:
- Samples depth indices from the policy distribution during training
- Uses Gumbel-Softmax for differentiable sampling
- Updates policy based on advantage (reward - baseline)
- Maintains exponential moving average baseline for stable training

#### Step 6: Softmax Normalization
```
attn_4d = (attn_5d * depth_weights).sum(dim=-1)
Shape: (batch_size, num_heads, seq_len_q, seq_len_k)

attn_weights = softmax(attn_4d, dim=-1)
```
After depth selection reduces the 5D tensor to 4D, standard softmax is applied over the key dimension to create proper attention probabilities.

#### Step 7: Value Multiplication
```
output = attn_weights @ value
Shape: (batch_size, num_heads, seq_len_q, head_dim)
```
Standard attention mechanism: multiply attention weights with value vectors.

## Key Features

### 1. Multi-Perspective Attention
The depth dimension (`depth_dim=5` by default) allows the model to maintain multiple attention "perspectives" simultaneously, which are then intelligently combined via GRPO.

### 2. Learnable Attention Selection
Unlike fixed attention mechanisms, MAW learns which attention perspective is best for each input through reinforcement learning (GRPO).

### 3. Reward-Driven Optimization
The reward function (negative attention variance) encourages the model to learn focused, interpretable attention patterns.

### 4. Differentiable Training
Despite using sampling, the Gumbel-Softmax trick ensures end-to-end differentiability during training.

## Implementation Details

### GRPO Algorithm

```python
1. Compute policy_probs from pooled hidden states
2. Sample depth_index from Categorical(policy_probs)
3. Compute reward based on attention quality
4. Calculate advantage = reward - baseline
5. Policy loss = -log_prob(depth_index) * advantage
6. Value loss = MSE(value_estimate, reward)
7. Update baseline with exponential moving average
```

### Training vs Inference

**Training Mode:**
- Uses Gumbel-Softmax for soft depth selection
- Samples from policy distribution
- Computes GRPO loss for policy updates
- Updates baseline reward

**Inference Mode:**
- Uses greedy selection (argmax of policy)
- Creates one-hot depth weights
- No GRPO loss computation

## Configuration

```python
BenchmarkConfig(
    use_maw=True,           # Enable MAW
    maw_depth_dim=5,        # Number of depth dimensions
    maw_num_heads=12,       # Number of attention heads
    maw_layers=1,           # Number of MAW layers
)
```

## Benefits

1. **Adaptive Attention**: Learns to select attention patterns based on input
2. **Interpretability**: Depth selection decisions can be analyzed
3. **Improved Performance**: Multiple perspectives capture richer relationships
4. **Stable Training**: Baseline subtraction reduces policy gradient variance

## Mathematical Formulation

Given:
- Q, K, V ∈ ℝ^(B×H×L×D) (batch, heads, length, dimension)
- depth_dim = d

MAW computes:
```
Q' ∈ ℝ^(B×H×d×L×1)
K' ∈ ℝ^(B×H×d×1×L)
A_5D = Q' ⊗ K' ∈ ℝ^(B×H×d×L×L)
A_5D = permute(A_5D) ∈ ℝ^(B×H×L×L×d)

π(s) = PolicyNet(s)  # Policy over depth indices
w ~ π(s)             # Sample or argmax
A = (A_5D * w).sum(dim=-1) ∈ ℝ^(B×H×L×L)

A_norm = softmax(A, dim=-1)
Output = A_norm @ V
```

## Testing

Run the test suite:
```bash
python test_maw.py
```

Tests verify:
- Forward/backward pass correctness
- Dimension transformations
- GRPO gradient flow
- Training/inference mode switching

## Integration

The MAW module integrates seamlessly into the HFTextEncoder:

```python
if config.use_maw:
    self.maw_layers = nn.ModuleList([
        TokenLevelMAW(hidden_size, config.maw_depth_dim, config.maw_num_heads) 
        for _ in range(config.maw_layers)
    ])
```

Applied after the base encoder:
```python
hidden_states = self.model(**tokenized).last_hidden_state
if self.maw_layers is not None:
    for layer in self.maw_layers:
        hidden_states = layer(hidden_states, tokenized["attention_mask"])
```

## Future Enhancements

1. **Multiple Depth Sampling**: Sample multiple depth indices and ensemble
2. **Hierarchical Depth Selection**: Different depth for different heads
3. **Curriculum Learning**: Gradually increase depth dimension during training
4. **Attention Visualization**: Tools to visualize depth selection decisions
