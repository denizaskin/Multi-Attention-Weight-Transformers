# MAW Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Multi-Attention-Weight (MAW)                          │
│                                                                              │
│  Input: hidden_states (B, L, H)                                             │
│         attention_mask (B, L)                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │  Q Proj   │   │  K Proj   │   │  V Proj   │
            │ (H → H)   │   │ (H → H)   │   │ (H → H)   │
            └───────────┘   └───────────┘   └───────────┘
                    │               │               │
                    │               │               │
                    ▼               ▼               │
            ┌───────────────────────────┐          │
            │  Reshape to Multi-Head    │          │
            │  (B, L, H) → (B, h, L, d) │          │
            │  h = num_heads            │          │
            │  d = head_dim             │          │
            └───────────────────────────┘          │
                    │               │               │
                    ▼               ▼               │
┌───────────────────────────────────────────────────────────────────────┐     │
│                  STEP 1 & 2: Expand to 5D                              │     │
│                                                                         │     │
│  Query:  (B, h, L, d) → (B, h, D, L, 1)                               │     │
│  Key:    (B, h, L, d) → (B, h, D, 1, L)                               │     │
│                                                                         │     │
│  Where D = depth_dim (typically 5)                                     │     │
└───────────────────────────────────────────────────────────────────────┘     │
                    │               │                                          │
                    └───────┬───────┘                                          │
                            │                                                  │
                            ▼                                                  │
        ┌───────────────────────────────────────────────┐                     │
        │  STEP 3: Matrix Multiply Q' ⊗ K'              │                     │
        │                                                │                     │
        │  A_5D = Q' @ K'                                │                     │
        │  Shape: (B, h, D, L, L)                        │                     │
        └───────────────────────────────────────────────┘                     │
                            │                                                  │
                            ▼                                                  │
        ┌───────────────────────────────────────────────┐                     │
        │  STEP 4: Transpose                             │                     │
        │                                                │                     │
        │  A_5D = permute(0, 1, 3, 4, 2)                 │                     │
        │  Shape: (B, h, L, L, D)                        │                     │
        └───────────────────────────────────────────────┘                     │
                            │                                                  │
        ┌───────────────────┴───────────────────────────────────┐             │
        │  STEP 5: GRPO (Group Relative Policy Optimization)    │             │
        │                                                        │             │
        │  ┌──────────────────────────────────────────────┐    │             │
        │  │  Policy Network                               │    │             │
        │  │  pooled_hidden → [hidden*2] → depth_dim       │    │             │
        │  │  Output: probs over depth indices             │    │             │
        │  └──────────────────────────────────────────────┘    │             │
        │                       │                               │             │
        │                       ▼                               │             │
        │  ┌──────────────────────────────────────────────┐    │             │
        │  │  Sampling (Training)                          │    │             │
        │  │  depth_idx ~ Categorical(policy_probs)        │    │             │
        │  │  OR                                            │    │             │
        │  │  Greedy (Inference)                           │    │             │
        │  │  depth_idx = argmax(policy_probs)             │    │             │
        │  └──────────────────────────────────────────────┘    │             │
        │                       │                               │             │
        │                       ▼                               │             │
        │  ┌──────────────────────────────────────────────┐    │             │
        │  │  Compute Reward                               │    │             │
        │  │  reward = -variance(attention_weights)        │    │             │
        │  │  (Encourages focused attention)               │    │             │
        │  └──────────────────────────────────────────────┘    │             │
        │                       │                               │             │
        │                       ▼                               │             │
        │  ┌──────────────────────────────────────────────┐    │             │
        │  │  Value Network                                │    │             │
        │  │  pooled_hidden → [hidden] → 1                 │    │             │
        │  │  Output: expected reward                      │    │             │
        │  └──────────────────────────────────────────────┘    │             │
        │                       │                               │             │
        │                       ▼                               │             │
        │  ┌──────────────────────────────────────────────┐    │             │
        │  │  Policy Gradient Update                       │    │             │
        │  │  advantage = reward - baseline                │    │             │
        │  │  policy_loss = -log_prob * advantage          │    │             │
        │  │  value_loss = MSE(value_est, reward)          │    │             │
        │  └──────────────────────────────────────────────┘    │             │
        │                       │                               │             │
        │                       ▼                               │             │
        │  Output: depth_weights (B, D)                         │             │
        │  - Training: Gumbel-Softmax (differentiable)          │             │
        │  - Inference: One-hot from argmax                     │             │
        └────────────────────────────────────────────────────────┘             │
                            │                                                  │
                            ▼                                                  │
        ┌───────────────────────────────────────────────┐                     │
        │  Apply Depth Weights & Reduce 5D → 4D         │                     │
        │                                                │                     │
        │  depth_weights: (B, D) → (B, 1, 1, 1, D)      │                     │
        │  A_4D = (A_5D * depth_weights).sum(dim=-1)    │                     │
        │  Shape: (B, h, L, L)                           │                     │
        └───────────────────────────────────────────────┘                     │
                            │                                                  │
                            ▼                                                  │
        ┌───────────────────────────────────────────────┐                     │
        │  STEP 6: Softmax Normalization                 │                     │
        │                                                │                     │
        │  A_norm = softmax(A_4D, dim=-1)                │                     │
        │  Shape: (B, h, L, L)                           │                     │
        │  (Apply attention_mask before softmax)         │                     │
        └───────────────────────────────────────────────┘                     │
                            │                                                  │
                            │               ┌──────────────────────────────────┘
                            │               │
                            ▼               ▼
        ┌───────────────────────────────────────────────┐
        │  STEP 7: Multiply with Value                   │
        │                                                │
        │  Output = A_norm @ V                           │
        │  Shape: (B, h, L, d)                           │
        └───────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────────┐
        │  Reshape Back                                  │
        │  (B, h, L, d) → (B, L, H)                      │
        └───────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────────┐
        │  Output Projection + Residual                  │
        │                                                │
        │  out = output_proj(reshaped)                   │
        │  out = LayerNorm(hidden_states + out)          │
        └───────────────────────────────────────────────┘
                            │
                            ▼
                  Output: (B, L, H)


Legend:
  B = batch_size
  L = sequence_length
  H = hidden_size
  h = num_heads
  d = head_dim = H / h
  D = depth_dim (typically 5)


Key Components:
  ┌─────────┐  
  │ Network │  Neural network layer
  └─────────┘  

  ┌─────────┐  
  │ Process │  Computation step
  └─────────┘  

         │     
         ▼      Data flow direction


GRPO Training Flow:
  1. Policy network outputs probability distribution over depths
  2. Sample depth index (or argmax for inference)
  3. Compute reward based on attention quality
  4. Update policy using policy gradient
  5. Update value network to predict rewards
  6. Use Gumbel-Softmax for differentiable depth selection
```

## Simplified Data Flow

```
Input (B, L, H)
      │
      ├─→ Q Proj → (B, h, L, d) ──┐
      ├─→ K Proj → (B, h, L, d) ──┼─→ 5D Expansion
      └─→ V Proj → (B, h, L, d) ──┘    │
                                        ▼
                              (B, h, D, L, L)  ← 5D Attention
                                        │
                                        ▼
                            GRPO selects depth index
                                        │
                                        ▼
                              (B, h, L, L)  ← 4D Attention
                                        │
                                        ▼
                                   Softmax
                                        │
                                        ▼
                              Output (B, L, H)
```

## GRPO Learning Cycle

```
┌─────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                    │
│                                                          │
│  1. Current State: pooled hidden states                 │
│              │                                           │
│              ▼                                           │
│  2. Policy Network: predict depth distribution          │
│              │                                           │
│              ▼                                           │
│  3. Sample: depth_idx ~ Categorical(policy)             │
│              │                                           │
│              ▼                                           │
│  4. Action: use sampled depth for attention             │
│              │                                           │
│              ▼                                           │
│  5. Reward: -variance(attention_weights)                │
│              │                                           │
│              ▼                                           │
│  6. Advantage: reward - baseline                        │
│              │                                           │
│              ▼                                           │
│  7. Update Policy: -log_prob * advantage                │
│              │                                           │
│              ▼                                           │
│  8. Update Value Network: MSE(value, reward)            │
│              │                                           │
│              └────→ Improved depth selection!           │
└─────────────────────────────────────────────────────────┘
```
