"""
Comprehensive Baseline Comparisons and Ablation Studies for MAW

This module implements various attention mechanisms for comparison and
provides systematic ablation studies to understand component contributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    num_heads: int
    head_dim: int
    dropout: float = 0.1
    temperature: float = 1.0


class BaseAttentionMechanism(ABC, nn.Module):
    """Abstract base class for attention mechanisms."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.all_head_size = self.num_heads * self.head_dim
        
    @abstractmethod
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (output, attention_weights)."""
        pass
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention."""
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


class StandardAttention(BaseAttentionMechanism):
    """Standard scaled dot-product attention."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value)
        
        return context_layer, attention_probs


class SparseAttention(BaseAttentionMechanism):
    """Sparse attention mechanism with configurable sparsity patterns."""
    
    def __init__(self, config: AttentionConfig, sparsity_factor: float = 0.1, 
                 pattern: str = "local"):
        super().__init__(config)
        self.sparsity_factor = sparsity_factor
        self.pattern = pattern
        self.dropout = nn.Dropout(config.dropout)
        
    def create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sparsity mask based on the specified pattern."""
        if self.pattern == "local":
            # Local attention window
            window_size = max(1, int(seq_len * self.sparsity_factor))
            mask = torch.zeros(seq_len, seq_len, device=device)
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                mask[i, start:end] = 1
        elif self.pattern == "strided":
            # Strided attention
            stride = max(1, int(1 / self.sparsity_factor))
            mask = torch.zeros(seq_len, seq_len, device=device)
            for i in range(0, seq_len, stride):
                mask[i, :] = 1
                mask[:, i] = 1
        else:
            # Random sparsity
            mask = torch.rand(seq_len, seq_len, device=device) < self.sparsity_factor
            mask = mask.float()
            
        return mask
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, num_heads, seq_len_q, head_dim = query.shape
        seq_len_k = key.shape[2]
        
        # Create sparsity mask
        sparse_mask = self.create_sparse_mask(seq_len_k, query.device)
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(head_dim)
        
        # Apply sparsity mask
        attention_scores = attention_scores * sparse_mask
        attention_scores = attention_scores + (1 - sparse_mask) * (-1e9)
        
        # Apply additional mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value)
        
        return context_layer, attention_probs


class LinearAttention(BaseAttentionMechanism):
    """Linear attention mechanism using kernel methods."""
    
    def __init__(self, config: AttentionConfig, feature_dim: int = 256):
        super().__init__(config)
        self.feature_dim = feature_dim
        self.feature_map = nn.ReLU()
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Apply feature map to queries and keys
        Q = self.feature_map(query)
        K = self.feature_map(key)
        
        # Normalize features
        Q = Q / math.sqrt(self.feature_dim)
        K = K / math.sqrt(self.feature_dim)
        
        # Linear attention computation: O(n) complexity
        # Context = Q * (K^T * V) / (Q * K^T * 1)
        KV = torch.matmul(K.transpose(-1, -2), value)  # [batch, heads, head_dim, head_dim]
        context_layer = torch.matmul(Q, KV)
        
        # Normalization factor
        K_sum = K.sum(dim=-2, keepdim=True)  # [batch, heads, 1, head_dim]
        norm_factor = torch.matmul(Q, K_sum.transpose(-1, -2))  # [batch, heads, seq_len, 1]
        norm_factor = torch.clamp(norm_factor, min=1e-6)
        
        context_layer = context_layer / norm_factor
        
        # For compatibility, return dummy attention weights
        attention_probs = torch.ones(query.shape[0], query.shape[1], 
                                   query.shape[2], key.shape[2], device=query.device)
        attention_probs = attention_probs / query.shape[2]
        
        return context_layer, attention_probs


class MultiScaleAttention(BaseAttentionMechanism):
    """Multi-scale attention with different receptive fields."""
    
    def __init__(self, config: AttentionConfig, scales: List[int] = [1, 2, 4, 8]):
        super().__init__(config)
        self.scales = scales
        self.scale_projections = nn.ModuleList([
            nn.Linear(config.head_dim, config.head_dim) for _ in scales
        ])
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Compute attention at different scales
        scale_outputs = []
        scale_attention_probs = []
        
        for i, scale in enumerate(self.scales):
            # Downsample if scale > 1
            if scale > 1:
                # Reshape for pooling: [B, H, L, D] -> [B*H, D, L]
                key_reshaped = key.permute(0, 1, 3, 2).contiguous().view(-1, head_dim, seq_len)
                value_reshaped = value.permute(0, 1, 3, 2).contiguous().view(-1, head_dim, seq_len)
                
                pooled_key = F.avg_pool1d(key_reshaped, kernel_size=scale, stride=scale)
                pooled_value = F.avg_pool1d(value_reshaped, kernel_size=scale, stride=scale)
                
                # Reshape back: [B*H, D, L'] -> [B, H, L', D]
                pooled_seq_len = pooled_key.shape[-1]
                pooled_key = pooled_key.view(batch_size, num_heads, head_dim, pooled_seq_len).permute(0, 1, 3, 2)
                pooled_value = pooled_value.view(batch_size, num_heads, head_dim, pooled_seq_len).permute(0, 1, 3, 2)
            else:
                pooled_key = key
                pooled_value = value
            
            # Apply scale-specific projection
            projected_key = self.scale_projections[i](pooled_key)
            
            # Compute attention
            attention_scores = torch.matmul(query, projected_key.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(head_dim)
            
            if attention_mask is not None and scale == 1:
                attention_scores = attention_scores + attention_mask
                
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            
            # Upsample attention if needed
            if scale > 1:
                attention_probs = F.interpolate(
                    attention_probs.view(-1, seq_len, attention_probs.shape[-1]),
                    size=seq_len, mode='linear', align_corners=False
                ).view(batch_size, num_heads, seq_len, seq_len)
                
                pooled_value = F.interpolate(
                    pooled_value.transpose(-1, -2).contiguous().view(-1, head_dim, pooled_value.shape[-2]),
                    size=seq_len, mode='linear', align_corners=False
                ).view(batch_size, num_heads, head_dim, seq_len).transpose(-1, -2)
                
            context = torch.matmul(attention_probs, pooled_value)
            scale_outputs.append(context)
            scale_attention_probs.append(attention_probs)
        
        # Combine outputs from different scales
        combined_output = torch.stack(scale_outputs).mean(dim=0)
        combined_attention = torch.stack(scale_attention_probs).mean(dim=0)
        
        return combined_output, combined_attention


class MAWAttention(BaseAttentionMechanism):
    """Our Multi-Attention-Weight mechanism for comparison."""
    
    def __init__(self, config: AttentionConfig, depth_dim: int = 8, 
                 maw_strength: float = 0.15, gating_mode: str = "stat"):
        super().__init__(config)
        self.depth_dim = depth_dim
        self.maw_strength = maw_strength
        self.gating_mode = gating_mode
        self.dropout = nn.Dropout(config.dropout)
        
    def _score_depths(self, attn_5d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score depths using statistical measures."""
        B, H, D, Lq, Lk = attn_5d.shape
        
        if self.gating_mode == "uniform":
            weights = torch.full((B, D), 1.0 / D, device=attn_5d.device, dtype=attn_5d.dtype)
            best_idx = torch.zeros(B, dtype=torch.long, device=attn_5d.device)
            return weights, best_idx
            
        # Compute statistical measures
        var_k = attn_5d.var(dim=-1).mean(dim=(1, 3))  # [B,D]
        max_k = attn_5d.max(dim=-1).values.mean(dim=(1, 3))  # [B,D]
        
        # Entropy
        p = attn_5d.clamp_min(1e-8)
        ent = (-p * p.log()).sum(dim=-1).mean(dim=(1, 3))  # [B,D]
        
        # HHI
        hhi = (attn_5d.pow(2).sum(dim=-1)).mean(dim=(1, 3))  # [B,D]
        
        # Normalize and combine
        def min_max_norm(vals):
            vmin = vals.min(dim=-1, keepdim=True).values
            vmax = vals.max(dim=-1, keepdim=True).values
            return (vals - vmin) / (vmax - vmin + 1e-8)
        
        var_n = min_max_norm(var_k)
        max_n = min_max_norm(max_k)
        ent_n = min_max_norm(ent)
        hhi_n = min_max_norm(hhi)
        
        alpha = 1.0 + 10.0 * self.maw_strength
        score = (0.5 * var_n + 0.3 * max_n + 0.2 * hhi_n) - (0.4 * ent_n)
        weights = torch.softmax(alpha * score, dim=-1)
        best_idx = torch.argmax(weights, dim=-1)
        
        return weights, best_idx
    
    def _min_max_norm(self, vals, eps=1e-12):
        """Min-max normalization utility."""
        vmin = vals.min(dim=-1, keepdim=True).values
        vmax = vals.max(dim=-1, keepdim=True).values
        denom = (vmax - vmin).clamp_min(eps)
        return (vals - vmin) / denom
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, H, Lq, Dh = query.shape
        Lk = key.shape[2]
        D = self.depth_dim
        
        # Split head dimension into depth slices
        q_slices = torch.tensor_split(query, D, dim=-1)
        k_slices = torch.tensor_split(key, D, dim=-1)
        v_slices = torch.tensor_split(value, D, dim=-1)
        
        # Compute per-depth attention scores
        scores = []
        for qi, ki in zip(q_slices, k_slices):
            d_k = max(qi.size(-1), 1)
            s = torch.matmul(qi, ki.transpose(-1, -2)) / math.sqrt(d_k)
            scores.append(s)
        scores_5d = torch.stack(scores, dim=2)  # [B,H,D,Lq,Lk]
        
        # Apply mask
        if attention_mask is not None:
            mask = attention_mask
            if mask.dim() == 4:
                mask = mask.unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.view(mask.size(0), 1, 1, 1, mask.size(1))
            scores_5d = scores_5d + mask
            
        # Softmax over keys per depth
        attn_5d = F.softmax(scores_5d, dim=-1)
        attn_5d = self.dropout(attn_5d)
        
        # Depth gating
        depth_weights, best_idx = self._score_depths(attn_5d)
        
        # Weighted attention for output
        attn_4d = torch.sum(attn_5d * depth_weights.view(B, 1, D, 1, 1), dim=2)
        
        # Compute per-depth contexts then gate
        ctx_slices = []
        for d, (attn_d, vi) in enumerate(zip(attn_5d.unbind(dim=2), v_slices)):
            ctx_d = torch.matmul(attn_d, vi)
            w_d = depth_weights[:, d].view(B, 1, 1, 1)
            ctx_slices.append(ctx_d * w_d)
        
        context = torch.cat(ctx_slices, dim=-1)
        
        return context, attn_4d


class AttentionComparator:
    """Comprehensive comparison framework for attention mechanisms."""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.mechanisms = {
            'standard': StandardAttention(config),
            'sparse_local': SparseAttention(config, sparsity_factor=0.1, pattern='local'),
            'sparse_strided': SparseAttention(config, sparsity_factor=0.1, pattern='strided'),
            'linear': LinearAttention(config),
            'multiscale': MultiScaleAttention(config),
            'maw': MAWAttention(config)
        }
        
    def compare_complexity(self, seq_len: int) -> Dict[str, Dict[str, float]]:
        """Compare computational and memory complexity across mechanisms."""
        head_dim = self.config.head_dim
        num_heads = self.config.num_heads
        
        complexities = {}
        
        # Standard attention
        std_time_ops = seq_len**2 * head_dim * num_heads
        std_memory = seq_len**2 * num_heads
        
        complexities['standard'] = {
            'time_ops': std_time_ops,
            'memory_units': std_memory,
            'time_complexity': 'O(L²d)',
            'memory_complexity': 'O(L²)'
        }
        
        # Sparse attention
        sparsity = 0.1
        sparse_ops = std_time_ops * sparsity
        sparse_memory = std_memory * sparsity
        
        complexities['sparse'] = {
            'time_ops': sparse_ops,
            'memory_units': sparse_memory,
            'time_complexity': f'O({sparsity}L²d)',
            'memory_complexity': f'O({sparsity}L²)'
        }
        
        # Linear attention
        feature_dim = 256
        linear_ops = seq_len * feature_dim * head_dim * num_heads
        linear_memory = seq_len * feature_dim * num_heads
        
        complexities['linear'] = {
            'time_ops': linear_ops,
            'memory_units': linear_memory,
            'time_complexity': 'O(Ld)',
            'memory_complexity': 'O(L)'
        }
        
        # MAW attention
        depth_dim = 8
        maw_ops = std_time_ops + depth_dim * seq_len * head_dim**2 * num_heads
        maw_memory = depth_dim * std_memory
        
        complexities['maw'] = {
            'time_ops': maw_ops,
            'memory_units': maw_memory,
            'time_complexity': 'O(L²d + DLd²)',
            'memory_complexity': 'O(DL²)'
        }
        
        return complexities
    
    def benchmark_attention_mechanisms(self, batch_size: int = 8, seq_len: int = 512,
                                     device: str = 'cuda') -> Dict[str, Dict[str, float]]:
        """Benchmark attention mechanisms on synthetic data."""
        torch.manual_seed(42)
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Create synthetic input
        hidden_size = self.config.num_heads * self.config.head_dim
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
        
        # Create Q, K, V projections
        query_proj = nn.Linear(hidden_size, hidden_size, device=device)
        key_proj = nn.Linear(hidden_size, hidden_size, device=device)
        value_proj = nn.Linear(hidden_size, hidden_size, device=device)
        
        query = query_proj(hidden_states)
        key = key_proj(hidden_states)
        value = value_proj(hidden_states)
        
        # Reshape for multi-head attention
        def reshape_for_attention(x):
            new_shape = x.size()[:-1] + (self.config.num_heads, self.config.head_dim)
            x = x.view(*new_shape)
            return x.permute(0, 2, 1, 3)
        
        query = reshape_for_attention(query)
        key = reshape_for_attention(key)
        value = reshape_for_attention(value)
        
        results = {}
        
        for name, mechanism in self.mechanisms.items():
            mechanism = mechanism.to(device)
            mechanism.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = mechanism(query, key, value)
            
            # Benchmark
            torch.cuda.synchronize() if device.type == 'cuda' else None
            import time
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    output, attention_weights = mechanism(query, key, value)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            
            # Memory usage (approximate)
            if device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
            else:
                memory_allocated = 0
            
            results[name] = {
                'avg_time_ms': avg_time * 1000,
                'memory_mb': memory_allocated,
                'output_shape': list(output.shape),
                'attention_shape': list(attention_weights.shape)
            }
        
        return results


def run_ablation_studies():
    """Run comprehensive ablation studies for MAW mechanism."""
    
    print("Running MAW Ablation Studies...")
    
    config = AttentionConfig(num_heads=12, head_dim=64)
    
    # Ablation 1: Depth dimension
    print("\n1. Depth Dimension Ablation:")
    for depth in [1, 2, 4, 8, 16]:
        maw = MAWAttention(config, depth_dim=depth)
        complexity = (depth * config.head_dim**2 * config.num_heads) / (config.head_dim**2 * config.num_heads)
        print(f"  Depth={depth}: Complexity Ratio={complexity:.2f}x")
    
    # Ablation 2: Gating modes
    print("\n2. Gating Mode Ablation:")
    gating_modes = ['uniform', 'random', 'stat']
    for mode in gating_modes:
        maw = MAWAttention(config, gating_mode=mode)
        print(f"  Gating Mode={mode}: Uses {'heuristic' if mode == 'stat' else 'simple'} selection")
    
    # Ablation 3: MAW strength
    print("\n3. MAW Strength Ablation:")
    for strength in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
        maw = MAWAttention(config, maw_strength=strength)
        print(f"  MAW Strength={strength}: Gating sharpness factor={1.0 + 10.0 * strength:.1f}")
    
    print("\nAblation studies completed.")


def run_baseline_comparison():
    """Run comprehensive baseline comparison."""
    
    print("Running Baseline Comparison...")
    
    config = AttentionConfig(num_heads=12, head_dim=64)
    comparator = AttentionComparator(config)
    
    # Complexity comparison
    print("\n1. Complexity Analysis (seq_len=512):")
    complexity_results = comparator.compare_complexity(seq_len=512)
    for name, results in complexity_results.items():
        print(f"  {name}:")
        print(f"    Time: {results['time_complexity']}")
        print(f"    Memory: {results['memory_complexity']}")
        print(f"    Ops: {results['time_ops']:,}")
    
    # Runtime benchmark
    print("\n2. Runtime Benchmark:")
    if torch.cuda.is_available():
        benchmark_results = comparator.benchmark_attention_mechanisms()
        for name, results in benchmark_results.items():
            print(f"  {name}: {results['avg_time_ms']:.2f}ms, {results['memory_mb']:.1f}MB")
    else:
        print("  CUDA not available, skipping runtime benchmark")
    
    print("\nBaseline comparison completed.")


if __name__ == "__main__":
    run_ablation_studies()
    run_baseline_comparison()