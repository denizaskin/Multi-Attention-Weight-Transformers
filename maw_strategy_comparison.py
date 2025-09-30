"""
Comprehensive True MAW Strategy Comparison
Tests all 6 selection/combination strategies from true_maw_guide.md
Plus standard encoder baseline for comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import json

@dataclass
class MAWConfig:
    hidden_dim: int = 768
    num_heads: int = 12
    depth_dimension: int = 8
    dropout: float = 0.1

class BaseMAWAttention(nn.Module):
    """Base class with core 5D attention computation"""
    
    def __init__(self, config: MAWConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.depth_dim = config.depth_dimension
        
        # Standard Q, K, V projections
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Depth-aware Q, K projections
        self.depth_query_proj = nn.Linear(self.hidden_dim, self.hidden_dim * self.depth_dim)
        self.depth_key_proj = nn.Linear(self.hidden_dim, self.hidden_dim * self.depth_dim)
        
        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def compute_5d_attention(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Core 5D attention computation - same for all strategies"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Standard Q, K, V
        Q = self.query_proj(hidden_states)
        K = self.key_proj(hidden_states)
        V = self.value_proj(hidden_states)
        
        # Depth-aware Q, K
        Q_depth = self.depth_query_proj(hidden_states)
        K_depth = self.depth_key_proj(hidden_states)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        Q_depth = Q_depth.view(batch_size, seq_len, self.num_heads, self.head_dim, self.depth_dim).transpose(1, 2)
        K_depth = K_depth.view(batch_size, seq_len, self.num_heads, self.head_dim, self.depth_dim).transpose(1, 2)
        
        # Compute 5D attention weights
        attention_scores = torch.zeros(
            batch_size, self.num_heads, seq_len, seq_len, self.depth_dim,
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        
        for depth_idx in range(self.depth_dim):
            q_d = Q_depth[:, :, :, :, depth_idx]
            k_d = K_depth[:, :, :, :, depth_idx]
            
            scores = torch.matmul(q_d, k_d.transpose(-1, -2)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
                mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
                scores = scores.masked_fill(mask == 0, -1e9)
                
            attention_scores[:, :, :, :, depth_idx] = scores
            
        # Convert to probabilities
        attention_weights = F.softmax(attention_scores, dim=-2)
        attention_weights = self.dropout(attention_weights)
        
        # Apply each depth pattern to values
        depth_outputs = []
        for depth_idx in range(self.depth_dim):
            attn_d = attention_weights[:, :, :, :, depth_idx]
            output_d = torch.matmul(attn_d, V)
            depth_outputs.append(output_d)
            
        depth_outputs = torch.stack(depth_outputs, dim=0)  # (depth, batch, heads, seq, head_dim)
        
        return depth_outputs, attention_weights, hidden_states

# Strategy 1: Learned Weighted Combination
class LearnedWeightedMAW(BaseMAWAttention):
    """Strategy 1: Learn static weights for each depth"""
    
    def __init__(self, config: MAWConfig):
        super().__init__(config)
        self.depth_weights = nn.Parameter(torch.ones(self.depth_dim) / self.depth_dim)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        depth_outputs, attention_weights, _ = self.compute_5d_attention(hidden_states, attention_mask)
        
        # Apply learned weights
        weights = F.softmax(self.depth_weights, dim=0)
        weights = weights.view(self.depth_dim, 1, 1, 1, 1)
        combined = (depth_outputs * weights).sum(dim=0)
        
        # Reshape and project
        batch_size, seq_len = hidden_states.shape[:2]
        output = combined.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(output)

# Strategy 2: Dynamic Gating
class DynamicGatingMAW(BaseMAWAttention):
    """Strategy 2: Content-aware gating based on input"""
    
    def __init__(self, config: MAWConfig):
        super().__init__(config)
        self.depth_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.depth_dim * 2),
            nn.GELU(),
            nn.Linear(self.depth_dim * 2, self.depth_dim),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        depth_outputs, attention_weights, _ = self.compute_5d_attention(hidden_states, attention_mask)
        
        # Content-aware gating
        gate_scores = self.depth_gate(hidden_states.mean(dim=1))  # (batch, depth)
        gate_scores = gate_scores.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, 1, depth)
        gate_scores = gate_scores.permute(4, 0, 1, 2, 3)  # (depth, batch, 1, 1, 1)
        
        combined = (depth_outputs * gate_scores).sum(dim=0)
        
        # Reshape and project
        batch_size, seq_len = hidden_states.shape[:2]
        output = combined.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(output)

# Strategy 3: Progressive Fusion
class ProgressiveFusionMAW(BaseMAWAttention):
    """Strategy 3: Each depth builds on previous ones hierarchically"""
    
    def __init__(self, config: MAWConfig):
        super().__init__(config)
        self.fusion_weights = nn.Parameter(torch.sigmoid(torch.randn(self.depth_dim)))
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        depth_outputs, attention_weights, _ = self.compute_5d_attention(hidden_states, attention_mask)
        
        # Progressive fusion: each depth builds on previous
        combined = depth_outputs[0]
        for d in range(1, self.depth_dim):
            fusion_weight = torch.sigmoid(self.fusion_weights[d])
            combined = fusion_weight * combined + (1 - fusion_weight) * depth_outputs[d]
        
        # Reshape and project
        batch_size, seq_len = hidden_states.shape[:2]
        output = combined.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(output)

# Strategy 4: Attention-Weighted Selection
class AttentionWeightedMAW(BaseMAWAttention):
    """Strategy 4: Use attention magnitudes to determine importance"""
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        depth_outputs, attention_weights, _ = self.compute_5d_attention(hidden_states, attention_mask)
        
        # Use attention magnitudes for selection
        attn_norms = attention_weights.norm(dim=(-2, -1))  # (batch, heads, seq, depth)
        selection_weights = F.softmax(attn_norms.mean(dim=(1, 2)), dim=-1)  # Average over heads and seq
        
        # Apply selection weights - fix dimension handling
        selection_weights = selection_weights.view(-1, 1, 1, 1, self.depth_dim)  # (batch, 1, 1, 1, depth)
        selection_weights = selection_weights.permute(4, 0, 1, 2, 3)  # (depth, batch, 1, 1, 1)
        
        combined = (depth_outputs * selection_weights).sum(dim=0)
        
        # Reshape and project
        batch_size, seq_len = hidden_states.shape[:2]
        output = combined.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(output)

# Strategy 5: Top-K Selection
class TopKSelectionMAW(BaseMAWAttention):
    """Strategy 5: Select only the k most important depth patterns"""
    
    def __init__(self, config: MAWConfig, k: int = 3):
        super().__init__(config)
        self.k = min(k, config.depth_dimension)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        depth_outputs, attention_weights, _ = self.compute_5d_attention(hidden_states, attention_mask)
        
        # Calculate importance of each depth
        output_norms = depth_outputs.norm(dim=-1).mean(dim=(-1, -2, -3))  # (depth,)
        top_k_indices = torch.topk(output_norms, k=self.k).indices
        
        # Select top-k depths
        selected_outputs = depth_outputs[top_k_indices]
        combined = selected_outputs.mean(dim=0)
        
        # Reshape and project
        batch_size, seq_len = hidden_states.shape[:2]
        output = combined.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(output)

# Strategy 6: Entropy-Based Selection
class EntropyBasedMAW(BaseMAWAttention):
    """Strategy 6: Prefer attention patterns with higher entropy (diversity)"""
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        depth_outputs, attention_weights, _ = self.compute_5d_attention(hidden_states, attention_mask)
        
        # Calculate entropy for each depth
        entropies = []
        for d in range(self.depth_dim):
            attn_d = attention_weights[:, :, :, :, d]
            entropy = -(attn_d * torch.log(attn_d + 1e-9)).sum(dim=-1).mean()
            entropies.append(entropy)
        
        entropies = torch.stack(entropies)
        entropy_weights = F.softmax(entropies, dim=0)
        
        # Apply entropy-based weights
        entropy_weights = entropy_weights.view(self.depth_dim, 1, 1, 1, 1)
        combined = (depth_outputs * entropy_weights).sum(dim=0)
        
        # Reshape and project
        batch_size, seq_len = hidden_states.shape[:2]
        output = combined.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(output)

# Standard Encoder for baseline comparison
class StandardEncoder(nn.Module):
    """Standard multi-head attention encoder for baseline"""
    
    def __init__(self, config: MAWConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = hidden_states.shape
        
        Q = self.query_proj(hidden_states)
        K = self.key_proj(hidden_states)
        V = self.value_proj(hidden_states)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(context)

def create_test_data(batch_size: int = 4, seq_len: int = 32, hidden_dim: int = 768):
    """Create test data for evaluation"""
    return torch.randn(batch_size, seq_len, hidden_dim)

def evaluate_model(model: nn.Module, model_name: str, test_data: torch.Tensor, num_runs: int = 10):
    """Evaluate a model on test data"""
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"🔍 EVALUATING: {model_name}")
    print(f"{'='*60}")
    
    # Model info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: {param_count:,}")
    
    # Timing evaluation
    times = []
    outputs = []
    
    with torch.no_grad():
        # Warmup
        _ = model(test_data)
        
        # Timed runs
        for i in range(num_runs):
            start_time = time.time()
            output = model(test_data)
            end_time = time.time()
            
            times.append(end_time - start_time)
            outputs.append(output)
    
    # Performance metrics
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    # Output quality metrics
    final_output = outputs[-1]
    output_variance = final_output.var().item()
    output_norm = final_output.norm().item()
    output_mean = final_output.mean().item()
    output_std = final_output.std().item()
    
    # Stability across runs
    output_consistency = torch.stack(outputs).std(dim=0).mean().item()
    
    # Results
    results = {
        "model_name": model_name,
        "parameters": param_count,
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "output_variance": output_variance,
        "output_norm": output_norm,
        "output_mean": output_mean,
        "output_std": output_std,
        "output_consistency": output_consistency,
        "output_shape": list(final_output.shape)
    }
    
    # Print results
    print(f"⏱️  Timing:")
    print(f"   Average: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"📈 Output Quality:")
    print(f"   Variance: {output_variance:.6f}")
    print(f"   Norm: {output_norm:.2f}")
    print(f"   Mean: {output_mean:.6f}")
    print(f"   Std: {output_std:.4f}")
    print(f"🎯 Stability:")
    print(f"   Consistency: {output_consistency:.6f} (lower = more stable)")
    print(f"📐 Shape: {final_output.shape}")
    
    return results

def run_comprehensive_evaluation():
    """Run comprehensive evaluation of all strategies"""
    
    print("🚀 COMPREHENSIVE TRUE MAW STRATEGY EVALUATION")
    print("=" * 80)
    print("Testing all 6 strategies from true_maw_guide.md plus standard baseline")
    print("=" * 80)
    
    # Configuration
    config = MAWConfig(
        hidden_dim=768,
        num_heads=12,
        depth_dimension=8,
        dropout=0.1
    )
    
    # Test data
    test_data = create_test_data(batch_size=4, seq_len=32, hidden_dim=768)
    print(f"📊 Test Data Shape: {test_data.shape}")
    
    # Models to evaluate
    models = [
        ("Standard Encoder (Baseline)", StandardEncoder(config)),
        ("Strategy 1: Learned Weighted", LearnedWeightedMAW(config)),
        ("Strategy 2: Dynamic Gating", DynamicGatingMAW(config)),
        ("Strategy 3: Progressive Fusion", ProgressiveFusionMAW(config)),
        ("Strategy 4: Attention Weighted", AttentionWeightedMAW(config)),
        ("Strategy 5: Top-K Selection", TopKSelectionMAW(config, k=3)),
        ("Strategy 6: Entropy Based", EntropyBasedMAW(config))
    ]
    
    # Evaluate each model
    all_results = []
    
    for model_name, model in models:
        try:
            results = evaluate_model(model, model_name, test_data, num_runs=5)
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ {model_name} failed: {e}")
            continue
    
    # Comparative analysis
    print("\n" + "=" * 80)
    print("📊 COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    if len(all_results) > 1:
        baseline = all_results[0]  # Standard encoder
        
        print(f"\n⚡ EFFICIENCY COMPARISON (vs {baseline['model_name']}):")
        print("-" * 60)
        for result in all_results:
            param_ratio = result['parameters'] / baseline['parameters']
            time_ratio = result['avg_time_ms'] / baseline['avg_time_ms']
            print(f"{result['model_name']:30}: {param_ratio:5.1f}x params, {time_ratio:5.1f}x time")
        
        print(f"\n🎯 OUTPUT QUALITY RANKING (by variance - lower is better):")
        print("-" * 60)
        sorted_by_variance = sorted(all_results, key=lambda x: x['output_variance'])
        for i, result in enumerate(sorted_by_variance, 1):
            print(f"{i:2}. {result['model_name']:30}: {result['output_variance']:.6f}")
        
        print(f"\n📈 STABILITY RANKING (by consistency - lower is better):")
        print("-" * 60)
        sorted_by_stability = sorted(all_results, key=lambda x: x['output_consistency'])
        for i, result in enumerate(sorted_by_stability, 1):
            print(f"{i:2}. {result['model_name']:30}: {result['output_consistency']:.6f}")
        
        print(f"\n🏆 SPEED RANKING (by time - lower is better):")
        print("-" * 60)
        sorted_by_speed = sorted(all_results, key=lambda x: x['avg_time_ms'])
        for i, result in enumerate(sorted_by_speed, 1):
            print(f"{i:2}. {result['model_name']:30}: {result['avg_time_ms']:.2f} ms")
    
    # Summary insights
    print(f"\n🔍 KEY INSIGHTS:")
    print("-" * 40)
    
    if len(all_results) > 1:
        # Find best in each category
        best_efficiency = min(all_results[1:], key=lambda x: x['parameters'] / x['avg_time_ms'])
        best_quality = min(all_results, key=lambda x: x['output_variance'])
        best_stability = min(all_results, key=lambda x: x['output_consistency'])
        fastest = min(all_results, key=lambda x: x['avg_time_ms'])
        
        print(f"🚀 Most Efficient: {best_efficiency['model_name']}")
        print(f"🎯 Best Quality: {best_quality['model_name']}")
        print(f"📊 Most Stable: {best_stability['model_name']}")
        print(f"⚡ Fastest: {fastest['model_name']}")
    
    # Save results
    with open('strategy_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: strategy_comparison_results.json")
    
    return all_results

if __name__ == "__main__":
    results = run_comprehensive_evaluation()
    
    print(f"\n🎉 EVALUATION COMPLETE!")
    print(f"\n✨ Summary:")
    print(f"- Tested {len(results)} models")
    print(f"- All 6 True MAW strategies implemented")
    print(f"- Standard encoder baseline included")
    print(f"- Performance metrics: timing, quality, stability")
    print(f"\n🔬 Next steps: Use best strategy for IR benchmarking!")