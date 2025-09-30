"""
Minimal NON-MAW vs MAW with GRPO Router Comparison
Only the essential code for comparing the two approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

@dataclass
class Config:
    hidden_dim: int = 256
    num_heads: int = 8
    depth_dim: int = 8
    seq_len: int = 64
    vocab_size: int = 1000
    dropout: float = 0.1

class NonMAWEncoder(nn.Module):
    """Standard multi-head attention encoder (NON-MAW)"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        # Standard projections
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Standard attention
        Q = self.query_proj(hidden_states)
        K = self.key_proj(hidden_states)
        V = self.value_proj(hidden_states)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(context)

class GRPORouter(nn.Module):
    """Tiny trainable GRPO router for discrete depth selection"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.depth_dim = config.depth_dim
        
        # Tiny router network
        self.router = nn.Sequential(
            nn.Linear(config.depth_dim, 32),
            nn.ReLU(),
            nn.Linear(32, config.depth_dim)
        )
        
    def forward(self, attention_patterns: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_patterns: (batch, heads, seq_q, seq_k, depth)
        Returns:
            depth_selection: (batch,) - discrete depth indices
        """
        # Average attention patterns across heads and sequence dimensions
        pattern_summary = attention_patterns.mean(dim=(1, 2, 3))  # (batch, depth)
        
        # Router prediction
        router_logits = self.router(pattern_summary)  # (batch, depth)
        
        # Discrete selection using Gumbel softmax for differentiability during training
        if self.training:
            depth_probs = F.gumbel_softmax(router_logits, hard=True, dim=-1)
            depth_indices = depth_probs.argmax(dim=-1)
        else:
            depth_indices = router_logits.argmax(dim=-1)
            
        return depth_indices

class MAWWithGRPOEncoder(nn.Module):
    """MAW encoder with 5D attention and GRPO router"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.depth_dim = config.depth_dim
        
        # Standard projections
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Depth-aware projections
        self.depth_query_proj = nn.Linear(self.hidden_dim, self.hidden_dim * self.depth_dim)
        self.depth_key_proj = nn.Linear(self.hidden_dim, self.hidden_dim * self.depth_dim)
        
        # GRPO Router
        self.grpo_router = GRPORouter(config)
        
        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
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
        
        # GRPO Router: Select discrete depth
        depth_indices = self.grpo_router(attention_weights)  # (batch,)
        
        # Apply selected depth attention to values
        batch_outputs = []
        for batch_idx in range(batch_size):
            selected_depth = depth_indices[batch_idx].item()
            attn_selected = attention_weights[batch_idx, :, :, :, selected_depth]  # (heads, seq, seq)
            output_selected = torch.matmul(attn_selected, V[batch_idx])  # (heads, seq, head_dim)
            batch_outputs.append(output_selected)
        
        # Stack batch outputs
        context = torch.stack(batch_outputs, dim=0)  # (batch, heads, seq, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(context)

def create_synthetic_retrieval_data(config: Config, num_queries: int = 100, num_docs_per_query: int = 10):
    """Create simple synthetic retrieval data"""
    queries = []
    documents = []
    relevance_scores = []
    
    for _ in range(num_queries):
        # Query
        query = torch.randn(1, config.seq_len, config.hidden_dim)
        
        # Documents (some relevant, some not)
        query_docs = []
        query_relevance = []
        
        for doc_idx in range(num_docs_per_query):
            if doc_idx < 3:  # First 3 docs are relevant
                # Similar to query (add noise)
                doc = query + 0.1 * torch.randn_like(query)
                relevance = 1.0
            else:
                # Random document
                doc = torch.randn(1, config.seq_len, config.hidden_dim)
                relevance = 0.0
                
            query_docs.append(doc)
            query_relevance.append(relevance)
        
        queries.append(query)
        documents.append(query_docs)
        relevance_scores.append(query_relevance)
    
    return queries, documents, relevance_scores

def compute_retrieval_metrics(scores: List[List[float]], relevance: List[List[float]], k: int = 10):
    """Compute standard retrieval metrics"""
    
    def dcg_at_k(relevance_scores, k):
        """Discounted Cumulative Gain at k"""
        dcg = 0.0
        for i in range(min(k, len(relevance_scores))):
            dcg += relevance_scores[i] / math.log2(i + 2)
        return dcg
    
    def ndcg_at_k(pred_relevance, true_relevance, k):
        """Normalized DCG at k"""
        dcg = dcg_at_k(pred_relevance[:k], k)
        idcg = dcg_at_k(sorted(true_relevance, reverse=True)[:k], k)
        return dcg / idcg if idcg > 0 else 0.0
    
    total_precision = 0.0
    total_recall = 0.0
    total_ndcg = 0.0
    total_map = 0.0
    
    num_queries = len(scores)
    
    for query_idx in range(num_queries):
        query_scores = scores[query_idx]
        query_relevance = relevance[query_idx]
        
        # Sort by scores (descending)
        sorted_indices = sorted(range(len(query_scores)), key=lambda i: query_scores[i], reverse=True)
        sorted_relevance = [query_relevance[i] for i in sorted_indices]
        
        # Precision@k
        relevant_at_k = sum(sorted_relevance[:k])
        precision_k = relevant_at_k / min(k, len(sorted_relevance))
        
        # Recall@k
        total_relevant = sum(query_relevance)
        recall_k = relevant_at_k / total_relevant if total_relevant > 0 else 0.0
        
        # NDCG@k
        ndcg_k = ndcg_at_k(sorted_relevance, query_relevance, k)
        
        # MAP (simplified)
        ap = 0.0
        relevant_count = 0
        for i, rel in enumerate(sorted_relevance):
            if rel > 0:
                relevant_count += 1
                ap += relevant_count / (i + 1)
        ap = ap / total_relevant if total_relevant > 0 else 0.0
        
        total_precision += precision_k
        total_recall += recall_k
        total_ndcg += ndcg_k
        total_map += ap
    
    return {
        'precision_at_k': total_precision / num_queries,
        'recall_at_k': total_recall / num_queries,
        'ndcg_at_k': total_ndcg / num_queries,
        'map': total_map / num_queries
    }

def train_grpo_router(model: MAWWithGRPOEncoder, queries: List[torch.Tensor], 
                     documents: List[List[torch.Tensor]], relevance: List[List[float]], 
                     epochs: int = 10):
    """Train the GRPO router"""
    
    print(f"🔥 Training GRPO Router for {epochs} epochs...")
    
    optimizer = torch.optim.Adam(model.grpo_router.parameters(), lr=0.001)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for query_idx in range(len(queries)):
            query = queries[query_idx]
            query_docs = documents[query_idx]
            query_relevance = relevance[query_idx]
            
            # Forward pass through MAW encoder
            query_output = model(query)
            
            # Compute losses for each document
            batch_loss = 0.0
            for doc_idx, (doc, rel_score) in enumerate(zip(query_docs, query_relevance)):
                doc_output = model(doc)
                
                # Similarity score
                similarity = F.cosine_similarity(query_output.mean(dim=1), doc_output.mean(dim=1), dim=-1)
                
                # Loss: encourage high similarity for relevant docs
                if rel_score > 0:
                    loss = 1.0 - similarity.mean()  # Want similarity close to 1
                else:
                    loss = F.relu(similarity.mean() - 0.5)  # Want similarity < 0.5
                
                batch_loss += loss
            
            batch_loss = batch_loss / len(query_docs)
            
            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
    
    print("✅ GRPO Router training complete!")

def evaluate_model_on_retrieval(model: nn.Module, model_name: str, 
                               queries: List[torch.Tensor], documents: List[List[torch.Tensor]], 
                               relevance: List[List[float]]):
    """Evaluate model on retrieval task"""
    
    print(f"\n🔍 Evaluating {model_name}...")
    
    model.eval()
    all_scores = []
    
    with torch.no_grad():
        for query_idx in range(len(queries)):
            query = queries[query_idx]
            query_docs = documents[query_idx]
            
            # Get query representation
            query_output = model(query)
            query_repr = query_output.mean(dim=1)  # (1, hidden_dim)
            
            # Score each document
            doc_scores = []
            for doc in query_docs:
                doc_output = model(doc)
                doc_repr = doc_output.mean(dim=1)  # (1, hidden_dim)
                
                # Cosine similarity
                score = F.cosine_similarity(query_repr, doc_repr, dim=-1).item()
                doc_scores.append(score)
            
            all_scores.append(doc_scores)
    
    # Compute metrics
    metrics = compute_retrieval_metrics(all_scores, relevance, k=10)
    
    print(f"📊 Results for {model_name}:")
    print(f"   Precision@10: {metrics['precision_at_k']:.4f}")
    print(f"   Recall@10: {metrics['recall_at_k']:.4f}")
    print(f"   F1@10: {2 * metrics['precision_at_k'] * metrics['recall_at_k'] / (metrics['precision_at_k'] + metrics['recall_at_k'] + 1e-9):.4f}")
    print(f"   NDCG@10: {metrics['ndcg_at_k']:.4f}")
    print(f"   MAP: {metrics['map']:.4f}")
    
    return metrics

def main():
    """Main comparison function"""
    
    print("🚀 NON-MAW vs MAW with GRPO Router Comparison")
    print("=" * 60)
    
    # Configuration
    config = Config(
        hidden_dim=256,
        num_heads=8,
        depth_dim=8,
        seq_len=64,
        dropout=0.1
    )
    
    print(f"📋 Configuration:")
    print(f"   Hidden dim: {config.hidden_dim}")
    print(f"   Num heads: {config.num_heads}")
    print(f"   Depth dim: {config.depth_dim}")
    print(f"   Sequence length: {config.seq_len}")
    
    # Create models
    non_maw_model = NonMAWEncoder(config)
    maw_model = MAWWithGRPOEncoder(config)
    
    print(f"\n📊 Model Parameters:")
    non_maw_params = sum(p.numel() for p in non_maw_model.parameters())
    maw_params = sum(p.numel() for p in maw_model.parameters())
    print(f"   NON-MAW: {non_maw_params:,}")
    print(f"   MAW+GRPO: {maw_params:,}")
    print(f"   Ratio: {maw_params/non_maw_params:.1f}x")
    
    # Create data
    print(f"\n🔄 Creating synthetic retrieval data...")
    train_queries, train_docs, train_relevance = create_synthetic_retrieval_data(config, num_queries=50, num_docs_per_query=10)
    eval_queries, eval_docs, eval_relevance = create_synthetic_retrieval_data(config, num_queries=30, num_docs_per_query=10)
    
    print(f"   Training: {len(train_queries)} queries")
    print(f"   Evaluation: {len(eval_queries)} queries")
    
    # Train MAW model (GRPO router)
    train_grpo_router(maw_model, train_queries, train_docs, train_relevance, epochs=10)
    
    # Evaluate both models
    print(f"\n" + "=" * 60)
    print("📈 EVALUATION RESULTS")
    print("=" * 60)
    
    non_maw_results = evaluate_model_on_retrieval(non_maw_model, "NON-MAW", eval_queries, eval_docs, eval_relevance)
    maw_results = evaluate_model_on_retrieval(maw_model, "MAW+GRPO", eval_queries, eval_docs, eval_relevance)
    
    # Comparison
    print(f"\n🏆 COMPARISON:")
    print("=" * 40)
    for metric in ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'map']:
        non_maw_val = non_maw_results[metric]
        maw_val = maw_results[metric]
        improvement = ((maw_val - non_maw_val) / non_maw_val * 100) if non_maw_val > 0 else 0
        
        if improvement > 0:
            status = "🟢 +"
        elif improvement < 0:
            status = "🔴 "
        else:
            status = "⚪ "
            
        print(f"{metric:15}: NON-MAW={non_maw_val:.4f}, MAW+GRPO={maw_val:.4f} {status}{improvement:+.1f}%")
    
    print(f"\n✅ Evaluation Complete!")

if __name__ == "__main__":
    main()

with open('/workspace/MAW-Clean/.gitignore', 'w') as f:
    f.write("""# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Data and result files
*.json
*.log
*results*
*data*
*dataset*
*output*
*summary*

# Model files
*.pt
*.pth
*.ckpt
*.h5
*.pkl

# Environment
.venv/
env/
venv/

# IDE
.vscode/
*.vsix

# Temporary files
*.tmp
*~
*.bak

# Large files
*.zip
*.tar.gz
""")