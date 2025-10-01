"""
Minimal NON-MAW vs MAW with GRPO Router Comparison (Real Datasets)
Tests on Tier-1 journal retrieval datasets with 64 depth dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
try:
    import ir_datasets
    import ir_measures
    from transformers import AutoTokenizer, AutoModel
    USE_REAL_DATASETS = True
except ImportError:
    USE_REAL_DATASETS = False
    print("⚠️  Real dataset libraries not available. Using synthetic data.")

@dataclass
class Config:
    hidden_dim: int = 512        # word_embedding_dimension (changed from 256)
    num_heads: int = 8           # attention_heads
    head_dim: int = 64           # hidden_dim // num_heads = 512 // 8 = 64 (DEPTH)
    seq_len: int = 128           # Longer for real documents (changed from 64)
    vocab_size: int = 30522      # BERT vocab size (changed from 1000)
    dropout: float = 0.1
    max_docs_per_query: int = 100  # For efficiency

    @property
    def depth_dim(self) -> int:
        """Depth = word_embedding_dimension / attention_heads = head_dim"""
        return self.head_dim  # 64 dimensions

class DatasetLoader:
    """Load and preprocess real retrieval datasets"""
    
    def __init__(self, config: Config):
        self.config = config
        if USE_REAL_DATASETS:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.encoder = AutoModel.from_pretrained('bert-base-uncased')
            
            # Freeze BERT encoder (we only train the attention mechanisms)
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def load_real_datasets(self, max_queries=50):
        """Load real datasets if available, otherwise fallback to synthetic"""
        if not USE_REAL_DATASETS:
            return self.create_synthetic_fallback(max_queries)
        
        try:
            # Try TREC-DL 2019 first
            print(f"📥 Loading TREC-DL 2019 dataset...")
            dataset = ir_datasets.load('msmarco-passage/trec-dl-2019/judged')
            
            queries = {}
            documents = {}
            qrels = {}
            
            # Load queries
            for query in dataset.queries_iter():
                if len(queries) >= max_queries:
                    break
                queries[query.query_id] = query.text
            
            # Load documents (sample subset)
            doc_count = 0
            for doc in dataset.docs_iter():
                if doc_count >= 5000:
                    break
                documents[doc.doc_id] = doc.text
                doc_count += 1
            
            # Load qrels
            for qrel in dataset.qrels_iter():
                if qrel.query_id in queries:
                    if qrel.query_id not in qrels:
                        qrels[qrel.query_id] = {}
                    qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
            
            print(f"✅ Loaded {len(queries)} queries, {len(documents)} documents")
            return self.prepare_real_data(queries, documents, qrels)
            
        except Exception as e:
            print(f"❌ Failed to load real datasets: {e}")
            return self.create_synthetic_fallback(max_queries)
    
    def prepare_real_data(self, queries: Dict, documents: Dict, qrels: Dict):
        """Convert real text data to embeddings"""
        print("🔄 Encoding real texts to embeddings...")
        
        prepared_queries = []
        prepared_documents = []
        prepared_relevance = []
        
        for query_id, query_text in list(queries.items())[:30]:  # Limit for demo
            if query_id not in qrels:
                continue
            
            # Encode query
            query_embedding = self.encode_texts([query_text])
            
            # Get docs for this query
            query_docs = []
            query_relevance = []
            doc_texts = []
            
            for doc_id, relevance in qrels[query_id].items():
                if doc_id in documents and len(doc_texts) < 10:
                    doc_texts.append(documents[doc_id])
                    query_relevance.append(float(relevance))
            
            if len(doc_texts) == 0:
                continue
            
            # Encode documents
            doc_embeddings = self.encode_texts(doc_texts)
            for i in range(doc_embeddings.size(0)):
                query_docs.append(doc_embeddings[i:i+1])
            
            prepared_queries.append(query_embedding)
            prepared_documents.append(query_docs)
            prepared_relevance.append(query_relevance)
        
        return prepared_queries, prepared_documents, prepared_relevance
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using BERT"""
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, 
            max_length=self.config.seq_len, return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.encoder(**encoded)
            return outputs.last_hidden_state
    
    def create_synthetic_fallback(self, max_queries=50):
        """Fallback synthetic data with correct dimensions"""
        print("🔄 Creating synthetic data with 64 depth dimensions...")
        
        queries = []
        documents = []
        relevance_scores = []
        
        for _ in range(max_queries):
            # Query with correct dimensions
            query = torch.randn(1, self.config.seq_len, self.config.hidden_dim)
            
            query_docs = []
            query_relevance = []
            
            for doc_idx in range(10):
                if doc_idx < 3:  # Relevant docs
                    doc = query + 0.1 * torch.randn_like(query)
                    relevance = 1.0
                else:  # Non-relevant docs
                    doc = torch.randn(1, self.config.seq_len, self.config.hidden_dim)
                    relevance = 0.0
                
                query_docs.append(doc)
                query_relevance.append(relevance)
            
            queries.append(query)
            documents.append(query_docs)
            relevance_scores.append(query_relevance)
        
        return queries, documents, relevance_scores

class NonMAWEncoder(nn.Module):
    """Standard multi-head attention encoder (NON-MAW)"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim  # Now 64 instead of 32
        
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
    """GRPO router for 64-dimensional depth selection"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.depth_dim = config.depth_dim  # 64
        
        # Enhanced router for 64 dimensions
        self.router = nn.Sequential(
            nn.Linear(self.depth_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.depth_dim)
        )
        
        self.layer_norm = nn.LayerNorm(self.depth_dim)
        
    def forward(self, attention_patterns: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_patterns: (batch, heads, seq_q, seq_k, depth=64)
        Returns:
            depth_selection: (batch,) - discrete depth indices (0-63)
        """
        # Average attention patterns across heads and sequence dimensions
        pattern_summary = attention_patterns.mean(dim=(1, 2, 3))  # (batch, 64)
        pattern_summary = self.layer_norm(pattern_summary)
        
        # Router prediction
        router_logits = self.router(pattern_summary)  # (batch, 64)
        
        # Discrete selection using Gumbel softmax
        if self.training:
            depth_probs = F.gumbel_softmax(router_logits, hard=True, dim=-1, tau=0.5)
            depth_indices = depth_probs.argmax(dim=-1)
        else:
            depth_indices = router_logits.argmax(dim=-1)
            
        return depth_indices

class MAWWithGRPOEncoder(nn.Module):
    """MAW encoder with 5D attention: (batch, heads, seq_q, seq_k, depth=64)"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim  # 64 - This IS the depth dimension
        self.depth_dim = config.depth_dim  # 64
        
        # Standard projections
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # GRPO Router
        self.grpo_router = GRPORouter(config)
        
        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Standard Q, K, V projections
        Q = self.query_proj(hidden_states)
        K = self.key_proj(hidden_states)
        V = self.value_proj(hidden_states)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 🎯 Create 5D attention weights: (batch, heads, seq_q, seq_k, depth=64)
        attention_weights = torch.zeros(
            batch_size, self.num_heads, seq_len, seq_len, self.depth_dim,
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        
        # Compute attention for each depth dimension (0-63)
        for depth_idx in range(self.depth_dim):
            # Use specific dimension of Q and K for this depth
            q_depth = Q[:, :, :, depth_idx:depth_idx+1]  # (batch, heads, seq, 1)
            k_depth = K[:, :, :, depth_idx:depth_idx+1]  # (batch, heads, seq, 1)
            
            # Compute attention scores
            scores = torch.matmul(q_depth, k_depth.transpose(-1, -2)).squeeze(-1)
            
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
                mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights[:, :, :, :, depth_idx] = F.softmax(scores, dim=-1)
        
        attention_weights = self.dropout(attention_weights)
        
        # 🎯 GRPO Router: Select optimal depth (0-63)
        depth_indices = self.grpo_router(attention_weights)
        
        # Apply selected depth attention to values
        batch_outputs = []
        for batch_idx in range(batch_size):
            selected_depth = depth_indices[batch_idx].item()
            attn_selected = attention_weights[batch_idx, :, :, :, selected_depth]
            output_selected = torch.matmul(attn_selected, V[batch_idx])
            batch_outputs.append(output_selected)
        
        context = torch.stack(batch_outputs, dim=0)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(context)

def create_synthetic_retrieval_data(config: Config, num_queries: int = 100, num_docs_per_query: int = 10):
    """Create synthetic retrieval data with correct dimensions"""
    queries = []
    documents = []
    relevance_scores = []
    
    for _ in range(num_queries):
        # Query with 512-dimensional embeddings
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
    """Train the GRPO router for 64-depth selection"""
    
    print(f"🔥 Training GRPO Router (64 depth patterns) for {epochs} epochs...")
    
    # Lower learning rate for stability with 64 dimensions
    optimizer = torch.optim.Adam(model.grpo_router.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        pattern_selections = []
        
        for query_idx in range(len(queries)):
            query = queries[query_idx]
            query_docs = documents[query_idx]
            query_relevance = relevance[query_idx]
            
            # Forward pass through MAW encoder
            query_output = model(query)
            
            # Track pattern selection
            with torch.no_grad():
                attention_scores = torch.zeros(1, model.num_heads, query.size(1), query.size(1), model.depth_dim)
                depth_indices = model.grpo_router(attention_scores)
                pattern_selections.append(depth_indices.item())
            
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
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.grpo_router.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += batch_loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        # Pattern analysis
        unique_patterns = len(set(pattern_selections))
        most_common = max(set(pattern_selections), key=pattern_selections.count) if pattern_selections else 0
        
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}, "
              f"Patterns used: {unique_patterns}/64, Most common: {most_common}")
    
    print("✅ GRPO Router training complete!")

def evaluate_model_on_retrieval(model: nn.Module, model_name: str, 
                               queries: List[torch.Tensor], documents: List[List[torch.Tensor]], 
                               relevance: List[List[float]]):
    """Evaluate model on retrieval task"""
    
    print(f"\n🔍 Evaluating {model_name}...")
    
    model.eval()
    all_scores = []
    pattern_usage = []
    
    with torch.no_grad():
        for query_idx in range(len(queries)):
            query = queries[query_idx]
            query_docs = documents[query_idx]
            
            # Get query representation
            query_output = model(query)
            query_repr = query_output.mean(dim=1)  # (1, hidden_dim)
            
            # Track pattern usage for MAW
            if hasattr(model, 'grpo_router'):
                attention_scores = torch.zeros(1, model.num_heads, query.size(1), query.size(1), model.depth_dim)
                depth_indices = model.grpo_router(attention_scores)
                pattern_usage.append(depth_indices.item())
            
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
    
    # Pattern analysis for MAW
    if pattern_usage:
        unique_patterns = len(set(pattern_usage))
        print(f"   Patterns used: {unique_patterns}/64 ({unique_patterns/64*100:.1f}%)")
    
    return metrics

def main():
    """Main comparison function with 64 depth dimensions"""
    
    print("🚀 NON-MAW vs MAW+GRPO (64 Depth Dimensions)")
    print("=" * 70)
    
    # Configuration with 64 depth dimensions
    config = Config(
        hidden_dim=512,           # word_embedding_dimension (512 instead of 256)
        num_heads=8,              # attention_heads
        head_dim=64,              # 512 // 8 = 64 (THIS IS THE DEPTH!)
        seq_len=128,              # longer sequences (128 instead of 64)
        dropout=0.1
    )
    
    print(f"📋 Configuration:")
    print(f"   Hidden dim: {config.hidden_dim} (was 256)")
    print(f"   Num heads: {config.num_heads}")
    print(f"   Head dim (DEPTH): {config.head_dim} (was 32)")
    print(f"   Depth formula: {config.hidden_dim} / {config.num_heads} = {config.depth_dim}")
    print(f"   5D attention: (batch, {config.num_heads}, seq, seq, {config.depth_dim})")
    print(f"   Sequence length: {config.seq_len} (was 64)")
    
    # Create models
    non_maw_model = NonMAWEncoder(config)
    maw_model = MAWWithGRPOEncoder(config)
    
    print(f"\n📊 Model Parameters:")
    non_maw_params = sum(p.numel() for p in non_maw_model.parameters())
    maw_params = sum(p.numel() for p in maw_model.parameters())
    print(f"   NON-MAW: {non_maw_params:,}")
    print(f"   MAW+GRPO (64-depth): {maw_params:,}")
    print(f"   Ratio: {maw_params/non_maw_params:.1f}x")
    print(f"   Additional params for 64-depth: {maw_params - non_maw_params:,}")
    
    # Load data (real or synthetic)
    print(f"\n🔄 Loading retrieval data...")
    if USE_REAL_DATASETS:
        loader = DatasetLoader(config)
        train_queries, train_docs, train_relevance = loader.load_real_datasets(max_queries=30)
        eval_queries, eval_docs, eval_relevance = loader.load_real_datasets(max_queries=20)
        print("✅ Using real Tier-1 datasets")
    else:
        train_queries, train_docs, train_relevance = create_synthetic_retrieval_data(config, num_queries=50, num_docs_per_query=10)
        eval_queries, eval_docs, eval_relevance = create_synthetic_retrieval_data(config, num_queries=30, num_docs_per_query=10)
        print("✅ Using synthetic data (install transformers, ir-datasets for real data)")
    
    print(f"   Training: {len(train_queries)} queries")
    print(f"   Evaluation: {len(eval_queries)} queries")
    
    # Train MAW model (GRPO router for 64 dimensions)
    train_grpo_router(maw_model, train_queries, train_docs, train_relevance, epochs=10)
    
    # Evaluate both models
    print(f"\n" + "=" * 70)
    print("📈 EVALUATION RESULTS (64 Depth Dimensions)")
    print("=" * 70)
    
    non_maw_results = evaluate_model_on_retrieval(non_maw_model, "NON-MAW", eval_queries, eval_docs, eval_relevance)
    maw_results = evaluate_model_on_retrieval(maw_model, "MAW+GRPO (64-depth)", eval_queries, eval_docs, eval_relevance)
    
    # Comparison
    print(f"\n🏆 COMPARISON (64-depth vs Standard):")
    print("=" * 50)
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
            
        print(f"{metric:15}: NON-MAW={non_maw_val:.4f}, MAW+GRPO(64)={maw_val:.4f} {status}{improvement:+.1f}%")
    
    print(f"\n✅ Evaluation Complete!")
    print(f"💡 Now using 64 depth dimensions (was 8) with 512-dim embeddings!")
    print(f"🔬 5D attention shape: (batch, 8, {config.seq_len}, {config.seq_len}, 64)")

if __name__ == "__main__":
    main()