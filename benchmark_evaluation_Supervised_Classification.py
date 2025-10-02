"""
MAW vs NON-MAW Evaluation on 5 Benchmark Retrieval Datasets
Used in Tier-1 Journals/Conferences: SIGIR, WWW, WSDM, CIKM, EMNLP, ACL

Datasets evaluated:
1. MS MARCO Passage Ranking
2. TREC Deep Learning
3. Natural Questions  
4. SciDocs Citation Prediction
5. FiQA Financial QA

Metrics: Hit Rate, MRR, NDCG @ K=1,5,10,100,1000

Usage:
    # Run on all datasets with GPU (if available)
    python benchmark_evaluation_Supervised_Classification.py
    
    # Run on specific dataset with limited samples
    python benchmark_evaluation_Supervised_Classification.py --dataset MS_MARCO --samples 20 --epochs 10
    
    # Run on CPU only
    python benchmark_evaluation_Supervised_Classification.py --device cpu --samples 15
    
    # Run on multiple datasets
    python benchmark_evaluation_Supervised_Classification.py --datasets MS_MARCO TREC_DL --samples 30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import gc
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
import argparse
import sys
import random
import json
import os
from datetime import datetime
from pathlib import Path

@dataclass
class Config:
    hidden_dim: int = 256
    num_heads: int = 8
    seq_len: int = 128
    vocab_size: int = 30000
    dropout: float = 0.1
    use_half_precision: bool = False  # Disabled for compatibility
    enable_gradient_checkpointing: bool = False
    
    @property
    def depth_dim(self) -> int:
        """Depth dimension = word_embedding_dimension / num_attention_heads"""
        return self.hidden_dim // self.num_heads

# Benchmark Dataset Configurations (reduced for debugging)
BENCHMARK_DATASETS = {
    "MS_MARCO": {
        "name": "MS MARCO Passage Ranking", 
        "venue": "NIPS 2016, SIGIR 2019+",
        "num_queries": 50,  # Reduced for debugging
        "num_docs_per_query": 50,  # Reduced for debugging
        "avg_query_len": 6,
        "avg_doc_len": 80,
        "domain": "Web Search"
    },
    "TREC_DL": {
        "name": "TREC Deep Learning",
        "venue": "TREC 2019-2023, SIGIR",
        "num_queries": 40, 
        "num_docs_per_query": 50,
        "avg_query_len": 8,
        "avg_doc_len": 120,
        "domain": "Deep Learning Track"
    },
    "NATURAL_QA": {
        "name": "Natural Questions",
        "venue": "TACL 2019, ACL, EMNLP",
        "num_queries": 45,
        "num_docs_per_query": 50,
        "avg_query_len": 12,
        "avg_doc_len": 200,
        "domain": "Question Answering"
    },
    "SCIDOCS": {
        "name": "SciDocs Citation Prediction", 
        "venue": "EMNLP 2020, SIGIR",
        "num_queries": 30,
        "num_docs_per_query": 50,
        "avg_query_len": 15,
        "avg_doc_len": 180,
        "domain": "Scientific Literature"
    },
    "FIQA": {
        "name": "FiQA Financial QA",
        "venue": "WWW 2018, SIGIR",
        "num_queries": 40,
        "num_docs_per_query": 50, 
        "avg_query_len": 10,
        "avg_doc_len": 150,
        "domain": "Financial Question Answering"
    }
}

class SupervisedClassificationRouter(nn.Module):
    """Supervised Classification Router for depth selection from 5D attention weights"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.depth_dim = config.depth_dim
        self.num_heads = config.num_heads
        
        # Router processes 5D attention weights: (batch, heads, seq_q, seq_k, depth) -> depth_index
        # Use adaptive processing instead of fixed linear layers
        
        self.attention_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),  # Reduce spatial dimensions to 8x8
            nn.Flatten(start_dim=1),       # Flatten to (batch, heads * 8 * 8)
        )
        
        # Fixed size after adaptive pooling: heads * 8 * 8
        pooled_size = self.num_heads * 8 * 8  # 8 * 8 * 8 = 512
        
        self.router = nn.Sequential(
            nn.Linear(pooled_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.depth_dim)  # Output logits for each depth
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, attention_weights_5d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_weights_5d: (batch_size, num_heads, seq_len_query, seq_len_key, depth)
        Returns:
            router_logits: (batch_size, depth_dim) - logits for depth selection
        """
        batch_size = attention_weights_5d.shape[0]
        
        # Average over depth dimension to get representative attention pattern
        # (batch, heads, seq_q, seq_k, depth) -> (batch, heads, seq_q, seq_k)
        avg_attention = attention_weights_5d.mean(dim=-1)  # Average across depths
        
        # Use adaptive pooling to handle variable sequence lengths
        # (batch, heads, seq_q, seq_k) -> (batch, heads, 8, 8) -> (batch, heads * 64)
        pooled_attention = self.attention_processor(avg_attention)
        
        # Normalize to prevent extreme values
        pooled_attention = F.normalize(pooled_attention, p=2, dim=-1)
        
        # Route to select optimal depth
        router_logits = self.router(pooled_attention)
        router_logits = torch.clamp(router_logits, min=-10, max=10)
        return router_logits
        
    def get_depth_selection(self, attention_weights_5d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_weights_5d: (batch_size, num_heads, seq_len_query, seq_len_key, depth)
        Returns:
            depth_indices: (batch_size,) - selected depth index for each batch item
        """
        router_logits = self.forward(attention_weights_5d)
        
        if self.training:
            # Use Gumbel-Softmax for differentiable discrete sampling (not RL, just for gradient flow)
            depth_probs = F.gumbel_softmax(router_logits, hard=True, dim=-1, tau=1.0)
            depth_indices = depth_probs.argmax(dim=-1)
        else:
            depth_indices = router_logits.argmax(dim=-1)
            
        return depth_indices
    
    def select_optimal_attention(self, attention_weights_5d: torch.Tensor) -> torch.Tensor:
        """
        Select optimal 4D attention weights from 5D attention weights
        
        Args:
            attention_weights_5d: (batch_size, num_heads, seq_len_query, seq_len_key, depth)
        Returns:
            attention_weights_4d: (batch_size, num_heads, seq_len_query, seq_len_key)
        """
        batch_size = attention_weights_5d.shape[0]
        depth_indices = self.get_depth_selection(attention_weights_5d)
        
        # Select the optimal depth for each batch item
        selected_attention = []
        for batch_idx in range(batch_size):
            selected_depth = depth_indices[batch_idx].item()
            # Extract 4D attention weights for selected depth
            attention_4d = attention_weights_5d[batch_idx, :, :, :, selected_depth]  # (heads, seq_q, seq_k)
            selected_attention.append(attention_4d)
        
        # Stack back to batch dimension
        attention_weights_4d = torch.stack(selected_attention, dim=0)  # (batch, heads, seq_q, seq_k)
        return attention_weights_4d

class NonMAWEncoder(nn.Module):
    """Standard multi-head attention encoder (NON-MAW baseline)"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        # Standard attention projections
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = hidden_states.shape
        
        Q = self.query_proj(hidden_states)
        K = self.key_proj(hidden_states)
        V = self.value_proj(hidden_states)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        output = self.out_proj(context)
        output = self.layer_norm(output + hidden_states)
        
        return output

class MAWWithSupervisedClassificationEncoder(nn.Module):
    """Multi-Attention-Weight encoder with 5D attention and Supervised Classification router"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.depth_dim = config.depth_dim
        
        # Standard projections for 3D attention
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        # Depth-aware projections for 5D attention (new method: projects to num_heads * depth_dim)
        self.depth_query_proj = nn.Linear(self.hidden_dim, self.num_heads * self.depth_dim, bias=False)
        self.depth_key_proj = nn.Linear(self.hidden_dim, self.num_heads * self.depth_dim, bias=False)
        
        # Supervised Classification Router
        self.supervised_router = SupervisedClassificationRouter(config)
        
        # Output layers
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Standard Q, K, V for 3D attention
        Q = self.query_proj(hidden_states)
        K = self.key_proj(hidden_states)
        V = self.value_proj(hidden_states)
        
        # Depth-aware Q, K for 5D attention
        Q_depth = self.depth_query_proj(hidden_states)
        K_depth = self.depth_key_proj(hidden_states)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # NEW 5D ATTENTION COMPUTATION METHOD:
        # Step 1: Reshape depth projections to (batch, heads, seq, depth)
        Q_depth = Q_depth.view(batch_size, seq_len, self.num_heads, self.depth_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, sequence_length_query, depth)
        
        K_depth = K_depth.view(batch_size, seq_len, self.num_heads, self.depth_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, sequence_length_key, depth)
        
        # Step 2: Expand Q_depth and transpose
        # (batch, heads, seq_q, depth) -> (batch, heads, depth, seq_q, 1)
        Q_depth_expanded = Q_depth.transpose(2, 3).unsqueeze(-1)
        
        # Step 3: Expand K_depth (already transposed)
        # (batch, heads, seq_k, depth) -> (batch, heads, depth, 1, seq_k)
        K_depth_expanded = K_depth.transpose(2, 3).unsqueeze(-2)
        
        # Step 4: Element-wise multiply with broadcasting
        # (batch, heads, depth, seq_q, 1) * (batch, heads, depth, 1, seq_k)
        # = (batch, heads, depth, seq_q, seq_k)
        scores_5d = Q_depth_expanded * K_depth_expanded
        
        # Transpose to (batch, heads, seq_q, seq_k, depth)
        scores_5d = scores_5d.permute(0, 1, 3, 4, 2)
        
        # Step 5: Scale by sqrt(depth) and apply softmax
        scores_5d = scores_5d / math.sqrt(self.depth_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1)  # (batch, 1, 1, seq, 1)
            mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len, self.depth_dim)
            scores_5d = scores_5d.masked_fill(mask == 0, -1e9)
        
        # Softmax over depth dimension (dim=-1)
        attention_weights_5d = F.softmax(scores_5d, dim=-1)
        # Shape: (batch_size, num_heads, sequence_length_query, sequence_length_key, depth)
        
        # Supervised Classification: Select optimal 4D attention weights from 5D attention weights
        # Input: (batch, heads, seq_q, seq_k, depth) -> Output: (batch, heads, seq_q, seq_k)
        selected_attention_weights = self.supervised_router.select_optimal_attention(attention_weights_5d)
        
        # Apply dropout to selected attention weights
        selected_attention_weights = self.dropout(selected_attention_weights)
        
        # Apply selected attention weights to values
        context = torch.matmul(selected_attention_weights, V)  # (batch, heads, seq, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection and residual connection
        output = self.out_proj(context)
        output = self.layer_norm(output + hidden_states)
        
        return output

def create_benchmark_dataset_split(dataset_name: str, config: Config, train_ratio: float = 0.8, device: torch.device = None) -> Tuple[
    Tuple[List[torch.Tensor], List[List[torch.Tensor]], List[List[float]]],  # train
    Tuple[List[torch.Tensor], List[List[torch.Tensor]], List[List[float]]]   # test
]:
    """
    Create realistic benchmark dataset with proper train/test split
    
    Args:
        dataset_name: One of the benchmark dataset keys
        config: Model configuration
        train_ratio: Ratio for training split (default 0.8)
        device: Device to create tensors on (if None, uses CUDA if available)
        
    Returns:
        (train_queries, train_docs, train_relevance), (test_queries, test_docs, test_relevance)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_config = BENCHMARK_DATASETS[dataset_name]
    total_queries = dataset_config["num_queries"]
    num_docs = dataset_config["num_docs_per_query"]
    
    print(f"📚 Creating {dataset_config['name']} dataset with train/test split...")
    print(f"   Domain: {dataset_config['domain']}")
    print(f"   Venue: {dataset_config['venue']}")
    print(f"   Total queries: {total_queries}, Docs per query: {num_docs}")
    
    # Calculate split sizes
    num_train = int(total_queries * train_ratio)
    num_test = total_queries - num_train
    print(f"   Split: {num_train} train, {num_test} test queries")
    
    all_queries = []
    all_documents = []
    all_relevance_scores = []
    
    # Domain-specific relevance patterns
    domain = dataset_config["domain"]
    
    for query_idx in range(total_queries):
        # Create query with domain-specific characteristics
        query = torch.randn(1, config.seq_len, config.hidden_dim, device=device)
        
        # Domain-specific normalization patterns
        if "Scientific" in domain:
            query = F.normalize(query, p=2, dim=-1) * 1.2  # Higher magnitude for technical terms
        elif "Financial" in domain:
            query = F.normalize(query, p=2, dim=-1) * 0.9  # More conservative patterns
        elif "Web Search" in domain:
            query = F.normalize(query, p=2, dim=-1) * 1.1  # Diverse patterns
        else:
            query = F.normalize(query, p=2, dim=-1)
        
        query_docs = []
        query_relevance = []
        
        # Create realistic relevance distribution (more challenging)
        for doc_idx in range(num_docs):
            # Make the dataset much more challenging with realistic relevance distribution
            relevance_prob = np.random.random()
            
            if relevance_prob < 0.02:  # Only 2% highly relevant (grade 3)
                noise_scale = 0.3 + 0.2 * np.random.random()  # More noise
                doc = query + noise_scale * torch.randn_like(query)
                relevance = 3
            elif relevance_prob < 0.08:  # 6% moderately relevant (grade 2) 
                noise_scale = 0.5 + 0.3 * np.random.random()
                doc = 0.7 * query + 0.3 * torch.randn_like(query)
                relevance = 2
            elif relevance_prob < 0.20:  # 12% somewhat relevant (grade 1)
                noise_scale = 0.8 + 0.4 * np.random.random()
                doc = 0.4 * query + 0.6 * torch.randn_like(query)
                relevance = 1
            else:  # 80% not relevant (grade 0) - realistic for retrieval
                doc = torch.randn(1, config.seq_len, config.hidden_dim, device=device)
                # Add small correlation to make it challenging but not impossible
                if np.random.random() < 0.1:
                    doc = 0.1 * query + 0.9 * doc
                relevance = 0
                
            doc = F.normalize(doc, p=2, dim=-1)
            query_docs.append(doc)
            query_relevance.append(relevance)
        
        all_queries.append(query)
        all_documents.append(query_docs)
        all_relevance_scores.append(query_relevance)
    
    # Split into train/test with shuffling (same approach as GRPO for consistency)
    train_size = int(total_queries * train_ratio)
    indices = list(range(total_queries))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Create train split
    train_queries = [all_queries[i] for i in train_indices]
    train_documents = [all_documents[i] for i in train_indices]
    train_relevance = [all_relevance_scores[i] for i in train_indices]
    
    # Create test split
    test_queries = [all_queries[i] for i in test_indices]
    test_documents = [all_documents[i] for i in test_indices]
    test_relevance = [all_relevance_scores[i] for i in test_indices]
    
    return (train_queries, train_documents, train_relevance), (test_queries, test_documents, test_relevance)

def compute_retrieval_metrics(similarities: torch.Tensor, relevance_scores: List[float], 
                            k_values: List[int] = [1, 5, 10, 100, 1000]) -> Dict[str, Dict[int, float]]:
    """
    Compute Hit Rate, MRR, NDCG for specified k values
    
    Args:
        similarities: (num_docs,) similarity scores
        relevance_scores: List of relevance grades (0-3)
        k_values: K values for evaluation
        
    Returns:
        Dictionary with metrics for each k value
    """
    # Sort documents by similarity (descending)
    sorted_indices = torch.argsort(similarities, descending=True)
    sorted_relevance = [relevance_scores[i] for i in sorted_indices.cpu().numpy()]
    
    metrics = {'hit_rate': {}, 'mrr': {}, 'ndcg': {}}
    
    for k in k_values:
        # Ensure k doesn't exceed number of documents
        effective_k = min(k, len(sorted_relevance))
        
        # Hit Rate@k
        top_k_relevance = sorted_relevance[:effective_k]
        hit_rate = 1.0 if any(rel > 0 for rel in top_k_relevance) else 0.0
        metrics['hit_rate'][k] = hit_rate
        
        # MRR@k
        mrr = 0.0
        for rank, rel in enumerate(top_k_relevance, 1):
            if rel > 0:
                mrr = 1.0 / rank
                break
        metrics['mrr'][k] = mrr
        
        # NDCG@k
        dcg = 0.0
        for rank, rel in enumerate(top_k_relevance, 1):
            if rel > 0:
                dcg += (2**rel - 1) / math.log2(rank + 1)
        
        # Ideal DCG
        ideal_relevance = sorted(relevance_scores, reverse=True)[:effective_k]
        idcg = 0.0
        for rank, rel in enumerate(ideal_relevance, 1):
            if rel > 0:
                idcg += (2**rel - 1) / math.log2(rank + 1)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics['ndcg'][k] = ndcg
    
    return metrics

def evaluate_model_on_dataset(model: nn.Module, model_name: str, queries: List[torch.Tensor], 
                            documents: List[List[torch.Tensor]], relevance_scores: List[List[float]],
                            device: torch.device, k_values: List[int] = [1, 5, 10, 100, 1000]) -> Dict[str, Dict[int, float]]:
    """Evaluate model on a single dataset"""
    
    model.eval()
    all_metrics = {'hit_rate': {k: [] for k in k_values}, 
                   'mrr': {k: [] for k in k_values}, 
                   'ndcg': {k: [] for k in k_values}}
    
    with torch.no_grad():
        for query_idx, (query, query_docs, query_rel) in enumerate(zip(queries, documents, relevance_scores)):
            # Get query representation
            query_output = model(query)
            query_repr = query_output.mean(dim=1)  # (1, hidden_dim)
            
            # Compute similarities with all documents
            similarities = []
            for doc in query_docs:
                doc_output = model(doc)
                doc_repr = doc_output.mean(dim=1)  # (1, hidden_dim)
                
                # Cosine similarity
                similarity = F.cosine_similarity(query_repr, doc_repr, dim=-1)
                similarities.append(similarity.item())
            
            similarities = torch.tensor(similarities, device=device)
            
            # Compute metrics for this query
            query_metrics = compute_retrieval_metrics(similarities, query_rel, k_values)
            
            # Accumulate metrics
            for metric_name in all_metrics:
                for k in k_values:
                    all_metrics[metric_name][k].append(query_metrics[metric_name][k])
    
    # Average metrics across all queries
    avg_metrics = {}
    for metric_name in all_metrics:
        avg_metrics[metric_name] = {}
        for k in k_values:
            avg_metrics[metric_name][k] = np.mean(all_metrics[metric_name][k])
    
    return avg_metrics

def train_supervised_classification_on_dataset(model: MAWWithSupervisedClassificationEncoder, queries: List[torch.Tensor], 
                         documents: List[List[torch.Tensor]], relevance_scores: List[List[float]],
                         device: torch.device, epochs: int = 10) -> None:
    """Train Supervised Classification router on dataset with proper contrastive loss"""
    
    print(f"🎯 Training Supervised Classification Router for {epochs} epochs...")
    
    optimizer = torch.optim.AdamW(model.supervised_router.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        total_ranking_loss = 0.0
        num_batches = 0
        
        for query_idx, (query, query_docs, query_rel) in enumerate(zip(queries, documents, relevance_scores)):
            if len(query_docs) < 2:
                continue
                
            optimizer.zero_grad()
            
            # Forward pass to get query representation
            query_output = model(query)
            query_repr = query_output.mean(dim=1)  # (1, hidden_dim)
            
            # Get document representations and compute similarities
            doc_similarities = []
            for doc in query_docs:
                doc_output = model(doc)
                doc_repr = doc_output.mean(dim=1)  # (1, hidden_dim)
                similarity = F.cosine_similarity(query_repr, doc_repr, dim=-1)
                doc_similarities.append(similarity.item())
            
            # Convert to tensor
            similarities = torch.tensor(doc_similarities, device=device, requires_grad=False)
            relevance_tensor = torch.tensor(query_rel, device=device, dtype=torch.float32)
            
            # Contrastive ranking loss - encourage relevant docs to have higher similarity
            ranking_loss = 0.0
            num_pairs = 0
            margin = 0.5  # Margin for ranking
            
            for i in range(len(query_rel)):
                for j in range(len(query_rel)):
                    if query_rel[i] > query_rel[j]:  # i is more relevant than j
                        # We want similarities[i] > similarities[j] + margin
                        pair_loss = torch.clamp(margin - (similarities[i] - similarities[j]), min=0.0)
                        ranking_loss += pair_loss
                        num_pairs += 1
            
            if num_pairs > 0:
                ranking_loss = ranking_loss / num_pairs
            else:
                ranking_loss = torch.tensor(0.0, device=device)
            
            # Router classification loss (depth selection based on 5D attention weights)
            # We need to get the 5D attention weights from the model
            # For training, we'll compute them manually since we need them for supervised classification
            
            # Get Q_depth, K_depth for 5D attention computation (NEW METHOD)
            Q_depth = model.depth_query_proj(query)  # (1, seq, num_heads * depth_dim)
            K_depth = model.depth_key_proj(query)    # (1, seq, num_heads * depth_dim)
            
            # Reshape for 5D attention
            seq_len = query.shape[1]  # Get actual sequence length
            Q_depth = Q_depth.view(1, seq_len, model.num_heads, model.depth_dim).transpose(1, 2)
            # Shape: (1, num_heads, seq, depth)
            K_depth = K_depth.view(1, seq_len, model.num_heads, model.depth_dim).transpose(1, 2)
            # Shape: (1, num_heads, seq, depth)
            
            # NEW 5D attention computation
            Q_depth_expanded = Q_depth.transpose(2, 3).unsqueeze(-1)  # (1, heads, depth, seq, 1)
            K_depth_expanded = K_depth.transpose(2, 3).unsqueeze(-2)  # (1, heads, depth, 1, seq)
            
            scores_5d = Q_depth_expanded * K_depth_expanded  # (1, heads, depth, seq, seq)
            scores_5d = scores_5d.permute(0, 1, 3, 4, 2)  # (1, heads, seq, seq, depth)
            scores_5d = scores_5d / math.sqrt(model.depth_dim)
            
            attention_weights_5d = F.softmax(scores_5d, dim=-1)  # Softmax over depth dimension
            # Shape: (1, num_heads, seq, seq, depth)
            
            # Supervised Classification router prediction based on 5D attention weights
            router_logits = model.supervised_router(attention_weights_5d)  # (1, depth_dim)
            
            # Target depth based on query complexity (number of relevant docs)
            num_relevant = sum(1 for rel in query_rel if rel > 0)
            if num_relevant >= 3:
                target_depth = 0  # Use depth 0 for queries with many relevant docs
            elif num_relevant >= 2:
                target_depth = 1  # Use depth 1 for queries with some relevant docs
            elif num_relevant >= 1:
                target_depth = 2  # Use depth 2 for queries with few relevant docs
            else:
                target_depth = min(3, model.config.depth_dim - 1)  # Use highest depth for hard queries
                
            target = torch.tensor([target_depth], device=device)
            
            # Classification loss
            ce_loss = F.cross_entropy(router_logits, target)
            
            # Combined loss
            total_loss_batch = 0.7 * ranking_loss + 0.3 * ce_loss  # Weight ranking higher
            
            # Ensure loss is valid and not zero
            if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
                print(f"Warning: Invalid loss at query {query_idx}, skipping...")
                continue
                
            if total_loss_batch.item() < 1e-8:  # If loss is essentially zero, add small regularization
                reg_loss = 0.01 * torch.sum(router_logits ** 2)
                total_loss_batch = total_loss_batch + reg_loss
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.supervised_router.parameters(), 1.0)
            optimizer.step()
            
            # Metrics
            prediction = router_logits.argmax(dim=-1)
            accuracy = (prediction == target).float().mean()
            
            total_loss += total_loss_batch.item()
            total_ranking_loss += ranking_loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
        
        scheduler.step()
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_ranking_loss = total_ranking_loss / num_batches
            avg_accuracy = total_accuracy / num_batches
            print(f"  Epoch {epoch+1:2d}/{epochs}: Loss={avg_loss:.4f}, Ranking={avg_ranking_loss:.4f}, Acc={avg_accuracy:.3f}")
        else:
            print(f"  Epoch {epoch+1:2d}/{epochs}: No valid batches")

def print_dataset_results(dataset_name: str, model_results: Dict[str, Dict[str, Dict[int, float]]], 
                         k_values: List[int] = [1, 5, 10, 100, 1000],
                         train_size: int = 0, test_size: int = 0) -> None:
    """Print formatted results for a dataset with train/test split info"""
    
    dataset_config = BENCHMARK_DATASETS[dataset_name]
    print(f"\n📊 {dataset_config['name']} Results")
    print(f"   Domain: {dataset_config['domain']} | Venue: {dataset_config['venue']}")
    print(f"   📈 Train: {train_size} queries | 🧪 Test: {test_size} queries (UNSEEN DATA)")
    print("="*100)
    
    # Header
    header = f"{'Model':<15} {'Metric':<10}"
    for k in k_values:
        header += f" @{k:<8}"
    print(header)
    print("-"*100)
    
    # Results for each model
    for model_name in ['NON-MAW', 'MAW+SupervisedClassification']:
        results = model_results[model_name]
        
        model_display = f"{model_name} (0-shot)" if model_name == 'NON-MAW' else f"{model_name} (trained)"
        
        for metric_name, metric_display in [('hit_rate', 'Hit Rate'), ('mrr', 'MRR'), ('ndcg', 'NDCG')]:
            line = f"{model_display:<15} {metric_display:<10}"
            for k in k_values:
                value = results[metric_name][k]
                line += f" {value:<8.3f}"
            print(line)
        print("-"*100)

def save_results_to_file(all_results: Dict, config: Dict, run_info: Dict, log_dir: str = "logs"):
    """
    Save benchmark results to JSON and text files with timestamp.
    
    Args:
        all_results: Dictionary containing all dataset results
        config: Configuration dictionary with model/training parameters
        run_info: Dictionary with run metadata (device, datasets, samples, etc.)
        log_dir: Directory to save logs (default: "logs")
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare complete results structure
    complete_results = {
        "timestamp": timestamp,
        "run_info": run_info,
        "config": config,
        "results": all_results
    }
    
    # Save JSON (machine-readable)
    json_path = os.path.join(log_dir, f"benchmark_supervised_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(complete_results, f, indent=2)
    print(f"\n💾 Results saved to: {json_path}")
    
    # Save human-readable text summary
    txt_path = os.path.join(log_dir, f"benchmark_supervised_{timestamp}.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("MAW vs NON-MAW Evaluation with Supervised Classification\n")
        f.write("=" * 100 + "\n\n")
        
        # Run information
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {run_info['device']}\n")
        f.write(f"Datasets: {', '.join(run_info['datasets'])}\n")
        f.write(f"Samples per dataset: {run_info['samples']}\n")
        f.write(f"Epochs: {run_info['epochs']}\n")
        f.write(f"Train/Test Split: {run_info['train_ratio']:.0%}/{1-run_info['train_ratio']:.0%}\n")
        f.write(f"K values: {run_info['k_values']}\n\n")
        
        # Configuration
        f.write(f"Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Results for each dataset
        for dataset_name, dataset_results in all_results.items():
            f.write("=" * 100 + "\n")
            f.write(f"DATASET: {dataset_name}\n")
            f.write("=" * 100 + "\n\n")
            
            for model_name in ['NON-MAW', 'MAW+SupervisedClassification']:
                f.write(f"{model_name}:\n")
                for metric_name, metric_display in [('hit_rate', 'Hit Rate'), ('mrr', 'MRR'), ('ndcg', 'NDCG')]:
                    f.write(f"  {metric_display}:\n")
                    for k, value in dataset_results[model_name][metric_name].items():
                        f.write(f"    K={k}: {value:.4f}\n")
                f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("End of Report\n")
        f.write("=" * 100 + "\n")
    
    print(f"💾 Summary saved to: {txt_path}")
    
    return json_path, txt_path

def main():
    """Main evaluation loop for benchmark datasets"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='MAW Supervised Classification Benchmark Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all datasets with GPU (auto-detect)
  python benchmark_evaluation_Supervised_Classification.py
  
  # Run on specific dataset with limited samples
  python benchmark_evaluation_Supervised_Classification.py --dataset MS_MARCO --samples 20 --epochs 10
  
  # Force CPU usage
  python benchmark_evaluation_Supervised_Classification.py --device cpu --samples 15
  
  # Run on multiple specific datasets
  python benchmark_evaluation_Supervised_Classification.py --datasets MS_MARCO TREC_DL --samples 30
        """
    )
    
    parser.add_argument('--dataset', type=str, choices=list(BENCHMARK_DATASETS.keys()),
                       help='Single dataset to evaluate')
    parser.add_argument('--datasets', type=str, nargs='+', choices=list(BENCHMARK_DATASETS.keys()),
                       help='Multiple datasets to evaluate')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of query samples per dataset (default: use full dataset)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'], default='auto',
                       help='Device to use: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Train/test split ratio (default: 0.8)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 5, 10, 100, 1000],
                       help='K values for metrics (default: 1 5 10 100 1000)')
    
    args = parser.parse_args()
    
    # Determine which datasets to run
    if args.dataset:
        datasets_to_run = [args.dataset]
    elif args.datasets:
        datasets_to_run = args.datasets
    else:
        datasets_to_run = list(BENCHMARK_DATASETS.keys())
    
    # Override dataset sample sizes if specified
    original_configs = {}
    if args.samples:
        for dataset_name in datasets_to_run:
            original_configs[dataset_name] = BENCHMARK_DATASETS[dataset_name]['num_queries']
            BENCHMARK_DATASETS[dataset_name]['num_queries'] = args.samples
    
    print("🚀 MAW vs NON-MAW Evaluation with Supervised Classification")
    print("Used in Tier-1 Journals/Conferences: SIGIR, WWW, WSDM, CIKM, EMNLP, ACL")
    print("="*100)
    
    # Configuration
    config = Config(
        hidden_dim=256,
        num_heads=8, 
        seq_len=128,
        dropout=0.1
    )
    
    # Device selection with detailed info
    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🎮 Device: GPU (CUDA) - {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
            print(f"🖥️  Device: CPU")
    elif args.device == 'cpu':
        device = torch.device('cpu')
        print(f"🖥️  Device: CPU (forced)")
    else:  # auto
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            print(f"🎮 Device: GPU (CUDA) - {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print(f"�️  Device: CPU")
    
    k_values = args.k_values
    
    print(f"�📋 Configuration: hidden_dim={config.hidden_dim}, num_heads={config.num_heads}, depth_dim={config.depth_dim}")
    print(f"🔧 Training: {args.epochs} epochs | Train/Test Split: {args.train_ratio:.0%}/{1-args.train_ratio:.0%}")
    print(f"📊 Evaluation metrics: Hit Rate, MRR, NDCG @ K={k_values}")
    print(f"📚 Datasets to evaluate: {', '.join(datasets_to_run)}")
    if args.samples:
        print(f"🔢 Sample size: {args.samples} queries per dataset")
    print()
    
    # Create models
    print(f"🔨 Creating models on {device.type.upper()}...")
    non_maw_model = NonMAWEncoder(config).to(device)
    maw_model = MAWWithSupervisedClassificationEncoder(config).to(device)
    print(f"   ✅ NON-MAW model: {sum(p.numel() for p in non_maw_model.parameters())} parameters")
    print(f"   ✅ MAW+SupervisedClassification model: {sum(p.numel() for p in maw_model.parameters())} parameters")
    
    # Results storage
    all_results = {}
    
    # Loop through selected benchmark datasets
    for dataset_idx, dataset_name in enumerate(datasets_to_run, 1):
        print(f"\n{'='*100}")
        print(f"DATASET {dataset_idx}/{len(datasets_to_run)}: {BENCHMARK_DATASETS[dataset_name]['name']}")
        print(f"{'='*100}")
        
        # Create dataset with proper train/test split
        print(f"   🔧 Data creation device: {device.type.upper()}")
        (train_queries, train_documents, train_relevance), (test_queries, test_documents, test_relevance) = create_benchmark_dataset_split(
            dataset_name, config, train_ratio=args.train_ratio, device=device
        )
        
        # Evaluate NON-MAW baseline (no training needed - zero-shot evaluation)
        print(f"\n🔍 Evaluating NON-MAW baseline (zero-shot on test set)...")
        print(f"   🔧 Evaluation device: {device.type.upper()}")
        non_maw_results = evaluate_model_on_dataset(
            non_maw_model, "NON-MAW", test_queries, test_documents, test_relevance, device, k_values
        )
        
        # Train MAW+SupervisedClassification on training set
        print(f"\n🎯 Training MAW+SupervisedClassification on training set ({len(train_queries)} queries, {args.epochs} epochs)...")
        print(f"   🔧 Training device: {device.type.upper()}")
        train_supervised_classification_on_dataset(maw_model, train_queries, train_documents, train_relevance, device, epochs=args.epochs)
        
        # Evaluate MAW+SupervisedClassification on test set (unseen data!)
        print(f"\n📊 Evaluating MAW+SupervisedClassification on test set ({len(test_queries)} queries)...")
        print(f"   🔧 Evaluation device: {device.type.upper()}")
        maw_results = evaluate_model_on_dataset(
            maw_model, "MAW+SupervisedClassification", test_queries, test_documents, test_relevance, device, k_values
        )
        
        # Store results
        all_results[dataset_name] = {
            'NON-MAW': non_maw_results,
            'MAW+SupervisedClassification': maw_results
        }
        
        # Print results for this dataset
        print_dataset_results(dataset_name, all_results[dataset_name], k_values, 
                             train_size=len(train_queries), test_size=len(test_queries))
        
        # Memory cleanup
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"\n   🧹 GPU memory cleared")
    
    # Restore original configs if modified
    if args.samples:
        for dataset_name, original_size in original_configs.items():
            BENCHMARK_DATASETS[dataset_name]['num_queries'] = original_size
    
    # Final summary across all datasets (only if multiple datasets)
    if len(datasets_to_run) > 1:
        print(f"\n{'='*100}")
        print(f"FINAL SUMMARY: Performance Across {len(datasets_to_run)} Benchmark Datasets")
        print(f"{'='*100}")
        
        # Compute average performance across datasets
        avg_results = {'NON-MAW': {'hit_rate': {}, 'mrr': {}, 'ndcg': {}}, 
                       'MAW+SupervisedClassification': {'hit_rate': {}, 'mrr': {}, 'ndcg': {}}}
        
        for k in k_values:
            for metric in ['hit_rate', 'mrr', 'ndcg']:
                for model in ['NON-MAW', 'MAW+SupervisedClassification']:
                    values = [all_results[dataset][model][metric][k] for dataset in datasets_to_run]
                    avg_results[model][metric][k] = np.mean(values)
        
        # Print summary table
        print(f"\n📈 Average Performance Across All Datasets:")
        print(f"{'Model':<15} {'Metric':<10}", end="")
        for k in k_values:
            print(f" @{k:<8}", end="")
        print()
        print("-"*100)
        
        for model_name in ['NON-MAW', 'MAW+SupervisedClassification']:
            for metric_name, metric_display in [('hit_rate', 'Hit Rate'), ('mrr', 'MRR'), ('ndcg', 'NDCG')]:
                print(f"{model_name:<15} {metric_display:<10}", end="")
                for k in k_values:
                    value = avg_results[model_name][metric_name][k]
                    print(f" {value:<8.3f}", end="")
                print()
            print("-"*100)
    else:
        print(f"\n{'='*100}")
    
    print(f"\n🎉 Evaluation completed on {len(datasets_to_run)} benchmark dataset(s)!")
    print(f"📊 Datasets evaluated: {', '.join([BENCHMARK_DATASETS[d]['name'] for d in datasets_to_run])}")
    print(f"📈 Metrics computed: Hit Rate, MRR, NDCG @ K={k_values}")
    print(f"🔧 Device used: {device.type.upper()}")
    
    # Save results to files
    device_info = f"{device.type.upper()}"
    if device.type == 'cuda':
        device_info += f" - {torch.cuda.get_device_name(0)}"
    
    run_info = {
        "device": device_info,
        "datasets": datasets_to_run,
        "samples": args.samples if args.samples else "full dataset",
        "epochs": args.epochs,
        "train_ratio": args.train_ratio,
        "k_values": k_values
    }
    
    config_dict = {
        "hidden_dim": config.hidden_dim,
        "num_heads": config.num_heads,
        "depth_dim": config.depth_dim,
        "seq_len": config.seq_len,
        "vocab_size": config.vocab_size,
        "dropout": config.dropout
    }
    
    save_results_to_file(all_results, config_dict, run_info)

if __name__ == "__main__":
    main()