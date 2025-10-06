#!/usr/bin/env python3
"""
Test BaselineRetriever WITHOUT DataParallel
"""

import torch
import torch.nn as nn
from tier_1 import Tier1Config, BaselineRetriever

device = torch.device('cuda:0')  # Single GPU only

print("Creating BaselineRetriever (single GPU, no DataParallel)...")
config = Tier1Config(
    hidden_dim=768,  # Must be divisible by num_heads (768 / 12 = 64)
    num_heads=12,
    num_layers=2,
    use_multi_gpu=False  # Explicitly disable
)

model = BaselineRetriever(config).to(device)

print("✅ Model created on single GPU")

# Create input
batch_size = 4
seq_len = 8

dummy_query = torch.randn(batch_size, seq_len, config.hidden_dim, device=device)
dummy_doc = torch.randn(batch_size, seq_len, config.hidden_dim, device=device)

print(f"Running forward pass...")

try:
    with torch.no_grad():
        scores = model(dummy_query, dummy_doc)
    
    print(f"✅ Forward pass successful!")
    print(f"   Output shape: {scores.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
