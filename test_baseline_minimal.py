#!/usr/bin/env python3
"""
Minimal test of BaselineRetriever with DataParallel
"""

import torch
import torch.nn as nn
from tier_1 import Tier1Config, BaselineRetriever

print("Importing modules...")

device = torch.device('cuda')
num_gpus = torch.cuda.device_count()

print(f"✅ Found {num_gpus} GPUs")

# Create config with minimal layers
config = Tier1Config(
    hidden_dim=768,  # Must be divisible by num_heads (768 / 12 = 64)
    num_heads=12,
    num_layers=2,  # Only 2 layers for fast testing
    use_multi_gpu=True
)

print(f"Creating BaselineRetriever...")
model = BaselineRetriever(config).to(device)

print(f"Model created on device")

if num_gpus > 1:
    print(f"Wrapping with DataParallel...")
    model = nn.DataParallel(model)
    print(f"✅ DataParallel wrapped")

# Create tiny input
batch_size = 4
seq_len = 8
print(f"\nCreating input tensors: batch={batch_size}, seq_len={seq_len}...")

dummy_query = torch.randn(batch_size, seq_len, config.hidden_dim, device=device)
dummy_doc = torch.randn(batch_size, seq_len, config.hidden_dim, device=device)

print(f"Input tensors created on device")
print(f"Query shape: {dummy_query.shape}")
print(f"Doc shape: {dummy_doc.shape}")

print(f"\n🔄 Running forward pass...")

try:
    with torch.no_grad():
        scores = model(dummy_query, dummy_doc)
    
    print(f"✅ Forward pass successful!")
    print(f"   Output shape: {scores.shape}")
    print(f"   Output device: {scores.device}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
