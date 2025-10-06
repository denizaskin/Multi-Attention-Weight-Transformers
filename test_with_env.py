#!/usr/bin/env python3
"""
Test with environment variables to fix DataParallel hang
"""

import os

# Set environment variables BEFORE importing torch
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable peer-to-peer
os.environ['NCCL_IB_DISABLE'] = '1'   # Disable InfiniBand
os.environ['NCCL_DEBUG'] = 'INFO'

import torch
import torch.nn as nn
from tier_1 import Tier1Config, BaselineRetriever

device = torch.device('cuda')
num_gpus = torch.cuda.device_count()

print(f"✅ Found {num_gpus} GPUs")
print(f"NCCL settings: blocking_wait=1, p2p_disable=1, ib_disable=1")

config = Tier1Config(
    hidden_dim=768,
    num_heads=12,
    num_layers=2,
    use_multi_gpu=True
)

print(f"\n📦 Creating BaselineRetriever...")
model = BaselineRetriever(config).to(device)
print(f"✅ Model created on GPU 0")

if num_gpus > 1:
    print(f"\n🔄 Wrapping with DataParallel (using GPUs 0-{num_gpus-1})...")
    model = nn.DataParallel(model)
    print(f"✅ DataParallel wrapped")

batch_size = 4
seq_len = 8

print(f"\n📝 Creating input tensors...")
dummy_query = torch.randn(batch_size, seq_len, config.hidden_dim, device=device)
dummy_doc = torch.randn(batch_size, seq_len, config.hidden_dim, device=device)
print(f"✅ Input tensors created")

print(f"\n🚀 Running forward pass...")
try:
    with torch.no_grad():
        scores = model(dummy_query, dummy_doc)
    
    print(f"\n✅ SUCCESS! Forward pass completed!")
    print(f"   Output shape: {scores.shape}")
    print(f"   Output values: {scores}")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
