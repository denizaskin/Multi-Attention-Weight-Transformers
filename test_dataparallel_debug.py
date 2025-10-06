#!/usr/bin/env python3
"""
Test DataParallel with explicit configuration
"""

import os
import torch
import torch.nn as nn

# Set environment variables for better NCCL behavior
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NCCL_BLOCKING_WAIT'] = '0'

def test_with_single_gpu():
    print("=" * 80)
    print("TEST 1: Single GPU (Baseline)")
    print("=" * 80)
    
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 64)
        
        def forward(self, x):
            return self.fc(x)
    
    device = torch.device('cuda:0')
    model = TinyModel().to(device)
    
    x = torch.randn(4, 64, device=device)
    
    print("Running forward pass on single GPU...")
    with torch.no_grad():
        y = model(x)
    
    print(f"✅ Single GPU test PASSED! Output shape: {y.shape}")
    return True


def test_with_dataparallel():
    print("\n" + "=" * 80)
    print("TEST 2: DataParallel (Multi-GPU)")
    print("=" * 80)
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus < 2:
        print("⚠️  Need 2+ GPUs, skipping")
        return True
    
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 64)
        
        def forward(self, x):
            print(f"  [Forward] Input device: {x.device}, shape: {x.shape}")
            out = self.fc(x)
            print(f"  [Forward] Output device: {out.device}, shape: {out.shape}")
            return out
    
    device = torch.device('cuda:0')
    model = TinyModel().to(device)
    
    print(f"\nWrapping model with DataParallel across GPUs: [0, 1]...")
    model = nn.DataParallel(model, device_ids=[0, 1])  # Use only 2 GPUs to start
    
    # Small batch that can be split
    batch_size = 4
    x = torch.randn(batch_size, 64, device=device)
    
    print(f"\nRunning forward pass with batch size {batch_size}...")
    print(f"Input device: {x.device}")
    
    try:
        with torch.no_grad():
            y = model(x)
        
        print(f"\n✅ DataParallel test PASSED!")
        print(f"   Output shape: {y.shape}")
        print(f"   Output device: {y.device}")
        return True
    except Exception as e:
        print(f"\n❌ DataParallel test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_with_single_gpu()
    
    if success1:
        success2 = test_with_dataparallel()
    
    print("\n" + "=" * 80)
    if success1 and success2:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)
