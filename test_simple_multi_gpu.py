#!/usr/bin/env python3
"""
Simple multi-GPU test without MAW/GRPO complexity
"""

import torch
import torch.nn as nn

def test_simple_dataparallel():
    print("="*80)
    print("SIMPLE DATAPARALLEL TEST (No MAW/GRPO)")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"\n✅ Found {num_gpus} GPU(s)")
    
    if num_gpus < 2:
        print("⚠️  Need 2+ GPUs for DataParallel test")
        return True
    
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 512)
            self.fc2 = nn.Linear(512, 256)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    device = torch.device('cuda')
    model = SimpleModel().to(device)
    model = nn.DataParallel(model)
    
    print(f"✅ Model wrapped with DataParallel")
    
    # Test forward pass
    batch_size = 16
    input_tensor = torch.randn(batch_size, 256, device=device)
    
    print(f"🔄 Running forward pass with batch size {batch_size}...")
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"✅ Forward pass successful!")
    print(f"   Input shape: {input_tensor.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Check GPU utilization
    print(f"\n📊 GPU Memory After Forward Pass:")
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / (1024**2)
        print(f"   GPU {i}: {allocated:.1f} MB allocated")
    
    print(f"\n✅ Simple DataParallel test PASSED!")
    return True


if __name__ == "__main__":
    test_simple_dataparallel()
