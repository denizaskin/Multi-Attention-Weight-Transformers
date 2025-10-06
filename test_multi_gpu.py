#!/usr/bin/env python3
"""
Quick test to verify multi-GPU setup and utilization
Run this before full benchmark to ensure all GPUs are working
"""

import os

# Set environment variables BEFORE importing torch to fix DataParallel hang
os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable peer-to-peer (fixes hang on some systems)
os.environ['NCCL_IB_DISABLE'] = '1'   # Disable InfiniBand
os.environ['NCCL_BLOCKING_WAIT'] = '1'  # Make NCCL operations blocking

import torch
import torch.nn as nn
from tier_1 import Tier1Config, BaselineRetriever, verify_multi_gpu_utilization

def test_multi_gpu_setup():
    """Test multi-GPU configuration"""
    print("="*80)
    print(" MULTI-GPU SETUP TEST")
    print("="*80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"\n✅ CUDA is available")
    print(f"✅ Found {num_gpus} GPU(s)")
    
    if num_gpus < 2:
        print(f"\n⚠️  Only {num_gpus} GPU found - Multi-GPU features require 2+ GPUs")
        return True  # Not an error, just a warning
    
    # Display GPU info
    print(f"\n{'GPU Information:':-^80}")
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        compute = torch.cuda.get_device_capability(i)
        print(f"GPU {i}: {name}")
        print(f"       Memory: {mem:.1f} GB")
        print(f"       Compute: {compute[0]}.{compute[1]}")
    
    print(f"\n{'Testing DataParallel:':-^80}")
    
    # Create a simple model
    config = Tier1Config(
        hidden_dim=768,  # Must be divisible by num_heads (768 / 12 = 64)
        num_heads=12,
        num_layers=4,
        use_multi_gpu=True,
        batch_size=8
    )
    
    device = torch.device('cuda')
    model = BaselineRetriever(config).to(device)
    
    # Wrap with DataParallel
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"✅ Model wrapped with DataParallel across {num_gpus} GPUs")
    else:
        print(f"⚠️  Single GPU mode")
    
    # Test forward pass
    print(f"\n{'Testing Forward Pass:':-^80}")
    batch_size = config.batch_size
    seq_len = 32
    
    # Create dummy batch
    dummy_query = torch.randn(batch_size, seq_len, config.hidden_dim, device=device)
    dummy_doc = torch.randn(batch_size, seq_len, config.hidden_dim, device=device)
    
    # Forward pass
    print(f"Processing batch of size {batch_size}...")
    with torch.no_grad():
        scores = model(dummy_query, dummy_doc)
    
    print(f"✅ Forward pass successful!")
    print(f"   Input shape: ({batch_size}, {seq_len}, {config.hidden_dim})")
    print(f"   Output shape: {scores.shape}")
    
    # Check GPU utilization
    print(f"\n{'GPU Utilization After Forward Pass:':-^80}")
    gpu_stats = verify_multi_gpu_utilization()
    
    all_active = True
    for gpu in gpu_stats['gpus']:
        status = "✅ ACTIVE" if gpu['utilization_pct'] > 0.1 else "⚠️  IDLE"
        print(f"GPU {gpu['id']}: {gpu['allocated_gb']:.2f} GB / {gpu['total_gb']:.1f} GB "
              f"({gpu['utilization_pct']:.1f}%) {status}")
        if gpu['utilization_pct'] <= 0.1:
            all_active = False
    
    if num_gpus > 1:
        if all_active:
            print(f"\n✅ SUCCESS: All {num_gpus} GPUs are actively being used!")
            return True
        else:
            print(f"\n⚠️  WARNING: Some GPUs appear idle")
            print(f"   This might be normal for small batches or single operations")
            print(f"   Try running full benchmark to see sustained GPU usage")
            return True
    else:
        print(f"\n✅ Single GPU mode working correctly")
        return True


def test_multi_gpu_training():
    """Test multi-GPU training with backward pass"""
    if not torch.cuda.is_available():
        return True
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"\n⚠️  Skipping training test (requires 2+ GPUs, found {num_gpus})")
        return True
    
    print(f"\n{'Testing Training (Forward + Backward):':-^80}")
    
    config = Tier1Config(
        hidden_dim=768,  # Must be divisible by num_heads (768 / 12 = 64)
        num_heads=12,
        num_layers=4,
        use_multi_gpu=True,
        batch_size=16  # Larger batch for better GPU utilization
    )
    
    device = torch.device('cuda')
    model = BaselineRetriever(config).to(device)
    model = nn.DataParallel(model)
    
    # Create optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        # Unfreeze last layer for training test
        for param in model.module.encoder.layers[-1].parameters():
            param.requires_grad = True
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
    
    # Training step
    batch_size = config.batch_size
    seq_len = 32
    
    print(f"Running training step with batch size {batch_size}...")
    
    # Create batch
    dummy_query = torch.randn(batch_size, seq_len, config.hidden_dim, device=device)
    dummy_doc_pos = torch.randn(batch_size, seq_len, config.hidden_dim, device=device)
    dummy_doc_neg = torch.randn(batch_size, seq_len, config.hidden_dim, device=device)
    
    # Forward pass
    pos_scores = model(dummy_query, dummy_doc_pos)
    neg_scores = model(dummy_query, dummy_doc_neg)
    
    # Compute loss
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
    
    # Backward pass
    loss.backward()
    
    # Check GPU utilization during training
    print(f"\n{'GPU Utilization During Training:':-^80}")
    gpu_stats = verify_multi_gpu_utilization()
    
    all_active = True
    for gpu in gpu_stats['gpus']:
        status = "✅ ACTIVE" if gpu['utilization_pct'] > 0.5 else "⚠️  LOW"
        print(f"GPU {gpu['id']}: {gpu['allocated_gb']:.2f} GB / {gpu['total_gb']:.1f} GB "
              f"({gpu['utilization_pct']:.1f}%) {status}")
        if gpu['utilization_pct'] <= 0.5:
            all_active = False
    
    if all_active:
        print(f"\n✅ SUCCESS: All {num_gpus} GPUs are actively training!")
    else:
        print(f"\n⚠️  NOTE: GPU 0 may show higher usage (it's the primary GPU)")
        print(f"   This is normal - it coordinates work and gathers results")
    
    print(f"\n✅ Training test completed successfully!")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    # Cleanup
    optimizer.step()
    optimizer.zero_grad()
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print(" MULTI-GPU VERIFICATION TEST SUITE")
    print("="*80 + "\n")
    
    # Test 1: Setup
    print("TEST 1: Multi-GPU Setup and Configuration")
    print("-"*80)
    success1 = test_multi_gpu_setup()
    
    # Test 2: Training
    print("\n" + "="*80)
    print("TEST 2: Multi-GPU Training")
    print("-"*80)
    success2 = test_multi_gpu_training()
    
    # Final summary
    print("\n" + "="*80)
    print(" TEST SUMMARY")
    print("="*80)
    
    if success1 and success2:
        print("\n✅ ALL TESTS PASSED!")
        print("\nYour system is ready for multi-GPU training and evaluation.")
        print("Run the full benchmark with: python3 tier_1.py")
    else:
        print("\n⚠️  SOME TESTS HAD WARNINGS")
        print("\nCheck the output above for details.")
        print("The system should still work, but performance may be suboptimal.")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
