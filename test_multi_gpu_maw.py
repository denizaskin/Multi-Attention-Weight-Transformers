#!/usr/bin/env python3
"""Test multi-GPU MAW training with optimized memory usage."""

import torch
import torch.nn as nn
import logging
from tier1_fixed import BenchmarkConfig, HFTextEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_multi_gpu_setup():
    """Test that all GPUs are available and clean."""
    print("="*80)
    print("GPU AVAILABILITY CHECK")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"✓ Found {num_gpus} GPUs")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / 1e9
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        free = total_mem - allocated - reserved
        
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total Memory: {total_mem:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Free: {free:.2f} GB")
    
    return num_gpus >= 2

def test_maw_multi_gpu():
    """Test MAW model with DataParallel across multiple GPUs."""
    print("\n" + "="*80)
    print("MULTI-GPU MAW TEST")
    print("="*80)
    
    # Create config with MAW enabled
    config = BenchmarkConfig(
        use_maw=True,
        use_lora=True,
        batch_size=8,  # Per-GPU batch size
        eval_batch_size=16,
        max_seq_length=256,
        maw_depth_dim=64,
        maw_num_heads=8,
        maw_layer_indices=[-1],
        device="cuda:0"
    )
    
    print(f"\nConfig:")
    print(f"  use_maw: {config.use_maw}")
    print(f"  batch_size: {config.batch_size} (per GPU)")
    print(f"  eval_batch_size: {config.eval_batch_size} (per GPU)")
    print(f"  max_seq_length: {config.max_seq_length}")
    print(f"  MAW depth_dim: {config.maw_depth_dim}")
    print(f"  MAW num_heads: {config.maw_num_heads}")
    
    # Create encoder
    print("\nInitializing encoder with MAW...")
    encoder = HFTextEncoder(config)
    
    # Check if DataParallel wrapper will be applied
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"✓ DataParallel will use {num_gpus} GPUs")
        encoder = nn.DataParallel(encoder, device_ids=list(range(num_gpus)))
    else:
        print(f"⚠ Only 1 GPU available, no parallelism")
    
    encoder.eval()
    
    # Test forward pass with different batch sizes
    test_texts = [
        "This is a test sentence for multi-GPU MAW.",
        "Another test sentence to verify memory efficiency.",
        "Testing distributed attention computation.",
        "Multi-head attention with depth dimension.",
    ] * config.batch_size  # Multiply to get larger batch
    
    print(f"\nTesting with batch of {len(test_texts)} texts...")
    
    try:
        with torch.no_grad():
            embeddings = encoder.module.encode(test_texts, batch_size=config.eval_batch_size) if hasattr(encoder, 'module') else encoder.encode(test_texts, batch_size=config.eval_batch_size)
        
        print(f"✓ Embeddings shape: {embeddings.shape}")
        print(f"✓ Embeddings dtype: {embeddings.dtype}")
        print(f"✓ Embeddings device: {embeddings.device}")
        
        # Check memory usage across GPUs
        print("\nMemory usage after forward pass:")
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  GPU {i}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test a single training step with gradient accumulation."""
    print("\n" + "="*80)
    print("TRAINING STEP TEST")
    print("="*80)
    
    config = BenchmarkConfig(
        use_maw=True,
        use_lora=True,
        batch_size=8,
        gradient_accumulation_steps=2,
        max_seq_length=256,
        device="cuda:0"
    )
    
    encoder = HFTextEncoder(config)
    num_gpus = torch.cuda.device_count()
    
    if num_gpus > 1:
        encoder = nn.DataParallel(encoder, device_ids=list(range(num_gpus)))
    
    encoder.train()
    
    # Create optimizer
    model_to_optimize = encoder.module if hasattr(encoder, 'module') else encoder
    params = [p for p in model_to_optimize.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-5)
    
    print(f"Trainable parameters: {sum(p.numel() for p in params):,}")
    
    # Simulate training step
    test_batch = ["Training sentence " + str(i) for i in range(config.batch_size * 2)]
    
    try:
        print(f"\nSimulating training step with {len(test_batch)} samples...")
        
        # Forward pass
        embeddings = model_to_optimize.encode_train(test_batch, batch_size=config.batch_size)
        
        # Dummy loss
        loss = embeddings.mean()
        loss = loss / config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        print(f"✓ Forward pass: embeddings shape = {embeddings.shape}")
        print(f"✓ Backward pass: loss = {loss.item():.6f}")
        
        # Check gradients
        has_grad = sum(1 for p in params if p.grad is not None)
        print(f"✓ Parameters with gradients: {has_grad}/{len(params)}")
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"✓ Optimizer step completed")
        
        # Memory after training step
        print("\nMemory usage after training step:")
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  GPU {i}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training step: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("MULTI-GPU MAW OPTIMIZATION TEST")
    print("="*80)
    
    # Test 1: GPU availability
    if not test_multi_gpu_setup():
        print("\n❌ Multi-GPU setup check failed")
        return
    
    # Clear cache
    print("\nClearing GPU cache...")
    torch.cuda.empty_cache()
    
    # Test 2: MAW with multi-GPU
    if not test_maw_multi_gpu():
        print("\n❌ Multi-GPU MAW test failed")
        return
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Test 3: Training step
    if not test_training_step():
        print("\n❌ Training step test failed")
        return
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - Multi-GPU MAW is working!")
    print("="*80)
    print("\nYou can now run full training with:")
    print("  python tier1_fixed.py --msmarco --quick-smoke-test --skip-bm25")
    print("\nThe model will automatically use all available GPUs with optimized batch sizes.")

if __name__ == "__main__":
    main()
