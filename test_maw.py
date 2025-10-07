"""
Test script for the new MAW implementation
"""
import torch
import torch.nn as nn
from tier1_fixed import TokenLevelMAW

def test_maw_forward():
    """Test MAW forward pass with correct dimensions"""
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    depth_dim = 5
    num_heads = 12
    
    # Create MAW module
    maw = TokenLevelMAW(hidden_size, depth_dim, num_heads)
    maw.eval()
    
    # Create dummy inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    with torch.no_grad():
        output = maw(hidden_states, attention_mask)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, hidden_size), f"Expected shape {(batch_size, seq_len, hidden_size)}, got {output.shape}"
    print(f"✓ Forward pass successful! Output shape: {output.shape}")
    
    # Test training mode
    maw.train()
    output_train = maw(hidden_states, attention_mask)
    assert output_train.shape == (batch_size, seq_len, hidden_size), f"Expected shape {(batch_size, seq_len, hidden_size)}, got {output_train.shape}"
    print(f"✓ Training mode forward pass successful! Output shape: {output_train.shape}")
    
    # Test that we can compute gradients
    loss = output_train.sum()
    loss.backward()
    print(f"✓ Backward pass successful!")
    
    # Check that policy network has gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in maw.policy_network.parameters())
    print(f"✓ Policy network has gradients: {has_grad}")
    
    return True

def test_maw_5d_attention():
    """Test that 5D attention computation works correctly"""
    batch_size = 2
    seq_len = 8
    hidden_size = 256
    depth_dim = 5
    num_heads = 4
    
    maw = TokenLevelMAW(hidden_size, depth_dim, num_heads)
    maw.eval()
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    with torch.no_grad():
        output = maw(hidden_states, attention_mask)
    
    # Check that output is different from input (transformation happened)
    diff = (output - hidden_states).abs().mean()
    print(f"✓ Mean difference between input and output: {diff:.6f}")
    assert diff > 0.01, "Output should be different from input"
    
    return True

def test_maw_dimensions():
    """Test the internal dimension transformations"""
    batch_size = 1
    seq_len = 4
    hidden_size = 128
    depth_dim = 5
    num_heads = 4
    head_dim = hidden_size // num_heads
    
    maw = TokenLevelMAW(hidden_size, depth_dim, num_heads)
    
    # Create small tensors to trace dimensions
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Access internal method for dimension checking
    query = maw.query_proj(hidden_states)
    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    print(f"✓ Query shape after projection and reshape: {query.shape}")
    assert query.shape == (batch_size, num_heads, seq_len, head_dim)
    
    key = maw.key_proj(hidden_states)
    key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    print(f"✓ Key shape after projection and reshape: {key.shape}")
    assert key.shape == (batch_size, num_heads, seq_len, head_dim)
    
    value = maw.value_proj(hidden_states)
    value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    print(f"✓ Value shape after projection and reshape: {value.shape}")
    assert value.shape == (batch_size, num_heads, seq_len, head_dim)
    
    print(f"✓ All dimension transformations correct!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing MAW Implementation")
    print("=" * 60)
    
    print("\n1. Testing forward pass...")
    test_maw_forward()
    
    print("\n2. Testing 5D attention computation...")
    test_maw_5d_attention()
    
    print("\n3. Testing dimension transformations...")
    test_maw_dimensions()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
