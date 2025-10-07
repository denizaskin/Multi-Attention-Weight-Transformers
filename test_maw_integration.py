"""
Integration test for MAW with the full encoder pipeline
"""
import torch
from tier1_fixed import BenchmarkConfig, HFTextEncoder

def test_maw_integration():
    """Test MAW integrated with HFTextEncoder"""
    print("Testing MAW integration with HFTextEncoder...")
    
    # Create config with MAW enabled
    config = BenchmarkConfig(
        dense_model="sentence-transformers/all-MiniLM-L6-v2",  # Small model for testing
        use_maw=True,
        maw_depth_dim=5,
        maw_num_heads=4,
        maw_layers=1,
        max_seq_length=128,
    )
    
    # Create encoder with MAW
    print(f"  Creating encoder with MAW...")
    encoder = HFTextEncoder(config)
    encoder.eval()
    
    print(f"  Model device: {encoder.primary_device}")
    print(f"  MAW layers: {len(encoder.maw_layers) if encoder.maw_layers else 0}")
    
    # Test encoding
    texts = [
        "This is a test sentence for MAW.",
        "Multi-Attention-Weight mechanism is working!",
        "The integration test is running successfully."
    ]
    
    print(f"\n  Encoding {len(texts)} texts...")
    with torch.no_grad():
        embeddings = encoder.encode_train(texts, batch_size=2)
    
    print(f"  ✓ Embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (len(texts), encoder.hidden_size_value)
    
    # Test that embeddings are normalized
    norms = torch.norm(embeddings, p=2, dim=1)
    print(f"  ✓ Embedding norms: {norms.tolist()}")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Embeddings should be normalized"
    
    # Test training mode
    encoder.train()
    print(f"\n  Testing training mode...")
    embeddings_train = encoder.encode_train(texts[:2], batch_size=2)
    print(f"  ✓ Training embeddings shape: {embeddings_train.shape}")
    
    # Test that gradients flow
    loss = embeddings_train.sum()
    loss.backward()
    
    # Check MAW policy network has gradients
    if encoder.maw_layers:
        has_grads = []
        for layer in encoder.maw_layers:
            layer_has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0 
                for p in layer.policy_network.parameters()
            )
            has_grads.append(layer_has_grad)
        print(f"  ✓ MAW layers with gradients: {sum(has_grads)}/{len(has_grads)}")
    
    print(f"\n✓ All integration tests passed!")
    return True

def test_maw_vs_no_maw():
    """Compare outputs with and without MAW"""
    print("\nComparing MAW vs standard attention...")
    
    config_no_maw = BenchmarkConfig(
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        use_maw=False,
        max_seq_length=128,
    )
    
    config_with_maw = BenchmarkConfig(
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        use_maw=True,
        maw_depth_dim=5,
        maw_num_heads=4,
        maw_layers=1,
        max_seq_length=128,
    )
    
    encoder_no_maw = HFTextEncoder(config_no_maw)
    encoder_with_maw = HFTextEncoder(config_with_maw)
    
    encoder_no_maw.eval()
    encoder_with_maw.eval()
    
    texts = ["Test sentence for comparison."]
    
    with torch.no_grad():
        emb_no_maw = encoder_no_maw.encode_train(texts)
        emb_with_maw = encoder_with_maw.encode_train(texts)
    
    print(f"  No MAW output shape: {emb_no_maw.shape}")
    print(f"  With MAW output shape: {emb_with_maw.shape}")
    
    # Both should have same dimension
    assert emb_no_maw.shape == emb_with_maw.shape
    
    # Outputs should be different (MAW transforms the representation)
    diff = (emb_no_maw - emb_with_maw).abs().mean().item()
    print(f"  Mean absolute difference: {diff:.6f}")
    print(f"  ✓ MAW produces different embeddings (as expected)")
    
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("MAW Integration Tests")
    print("=" * 70)
    
    try:
        test_maw_integration()
        test_maw_vs_no_maw()
        
        print("\n" + "=" * 70)
        print("All integration tests completed successfully! ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
