"""
Test MAW layer selection functionality
"""
import torch
from tier1_fixed import BenchmarkConfig, HFTextEncoder

def test_maw_last_layer_default():
    """Test that MAW is applied to last layer by default"""
    print("Test 1: MAW on last layer (default)")
    
    config = BenchmarkConfig(
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        use_maw=True,
        maw_layer_indices=[-1],  # Last layer (default)
        maw_depth_dim=5,
        maw_num_heads=4,
    )
    
    encoder = HFTextEncoder(config)
    num_layers = encoder._get_num_encoder_layers(encoder.model.config)
    
    print(f"  Total encoder layers: {num_layers}")
    print(f"  MAW applied to layers: {encoder.maw_layer_indices}")
    print(f"  Expected: [{num_layers - 1}] (last layer)")
    
    assert encoder.maw_layer_indices == [num_layers - 1], f"Expected last layer {num_layers-1}, got {encoder.maw_layer_indices}"
    print("  ✓ MAW correctly applied to last layer only\n")


def test_maw_specific_layers():
    """Test MAW on specific layers"""
    print("Test 2: MAW on specific layers (0, 3, 5)")
    
    config = BenchmarkConfig(
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        use_maw=True,
        maw_layer_indices=[0, 3, 5],
        maw_depth_dim=5,
        maw_num_heads=4,
    )
    
    encoder = HFTextEncoder(config)
    
    print(f"  Total encoder layers: {encoder._get_num_encoder_layers(encoder.model.config)}")
    print(f"  MAW applied to layers: {encoder.maw_layer_indices}")
    print(f"  Expected: [0, 3, 5]")
    
    assert encoder.maw_layer_indices == [0, 3, 5], f"Expected [0, 3, 5], got {encoder.maw_layer_indices}"
    print("  ✓ MAW correctly applied to specified layers\n")


def test_maw_negative_indexing():
    """Test MAW with negative indices"""
    print("Test 3: MAW on last two layers (negative indexing)")
    
    config = BenchmarkConfig(
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        use_maw=True,
        maw_layer_indices=[-1, -2],  # Last two layers
        maw_depth_dim=5,
        maw_num_heads=4,
    )
    
    encoder = HFTextEncoder(config)
    num_layers = encoder._get_num_encoder_layers(encoder.model.config)
    
    print(f"  Total encoder layers: {num_layers}")
    print(f"  MAW applied to layers: {encoder.maw_layer_indices}")
    print(f"  Expected: [{num_layers - 2}, {num_layers - 1}] (last two layers)")
    
    assert encoder.maw_layer_indices == [num_layers - 2, num_layers - 1], \
        f"Expected last two layers, got {encoder.maw_layer_indices}"
    print("  ✓ MAW correctly applied to last two layers\n")


def test_maw_all_layers():
    """Test MAW on all layers"""
    print("Test 4: MAW on all layers")
    
    config = BenchmarkConfig(
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        use_maw=True,
        maw_layer_indices=list(range(100)),  # Simulates 'all'
        maw_depth_dim=5,
        maw_num_heads=4,
    )
    
    encoder = HFTextEncoder(config)
    num_layers = encoder._get_num_encoder_layers(encoder.model.config)
    
    print(f"  Total encoder layers: {num_layers}")
    print(f"  MAW applied to layers: {encoder.maw_layer_indices}")
    print(f"  Expected: all layers [0, 1, ..., {num_layers-1}]")
    
    assert encoder.maw_layer_indices == list(range(num_layers)), \
        f"Expected all layers, got {encoder.maw_layer_indices}"
    print("  ✓ MAW correctly applied to all layers\n")


def test_maw_forward_pass():
    """Test that forward pass works with MAW on different layers"""
    print("Test 5: Forward pass with MAW on last layer")
    
    config = BenchmarkConfig(
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        use_maw=True,
        maw_layer_indices=[-1],
        maw_depth_dim=5,
        maw_num_heads=4,
    )
    
    encoder = HFTextEncoder(config)
    encoder.eval()
    
    texts = ["This is a test.", "Another test sentence."]
    
    with torch.no_grad():
        embeddings = encoder.encode_train(texts)
    
    print(f"  Embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (2, encoder.hidden_size_value)
    print("  ✓ Forward pass successful\n")


def test_parse_layer_spec():
    """Test the layer specification parsing"""
    print("Test 6: Layer specification parsing")
    
    from tier1_fixed import _parse_maw_layer_indices
    
    # Test default
    result = _parse_maw_layer_indices("-1")
    print(f"  '-1' -> {result}")
    assert result == [-1], f"Expected [-1], got {result}"
    
    # Test multiple layers
    result = _parse_maw_layer_indices("0,5,11")
    print(f"  '0,5,11' -> {result}")
    assert result == [0, 5, 11], f"Expected [0, 5, 11], got {result}"
    
    # Test negative indices
    result = _parse_maw_layer_indices("-1,-2")
    print(f"  '-1,-2' -> {result}")
    assert result == [-1, -2], f"Expected [-1, -2], got {result}"
    
    # Test 'all'
    result = _parse_maw_layer_indices("all")
    print(f"  'all' -> range(0, 100) [first few: {result[:5]}...]")
    assert len(result) == 100, f"Expected 100 elements for 'all', got {len(result)}"
    
    print("  ✓ All parsing tests passed\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing MAW Layer Selection")
    print("=" * 70)
    print()
    
    try:
        test_parse_layer_spec()
        test_maw_last_layer_default()
        test_maw_specific_layers()
        test_maw_negative_indexing()
        test_maw_all_layers()
        test_maw_forward_pass()
        
        print("=" * 70)
        print("All MAW layer selection tests passed! ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
