"""
Example usage of MAW with layer selection

This script demonstrates how to use the MAW layer selection feature
in different configurations.
"""
import torch
from tier1_fixed import BenchmarkConfig, HFTextEncoder

def example_default():
    """Example 1: Default configuration (last layer only)"""
    print("=" * 70)
    print("Example 1: Default MAW (Last Layer Only)")
    print("=" * 70)
    
    config = BenchmarkConfig(
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        use_maw=True,
        # maw_layer_indices defaults to [-1]
        maw_depth_dim=5,
        maw_num_heads=4,
    )
    
    encoder = HFTextEncoder(config)
    print(f"Model: {config.dense_model}")
    print(f"Total layers: {encoder._get_num_encoder_layers(encoder.model.config)}")
    print(f"MAW applied to: {encoder.maw_layer_indices}")
    print(f"This is the RECOMMENDED configuration for most users!")
    print()


def example_multiple_layers():
    """Example 2: Multiple specific layers"""
    print("=" * 70)
    print("Example 2: MAW on Multiple Specific Layers")
    print("=" * 70)
    
    config = BenchmarkConfig(
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        use_maw=True,
        maw_layer_indices=[0, 2, 5],  # First, middle, and last layers
        maw_depth_dim=5,
        maw_num_heads=4,
    )
    
    encoder = HFTextEncoder(config)
    print(f"Model: {config.dense_model}")
    print(f"Total layers: {encoder._get_num_encoder_layers(encoder.model.config)}")
    print(f"MAW applied to: {encoder.maw_layer_indices}")
    print(f"Use case: When you want MAW at strategic points in the encoder")
    print()


def example_top_layers():
    """Example 3: Last N layers"""
    print("=" * 70)
    print("Example 3: MAW on Last 3 Layers")
    print("=" * 70)
    
    config = BenchmarkConfig(
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        use_maw=True,
        maw_layer_indices=[-1, -2, -3],  # Last 3 layers
        maw_depth_dim=5,
        maw_num_heads=4,
    )
    
    encoder = HFTextEncoder(config)
    num_layers = encoder._get_num_encoder_layers(encoder.model.config)
    print(f"Model: {config.dense_model}")
    print(f"Total layers: {num_layers}")
    print(f"MAW applied to: {encoder.maw_layer_indices}")
    print(f"These are layers {num_layers-3}, {num_layers-2}, {num_layers-1}")
    print(f"Use case: Enhanced final representations")
    print()


def example_all_layers():
    """Example 4: All layers"""
    print("=" * 70)
    print("Example 4: MAW on All Layers")
    print("=" * 70)
    
    config = BenchmarkConfig(
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        use_maw=True,
        maw_layer_indices=list(range(100)),  # Will be clamped to actual layer count
        maw_depth_dim=5,
        maw_num_heads=4,
    )
    
    encoder = HFTextEncoder(config)
    print(f"Model: {config.dense_model}")
    print(f"Total layers: {encoder._get_num_encoder_layers(encoder.model.config)}")
    print(f"MAW applied to: {encoder.maw_layer_indices}")
    print(f"Use case: Maximum expressiveness (but slowest)")
    print(f"WARNING: High memory and compute requirements!")
    print()


def example_encoding():
    """Example 5: Actual encoding with MAW"""
    print("=" * 70)
    print("Example 5: Encoding Text with MAW")
    print("=" * 70)
    
    config = BenchmarkConfig(
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        use_maw=True,
        maw_layer_indices=[-1],  # Last layer (default)
        maw_depth_dim=5,
        maw_num_heads=4,
    )
    
    encoder = HFTextEncoder(config)
    encoder.eval()
    
    texts = [
        "The cat sits on the mat.",
        "A dog plays in the park.",
        "Birds fly in the sky."
    ]
    
    print(f"Encoding {len(texts)} texts with MAW on layer {encoder.maw_layer_indices}...")
    
    with torch.no_grad():
        embeddings = encoder.encode_train(texts)
    
    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"✓ Embedding dimension: {embeddings.shape[1]}")
    
    # Check normalization (move to same device for comparison)
    norms = torch.norm(embeddings, dim=1)
    expected = torch.ones(len(texts), device=embeddings.device)
    is_normalized = torch.allclose(norms, expected, atol=1e-5)
    print(f"✓ L2 normalized: {is_normalized}")
    print()


def example_cli_equivalents():
    """Example 6: CLI command equivalents"""
    print("=" * 70)
    print("Example 6: Equivalent CLI Commands")
    print("=" * 70)
    
    examples = [
        ("Last layer (default)", 
         "python tier1_fixed.py --use-maw",
         "maw_layer_indices=[-1]"),
        
        ("Specific layers",
         "python tier1_fixed.py --use-maw --maw-layer-indices 0,3,5",
         "maw_layer_indices=[0, 3, 5]"),
        
        ("Last 2 layers",
         "python tier1_fixed.py --use-maw --maw-layer-indices=-1,-2",
         "maw_layer_indices=[-1, -2]"),
        
        ("All layers",
         "python tier1_fixed.py --use-maw --maw-layer-indices all",
         "maw_layer_indices=list(range(100))"),
    ]
    
    for name, cli, code in examples:
        print(f"\n{name}:")
        print(f"  CLI:  {cli}")
        print(f"  Code: BenchmarkConfig(use_maw=True, {code})")
    
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "MAW Layer Selection Examples" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    example_default()
    example_multiple_layers()
    example_top_layers()
    example_all_layers()
    example_encoding()
    example_cli_equivalents()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  • Default (last layer only) works great for most tasks")
    print("  • Use negative indexing for top layers (portable across models)")
    print("  • More layers = more compute/memory but potentially better results")
    print("  • Always benchmark different configurations for your specific task")
    print()
