"""
Test CLI argument parsing for MAW layer selection
"""
import sys
import argparse

# Add the workspace to path
sys.path.insert(0, '/workspace/Multi-Attention-Weight-Transformers')

from tier1_fixed import parse_args, build_config, _parse_maw_layer_indices

def test_cli_parsing():
    """Test that CLI arguments are parsed correctly"""
    print("Testing CLI Argument Parsing\n")
    print("=" * 70)
    
    # Test 1: Default (last layer)
    print("\nTest 1: Default behavior (no maw args)")
    sys.argv = ['test_cli.py']
    args = parse_args()
    layer_indices = _parse_maw_layer_indices(args.maw_layer_indices)
    print(f"  --maw-layer-indices: {args.maw_layer_indices}")
    print(f"  Parsed as: {layer_indices}")
    assert layer_indices == [-1], f"Expected [-1], got {layer_indices}"
    print("  ✓ Default is last layer\n")
    
    # Test 2: Last layer explicit
    print("Test 2: Explicit last layer")
    sys.argv = ['test_cli.py', '--maw-layer-indices', '-1']
    args = parse_args()
    layer_indices = _parse_maw_layer_indices(args.maw_layer_indices)
    print(f"  --maw-layer-indices -1")
    print(f"  Parsed as: {layer_indices}")
    assert layer_indices == [-1], f"Expected [-1], got {layer_indices}"
    print("  ✓ Explicit last layer works\n")
    
    # Test 3: Multiple specific layers
    print("Test 3: Multiple specific layers")
    sys.argv = ['test_cli.py', '--maw-layer-indices', '0,5,11']
    args = parse_args()
    layer_indices = _parse_maw_layer_indices(args.maw_layer_indices)
    print(f"  --maw-layer-indices 0,5,11")
    print(f"  Parsed as: {layer_indices}")
    assert layer_indices == [0, 5, 11], f"Expected [0, 5, 11], got {layer_indices}"
    print("  ✓ Multiple layers work\n")
    
    # Test 4: Negative indexing (needs to be in quotes for CLI)
    print("Test 4: Negative indexing")
    sys.argv = ['test_cli.py', '--maw-layer-indices=-1,-2,-3']  # Use = to avoid argparse issue
    args = parse_args()
    layer_indices = _parse_maw_layer_indices(args.maw_layer_indices)
    print(f"  --maw-layer-indices=-1,-2,-3")
    print(f"  Parsed as: {layer_indices}")
    assert layer_indices == [-1, -2, -3], f"Expected [-1, -2, -3], got {layer_indices}"
    print("  ✓ Negative indexing works\n")
    
    # Test 5: All layers
    print("Test 5: All layers")
    sys.argv = ['test_cli.py', '--maw-layer-indices', 'all']
    args = parse_args()
    layer_indices = _parse_maw_layer_indices(args.maw_layer_indices)
    print(f"  --maw-layer-indices all")
    print(f"  Parsed as: range(0, 100) (will be clamped to model's layer count)")
    assert len(layer_indices) == 100, f"Expected 100 elements, got {len(layer_indices)}"
    print("  ✓ 'all' works\n")
    
    # Test 6: Build config integration
    print("Test 6: Full config building")
    sys.argv = ['test_cli.py', '--maw-layer-indices', '3,6,9']
    args = parse_args()
    config = build_config(args)
    print(f"  --maw-layer-indices 3,6,9")
    print(f"  Config.maw_layer_indices: {config.maw_layer_indices}")
    assert config.maw_layer_indices == [3, 6, 9], f"Expected [3, 6, 9], got {config.maw_layer_indices}"
    print("  ✓ Config building works\n")
    
    print("=" * 70)
    print("All CLI parsing tests passed! ✓")
    print("=" * 70)

if __name__ == "__main__":
    try:
        test_cli_parsing()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
