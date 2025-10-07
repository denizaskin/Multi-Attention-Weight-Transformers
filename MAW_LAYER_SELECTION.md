# MAW Layer Selection - Feature Documentation

## Overview

MAW (Multi-Attention-Weight) can now be applied to specific layers of the transformer encoder, providing fine-grained control over where the attention mechanism is enhanced.

## Default Behavior

**By default, MAW is applied ONLY to the last layer of the encoder.**

This is the most common use case as the last layer typically contains the most refined representations before pooling.

## Configuration Options

### 1. Last Layer Only (Default)

```bash
python tier1_fixed.py --use-maw --maw-layer-indices "-1"
```

Or in code:
```python
config = BenchmarkConfig(
    use_maw=True,
    maw_layer_indices=[-1],  # Last layer
)
```

### 2. Specific Layers

Apply MAW to layers 0, 5, and 11:

```bash
python tier1_fixed.py --use-maw --maw-layer-indices "0,5,11"
```

Or in code:
```python
config = BenchmarkConfig(
    use_maw=True,
    maw_layer_indices=[0, 5, 11],
)
```

### 3. Last N Layers (Negative Indexing)

Apply MAW to the last 2 layers:

```bash
python tier1_fixed.py --use-maw --maw-layer-indices "-1,-2"
```

Or in code:
```python
config = BenchmarkConfig(
    use_maw=True,
    maw_layer_indices=[-1, -2],  # Last two layers
)
```

### 4. All Layers

Apply MAW to every encoder layer:

```bash
python tier1_fixed.py --use-maw --maw-layer-indices "all"
```

Or in code:
```python
config = BenchmarkConfig(
    use_maw=True,
    maw_layer_indices=list(range(100)),  # Will be clamped to actual layer count
)
```

## Layer Indexing

### Positive Indices (0-based)
- `0` = First encoder layer
- `1` = Second encoder layer
- `11` = Twelfth encoder layer (for 12-layer models)

### Negative Indices (Python-style)
- `-1` = Last layer
- `-2` = Second-to-last layer
- `-3` = Third-to-last layer

### Special Values
- `all` = All encoder layers

## Examples

### Example 1: Default Usage (Last Layer)
```bash
# Most common use case
python tier1_fixed.py --quick-smoke-test --msmarco --use-maw
```

This applies MAW to the last layer only (layer 11 in BERT-base, layer 5 in MiniLM, etc.)

### Example 2: Multiple Specific Layers
```bash
# Apply MAW to first, middle, and last layers of a 12-layer model
python tier1_fixed.py --msmarco --use-maw --maw-layer-indices "0,5,11"
```

### Example 3: Top 3 Layers
```bash
# Apply MAW to the last 3 layers
python tier1_fixed.py --msmarco --use-maw --maw-layer-indices "-1,-2,-3"
```

### Example 4: All Layers
```bash
# Apply MAW to every layer (most expensive, but potentially most expressive)
python tier1_fixed.py --msmarco --use-maw --maw-layer-indices "all"
```

### Example 5: Programmatic Configuration
```python
from tier1_fixed import BenchmarkConfig, BenchmarkRunner

# Apply MAW to layers 3, 6, and 9
config = BenchmarkConfig(
    dense_model="facebook/contriever",
    use_maw=True,
    maw_layer_indices=[3, 6, 9],
    maw_depth_dim=5,
    maw_num_heads=12,
)

runner = BenchmarkRunner(config)
runner.run()
```

## Technical Details

### How It Works

1. **Layer Detection**: The encoder automatically detects the number of layers in the model
2. **Index Resolution**: Negative indices are converted to positive (e.g., -1 becomes 11 for a 12-layer model)
3. **Validation**: Out-of-range indices are filtered out with warnings
4. **Hook Integration**: MAW modules are applied to the output of specified encoder layers
5. **Hidden States**: The model uses `output_hidden_states=True` to access intermediate representations

### Implementation

The MAW mechanism is inserted **after** the specified encoder layers:

```
Input Embeddings
      ↓
Encoder Layer 0 ──→ [MAW if 0 in maw_layer_indices]
      ↓
Encoder Layer 1 ──→ [MAW if 1 in maw_layer_indices]
      ↓
     ...
      ↓
Encoder Layer N-1 ──→ [MAW if N-1 in maw_layer_indices]
      ↓
Mean Pooling
      ↓
L2 Normalization
      ↓
Output
```

### Performance Considerations

| Configuration | Relative Speed | Memory Usage | Expressiveness |
|--------------|----------------|--------------|----------------|
| Last layer only (default) | Fast (1.0x) | Low | Good |
| 2-3 specific layers | Medium (0.7x) | Medium | Better |
| All layers | Slow (0.3x) | High | Best |

**Recommendation**: Start with the default (last layer only), then experiment with more layers if needed.

## Validation

The implementation includes automatic validation:

- ✅ Out-of-range indices are skipped with warnings
- ✅ Duplicate indices are automatically removed
- ✅ Indices are sorted for consistent application order
- ✅ Invalid specifications fall back to default (-1)

## Logging

When MAW is enabled, you'll see:

```
INFO - MAW enabled on encoder layers: [5]
```

Or for multiple layers:

```
INFO - MAW enabled on encoder layers: [0, 3, 5]
```

## Testing

Run the layer selection tests:

```bash
python test_maw_layers.py
```

This validates:
- ✅ Default last layer selection
- ✅ Specific layer selection
- ✅ Negative indexing
- ✅ All layers selection
- ✅ Forward pass correctness

## Migration from Old API

If you were using the old `--maw-layers` parameter:

**Old (deprecated):**
```bash
python tier1_fixed.py --maw-layers 3
```

**New:**
```bash
# To apply to last 3 layers:
python tier1_fixed.py --maw-layer-indices "-1,-2,-3"

# Or to apply to layers 0, 1, 2:
python tier1_fixed.py --maw-layer-indices "0,1,2"
```

## Troubleshooting

### Warning: "MAW layer index X is out of range"

**Cause**: You specified a layer index that doesn't exist in the model.

**Solution**: Check your model's architecture. Most models have 6-12 layers.

```python
# Check number of layers
encoder = HFTextEncoder(config)
num_layers = encoder._get_num_encoder_layers(encoder.model.config)
print(f"Model has {num_layers} encoder layers")
```

### No MAW modules created

**Cause**: All specified indices were out of range or invalid.

**Solution**: Use valid indices or stick with the default `-1`.

## Best Practices

1. **Start Simple**: Use the default (last layer) first
2. **Gradual Expansion**: Add more layers only if performance improves
3. **Monitor Memory**: More layers = more memory usage
4. **Benchmark**: Compare performance with different layer configurations
5. **Task-Specific**: Different tasks may benefit from different layer selections

## Summary

The new layer selection feature provides:

- ✅ **Default**: Last layer only (optimal for most use cases)
- ✅ **Flexible**: Specify any combination of layers
- ✅ **Intuitive**: Negative indexing for top layers
- ✅ **Powerful**: Apply to all layers if needed
- ✅ **Safe**: Automatic validation and error handling

**Default is best for most users!** Experiment with other configurations only if you need specialized behavior.
