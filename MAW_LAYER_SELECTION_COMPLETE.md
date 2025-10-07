# MAW Layer Selection - Implementation Complete ✅

## Summary of Changes

The MAW (Multi-Attention-Weight) mechanism now supports **fine-grained layer selection**, allowing you to apply it to specific transformer encoder layers instead of applying it sequentially after the encoder.

## Key Features

### ✅ Default: Last Layer Only
By default, MAW is applied **ONLY to the last layer** of the encoder, which is the most common and effective use case.

### ✅ Flexible Layer Selection
You can specify:
- Single layer: `--maw-layer-indices -1`
- Multiple layers: `--maw-layer-indices 0,5,11`
- Last N layers: `--maw-layer-indices=-1,-2,-3`
- All layers: `--maw-layer-indices all`

### ✅ Intelligent Resolution
- Negative indexing (Python-style): `-1` = last layer, `-2` = second-to-last, etc.
- Automatic validation: Out-of-range indices are skipped with warnings
- Duplicate removal: Each layer is processed only once

## Implementation Details

### Code Changes

**1. BenchmarkConfig (tier1_fixed.py, line ~107)**
```python
# OLD
maw_layers: int = 1  # Number of sequential MAW layers

# NEW
maw_layer_indices: List[int] = field(default_factory=lambda: [-1])  # Which encoder layers
```

**2. HFTextEncoder Initialization (tier1_fixed.py, line ~863)**
```python
# Creates MAW modules for specific encoder layers
self.maw_modules = nn.ModuleDict({
    str(idx): TokenLevelMAW(...)
    for idx in self.maw_layer_indices
})
```

**3. Forward Pass (tier1_fixed.py, line ~927)**
```python
# Gets hidden states from all layers
outputs = self.model(**tokenized, output_hidden_states=True)

# Applies MAW to specified layers
for layer_idx in self.maw_layer_indices:
    modified_states[layer_idx + 1] = maw_module(...)
```

**4. CLI Arguments (tier1_fixed.py, line ~1891)**
```python
parser.add_argument(
    "--maw-layer-indices", 
    type=str, 
    default="-1",
    help="Comma-separated layer indices..."
)
```

### Architecture

```
┌──────────────────────────────────────────────┐
│         Transformer Encoder                  │
│                                              │
│  Embeddings                                  │
│      ↓                                       │
│  Layer 0  ──→ [MAW if 0 in indices]         │
│      ↓                                       │
│  Layer 1  ──→ [MAW if 1 in indices]         │
│      ↓                                       │
│   ...                                        │
│      ↓                                       │
│  Layer N-1 ──→ [MAW if N-1 in indices]      │
│      ↓                                       │
│  Mean Pool                                   │
│      ↓                                       │
│  L2 Normalize                                │
│      ↓                                       │
│  Output                                      │
└──────────────────────────────────────────────┘
```

## Usage Examples

### Command Line

```bash
# Default: Last layer only
python tier1_fixed.py --quick-smoke-test --msmarco --use-maw

# Specific layers
python tier1_fixed.py --msmarco --use-maw --maw-layer-indices 0,5,11

# Last 2 layers (note: use = for negative indices)
python tier1_fixed.py --msmarco --use-maw --maw-layer-indices=-1,-2

# All layers
python tier1_fixed.py --msmarco --use-maw --maw-layer-indices all
```

### Programmatic

```python
from tier1_fixed import BenchmarkConfig, HFTextEncoder

# Last layer only (default)
config = BenchmarkConfig(
    use_maw=True,
    maw_layer_indices=[-1],
)

# Multiple specific layers
config = BenchmarkConfig(
    use_maw=True,
    maw_layer_indices=[0, 5, 11],
)

# All layers
config = BenchmarkConfig(
    use_maw=True,
    maw_layer_indices=list(range(100)),  # Will be clamped
)

encoder = HFTextEncoder(config)
```

## Testing

### Unit Tests
```bash
python test_maw_layers.py
```

Tests:
- ✅ Default last layer selection
- ✅ Specific layer selection
- ✅ Negative indexing
- ✅ All layers selection
- ✅ Forward pass correctness

### CLI Tests
```bash
python test_cli_parsing.py
```

Tests:
- ✅ Argument parsing
- ✅ Default values
- ✅ Multiple formats
- ✅ Config building

### Integration Tests
```bash
python test_maw_integration.py
```

Tests:
- ✅ Full pipeline integration
- ✅ Encoder compatibility
- ✅ Gradient flow

## Performance Comparison

| Configuration | Speed | Memory | Use Case |
|--------------|-------|--------|----------|
| **Last layer only** (default) | ✅ Fast | ✅ Low | Most tasks |
| 2-3 specific layers | ⚠️ Medium | ⚠️ Medium | Fine-tuning |
| All layers | ❌ Slow | ❌ High | Research |

**Recommendation**: Start with default (last layer), experiment if needed.

## Migration Guide

### From Old API

**Before:**
```python
config = BenchmarkConfig(
    use_maw=True,
    maw_layers=3,  # DEPRECATED: Applied sequentially after encoder
)
```

**After:**
```python
# To apply to last 3 encoder layers:
config = BenchmarkConfig(
    use_maw=True,
    maw_layer_indices=[-1, -2, -3],
)

# Or to apply to first 3 encoder layers:
config = BenchmarkConfig(
    use_maw=True,
    maw_layer_indices=[0, 1, 2],
)
```

## Validation & Logging

### Automatic Validation
- Out-of-range indices → Warning + skip
- Duplicate indices → Automatic removal
- Invalid formats → Fall back to default `[-1]`

### Logging Output
```
INFO - MAW enabled on encoder layers: [5]
```

For multiple layers:
```
INFO - MAW enabled on encoder layers: [0, 3, 5]
```

## Files Created/Modified

| File | Status | Purpose |
|------|--------|---------|
| `tier1_fixed.py` | ✅ Modified | Core implementation |
| `test_maw_layers.py` | ✅ Created | Layer selection tests |
| `test_cli_parsing.py` | ✅ Created | CLI parsing tests |
| `MAW_LAYER_SELECTION.md` | ✅ Created | User documentation |
| `MAW_LAYER_SELECTION_COMPLETE.md` | ✅ Created | This summary |

## Technical Notes

### Layer Index Mapping
- `hidden_states[0]` = Input embeddings
- `hidden_states[i+1]` = Output of encoder layer `i`
- MAW is applied to `hidden_states[layer_idx + 1]`

### Hidden States Access
- Requires `output_hidden_states=True` in model forward pass
- Returns tuple of (num_layers + 1) tensors
- Slight memory overhead (minimal for most models)

### Gradient Flow
- ✅ Gradients flow correctly through MAW modules
- ✅ GRPO policy/value networks update properly
- ✅ Compatible with LoRA and full fine-tuning

## Troubleshooting

### Issue: "MAW layer index X is out of range"
**Solution**: Check your model's layer count:
```python
encoder = HFTextEncoder(config)
num_layers = encoder._get_num_encoder_layers(encoder.model.config)
print(f"Model has {num_layers} layers")
```

### Issue: Negative indices not working in CLI
**Solution**: Use `=` syntax:
```bash
# Instead of: --maw-layer-indices -1,-2
# Use:
python tier1_fixed.py --maw-layer-indices=-1,-2
```

### Issue: No MAW modules created
**Solution**: Check that `use_maw=True` and at least one valid index is specified.

## Best Practices

1. ✅ **Start with default**: Last layer works well for most tasks
2. ✅ **Measure performance**: Compare different configurations
3. ✅ **Monitor memory**: More layers = more memory
4. ✅ **Use negative indexing**: More portable across models
5. ✅ **Document your choice**: Note which layers you use in experiments

## What's Next?

### Possible Extensions
- [ ] Layer-specific MAW hyperparameters (different depth_dim per layer)
- [ ] Automatic layer selection based on task
- [ ] Visualization of layer-wise attention patterns
- [ ] Pruning: automatically remove ineffective layers

### Research Questions
- Which layers benefit most from MAW?
- Does early-layer vs late-layer MAW differ?
- Can we learn optimal layer selection?

## Conclusion

✅ **Implementation Complete**

The MAW mechanism now supports flexible layer selection with:
- Default last-layer behavior (optimal for most users)
- Full control over which layers to enhance
- Intelligent validation and error handling
- Comprehensive testing and documentation

**Ready for production use!** 🎉

---

## Quick Reference

```bash
# Default (recommended)
python tier1_fixed.py --use-maw

# Specific layers
python tier1_fixed.py --use-maw --maw-layer-indices 0,5,11

# Last N layers
python tier1_fixed.py --use-maw --maw-layer-indices=-1,-2,-3

# All layers
python tier1_fixed.py --use-maw --maw-layer-indices all
```

**Remember**: Default is last layer only, which works great for most tasks!
