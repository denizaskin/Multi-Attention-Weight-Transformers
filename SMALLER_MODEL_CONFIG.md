# Smaller Model Configuration - Performance Optimization

## Model Change Summary

### Previous: facebook/contriever
- **Parameters**: ~110 million
- **Hidden size**: 768
- **Layers**: 12
- **Memory footprint**: ~440 MB base model
- **Max batch size (standard)**: 32
- **Max batch size (MAW)**: 8

### Current: sentence-transformers/all-MiniLM-L6-v2
- **Parameters**: ~22 million (5x smaller!)
- **Hidden size**: 384 (half the size)
- **Layers**: 6 (half the layers)
- **Memory footprint**: ~90 MB base model (5x smaller!)
- **Max batch size (standard)**: 64 (2x larger!)
- **Max batch size (MAW)**: 16 (2x larger!)

## New Batch Size Configuration

### Standard Models (DenseLoRA)
- **batch_size**: 64 (increased from 32)
- **eval_batch_size**: 128 (increased from 64)
- **Expected memory**: ~1-2 GB per GPU
- **Training speed**: ~2x faster (larger batches + smaller model)

### MAW Models (MAWLoRA, MAWFullFT)
- **batch_size**: 16 (increased from 8)
- **eval_batch_size**: 32 (increased from 16)
- **Expected memory**: ~5-8 GB per GPU
- **Training speed**: ~2x faster

## Expected Performance Improvements

### 1. Training Speed
- **2-3x faster** due to:
  - Larger batch sizes (2x)
  - Smaller model (20% faster per batch)
  - Better GPU utilization

### 2. Memory Usage
- **5x less base memory** for the model
- Can use **2x larger batches** without OOM
- More memory available for MAW's 5D attention

### 3. Throughput
- **Standard models**: Can process 64 samples/batch vs 32
- **MAW models**: Can process 16 samples/batch vs 8
- **Evaluation**: Can process 128 samples/batch vs 64

### 4. GPU Utilization
- Better utilization across all 4 GPUs
- Less memory fragmentation
- More consistent performance

## Quality Considerations

### Pros
- ✅ Much faster training and evaluation
- ✅ Can train with larger batches
- ✅ Less prone to OOM errors
- ✅ Better GPU utilization
- ✅ Easier to scale to larger datasets

### Cons
- ⚠️ Smaller hidden dimension (384 vs 768) may reduce representation capacity
- ⚠️ Fewer layers (6 vs 12) may affect complex pattern learning
- ⚠️ May need to adjust learning rate or training epochs

### Mitigation
- The MAW mechanism can compensate for smaller base model
- Depth-aware attention provides additional expressiveness
- Can always switch back to Contriever if quality drops

## How to Run

### Quick Test with New Model
```bash
cd /workspace/Multi-Attention-Weight-Transformers
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
timeout 600 python tier1_fixed.py --msmarco --quick-smoke-test --skip-bm25
```

### Expected Results
- DenseLoRA: ~8-10 seconds (vs 15 seconds before)
- MAWLoRA: ~15-18 seconds (vs 26 seconds before)
- MAWFullFT: ~12-15 seconds (vs 23 seconds before)

### Full Training
```bash
cd /workspace/Multi-Attention-Weight-Transformers
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python tier1_fixed.py --msmarco --skip-bm25
```

## Alternative Small Models

If you want to try other small models:

### Option 1: Even Smaller
```python
dense_model: str = "sentence-transformers/all-MiniLM-L12-v2"
# 33M params, 384 hidden, 12 layers
# Good balance between size and quality
```

### Option 2: Tiny Model (Fastest)
```python
dense_model: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
# 17M params, 384 hidden, 3 layers
# Extremely fast but may sacrifice quality
```

### Option 3: Multilingual Small
```python
dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 118M params, but supports 50+ languages
```

## Memory Estimation

With the new model, here's the memory breakdown per GPU:

### Training Phase
```
Base model:           ~90 MB
LoRA adapters:        ~50 MB
Optimizer states:     ~200 MB
Activations (batch):  ~500 MB (batch_size=64)
Gradients:            ~200 MB
Total:                ~1.0 GB (standard models)
```

### MAW Training Phase
```
Base model:           ~90 MB
LoRA adapters:        ~50 MB
MAW modules:          ~100 MB
Optimizer states:     ~400 MB
Activations (batch):  ~2.0 GB (batch_size=16, 5D attention)
Gradients:            ~400 MB
Total:                ~3.0 GB (MAW models)
```

### Evaluation Phase
```
Base model:           ~90 MB
MAW modules:          ~100 MB (if MAW)
Activations (batch):  ~1.0 GB (batch_size=128)
Total:                ~1.2 GB (standard) or ~1.3 GB (MAW)
```

## Monitoring GPU Usage

Check GPU memory during training:
```bash
# In another terminal
watch -n 1 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv'
```

Expected utilization:
- **GPU 0**: 1-3 GB during standard training, 3-5 GB during MAW training
- **GPU 1-3**: Minimal usage during training, utilized during evaluation

## Configuration Summary

```python
# tier1_fixed.py - Current Configuration
dense_model: "sentence-transformers/all-MiniLM-L6-v2"
max_seq_length: 256
batch_size: 64
eval_batch_size: 128
gradient_accumulation_steps: 2

# MAW variants
maw_batch_size: 16
maw_eval_batch_size: 32
maw_chunk_size: 2
```

## Switching Back to Contriever

If you need to switch back:
```python
dense_model: str = "facebook/contriever"
batch_size: int = 32
eval_batch_size: int = 64
# MAW variants
maw_batch_size: 8
maw_eval_batch_size: 16
```

The smaller model should provide **2-3x faster training** with similar or better results (MAW can compensate for the smaller base model)!
