# Distributed Training Hang Issue

## Symptom
When running with `torch.distributed.run --nproc_per_node=4`, the script hangs after:
```
2025-10-07 01:06:54,297 - INFO - Building BM25 index for msmarco with 9271 documents...
2025-10-07 01:06:54,593 - INFO - BM25 index built for msmarco
```

No further output appears, and the process hangs indefinitely.

## Diagnosis
The issue is likely related to distributed synchronization barriers. Possible causes:
1. **HuggingFace model loading**: Multiple ranks trying to download/cache models simultaneously
2. **BM25 evaluation**: Only rank 0 builds/evaluates BM25, but no proper synchronization before moving to dense evaluation
3. **Corpus embedding caching**: Race conditions in memmap file creation/access

## Attempted Fixes
1. Added `dist.barrier()` after base encoder initialization
2. Added `dist.barrier()` after BM25 evaluation
3. Existing barriers in `_prepare_corpus_embeddings` already present

## Workaround
**For smoke tests, use single-process mode:**
```bash
# Works reliably
python tier1_fixed.py --quick-smoke-test --msmarco

# Hangs with multi-process
python run_smoke_test.py --nproc 4 --extra-args --msmarco
```

## Recommended Solution
For production use with multi-GPU:
1. Use DDP (Distributed Data Parallel) wrapper around the encoder models
2. Ensure all file I/O operations (model loading, cache creation) are properly synchronized
3. Consider using shared filesystem for model cache to avoid download conflicts
4. Add verbose logging on all ranks (not just rank 0) to diagnose exact hang location

## Alternative Approach
Instead of `torchrun` with multiple processes, use:
- **DataParallel** for single-node multi-GPU (simpler, no process spawning)
- **Single GPU** for smoke tests (fastest for debugging)
- **Full DDP setup** only for large-scale production runs

The smoke test is designed to be quick (~5 min), so single-GPU execution is acceptable.
