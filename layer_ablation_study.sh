#!/bin/bash
# Layer Ablation Study - Find the sweet spot for MAW layers
# Hypothesis: More MAW layers is NOT always better

echo "🔬 Starting Layer Ablation Study..."
echo "Testing different numbers of MAW layers with 100 samples"
echo "================================================"

# Test 1: 1 Layer (baseline)
echo "Test 1: 1 layer, MAW on all"
python benchmark_evaluation_GRPO.py \
    --dataset MS_MARCO \
    --samples 100 \
    --epochs 10 \
    --num-layers 1 \
    --maw-layers "all" \
    --seed 42

# Test 2: 2 Layers, MAW on all
echo "Test 2: 2 layers, MAW on all"
python benchmark_evaluation_GRPO.py \
    --dataset MS_MARCO \
    --samples 100 \
    --epochs 10 \
    --num-layers 2 \
    --maw-layers "all" \
    --seed 42

# Test 3: 3 Layers, MAW on all
echo "Test 3: 3 layers, MAW on all"
python benchmark_evaluation_GRPO.py \
    --dataset MS_MARCO \
    --samples 100 \
    --epochs 10 \
    --num-layers 3 \
    --maw-layers "all" \
    --seed 42

# Test 4: 4 Layers, MAW on all
echo "Test 4: 4 layers, MAW on all"
python benchmark_evaluation_GRPO.py \
    --dataset MS_MARCO \
    --samples 100 \
    --epochs 10 \
    --num-layers 4 \
    --maw-layers "all" \
    --seed 42

# Test 5: 6 Layers, MAW on first layer only
echo "Test 5: 6 layers, MAW on layer 1 only"
python benchmark_evaluation_GRPO.py \
    --dataset MS_MARCO \
    --samples 100 \
    --epochs 10 \
    --num-layers 6 \
    --maw-layers "1" \
    --seed 42

# Test 6: 6 Layers, MAW on first 2 layers
echo "Test 6: 6 layers, MAW on layers 1,2"
python benchmark_evaluation_GRPO.py \
    --dataset MS_MARCO \
    --samples 100 \
    --epochs 10 \
    --num-layers 6 \
    --maw-layers "1,2" \
    --seed 42

echo "================================================"
echo "✅ Layer Ablation Study Complete!"
echo "Check logs/ directory for detailed results"
