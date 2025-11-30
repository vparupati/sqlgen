#!/bin/bash

# Activate venv
source venv/bin/activate

# Model to fine-tune (using 4-bit quantized version for efficiency)
MODEL="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

# Output directory for adapters
ADAPTER_PATH="adapters"

# Training configuration
BATCH_SIZE=2
LORA_LAYERS=8
ITERS=2000
LEARNING_RATE=1e-5

echo "Starting fine-tuning with MLX..."
echo "Model: $MODEL"
echo "Data: data/"
echo "Adapters: $ADAPTER_PATH"

# Run training
python3 -m mlx_lm.lora \
    --model $MODEL \
    --train \
    --data data \
    --batch-size $BATCH_SIZE \
    --num-layers $LORA_LAYERS \
    --iters $ITERS \
    --learning-rate $LEARNING_RATE \
    --adapter-path $ADAPTER_PATH \
    --save-every 100 \
    --steps-per-eval 100 \
    --max-seq-length 2048 \
    --grad-checkpoint

echo "Training complete! Adapters saved to $ADAPTER_PATH"
