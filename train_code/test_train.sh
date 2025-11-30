#!/bin/bash
source venv/bin/activate
MODEL="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
python3 -m mlx_lm.lora \
    --model $MODEL \
    --train \
    --data data \
    --batch-size 2 \
    --num-layers 4 \
    --iters 10 \
    --adapter-path adapters_test \
    --save-every 10 \
    --steps-per-eval 10
