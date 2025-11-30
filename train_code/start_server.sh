#!/bin/bash
source venv/bin/activate
python3 -m mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --adapter-path adapters --port 8081
