# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
#!/bin/bash
# Run GPT-2 on the built-in Apple M2 GPU via tinygrad's METAL backend.
# Usage: ./run_gpt2.sh "your prompt" [token_count]
eval "$(/opt/homebrew/bin/brew shellenv zsh)"
source ~/tg311/bin/activate
export PYTHONPATH=~/tinygrad-src
cd ~/tinygrad-src/examples
PROMPT="${1:-Once upon a time, in a land far away,}"
COUNT="${2:-25}"
python gpt2.py --model_size gpt2 --prompt "$PROMPT" --count "$COUNT" --temperature 0.8
