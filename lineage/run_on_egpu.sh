# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
#!/bin/bash
# Run local LLMs on the RTX 4070 Ti Super eGPU via tinygrad's NV backend (TinyGPU).
# Usage: ./run_on_egpu.sh [size]
#   size: 1B (default) | 8B | 70B
eval "$(/opt/homebrew/bin/brew shellenv zsh)"
source ~/tg311/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=~/tinygrad-src
cd ~/tinygrad-src/examples
SIZE="${1:-1B}"
DEV=NV python llama3.py --size "$SIZE" --benchmark
