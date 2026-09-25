# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
#!/bin/zsh
# TLX W2 (V-27): the API wrapper sources ops/env.canonical too (the API needs
# TLX_ADMIN_TOKEN for /health debug fields + drift alarm parity, and
# TLX_MODEL_PATH/GGUF for the expected-config fingerprint). System python3 +
# --user packages (pip3 install --user fastapi uvicorn jinja2). Zero GPU imports.
set -u
OPS=~/tinygrad-metal/engine0/ops
[ -f "$OPS/env.canonical" ] && { set -a; source "$OPS/env.canonical"; set +a; }
cd ~/tinygrad-metal/engine0 || exit 1
export PATH="$HOME/Library/Python/3.9/bin:$PATH"
export GGUF="${TLX_MODEL_PATH:-${GGUF:-~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf}}"
exec /usr/bin/python3 ~/tinygrad-metal/engine0/api_server.py
