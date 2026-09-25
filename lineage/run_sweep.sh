# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
#!/bin/zsh
cd ~/tinygrad-metal
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin"
export PYTHONPATH=~/tinygrad-src
for L in 2048 8192 32768; do
  env DEV=NV BEAM=1 MEASURE_N=50 ~/tg311/bin/python -u bench_ctx.py $L >> ~/tinygrad-metal/stage_sweep.log 2>&1
done
touch ~/tinygrad-metal/stage_sweep.done
