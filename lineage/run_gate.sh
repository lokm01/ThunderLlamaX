# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
#!/bin/zsh
cd ~/tinygrad-metal
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin"
export PYTHONPATH=~/tinygrad-src
env DEV=NV BEAM=0 JIT=2 MTP_K=2 ~/tg311/bin/python -u mtp_spec.py spec 20 > ~/tinygrad-metal/stage_gate.log 2>&1
echo "GATE_EXIT=$?" >> ~/tinygrad-metal/stage_gate.log
touch ~/tinygrad-metal/stage_gate.done
