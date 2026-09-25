# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
#!/bin/zsh
cd ~/tinygrad-metal
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin"
export PYTHONPATH=~/tinygrad-src
echo "=== SKIP=OFF ===" > ~/tinygrad-metal/stage_vram.log
env DEV=NV MTP_SKIP_DUPHEAD=0 ~/tg311/bin/python -u vram_check.py >> ~/tinygrad-metal/stage_vram.log 2>&1
echo "=== SKIP=ON ===" >> ~/tinygrad-metal/stage_vram.log
env DEV=NV MTP_SKIP_DUPHEAD=1 ~/tg311/bin/python -u vram_check.py >> ~/tinygrad-metal/stage_vram.log 2>&1
touch ~/tinygrad-metal/stage_p0b.done
