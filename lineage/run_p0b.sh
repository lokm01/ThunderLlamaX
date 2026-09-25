# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
#!/bin/zsh
cd ~/tinygrad-metal
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin"
export PYTHONPATH=~/tinygrad-src
env DEV=NV BEAM=0 ITERS=30 ~/tg311/bin/python -u kv_bench.py > ~/tinygrad-metal/stage_kv.log 2>&1
echo "KV_EXIT=$?" >> ~/tinygrad-metal/stage_kv.log
env DEV=NV BEAM=0 NBATCH=20 ITERS=100 ~/tg311/bin/python -u gdn_chain_bench.py > ~/tinygrad-metal/stage_gdn.log 2>&1
echo "GDN_EXIT=$?" >> ~/tinygrad-metal/stage_gdn.log
echo ALL_DONE > ~/tinygrad-metal/stage_p0b.done
