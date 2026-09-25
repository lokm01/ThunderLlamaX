# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
#!/bin/bash
# Fires when the current mtp_v3 (8k run) exits: 100k baseline then 100k MTP bench.
export DOCKER_HOST=unix://<colima-socket>
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
cd ~/tinygrad-metal
echo "[chain] waiting for 8k run to finish..." >> ~/chain100k.log
while pgrep -f "mtp_v3.py" > /dev/null; do sleep 60; done
echo "[chain] 8k done at $(date). Generating 100k baseline..." >> ~/chain100k.log
DEV=NV BEAM=1 JIT=1 MTP_MAXCTX=100352 MTP_PROMPT_FILE=~/prompt100k.txt GEN_N=60 \
  GEN_OUT=~/tinygrad-metal/spec_base_100k.json \
  ~/tg311/bin/python -u gen_basex.py > ~/gen_100k.log 2>&1
echo "[chain] baseline done at $(date). Launching 100k MTP..." >> ~/chain100k.log
DEV=NV BEAM=1 MTP_A3_OVERRIDE=$HOME/tinygrad-metal/a3b/override.json MTP_A3C_OFF=1 \
  MTP_T3_LAZY=1 MTP_SEQ_ATTN=1 MTP_STEP_STATES=1 MTP_PROBE_RO=1 \
  MTP_CHUNKED_PREFILL=1 MTP_CHUNK_EAGERDRAFT=1 \
  MTP_MAXCTX=100352 MTP_PROMPT_FILE=~/prompt100k.txt \
  MTP_BASE_JSON=~/tinygrad-metal/spec_base_100k.json NTOK=60 \
  ~/tg311/bin/python -u mtp_v3.py > ~/mtp_100k.log 2>&1
echo "[chain] 100k MTP done at $(date)." >> ~/chain100k.log
