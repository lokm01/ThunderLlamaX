# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
#!/bin/zsh
# 100k MTP with JIT=1 probe (NOBIND) — resumes from ckpt@94208 (~3600 prefill tokens left)
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin"
export DOCKER_HOST=unix://<colima-socket>
cd ~/tinygrad-metal
sudo -n rm -f $TMPDIR/nv_usb4.lock 2>/dev/null
env DEV=NV BEAM=1 MTP_K3=3 MTP_BEAM_MAXGS=8192 MTP_EC_DRAIN=2 \
  MTP_A3_OVERRIDE=$HOME/tinygrad-metal/a3b/override.json MTP_A3C_OFF=1 \
  MTP_T3_LAZY=1 MTP_SEQ_ATTN=1 MTP_STEP_STATES=1 MTP_PROBE_RO=1 \
  MTP_EMB_GATHER=1 MTP_HEAD16_DIRECT=1 MTP_BLKGROUP=8 \
  MTP_CHUNKED_PREFILL=1 MTP_CHUNK_EAGERDRAFT=1 \
  MTP_GRAPH_NOBIND=1 MTP_CKPT=1 MTP_DRAFT_SLICE=40960 MTP_EARLY_CAPTURE=1 MTP_EC_SS=0 MTP_COMMIT_MODE=1 MTP_COMMIT_JIT=1 MTP_SELFIN=1 MTP_DRAFT_JIT1=1 MTP_HM_DEVSTORE=1 MTP_SEL_JIT1=1 MTP_HSEED_SEL=1 MTP_A3E_T3=1 MTP_PROBE_AM=1 MTP_DTAIL_JIT=1 MTP_SEQ_ATTN_BATCH=1 MTP_A3E_T3=1 MTP_SKV=1 \
  MTP_MAXCTX=100352 MTP_PROMPT_FILE=~/prompt100k.txt \
  MTP_BASE_JSON=~/tinygrad-metal/spec_base_100k.json NTOK=60 \
  ~/tg311/bin/python -u mtp_v3.py > ~/mtp_100k_k3d2.log 2>&1
echo "EXIT=$?" >> ~/mtp_100k_k3d2.log
