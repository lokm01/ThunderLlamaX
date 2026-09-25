# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
#!/bin/zsh
# T2 W4A8 gates: PF_W4A8=1 (+PF_W4A8_MB) — the Tier-2 battery per the P7E7
# convention: 2k/8k gates + 100k rebuild + Tier-1 decode-untouched + kill-switch.
# Args: $1 = MB budget (default 2000), $2 = which set (2k|8k|100k|t1|ks2k)
MB=${1:-2000}
SET=${2:-2k}
cd ~/tinygrad-metal/engine0
BASEENV="PATH=~/.local/bin:/opt/homebrew/bin:/usr/bin:/bin \
  DOCKER_HOST=unix://~/.colima/default/docker.sock \
  DEV=NV SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1 PVH=1 HM=1 \
  MTP_KERNARGS_MB=256 PF_PREFILL=1 PF_GEMM3=1 PF_ATTN32=1 \
  NV_SMEM_CFG_AUTO=1 NV_SMEM_CFG_AUTO_NAMES=pfg,pfa32c PF_N32=1 PF_PRE32=1 PF_SCAN32=1 PF_M64=1 \
  PF_M128=1 PF_DR7=1 PF_ATTNW=1 PF_SCANC=1 PF_SCANC_N2=1 PF_M64QKV=1 PF_ABW=1 \
  PF_RING4=1 PF_QKV1=1 PF_P5=1 PF_OP64=1 PC_ENABLED=0"
W4="PF_W4A8=1 PF_W4A8_MB=$MB"
case $SET in
  2k)   env $=BASEENV $=W4 LOOKUP_K=0 PF_GATE=1 PF_GATE_TIEOK=1 PF_TRUNC=2048 NTOK=60 ~/tg311/bin/python -u test_w100k.py > ~/t2w4_g2k.log 2>&1 ;;
  8k)   env $=BASEENV $=W4 LOOKUP_K=0 PF_GATE=1 PF_GATE_TIEOK=1 PF_TRUNC=8192 NTOK=60 ~/tg311/bin/python -u test_w100k.py > ~/t2w4_g8k.log 2>&1 ;;
  100k) env $=BASEENV $=W4 LOOKUP_K=0 PF_GATE100K=1 NTOK=60 ~/tg311/bin/python -u test_w100k.py > ~/t2w4_r100k.log 2>&1 ;;
  t1)   env $=BASEENV $=W4 LOOKUP_K=8 NTOK=60 ~/tg311/bin/python -u test_w100k.py > ~/t2w4_t1.log 2>&1 ;;
  ks2k) env $=BASEENV LOOKUP_K=0 PF_GATE=1 PF_GATE_TIEOK=1 PF_TRUNC=2048 NTOK=60 ~/tg311/bin/python -u test_w100k.py > ~/t2w4_ks2k.log 2>&1 ;;
esac
echo "$SET exit $?"
