#!/bin/bash
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 1 Stage B — engine-trace adaptation (after Stage A).
# Usage: bash run_stageb.sh <stage>   (b1 build | b2 train | b3 pack)
# BEST_CKPT env: the Stage-A checkpoint file to adapt from.
set -e
cd /root/ttt
export PATH=/opt/conda/bin:$PATH
export TOKENIZERS_PARALLELISM=false
D=/root/data
W=/root/w
BEST_CKPT=${BEST_CKPT:-/root/runs/stagea/last.pt}
stage=$1

case $stage in
b1)
  python stageb_build.py --traces /root/traces --out $D/sb --weights $W
  ;;
b2)
  python train_stageb.py --ckpt-in $BEST_CKPT --out /root/runs/stageb \
      --data "$D/sb/prose16k:0.45,$D/sb/code8k:0.2,$D/sb/gsm8k:0.35" \
      --r8 $D/sb/r8_anchored.pt --r8-weight 0.30 \
      --steps 400 --S 6 --lr 6e-6 --warmup 25 --batch 4 --anchors 2 \
      --lmax 16384 --ckpt-tokens 200000
  ;;
b3)
  mkdir -p /root/packs
  for c in /root/runs/stageb/ckpt_*.pt; do
    n=$(basename $c .pt)
    [ -d /root/packs/sb_${n}_rtn ] || python pack_trained.py --ckpt $c --out /root/packs/sb_${n}_rtn --mode rtn
  done
  # GPTQ variant for the final stage-B ckpt, calibrated on the ENGINE traces
  python train.py --mode calib --ckpt-in /root/runs/stageb/last.pt --weights $W \
      --data "$D/sb/prose16k,$D/sb/code8k,$D/sb/gsm8k" --calib-tokens 120000 \
      --out /root/runs/stageb --lmax 16384
  python pack_trained.py --ckpt /root/runs/stageb/last.pt --out /root/packs/sb_final_gptq \
      --mode gptq --calib /root/runs/stageb/calib
  ls -la /root/packs/ | tail -20
  ;;
esac
echo "STAGE $stage DONE"
