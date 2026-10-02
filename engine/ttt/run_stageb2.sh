#!/bin/bash
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 2 — anchor-scale Stage B v2 (after the anchor dump).
# Usage: bash run_stageb2.sh <stage>   (b1 build | b2 train | b3 pack)
# BEST_CKPT env: ckpt to adapt from (default the Stage-A winner ckpt_6000).
set -e
cd /root/ttt
export PATH=/opt/conda/bin:$PATH
export TOKENIZERS_PARALLELISM=false
D=/root/data
W=/root/w
BEST_CKPT=${BEST_CKPT:-/root/ckpt_6000.pt}
stage=$1

case $stage in
b1)
  # corpus shards (the protective mix) — v1 builder, unchanged semantics
  python stageb_build.py --traces /root/traces --out $D/sb --weights $W
  # the v2 anchored sets (train + canary)
  python stageb2_build.py --anchors /root/anchors --out $D/sb2
  ;;
b2)
  python train_stageb2.py --ckpt-in $BEST_CKPT --out /root/runs/stageb2 \
      --data "$D/sb/prose16k:0.45,$D/sb/code8k:0.22,$D/sb/gsm8k:0.33" \
      --anchors $D/sb2/anchors_train.pt --canary $D/sb2/anchors_canary.pt \
      --anchor-weight 0.55 \
      --steps 800 --S 6 --lr 6e-6 --warmup 25 --batch 8 --corpus-batch 4 \
      --lmax 16384 --eval-every 50 --eval-n 48 --patience 4
  ;;
b3)
  mkdir -p /root/packs2
  # pack BOTH the best-canary ckpt (the ship candidate) and last (the curve end)
  for c in /root/runs/stageb2/best.pt /root/runs/stageb2/last.pt; do
    [ -f "$c" ] || continue
    n=$(basename $c .pt)
    [ -d /root/packs2/sb2_${n}_rtn ] || python pack_trained.py --ckpt $c --out /root/packs2/sb2_${n}_rtn --mode rtn
  done
  # GPTQ for the best-canary ckpt, calibrated on the ENGINE traces (v1 recipe)
  python train.py --mode calib --ckpt-in /root/runs/stageb2/best.pt --weights $W \
      --data "$D/sb/prose16k,$D/sb/code8k,$D/sb/gsm8k" --calib-tokens 120000 \
      --out /root/runs/stageb2 --lmax 16384
  python pack_trained.py --ckpt /root/runs/stageb2/best.pt --out /root/packs2/sb2_best_gptq \
      --mode gptq --calib /root/runs/stageb2/calib
  ls -la /root/packs2/ | tail -10
  ;;
esac
echo "STAGE $stage DONE"
