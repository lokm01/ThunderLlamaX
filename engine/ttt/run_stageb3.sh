#!/bin/bash
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 2 — Stage B v3 (FIRST-PARTY bf16 init, LR 2-3e-6 sweep).
# Usage: bash run_stageb3.sh <stage>   (b1 build | b2 train | b3 pack)
# v3 = v2 recipe EXACTLY, changed ONLY: init (pristine bf16 /root/w_init) + LR.
set -e
cd /root/ttt
export PATH=/opt/conda/bin:$PATH
export TOKENIZERS_PARALLELISM=false
D=/root/data
W=/root/w
LRs=${LRs:-"2e-6 3e-6"}
STEPS=${STEPS:-1600}
stage=$1

case $stage in
b1)
  # corpus shards (the protective mix) — v1 builder, --skip-r8 (v3 anchors replace r8)
  python stageb_build.py --traces /root/traces --out $D/sb --weights $W --skip-r8
  # the v2 anchored sets (train + canary) — unchanged builder
  python stageb2_build.py --anchors /root/anchors --out $D/sb2
  ;;
b2)
  for lr in $LRs; do
    tag=$(echo $lr | tr -d '-')
    python train_stageb3.py --init-dir $W --out /root/runs/sb3_lr${tag} \
        --data "$D/sb/prose16k:0.45,$D/sb/code8k:0.22,$D/sb/gsm8k:0.33" \
        --anchors $D/sb2/anchors_train.pt --canary $D/sb2/anchors_canary.pt \
        --anchor-weight 0.55 \
        --steps $STEPS --S 6 --lr $lr --warmup 25 --batch 8 --corpus-batch 4 \
        --lmax 16384 --eval-every 50 --eval-n 48 --patience 4 --curve-every 100
  done
  ;;
b3)
  mkdir -p /root/packs3
  for lr in $LRs; do
    tag=$(echo $lr | tr -d '-')
    R=/root/runs/sb3_lr${tag}
    [ -f $R/best.pt ] || continue
    # RTN curve packs (cheap, no calib): best + the periodic curve ckpts
    for c in $R/best.pt $R/ckpt_*.pt; do
      [ -f "$c" ] || continue
      n=$(basename $c .pt)
      [ -d /root/packs3/sb3_lr${tag}_${n}_rtn ] || \
        python pack_trained.py --ckpt $c --out /root/packs3/sb3_lr${tag}_${n}_rtn --mode rtn
    done
    # GPTQ for the best-canary ckpt (the ship candidate), calibrated on the
    # ENGINE traces (v2 recipe verbatim)
    python train.py --mode calib --ckpt-in $R/best.pt --weights $W \
        --data "$D/sb/prose16k,$D/sb/code8k,$D/sb/gsm8k" --calib-tokens 120000 \
        --out $R --lmax 16384
    python pack_trained.py --ckpt $R/best.pt --out /root/packs3/sb3_lr${tag}_best_gptq \
        --mode gptq --calib $R/calib
  done
  ls -la /root/packs3/ | tail -20
  ;;
esac
echo "STAGE $stage DONE"
