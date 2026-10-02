#!/bin/bash
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# v3 per-arm pack stage (disk-adapted: pack arm -> cleanup -> next arm)
set -e
cd /root/ttt
export PATH=/opt/conda/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=$1
R=/root/runs/$TAG
mkdir -p /root/packs3
echo "== RTN curve packs =="
for c in $R/best.pt $R/ckpt_*.pt; do
  [ -f "$c" ] || continue
  n=$(basename $c .pt)
  [ -d /root/packs3/${TAG}_${n}_rtn ] || python pack_trained.py --ckpt $c --out /root/packs3/${TAG}_${n}_rtn --mode rtn
done
echo "== GPTQ calib + pack (best) =="
python train.py --mode calib --ckpt-in $R/best.pt --weights /root/w \
    --data "/root/data/sb/prose16k,/root/data/sb/code8k,/root/data/sb/gsm8k" --calib-tokens 120000 \
    --out $R --lmax 16384
python pack_trained.py --ckpt $R/best.pt --out /root/packs3/${TAG}_best_gptq --mode gptq --calib $R/calib
echo "== cleanup arm ckpts (packs + calib + hist kept) =="
rm -f $R/ckpt_*.pt $R/best.pt $R/last.pt
echo "B3-ARM $TAG DONE"
ls -la /root/packs3/ | tail -15
df -h / | tail -1
