#!/bin/bash
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 1 Stage A — the full-run orchestration (rental H100 NVL).
# Usage: bash run_stagea.sh <stage>   (s1 prompts | s2 gen | s3 dump | s4 val | s5 train | s6 pack)
set -e
cd /root/ttt
export PATH=/opt/conda/bin:$PATH
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=/root/data
W=/root/w
stage=$1

case $stage in
s1)
  mkdir -p $D
  python longcorpus.py --out-tf $D/books_tf.jsonl --out-sg $D/books_sg.jsonl \
      --out-prose $D/prose_short.jsonl --n-tf 170 --n-sg 80 --n-prose 800 \
      --min-chars 240000 --sg-prompt-tok 36000 --sg-max-new 2560
  python gen_prompts.py --out $D/prompts.jsonl --n-reason 5000 --n-code 2000 \
      --n-chat 2500 --n-prose 0 --n-docqa 1500 --n-long 0
  python - <<'EOF'
import json
rows = [json.loads(l) for l in open('/root/data/prose_short.jsonl')]
with open('/root/data/prompts_all.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r)+'\n')
EOF
  wc -l $D/*.jsonl
  ;;
s2)
  # short/mid generation (vLLM, 8k ctx)
  python teacher_gen.py --prompts $D/prompts.jsonl --out $D/gen.jsonl \
      --max-model-len 8192 --batch 512 --max-num-seqs 512
  # long self-gen (49k ctx, small batch)
  python teacher_gen.py --prompts $D/books_sg.jsonl --out $D/gen_sg.jsonl \
      --max-model-len 49152 --batch 16 --max-num-seqs 16
  ;;
s3)
  mkdir -p $D
  dump() { out=$1; shift; if [ -f $out/meta.json ]; then echo "SKIP $out (done)"; else python dump_features.py --out $out "$@"; fi; }
  dump $D/sh_books_long --gen $D/books_tf.jsonl --field text --lmax 32768 --batch 1 \
      --split-windows --abs-cap 108000 --max-win 3 --anchors 4 --tail-anchor 1 \
      --batch-hint 2 --holdout 0
  dump $D/sh_sg_long --gen $D/gen_sg.jsonl --filter-cls sg_long --lmax 32768 \
      --batch 1 --anchors 4 --tail-anchor 1 --batch-hint 2 --holdout 0
  dump $D/sh_books_mid --gen $D/books_tf.jsonl --field text --lmax 8192 --batch 1 \
      --split-windows --abs-cap 104000 --max-win 3 --win-stride 4 --anchors 2 \
      --batch-hint 4 --holdout 0
  dump $D/sh_gsm --gen $D/gen.jsonl --filter-cls gsm8k,metamath --lmax 2048 --batch 8 --holdout 0
  dump $D/sh_chat --gen $D/gen.jsonl --filter-cls ultrachat --lmax 4096 --batch 8 --holdout 0
  dump $D/sh_code --gen $D/gen.jsonl --filter-cls mbpp,code --lmax 4096 --batch 8 --holdout 0
  dump $D/sh_docqa --gen $D/gen.jsonl --filter-cls docqa --lmax 1024 --batch 8 --holdout 0
  dump $D/sh_prose_s --gen $D/prompts_all.jsonl --filter-cls prose --lmax 2048 --batch 8 --holdout 0
  df -h /root | tail -1
  ;;
s4)
  python train.py --mode overfit-synthetic --S 6 --overfit-steps 300
  python train.py --mode overfit-long --S 6 --overfit-steps 300
  python train.py --mode overfit-real --S 6 --overfit-steps 300 --weights $W \
      --data $D/sh_gsm --feedback slice
  ;;
s5)
  python train.py --mode train --weights $W --out /root/runs/stagea \
      --data "$D/sh_books_long:0.22,$D/sh_sg_long:0.11,$D/sh_books_mid:0.14,$D/sh_gsm:0.23,$D/sh_chat:0.10,$D/sh_code:0.10,$D/sh_docqa:0.05,$D/sh_prose_s:0.05" \
      --steps 6000 --S 6 --lr 5e-5 --warmup 300 --lmax 32768 --batch 8 \
      --feedback slice --ckpt-tokens 25000000 --calib-tokens 300000 --log 10
  ;;
s6)
  # RTN packs for every ckpt (fast) + GPTQ for the final
  mkdir -p /root/packs
  for c in /root/runs/stagea/ckpt_*.pt; do
    n=$(basename $c .pt)
    [ -d /root/packs/${n}_rtn ] || python pack_trained.py --ckpt $c --out /root/packs/${n}_rtn --mode rtn
  done
  python pack_trained.py --ckpt /root/runs/stagea/last.pt --out /root/packs/final_gptq \
      --mode gptq --calib /root/runs/stagea/calib
  ls -la /root/packs/
  ;;
esac
echo "STAGE $stage DONE"
