#!/bin/zsh
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 2 — anchor-scale dump RUNNER (relaunch across faults).
# The harness is idempotent (sessions with manifest.json are skipped), so any
# process-death/box-reset class just means relaunch. p0_run.sh takes the GPU
# lock. ALL DONE / exit 0 = stop relaunching.
set -u
LOG="$HOME/anchor_dump.log"
: > "$LOG"
for i in $(seq 1 40); do
  print -- "=== runner attempt $i $(date) ===" >> "$LOG"
  "$HOME/p0_run.sh" env PF_W4A8=0 TLX_T1_MODE=0 TLX_EAGLE_K=4 \
      TLX_EAGLE_PROSE_TRIG=0 PC_ENABLED=0 ANCHOR_NCYC=250 \
      /Users/lokm/tg311/bin/python -u "$HOME/anchor_scale_dump.py" >> "$LOG" 2>&1
  rc=$?
  print -- "=== runner attempt $i exit $rc $(date) ===" >> "$LOG"
  if [ $rc -eq 0 ]; then
    print -- "RUNNER_DONE rc=0" >> "$LOG"
    exit 0
  fi
  sleep 90   # reset/boot guard before the next boot (~4-6 min engine load)
done
print -- "RUNNER_EXHAUSTED" >> "$LOG"
exit 1
