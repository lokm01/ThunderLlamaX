# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P12 builder: the 32-row M32-merge twins (bit-identical per-row math, one
launch per block instead of 2x16 halves).
  pfk_pre32_100k — pf_kpre32.cu (-DCTXK=100352), TROWS=32 kv8 append + qw
  pfs32          — pf_scan32.cu (no defines), TROWS=32 GDN scan
Same docker nvcc + symbol-check pattern as build_p6.build.
"""
import sys, os
sys.path.insert(0, "~/tinygrad-metal/engine0")
from build_p6 import build

TARGETS = [
  ("pfk_pre32_100k", "pf_kpre32.cu", ["-DCTXK=100352"], "pfk_pre32"),
  ("pfs32",          "pf_scan32.cu", [],                 "pfs32"),
]

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  only = sys.argv[1:] if len(sys.argv) > 1 else None
  n_ok = n_fail = 0
  for name, src, extra, kname in TARGETS:
    if only and not any(o in name for o in only): continue
    if build(env, name, src, extra, kname): n_ok += 1
    else: n_fail += 1
  print(f"[build_p12 done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
