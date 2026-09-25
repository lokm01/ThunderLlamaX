# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P15 builder: pfs64 -- pf_scan64.cu (TROWS=64 twin of pfs32, per-row op
order verbatim; the P12 TROWS-merge law). Same docker nvcc + symbol-check
pattern as build_p6.build."""
import sys, os
sys.path.insert(0, "~/tinygrad-metal/engine0")
from build_p6 import build

TARGETS = [("pfs64", "pf_scan64.cu", [], "pfs64")]

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  n_ok = n_fail = 0
  for name, src, extra, kname in TARGETS:
    if build(env, name, src, extra, kname): n_ok += 1
    else: n_fail += 1
  print(f"[build_p15 done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
