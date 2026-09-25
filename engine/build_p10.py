# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P10 builder: the t32 attention ship set.
  pfc16t_s13         — S13/NHP3 layout combine (grid 24 x 256thr; pfc16 epilogue)
  pfa32ctl_s13_100k  — t32 + PFA_LC self-resetting last-CTA combine (10 args)
  pfa32ctl_s13_2k    — 2k-shape build for the standalone corr probe
Same docker nvcc + symbol-check pattern as build_p6.py (-Xptxas -v readout).
"""
import sys, os
sys.path.insert(0, "~/tinygrad-metal/engine0")
from build_p6 import build

C100 = ["-DCTXK=100352"]
C2K  = ["-DCTXK=2048"]
T32  = ["-DS=13", "-DPFA_T32=1"]
LC   = ["-DMINB=2", "-DPFA_LC=1"]
TARGETS = [
  ("pfa32c_t32_s26_100k", "pf_attn32c.cu", C100 + ["-DS=26", "-DPFA_T32=1", "-DMINB=2"], "pfa32ct"),
  ("pfc16t_s26",          "pf_attn32c.cu", C100 + ["-DS=26", "-DPFA_T32=1", "-DPFC_T32=1"], "pfc16t"),
  ("pfc16t_s13",        "pf_attn32c.cu", C100 + T32 + ["-DPFC_T32=1"],          "pfc16t"),
  ("pfa32ctl_s13_100k", "pf_attn32c.cu", C100 + T32 + LC,                       "pfa32ctl"),
  ("pfa32ctl_s13_2k",   "pf_attn32c.cu", C2K  + T32 + LC,                       "pfa32ctl"),
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
  print(f"[build_p10 done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
