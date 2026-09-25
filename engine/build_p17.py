# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P17 builder: the wide-M attention shape sweep (pf_attnw.cu).
Shapes (all TILE=32 K-staged, per-row math VERBATIM from the shipped t32):
  pfaw_w32_s13/s26_100k  — ROWS=32 HRP=2 NW=16 (512thr, RMAX=64, grid 4*S*3)
  pfaw_w64_s13_100k      — ROWS=64 HRP=2 NW=16 (RMAX=128, smem = 48KB EXACT)
  pfaw_w64q_s13_100k     — ROWS=64 HRP=2 NW=32 (1024thr; 64-reg cap check)
  pfaw_w64h_s13_100k     — ROWS=64 HRP=1 NW=16 (RMAX=64, grid 4*S*6 = 312)
  combines pfcw* (same epilogue/op-order as pfc16t; t < ROWS loop)
  + 2k-shape corr twins for w32.
Same docker nvcc + symbol-check pattern as build_p6.py (-Xptxas -v readout).
"""
import sys, os
sys.path.insert(0, "~/tinygrad-metal/engine0")
from build_p6 import build

C100 = ["-DCTXK=100352"]
C2K  = ["-DCTXK=2048"]
W32  = ["-DROWS=32", "-DHRP=2", "-DNW=16", "-DMINB=1"]
W64  = ["-DROWS=64", "-DHRP=2", "-DNW=16", "-DMINB=1"]
W64Q = ["-DROWS=64", "-DHRP=2", "-DNW=32", "-DMINB=1"]
W64H = ["-DROWS=64", "-DHRP=1", "-DNW=16", "-DMINB=1"]
TARGETS = [
  ("pfaw_w32_s13_100k",  "pf_attnw.cu", C100 + ["-DS=13"] + W32,  "pfaw32"),
  ("pfaw_w32_s26_100k",  "pf_attnw.cu", C100 + ["-DS=26"] + W32,  "pfaw32"),
  ("pfaw_w32_s13_2k",    "pf_attnw.cu", C2K  + ["-DS=13"] + W32,  "pfaw32"),
  ("pfaw_w64_s13_100k",  "pf_attnw.cu", C100 + ["-DS=13"] + W64,  "pfaw64"),
  ("pfaw_w64q_s13_100k", "pf_attnw.cu", C100 + ["-DS=13"] + W64Q, "pfaw64q"),
  ("pfaw_w64h_s13_100k", "pf_attnw.cu", C100 + ["-DS=13"] + W64H, "pfaw64h"),
  ("pfcw32_s13",         "pf_attnw.cu", C100 + ["-DS=13", "-DROWS=32", "-DHRP=2", "-DPFC_T32=1"], "pfcw32"),
  ("pfcw32_s26",         "pf_attnw.cu", C100 + ["-DS=26", "-DROWS=32", "-DHRP=2", "-DPFC_T32=1"], "pfcw32"),
  ("pfcw32_s13_2k",      "pf_attnw.cu", C2K  + ["-DS=13", "-DROWS=32", "-DHRP=2", "-DPFC_T32=1"], "pfcw32"),
  ("pfcw64_s13",         "pf_attnw.cu", C100 + ["-DS=13", "-DROWS=64", "-DHRP=2", "-DPFC_T32=1"], "pfcw64"),
  ("pfcw64h_s13",        "pf_attnw.cu", C100 + ["-DS=13", "-DROWS=64", "-DHRP=1", "-DPFC_T32=1"], "pfcw64h"),
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
  print(f"[build_p17 done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
