# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P9 builder: pfa32c co-resident attention variants (t64 main / t32 K-staged /
16c). Same docker nvcc + symbol-check pattern as build_p6.py. -Xptxas -v gives
the GATE 1 register/spill readout. MINB sets __launch_bounds__ minBlocks/SM:
2 at 512thr -> ptxas targets 64 regs; 3 at 256thr -> 80 regs.
"""
import sys, os
sys.path.insert(0, "~/tinygrad-metal/engine0")
from build_p6 import build

C100 = ["-DCTXK=100352"]
C2K  = ["-DCTXK=2048"]
T64  = ["-DKNAME=pfa32c",  "-DS=13", "-DMINB=2"]
T64N = ["-DKNAME=pfa32c",  "-DS=13", "-DMINB=1"]   # diagnostic: natural reg count
T32  = ["-DKNAME=pfa32ct", "-DS=13", "-DMINB=2", "-DPFA_T32=1"]
R16  = ["-DKNAME=pfa32c16","-DS=10", "-DMINB=3", "-DPFA_R16=1"]
TARGETS = [
  ("pfa32c_t64_s13_100k",   "pf_attn32c.cu", C100 + T64,  "pfa32c"),
  ("pfa32c_t64n_s13_100k",  "pf_attn32c.cu", C100 + T64N, "pfa32c"),
  ("pfa32c_t32_s13_100k",   "pf_attn32c.cu", C100 + T32,  "pfa32ct"),
  ("pfa32c_16c_s10_100k",   "pf_attn32c.cu", C100 + R16,  "pfa32c16"),
  ("pfa32c_t64_s13_2k",     "pf_attn32c.cu", C2K + T64,   "pfa32c"),
  ("pfa32c_t32_s13_2k",     "pf_attn32c.cu", C2K + T32,  "pfa32ct"),
  ("pfa32c_16c_s10_2k",     "pf_attn32c.cu", C2K + R16,  "pfa32c16"),
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
  print(f"[build_p9 done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
