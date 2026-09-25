# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P16 builder: the ws ping-pong GEMM variants (pf_gemm3.cu PING section).
Same docker nvcc + symbol-check pattern as build_p6.build.
  pfg3_iq3d_r7pp1_m32_nw8k128  43520B  discriminator A (decode off the mma->mma path)
  pfg3_iq3d_r7pp2_m32_nw4k128  34816B  the FULL ping-pong (ONE barrier/chunk), nt32 geometry
  pfg3_iq3d_r7pp1_m64_nw4k128  34816B  the gemm_fd-class shippable candidate (m64 twin, nt32)
  pfg3_ffn_r7pp1_m32_nw4k128   43520B  the FFN-class ping (W7 fg/fu covered case)
"""
import sys, os
sys.path.insert(0, "~/tinygrad-metal/engine0")
from build_p6 import build

HM = ["-DHMMA=1"]
R7 = ["-DQCLASS=1", "-DREPACK=1"]
TARGETS = [
  ("pfg3_iq3d_r7pp1_m32_nw8k128", "pf_gemm3.cu", R7 + ["-DKDIM=17408", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=32", "-DFFN=0", "-DRES=1", "-DPING=1"] + HM, None),
  ("pfg3_iq3d_r7pp2_m32_nw4k128", "pf_gemm3.cu", R7 + ["-DKDIM=17408", "-DNDIM=5120", "-DNTHR=128", "-DNTILE=32", "-DKCH=128", "-DMTILE=32", "-DFFN=0", "-DRES=1", "-DPING=2"] + HM, None),
  ("pfg3_iq3d_r7pp1_m64_nw4k128", "pf_gemm3.cu", R7 + ["-DKDIM=17408", "-DNDIM=5120", "-DNTHR=128", "-DNTILE=32", "-DKCH=128", "-DMTILE=64", "-DFFN=0", "-DRES=1", "-DPING=1"] + HM, None),
  ("pfg3_ffn_r7pp1_m32_nw4k128", "pf_gemm3.cu", R7 + ["-DKDIM=5120", "-DNDIM=17408", "-DNTHR=128", "-DNTILE=32", "-DKCH=128", "-DMTILE=32", "-DFFN=1", "-DRES=0", "-DPING=1"] + HM, None),
]

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  n_ok = n_fail = 0
  for name, src, extra, kname in TARGETS:
    kname = kname or name
    if build(env, name, src, extra, kname): n_ok += 1
    else: n_fail += 1
  print(f"[build_p16 done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
