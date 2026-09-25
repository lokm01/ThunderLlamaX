# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P11 G1 builder: K-split-2 twins of the classic M32 GEMM family + the
fixed-order combine kernel. Grid doubles 80 -> 160 CTAs (the outstanding-loads
theory vs the dext's load-stream wall). KS splits the K-chunk range; each half
writes EXACT fp32 partials; pfg_ks2c* adds p0+p1 in a fixed order (+res for
the iq3d RES class). Numerics: Tier-2 (fp32 partial-add reassociation).
"""
import sys
sys.path.insert(0, "~/tinygrad-metal/engine0")
from build_p6 import build
import os

G16 = ["-DHMMA=1", "-DM32=1"]
TARGETS = [
  ("pfg_iq3d_m32_ks2_res_hm_nw8k128", "pf_gemm.cu", ["-DQCLASS=1", "-DKDIM=17408", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DFFN=0", "-DRES=1", "-DKS=2"] + G16, None),
  ("pfg_iq3o_m32_ks2_hm_nw8k128",     "pf_gemm.cu", ["-DQCLASS=1", "-DKDIM=6144",  "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DFFN=0", "-DRES=0", "-DKS=2"] + G16, None),
  ("pfg_iq3s_m32_ks2_hm_nw8k128",     "pf_gemm.cu", ["-DQCLASS=5", "-DKDIM=6144",  "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DFFN=0", "-DRES=0", "-DKS=2"] + G16, None),
  ("pfg_ks2c_res_hm",  "pf_ks2c.cu", ["-DKS2C=1", "-DKS2C_RES=1", "-DNDIM=5120", "-DNTHR=256"], None),
  ("pfg_ks2c_hm",      "pf_ks2c.cu", ["-DKS2C=1", "-DNDIM=5120", "-DNTHR=256"], None),
]

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  only = sys.argv[1:] if len(sys.argv) > 1 else None
  n_ok = n_fail = 0
  for name, src, extra, kname in TARGETS:
    if only and not any(o in name for o in only): continue
    if build(env, name, src, extra, kname or name): n_ok += 1
    else: n_fail += 1
  print(f"[build_p11 done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
