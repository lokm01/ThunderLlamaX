# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P8 builder: (1) gdnqg packed5 twin (pf_gemm3m -DP5A=1 -DRPA=2: the qkv Q5_K
seg reads the pack_w5 true-16B layout; gate seg B packed7 RING4 verbatim);
(2) iq3s o-proj M-grid fold (pf_gemm3 QCLASS=5 classic, MTILE=64 — ONE g=160
launch replaces 4x m32 g=80 per attn block; the P7E4/R2d fold pattern).
Same docker nvcc + symbol-check pattern as build_r2d.py."""
import subprocess, os, sys, time
BASE = os.path.dirname(os.path.abspath(__file__))

def sh(cmd, env, tries=4):
  r = None
  for t in range(tries):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode == 0 or "failed to connect to the docker API" not in r.stderr: return r
    print(f"[build] transient docker API failure (try {t+1}), retrying...", flush=True)
    time.sleep(2)
  return r

def build(env, name, src, extra, kname):
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={kname}"] + extra + \
        ["-Xptxas", "-v", f"--output-file={BASE}/{name}.cubin", f"{BASE}/{src}"]
  r = sh(cmd, env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-2500:]}"); return False
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l.lower() or "spill" in l.lower()]
  d = sh(["docker", "exec", "cuda-nvcc-persistent", "cuobjdump", "-symbols", f"{BASE}/{name}.cubin"], env)
  if kname not in [l.split()[-1] for l in (d.stdout or "").splitlines() if l.strip().endswith(kname)]:
    print(f"[build] {name} SYMBOL CHECK FAIL:\n{d.stdout}\n{d.stderr}"); return False
  print(f"[build] {name} OK {' | '.join(regs[-3:])}", flush=True)
  return True

HM = ["-DHMMA=1"]
TARGETS = [
  # gdnqg packed5: seg A qkv Q5_K -> packed5 ring2; seg B gate packed7 ring4 (R2d verbatim)
  ("pfg3m_gdnqg_r7q4p5_m64_nw8k128", "pf_gemm3m.cu",
   ["-DSEGN=2", "-DKDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64",
    "-DRING4=1", "-DP5A=1"] + HM +
   ["-DQCA=2", "-DNDA=10240", "-DGRID_A=160", "-DRPA=2", "-DQCB=1", "-DNDB=6144", "-DGRID_B=96", "-DRPB=1"], None),
  # iq3s o-proj M-grid fold: classic QCLASS=5, MTILE=64, NTILE=64 (the pfg3_iq3o geometry)
  ("pfg3_iq3s_m64_nw8k128", "pf_gemm3.cu",
   ["-DQCLASS=5", "-DKDIM=6144", "-DNDIM=5120", "-DNTHR=256", "-DNTILE=64", "-DKCH=128",
    "-DMTILE=64", "-DFFN=0", "-DRES=0", "-DREPACK=0"] + HM, None),
]

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  only = sys.argv[1:] if len(sys.argv) > 1 else None
  n_ok = n_fail = 0
  for name, src, extra, kname in TARGETS:
    if only and not any(o in name for o in only): continue
    if build(env, name, src, extra, kname or name): n_ok += 1
    else: n_fail += 1
  print(f"[build_p8 done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
