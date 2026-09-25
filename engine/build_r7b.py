# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7b builder: pf_gemm3w warp-spec cubins (single-class). One kernel per cubin,
warp-token names (ws6p2 = 6 warps 2 producers, k64 = 64-k stages)."""
import subprocess, os, sys, time
BASE = os.path.dirname(os.path.abspath(__file__))

def sh(cmd, env, tries=4):
  r = None
  for t in range(tries):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode == 0 or "failed to connect" not in (r.stderr or ""): return r
    print(f"[build] transient docker failure (try {t+1})", flush=True)
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

R7 = ["-DQCLASS=1", "-DREPACK=1"]
HM = ["-DHMMA=1"]
W = ["-DKCHW=64"]
TARGETS = [
  # ffn: NTILE=32, 4 consumer + 2 producer warps = 192 thr
  ("pfg3w_ffn_r7_m64_ws6p2k64", "pf_gemm3w.cu", R7 + W + ["-DKDIM=5120", "-DNDIM=17408", "-DNTHR=192", "-DNTILE=32",
    "-DMTILE=64", "-DFFN=1", "-DRES=0", "-DNCONS=4", "-DNPROD=2"] + HM, None),
  # fd (iq3d, RES): NTILE=64, 8+2 = 320 thr
  ("pfg3w_iq3d_r7_m64_ws10p2k64", "pf_gemm3w.cu", R7 + W + ["-DKDIM=17408", "-DNDIM=5120", "-DNTHR=320", "-DNTILE=64",
    "-DMTILE=64", "-DFFN=0", "-DRES=1", "-DNCONS=8", "-DNPROD=2"] + HM, None),
  # out (iq3o): NTILE=64, 8+2 = 320 thr
  ("pfg3w_iq3o_r7_m64_ws10p2k64", "pf_gemm3w.cu", R7 + W + ["-DKDIM=6144", "-DNDIM=5120", "-DNTHR=320", "-DNTILE=64",
    "-DMTILE=64", "-DFFN=0", "-DRES=0", "-DNCONS=8", "-DNPROD=2"] + HM, None),
  # fd 3-producer variant (12 warps = 384 thr) for the producer-count sweep
  ("pfg3w_iq3d_r7_m64_ws11p3k64", "pf_gemm3w.cu", R7 + W + ["-DKDIM=17408", "-DNDIM=5120", "-DNTHR=352", "-DNTILE=64",
    "-DMTILE=64", "-DFFN=0", "-DRES=1", "-DNCONS=8", "-DNPROD=3"] + HM, None),
]

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  ok = all(build(env, n, s, e, n) for n, s, e, _ in TARGETS)
  sys.exit(0 if ok else 1)

if __name__ == "__main__":
  main()
