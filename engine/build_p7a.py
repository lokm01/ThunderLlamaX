# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-A Stage-0 builder: the four microbenchmark probes.
Same docker-nvcc + cuobjdump symbol-check pattern as build_p5.
Laws honored: per-kernel cubin, nw-token names, hardcoded sizes via -D."""
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

def build(env, name, src, extra):
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={name}"] + extra + \
        ["-Xptxas", "-v", f"--output-file={BASE}/{name}.cubin", f"{BASE}/{src}"]
  r = sh(cmd, env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-2500:]}"); return False
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l.lower() or "spill" in l.lower()]
  d = sh(["docker", "exec", "cuda-nvcc-persistent", "cuobjdump", "-symbols", f"{BASE}/{name}.cubin"], env)
  if name not in [l.split()[-1] for l in (d.stdout or "").splitlines() if l.strip().endswith(name)]:
    print(f"[build] {name} SYMBOL CHECK FAIL:\n{d.stdout}\n{d.stderr}"); return False
  print(f"[build] {name} OK {' | '.join(regs[-3:])}", flush=True)
  return True

IMMA = ["-DNTHR=128"]
TARGETS = [
  # probe 1: IMMA (int8 tensor cores)
  ("p7a_imma16832_nw4",   "p7a_imma.cu",   IMMA + ["-DILP=1"]),
  ("p7a_imma16832i4_nw4", "p7a_imma.cu",   IMMA + ["-DILP=4"]),
  ("p7a_imma16832u8_nw4", "p7a_imma.cu",   IMMA + ["-DILP=4", "-DU8=1"]),
  ("p7a_imma8816_nw4",    "p7a_imma.cu",   IMMA + ["-DMMA8816=1", "-DILP=4"]),
  # probe 2: quant-word stream, strided vs repacked
  ("p7a_wstrd_nw8",  "p7a_repack.cu", ["-DNTHR=256", "-DKERNEL=0"]),
  ("p7a_wstrd_nw16", "p7a_repack.cu", ["-DNTHR=512", "-DKERNEL=0"]),
  ("p7a_wrep8_nw8",  "p7a_repack.cu", ["-DNTHR=256", "-DKERNEL=1", "-DRING=8"]),
  ("p7a_wrep8_nw16", "p7a_repack.cu", ["-DNTHR=512", "-DKERNEL=1", "-DRING=8"]),
  ("p7a_wrep16_nw8", "p7a_repack.cu", ["-DNTHR=256", "-DKERNEL=1", "-DRING=16"]),
  # probe 3: int8-KV slab stream
  ("p7a_kvs_nw32", "p7a_kv.cu", ["-DNTHR=1024", "-DCTXK=100352"]),
  ("p7a_kvs_nw16", "p7a_kv.cu", ["-DNTHR=512",  "-DCTXK=100352"]),
  # probe 4: ldmatrix
  ("p7a_ldmx_nw8", "p7a_ldmx.cu", ["-DNTHR=256"]),
]

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  only = sys.argv[1:] if len(sys.argv) > 1 else None
  n_ok = n_fail = 0
  for name, src, extra in TARGETS:
    if only and not any(o in name for o in only): continue
    if build(env, name, src, extra): n_ok += 1
    else: n_fail += 1
  print(f"[build] done ok={n_ok} fail={n_fail}")
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
