# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P8w4 builder: the W4A8 FFN pair — pfk_q8.cu (act quant) + p8_w4ffn.cu (fused
gate+up IMMA GEMM with the silu*u epilogue). Same docker nvcc + symbol-check
pattern as build_p8.py."""
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

TARGETS = [
  ("p8q8x_nw8k128", "pfk_q8.cu", ["-DKDIM=5120"], "p8q8x"),
  ("p8w4ffn_nw8k128", "p8_w4ffn.cu",
   ["-DKDIM=5120", "-DNDIM=17408", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64"], "p8w4ffn"),
  ("p8w4ffn7_nw8k128", "p8_w4ffn7.cu",
   ["-DKDIM=5120", "-DNDIM=17408", "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=64"], "p8w4ffn7"),
]

def main():
  env = dict(os.environ)
  env["PATH"] = "~/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:" + env.get("PATH", "")
  env["DOCKER_HOST"] = "unix://~/.colima/default/docker.sock"
  ok = all(build(env, n, s, e, k) for n, s, e, k in TARGETS)
  sys.exit(0 if ok else 1)

if __name__ == "__main__":
  main()
