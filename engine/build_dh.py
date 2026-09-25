# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2G L3: build half2-core draft GEMV cubins (q4vh/dkvh/dfguh sources).
Same -D scheme as build_w2.py jobs; unique names per per-kernel cubin law;
cuobjdump symbol check inside the container."""
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

def sym_ok(env, name):
  d = sh(["docker", "exec", "cuda-nvcc-persistent", "cuobjdump", "-symbols", f"{BASE}/{name}.cubin"], env)
  if name not in [l.split()[-1] for l in (d.stdout or "").splitlines() if l.strip().endswith(name)]:
    print(f"[build] {name} SYMBOL CHECK FAIL:\n{d.stdout}\n{d.stderr}"); sys.exit(1)
  print(f"[build] {name} symbol OK", flush=True)

def build(env, name, src, extra=()):
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={name}", *extra,
         f"--output-file={BASE}/{name}.cubin", f"{BASE}/{src}"]
  r = sh(cmd, env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-3000:]}"); sys.exit(1)
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "spill" in l]
  print(f"[build] {name} OK {' | '.join(regs[-2:])}", flush=True)
  sym_ok(env, name)

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  # (cubin name, NOUT, NGRP, ADDHH) — same jobs as build_w2.py
  for nm, nout, ngrp, addhh in (("ehprojh", 5120, 40, 1), ("dqh", 12288, 20, 0),
                                 ("doprojh", 5120, 24, 0), ("ddownh", 5120, 68, 1)):
    extra = [f"-DNOUT={nout}", f"-DNGRP={ngrp}"] + (["-DADDHH=1"] if addhh else [])
    build(env, nm, "q4vh.cu", extra)
  build(env, "dkvh", "dkvh.cu")
  build(env, "dfguh", "dfguh.cu")
  print("[build_dh done]", flush=True)

if __name__ == "__main__":
  main()
