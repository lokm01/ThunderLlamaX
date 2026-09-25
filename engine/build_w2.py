# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Build W2-MTP kernels: split m3.cu/m3b.cu/mtpd.cu per-kernel (own cubin each),
compile q4v.cu per-use with -D flags. Same pattern as build_w1c.py."""
import re, subprocess, os, sys, time
BASE = os.path.dirname(os.path.abspath(__file__))

def sh(cmd, env, tries=4):
  r = None
  for t in range(tries):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode == 0 or "failed to connect to the docker API" not in r.stderr: return r
    print(f"[build] transient docker API failure (try {t+1}), retrying...", flush=True)
    time.sleep(2)
  return r

def split_build(srcfile):
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  src = open(f"{BASE}/{srcfile}").read()
  hdr = src[:src.find('extern "C" __global__')]
  ks = re.findall(r'extern "C" __global__ void __launch_bounds__\(\d+\) (\w+)\(', src)
  bodies = re.split(r'(?=extern "C" __global__ void)', src[src.find('extern "C" __global__'):])
  bodies = [b for b in bodies if b.strip()]
  assert len(bodies) == len(ks), (srcfile, len(bodies), len(ks))
  for name, body in zip(ks, bodies):
    open(f"{BASE}/{name}.cu", "w").write(hdr + body.rstrip() + "\n")
    r = sh(["nvcc", "-arch=sm_86", "-cubin", f"--output-file={BASE}/{name}.cubin", f"{BASE}/{name}.cu"], env)
    if r.returncode:
      print(f"[build] {name} FAIL\n{r.stderr[-1500:]}"); sys.exit(1)
    print(f"[build] {name} OK", flush=True)

def build_q4(env):
  # (cubin name, NOUT, NGRP, ADDHH)
  jobs = [("ehproj", 5120, 40, 1), ("dq", 12288, 20, 0), ("doproj", 5120, 24, 0), ("ddown", 5120, 68, 1)]
  for nm, nout, ngrp, addhh in jobs:
    cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={nm}", f"-DNOUT={nout}", f"-DNGRP={ngrp}"]
    if addhh: cmd.append("-DADDHH=1")
    cmd += [f"--output-file={BASE}/{nm}.cubin", f"{BASE}/q4v.cu"]
    r = sh(cmd, env)
    if r.returncode:
      print(f"[build] {nm} FAIL\n{r.stderr[-1500:]}"); sys.exit(1)
    print(f"[build] {nm} OK (NOUT={nout} NGRP={ngrp} ADDHH={addhh})", flush=True)

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  for f in ("m3.cu", "m3b.cu", "mtpd.cu"):
    split_build(f)
  build_q4(env)
  print("[build all done]", flush=True)

if __name__ == "__main__":
  main()
