# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2c: build the decode-side r7 GEMV cubins from r7d.cu (build_w1c.py pattern:
split per-kernel .cu -> per-kernel cubin; the multi-kernel cubin mis-load law).
-Xptxas -v captured per kernel; regs + spill printed (the P18 spill law)."""
import re, subprocess, os, sys
BASE = os.path.dirname(os.path.abspath(__file__))

def sh(cmd, env, tries=4):
  import time
  r = None
  for t in range(tries):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode == 0 or "failed to connect to the docker API" not in r.stderr: return r
    print(f"[build] transient docker API failure (try {t+1}), retrying...", flush=True)
    time.sleep(2)
  return r

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  src = open(f"{BASE}/r7d.cu").read()
  hdr = src[:src.find('extern "C" __global__')]
  ks = re.findall(r'extern "C" __global__ void __launch_bounds__\(\d+\) (\w+)\(', src)
  bodies = re.split(r'(?=extern "C" __global__ void)', src[src.find('extern "C" __global__'):])
  bodies = [b for b in bodies if b.strip()]
  assert len(bodies) == len(ks), (len(bodies), len(ks))
  for name, body in zip(ks, bodies):
    open(f"{BASE}/{name}.cu", "w").write(hdr + body.rstrip() + "\n")
    r = sh(["bash", "-c", f"nvcc -arch=sm_86 -cubin -Xptxas -v --output-file={BASE}/{name}.cubin {BASE}/{name}.cu 2> {BASE}/.{name}.ptxas"], env)
    if r.returncode:
      print(f"[build] {name} FAIL\n{r.stderr[-1500:]}"); sys.exit(1)
    pt = open(f"{BASE}/.{name}.ptxas").read()
    regs = re.search(r"Used (\d+) registers", pt)
    spill = re.search(r"(\d+) bytes spill stores", pt)
    print(f"[build] {name} OK regs={regs.group(1) if regs else '?'} spill={spill.group(1) if spill else '0'}", flush=True)
  print("[build all done]", flush=True)

if __name__ == "__main__":
  main()
