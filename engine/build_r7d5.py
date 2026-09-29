# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P10-dense rung 2: build ONLY the two new r7d.cu kernels (ffn8v5r7 /
down8nw32v5r7) as per-kernel cubins (build_r7d.py pattern; the multi-kernel
cubin mis-load law). -Xptxas -v regs/spill audit — the P18 spill law (HARD
fail on any spill)."""
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

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "start", "cuda-nvcc-persistent"], env); sh(["docker", "ps"], env)
  src = open(f"{BASE}/r7d.cu").read()
  hdr = src[:src.find('extern "C" __global__')]
  want = ("ffn8v5r7", "down8nw32v5r7")
  bodies = re.split(r'(?=extern "C" __global__ void)', src[src.find('extern "C" __global__'):])
  bodies = [b for b in bodies if b.strip()]
  found = {}
  for b in bodies:
    m = re.match(r'extern "C" __global__ void __launch_bounds__\((\d+)\) (\w+)\(', b)
    if m and m.group(2) in want: found[m.group(2)] = b
  assert set(found) == set(want), (sorted(found), want)
  for name in want:
    open(f"{BASE}/{name}.cu", "w").write(hdr + found[name].rstrip() + "\n")
    r = sh(["bash", "-c", f"nvcc -arch=sm_86 -cubin -Xptxas -v --output-file={BASE}/{name}.cubin {BASE}/{name}.cu 2> {BASE}/.{name}.ptxas"], env)
    if r.returncode:
      print(f"[build] {name} FAIL\n{r.stderr[-2000:]}"); sys.exit(1)
    pt = open(f"{BASE}/.{name}.ptxas").read()
    regs = re.search(r"Used (\d+) registers", pt)
    sp = re.search(r"(\d+) bytes spill stores", pt)
    nsp = int(sp.group(1)) if sp else 0
    if nsp > 0:
      print(f"[build] {name} SPILL {nsp}B — HARD FAIL (the P18 spill law)"); sys.exit(1)
    print(f"[build] {name} OK regs={regs.group(1) if regs else '?'} spill=0", flush=True)
  print("[build r7d5 done]", flush=True)

if __name__ == "__main__":
  main()
