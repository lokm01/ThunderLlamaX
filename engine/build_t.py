# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Build W2D T-layout kernels per-kernel (own cubin each)."""
import re, subprocess, os, sys, time
BASE = os.path.dirname(os.path.abspath(__file__))
def sh(cmd, env, tries=4):
  r = None
  for t in range(tries):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode == 0 or "failed to connect to the docker API" not in r.stderr: return r
    print(f"[build] transient docker API failure (try {t+1}), retrying...", flush=True); time.sleep(2)
  return r
env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
           DOCKER_HOST="unix://~/.colima/default/docker.sock")
sh(["docker", "ps"], env)
src = open(f"{BASE}/m3v2.cu").read()
hdr = src[:src.find('extern "C" __global__')]
ks = re.findall(r'extern "C" __global__ void __launch_bounds__\(\d+\) (\w+)\(', src)
bodies = re.split(r'(?=extern "C" __global__ void)', src[src.find('extern "C" __global__'):])
bodies = [b for b in bodies if b.strip()]
assert len(bodies) == len(ks), (len(bodies), len(ks))
for name, body in zip(ks, bodies):
  open(f"{BASE}/{name}.cu", "w").write(hdr + body.rstrip() + "\n")
  r = sh(["nvcc", "-arch=sm_86", "-cubin", f"--output-file={BASE}/{name}.cubin", f"{BASE}/{name}.cu"], env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-1500:]}"); sys.exit(1)
  print(f"[build] {name} OK", flush=True)
print("[build t done]", flush=True)
