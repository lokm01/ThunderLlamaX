# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7a norms rung: rebuild the per-row-CTA norm/emb kernels (m8.cu + m3.cu
splits; one kernel per cubin per the MULTI-KERNEL CUBIN law)."""
import re, subprocess, os, sys, time
BASE = os.path.dirname(os.path.abspath(__file__))
WANT = {"h_embed8", "k0n8", "k0ab8", "hh8", "h_embed3", "k0n3", "k0ab3", "hh3"}

def sh(cmd, env, tries=4):
  r = None
  for t in range(tries):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode == 0 or "failed to connect to the docker API" not in r.stderr: return r
    time.sleep(2)
  return r

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "start", "cuda-nvcc-persistent"], env); sh(["docker", "ps"], env)
  for srcname in ("m8.cu", "m3.cu"):
    src = open(f"{BASE}/{srcname}").read()
    hdr = src[:src.find('extern "C" __global__')]
    ks = re.findall(r'extern "C" __global__ void __launch_bounds__\(\d+\) (\w+)\(', src)
    bodies = re.split(r'(?=extern "C" __global__ void)', src[src.find('extern "C" __global__'):])
    bodies = [b for b in bodies if b.strip()]   # the empty-first-element law (build_r7d.py)
    assert len(bodies) == len(ks), (srcname, len(ks), len(bodies))
    for name, body in zip(ks, bodies):
      if name not in WANT: continue
      open(f"{BASE}/{name}.cu", "w").write(hdr + body.rstrip() + "\n")
      r = sh(["bash", "-c", f"nvcc -arch=sm_86 -cubin -Xptxas -v --output-file={BASE}/{name}.cubin {BASE}/{name}.cu 2> {BASE}/.{name}.ptxas"], env)
      if r.returncode: print(f"[build] {name} FAIL\n{r.stderr[-1500:]}"); sys.exit(1)
      pt = open(f"{BASE}/.{name}.ptxas").read()
      regs = re.search(r"Used (\d+) registers", pt)
      spill = re.search(r"(\d+) bytes spill stores", pt)
      n_spill = int(spill.group(1)) if spill else 0
      print(f"[build] {name} OK regs={regs.group(1) if regs else '?'} spill={n_spill}", flush=True)
      assert n_spill == 0, name
  print("[build norms done]", flush=True)

if __name__ == "__main__":
  main()
