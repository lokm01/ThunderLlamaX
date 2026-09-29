# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""TLX P10-dense: build the full-vocab draft-head cubins (build_r7d.py pattern:
per-kernel .cu -> cubin via docker nvcc sm_86; -Xptxas -v regs/spill audit —
the P18 spill law)."""
import subprocess, os, sys, re
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
  sh(["docker", "start", "cuda-nvcc-persistent"], env)
  sh(["docker", "ps"], env)
  for name in ("sheadf", "samxf"):
    r = sh(["bash", "-c", f"nvcc -arch=sm_86 -cubin -Xptxas -v --output-file={BASE}/{name}.cubin {BASE}/{name}.cu 2> {BASE}/.{name}.ptxas"], env)
    if r.returncode:
      print(f"[build] {name} FAIL\n{r.stderr[-1500:]}"); sys.exit(1)
    pt = open(f"{BASE}/.{name}.ptxas").read()
    regs = re.search(r"Used (\d+) registers", pt)
    spill = re.search(r"(\d+) bytes spill stores", pt)
    _rq = regs.group(1) if regs else "?"
    _sp = spill.group(1) if spill else "0"
    print(f"[build] {name} OK regs={_rq} spill={_sp}", flush=True)
  print("[build all done]", flush=True)
if __name__ == "__main__":
  main()
