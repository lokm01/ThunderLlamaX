# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""D0 build: ffn8l_3 cubin + cuobjdump reg/smem check."""
import subprocess, os, sys, time
BASE = "~/tinygrad-metal/engine0"
env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
           DOCKER_HOST="unix://~/.colima/default/docker.sock")
def sh(cmd, tries=4):
  r = None
  for t in range(tries):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode == 0 or "failed to connect to the docker API" not in r.stderr: return r
    print(f"[build] transient docker API failure (try {t+1}), retry...", flush=True); time.sleep(2)
  return r
r = sh(["nvcc", "-arch=sm_86", "-cubin", f"--output-file={BASE}/ffn8l_3.cubin", f"{BASE}/ffn8l_3.cu"])
if r.returncode:
  print(f"[build] FAIL\n{r.stderr[-2000:]}"); sys.exit(1)
print("[build] ffn8l_3 OK", flush=True)
r = sh(["docker", "exec", "cuda-nvcc-persistent", "cuobjdump", "-res-usage", f"{BASE}/ffn8l_3.cubin"])
print(r.stdout[-1500:] if r.returncode == 0 else f"cuobjdump FAIL {r.stderr[-500:]}")
