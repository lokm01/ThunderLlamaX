# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Build split-KV cubins (per-kernel cubin law). skv.cu split into 3 bodies;
each compiled with -D renames + ctx/rows/splits defines."""
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

def split_srcs():
  src = open(f"{BASE}/skv.cu").read()
  marker = chr(101)+chr(120)+chr(116)+chr(101)+chr(114)+chr(110)  # avoids quote nesting in this file
  full = marker + ' "C" ' + chr(95)*2 + 'global' + chr(95)*2
  hdr = src[:src.find(full)]
  bodies = [b for b in src[src.find(full):].split("(?=") if b.strip()]
  # manual split on the marker
  tail = src[src.find(full):]
  parts = []
  idx = 0
  while True:
    nxt = tail.find(full, idx + 1)
    if nxt == -1:
      parts.append(tail[idx:]); break
    parts.append(tail[idx:nxt]); idx = nxt
  out = {}
  for b in parts:
    if not b.strip(): continue
    m = re.match(full + r' void ' + chr(95)*2 + r'launch_bounds' + chr(95)*2 + r'\(\d+\) (\w+)\(', b)
    out[m.group(1)] = b.rstrip() + "\n"
  assert set(out) == {"KPRE", "K1S", "K2S"}, out.keys()
  return hdr, out

def build(env, name, kernel, ctxk, rows, S, unroll=4):
  hdr, bodies = split_srcs()
  open(f"{BASE}/{name}.cu", "w").write(hdr + bodies[kernel].replace("#pragma unroll UNROLL", f"#pragma unroll {unroll}"))
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-D{kernel}={name}", f"-DCTXK={ctxk}",
         f"-DROWS={rows}", f"-DS={S}", f"-DCH={ctxk//S}", f"-DUNROLL={unroll}",
         f"--output-file={BASE}/{name}.cubin", f"{BASE}/{name}.cu"]
  r = sh(cmd, env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-2000:]}"); sys.exit(1)
  print(f"[build] {name} OK (CTXK={ctxk} ROWS={rows} S={S} CH={ctxk//S} U={unroll})", flush=True)

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  jobs = [
    ("spk_pre3_2k",   "KPRE", 2304,   3, 32), ("spk_pre3_100k", "KPRE", 100352, 3, 32),
    ("spk_a3_2k",    "K1S",  2304,   3, 32), ("spk_a3_100k",  "K1S",  100352, 3, 32),
    ("spk_c3",       "K2S",  0,      3, 32),
    ("spk_pre1_2k",   "KPRE", 2304,   1, 32), ("spk_pre1_100k", "KPRE", 100352, 1, 32),
    ("spk_a1_2k",    "K1S",  2304,   1, 32), ("spk_a1_100k",  "K1S",  100352, 1, 32),
    ("spk_c1",       "K2S",  0,      1, 32),
  ]
  for j in jobs: build(env, *j)
  print("[build_skv done]", flush=True)

if __name__ == "__main__":
  main()
