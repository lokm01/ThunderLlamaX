# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R4: build M=5 probe kernels (m5.cu per-kernel split) + lookup5 + acceptk +
accept5k + ROWS=5 split-KV attention (spk_g.cu is ROWS-generic per W2D)."""
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
# 1) m5.cu per-kernel split
src = open(f"{BASE}/m5.cu").read()
hdr = src[:src.find('extern "C" __global__')]
ks = re.findall(r'extern "C" __global__ void __launch_bounds__\(\d+\) (\w+)\(', src)
bodies = re.split(r'(?=extern "C" __global__ void)', src[src.find('extern "C" __global__'):])
bodies = [b for b in bodies if b.strip()]
assert len(bodies) == len(ks), (len(bodies), len(ks))
for name, body in zip(ks, bodies):
  open(f"{BASE}/{name}.cu", "w").write(hdr + body.rstrip() + "\n")
  r = sh(["nvcc", "-arch=sm_86", "-cubin", f"--output-file={BASE}/{name}.cubin", f"{BASE}/{name}.cu"], env)
  if r.returncode: print(f"[build] {name} FAIL\n{r.stderr[-1500:]}"); sys.exit(1)
  print(f"[build] {name} OK", flush=True)
# 2) lookup5 + accepts
for n in ("lookup5_nw32", "acceptk", "accept5k"):
  r = sh(["nvcc", "-arch=sm_86", "-cubin", f"--output-file={BASE}/{n}.cubin", f"{BASE}/{n}.cu"], env)
  if r.returncode: print(f"[build] {n} FAIL\n{r.stderr[-1500:]}"); sys.exit(1)
  print(f"[build] {n} OK", flush=True)
# 3) split-KV attention ROWS=5 (same splitter as build_k3.py)
def split_skv():
  src = open(f"{BASE}/skv_split.cu").read()
  full = 'extern "C" __global__'
  hdr = src[:src.find(full)]
  tail = src[src.find(full):]
  parts, idx = [], 0
  while True:
    nxt = tail.find(full, idx + 1)
    if nxt == -1:
      parts.append(tail[idx:]); break
    parts.append(tail[idx:nxt]); idx = nxt
  out = {}
  for b in parts:
    if not b.strip(): continue
    m = re.match(re.escape(full) + r" void __launch_bounds__\(\d+\) (\w+)\(", b)
    out[m.group(1)] = b.rstrip() + "\n"
  assert set(out) == {"KPRE", "K1S", "K2S"}, out.keys()
  return hdr, out
_SKVHDR, _SKVBOD = split_skv()
C2, C100 = 2304, 100352
for suf, C, S in (("2k", C2, 32), ("100k", C100, 256)):
  for kern, nm in (("KPRE", f"spk_pre5_{suf}"), ("K2S", "spk_c5" if suf == "2k" else "spk_c5g_100k")):
    open(f"{BASE}/{nm}.cu", "w").write(_SKVHDR + _SKVBOD[kern])
    r = sh(["nvcc", "-arch=sm_86", "-cubin", f"-D{kern}={nm}", f"-DCTXK={C}", "-DROWS=5", f"-DS={S}", f"-DCH={C//S}", "-DUNROLL=4", f"--output-file={BASE}/{nm}.cubin", f"{BASE}/{nm}.cu"], env)
    if r.returncode: print(f"[build] {nm} FAIL\n{r.stderr[-2000:]}"); sys.exit(1)
    print(f"[build] {nm} OK (CTXK={C} ROWS=5 S={S})", flush=True)
  nm = f"spk_g4nw32a5_{suf}"
  r = sh(["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={nm}", f"-DCTXK={C}", "-DROWS=5", f"-DS={S}", f"-DCH={C//S}", "-DTILE=32", "-DNW=32", f"--output-file={BASE}/{nm}.cubin", f"{BASE}/spk_g4.cu"], env)
  if r.returncode: print(f"[build] {nm} FAIL\n{r.stderr[-2000:]}"); sys.exit(1)
  print(f"[build] {nm} OK (ROWS=5 S={S} NW=32)", flush=True)
print("[build k4 done]", flush=True)
