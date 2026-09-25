# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R5 K=5: build the M=6/T=6 kernel set. (1) split m6.cu per-kernel (the
per-kernel-cubin law) + nvcc each with -Xptxas -v (zero-spill gate);
(2) lookup6_nw32 + accept6k + acceptsel6k; (3) ROWS=6 split-KV attention:
spk_pre6qh_100k (preqh source), spk_g4nw32hm6_100k (g4hm source),
spk_c6g_100k (skv_split K2S body)."""
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
sh(["docker", "start", "cuda-nvcc-persistent"], env)
sh(["docker", "ps"], env)

def build(name, extra=(), src=None):
    r = sh(["nvcc", "-arch=sm_86", "-cubin", "-Xptxas", "-v", *extra,
            f"--output-file={BASE}/{name}.cubin", f"{BASE}/{src or name}.cu"], env)
    if r.returncode:
        print(f"[build] {name} FAIL\n{r.stderr[-2000:]}"); sys.exit(1)
    regs = [l.strip() for l in (r.stderr or "").splitlines() if "registers" in l or "spill" in l]
    spill = [l for l in regs if "spill" in l and "0 bytes spill stores" not in l]
    if spill:
        print(f"[build] {name} SPILL:\n" + "\n".join(spill)); sys.exit(1)
    print(f"[build] {name} OK | {regs[-1] if regs else ''}", flush=True)

# 1) m6.cu per-kernel split
src = open(f"{BASE}/m6.cu").read()
hdr = src[:src.find('extern "C" __global__')]
ks = re.findall(r'extern "C" __global__ void __launch_bounds__\(\d+\) (\w+)\(', src)
bodies = re.split(r'(?=extern "C" __global__ void)', src[src.find('extern "C" __global__'):])
bodies = [b for b in bodies if b.strip()]
assert len(bodies) == len(ks) == 14, (len(bodies), len(ks))
for name, body in zip(ks, bodies):
    open(f"{BASE}/{name}.cu", "w").write(hdr + body.rstrip() + "\n")
    build(name)

# 2) lookup6 + accepts
for n in ("lookup6_nw32", "accept6k", "acceptsel6k"):
    build(n)

# 3) ROWS=6 split-KV attention (canonical QH+HMMA path)
C100, S = 100352, 256
build("spk_pre6qh_100k", src="spk_preqh", extra=("-DKPRE=spk_pre6qh_100k", f"-DCTXK={C100}", "-DROWS=6"))
# R5: ROWS=6 spills at full ks4 unroll under the 64-reg/1024-thread budget;
# build from a tuned variant (ks4 unroll 8 — scheduling-only, per-row fp op
# order unchanged; the hm3/hm5 sources stay untouched).
_g6 = open(f"{BASE}/spk_g4hm.cu").read()
_old = chr(95)+chr(80)+"ragma(" + chr(34) + "unroll" + chr(34) + ")" + chr(10) + "        for (int ks4 = 0; ks4 < QK_KS; ++ks4) {"
assert _old in _g6
open(f"{BASE}/.spk_g4hm_r6.cu", "w").write(_g6.replace(_old, "#pragma unroll 4" + chr(10) + "        for (int ks4 = 0; ks4 < QK_KS; ++ks4) {", 1))
build("spk_g4nw32hm6_100k", src=".spk_g4hm_r6", extra=(f"-DKNAME=spk_g4nw32hm6_100k", f"-DCTXK={C100}", "-DROWS=6",
                                   f"-DS={S}", f"-DCH={C100//S}", "-DTILE=32", "-DNW=32"))
def split_skv():
    src = open(f"{BASE}/skv_split.cu").read()
    full = 'extern "C" __global__'
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
    return src[:src.find(full)], out
_h, _b = split_skv()
open(f"{BASE}/spk_c6g_100k.cu", "w").write(_h + _b["K2S"])
build("spk_c6g_100k", extra=("-DK2S=spk_c6g_100k", f"-DCTXK={C100}", "-DROWS=6", f"-DS={S}", f"-DCH={C100//S}", "-DUNROLL=4"))
print("[build k5 done]")
