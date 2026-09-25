# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R5d K=7: build the M=8/T=8 kernel set (mirrors build_k6.py)."""
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

src = open(f"{BASE}/m8.cu").read()
hdr = src[:src.find('extern "C" __global__')]
ks = re.findall(r'extern "C" __global__ void __launch_bounds__\(\d+\) (\w+)\(', src)
bodies = re.split(r'(?=extern "C" __global__ void)', src[src.find('extern "C" __global__'):])
bodies = [b for b in bodies if b.strip()]
assert len(bodies) == len(ks) == 14, (len(bodies), len(ks))
for name, body in zip(ks, bodies):
    open(f"{BASE}/{name}.cu", "w").write(hdr + body.rstrip() + "\n")
    build(name)

for n in ("lookup8_nw32", "accept8k", "acceptsel8k"):
    build(n)

C100, S = 100352, 256
build("spk_pre8qh_100k", src="spk_preqh", extra=("-DKPRE=spk_pre8qh_100k", f"-DCTXK={C100}", "-DROWS=8"))
# hm8: ks4-unroll-4 variant for the 64-reg/1024-thr budget (same knob as hm6/hm7).
# ROWS=8 -> RMAX=48, RP=48 (NO padding rows), MAXOWN=2 owner path.
_g8 = open(f"{BASE}/spk_g4hm.cu").read()
_prag = chr(95) + chr(80) + 'ragma("' + "unroll" + '")' + chr(10) + "        for (int ks4 = 0; ks4 < QK_KS; ++ks4) {"
assert _prag in _g8
open(f"{BASE}/.spk_g4hm_r8.cu", "w").write(_g8.replace(_prag, "#pragma unroll 4" + chr(10) + "        for (int ks4 = 0; ks4 < QK_KS; ++ks4) {", 1))
build("spk_g4nw32hm8_100k", src=".spk_g4hm_r8", extra=(f"-DKNAME=spk_g4nw32hm8_100k", f"-DCTXK={C100}", "-DROWS=8",
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
open(f"{BASE}/spk_c8g_100k.cu", "w").write(_h + _b["K2S"])
build("spk_c8g_100k", extra=("-DK2S=spk_c8g_100k", f"-DCTXK={C100}", "-DROWS=8", f"-DS={S}", f"-DCH={C100//S}", "-DUNROLL=4"))
print("[build k8 done]")
