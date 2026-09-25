# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R8 K=10: build the M=11/T=11 probe cubins (m11.cu splits + r7d ffn8v11r7/
down8nw32v11r7 via build_r7d) + lookup11/accept11k/acceptsel11k + the ROWS=11
split-KV attention set (RMAX=66, RP=80, MAXOWN=3 — the R5 owner law; 14 pad rows)."""
import re, subprocess, os, sys, time
BASE = os.path.dirname(os.path.abspath(__file__))

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

  def build(name, extra=(), src=None):
    r = sh(["nvcc", "-arch=sm_86", "-cubin", "-Xptxas", "-v", *extra,
            f"--output-file={BASE}/{name}.cubin", f"{BASE}/{src or name}.cu"], env)
    if r.returncode:
      print(f"[build] {name} FAIL\n{r.stderr[-2000:]}"); sys.exit(1)
    regs = [l.strip() for l in (r.stderr or "").splitlines() if "registers" in l or "spill" in l]
    spill = [l for l in regs if "spill" in l and "0 bytes spill stores" not in l]
    # hm11 (ROWS=11: RMAX=66/RP=80/MAXOWN=3) carries ONE 4B index spill —
    # SASS-audited COLD (STL in the staging phase + one predicated LDL in the
    # reduce phase; the 12 HMMA are STL/LDL-free). Far below the P18 >~100B
    # nondet class; deterministic private-memory traffic.
    allow = (name == "spk_g4nw32hm11_100k" and all("8 bytes stack frame" in l for l in spill) and len(spill) <= 2)
    if spill and not allow:
      print(f"[build] {name} SPILL:\n" + "\n".join(spill)); sys.exit(1)
    if spill:
      print(f"[build] {name} NOTE: documented cold 8B spill (SASS-audited)", flush=True)
    print(f"[build] {name} OK | {regs[-1] if regs else ''}", flush=True)

  src = open(f"{BASE}/m11.cu").read()
  hdr = src[:src.find('extern "C" __global__')]
  ks = re.findall(r'extern "C" __global__ void __launch_bounds__\(\d+\) (\w+)\(', src)
  bodies = re.split(r'(?=extern "C" __global__ void)', src[src.find('extern "C" __global__'):])
  bodies = [b for b in bodies if b.strip()]   # the empty-first-element law
  assert len(bodies) == len(ks) == 14, (len(bodies), len(ks))
  for name, body in zip(ks, bodies):
    open(f"{BASE}/{name}.cu", "w").write(hdr + body.rstrip() + "\n")
    build(name)

  for n in ("lookup11_nw32", "accept11k", "acceptsel11k"):
    build(n)

  C100, S = 100352, 256
  build("spk_pre11qh_100k", src="spk_preqh", extra=("-DKPRE=spk_pre11qh_100k", f"-DCTXK={C100}", "-DROWS=11"))
  # hm11: ROWS=11 -> RMAX=66, RP=80 (14 pad rows, never stored/read); MAXOWN=3.
  _g = open(f"{BASE}/spk_g4hm.cu").read()
  _prag = chr(95) + chr(80) + 'ragma("' + "unroll" + '")' + chr(10) + "        for (int ks4 = 0; ks4 < QK_KS; ++ks4) {"
  assert _prag in _g
  open(f"{BASE}/.spk_g4hm_r11.cu", "w").write(_g.replace(_prag, "#pragma unroll 2" + chr(10) + "        for (int ks4 = 0; ks4 < QK_KS; ++ks4) {", 1))
  build("spk_g4nw32hm11_100k", src=".spk_g4hm_r11", extra=(f"-DKNAME=spk_g4nw32hm11_100k", f"-DCTXK={C100}", "-DROWS=11",
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
  open(f"{BASE}/spk_c11g_100k.cu", "w").write(_h + _b["K2S"])
  build("spk_c11g_100k", extra=("-DK2S=spk_c11g_100k", f"-DCTXK={C100}", "-DROWS=11", f"-DS={S}", f"-DCH={C100//S}", "-DUNROLL=4"))
  print("[build k11 done]")

if __name__ == "__main__":
  main()
