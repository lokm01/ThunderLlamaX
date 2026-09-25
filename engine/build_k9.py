# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7a K=8: build the M=9/T=9 probe cubins (m9.cu splits + r7d ffn8v9r7/
down8nw32v9r7 via build_r7d) + lookup9/accept9k/acceptsel9k + the ROWS=9
split-KV attention set (RMAX=54, RP=64, MAXOWN=2 — the R5 owner law)."""
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
    if spill:
      print(f"[build] {name} SPILL:\n" + "\n".join(spill)); sys.exit(1)
    print(f"[build] {name} OK | {regs[-1] if regs else ''}", flush=True)

  src = open(f"{BASE}/m9.cu").read()
  hdr = src[:src.find('extern "C" __global__')]
  ks = re.findall(r'extern "C" __global__ void __launch_bounds__\(\d+\) (\w+)\(', src)
  bodies = re.split(r'(?=extern "C" __global__ void)', src[src.find('extern "C" __global__'):])
  bodies = [b for b in bodies if b.strip()]   # the empty-first-element law
  assert len(bodies) == len(ks) == 14, (len(bodies), len(ks))
  for name, body in zip(ks, bodies):
    open(f"{BASE}/{name}.cu", "w").write(hdr + body.rstrip() + "\n")
    build(name)

  for n in ("lookup9_nw32", "accept9k", "acceptsel9k"):
    build(n)

  C100, S = 100352, 256
  build("spk_pre9qh_100k", src="spk_preqh", extra=("-DKPRE=spk_pre9qh_100k", f"-DCTXK={C100}", "-DROWS=9"))
  # hm9: ROWS=9 -> RMAX=54, RP=64 (10 pad rows, never stored/read); MAXOWN=2.
  _g = open(f"{BASE}/spk_g4hm.cu").read()
  _prag = chr(95) + chr(80) + 'ragma("' + "unroll" + '")' + chr(10) + "        for (int ks4 = 0; ks4 < QK_KS; ++ks4) {"
  assert _prag in _g
  open(f"{BASE}/.spk_g4hm_r9.cu", "w").write(_g.replace(_prag, "#pragma unroll 4" + chr(10) + "        for (int ks4 = 0; ks4 < QK_KS; ++ks4) {", 1))
  build("spk_g4nw32hm9_100k", src=".spk_g4hm_r9", extra=(f"-DKNAME=spk_g4nw32hm9_100k", f"-DCTXK={C100}", "-DROWS=9",
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
  open(f"{BASE}/spk_c9g_100k.cu", "w").write(_h + _b["K2S"])
  build("spk_c9g_100k", extra=("-DK2S=spk_c9g_100k", f"-DCTXK={C100}", "-DROWS=9", f"-DS={S}", f"-DCH={C100//S}", "-DUNROLL=4"))
  print("[build k9 done]")

if __name__ == "__main__":
  main()
