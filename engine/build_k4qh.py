# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R4: ROWS=5 QH-path attention cubins. (1) patch spk_preqh.cu's ROWS==3
ternaries to the ROWS-generic form (bit-identical for ROWS in {1,3}, legal 5 —
the skv_split law); (2) build spk_pre5qh_100k + spk_g4nw32hm5_100k."""
import subprocess, os, sys, time, re
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

src = open(f"{BASE}/spk_preqh.cu").read()
for a, b in [("(ROWS==3 ? t*12288 : 0)", "(ROWS==1 ? 0 : t*12288)"),
             ("(ROWS==3 ? t*24 : 0)", "(ROWS==1 ? 0 : t*24)"),
             ("(ROWS==3 ? t*1024 : 0)", "(ROWS==1 ? 0 : t*1024)")]:
  assert src.count(a) in (1, 2), (a, src.count(a))
  src = src.replace(a, b)
src = src.replace("// -DKPRE -DCTXK -DROWS (3|1).", "// -DKPRE -DCTXK -DROWS (1|3|5 — R4-generic ternaries).")
open(f"{BASE}/spk_preqh.cu", "w").write(src)
print("[patch] spk_preqh.cu ROWS-generic ternaries OK", flush=True)

C100 = 100352
r = sh(["nvcc", "-arch=sm_86", "-cubin", "-DKPRE=spk_pre5qh_100k", f"-DCTXK={C100}", "-DROWS=5",
        f"--output-file={BASE}/spk_pre5qh_100k.cubin", f"{BASE}/spk_preqh.cu"], env)
if r.returncode: print(f"[build] spk_pre5qh_100k FAIL\n{r.stderr[-2000:]}"); sys.exit(1)
print("[build] spk_pre5qh_100k OK", flush=True)
nm = "spk_g4nw32hm5_100k"
r = sh(["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={nm}", f"-DCTXK={C100}", "-DROWS=5", "-DS=256",
        f"-DCH={C100//256}", "-DTILE=32", "-DNW=32", "-Xptxas", "-v",
        f"--output-file={BASE}/{nm}.cubin", f"{BASE}/spk_g4hm.cu"], env)
if r.returncode: print(f"[build] {nm} FAIL\n{r.stderr[-2000:]}"); sys.exit(1)
regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l or "spill" in l]
print(f"[build] {nm} OK {' | '.join(regs[-3:])}", flush=True)
print("[build k4qh done]", flush=True)
