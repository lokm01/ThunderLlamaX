# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P13 builder: pf13_stream (kernel A mechanism microbench) + pf13_ffn
(kernel B persistent-CTA FFN tile) cubins. Same docker nvcc + symbol-check
pattern as build_p7b.py. Per-kernel cubin law; entry name == cubin name."""
import subprocess, os, sys, time
BASE = os.path.dirname(os.path.abspath(__file__))

def sh(cmd, env, tries=4):
  r = None
  for t in range(tries):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode == 0 or "failed to connect to the docker API" not in r.stderr: return r
    print(f"[build] transient docker API failure (try {t+1}), retrying...", flush=True)
    time.sleep(2)
  return r

def build(env, name, src, extra, kname):
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={kname}"] + extra + \
        ["-Xptxas", "-v", f"--output-file={BASE}/{name}.cubin", f"{BASE}/{src}"]
  r = sh(cmd, env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-2500:]}"); return False
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l.lower() or "spill" in l.lower()]
  d = sh(["docker", "exec", "cuda-nvcc-persistent", "cuobjdump", "-symbols", f"{BASE}/{name}.cubin"], env)
  if kname not in [l.split()[-1] for l in (d.stdout or "").splitlines() if l.strip().endswith(kname)]:
    print(f"[build] {name} SYMBOL CHECK FAIL:\n{d.stdout}\n{d.stderr}"); return False
  print(f"[build] {name} OK {' | '.join(regs[-3:])}", flush=True)
  return True

TARGETS = [
  # ---- kernel A: pure per-SM stream (grid literal NCTA; ring RINGD) ----
  ("pf13s_nw8d4",   "pf13_stream.cu", ["-DNTHR=256", "-DRINGD=4", "-DNCTA=82"]),
  ("pf13s_nw8d8",   "pf13_stream.cu", ["-DNTHR=256", "-DRINGD=8", "-DNCTA=82"]),
  ("pf13s_nw8d16",  "pf13_stream.cu", ["-DNTHR=256", "-DRINGD=16", "-DNCTA=82"]),
  ("pf13s_nw16d8",  "pf13_stream.cu", ["-DNTHR=512", "-DRINGD=8", "-DNCTA=82"]),
  ("pf13s_nw32d4",  "pf13_stream.cu", ["-DNTHR=1024", "-DRINGD=4", "-DNCTA=82"]),
  ("pf13s272_nw8d8","pf13_stream.cu", ["-DNTHR=256", "-DRINGD=8", "-DNCTA=272"]),  # the 3.3-wave reference
  # ---- kernel B: persistent FFN (m32 drop-in shapes; WRING register ring) ----
  ("pf13ffn_w2_nw8", "pf13_ffn.cu", ["-DWRING=2", "-DNBLK=8", "-DKDIM=5120", "-DNDIM=17408",
                                     "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=32"]),
  ("pf13ffn_w4_nw8", "pf13_ffn.cu", ["-DWRING=4", "-DNBLK=8", "-DKDIM=5120", "-DNDIM=17408",
                                     "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=32",
                                     "-Xptxas", "-maxrregcount=176"]),   # regs free at grid 82 (1 CTA/SM)
  ("pf13ffn_w8_nw8", "pf13_ffn.cu", ["-DWRING=8", "-DNBLK=8", "-DKDIM=5120", "-DNDIM=17408",
                                     "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=32"]),
  # P14: NBLK=1 in-plan drop-ins (per-block persistent launches; the model
  # blocks are sequential so one launch spans ONE block = drop-in for the r7 ffn line)
  ("pf13ffn_w2_n1", "pf13_ffn.cu", ["-DWRING=2", "-DNBLK=1", "-DKDIM=5120", "-DNDIM=17408",
                                     "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=32"]),
  ("pf13ffn_w4_n1", "pf13_ffn.cu", ["-DWRING=4", "-DNBLK=1", "-DKDIM=5120", "-DNDIM=17408",
                                     "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=32",
                                     "-Xptxas", "-maxrregcount=176"]),
  ("pf13ffn_w8_n1", "pf13_ffn.cu", ["-DWRING=8", "-DNBLK=1", "-DKDIM=5120", "-DNDIM=17408",
                                     "-DNTHR=256", "-DNTILE=64", "-DKCH=128", "-DMTILE=32"]),
]

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  only = sys.argv[1:] if len(sys.argv) > 1 else None
  n_ok = n_fail = 0
  for name, src, extra in TARGETS:
    if only and not any(o in name for o in only): continue
    if build(env, name, src, extra, name): n_ok += 1
    else: n_fail += 1
  print(f"[build_p13 done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
