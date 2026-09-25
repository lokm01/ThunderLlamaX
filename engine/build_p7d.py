# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-D builder: pf_attn2.cu (pfaW/pfcW widened attention) + pf_kpre64.cu.
Same docker nvcc + symbol-check pattern as build_p7c.py. Warp-token names.
Register law: R192/R144 run NW=16 (512thr) — acc 96/72 regs + ~25 working;
the 64-reg wall at 1024thr rules out NW=32 for R>~124 (see P7DE doc).
smem @R192/T16: 29920B (in-graph legal); @R144: 23744B.
"""
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
    print(f"[build] {name} FAIL\n{r.stderr[-3000:]}"); return False
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l.lower() or "spill" in l.lower()]
  d = sh(["docker", "exec", "cuda-nvcc-persistent", "cuobjdump", "-symbols", f"{BASE}/{name}.cubin"], env)
  if kname not in [l.split()[-1] for l in (d.stdout or "").splitlines() if l.strip().endswith(kname)]:
    print(f"[build] {name} SYMBOL CHECK FAIL:\n{d.stdout}\n{d.stderr}"); return False
  print(f"[build] {name} OK {' | '.join(regs[-3:])}", flush=True)
  return True

def attn(name, ksel, rows, s, nw=16, tile=16):
  return (name, "pf_attn2.cu", [f"-DKSEL={ksel}", f"-DROWS={rows}", f"-DTILE={tile}", f"-DNW={nw}",
          f"-DS={s}", "-DCTXK=100352"], name.split("_")[0])

def kpre(name, trows=64):
  return (name, "pf_kpre64.cu", ["-DCTXK=100352", f"-DTROWS={trows}"], "pfk_pre64")

TARGETS = [
  # K1 widened tiers (S sweep on R192; R144 + R96 structure control at S=32)
  attn("pfa32nw16_s32_100k", 1, 32, 32),
  attn("pfa32nw16_s64_100k", 1, 32, 64),
  attn("pfa32nw16_s128_100k", 1, 32, 128),
  attn("pfa32nw16_s256_100k", 1, 32, 256),
  attn("pfa32nw16_s8_100k", 1, 32, 8),
  attn("pfa24nw16_s32_100k", 1, 24, 32),
  attn("pfa16ctl_nw16_s32_100k", 1, 16, 32),
  # K2 combine (ROWS/S must match the K1 tier)
  attn("pfc32_s32", 2, 32, 32),
  attn("pfc32_s64", 2, 32, 64),
  attn("pfc32_s128", 2, 32, 128),
  attn("pfc32_s256", 2, 32, 256),
  attn("pfc32_s8", 2, 32, 8),
  attn("pfc24_s32", 2, 24, 32),
  attn("pfc16ctl_s32", 2, 16, 32),
  # 64-row kpre
  kpre("pfk_pre64_100k"),
]

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  only = sys.argv[1:] if len(sys.argv) > 1 else None
  ok = True
  for name, src, extra, kname in TARGETS:
    if only and not any(o in name for o in only): continue
    ok &= build(env, name, src, extra, kname)
  sys.exit(0 if ok else 1)

if __name__ == "__main__":
  main()
