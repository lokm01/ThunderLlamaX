# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P5 builder: FFN/IQ3 restructured pGEMM cubins (pf_gemm.cu SYNCW/SUB/H2
paths). Same docker-nvcc + symbol-check pattern as build_pf2. Names carry the
plan: h2 = LUT decode, sw = SYNCW, sN = SUB subtiles/warp.
Laws: per-kernel cubin + warp-token names; hardcoded sizes; smem combos
manually vetted <= 44KB."""
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

def build(env, name, extra):
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={name}"] + extra + \
        ["-Xptxas", "-v", f"--output-file={BASE}/{name}.cubin", f"{BASE}/pf_gemm.cu"]
  r = sh(cmd, env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-2500:]}"); return False
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l.lower() or "spill" in l.lower()]
  d = sh(["docker", "exec", "cuda-nvcc-persistent", "cuobjdump", "-symbols", f"{BASE}/{name}.cubin"], env)
  if name not in [l.split()[-1] for l in (d.stdout or "").splitlines() if l.strip().endswith(name)]:
    print(f"[build] {name} SYMBOL CHECK FAIL:\n{d.stdout}\n{d.stderr}"); return False
  print(f"[build] {name} OK {' | '.join(regs[-3:])}", flush=True)
  return True

FFN = ["-DQCLASS=1", "-DKDIM=5120", "-DNDIM=17408", "-DHMMA=1", "-DFFN=1", "-DRES=0"]
# (name, class-dims, NTHR, NTILE, KCH, SUB, H2)
TARGETS = [
  # FFN family (fused gate+up, x64 instances/chunk)
  ("pfg_ffnX_sw_nw8k128",   FFN, 256, 64, 128, 1, 0),   # bit-identical control (SYNCW only)
  ("pfg_ffnX_h2sw_nw8k128", FFN, 256, 64, 128, 1, 1),   # direct swap, same 272 CTAs
  ("pfg_ffnX_h2sw_nw8k64s2", FFN, 256, 128, 64, 2, 1),  # 136 CTAs (2 waves)
  ("pfg_ffnX_h2sw_nw8k32s4", FFN, 256, 256, 32, 4, 1),  # 68 CTAs (1 wave), 256thr
  ("pfg_ffnX_h2sw_nw32k32",  FFN, 1024, 256, 32, 1, 1), # 68 CTAs, 1024thr fat CTA
  ("pfg_ffnX_h2sw_nw16k64",  FFN, 512, 128, 64, 1, 1),  # 136 CTAs, 512thr
  # down-projection RES (x64)
  ("pfg_iq3dX_h2sw_nw8k128", ["-DQCLASS=1", "-DKDIM=17408", "-DNDIM=5120", "-DHMMA=1", "-DFFN=0", "-DRES=1"], 256, 64, 128, 1, 1),
  # attn q iq3 (x8)
  ("pfg_iq3qX_h2sw_nw8k128", ["-DQCLASS=1", "-DKDIM=5120", "-DNDIM=12288", "-DHMMA=1", "-DFFN=0", "-DRES=0"], 256, 64, 128, 1, 1),
  # ssm_out iq3 (x24)
  ("pfg_iq3oX_h2sw_nw8k128", ["-DQCLASS=1", "-DKDIM=6144", "-DNDIM=5120", "-DHMMA=1", "-DFFN=0", "-DRES=0"], 256, 64, 128, 1, 1),
  # CLASSIC structure + H2 LUT decode (no SYNCW) — the decode-only ablation
  ("pfg_ffnX_h2_nw8k128",   FFN, 256, 64, 128, 1, 1),
  ("pfg_iq3dX_h2_nw8k128",  ["-DQCLASS=1", "-DKDIM=17408", "-DNDIM=5120", "-DHMMA=1", "-DFFN=0", "-DRES=1"], 256, 64, 128, 1, 1),
  ("pfg_iq3qX_h2_nw8k128",  ["-DQCLASS=1", "-DKDIM=5120", "-DNDIM=12288", "-DHMMA=1", "-DFFN=0", "-DRES=0"], 256, 64, 128, 1, 1),
  ("pfg_iq3oX_h2_nw8k128",  ["-DQCLASS=1", "-DKDIM=6144", "-DNDIM=5120", "-DHMMA=1", "-DFFN=0", "-DRES=0"], 256, 64, 128, 1, 1),
]

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  only = sys.argv[1:] if len(sys.argv) > 1 else None
  n_ok = n_fail = 0
  for name, cls, nthr, ntile, kch, sub, h2 in TARGETS:
    if only and not any(o in name for o in only): continue
    extra = cls + [f"-DNTHR={nthr}", f"-DNTILE={ntile}", f"-DKCH={kch}",
                   f"-DSYNCW={'0' if '_h2_' in name else '1'}", f"-DSUB={sub}", f"-DH2={h2}"]
    smem = 2*16*(kch+8)*2 + (2 if "-DFFN=1" in cls else 1)*ntile*(kch+8)*2
    assert smem <= 45056, f"{name} smem {smem} over 44KB"
    if build(env, name, extra): n_ok += 1
    else: n_fail += 1
  print(f"[build_p5 done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
