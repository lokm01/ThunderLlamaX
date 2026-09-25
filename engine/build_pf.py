# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P1 pGEMM builder: per-kernel cubins from pf_gemm.cu for every (class, shape)
combo in the sweep. Names carry the warp token (NAME-ENCODED LAUNCH CONFIG law:
nw8/nw16/nw32 -> 256/512/1024 threads; bare = 256) + k-tile + engine/engine-lite
tokens. Per-kernel cubin law + in-container cuobjdump symbol check."""
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

# (class_token, QCLASS, KDIM, NDIM, FFN) — dims from the trunk weight audit
CLASSES = {
  # IQ3_XXS packed: gate K=5120 N=6144; fg/fu K=5120 N=17408; fd K=17408 N=5120
  "iq3g":  (1, 5120, 6144, 0),
  "iq3n":  (1, 5120, 17408, 0),   # fg/fu shape
  "iq3d":  (1, 17408, 5120, 0),   # fd shape
  "iq3o":  (1, 6144, 5120, 0),    # ssm_out type-18 shape
  "iq3q":  (1, 5120, 12288, 0),   # attn q type-18 shape
  "iq3k":  (1, 5120, 1024, 0),    # attn k shape
  # Q5_K raw: qkv K=5120 N=10240 ; head K=5120 N=248320
  "q5kv":  (2, 5120, 10240, 0),
  "q5h":   (2, 5120, 248320, 0),
  # Q6_K packed: attn q K=5120 N=12288
  "q6q":   (3, 5120, 12288, 0),
  # Q4_K raw: attn v K=5120 N=1024
  "q4v":   (4, 5120, 1024, 0),
  # IQ3_S raw: attn o K=6144 N=5120
  "iq3s":  (5, 6144, 5120, 0),
  # Q8_0 raw: ssm_out type-8 K=6144 N=5120
  "q8o":   (6, 6144, 5120, 0),
  # Q4_0 draft packed: fg/fu K=5120 N=17408
  "q4dn":  (7, 5120, 17408, 0),
  # fused FFN gate+up (IQ3 packed): K=5120 N=17408, FFN epilogue
  "ffn":   (1, 5120, 17408, 1),
}

# sweep: (NTHR, NTILE, KCH) — NTILE = NTHR/32*8 always; smem combos kept legal
SHAPES = [
  (256, 64, 64),
  (256, 64, 128),
  (512, 128, 64),
  (512, 128, 128),
  (1024, 256, 64),
]  # LAW: NTILE == NTHR/32*8 (warp out-tile hardcoded 8 cols; NTILE=32 faults)

def build(env, name, qcls, kd, nd, ffn, nthr, ntile, kch, hmma):
  cmd = ["nvcc", "-arch=sm_86", "-cubin", f"-DKNAME={name}", f"-DQCLASS={qcls}",
         f"-DKDIM={kd}", f"-DNDIM={nd}", f"-DNTHR={nthr}", f"-DNTILE={ntile}",
         f"-DKCH={kch}", f"-DHMMA={hmma}", f"-DFFN={ffn}", "-Xptxas", "-v",
         f"--output-file={BASE}/{name}.cubin", f"{BASE}/pf_gemm.cu"]
  r = sh(cmd, env)
  if r.returncode:
    print(f"[build] {name} FAIL\n{r.stderr[-2500:]}"); return False
  regs = [l for l in (r.stderr or "").splitlines() if "registers" in l or "smem" in l.lower() or "spill" in l.lower()]
  d = sh(["docker", "exec", "cuda-nvcc-persistent", "cuobjdump", "-symbols", f"{BASE}/{name}.cubin"], env)
  if name not in [l.split()[-1] for l in (d.stdout or "").splitlines() if l.strip().endswith(name)]:
    print(f"[build] {name} SYMBOL CHECK FAIL:\n{d.stdout}\n{d.stderr}"); return False
  print(f"[build] {name} OK {' | '.join(regs[-3:])}", flush=True)
  return True

def main():
  env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
             DOCKER_HOST="unix://~/.colima/default/docker.sock")
  sh(["docker", "ps"], env)
  only = sys.argv[1:] if len(sys.argv) > 1 else None
  n_ok = n_fail = 0
  for ctok, (qcls, kd, nd, ffn) in CLASSES.items():
    for (nthr, ntile, kch) in SHAPES:
      # smem legality: xs 16*(kch+8)*2 + ws (ffn?2:1)*ntile*(kch+8)*2 <= 40000
      smem = 16*(kch+8)*2 + (2 if ffn else 1)*ntile*(kch+8)*2
      if smem > 40000: continue
      # fused FFN only on the 'ffn' token; plain only elsewhere
      for hmma, mtok in ((1, "hm"), (0, "hf")):
        nw = nthr // 32
        name = f"pfg_{ctok}_{mtok}_nw{nw}k{kch}"
        if only and not any(o in name for o in only): continue
        if build(env, name, qcls, kd, nd, ffn, nthr, ntile, kch, hmma): n_ok += 1
        else: n_fail += 1
  print(f"[build_pf done] ok={n_ok} fail={n_fail}", flush=True)
  sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
  main()
