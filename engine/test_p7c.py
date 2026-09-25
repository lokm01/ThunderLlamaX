# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-C validate+bench: the chunked delta-rule scan (FLA WY) vs the sequential
oracle. READOUT-ORDER LAW: gates from the first clean run; timing after.

Gates:
  1. single chunk (C=64 NC=1, one block): O/z/rec vs numpy fp64 token-loop
     oracle (pf_scan16 math VERBATIM) — O med relerr <= 1e-3 class.
  2. 64 tokens vs the REAL pf_scan16 cubin (4 chained 16-step launches):
     z + final rec relerr <= ~2e-3 class (fp16-reassociation dominated).
  3. 512 tokens (NC=8) vs 32 chained pf_scan16 launches: state accumulation
     drift budget rec <= 3e-2 (the P6 100k class), z med relerr report.
Bench: synced min-of-N; per-kernel breakdown per 512-token super-chunk per
block vs the sequential pfs16 control (same session).
Usage: ~/tg311/bin/python -u test_p7c.py [gate1|gate2|gate3|bench] (default all)
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
LS = (256, 1, 1)
P = Bufs()
_cache = {}
def prog(n):
  if n not in _cache:
    lib = open(f"{BASE}/{n}.cubin", "rb").read()
    _cache[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
  return _cache[n]

PF16 = None
WP_STRIDE_F = 47200  # floats per block: convw 40960 | dtb 48 | ssma 48 | snw 6144
HC_BYTES = {64: 263696, 32: 125712}  # R2b: hi-lo sizes (P7E5 193040 -> P7E6 263696; c32 125712) — the P7C-era dict was 2.5x undersized

def relerr(mine, ref, floor=1e-3):
  ref = np.asarray(ref); mine = np.asarray(mine)
  act = np.abs(ref) > floor * max(np.abs(ref).max(), 1e-9)
  e = np.abs(mine[act].astype(np.float64) - ref[act].astype(np.float64)) / np.abs(ref[act].astype(np.float64))
  return (float(np.median(e)) if e.size else 0.0,
          float(np.linalg.norm(mine.astype(np.float64) - ref.astype(np.float64)) / max(np.linalg.norm(ref.astype(np.float64)), 1e-9)))

# ---------------- numpy fp64 oracle (pf_scan16 math VERBATIM) ----------------
def scan_ref(qkv, gate, araw, braw, convw, dtb, ssma, snw, conv0, rec0):
  T = qkv.shape[0]
  convw = convw.reshape(10240, 4)
  conv_live = conv0.astype(np.float64).copy()
  S = rec0.astype(np.float64).reshape(48, 128, 128).copy()  # [h][v][k]
  O = np.zeros((T, 48 * 128), dtype=np.float64)
  Z = np.zeros((T, 6144), dtype=np.float16)
  snw_h = snw.astype(np.float64).reshape(48, 128)
  for t in range(T):
    rt = qkv[t].astype(np.float64)
    r3 = qkv[t - 3].astype(np.float64) if t >= 3 else conv_live[t + 0]
    r2 = qkv[t - 2].astype(np.float64) if t >= 2 else conv_live[t + 1]
    r1 = qkv[t - 1].astype(np.float64) if t >= 1 else conv_live[t + 2]
    for h in range(48):
      kh = h % 16
      qc0, kc0, vc0 = kh * 128, 2048 + kh * 128, 4096 + h * 128
      sq = r3[qc0:qc0+128]*convw[qc0:qc0+128,0] + r2[qc0:qc0+128]*convw[qc0:qc0+128,1] + r1[qc0:qc0+128]*convw[qc0:qc0+128,2] + rt[qc0:qc0+128]*convw[qc0:qc0+128,3]
      sk = r3[kc0:kc0+128]*convw[kc0:kc0+128,0] + r2[kc0:kc0+128]*convw[kc0:kc0+128,1] + r1[kc0:kc0+128]*convw[kc0:kc0+128,2] + rt[kc0:kc0+128]*convw[kc0:kc0+128,3]
      sv = r3[vc0:vc0+128]*convw[vc0:vc0+128,0] + r2[vc0:vc0+128]*convw[vc0:vc0+128,1] + r1[vc0:vc0+128]*convw[vc0:vc0+128,2] + rt[vc0:vc0+128]*convw[vc0:vc0+128,3]
      sq = sq * (1.0 / (1.0 + np.exp(-sq)))
      sk = sk * (1.0 / (1.0 + np.exp(-sk)))
      sv = sv * (1.0 / (1.0 + np.exp(-sv)))
      qn = (1.0 / max(np.sqrt((sq * sq).sum()), 1e-6)) * 0.08838834764831845
      kn = 1.0 / max(np.sqrt((sk * sk).sum()), 1e-6)
      qh_, khh, vh = sq * qn, sk * kn, sv
      x = float(araw[t, h] + dtb[h])
      sp = max(x, 0.0) + float(np.log1p(np.exp(-abs(x))))
      al = np.exp(sp * ssma[h])
      be = 1.0 / (1.0 + np.exp(-float(braw[t, h])))
      s = S[h] * al
      kd = s @ khh
      dl = (vh - kd) * be
      s = s + np.outer(dl, khh)
      S[h] = s
      O[t, h * 128:(h + 1) * 128] = s @ qh_
    core = O[t].reshape(48, 128)
    zz = (core * core).sum(axis=1, keepdims=True)
    rz = 1.0 / np.sqrt(zz / 128 + 1e-6)
    zc = core * rz * snw_h
    g = gate[t].reshape(48, 128).astype(np.float64)
    Z[t] = (zc * (g * (1.0 / (1.0 + np.exp(-g))))).reshape(6144).astype(np.float16)  # pfs16 gate = g*sigmoid(g)
  convf = qkv[T - 3:T].astype(np.float32).reshape(-1).copy()
  return O, Z, S, convf

# ---------------- inputs ----------------
def make_inputs(T, seed):
  rng = np.random.default_rng(seed)
  return dict(
    qkv=(rng.standard_normal((T, 10240)) * 0.9).astype(np.float16),
    gate=(rng.standard_normal((T, 6144)) * 0.8).astype(np.float16),
    araw=(rng.standard_normal((T, 48)) * 0.7).astype(np.float32),
    braw=(rng.standard_normal((T, 48)) * 0.7).astype(np.float32),
    convw=(rng.standard_normal((10240, 4)) * 0.25).astype(np.float32),
    dtb=(rng.standard_normal(48) * 0.5).astype(np.float32),
    ssma=(-rng.uniform(0.15, 3.5, 48)).astype(np.float32),
    snw=(1.0 + rng.standard_normal(6144) * 0.08).astype(np.float32),
    conv0=(rng.standard_normal((3, 10240)) * 0.9).astype(np.float32),
    rec0=(rng.standard_normal(48 * 128 * 128) * 0.25).astype(np.float32),
  )

def up_weights(P, inp):
  plane = np.zeros(WP_STRIDE_F, dtype=np.float32)
  plane[0:40960] = inp["convw"].reshape(-1)
  plane[40960:41008] = inp["dtb"]
  plane[41008:41056] = inp["ssma"]
  plane[41056:47200] = inp["snw"]
  P.up("wp0", plane)

def up_run_bufs(P, tag, T, inp):
  scr = 48 * HC_BYTES[64]  # NC chunks worth is allocated by caller override
  P.up(f"{tag}_qkv", inp["qkv"].reshape(-1))
  P.up(f"{tag}_gate", inp["gate"].reshape(-1))
  P.up(f"{tag}_ar", inp["araw"].reshape(-1))
  P.up(f"{tag}_br", inp["braw"].reshape(-1))
  P.up(f"{tag}_conv", inp["conv0"].reshape(-1))
  P.up(f"{tag}_rec", inp["rec0"])
  P.poison(f"{tag}_z", T * 6144 * 2, np.float16, 7.7)
  for nm in ("sq16", "sk16", "sv16", "sc16"): P.poison(f"{tag}_{nm}", 48 * 128 * 4, np.float32, 7.7e31)
  dev.synchronize(); P._keep.clear()

def run_mine(P, tag, C, NC, T, inp):
  P.poison(f"{tag}_scr", 48 * NC * HC_BYTES[C], np.uint8, 0xAB)
  P.poison(f"{tag}_o", T * 6144 * 4, np.float32, 7.7e31)
  dev.synchronize(); P._keep.clear()
  ca, cb, cz = prog(f"pfca_c{C}_nc{NC}_nw16"), prog(f"pfcb_c{C}_nc{NC}_nw8"), prog(f"pfcz_c{C}_nc{NC}_nw8")
  ca(P.d["wp0"], P.d[f"{tag}_conv"], P.d[f"{tag}_qkv"], P.d[f"{tag}_ar"], P.d[f"{tag}_br"], P.d[f"{tag}_scr"],
     global_size=(48 * NC, 1, 1), local_size=(512, 1, 1))
  cb(P.d[f"{tag}_scr"], P.d[f"{tag}_rec"], P.d[f"{tag}_o"], global_size=(192, 1, 1), local_size=LS)
  cz(P.d[f"{tag}_o"], P.d[f"{tag}_gate"], P.d["wp0"].offset(offset=41056 * 4, size=6144 * 4),
     P.d[f"{tag}_z"], P.d[f"{tag}_qkv"], P.d[f"{tag}_conv"], global_size=(48 * NC * (C // 8), 1, 1), local_size=LS)
  dev.synchronize()
  return (P.down(f"{tag}_o", (T, 6144), np.float32).copy(),
          P.down(f"{tag}_z", (T, 6144), np.float16).copy(),
          P.down(f"{tag}_rec", (48 * 128 * 128,), np.float32).copy(),
          P.down(f"{tag}_conv", (3 * 10240,), np.float32).copy())

def run_seq16(P, tag, T, inp):
  """Chained pf_scan16 (the REAL cubin) over T tokens."""
  PF16 = prog("pfs16")
  dtbb = P.d["wp0"].offset(offset=40960 * 4, size=48 * 4)
  ssmab = P.d["wp0"].offset(offset=41008 * 4, size=48 * 4)
  snwb = P.d["wp0"].offset(offset=41056 * 4, size=6144 * 4)
  for t0 in range(0, T, 16):
    PF16(P.d[f"{tag}_conv"], P.d[f"{tag}_rec"],
         P.d[f"{tag}_qkv"].offset(offset=t0 * 10240 * 2, size=16 * 10240 * 2),
         P.d[f"{tag}_gate"].offset(offset=t0 * 6144 * 2, size=16 * 6144 * 2),
         P.d["wp0"], dtbb, ssmab,
         P.d[f"{tag}_ar"].offset(offset=t0 * 48 * 4, size=16 * 48 * 4),
         P.d[f"{tag}_br"].offset(offset=t0 * 48 * 4, size=16 * 48 * 4),
         P.d[f"{tag}_sq16"], P.d[f"{tag}_sk16"], P.d[f"{tag}_sv16"], P.d[f"{tag}_sc16"], snwb,
         P.d[f"{tag}_z"].offset(offset=t0 * 6144 * 2, size=16 * 6144 * 2),
         global_size=(48, 1, 1), local_size=LS)
  dev.synchronize()
  return (P.down(f"{tag}_z", (T, 6144), np.float16).copy(),
          P.down(f"{tag}_rec", (48 * 128 * 128,), np.float32).copy(),
          P.down(f"{tag}_conv", (3 * 10240,), np.float32).copy())

# ============================== main ==============================
if __name__ == "__main__":
  arg = next((a for a in sys.argv[1:] if a in ("gate1", "gate2", "gate3", "bench")), None)

  if arg in (None, "gate1"):
    t0 = time.perf_counter()
    inp = make_inputs(64, 11)
    up_weights(P, inp)
    O_ref, Z_ref, S_ref, convf_ref = scan_ref(inp["qkv"], inp["gate"], inp["araw"], inp["braw"],
                                              inp["convw"], inp["dtb"], inp["ssma"], inp["snw"],
                                              inp["conv0"], inp["rec0"])
    print(f"[g1] fp64 oracle done in {time.perf_counter()-t0:.1f}s", flush=True)
    up_run_bufs(P, "g1", 64, inp)
    O_m, Z_m, rec_m, conv_m = run_mine(P, "g1", 64, 1, 64, inp)
    mo, fo = relerr(O_m, O_ref); mz, fz = relerr(Z_m, Z_ref)
    mr, fr = relerr(rec_m, S_ref.reshape(-1))
    cv = float(np.abs(conv_m - convf_ref).max())
    print(f"[g1] O: med {mo:.3e} F {fo:.3e} | z: med {mz:.3e} F {fz:.3e} | rec: med {mr:.3e} F {fr:.3e} | conv maxdiff {cv:.1e}", flush=True)
    print(f"[g1] {'PASS' if mo <= 1e-3 and mr <= 2e-3 else 'FAIL'} (O med<=1e-3, rec<=2e-3)", flush=True)

  if arg in (None, "gate2"):
    inp = make_inputs(64, 22)
    up_weights(P, inp)
    up_run_bufs(P, "g2s", 64, inp)
    Z_s, rec_s, conv_s = run_seq16(P, "g2s", 64, inp)
    up_run_bufs(P, "g2m", 64, inp)
    Z_m, rec_m, conv_m = run_mine(P, "g2m", 64, 1, 64, inp)[1:]
    mz, fz = relerr(Z_m, Z_s, floor=0.05)
    mr, fr = relerr(rec_m, rec_s)
    cv = float(np.abs(conv_m - conv_s).max())
    print(f"[g2] z: med {mz:.3e} F {fz:.3e} | rec: med {mr:.3e} F {fr:.3e} | conv maxdiff {cv:.1e}", flush=True)
    print(f"[g2] {'PASS' if mr <= 2e-3 and mz <= 4e-3 else 'FAIL'} (rec<=2e-3, z<=4e-3)", flush=True)

  if arg in (None, "gate3"):
    inp = make_inputs(512, 33)
    up_weights(P, inp)
    up_run_bufs(P, "g3m", 512, inp)
    Z_m, rec_m, conv_m = run_mine(P, "g3m", 64, 8, 512, inp)[1:]
    up_run_bufs(P, "g3s", 512, inp)
    Z_s, rec_s, conv_s = run_seq16(P, "g3s", 512, inp)
    mz, fz = relerr(Z_m, Z_s, floor=0.05)
    mr, fr = relerr(rec_m, rec_s)
    cv = float(np.abs(conv_m - conv_s).max())
    print(f"[g3] z: med {mz:.3e} F {fz:.3e} | rec: med {mr:.3e} F {fr:.3e} | conv maxdiff {cv:.1e}", flush=True)
    print(f"[g3] {'PASS' if mr <= 3e-2 and mz <= 8e-3 else 'FAIL'} (rec<=3e-2 drift budget, z med<=8e-3)", flush=True)

  if arg in (None, "bench"):
    inp = make_inputs(512, 33)
    up_weights(P, inp)
    up_run_bufs(P, "b", 512, inp)
    P.poison("b_scr", 48 * 8 * HC_BYTES[64], np.uint8, 0xAB)
    P.poison("b_o", 512 * 6144 * 4, np.float32, 7.7e31)
    dev.synchronize(); P._keep.clear()
    ca, cb, cz = prog("pfca_c64_nc8_nw16"), prog("pfcb_c64_nc8_nw8"), prog("pfcz_c64_nc8_nw8")
    snwb = P.d["wp0"].offset(offset=41056 * 4, size=6144 * 4)
    dtbb = P.d["wp0"].offset(offset=40960 * 4, size=48 * 4)
    ssmab = P.d["wp0"].offset(offset=41008 * 4, size=48 * 4)
    def k_ca(): ca(P.d["wp0"], P.d["b_conv"], P.d["b_qkv"], P.d["b_ar"], P.d["b_br"], P.d["b_scr"], global_size=(384, 1, 1), local_size=(512, 1, 1))
    def k_cb(): cb(P.d["b_scr"], P.d["b_rec"], P.d["b_o"], global_size=(192, 1, 1), local_size=LS)
    def k_cz(): cz(P.d["b_o"], P.d["b_gate"], snwb, P.d["b_z"], P.d["b_qkv"], P.d["b_conv"], global_size=(384 * 8, 1, 1), local_size=LS)
    def one_block_chunked(): k_ca(); k_cb(); k_cz()
    def one_block_seq():
      for t0 in range(0, 512, 16):
        PF16(P.d["b_conv"], P.d["b_rec"],
             P.d["b_qkv"].offset(offset=t0 * 10240 * 2, size=16 * 10240 * 2),
             P.d["b_gate"].offset(offset=t0 * 6144 * 2, size=16 * 6144 * 2),
             P.d["wp0"], dtbb, ssmab,
             P.d["b_ar"].offset(offset=t0 * 48 * 4, size=16 * 48 * 4),
             P.d["b_br"].offset(offset=t0 * 48 * 4, size=16 * 48 * 4),
             P.d["b_sq16"], P.d["b_sk16"], P.d["b_sv16"], P.d["b_sc16"], snwb,
             P.d["b_z"].offset(offset=t0 * 6144 * 2, size=16 * 6144 * 2), global_size=(48, 1, 1), local_size=LS)
    PF16 = prog("pfs16")
    res = {}
    for nm, fn in (("pfca", k_ca), ("pfcb", k_cb), ("pfcz", k_cz)):
      fn(); dev.synchronize()
      best = 1e9
      for _ in range(10):
        t0 = time.perf_counter(); fn(); dev.synchronize(); best = min(best, time.perf_counter() - t0)
      res[nm] = best
    print(f"[bench] per-block (8x64 chunks): " + " ".join(f"{k} {v*1e3:.3f}ms" for k, v in res.items()), flush=True)
    tot = sum(res.values()) * 48
    print(f"[bench] CHUNKED 512-tok super-chunk x48 GDN blocks: {tot*1e3:.1f} ms (sum-of-kernels)", flush=True)
    best = 1e9
    for _ in range(3):
      t0 = time.perf_counter()
      for _bl in range(48): one_block_seq()
      dev.synchronize(); best = min(best, time.perf_counter() - t0)
    print(f"[bench] SEQUENTIAL control (pfs16, 48 blocks x 32 launches): {best*1e3:.1f} ms -> speedup {best/tot:.1f}x", flush=True)
    best = 1e9
    for _ in range(5):
      t0 = time.perf_counter()
      for _bl in range(48): one_block_chunked()
      dev.synchronize(); best = min(best, time.perf_counter() - t0)
    print(f"[bench] CHUNKED end-to-end (48 blocks x 3 launches): {best*1e3:.1f} ms", flush=True)
