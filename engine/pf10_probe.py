# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P10 probe: the t32 ship set standalone validation + bench.
corr  = 2k-class correctness vs the pfa16r_s8_2k reference + numpy gate truth:
        (a) pfa32ct + pfc16t combine  (b) pfa32ctl (fused last-CTA combine).
        Determinism x2 (the LC arm re-launches with the SAME ctr buffer ->
        exercises the self-reset). Poison-first.
bench = synced min-of-5 @pos=100336, CFG=100 via AUTO_NAMES=pfg,pfa32c:
        t32 / t32+pfc16t pair / t32ctl fused / pfa16 shipped ref.
Run: cd ~/tinygrad-metal/engine0 && <FULL env> ~/tg311/bin/python -u pf10_probe.py corr|bench
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad.device import Device, TinyELF
from tinygrad import dtypes
from tinygrad.runtime.ops_nv import NVProgram
from engine0 import Bufs

BASE = "~/tinygrad-metal/engine0"
dev = Device["NV"]
P = Bufs()
mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
CTXK = 2048 if mode == "corr" else 100352
rng = np.random.default_rng(11)
print(f"[env] AUTO={os.getenv('NV_SMEM_CFG_AUTO','-')} ANAMES={os.getenv('NV_SMEM_CFG_AUTO_NAMES','-')} "
      f"TGT={os.getenv('NV_SMEM_CFG_AUTO_TGT','-')} mode={mode} CTXK={CTXK}", flush=True)

_pc = {}
def prog(cub, ent):
  if (cub, ent) in _pc: return _pc[(cub, ent)]
  lib = open(f"{BASE}/{cub}.cubin", "rb").read()
  _pc[(cub, ent)] = NVProgram(dev, TinyELF(lib=lib, name=ent, target=dev.renderer.target, signature=tuple()))
  return _pc[(cub, ent)]

S13, NHP3, RMAX32, NTHRA, NCTA = 13, 3, 32, 512, 4*13*3
def run_t32(pos, fused):
  gs = NCTA
  P.poison("pm", gs*RMAX32*4, np.float32, 7.7e31); P.poison("ps", gs*RMAX32*4, np.float32, 7.7e31)
  P.poison("pA", gs*RMAX32*256*4, np.float32, 7.7e31)
  P.win_up("pos_slot", 0, np.array([pos], dtype=np.int32)); dev.synchronize()
  pk = prog(f"pfa32ctl_s13_{'2k' if mode=='corr' else '100k'}", "pfa32ctl") if fused else \
       prog(f"pfa32c_t32_s13_{'2k' if mode=='corr' else '100k'}", "pfa32ct")
  if fused:
    P.win_up("ao16", 0, np.full(16*6144, 7.7, dtype=np.float16)); dev.synchronize()
    pk(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
       P.d["qrow16"], P.d["ao16"], P.d["atc_ctr"], global_size=(gs,1,1), local_size=(NTHRA,1,1))
    dev.synchronize()
    return P.down("ao16", (16, 6144), np.float16).astype(np.float32).copy()
  pk(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
     global_size=(gs,1,1), local_size=(NTHRA,1,1))
  dev.synchronize()
  pm = P.down("pm", (gs, RMAX32), np.float32).copy()
  ps = P.down("ps", (gs, RMAX32), np.float32).copy()
  pA = P.down("pA", (gs, RMAX32, 256), np.float32).copy()
  return pm, ps, pA

def combine_t32(pm, ps, pA, qrow):
  gs = pm.shape[0]
  rows = np.zeros((gs, RMAX32), dtype=np.int64)
  for bx in range(gs):
    hp = bx % NHP3; s = (bx // NHP3) % S13; g = bx // (NHP3*S13)
    for r in range(RMAX32): rows[bx, r] = (r % 16)*24 + g*6 + hp*2 + r//16
  m = np.full((384, gs), -1e30, dtype=np.float32); nu = np.zeros((384, gs), dtype=np.float32)
  O = np.zeros((384, gs, 256), dtype=np.float32)
  for bx in range(gs):
    m[rows[bx], bx] = pm[bx]; nu[rows[bx], bx] = ps[bx]; O[rows[bx], bx] = pA[bx]
  gmax = m.max(axis=1, keepdims=True)
  with np.errstate(under="ignore"): w = np.exp(m - gmax)
  out = (O * w[:, :, None]).sum(axis=1); den = (nu * w).sum(axis=1)
  out = out / np.maximum(den, 1e-30)[:, None]
  # sigmoid gate epilogue (pfc16 math), ao[t, h*256+d]
  qr = qrow.reshape(16, 12288).astype(np.float32)
  ao = np.zeros((16, 6144), dtype=np.float32)
  _tt = np.arange(16)
  for h in range(24):
    gf = qr[:, h*512+256:(h+1)*512]
    sg = 1.0 / (1.0 + np.exp(-gf))
    ao[:, h*256:(h+1)*256] = out[h + 24*_tt] * sg   # q-row = h + 24*t (t-major)
  return ao

def run_pfc16t(pm, ps, pA, qrow):
  P.up("pm2", pm); P.up("ps2", ps); P.up("pA2", pA.reshape(-1)); P.up("qrow2", qrow)
  P.win_up("ao16", 0, np.full(16*6144, 7.7, dtype=np.float16)); dev.synchronize()
  prog("pfc16t_s13", "pfc16t")(P.d["pm2"], P.d["ps2"], P.d["pA2"], P.d["qrow2"], P.d["ao16"],
                               global_size=(24,1,1), local_size=(256,1,1))
  dev.synchronize()
  o = P.down("ao16", (16, 6144), np.float16).astype(np.float32).copy()
  P._keep.clear()
  return o

def mapping_probe():
  """Reveal which partial slot the pfc16t kernel actually reads per (h,t):
  pm=0 everywhere, ps=1, pA[pb,d]=pb -> ao = mean_s2(slot(h,t,s2)) * sigmoid."""
  gs = NCTA
  n_slot = gs*RMAX32
  pm = np.zeros((gs, RMAX32), np.float32); ps = np.ones((gs, RMAX32), np.float32)
  pA = np.repeat(np.arange(n_slot, dtype=np.float32)[:, None], 256, axis=1).reshape(gs, RMAX32, 256)
  q1 = np.full((16, 12288), 50.0, np.float16)   # sigmoid ~ 1
  ao = run_pfc16t(pm, ps, pA, q1)
  # candidates for slot(h,t,s2) = (g*39 + s2*X + Y)*32 + Z*16 + t
  for name, X, Y, Z in [("A hp", 3, "hp", "hli"), ("B hli", 3, "hli", "hp"), ("D s/hp", 1, "s", "hp")]:
    pred = np.zeros((16, 24), np.float32)
    for h in range(24):
      g, hl = h//6, h%6; hp, hli = hl//2, hl%2
      for t in range(16):
        ss = []
        for s2 in range(S13):
          y = {"hp": hp, "hli": hli, "s": s2}[Y]; z = {"hp": hp, "hli": hli}[Z]
          ss.append((g*(NHP3*S13) + s2*X + y)*RMAX32 + z*16 + t)
        pred[t, h] = np.mean(ss)
    err = np.abs(ao[:, :24*256].reshape(16, 24, 256).mean(axis=2) - pred)
    print(f"[map] cand {name}: med {np.median(err):.3f} max {err.max():.1f}", flush=True)

# ---- world buffers (real-scale synthetic, P7F2 distributions) ----
kv = np.clip(rng.integers(0, 256, (2*4*CTXK*256,)).astype(np.uint8), 0, 255)
P.poison("kv", 2*4*CTXK*256, np.uint8, 200); P.up("kv", kv)
sc = (rng.uniform(0.001, 0.02, (2*4*CTXK*8))).astype(np.float16)
P.poison("sc", 2*4*CTXK*8*2, np.float16, np.float16(7.7)); P.up("sc", sc)
qw = (rng.standard_normal(32*24*256) * 0.5).astype(np.float16)
P.poison("qw", 32*24*256*2, np.float16, np.float16(7.7)); P.up("qw", qw)
qrow = (rng.standard_normal(16*12288) * 0.3).astype(np.float16)
P.poison("qrow16", 16*12288*2, np.float16, np.float16(7.7)); P.up("qrow16", qrow)
P.poison("ao16", 16*6144*2, np.float16, np.float16(7.7))
P.poison("atc_ctr", 4, np.uint32, 0xFF)
P.up("atc_ctr", np.zeros(1, dtype=np.uint32))   # LC ticket starts at 0 (self-reset keeps it 0)
P.poison("pos_slot", 4, np.int32, -1)
P._keep.clear(); dev.synchronize()
qrow = qrow.reshape(16, 12288)  # keep numpy view after device copy

pos = int(os.getenv("POS", "2032" if mode == "corr" else "100336"))
if mode == "corr":
  # reference: pfa16r_s8_2k partials -> numpy combine + gate
  S8, RMAX96 = 8, 96
  gs8 = 4*S8
  P.poison("pm", gs8*RMAX96*4, np.float32, 7.7e31); P.poison("ps", gs8*RMAX96*4, np.float32, 7.7e31)
  P.poison("pA", gs8*RMAX96*256*4, np.float32, 7.7e31)
  P.win_up("pos_slot", 0, np.array([pos], dtype=np.int32)); dev.synchronize()
  prog("pfa16r_s8_2k", "pfa16r")(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
                                 global_size=(gs8,1,1), local_size=(1024,1,1))
  dev.synchronize()
  pm8 = P.down("pm", (gs8, RMAX96), np.float32).copy(); ps8 = P.down("ps", (gs8, RMAX96), np.float32).copy()
  pA8 = P.down("pA", (gs8, RMAX96, 256), np.float32).copy()
  rows8 = np.zeros((gs8, RMAX96), dtype=np.int64)
  for bx in range(gs8):
    g, s = bx // S8, bx % S8
    for r in range(RMAX96): rows8[bx, r] = (r % 16)*24 + g*6 + r//16
  m = np.full((384, gs8), -1e30, dtype=np.float32); nu = np.zeros((384, gs8), dtype=np.float32)
  O = np.zeros((384, gs8, 256), dtype=np.float32)
  for bx in range(gs8):
    m[rows8[bx], bx] = pm8[bx]; nu[rows8[bx], bx] = ps8[bx]; O[rows8[bx], bx] = pA8[bx]
  gmax = m.max(axis=1, keepdims=True)
  with np.errstate(under="ignore"): w = np.exp(m - gmax)
  out8 = (O * w[:, :, None]).sum(axis=1); den8 = (nu * w).sum(axis=1)
  ref = out8 / np.maximum(den8, 1e-30)[:, None]
  qr = qrow.astype(np.float32)
  ref_ao = np.zeros((16, 6144), dtype=np.float32)
  _tt = np.arange(16)
  for h in range(24):
    sg = 1.0 / (1.0 + np.exp(-qr[:, h*512+256:(h+1)*512]))
    ref_ao[:, h*256:(h+1)*256] = ref[h + 24*_tt] * sg   # q-row = h + 24*t

  # (a) t32 + pfc16t
  pm, ps, pA = run_t32(pos, False); ao_a = run_pfc16t(pm, ps, pA, qrow)
  e = np.abs(ao_a - ref_ao) / np.maximum(np.abs(ref_ao), 1e-3)
  print(f"[corr] t32+pfc16t vs pfa16-ref: relerr med {np.median(e):.3e} max {np.max(e):.3e} "
        f"| poison cells {(ao_a == 7.7).sum()}/{ao_a.size}", flush=True)
  # numpy cross-check of the S13 mapping itself
  ao_np = combine_t32(pm, ps, pA, qrow)
  e2 = np.abs(ao_a - ao_np) / np.maximum(np.abs(ao_np), 1e-3)
  e4 = np.abs(ao_np - ref_ao) / np.maximum(np.abs(ref_ao), 1e-3)
  print(f"[corr] pfc16t vs numpy-combine: relerr med {np.median(e2):.3e} max {np.max(e2):.3e}", flush=True)
  print(f"[corr] numpy-combine vs pfa16-ref: relerr med {np.median(e4):.3e} max {np.max(e4):.3e}", flush=True)
  _bad = np.argwhere(e2 > 1e-2)[:6]
  for r, c in _bad:
    print(f"  [dbg] ao[{r},{c}]: kern {ao_a[r,c]:.5f} np {ao_np[r,c]:.5f} ref {ref_ao[r,c]:.5f}", flush=True)
  # (b) fused LC
  ao_b1 = run_t32(pos, True); ao_b2 = run_t32(pos, True)
  det = np.array_equal(ao_b1, ao_b2)
  e3 = np.abs(ao_b1 - ao_np) / np.maximum(np.abs(ao_np), 1e-3)
  print(f"[corr] t32ctl(fused) vs numpy-combine: relerr med {np.median(e3):.3e} max {np.max(e3):.3e} | det x2 {det}", flush=True)
  ctr = P.down("atc_ctr", (1,), np.uint32)
  print(f"[corr] ctr after 2 fused launches = {ctr[0]} (must be 0: self-reset)", flush=True)
  # isolate: uniform gate (sigmoid~1) on REAL partials
  q1 = np.full((16, 12288), 50.0, np.float16)
  ao_u = run_pfc16t(pm, ps, pA, q1)
  def combine_t32_nogate(pm, ps, pA):
    gs = pm.shape[0]
    rows = np.zeros((gs, RMAX32), dtype=np.int64)
    for bx in range(gs):
      hp = bx % NHP3; s = (bx // NHP3) % S13; g = bx // (NHP3*S13)
      for r in range(RMAX32): rows[bx, r] = (r % 16)*24 + g*6 + hp*2 + r//16
    m = np.full((384, gs), -1e30, np.float32); nu = np.zeros((384, gs), np.float32)
    O = np.zeros((384, gs, 256), dtype=np.float32)
    for bx in range(gs):
      m[rows[bx], bx] = pm[bx]; nu[rows[bx], bx] = ps[bx]; O[rows[bx], bx] = pA[bx]
    gmax = m.max(axis=1, keepdims=True)
    with np.errstate(under="ignore"): w = np.exp(m - gmax)
    out = (O * w[:, :, None]).sum(axis=1); den = (nu * w).sum(axis=1)
    return (out / np.maximum(den, 1e-30)[:, None]).reshape(16, 24, 256)
  eu = np.abs(ao_u.reshape(16,24,256) - combine_t32_nogate(pm, ps, pA))
  print(f"[corr] UNGATED kernel vs numpy: med {np.median(eu):.3e} max {eu.max():.3e}", flush=True)
  # gate factor per (t,h): kern_ungated * sigmoid(npm gate) vs kern real
  qr32 = qrow.astype(np.float32)
  for hh in range(3):
    sg = 1.0/(1.0+np.exp(-qr32[:, hh*512+256:(hh+1)*512]))
    pred = ao_u.reshape(16,24,256)[:, hh, :] * sg
    got = ao_a.reshape(16,24,256)[:, hh, :]
    ee = np.abs(pred - got).mean()
    print(f"  [gate] h={hh}: mean|kern - kern_ungated*sigmoid(npm gate)| = {ee:.4f}", flush=True)
else:
  # bench @100336: shipped pfa16 ref, t32, t32+pfc16t pair, t32ctl fused
  S32, RMAX96, GS32 = 32, 96, 4*32
  def bench(name, fn):
    for _ in range(2): fn()
    dev.synchronize()
    best = 1e9
    for _ in range(5):
      t0 = time.perf_counter(); fn(); dev.synchronize()
      best = min(best, time.perf_counter() - t0)
    print(f"[bench] {name} pos={pos}: {best*1e3:8.2f} ms", flush=True)
  P.win_up("pos_slot", 0, np.array([pos], dtype=np.int32)); dev.synchronize()
  P.poison("pm", 4*32*96*4, np.float32, 7.7e31); P.poison("ps", 4*32*96*4, np.float32, 7.7e31)
  P.poison("pA", 4*32*96*256*4, np.float32, 7.7e31); dev.synchronize()
  bench("pfa16 (shipped)", lambda: prog("pfa16nw32_s32_100k", "pfa16")(
    P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
    global_size=(GS32,1,1), local_size=(1024,1,1)))
  bench("t32", lambda: prog("pfa32c_t32_s13_100k", "pfa32ct")(
    P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
    global_size=(NCTA,1,1), local_size=(NTHRA,1,1)))
  for _s in (26,):
    _gs = 4*_s*3
    bench(f"t32_s{_s}", lambda: prog(f"pfa32c_t32_s{_s}_100k", "pfa32ct")(
      P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
      global_size=(_gs,1,1), local_size=(NTHRA,1,1)))
    _pkt = prog(f"pfa32c_t32_s{_s}_100k", "pfa32ct"); _pcc = prog(f"pfc16t_s{_s}", "pfc16t")
    def pair_s():
      _pkt(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
           global_size=(_gs,1,1), local_size=(NTHRA,1,1))
      _pcc(P.d["pm"], P.d["ps"], P.d["pA"], P.d["qrow16"], P.d["ao16"], global_size=(24,1,1), local_size=(256,1,1))
    bench(f"t32_s{_s}+pfc pair", pair_s)
  pk_t = prog("pfa32c_t32_s13_100k", "pfa32ct"); pk_c = prog("pfc16t_s13", "pfc16t")
  def pair():
    pk_t(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
         global_size=(NCTA,1,1), local_size=(NTHRA,1,1))
    pk_c(P.d["pm"], P.d["ps"], P.d["pA"], P.d["qrow16"], P.d["ao16"], global_size=(24,1,1), local_size=(256,1,1))
  bench("t32+pfc16t pair", pair)
  pk_l = prog("pfa32ctl_s13_100k", "pfa32ctl")
  P.win_up("atc_ctr", 0, np.zeros(1, dtype=np.uint32)); dev.synchronize()
  bench("t32ctl fused", lambda: pk_l(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"], P.d["pm"], P.d["ps"], P.d["pA"],
       P.d["qrow16"], P.d["ao16"], P.d["atc_ctr"], global_size=(NCTA,1,1), local_size=(NTHRA,1,1)))
P._keep.clear()
print(f"[{mode}] DONE", flush=True)