# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P8w4-v2 A/B: the packed7-reading fused W4A8 FFN kernel (p8w4ffn7) vs the
shipped pfg3_ffn fp16 stream, REAL weights from the resident W7 buffers.
Numerics vs a numpy reference of the LINEARIZED quantized values (Glin grid +
fp16-rounded sdb scales, same ints). Warm engine world (P8 order law)."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from mtp import MTPEngine, CBLK
import pf_prefill

E = MTPEngine(theta=1e7)
E.reset_fresh(101)
E.stload_trunk()
for i in E.gdn_idx:
  E._mfill(f"conv{i}_1", 0, CBLK)
dev.synchronize()
pf_prefill.ensure64(E)
pf_prefill.ensure128(E)
dev.synchronize(); E.P._keep.clear()
P, d, W, pr = E.P, E.P.d, E.W, E.pr
W7 = getattr(E, "_pf_W7", {})
LS = (256, 1, 1)

from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram
pr["p8q8x_nw8k128"] = NVProgram(dev, TinyELF(
    lib=open("~/tinygrad-metal/engine0/p8q8x_nw8k128.cubin", "rb").read(),
    name="p8q8x", target=dev.renderer.target, signature=tuple()))
pr["p8w4ffn7_nw8k128"] = NVProgram(dev, TinyELF(
    lib=open("~/tinygrad-metal/engine0/p8w4ffn7_nw8k128.cubin", "rb").read(),
    name="p8w4ffn7", target=dev.renderer.target, signature=tuple()))

# the linearized LUT (1024 s8)
sys.path.insert(0, "~/tinygrad-metal/engine0")
from pack_w4 import grid_f32
DELTA = 4.0507
G = grid_f32().reshape(-1)
lut8 = np.clip(np.rint(G / DELTA), 0, 15).astype(np.int8)
P.up("w7t_lut", lut8)
dev.synchronize(); P._keep.clear()

KD, ND, M = 5120, 17408, 128
NCH = KD // 128
rng = np.random.default_rng(11)
BLKS = [int(os.getenv("P8W4_BLK", "3")), int(os.getenv("P8W4_BLK2", "40"))]

# ---- numpy LINEARIZED reference (per BLK, from the packed/ raw bytes) ----
from pack_w4 import dequant_iq3xxs
import pack_w4
def lin_ref(i):
    arr = np.load(f"~/tinygrad-metal/engine0/packed/fg{i}.npy")
    if arr.ndim == 1: arr = arr.reshape(arr.shape[0], -1)
    arru = np.load(f"~/tinygrad-metal/engine0/packed/fu{i}.npy")
    if arru.ndim == 1: arru = arru.reshape(arru.shape[0], -1)
    def dq_lin(a):
        # replicate dequant_iq3xxs but with the linearized grid AND fp16 sdb
        n = a.shape[0]; nb = KD >> 8
        row16 = a.view(np.uint16)
        q = row16[:, :nb*32]
        scp = np.ascontiguousarray(a[:, 64*nb:96*nb]).view(np.uint32)
        dpp = np.ascontiguousarray(row16[:, nb*48:nb*49]).view(np.float16)
        d = dpp.astype(np.float32)
        Wl = np.empty((n, nb, 32, 8), dtype=np.float32)
        for lc in range(32):
            cc = lc & 3
            qv = q[:, lc::32].astype(np.uint32)
            swv = scp[:, (lc >> 2)::8]
            sdb = d * ((swv >> 28).astype(np.float32) + 0.5) * 0.5 * DELTA
            sidx = (swv >> (7 * cc)) & 0x7F
            spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1
            i0 = ((qv & 0xFF) << 2)[:, :, None] + np.arange(4)[None, None, :]
            i1 = ((qv >> 8) << 2)[:, :, None] + np.arange(4)[None, None, :]
            l0 = lut8.astype(np.float32)[i0.reshape(-1)].reshape(n, -1, 4)
            l1 = lut8.astype(np.float32)[i1.reshape(-1)].reshape(n, -1, 4)
            sg = np.empty((n, sidx.shape[1], 8), dtype=np.float32)
            for b in range(7):
                sg[:, :, b] = np.where((sidx >> b) & 1, -1.0, 1.0)
            sg[:, :, 7] = np.where(spar != 0, -1.0, 1.0)
            lv = np.concatenate([l0, l1], axis=2) * sg
            Wl[:, :, lc, :] = sdb[..., None] * lv
        return Wl.reshape(n, KD)
    wg = dq_lin(arr); wu = dq_lin(arru)
    xdeq = xq.astype(np.float64) * sx.astype(np.float64)[:, (np.arange(KD) // 128)]
    g = xdeq @ wg.T.astype(np.float64); u = xdeq @ wu.T.astype(np.float64)
    hgf = g.astype(np.float16).astype(np.float32)
    huf = u.astype(np.float16).astype(np.float32)
    sil = hgf * (1.0 / (1.0 + np.exp2(hgf * -1.4423828125)))
    sil = sil.astype(np.float16).astype(np.float32)
    return (sil.astype(np.float16).astype(np.float32) * huf.astype(np.float16).astype(np.float32)).astype(np.float16)

# ---- random fp16 x + quant ----
xf = (rng.standard_normal((M, KD)) * 0.7).astype(np.float16)
xq = np.zeros((M, KD), dtype=np.int8); sx = np.zeros((M, NCH), dtype=np.float32)
for c in range(NCH):
  seg = xf[:, c*128:(c+1)*128].astype(np.float32)
  am = np.maximum(np.abs(seg).max(axis=1), 1e-6)
  s = am * (1.0 / 127.0)
  sx[:, c] = s
  xq[:, c*128:(c+1)*128] = np.clip(np.rint(seg / s[:, None]), -127, 127).astype(np.int8)
P.up("w7t_hhx", xf); P.up("w7t_xq", xq); P.up("w7t_sx", sx)
P.poison("w7t_rsk", M*NCH*4, np.int32, 0x5a5a5a5a)
P.poison("w7t_out", M*ND*2, np.float16, 7.7)
P.poison("w7t_gact", M*ND*2, np.float16, 7.7)
dev.synchronize(); P._keep.clear()

# ---- warm the engine world ----
import json as _json
_wids = [int(t) for t in _json.load(open("~/ids8k.json"))[:2048]]
from mtp import SLICE
_seen, sl = set(), []
for t in _wids:
  if t not in _seen:
    _seen.add(t); sl.append(t)
_base = sl[:]
while len(sl) < SLICE:
  sl += _base
E.init_draft(sl[:SLICE])
E.reset_fresh(_wids[0]); E.stload_trunk()
for i in E.gdn_idx:
  E._mfill(f"conv{i}_1", 0, CBLK)
dev.synchronize()
pf_prefill.prefill_batch(E, None, _wids)
dev.synchronize(); E.P._keep.clear()
print("[warm] 2k prefill done", flush=True)

# ---- numerics ----
for i in BLKS:
  pr["p8w4ffn7_nw8k128"](W7[("fg", i)], W7[("fu", i)], d["w7t_lut"], d["w7t_xq"], d["w7t_sx"], d["w7t_out"],
                        global_size=(544, 1, 1), local_size=LS, wait=True)
  got = P.down("w7t_out", (M, ND), np.float16).astype(np.float64)
  ref = lin_ref(i).astype(np.float64)
  rel = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9)
  # fp16 oracle: the shipped kernel on the same input
  pr["pfg3_ffn_r7_m64_nw4k128"](W7[("fg", i)], W7[("fu", i)], d["gridf"], d["w7t_hhx"], d["w7t_gact"],
      global_size=(1088,1,1), local_size=(128,1,1), wait=True)
  gf16 = P.down("w7t_gact", (M, ND), np.float16).astype(np.float64)
  rel16 = np.linalg.norm(got - gf16) / max(np.linalg.norm(gf16), 1e-9)
  relref16 = np.linalg.norm(ref - gf16) / max(np.linalg.norm(gf16), 1e-9)
  print(f"[w4ffn7] blk {i}: vs linref {rel:.3e} | vs fp16 {rel16:.3e} | linref vs fp16 {relref16:.3e}", flush=True)
  P._keep.clear()

# ---- DBG probe: single nonzero x element localizes the decode bug ----
if os.getenv("P8W4_DBG") == "1":
  i = BLKS[0]
  xqd = np.zeros((M, KD), dtype=np.int8); xqd[0, 0] = 127
  sxd = np.ones((M, NCH), dtype=np.float32)
  P.up("dbg_xq", xqd); P.up("dbg_sx", sxd)
  dev.synchronize(); P._keep.clear()
  pr["p8w4ffn7_nw8k128"](W7[("fg", i)], W7[("fu", i)], d["w7t_lut"], d["dbg_xq"], d["dbg_sx"], d["w7t_out"],
                        global_size=(544, 1, 1), local_size=LS, wait=True)
  gotd = P.down("w7t_out", (M, ND), np.float16).astype(np.float64)[0]
  P._keep.clear()
  # expected: silu(g0)*u0 where g0 = 1.0 * Wg[n,0], u0 = 1.0 * Wu[n,0]
  from pack_w4 import dequant_iq3xxs
  arrf = np.load(f"~/tinygrad-metal/engine0/packed/fg{i}.npy")
  if arrf.ndim == 1: arrf = arrf.reshape(arrf.shape[0], -1)
  arru = np.load(f"~/tinygrad-metal/engine0/packed/fu{i}.npy")
  if arru.ndim == 1: arru = arru.reshape(arru.shape[0], -1)
  Wg = dequant_iq3xxs(arrf, KD)[:, 0]; Wu = dequant_iq3xxs(arru, KD)[:, 0]
  import math
  def silu_h(v):
    hf = np.float16(v)
    s = np.float16(float(hf) * (1.0/(1.0+2.0**(float(hf)*-1.4423828125))))
    return float(np.float16(s))
  expd = np.array([silu_h(127.0*wg)*float(np.float16(127.0*wu)) for wg, wu in zip(Wg, Wu)])
  for sl in (slice(0,8), slice(32,40), slice(256,264), slice(1024,1032)):
    print(f"[dbg] n={sl.start}: got {np.round(gotd[sl],3)}", flush=True)
    print(f"[dbg]        exp {np.round(expd[sl],3)}", flush=True)
  err = np.abs(gotd - expd) / (np.abs(expd) + 1e-3)
  # per-256-col-block profile (N side) and per-warp-row profile
  nblk = ND // 256
  prof = [float(err[b*256:(b+1)*256].mean()) for b in range(min(nblk, 68))]
  print(f"[dbg] per-colblock err mean: {[round(p,2) for p in prof[:24]]}", flush=True)
  print(f"[dbg] ...{[round(p,2) for p in prof[24:48]]}", flush=True)
  print(f"[dbg] col%%8 profile: {[round(float(err[n::8].mean()),2) for n in range(8)]}", flush=True)
  cc = np.corrcoef(gotd, expd)[0,1]
  print(f"[dbg] corr(got,exp) = {cc:.4f}", flush=True)
  arrfW = dequant_iq3xxs(arrf, KD); arruW = dequant_iq3xxs(arru, KD)
  # per-k probe with mixing matrix: which W column does each k actually read?
  arrfWl = None
  Evec = {}
  for k0 in range(8):
      xqd = np.zeros((M, KD), dtype=np.int8); sxd = np.ones((M, NCH), dtype=np.float32)
      xqd[0, k0] = 127
      P.up("dbg_xq", xqd); P.up("dbg_sx", sxd)
      dev.synchronize(); P._keep.clear()
      pr["p8w4ffn7_nw8k128"](W7[("fg", i)], W7[("fu", i)], d["w7t_lut"], d["dbg_xq"], d["dbg_sx"], d["w7t_out"],
                            global_size=(544, 1, 1), local_size=LS, wait=True)
      Evec[k0] = P.down("w7t_out", (M, ND), np.float16).astype(np.float64)[0].copy()
      P._keep.clear()
  P._keep.clear()
  # predictions: linearized G,U columns (k 0..11)
  Glin = np.clip(np.rint(G / 4.0507), 0, 15) * 4.0507
  def lin_cols(arr):
      # reuse lin_ref machinery quickly: recompute dequant with Glin (fp32 sdb)
      n_ = arr.shape[0]; nb = KD >> 8
      row16 = arr.view(np.uint16)
      q = row16[:, :nb*32]
      scp = np.ascontiguousarray(arr[:, 64*nb:96*nb]).view(np.uint32)
      dpp = np.ascontiguousarray(row16[:, nb*48:nb*49]).view(np.float16)
      df_ = dpp.astype(np.float32)
      out = np.empty((n_, nb, 32, 8), dtype=np.float32)
      for lc in range(32):
          cc = lc & 3
          qv = q[:, lc::32].astype(np.uint32)
          swv = scp[:, (lc >> 2)::8]
          sdb = df_ * ((swv >> 28).astype(np.float32) + 0.5) * 0.5 * 4.0507
          sidx = (swv >> (7 * cc)) & 0x7F
          spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1
          def lv(qvx):
              idx = (qvx.astype(np.int64) << 2)
              l0 = lut8.astype(np.float32)[(idx[:, :, None] + np.arange(4)[None, None, :]).reshape(-1)].reshape(qvx.shape[0], -1, 4)
              return l0
          l0, l1 = lv(qv & 0xFF), lv(qv >> 8)
          sg = np.empty((n_, sidx.shape[1], 8), dtype=np.float32)
          for b in range(7):
              sg[:, :, b] = np.where((sidx >> b) & 1, -1.0, 1.0)
          sg[:, :, 7] = np.where(spar != 0, -1.0, 1.0)
          out[:, :, lc, :] = sdb[..., None] * (np.concatenate([l0, l1], axis=2) * sg)
      return out.reshape(n_, KD)
  Wgl, Wul = lin_cols(arrf), lin_cols(arru)
  preds = {}
  for kk in range(12):
      g = 127.0 * Wgl[:, kk]; u = 127.0 * Wul[:, kk]
      hgf = g.astype(np.float16).astype(np.float32)
      sil = hgf * (1.0 / (1.0 + np.exp2(hgf * -1.4423828125)))
      sil = sil.astype(np.float16).astype(np.float32)
      preds[kk] = (sil.astype(np.float16).astype(np.float32) * u.astype(np.float16).astype(np.float32)).astype(np.float64)
  print("[mix] rows=k-probe, cols=predicted-from-k:', corr matrix:", flush=True)
  hdr = "      " + " ".join(f"c{j}" for j in range(12))
  print(hdr, flush=True)
  for k0 in range(8):
      row = []
      for kk in range(12):
          a, b = Evec[k0], preds[kk]
          if np.std(b) < 1e-9: row.append(0.0); continue
          row.append(float(np.corrcoef(a, b)[0, 1]))
      print(f"k={k0}: " + " ".join(f"{v:+.2f}" for v in row), flush=True)
  # chunk + quarter profile: activate one k-chunk (or quarter) at a time
  for c in (0, 1, 2, 3, 19, 39):
    for q in (None, 0, 1, 2, 3):
      xqd = np.zeros((M, KD), dtype=np.int8); sxd = np.ones((M, NCH), dtype=np.float32)
      lo = c*128 + (q*32 if q is not None else 0); hi = c*128 + (q*32+32 if q is not None else 128)
      xqd[0, lo:hi] = 127
      P.up("dbg_xq", xqd); P.up("dbg_sx", sxd)
      dev.synchronize(); P._keep.clear()
      pr["p8w4ffn7_nw8k128"](W7[("fg", i)], W7[("fu", i)], d["w7t_lut"], d["dbg_xq"], d["dbg_sx"], d["w7t_out"],
                            global_size=(544, 1, 1), local_size=LS, wait=True)
      g2 = P.down("w7t_out", (M, ND), np.float16).astype(np.float64)[0]
      P._keep.clear()
      e2 = np.array([silu_h(127.0*float(np.sum(arrfW[n, lo:hi])))*float(np.float16(127.0*float(np.sum(arruW[n, lo:hi])))) for n in range(ND)])
      r2 = np.linalg.norm(g2 - e2) / max(np.linalg.norm(e2), 1e-9)
      tag = f"c{c}" + (f"q{q}" if q is not None else " full")
      print(f"[dbg] {tag}: relerr {r2:.3e} corr {np.corrcoef(g2, e2)[0,1]:.4f}", flush=True)
  # k-shift search: correlate got vs Wg at shifted k
  for kshift in (0, 4, 8, 16, 32, 64):
    Wgs = dequant_iq3xxs(arrf, KD)[:, kshift] if kshift else Wg
    exps = np.array([silu_h(wg)*float(np.float16(wu)) for wg, wu in zip(Wgs, Wu)])
    try: c2 = np.corrcoef(gotd, exps)[0,1]
    except Exception: c2 = float("nan")
    print(f"[dbg] kshift {kshift}: corr {c2:.4f}", flush=True)

# ---- timing ----
ffn_blocks = [i for i in E.gdn_idx if ("fg", i) in W7][:8]
def run_ffn(obuf, i):
  pr["pfg3_ffn_r7_m64_nw4k128"](W7[("fg", i)], W7[("fu", i)], d["gridf"], d["w7t_hhx"], obuf,
      global_size=(1088,1,1), local_size=(128,1,1), wait=True)
ts = []
for k, i in enumerate(ffn_blocks):
  run_ffn(d["w7t_gact"], i); dev.synchronize()
  t0 = time.perf_counter(); run_ffn(d["w7t_gact"], i); dev.synchronize(); ts.append(time.perf_counter() - t0)
p_ffn = min(ts)
print("[phase] ffn bench done", flush=True)

def run_w4(i):
  pr["p8q8x_nw8k128"](d["w7t_hhx"], d["w7t_xq"], d["w7t_sx"], d["w7t_rsk"], d["gridf"], d["gridf"],
                      global_size=(M, 1, 1), local_size=LS, wait=True)
  pr["p8w4ffn7_nw8k128"](W7[("fg", i)], W7[("fu", i)], d["w7t_lut"], d["w7t_xq"], d["w7t_sx"], d["w7t_out"],
                        global_size=(544, 1, 1), local_size=LS, wait=True)
ts = []
for k in range(6):
  i = BLKS[k % len(BLKS)]
  run_w4(i); dev.synchronize()
  t0 = time.perf_counter(); run_w4(i); dev.synchronize(); ts.append(time.perf_counter() - t0)
  print(f"[bench:w4v2] launch {k} ok {ts[-1]*1e6:.1f}us", flush=True)
p_w4 = min(ts)
print(f"[perf] HMMA fused fp16 : {p_ffn*1e6:8.1f} us/launch (g=1088)", flush=True)
print(f"[perf] W4A8-v2 fused+q8: {p_w4*1e6:8.1f} us/launch (quant+g=544)", flush=True)
print(f"[verdict] speedup x{p_ffn/max(p_w4,1e-12):.3f} per launch", flush=True)
print("[w4test7] DONE", flush=True)
