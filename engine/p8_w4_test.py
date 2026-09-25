# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P8w4 A/B: the fused W4A8 FFN kernel vs the shipped pfg3_ffn fp16 stream,
REAL weights (packed4 planes), warm engine world. ORDER LAW (the P8 fault
class): a REAL 2k prefill runs FIRST — the never-executed-plan world faults
standalone launches; all standalone A/B work happens after it.
[1] warm 2k prefill (real plan executes) [2] pfk_q8 vs numpy quantizer
[3] numerics vs fp64-of-quantized [4] census-methodology timing.
"""
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
for nm, k in (("p8q8x_nw8k128", "p8q8x"), ("p8w4ffn_nw8k128", "p8w4ffn")):
  lib = open(f"~/tinygrad-metal/engine0/{nm}.cubin", "rb").read()
  pr[nm] = NVProgram(dev, TinyELF(lib=lib, name=k, target=dev.renderer.target, signature=tuple()))
dev.synchronize()

KD, ND, M = 5120, 17408, 128
NCH = KD // 128
rng = np.random.default_rng(11)
BLKS = [int(os.getenv("P8W4_BLK", "3")), int(os.getenv("P8W4_BLK2", "40"))]

# ---- load real packed4 planes for the test blocks (uploads only, no launches) ----
w4 = {}
for i in BLKS:
  u1 = np.load(f"~/tinygrad-metal/engine0/packed4/fg{i}.npy")
  s1 = np.load(f"~/tinygrad-metal/engine0/packed4/fgs{i}.npy")
  u2 = np.load(f"~/tinygrad-metal/engine0/packed4/fu{i}.npy")
  s2 = np.load(f"~/tinygrad-metal/engine0/packed4/fus{i}.npy")
  P.up(f"w4t_fg_{i}", np.frombuffer(u1.tobytes(), dtype=np.uint8))
  P.up(f"w4t_fgs_{i}", s1)
  P.up(f"w4t_fu_{i}", np.frombuffer(u2.tobytes(), dtype=np.uint8))
  P.up(f"w4t_fus_{i}", s2)
  w4[i] = (u1, s1, u2, s2)
dev.synchronize(); P._keep.clear()

# ---- random fp16 x + the numpy quantizer reference ----
xf = (rng.standard_normal((M, KD)) * 0.7).astype(np.float16)
xq = np.zeros((M, KD), dtype=np.int8); sx = np.zeros((M, NCH), dtype=np.float32)
rs = np.zeros((M, NCH), dtype=np.int32)
for c in range(NCH):
  seg = xf[:, c*128:(c+1)*128].astype(np.float32)
  am = np.maximum(np.abs(seg).max(axis=1), 1e-6)
  s = am * (1.0 / 127.0)
  sx[:, c] = s
  q = np.clip(np.rint(seg / s[:, None]), -127, 127).astype(np.int8)
  xq[:, c*128:(c+1)*128] = q
  rs[:, c] = q.sum(axis=1).astype(np.int32)
P.up("w4t_xf", xf); P.up("w4t_xq", xq); P.up("w4t_sx", sx); P.up("w4t_rs", rs)
P.poison("w4t_xqk", M*KD, np.int8, -19)
P.poison("w4t_sxk", M*NCH*4, np.float32, 7.7e31)
P.poison("w4t_rsk", M*NCH*4, np.int32, 0x5a5a5a5a)
P.poison("w4t_out", M*ND*2, np.float16, 7.7)
P.poison("w4t_gact", M*ND*2, np.float16, 7.7)
P.up("w4t_hhx", xf)   # the quant kernel's fp16 input = the same random x
dev.synchronize(); P._keep.clear()

# ---- [1] WARM: a real 2k prefill (the plan/graph world executes) ----
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

# ---- [2] pfk_q8 kernel vs numpy quantizer (bit-level) ----
pr["p8q8x_nw8k128"](d["w4t_hhx"], d["w4t_xqk"], d["w4t_sxk"], d["w4t_rsk"], d["w4t_xq"], d["w4t_xq"],
                    global_size=(M, 1, 1), local_size=LS, wait=True)
gxq = P.down("w4t_xqk", (M, KD), np.int8)
gsx = P.down("w4t_sxk", (M, NCH), np.float32)
grs = P.down("w4t_rsk", (M, NCH), np.int32)
print(f"[q8] xq bit-identical: {bool((gxq == xq).all())}", flush=True)
print(f"[q8] sx  bit-identical: {bool((gsx == sx).all())}", flush=True)
print(f"[q8] rs  bit-identical: {bool((grs == rs).all())}", flush=True)
P._keep.clear()

# ---- [3] numpy fp64-of-quantized reference + kernel numerics ----
def np_ref(i):
  u1, s1, u2, s2 = w4[i]
  def deq(u, s):
    uu = u.reshape(ND // 8, NCH, 32, 4)
    wq = np.zeros((ND, KD), dtype=np.float64)
    for c in range(NCH):
      for qc in range(4):
        v = uu[:, c, qc::4, :]                      # [g, r, word]
        blk = np.zeros((ND // 8, 8, 32), dtype=np.float64)
        for w in range(32):
          blk[:, :, w] = (v[..., w >> 3] >> (4 * (w & 7))) & 0xF
        wq[:, c*128 + qc*32:c*128 + qc*32 + 32] = blk.reshape(ND, 32) - 8.0
    return wq * s.astype(np.float64)[:, (np.arange(KD) // 128)]
  wg = deq(u1, s1); wu = deq(u2, s2)
  xdeq = xq.astype(np.float64) * sx.astype(np.float64)[:, (np.arange(KD) // 128)]
  g = xdeq @ wg.T; u = xdeq @ wu.T
  hgf = g.astype(np.float16).astype(np.float32)
  huf = u.astype(np.float16).astype(np.float32)
  sil = hgf * (1.0 / (1.0 + np.exp2(hgf * -1.4423828125)))
  sil = sil.astype(np.float16).astype(np.float32)
  return (sil.astype(np.float16).astype(np.float32) * huf.astype(np.float16).astype(np.float32)).astype(np.float16)

for i in BLKS:
  pr["p8w4ffn_nw8k128"](d[f"w4t_fg_{i}"], d[f"w4t_fu_{i}"], d[f"w4t_fgs_{i}"], d[f"w4t_fus_{i}"],
                        d["w4t_xqk"], d["w4t_sxk"], d["w4t_rsk"], d["w4t_out"],
                        global_size=(544, 1, 1), local_size=LS, wait=True)
  got = P.down("w4t_out", (M, ND), np.float16).astype(np.float64)
  ref = np_ref(i).astype(np.float64)
  rel = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9)
  print(f"[w4ffn] blk {i}: relerr vs fp64-of-quantized {rel:.3e} (absmax {np.abs(got-ref).max():.4f})", flush=True)
  P._keep.clear()

# ---- [4] timing: shipped fp16 fused (g=1088) vs W4A8 (quant+g=544), min-of-N ----
ffn_blocks = [i for i in E.gdn_idx if ("fg", i) in W7][:8]
def run_ffn(obuf, i):
  pr["pfg3_ffn_r7_m64_nw4k128"](W7[("fg", i)], W7[("fu", i)], d["gridf"], d["w4t_hhx"], obuf,
      global_size=(1088,1,1), local_size=(128,1,1), wait=True)
ts = []
for k, i in enumerate(ffn_blocks):
  run_ffn(d["w4t_gact"], i); dev.synchronize()
  t0 = time.perf_counter(); run_ffn(d["w4t_gact"], i); dev.synchronize(); ts.append(time.perf_counter() - t0)
  print(f"[bench:ffn] launch {k} ok", flush=True)
p_ffn = min(ts)
print("[phase] ffn bench done", flush=True)

ts = []
def run_w4(i):
  pr["p8q8x_nw8k128"](d["w4t_hhx"], d["w4t_xqk"], d["w4t_sxk"], d["w4t_rsk"], d["w4t_xq"], d["w4t_xq"],
                      global_size=(M, 1, 1), local_size=LS, wait=True)
  pr["p8w4ffn_nw8k128"](d[f"w4t_fg_{i}"], d[f"w4t_fu_{i}"], d[f"w4t_fgs_{i}"], d[f"w4t_fus_{i}"],
                        d["w4t_xqk"], d["w4t_sxk"], d["w4t_rsk"], d["w4t_out"],
                        global_size=(544, 1, 1), local_size=LS, wait=True)
for k in range(6):
  i = BLKS[k % len(BLKS)]   # only the plane-resident test blocks
  run_w4(i); dev.synchronize()
  t0 = time.perf_counter(); run_w4(i); dev.synchronize(); ts.append(time.perf_counter() - t0)
  print(f"[bench:w4] launch {k} ok {ts[-1]*1e6:.1f}us", flush=True)
p_w4 = min(ts)
w_w4 = 2 * ND * KD * (0.5 + 2.0/128) / 1e6
print(f"[perf] HMMA fused fp16 : {p_ffn*1e6:8.1f} us/launch (g=1088)", flush=True)
print(f"[perf] W4A8 fused+q8  : {p_w4*1e6:8.1f} us/launch (quant+g=544) | W {w_w4:.1f} MB -> {w_w4/1e3/p_w4:6.1f} GB/s", flush=True)
print(f"[verdict] speedup x{p_ffn/max(p_w4,1e-12):.3f} per launch", flush=True)
print("[w4test] DONE", flush=True)
