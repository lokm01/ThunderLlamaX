# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Decode-value check: run pfg_dbg3 (decode dump) and diff ws[0..16][0..128]
against a numpy IQ3_XXS dequant of packed gate rows (same math as w1c.cu)."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev, iq3_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
PACKED = f"{BASE}/packed"
from engine0 import parse_gguf
ds, infos = parse_gguf()
gdn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" not in infos]
G0 = gdn_idx[0]

P = Bufs()
P.up("gridf", iq3_grid_f32())
W = np.load(f"{PACKED}/gate{G0}.npy")
P.up("w", W); dev.synchronize()
rng = np.random.default_rng(3)
x = (rng.standard_normal((16, 5120)) * 0.8).astype(np.float16)
P.up("x16buf", x.reshape(-1))
P.poison("out16", 16*6144*2, np.float16, 7.7); dev.synchronize()
lib = open(f"{BASE}/pfg_dbg3_k128.cubin", "rb").read()
pr = NVProgram(dev, TinyELF(lib=lib, name="pfg_dbg3_k128", target=dev.renderer.target, signature=tuple()))
pr(P.d["w"], P.d["gridf"], P.d["x16buf"], P.d["out16"], global_size=(96, 1, 1), local_size=(256, 1, 1))
dev.synchronize()
out = P.down("out16", (16, 6144), np.float16).astype(np.float32)

# numpy IQ3_XXS dequant of rows 0..15, k 0..127 (w1c.cu math)
rowb = 98 * 20
gridf = iq3_grid_f32().reshape(256, 4)
ref = np.zeros((16, 128), np.float32)
for r in range(16):
  row = W[r]
  qsp = row[:64*20].view(np.uint16)
  scp = row[64*20:96*20].view(np.uint32)
  dpp = row[96*20:98*20].view(np.uint16).view(np.float16)  # hmm: raw u16 -> half
  import struct
  dh = np.frombuffer(row[96*20:98*20].tobytes(), dtype="<f2")  # f16 d per block
  for lc in range(16):   # chunks covering k 0..127
    b, lcc = 0, lc
    d = float(np.frombuffer(row[96*20 + 2*b:96*20 + 2*b + 2].tobytes(), dtype="<f2")[0])
    sw = int(scp[8*b + (lcc>>2)])
    db = d * (((sw >> 28) & 0xF) + 0.5) * 0.5
    sidx = (sw >> (7*(lcc & 3))) & 0x7F
    spar = bin(sidx).count("1") & 1
    q = int(qsp[32*b + lcc])
    g0 = gridf[q & 0xFF]; g1 = gridf[q >> 8]
    for j in range(8):
      if j < 7: sg = -1.0 if ((sidx >> j) & 1) else 1.0
      else: sg = -1.0 if spar else 1.0
      gv = g0[j] if j < 4 else g1[j-4]
      ref[r, lc*8 + j] = db * gv * sg

mine = out[:, :128]
err = np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-6)
print(f"[dq] relerr max {err.max():.3e} med {np.median(err):.3e}")
bad = np.argwhere(err > 1e-2)
print(f"[dq] bad {len(bad)}/{err.size}; first 10 (row,col,mine,ref):")
for r, c in bad[:10]:
  print(f"  [{r},{c}] mine {mine[r,c]:.5f} ref {ref[r,c]:.5f}")
# column pattern check: is the error column-periodic (chunk mapping) or random?
if len(bad):
  cols = bad[:, 1]
  print(f"[dq] bad cols mod 32 hist: {np.bincount(cols % 32, minlength=32)}")
  print(f"[dq] bad cols // 32 hist: {np.bincount(cols // 32, minlength=4)}")
  print(f"[dq] bad rows hist: {np.bincount(bad[:, 0], minlength=16)}")

# ---- full-kernel A/B: hf kernel out[:, :64] vs numpy dot (CTA 0 tile) ----
P2 = None
lib = open(f"{BASE}/pfg_iq3g_hf_nw8k128.cubin", "rb").read()
prf = NVProgram(dev, TinyELF(lib=lib, name="pfg_iq3g_hf_nw8k128", target=dev.renderer.target, signature=tuple()))
P.poison("out16", 16*6144*2, np.float16, 7.7); dev.synchronize()
prf(P.d["w"], P.d["gridf"], P.d["x16buf"], P.d["out16"], global_size=(96,1,1), local_size=(256,1,1))
dev.synchronize()
full = P.down("out16", (16, 6144), np.float16).astype(np.float32)
# numpy dequant rows 0..63 FULL K, then dot
Wf = np.zeros((64, 5120), np.float32)
gridf2 = iq3_grid_f32().reshape(256, 4)
for r in range(64):
  row = W[r]
  qsp = row[:64*20].view(np.uint16); scp = row[64*20:96*20].view(np.uint32)
  for b in range(20):
    d = float(np.frombuffer(row[96*20 + 2*b:96*20 + 2*b + 2].tobytes(), dtype="<f2")[0])
    for lcc in range(32):
      sw = int(scp[8*b + (lcc>>2)])
      db = d * (((sw >> 28) & 0xF) + 0.5) * 0.5
      sidx = (sw >> (7*(lcc & 3))) & 0x7F
      spar = bin(sidx).count("1") & 1
      q = int(qsp[32*b + lcc])
      g0 = gridf2[q & 0xFF]; g1 = gridf2[q >> 8]
      for j in range(8):
        sg = (-1.0 if spar else 1.0) if j == 7 else (-1.0 if ((sidx >> j) & 1) else 1.0)
        Wf[r, b*256 + lcc*8 + j] = db * (g0[j] if j < 4 else g1[j-4]) * sg
refdot = x.astype(np.float32) @ Wf.T
err = np.abs(full[:, :64] - refdot) / np.maximum(np.abs(refdot), 1e-6)
print(f"[full-hf] relerr max {err.max():.3e} med {np.median(err):.3e}")
print(f"[full-hf] mine[0,:6] {full[0,:6]}")
print(f"[full-hf] ref [0,:6] {refdot[0,:6]}")
# per-column error pattern
badc = np.argwhere(err > 3e-3)
if len(badc):
  print(f"[full-hf] bad {len(badc)}; col hist mod 8: {np.bincount(badc[:,1] % 8, minlength=8)}; row hist: {np.bincount(badc[:,0], minlength=16)}")

# ---- ref-path repro: q5g8 as collected in validate() ----
print("[refrepro] collecting q5g8 gate_row refs like validate() does", flush=True)
P.poison("xh", 17408*2, np.float16, 7.7)
wqkvr = None
from engine0 import read_raw
wqkvr = np.frombuffer(read_raw(infos[f"blk.{G0}.attn_qkv.weight"], ds), dtype=np.uint8)
P.up("w_qkv", wqkvr)
P.poison("qkv_row", 10240*2, np.float16, 7.7)
P.poison("gate_row", 6144*2, np.float16, 7.7)
dev.synchronize()
refs = np.zeros((16, 6144), np.float32)
for m in range(16):
  P.win_up("xh", 0, x[m]); dev.synchronize()
  prq = NVProgram(dev, TinyELF(lib=open(f"{BASE}/q5g8.cubin","rb").read(), name="q5g8", target=dev.renderer.target, signature=tuple()))
  prq(P.d["w_qkv"], P.d["w"], P.d["gridf"], P.d["xh"], P.d["qkv_row"], P.d["gate_row"], global_size=(2048,1,1), local_size=(256,1,1), wait=True)
  refs[m] = P.down("gate_row", (6144,), np.float16).astype(np.float32)
print(f"[refrepro] ref[0,:6] {refs[0,:6]}")
print(f"[refrepro] npy [0,:6] {refdot[0,:6]}")
err2 = np.abs(full[:, :6144][:, :] - refs) / np.maximum(np.abs(refs), 1e-6)
print(f"[refrepro] full-vs-ref relerr med {np.median(err2):.3e} max {err2.max():.3e}")
bad2 = np.argwhere(err2 > 3e-3)
print(f"[refrepro] bad {len(bad2)}/{err2.size}")
print(f"[refrepro] bad row hist: {np.bincount(bad2[:,0], minlength=16)}")
print(f"[refrepro] bad col%8 hist: {np.bincount(bad2[:,1] % 8, minlength=8)}")
print(f"[refrepro] bad col//8%8 hist: {np.bincount((bad2[:,1]//8) % 8, minlength=8)}")
if len(bad2):
  r, c = bad2[0]
  print(f"[refrepro] first bad [{r},{c}] mine {full[r,c]:.5f} ref {refs[r,c]:.5f}")
  r, c = bad2[len(bad2)//2]
  print(f"[refrepro] mid bad   [{r},{c}] mine {full[r,c]:.5f} ref {refs[r,c]:.5f}")
