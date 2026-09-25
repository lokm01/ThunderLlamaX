# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-B validate+bench: pf_gemm3/pf_gemm3m (repacked-layout + M-grid) vs the
SHIPPED P6 M32 kernels on real weights.
Gates (READOUT-ORDER LAW: gates from the first clean run, timing after):
  1. gemm3-r7-m32 vs P6-m32 (32 rows): BIT-IDENTICAL (same decode words, same
     per-row k-order; the repack is a pure byte permutation).
  2. gemm3-m64 vs P6-m32 x2 (64 rows): BIT-IDENTICAL (the M-amortization law).
  3. gemm3-cl (classic control) vs P6-m32: BIT-IDENTICAL (template sanity).
Bench: async launch + dev.synchronize, min-of-10; weight GB/s (original
bytes), DRAM GB/s (repacked bytes = 1.3061x — the padding cost), TFLOPS,
MFU vs 71.2T, amortized GB/s (P6 def: (rows/16)*wb/T).
Usage: ~/tg311/bin/python -u test_p7b.py [filter ...]
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev, parse_gguf, read_raw, iq3_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
PACKED = f"{BASE}/packed"
P7 = f"{BASE}/packed7"
LS = (256, 1, 1)
P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

P.up("gridf", iq3_grid_f32())
dev.synchronize()

ds, infos = parse_gguf()
attn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" in infos]
gdn_idx = [i for i in range(64) if i not in set(attn_idx)]
G0 = gdn_idx[0]
A18 = next(i for i in attn_idx if infos[f"blk.{i}.attn_q.weight"][0] != 14)
print(f"[blocks] gdn0={G0} attn-iq3={A18}", flush=True)

# ---- weights: originals (for P6 refs + classic segs) + repacked ----
P.up("w_fg", np.load(f"{PACKED}/fg{G0}.npy"))
P.up("w_fu", np.load(f"{PACKED}/fu{G0}.npy"))
P.up("w_fd", np.load(f"{PACKED}/fd{G0}.npy"))
OQ = next(i for i in gdn_idx if os.path.exists(f"{PACKED}/out{i}.npy"))
P.up("w_o18", np.load(f"{PACKED}/out{OQ}.npy"))
P.up("w_qkv5", np.frombuffer(read_raw(infos[f"blk.{G0}.attn_qkv.weight"], ds), dtype=np.uint8))
P.up("w_v4", np.frombuffer(read_raw(infos[f"blk.{A18}.attn_v.weight"], ds), dtype=np.uint8))
P.up("r_fg", np.load(f"{P7}/fg{G0}.npy"))
P.up("r_fu", np.load(f"{P7}/fu{G0}.npy"))
P.up("r_fd", np.load(f"{P7}/fd{G0}.npy"))
P.up("r_o", np.load(f"{P7}/out{OQ}.npy"))
P.up("r_gate", np.load(f"{P7}/gate{G0}.npy"))
P.up("r_q", np.load(f"{P7}/q{A18}.npy"))
P.up("r_k", np.load(f"{P7}/k{A18}.npy"))
rng = np.random.default_rng(7)
P.up("res64", (rng.standard_normal((64, 5120)) * 4.0).astype(np.float32).reshape(-1))
dev.synchronize()
print("[weights] up", flush=True)

R7F = 512.0 / 392.0   # repacked/orig byte ratio (KCH=128 layout)

# ---- single classes: (tag, kd, nd, res, ffn, wref_names, r7_names, wb, grid32, variants)
SINGLES = [
  ("ffn", 5120, 17408, False, True, ("w_fg", "w_fu"), ("r_fg", "r_fu"), 2*17408*1960, 272, [
      ("m32r7", "pfg3_ffn_r7_m32_nw8k128", 32, 272, LS, 32, "r"),
      ("m64nw4", "pfg3_ffn_r7_m64_nw4k128", 64, 544, (128, 1, 1), 64, "r"),
      ("m32cl", "pfg3_ffn_cl_m32_nw8k128", 32, 272, LS, 32, "o"),
      ("m64cl", "pfg3_ffn_cl_m64_nw6k128", 64, 0, (192, 1, 1), 0)]),  # INVALID: 17408%48; skipped
  ("iq3d", 17408, 5120, True, False, ("w_fd",), ("r_fd",), 5120*6664, 80, [
      ("m32r7", "pfg3_iq3d_r7_m32_nw8k128", 32, 80, LS, 32, "r"),
      ("m64r7", "pfg3_iq3d_r7_m64_nw8k128", 64, 80, LS, 64),
      ("m64cl", "pfg3_iq3d_cl_m64_nw8k128", 64, 80, LS, 64, "o")]),
  ("iq3o", 6144, 5120, False, False, ("w_o18",), ("r_o",), 5120*2352, 80, [
      ("m32r7", "pfg3_iq3o_r7_m32_nw8k128", 32, 80, LS, 32, "r"),
      ("m64r7", "pfg3_iq3o_r7_m64_nw8k128", 64, 80, LS, 64, "r")]),
]
REF32 = {"ffn": ("pfg_ffn_m32_hm_nw8k128", ("w_fg", "w_fu"), False, True),
         "iq3d": ("pfg_iq3d_m32_res_hm_nw8k128", ("w_fd",), True, False),
         "iq3o": ("pfg_iq3o_m32_hm_nw8k128", ("w_o18",), False, False)}

only = [a for a in sys.argv[1:] if not a.startswith("--")] or None
ALL_OK = True
RESULTS = {}

def run_ref32(tag, rows32):
  """P6 m32 reference over `rows32` rows (one launch per 32 rows)."""
  name, wn, res, ffn = REF32[tag]
  pr = prog(name)
  outs = []
  for i, r0 in enumerate(range(0, rows32, 32)):
    x = P.d[f"x{tag}"] if i == 0 else P.d[f"x{tag}"].offset(offset=32*KD[tag]*2, size=32*KD[tag]*2)
    o = P.d[f"ref{tag}"] if i == 0 else P.d[f"ref{tag}"].offset(offset=32*ND[tag]*(4 if res else 2), size=32*ND[tag]*(4 if res else 2))
    rr = P.d["res64"] if i == 0 else P.d["res64"].offset(offset=32*5120*4, size=32*5120*4)
    a = tuple(P.d[w] for w in wn) + (P.d["gridf"], x)
    if res: a = a + (rr, o)
    else: a = a + (o,)
    pr(*a, global_size=(GRID[tag], 1, 1), local_size=LS)
  dev.synchronize()

KD = {t: s[1] for t, s in zip(["ffn", "iq3d", "iq3o"], SINGLES)}
ND = {t: s[2] for t, s in zip(["ffn", "iq3d", "iq3o"], SINGLES)}
GRID = {t: s[8] for t, s in zip(["ffn", "iq3d", "iq3o"], SINGLES)}

for tag, kd, nd, res, ffn, wn, rn, wb, g32, variants in SINGLES:
  if only and not any(o in tag for o in only): continue
  # 64-row x + poison outs
  P.up(f"x{tag}", (rng.standard_normal((64, kd)) * 0.8).astype(np.float16).reshape(-1))
  elt = 4 if res else 2
  P.poison(f"ref{tag}", 64*nd*elt, np.float32 if res else np.float16, 7.7e31 if res else 7.7)
  P.poison(f"mine{tag}", 64*nd*elt, np.float32 if res else np.float16, 7.7e31 if res else 7.7)
  dev.synchronize()
  run_ref32(tag, 64)
  ref = P.down(f"ref{tag}", (64, nd), np.float32 if res else np.float16).astype(np.float32)
  for vt in variants:
    vtag, kname, rows, gsz, lsz, mt = vt[:6]
    wt = vt[6] if len(vt) > 6 else "r"
    if mt == 0: continue
    wset = rn if wt == "r" else wn
    if only and not any(o in vtag or o in kname for o in only) and not any(o in tag for o in only): continue
    pr = prog(kname)
    P.poison(f"mine{tag}", 64*nd*elt, np.float32 if res else np.float16, 7.7e31 if res else 7.7)
    dev.synchronize()
    for i in range(64 // mt):
      x = P.d[f"x{tag}"] if i == 0 else P.d[f"x{tag}"].offset(offset=32*kd*2, size=32*kd*2)
      o = P.d[f"mine{tag}"] if i == 0 else P.d[f"mine{tag}"].offset(offset=32*nd*elt, size=32*nd*elt)
      rr = P.d["res64"] if i == 0 else P.d["res64"].offset(offset=32*5120*4, size=32*5120*4)
      a = tuple(P.d[w] for w in wset) + (P.d["gridf"], x)
      if res: a = a + (rr, o)
      else: a = a + (o,)
      pr(*a, global_size=(gsz, 1, 1), local_size=lsz)
    dev.synchronize()
    mine = P.down(f"mine{tag}", (64, nd), np.float32 if res else np.float16).astype(np.float32)
    nz = int((mine != ref).sum())
    ok = nz == 0
    ALL_OK &= ok
    msg = "BIT-IDENTICAL" if ok else f"DIFF nz={nz}/{mine.size}"
    if not ok:
      d = np.abs(mine - ref); rel = d / np.maximum(np.abs(ref), 1e-6)
      msg += f" maxrel {rel.max():.3e}"
    print(f"[gate] {tag:<5} {vtag:<6} vs P6-m32x2: {msg}", flush=True)
    if not ok: continue
    # bench (m32 configs: the honest 64-row cost = two launches)
    import time as _t
    def one64():
      for i in range(64 // mt):
        x = P.d[f"x{tag}"] if i == 0 else P.d[f"x{tag}"].offset(offset=32*kd*2, size=32*kd*2)
        o = P.d[f"mine{tag}"] if i == 0 else P.d[f"mine{tag}"].offset(offset=32*nd*elt, size=32*nd*elt)
        rr = P.d["res64"] if i == 0 else P.d["res64"].offset(offset=32*5120*4, size=32*5120*4)
        aa = tuple(P.d[w] for w in wset) + (P.d["gridf"], x)
        if res: aa = aa + (rr, o)
        else: aa = aa + (o,)
        pr(*aa, global_size=(gsz, 1, 1), local_size=lsz)
    for _ in range(2): one64()
    dev.synchronize()
    best = 1e9
    for _ in range(10):
      t0 = _t.perf_counter()
      one64()
      dev.synchronize()
      best = min(best, _t.perf_counter() - t0)
    best /= (64 // mt)   # per-launch time (one MTILE-row pass)
    r7 = "r7" in vtag or "r7" in kname
    dram = wb * (R7F if r7 else 1.0)
    tf = 2.0 * rows * nd * kd / best / 1e12
    RESULTS[(tag, vtag)] = (best, wb, dram, rows, nd, kd)
    print(f"[bench] {tag:<5} {vtag:<6} {best*1e3:7.3f} ms/{rows}r | wGB/s {wb/best/1e9:6.1f} | "
          f"DRAM {dram/best/1e9:6.1f} | amort {(mt/16)*wb/best/1e9:6.1f} | {tf:5.2f} TF {tf/71.2*100:4.1f}% MFU", flush=True)

# ---- merged twins ----
TWINS = [
  ("gdnqg", 5120, [("qkv32", 10240), ("gate32", 6144)],
   ("pfg2_gdnqg_m32_hm_nw16k128", ("w_qkv5", "w_gate_o"), 128, (512, 1, 1)),
   [
     ("m32r7", "pfg3m_gdnqg_r7_m32_nw16k128", 32, 128, (512, 1, 1), ("w_qkv5", "r_gate")),
     ("m64r7", "pfg3m_gdnqg_r7_m64_nw8k128", 64, 256, LS, ("w_qkv5", "r_gate")),
   ], 10240*176*20 + 6144*1960),
  ("attnqkvi3", 5120, [("qrow", 12288), ("krow", 1024), ("vrow", 1024)],
   ("pfg2_attnqkvi3_m32_hm_nw8k128", ("w_q18_o", "w_k18_o", "w_v4"), 224, LS),
   [
     ("m32r7", "pfg3m_attnqkvi3_r7_m32_nw8k128", 32, 224, LS, ("r_q", "r_k", "w_v4")),
     ("m64r7", "pfg3m_attnqkvi3_r7_m64_nw8k128", 64, 224, LS, ("r_q", "r_k", "w_v4")),
   ], 12288*1960 + 1024*1960 + 1024*144*20),
]
print("[twins] loading originals for refs", flush=True)
P.up("w_gate_o", np.load(f"{PACKED}/gate{G0}.npy"))
P.up("w_q18_o", np.load(f"{PACKED}/q{A18}.npy"))
P.up("w_k18_o", np.load(f"{PACKED}/k{A18}.npy"))
dev.synchronize()

for tag, kd, outs, refentry, variants, wb in TWINS:
  if only and not any(o in tag for o in only): continue
  osum = sum(n for _, n in outs)
  P.up(f"xt_{tag}", (rng.standard_normal((64, kd)) * 0.5).astype(np.float16).reshape(-1))
  for on, ondim in outs:
    P.poison(f"{on}_r", 64*ondim*2, np.float16, 7.7)
    P.poison(f"{on}_m", 64*ondim*2, np.float16, 7.7)
  dev.synchronize()
  # reference: P6 twin m32 x2 (rows 0-31, 32-63)
  if tag == "gdnqg":
    refn, refw, gsz, lsz = "pfg2_gdnqg_m32_hm_nw16k128", ("w_qkv5", "w_gate_o"), 128, (512, 1, 1)
  else:
    refn, refw, gsz, lsz = "pfg2_attnqkvi3_m32_hm_nw8k128", ("w_q18_o", "w_k18_o", "w_v4"), 224, LS
  pr = prog(refn)
  for i in range(2):
    x = P.d[f"xt_{tag}"] if i == 0 else P.d[f"xt_{tag}"].offset(offset=32*kd*2, size=32*kd*2)
    oas = tuple(P.d[f"{on}_r"] if i == 0 else P.d[f"{on}_r"].offset(offset=32*n*2, size=32*n*2) for on, n in outs)
    pr(*(tuple(P.d[w] for w in refw) + (P.d["gridf"], x) + oas), global_size=(gsz, 1, 1), local_size=lsz)
  dev.synchronize()
  refs = [P.down(f"{on}_r", (64, n), np.float16).astype(np.float32) for on, n in outs]
  for vtag, kname, rows, gsz2, lsz2, wn in variants:
    pr2 = prog(kname)
    for on, n in outs: P.poison(f"{on}_m", 64*n*2, np.float16, 7.7)
    dev.synchronize()
    mt = 32 if rows == 32 else 64
    for i in range(64 // mt):
      x = P.d[f"xt_{tag}"] if i == 0 else P.d[f"xt_{tag}"].offset(offset=32*kd*2, size=32*kd*2)
      oas = tuple(P.d[f"{on}_m"] if i == 0 else P.d[f"{on}_m"].offset(offset=32*n*2, size=32*n*2) for on, n in outs)
      a = tuple(P.d[w] for w in wn) + (P.d["gridf"], x) + oas
      pr2(*a, global_size=(gsz2, 1, 1), local_size=lsz2)
    dev.synchronize()
    nz = tot = 0
    for (on, n), rr in zip(outs, refs):
      mm = P.down(f"{on}_m", (64, n), np.float16).astype(np.float32)
      nz += int((mm != rr).sum()); tot += mm.size
    ok = nz == 0
    ALL_OK &= ok
    print(f"[gate] {tag:<9} {vtag:<6} vs P6-m32x2: {'BIT-IDENTICAL' if ok else f'DIFF nz={nz}/{tot}'}", flush=True)
    if not ok: continue
    def one64t():
      for i in range(64 // mt):
        x = P.d[f"xt_{tag}"] if i == 0 else P.d[f"xt_{tag}"].offset(offset=32*kd*2, size=32*kd*2)
        oas = tuple(P.d[f"{on}_m"] if i == 0 else P.d[f"{on}_m"].offset(offset=32*n*2, size=32*n*2) for on, n in outs)
        aa = tuple(P.d[w] for w in wn) + (P.d["gridf"], x) + oas
        pr2(*aa, global_size=(gsz2, 1, 1), local_size=lsz2)
    for _ in range(2): one64t()
    dev.synchronize()
    best = 1e9
    for _ in range(10):
      t0 = time.perf_counter()
      one64t()
      dev.synchronize()
      best = min(best, time.perf_counter() - t0)
    best /= (64 // mt)
    r7frac = (wb - (10240*176*20 if tag == "gdnqg" else 1024*144*20)) / wb
    dram = wb * (1.0 + (R7F - 1.0) * r7frac)
    tf = 2.0 * mt * sum(n for _, n in outs) * kd / best / 1e12
    print(f"[bench] {tag:<9} {vtag:<6} {best*1e3:7.3f} ms/{rows}r | wGB/s {wb/best/1e9:6.1f} | "
          f"DRAM {dram/best/1e9:6.1f} | amort {(mt/16)*wb/best/1e9:6.1f} | {tf:5.2f} TF {tf/71.2*100:4.1f}% MFU", flush=True)

print("[p7b] ALL BIT-IDENTICAL" if ALL_OK else "[p7b] DIFFERENCES FOUND", flush=True)
sys.exit(0 if ALL_OK else 1)
