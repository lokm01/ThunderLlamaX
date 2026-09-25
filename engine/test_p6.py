# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P6 validate+bench: the M32 pGEMM cubins vs the SHIPPED M=16 kernels on real
packed weights (ffn / iq3d / iq3o — the big single classes; the MERGED M32
kernels are validated in pf_fwd32.py where the engine weight world exists).
Gate: M32 output rows must be BIT-IDENTICAL to the M=16 kernel run on the same
32 input rows (same decode, same per-row k-order/fragment map) — plus
poison-free. Bench: a 32-token batch needs TWO M16 launches (two full weight
passes) vs ONE M32 launch; report wall x2-vs-x1, per-pass GB/s and the
32-tok-amortized GB/s (the honest in-chunk metric).
Usage: ~/tg311/bin/python -u test_p6.py [filter ...]"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev, parse_gguf, iq3_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
PACKED = f"{BASE}/packed"
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
oq_type = {i: infos[f"blk.{i}.ssm_out.weight"][0] for i in gdn_idx}
G0 = gdn_idx[0]
O18 = next(i for i in gdn_idx if oq_type[i] == 18)
print(f"[blocks] gdn0={G0} o18={O18}", flush=True)

P.up("w_fg", np.load(f"{PACKED}/fg{G0}.npy"))
P.up("w_fu", np.load(f"{PACKED}/fu{G0}.npy"))
P.up("w_fd", np.load(f"{PACKED}/fd{G0}.npy"))
P.up("w_o18", np.load(f"{PACKED}/out{O18}.npy"))
rng = np.random.default_rng(7)
P.up("res_in", (rng.standard_normal((32, 5120)) * 4.0).astype(np.float32).reshape(-1))
dev.synchronize()

# (tag, m16, m32, wnames, nd, kd, wbytes_one_pass, grid16, inst_per_chunk, res)
CLASSES = [
  ("ffn",  "pfg_ffn_hm_nw8k128",      "pfg_ffn_m32_hm_nw8k128",      ("w_fg", "w_fu"), 17408, 5120, 1960*17408*2, 272, 64, False),
  ("iq3d", "pfg_iq3d_res_hm_nw8k128", "pfg_iq3d_m32_res_hm_nw8k128", ("w_fd",),        5120, 17408, 6664*5120,    80, 64, True),
  ("iq3o", "pfg_iq3o_hm_nw8k128",     "pfg_iq3o_m32_hm_nw8k128",     ("w_o18",),       5120, 6144, 2352*5120,    80, 24, False),
]

only = sys.argv[1:] or None
ALL_OK = True
for tag, m16, m32, wn, nd, kd, wb, g16, inst, res in CLASSES:
  if only and not any(o in tag or o in m32 for o in only): continue
  x = (rng.standard_normal((32, kd)) * 0.8).astype(np.float16)
  P.up("x32", x.reshape(-1))
  elt = 4 if res else 2
  wargs = tuple(P.d[w] for w in wn)
  x2 = P.d["x32"].offset(offset=16*kd*2, size=16*kd*2)
  r2 = P.d["res_in"].offset(offset=16*5120*4, size=16*5120*4) if res else None
  # ---- reference: M=16 kernel, rows 0..15 then 16..31 (two launches, two views)
  P.poison("outA", 32*nd*elt, np.float32 if res else np.float16, 7.7e31 if res else 7.7)
  dev.synchronize()
  oA2 = P.d["outA"].offset(offset=16*nd*elt, size=16*nd*elt)
  pr16 = prog(m16)
  if res: aA = lambda o: wargs + (P.d["gridf"], P.d["x32"], P.d["res_in"], o)
  else:   aA = lambda o: wargs + (P.d["gridf"], P.d["x32"], o)
  pr16(*aA(P.d["outA"]), global_size=(g16, 1, 1), local_size=LS)
  pr16(*(wargs + (P.d["gridf"], x2, r2, oA2) if res else wargs + (P.d["gridf"], x2, oA2)),
       global_size=(g16, 1, 1), local_size=LS)
  dev.synchronize()
  ref = P.down("outA", (32, nd), np.float32 if res else np.float16).astype(np.float32)
  # bench M16 x2 (the true 32-token cost today)
  for _ in range(2):
    pr16(*aA(P.d["outA"]), global_size=(g16, 1, 1), local_size=LS)
    pr16(*(wargs + (P.d["gridf"], x2, r2, oA2) if res else wargs + (P.d["gridf"], x2, oA2)),
         global_size=(g16, 1, 1), local_size=LS)
  dev.synchronize()
  b16 = 1e9
  for _ in range(10):
    t0 = time.perf_counter()
    pr16(*aA(P.d["outA"]), global_size=(g16, 1, 1), local_size=LS)
    pr16(*(wargs + (P.d["gridf"], x2, r2, oA2) if res else wargs + (P.d["gridf"], x2, oA2)),
         global_size=(g16, 1, 1), local_size=LS)
    dev.synchronize()
    b16 = min(b16, time.perf_counter() - t0)
  # ---- M32: one launch, full 32 rows
  P.poison("outB", 32*nd*elt, np.float32 if res else np.float16, 7.7e31 if res else 7.7)
  dev.synchronize()
  pr32 = prog(m32)
  if res: aB = wargs + (P.d["gridf"], P.d["x32"], P.d["res_in"], P.d["outB"])
  else:   aB = wargs + (P.d["gridf"], P.d["x32"], P.d["outB"])
  pr32(*aB, global_size=(g16, 1, 1), local_size=LS)
  dev.synchronize()
  mine = P.down("outB", (32, nd), np.float32 if res else np.float16).astype(np.float32)
  nz = int((mine != ref).sum())
  ok = nz == 0
  ALL_OK &= ok
  msg = "BIT-IDENTICAL" if ok else "DIFF"
  if not ok:
    d = np.abs(mine - ref); rel = d / np.maximum(np.abs(ref), 1e-6)
    msg += f" maxrel {rel.max():.3e} med {np.median(rel[rel>0]) if (rel>0).any() else 0:.3e}"
  print(f"[gate] {tag:<5} M32 vs M16x2: mismatches {nz}/{mine.size} {msg}", flush=True)
  for _ in range(2): pr32(*aB, global_size=(g16, 1, 1), local_size=LS)
  dev.synchronize()
  b32 = 1e9
  for _ in range(10):
    t0 = time.perf_counter()
    pr32(*aB, global_size=(g16, 1, 1), local_size=LS)
    dev.synchronize()
    b32 = min(b32, time.perf_counter() - t0)
  print(f"[bench] {tag:<5} M16x2 {b16*1e3:7.3f} ms | M32 {b32*1e3:7.3f} ms | speedup {b16/b32:5.2f}x | "
        f"M32 per-pass {wb/b32/1e9:6.1f} GB/s | 32-tok amortized {2*wb/b32/1e9:6.1f} GB/s | "
        f"(M32 chunk-time x{inst}: {b32*1e3*inst:6.2f} ms/32rows)", flush=True)

print("[p6] ALL BIT-IDENTICAL" if ALL_OK else "[p6] DIFFERENCES FOUND", flush=True)
sys.exit(0 if ALL_OK else 1)
