# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P5 validate+bench: the restructured FFN/IQ3 pGEMM cubins vs the SHIPPED
P1/P4 kernels on real packed weights. non-H2 (SYNCW/SUB) must be BIT-IDENTICAL
to shipped; H2 validated F-norm class + poison-free. Synced timing min-of-10.
Usage: ~/tg311/bin/python -u test_p5.py [filter ...]"""
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
M = 16
P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

# ---- iq3 LUT: [0..2048) grid as half2 pairs (1024 halfs), [2048..4096) sign masks
g = np.asarray(iq3_grid_f32(), dtype=np.float32).reshape(256, 4)
gh = g.astype(np.float16)
if not np.array_equal(gh.astype(np.float32), g):
  bad = int((gh.astype(np.float32) != g).sum())
  print(f"[lut] WARNING: {bad}/1024 grid values NOT exact in fp16 (extra rounding)", flush=True)
else:
  print("[lut] grid exactly representable in fp16 (no extra rounding from gh2)", flush=True)
h16 = gh.reshape(-1).view(np.uint16)
sg = np.zeros((128, 4), np.uint32)
for sidx in range(128):
  bits = [(sidx >> j) & 1 for j in range(7)] + [bin(sidx).count("1") & 1]
  for pr in range(4):
    sg[sidx, pr] = (0x8000 if bits[2*pr] else 0) | (0x80000000 if bits[2*pr+1] else 0)
lut32 = np.zeros(1024, np.uint32)
lut32[:512] = h16.view(np.uint32)
lut32[512:1024] = sg.reshape(-1)
P.up("iq3lut", lut32)
P.up("gridf", iq3_grid_f32())
dev.synchronize()

ds, infos = parse_gguf()
attn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" in infos]
gdn_idx = [i for i in range(64) if i not in set(attn_idx)]
qtype = {i: infos[f"blk.{i}.attn_q.weight"][0] for i in attn_idx}
oq_type = {i: infos[f"blk.{i}.ssm_out.weight"][0] for i in gdn_idx}
G0 = gdn_idx[0]
A18 = next(i for i in attn_idx if qtype[i] == 18)
O18 = next(i for i in gdn_idx if oq_type[i] == 18)
print(f"[blocks] gdn0={G0} a18={A18} o18={O18}", flush=True)

P.up("w_fg", np.load(f"{PACKED}/fg{G0}.npy"))
P.up("w_fu", np.load(f"{PACKED}/fu{G0}.npy"))
P.up("w_fd", np.load(f"{PACKED}/fd{G0}.npy"))
P.up("w_q3", np.load(f"{PACKED}/q{A18}.npy"))
P.up("w_o18", np.load(f"{PACKED}/out{O18}.npy"))
rng = np.random.default_rng(7)
P.up("res_in", (rng.standard_normal((M, 5120)) * 4.0).astype(np.float32).reshape(-1))
dev.synchronize()

# class -> dict(shipped name, wnames, nd, kd, rowb, inst, res?)
CLASSES = [
  ("ffn",  "pfg_ffn_hm_nw8k128",     ("w_fg", "w_fu"), 17408, 5120, 1960, 64, False),
  ("iq3d", "pfg_iq3d_res_hm_nw8k128", ("w_fd",),        5120, 17408, 6664, 64, True),
  ("iq3q", "pfg_iq3q_hm_nw8k128",     ("w_q3",),       12288, 5120, 1960, 8, False),
  ("iq3o", "pfg_iq3o_hm_nw8k128",     ("w_o18",),       5120, 6144, 2352, 24, False),
]
CANDS = ["pfg_ffnX_sw_nw8k128", "pfg_ffnX_h2sw_nw8k128", "pfg_ffnX_h2sw_nw8k64s2",
         "pfg_ffnX_h2sw_nw8k32s4", "pfg_ffnX_h2sw_nw32k32", "pfg_ffnX_h2sw_nw16k64",
         "pfg_iq3dX_h2sw_nw8k128", "pfg_iq3qX_h2sw_nw8k128", "pfg_iq3oX_h2sw_nw8k128",
         "pfg_ffnX_h2_nw8k128", "pfg_iq3dX_h2_nw8k128", "pfg_iq3qX_h2_nw8k128", "pfg_iq3oX_h2_nw8k128"]
# (nthr, ntile) per candidate for grid/local_size
NT = {"pfg_ffnX_sw_nw8k128": (256, 64), "pfg_ffnX_h2sw_nw8k128": (256, 64),
      "pfg_ffnX_h2sw_nw8k64s2": (256, 128), "pfg_ffnX_h2sw_nw8k32s4": (256, 256),
      "pfg_ffnX_h2sw_nw32k32": (1024, 256), "pfg_ffnX_h2sw_nw16k64": (512, 128),
      "pfg_iq3dX_h2sw_nw8k128": (256, 64), "pfg_iq3qX_h2sw_nw8k128": (256, 64),
      "pfg_iq3oX_h2sw_nw8k128": (256, 64),
      "pfg_ffnX_h2_nw8k128": (256, 64), "pfg_iq3dX_h2_nw8k128": (256, 64),
      "pfg_iq3qX_h2_nw8k128": (256, 64), "pfg_iq3oX_h2_nw8k128": (256, 64)}

only = sys.argv[1:] or None
ALL_OK = True
for tag, shipped, wn, nd, kd, rowb, inst, res in CLASSES:
  if only and not any(o in (tag + shipped) or o in " ".join(CANDS) for o in only): continue
  x = (rng.standard_normal((M, kd)) * 0.8).astype(np.float16)
  P.up("x16b", x.reshape(-1))
  wargs = tuple(P.d[w] for w in wn)
  # shipped reference
  P.poison("outA", M*nd*(4 if res else 2), np.float32 if res else np.float16, 7.7e31 if res else 7.7)
  dev.synchronize()
  argsA = wargs + (P.d["gridf"], P.d["x16b"], P.d["res_in"], P.d["outA"]) if res else \
          wargs + (P.d["gridf"], P.d["x16b"], P.d["outA"])
  gridA = nd // 64
  prS = prog(shipped)
  prS(*argsA, global_size=(gridA, 1, 1), local_size=LS); dev.synchronize()
  ref = P.down("outA", (M, nd), np.float32 if res else np.float16).astype(np.float32)
  # shipped bench
  best = 1e9
  for _ in range(2): prS(*argsA, global_size=(gridA, 1, 1), local_size=LS)
  dev.synchronize()
  for _ in range(10):
    t0 = time.perf_counter(); prS(*argsA, global_size=(gridA, 1, 1), local_size=LS); dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  mult = 2 if tag == "ffn" else 1
  gbs = rowb*nd*mult/best/1e9
  print(f"[bench] {tag:<5} SHIPPED {shipped:<26} {best*1e3:7.3f} ms {gbs:6.1f} GB/s  (chunk {best*1e3*inst:6.2f} ms x{inst})", flush=True)
  for name in CANDS:
    if tag not in name and not (tag == "ffn" and name.startswith("pfg_ffn")): continue
    if tag != "ffn" and tag not in name: continue
    if only and not any(o in name for o in only): continue
    nthr, ntile = NT[name]
    grid = nd // ntile
    P.poison("outB", M*nd*(4 if res else 2), np.float32 if res else np.float16, 7.7e31 if res else 7.7)
    dev.synchronize()
    argsB = wargs + (P.d["gridf"], P.d["x16b"], P.d["res_in"], P.d["outB"]) if res else \
            wargs + (P.d["gridf"], P.d["x16b"], P.d["outB"])
    if "h2" in name: argsB += (P.d["iq3lut"],)
    prN = prog(name)
    try:
      prN(*argsB, global_size=(grid, 1, 1), local_size=(nthr, 1, 1)); dev.synchronize()
    except Exception as e:
      print(f"[val] {name}: FAULT {type(e).__name__}", flush=True); ALL_OK = False; continue
    mine = P.down("outB", (M, nd), np.float32 if res else np.float16)
    mz = mine.astype(np.float32)
    pz = int((np.abs(mz) > 1e30).sum())
    if "h2" not in name:
      nz = int((mine.astype(np.float32) != ref).sum())
      ok = nz == 0 and pz == 0
      print(f"[val] {name}: BIT-DIFF vs shipped nz={nz}/{ref.size} poison={pz} -> {'PASS' if ok else 'FAIL'}", flush=True)
      ALL_OK &= ok
    else:
      fn = np.linalg.norm(mz - ref) / max(np.linalg.norm(ref), 1e-9)
      act = np.abs(ref) > 1e-6
      med = float(np.median(np.abs(mz[act] - ref[act]) / np.abs(ref[act]))) if act.sum() else 0.0
      ok = fn <= 3e-3 and pz == 0
      print(f"[val] {name}: F {fn:.3e} med {med:.3e} poison={pz} -> {'PASS' if ok else 'FAIL'}", flush=True)
      ALL_OK &= ok
    b2 = 1e9
    try:
      for _ in range(2): prN(*argsB, global_size=(grid, 1, 1), local_size=(nthr, 1, 1))
      dev.synchronize()
      for _ in range(10):
        t0 = time.perf_counter(); prN(*argsB, global_size=(grid, 1, 1), local_size=(nthr, 1, 1)); dev.synchronize()
        b2 = min(b2, time.perf_counter() - t0)
    except Exception as e:
      print(f"[bench] {name}: FAULT {type(e).__name__}", flush=True); ALL_OK = False; continue
    g2 = rowb*nd*mult/b2/1e9
    print(f"[bench] {name:<28} {b2*1e3:7.3f} ms {g2:6.1f} GB/s  {g2/gbs:5.2f}x  (chunk {b2*1e3*inst:6.2f} ms)", flush=True)
print(f"[test_p5] {'ALL PASS' if ALL_OK else 'FAILURES PRESENT'}", flush=True)
