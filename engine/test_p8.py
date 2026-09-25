# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P8: NTILE=32 grid-growth twins vs shipped M32 cubins (CTASM item 2).
Gate: NT32 output BIT-IDENTICAL to the shipped M32 kernel (per-element k-order
unchanged; N-tile split only). r7 twins validated vs the SAME classic reference
(P7B law: r7 bit-identical to classic). Bench: min-of-10 synced, both grids.
The process env controls the carveout (NV_SMEM_CFG_AUTO*).
Usage: ~/tg311/bin/python -u test_p8.py [filter ...]"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev, parse_gguf, iq3_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
PACKED, PACKED7 = f"{BASE}/packed", f"{BASE}/packed7"
LS = (256, 1, 1)
P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

print(f"[env] NV_SMEM_CFG_AUTO={os.getenv(chr(78)+chr(86)+chr(95)+chr(83)+chr(77)+chr(69)+chr(77)+chr(95)+chr(67)+chr(70)+chr(71)+chr(95)+chr(65)+chr(85)+chr(84)+chr(79),chr(48))} TGT={os.getenv(chr(78)+chr(86)+chr(95)+chr(83)+chr(77)+chr(69)+chr(77)+chr(95)+chr(67)+chr(70)+chr(71)+chr(95)+chr(65)+chr(85)+chr(84)+chr(79)+chr(95)+chr(84)+chr(71)+chr(84),chr(50))} NAMES={os.getenv(chr(78)+chr(86)+chr(95)+chr(83)+chr(77)+chr(69)+chr(77)+chr(95)+chr(67)+chr(70)+chr(71)+chr(95)+chr(65)+chr(85)+chr(84)+chr(79)+chr(95)+chr(78)+chr(65)+chr(77)+chr(69)+chr(83),chr(45))}", flush=True)
P.up("gridf", iq3_grid_f32())
dev.synchronize()

ds, infos = parse_gguf()
attn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" in infos]
gdn_idx = [i for i in range(64) if i not in set(attn_idx)]
oq_type = {i: infos[f"blk.{i}.ssm_out.weight"][0] for i in gdn_idx}
G0 = gdn_idx[0]
O18 = next(i for i in gdn_idx if oq_type[i] == 18)
print(f"[blocks] gdn0={G0} o18={O18}", flush=True)

P.up("w_fd", np.load(f"{PACKED}/fd{G0}.npy"))
P.up("w_o18", np.load(f"{PACKED}/out{O18}.npy"))
P.up("w_fg", np.load(f"{PACKED}/fg{G0}.npy"))
P.up("w_fu", np.load(f"{PACKED}/fu{G0}.npy"))
P.up("w_fd7", np.load(f"{PACKED7}/fd{G0}.npy"))
P.up("w_o7", np.load(f"{PACKED7}/out{O18}.npy"))
rng = np.random.default_rng(7)
P.up("res_in", (rng.standard_normal((32, 5120)) * 4.0).astype(np.float32).reshape(-1))
dev.synchronize()

# tag, shipped, nt32, wnames, nd, kd, wbytes, g_ship, g_nt32, inst, res, ls_nt32
CLASSES = [
  ("iq3d",    "pfg_iq3d_m32_res_hm_nw8k128", "pfg_iq3d_m32_res_nt32_hm_nw4k128", ("w_fd",),  5120, 17408, 6664*5120,    80, 160, 64, True,  (128,1,1)),
  ("iq3o",    "pfg_iq3o_m32_hm_nw8k128",     "pfg_iq3o_m32_nt32_hm_nw4k128",     ("w_o18",), 5120, 6144,  2352*5120,    80, 160, 24, False, (128,1,1)),
  ("ffn",     "pfg_ffn_m32_hm_nw8k128",      "pfg_ffn_m32_nt32_hm_nw4k128",      ("w_fg","w_fu"), 17408, 5120, 1960*17408*2, 272, 544, 64, False, (128,1,1)),
  ("iq3d_r7", "pfg3_iq3d_r7_m32_nw8k128",    "pfg3_iq3d_r7_m32_nt32_nw4k128",    ("w_fd7",), 5120, 17408, 6664*5120,    80, 160, 64, True,  (128,1,1)),
  ("iq3o_r7", "pfg3_iq3o_r7_m32_nw8k128",    "pfg3_iq3o_r7_m32_nt32_nw4k128",    ("w_o7",),  5120, 6144,  2352*5120,    80, 160, 24, False, (128,1,1)),
]

only = sys.argv[1:] or None
ALL_OK = True
for tag, ship, nt32, wn, nd, kd, wb, gs, gn, inst, res, lsn in CLASSES:
  if only and not any(o in tag for o in only): continue
  x = (rng.standard_normal((32, kd)) * 0.8).astype(np.float16)
  P.up("x32", x.reshape(-1))
  elt = 4 if res else 2
  wargs = tuple(P.d[w] for w in wn)
  def args(wa, out):
    return wa + (P.d["gridf"], P.d["x32"], out) if not res else wa + (P.d["gridf"], P.d["x32"], P.d["res_in"], out)
  # reference: shipped M32
  P.poison("outA", 32*nd*elt, np.float32 if res else np.float16, 7.7e31 if res else 7.7)
  dev.synchronize()
  pr_s = prog(ship)
  pr_s(*args(wargs, P.d["outA"]), global_size=(gs,1,1), local_size=LS)
  dev.synchronize()
  ref = P.down("outA", (32, nd), np.float32 if res else np.float16).astype(np.float32)
  # nt32 twin
  P.poison("outB", 32*nd*elt, np.float32 if res else np.float16, 7.7e31 if res else 7.7)
  dev.synchronize()
  pr_n = prog(nt32)
  pr_n(*args(wargs, P.d["outB"]), global_size=(gn,1,1), local_size=lsn)
  dev.synchronize()
  mine = P.down("outB", (32, nd), np.float32 if res else np.float16).astype(np.float32)
  nz = int((mine != ref).sum())
  ALL_OK &= (nz == 0)
  msg = "BIT-IDENTICAL" if nz == 0 else f"DIFF {nz}/{mine.size}"
  if nz:
    d = np.abs(mine - ref); rel = d / np.maximum(np.abs(ref), 1e-6)
    msg += f" maxrel {rel.max():.3e}"
  # bench both
  def b(fn):
    for _ in range(2): fn()
    dev.synchronize()
    best = 1e9
    for _ in range(10):
      t0 = time.perf_counter(); fn(); dev.synchronize()
      best = min(best, time.perf_counter() - t0)
    return best
  bs = b(lambda: pr_s(*args(wargs, P.d["outA"]), global_size=(gs,1,1), local_size=LS))
  bn = b(lambda: pr_n(*args(wargs, P.d["outB"]), global_size=(gn,1,1), local_size=lsn))
  print(f"[p8] {tag:<8} {msg:<44} | ship {bs*1e3:6.3f} ms ({2*wb/bs/1e9:5.0f} GB/s am) | nt32 {bn*1e3:6.3f} ms ({2*wb/bn/1e9:5.0f} GB/s am) | x{bs/bn:.2f} | chunk {inst}*ship {bs*1e3*inst:6.1f} -> {bn*1e3*inst:6.1f} ms", flush=True)

print("[p8] ALL BIT-IDENTICAL" if ALL_OK else "[p8] DIFFERENCES FOUND", flush=True)
sys.exit(0 if ALL_OK else 1)
