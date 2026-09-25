# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P11 G1: K-split-2 twins + combine vs shipped classic M32 cubins.
Gate: correctness relerr vs the shipped kernel (Tier-2 expected — fp32 partial
reassociation; report med/max), determinism x2. Bench: min-of-10 synced, BOTH
grids + the combine launch (the in-plan cost is gemm+combine).
Decision gate: iq3d < 1.2x -> bank-and-stop the GEMM family.
Usage: ~/tg311/bin/python -u test_p11.py [filter ...]  (env: NV_SMEM_CFG_AUTO*)
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev, parse_gguf
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
PACKED = f"{BASE}/packed"
LS = (256, 1, 1)
P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

print(f"[env] AUTO={os.getenv('NV_SMEM_CFG_AUTO','0')} TGT={os.getenv('NV_SMEM_CFG_AUTO_TGT','2')} NAMES={os.getenv('NV_SMEM_CFG_AUTO_NAMES','-')}", flush=True)

ds, infos = parse_gguf()
attn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" in infos]
gdn_idx = [i for i in range(64) if i not in set(attn_idx)]
oq_type = {i: infos[f"blk.{i}.ssm_out.weight"][0] for i in gdn_idx}
G0 = gdn_idx[0]
O18 = next(i for i in gdn_idx if oq_type[i] == 18)
A0 = attn_idx[0]
print(f"[blocks] gdn0={G0} o18={O18} attn0={A0}", flush=True)

# real weights (classic packed layout)
P.up("w_fd", np.load(f"{PACKED}/fd{G0}.npy"))
P.up("w_o18", np.load(f"{PACKED}/out{O18}.npy"))
from engine0 import iq3_grid_f32
P.up("gridf", iq3_grid_f32())
rng = np.random.default_rng(7)
P.up("res_in", (rng.standard_normal((32, 5120)) * 4.0).astype(np.float32).reshape(-1))
dev.synchronize()

# tag, shipped, ks2, comb, wname, kd, wbytes, g_ship, res
CLASSES = [
  ("iq3d", "pfg_iq3d_m32_res_hm_nw8k128", "pfg_iq3d_m32_ks2_res_hm_nw8k128", "pfg_ks2c_res_hm", "w_fd", 17408, 6664*5120, 80, True, 64),
  ("iq3o", "pfg_iq3o_m32_hm_nw8k128",     "pfg_iq3o_m32_ks2_hm_nw8k128",     "pfg_ks2c_hm",     "w_o18", 6144, 2352*5120, 80, False, 24),
]

only = sys.argv[1:] or None
for tag, ship, ks2, comb, wn, kd, wb, gs, res, inst in CLASSES:
  if only and not any(o in tag for o in only): continue
  nd = 5120
  x = (rng.standard_normal((32, kd)) * 0.8).astype(np.float16)
  P.up("x32", x.reshape(-1))
  elt = 4 if res else 2
  w = P.d[wn]
  def args_ship(out):
    return (w, P.d["gridf"], P.d["x32"], P.d["res_in"], out) if res else (w, P.d["gridf"], P.d["x32"], out)
  P.poison("outA", 32*nd*elt, np.float32 if res else np.float16, 7.7e31 if res else 7.7)
  dev.synchronize()
  pr_s = prog(ship)
  pr_s(*args_ship(P.d["outA"]), global_size=(gs,1,1), local_size=LS)
  dev.synchronize()
  ref = P.down("outA", (32, nd), np.float32 if res else np.float16).astype(np.float32)

  # ks2 + combine (partials poison-first; combine per-class args)
  P.poison("pfks", 2*32*nd*4, np.float32, 7.7e31)
  P.poison("outB", 32*nd*elt, np.float32 if res else np.float16, 7.7e31 if res else 7.7)
  dev.synchronize()
  pr_k = prog(ks2); pr_c = prog(comb)
  if res:
    pr_k(w, P.d["gridf"], P.d["x32"], P.d["res_in"], P.d["pfks"], global_size=(2*gs,1,1), local_size=LS)
    pr_c(P.d["pfks"], P.d["res_in"], P.d["outB"], global_size=(160,1,1), local_size=LS)
  else:
    # LAW: KS keeps the RES||KS 5-arg signature (res16 slot unused) -> dummy arg
    pr_k(w, P.d["gridf"], P.d["x32"], P.d["res_in"], P.d["pfks"], global_size=(2*gs,1,1), local_size=LS)
    pr_c(P.d["pfks"], P.d["outB"], global_size=(160,1,1), local_size=LS)
  dev.synchronize()
  pf = P.down("pfks", (2, 32, nd), np.float32)
  print(f"[p11dbg] {tag} partials poison {int((pf == 7.7e31).sum())}/{pf.size} plane0max {np.abs(pf[0]).max():.3e} plane1max {np.abs(pf[1]).max():.3e}", flush=True)
  mine = P.down("outB", (32, nd), np.float32 if res else np.float16).astype(np.float32)
  act = np.abs(ref) > 1e-6
  e = np.abs(mine[act] - ref[act]) / np.abs(ref[act])
  # determinism x2
  pr_c(*( (P.d["pfks"], P.d["res_in"], P.d["outA"]) if res else (P.d["pfks"], P.d["outA"]) ), global_size=(160,1,1), local_size=LS)
  dev.synchronize()
  m2 = P.down("outA", (32, nd), np.float32 if res else np.float16).astype(np.float32)
  det = "det-ok" if np.array_equal(mine, m2) else "DET-FAIL"

  def b(fn):
    for _ in range(2): fn()
    dev.synchronize()
    best = 1e9
    for _ in range(10):
      t0 = time.perf_counter(); fn(); dev.synchronize()
      best = min(best, time.perf_counter() - t0)
    return best
  def run_ship(): pr_s(*args_ship(P.d["outA"]), global_size=(gs,1,1), local_size=LS)
  def run_ks2():
    if res:
      pr_k(w, P.d["gridf"], P.d["x32"], P.d["res_in"], P.d["pfks"], global_size=(2*gs,1,1), local_size=LS)
      pr_c(P.d["pfks"], P.d["res_in"], P.d["outB"], global_size=(160,1,1), local_size=LS)
    else:
      pr_k(w, P.d["gridf"], P.d["x32"], P.d["res_in"], P.d["pfks"], global_size=(2*gs,1,1), local_size=LS)
      pr_c(P.d["pfks"], P.d["outB"], global_size=(160,1,1), local_size=LS)
  bs = b(run_ship); bk = b(run_ks2)
  print(f"[p11] {tag:<5} relerr med {np.median(e):.2e} max {np.max(e):.2e} {det} | "
        f"ship {bs*1e3:6.3f} ms ({2*wb/bs/1e9:5.0f} GB/s am) | ks2+comb {bk*1e3:6.3f} ms ({2*wb/bk/1e9:5.0f} GB/s am) | "
        f"x{bs/bk:.2f} | chunk {inst}*ship {bs*1e3*inst:5.1f} -> {bk*1e3*inst:5.1f} ms", flush=True)

print("[p11] done", flush=True)
