# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7E6 M32-side funnel: C3 seed, 256-tok M32 run; capture rec + qkv32 + araw/braw
per pfs16 launch (block 0 only). Offline: exact per-window oracle from M32's own
inputs vs its rec; plus SC per-window states for the cross-diff."""
import os, sys
os.environ.setdefault("DEV", "NV")
os.environ["PF_SUPER"] = "0"; os.environ["PF_DFILL"] = "0"; os.environ["PF_G3SC"] = "0"; os.environ["PF_GEMM3"] = "1"
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from trunk_w1c import TrunkEngineW1C
import pf_prefill

E = TrunkEngineW1C(theta=1e7)
P, d = E.P, E.P.d
import json as _j; ids = np.array(_j.load(open("~/ids8k.json"))[:256], dtype=np.int32)
SNAPC = {i: np.load(f"~/snap100k/conv_{i}.npy").reshape(3,10240) for i in E.gdn_idx}

def reset():
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32)); dev.synchronize()
  for i in E.attn_idx:
    P.win_up(f"kv{i}", 0, np.zeros(2*4*100352*256, dtype=np.uint8)); P._keep.clear()
  for i in E.gdn_idx:
    z = np.zeros_like(SNAPC[i]); z[2] = SNAPC[i][2]
    P.win_up(f"conv{i}_0", 0, z.reshape(-1))
    P.win_up(f"conv{i}_1", 0, np.zeros(3*10240, dtype=np.float32))
    P.win_up(f"rec{i}", 0, np.zeros(48*128*128, dtype=np.float32))
  dev.synchronize(); P._keep.clear()
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32)); dev.synchronize()

class G: pass
CONVNAME = {id(d[f"conv{i}_0"]): i for i in E.gdn_idx}
CAP = {}
reset()
pf_prefill.ensure(E)
_pfplan = E._pf_plan
_LC = {}
def wrap(orig):
  def w(*a, **k):
    orig(*a, **k); dev.synchronize()
    bi = CONVNAME.get(id(a[0]))
    if bi is None: return
    _LC[bi] = _LC.get(bi, 0) + 1
    L = _LC[bi] - 1
    if bi != E.gdn_idx[0]:
      if L == 15: CAP[("recfinal", bi)] = P.down(f"rec{bi}", (48*128*128,), np.float32).copy()
      P._keep.clear(); return
    CAP[("rec", L)] = P.down(f"rec{bi}", (48*128*128,), np.float32).copy()
    CAP[("qkv32", L)] = P.down("qkv32", (32,10240), np.float16).copy()
    CAP[("araw32", L)] = P.down("araw32", (32*48,), np.float32).copy()
    CAP[("braw32", L)] = P.down("braw32", (32*48,), np.float32).copy()
    CAP[("conv0", L)] = P.down(f"conv{bi}_0", (3*10240,), np.float32).copy()
    if L < 2:
      CAP[("q", L)] = P.down("q", (48*128,), np.float32).copy()
      CAP[("k", L)] = P.down("k", (48*128,), np.float32).copy()
      CAP[("v", L)] = P.down("v", (48*128,), np.float32).copy()
      CAP[("core", L)] = P.down("core", (48*128,), np.float32).copy()
    P._keep.clear()
  return w
_pl = []
for ent in _pfplan:
  prog = ent[0]
  _pl.append(((wrap(prog) if prog is E.pr["pfs16"] else prog),) + tuple(ent[1:]))
E._pf_plan = _pl
pf_prefill.prefill_batch(E, G, ids); dev.synchronize()
E._pf_plan = _pfplan
np.save("~/p7e6_m32cap.npy", CAP, allow_pickle=True)
print("[m32cap saved]", len(CAP), "entries", flush=True)
