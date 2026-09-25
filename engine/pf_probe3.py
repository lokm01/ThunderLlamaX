# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7E6 funnel v2: per-16-token-window z diff (SC vs M32) under C3 seed at 256
tokens, blocks 0-3 detail + per-block rec; M32-side input dumps at block 0."""
import os, sys
os.environ.setdefault("DEV", "NV")
os.environ["PF_DFILL"] = "0"; os.environ["PF_G3SC"] = "0"; os.environ["PF_GEMM3"] = "1"
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
NC, C, HCB = 4, 64, 263696

def reset():
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32)); dev.synchronize()
  for i in E.attn_idx:
    P.win_up(f"kv{i}", 0, np.zeros(2*4*100352*256, dtype=np.uint8)); P._keep.clear()
  for i in E.gdn_idx:
    cv = SNAPC[i]
    z = np.zeros_like(cv)
    V = os.environ.get("P7SEED", "C3")
    if V == "C3": z[2] = cv[2]
    elif V == "C1": z[0] = cv[0]
    elif V == "C0": z[:] = cv[:]
    elif V == "CZ": pass
    P.win_up(f"conv{i}_0", 0, z.reshape(-1))
    P.win_up(f"conv{i}_1", 0, np.zeros(3*10240, dtype=np.float32))
    P.win_up(f"rec{i}", 0, np.zeros(48*128*128, dtype=np.float32))
  dev.synchronize(); P._keep.clear()
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32)); dev.synchronize()

class G: pass
CONVNAME = {id(d[f"conv{i}_0"]): i for i in E.gdn_idx}
Z32 = {}; REC = {}; SNIFF = {}

os.environ["PF_SUPER"] = "1"
reset()
pf_prefill.ensure(E); pf_prefill.ensure_sc(E)
plan = E._pfsc_plan
def scwrap(entry):
  prog = entry[0]; nm = str(getattr(prog, "name", ""))
  def w(*a, **k):
    prog(*a, **k)
    if "pfcz" in nm:
      dev.synchronize()
      bi = CONVNAME.get(id(a[5]))
      Z32[("sc", bi)] = P.down("zsc", (256,6144), np.float16).copy().astype(np.float32)
      REC[("sc", bi)] = P.down(f"rec{bi}", (48*128*128,), np.float32).copy()
      SNIFF[("sco", bi)] = P.down("sco", (256,6144), np.float32).copy()
      if bi == E.gdn_idx[0]:
        SNIFF["scscr"] = P.down("scscr", (48*NC*HCB//2,), np.uint16).copy()
        SNIFF["araw"] = P.down("arawsc", (256*48,), np.float32).copy()
        SNIFF["braw"] = P.down("brawsc", (256*48,), np.float32).copy()
        SNIFF["qkv"] = P.down("qkvsc", (256,10240), np.float16).copy()
      P._keep.clear()
  return w
E._pfsc_plan = [(scwrap(ent),) + ent[1:] for ent in plan]
pf_prefill.prefill_batch(E, G, ids); dev.synchronize()
E._pfsc_plan = plan
print("[sc done]", flush=True)

os.environ["PF_SUPER"] = "0"
reset()
_pfplan = E._pf_plan
_LC = {}
def m32wrap(orig):
  def w(*a, **k):
    orig(*a, **k); dev.synchronize()
    bi = CONVNAME.get(id(a[0]))
    if bi is None: return
    _LC[bi] = _LC.get(bi, 0) + 1
    L = _LC[bi] - 1
    half = L % 2; p = L // 2
    t0 = 32*p + 16*half
    nm = "z32" if id(a[14]) == id(d["z32"]) else None
    if nm is None and not hasattr(P.d, "get"): nm = None
    # decide buffer by identity against both candidates
    buf = P.down("z32", (32,6144), np.float16).copy().astype(np.float32)
    Z32[("m32", bi, L)] = (t0, buf[16*half:16*half+16])
    if bi == E.gdn_idx[0] and L == 0:
      SNIFF["araw32"] = P.down("araw32", (32*48,), np.float32).copy()
      SNIFF["braw32"] = P.down("braw32", (32*48,), np.float32).copy()
      SNIFF["qkv32"] = P.down("qkv32", (32,10240), np.float16).copy()
    if L == 15: REC[("m32", bi)] = P.down(f"rec{bi}", (48*128*128,), np.float32).copy()
    P._keep.clear()
  return w
_m32plan = []
for ent in _pfplan:
  prog = ent[0]
  _m32plan.append(((m32wrap(prog) if prog is E.pr["pfs16"] else prog),) + tuple(ent[1:]))
E._pf_plan = _m32plan
pf_prefill.prefill_batch(E, G, ids); dev.synchronize()
E._pf_plan = _pfplan
print("[m32 done]", flush=True)

print("\n=== per-16-tok-window z maxabs (blocks 0-3) ===", flush=True)
for bp in range(6, 18):
  i = E.gdn_idx[bp]
  zs = ZSC = Z32[("sc", i)]
  row = []
  for L in range(16):
    t0, zm = Z32[("m32", i, L)]
    row.append((t0, float(np.abs(zm - zs[t0:t0+16]).max())))
  print(f"blk {bp} (i={i}): " + " ".join(f"{t0}:{v:.1e}" for t0, v in row), flush=True)
for bp, i in enumerate(E.gdn_idx[:6]):
  if ("sc", i) in REC and ("m32", i) in REC:
    dd = np.abs(REC[("m32", i)].astype(np.float64) - REC[("sc", i)].astype(np.float64))
    print(f"rec blk {bp}: maxabs {dd.max():.3e}", flush=True)
np.save("~/p7e6_funnel2.npy", {"Z32": Z32, "REC": REC, "SNIFF": SNIFF}, allow_pickle=True)
print("[saved2]", flush=True)
