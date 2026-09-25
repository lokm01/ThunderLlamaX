# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7E6 forensic funnel: find the first diverging (block, window) under the C3
seed (convlive row2 only, rec=0) at 256 tokens (single chunk), SC vs M32.
Phase 1: wrap pfcz (SC) / pfs16 (M32), snapshot z per (block, window-32) +
rec per block -> divergence matrix. Phase 2: for first guilty block b*, fp64
oracle scan over dumped pfca outputs (all 4 windows, tokens 0-63 each) vs sco
-> first token. Exit leaves GPU clean."""
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
NCH = 10240
SNAPC = {i: np.load(f"~/snap100k/conv_{i}.npy").reshape(3,10240) for i in E.gdn_idx}
gi = E.gdn_idx[0]
print("SNAPC row absmax:", [float(np.abs(SNAPC[gi][r]).max()) for r in range(3)], flush=True)
NC, C, LDK, LDTC, HCB = 4, 64, 136, 72, 263696

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
    elif V == "C5": z[:] = cv[:]*0.01
    P.win_up(f"conv{i}_0", 0, z.reshape(-1))
    P.win_up(f"conv{i}_1", 0, np.zeros(3*10240, dtype=np.float32))
    P.win_up(f"rec{i}", 0, np.zeros(48*128*128, dtype=np.float32))
  dev.synchronize(); P._keep.clear()
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32)); dev.synchronize()

class G: pass
RECNAME = {id(d[f"rec{i}"]): i for i in E.gdn_idx}
CONVNAME = {id(d[f"conv{i}_0"]): i for i in E.gdn_idx}
ZSC = {}   # (world, blkidx_pos, w32) -> z rows (32,6144) fp32
REC = {}   # (world, blkpos) -> rec (48,128,128) fp32
SNIFF = {}

_MCNT = {}
def m32wrap(orig):
  def w(*a, **k):
    orig(*a, **k); dev.synchronize()
    bi = CONVNAME.get(id(a[0]))
    _MCNT[bi] = _MCNT.get(bi, 0) + 1
    p = (_MCNT[bi] - 1) % 8
    zz = P.down("zsc", (256,6144), np.float16).copy()
    ZSC[("m32", bi, p)] = zz[p*32:(p+1)*32].astype(np.float32)
    if bi is not None and p == 7: REC[("m32", bi)] = P.down(f"rec{bi}", (48*128*128,), np.float32).copy()
    P._keep.clear()
  return w

os.environ["PF_SUPER"] = "1"
reset()
pf_prefill.ensure(E); pf_prefill.ensure_sc(E)
plan = E._pfsc_plan
NPFCZ = 0
def scwrap(entry):
  prog = entry[0]; args = entry[1:]
  nm = getattr(prog, "name", "")
  def w(*a, **k):
    prog(*a, **k)
    if "pfcz" in str(nm):
      dev.synchronize()
      global NPFCZ
      bi = CONVNAME.get(id(a[5]))
      zz = P.down("zsc", (256,6144), np.float16).copy()
      ZSC[("sc", bi, -1)] = zz.astype(np.float32)
      REC[("sc", bi)] = P.down(f"rec{bi}", (48*128*128,), np.float32).copy()
      SNIFF[("sco", bi)] = P.down("sco", (256,6144), np.float32).copy()
      NPFCZ += 1
      if NPFCZ == 1:
        SNIFF["scscr"] = P.down("scscr", (48*NC*HCB//2,), np.uint16).copy()
        SNIFF["araw"] = P.down("arawsc", (256*48,), np.float32).copy()
        SNIFF["braw"] = P.down("brawsc", (256*48,), np.float32).copy()
        SNIFF["qkv"] = P.down("qkvsc", (256,10240), np.float16).copy()
        SNIFF["conv0"] = P.down(f"conv{E.gdn_idx[0]}_0", (3*10240,), np.float32).copy()
      P._keep.clear()
  return w
E._pfsc_plan = [(scwrap(ent),) + ent[1:] for ent in plan]
pf_prefill.prefill_batch(E, G, ids); dev.synchronize()
E._pfsc_plan = plan
print("[sc world done]", flush=True)

os.environ["PF_SUPER"] = "0"
reset()
_pfplan = E._pf_plan
_m32plan = []
for ent in _pfplan:
  prog = ent[0]
  _m32plan.append(((m32wrap(prog) if prog is E.pr["pfs16"] else prog),) + tuple(ent[1:]))
E._pf_plan = _m32plan
pf_prefill.prefill_batch(E, G, ids); dev.synchronize()
E._pf_plan = _pfplan
print("[m32 world done]", flush=True)

# ---- phase 1: divergence matrix over (blockpos, w32) ----
print("\n=== PHASE 1: z maxabs per (gdn-block-position, 32-tok window) ===", flush=True)
rows = []
first = None
for bp, i in enumerate(E.gdn_idx):
  zsc_ = ZSC.get(("sc", i, -1))
  if zsc_ is None: continue
  mx = []
  for p in range(8):
    zm = ZSC.get(("m32", i, p))
    if zm is None: mx.append(-1.0); continue
    mx.append(float(np.abs(zm - zsc_[p*32:(p+1)*32]).max()))
  rows.append((bp, i, mx))
  flag = " <== FIRST" if first is None and max(mx) > 2e-2 and bp > 0 or first is None and max(mx) > 2e-2 else ""
  if first is None and max(mx) > 2e-2: first = (bp, i, mx)
  print(f"blk {bp:2d} (i={i:2d}): " + " ".join(f"{v:.1e}" for v in mx) + flag, flush=True)
for bp, i, mx in rows[:3]:
  print(f"detail blk {bp}: " + ", ".join(f"{v:.3e}" for v in mx), flush=True)
print("rec final compare (block 0..5):", flush=True)
for bp, i in enumerate(E.gdn_idx[:6]):
  if ("sc", i) in REC and ("m32", i) in REC:
    dd = np.abs(REC[("m32", i)].astype(np.float64) - REC[("sc", i)].astype(np.float64))
    print(f"  blk {bp} rec maxabs {dd.max():.3e} medrel {np.median(dd/np.maximum(np.abs(REC[('m32',i)]),1e-9)):.3e}", flush=True)
np.save("~/p7e6_funnel.npy", {"ZSC": {k: v for k, v in list(ZSC.items())[:600]}, "REC": REC, "SNIFF": SNIFF}, allow_pickle=True)
print("[saved]", flush=True)
