# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R3 lookup_nw32 standalone validation: kernel proposals vs offline analyzer
semantics on the REAL 100k stream, at sample positions (hits + misses)."""
import os, sys, json
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
P = Bufs()
lib = open(f"{BASE}/lookup_nw32.cubin", "rb").read()
pr = NVProgram(dev, TinyELF(lib=lib, name="lookup_nw32", target=dev.renderer.target, signature=tuple()))
SNAP = "~/snap100k"
ids = np.load(f"{SNAP}/ids.npy").tolist()
out = json.load(open("~/tinygrad-metal/spec_base_100k.json"))
CTX = 100352
P.up("tok_hist", np.full(CTX + 128, -1, dtype=np.int32))
P.up("l_hist", np.zeros(1 << 20, dtype=np.int32))
P.up("pos_slot", np.zeros(1, dtype=np.int32))
P.up("cur_slot", np.zeros(1, dtype=np.int32))
P.up("cyc_slot", np.zeros(1, dtype=np.int32))
P.poison("dring0", 4, np.int32, -777)
P.poison("dring1", 4, np.int32, -777)
dev.synchronize(); d = P.d

def offline_best(fed, cur):
  pos = len(fed)
  S = np.array(fed[max(0, pos-7):] + [cur], dtype=np.int64)
  W = len(S)
  imax = pos - W - 3
  if imax < 0: return (0, -1, None, None)
  fed_a = np.array(fed, dtype=np.int64)
  ok = np.ones(imax+1, dtype=bool); l = np.zeros(imax+1, dtype=np.int64)
  for u in range(W):
    ok &= (fed_a[u:u+imax+1] == S[u]); l += ok
  cand = np.nonzero(l >= 6)[0]
  if len(cand) == 0: return (0, -1, None, None)
  ll = l[cand]; best = int(cand[np.lexsort((cand, ll))[-1]]); bl = int(l[best])
  return (bl, best, int(fed_a[best+bl]), int(fed_a[best+bl+1]))

ok_all = True
for j in (0, 3, 5, 12, 20, 33, 40, 47, 55, 57):
  fed = ids + out[:j]
  P.win_up("tok_hist", 0, np.array(fed, dtype=np.int32))
  P.win_up("pos_slot", 0, np.array([len(fed)], dtype=np.int32))
  P.win_up("cur_slot", 0, np.array([out[j]], dtype=np.int32))
  P.poison("dring0", 4, np.int32, -777)
  P.poison("dring1", 4, np.int32, -777)
  P.win_up("l_hist", 0, np.array([0], dtype=np.int32))
  dev.synchronize()
  pr(d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"], d["l_hist"], d["cyc_slot"],
     global_size=(1,1,1), local_size=(1024,1,1), wait=True)
  d0 = int(P.down("dring0", (1,), np.int32)[0]); d1 = int(P.down("dring1", (1,), np.int32)[0])
  lh = int(P.down("l_hist", (1,), np.int32)[0])
  bl, bi, bp1, bp2 = offline_best(fed, out[j])
  hit = bl >= 6
  got_hit = lh >= 7
  ok = (got_hit == hit) and (not hit or (lh - 1 == bl and d0 == bp1 and d1 == bp2))
  ok_all &= ok
  print(f"j={j:3d}: kernel l={lh-1} props=({d0},{d1}) | offline l={bl} i={bi} props=({bp1},{bp2}) -> {'OK' if ok else 'MISMATCH'}", flush=True)
print("[lut_test]", "ALL OK" if ok_all else "FAILURES", flush=True)
sys.exit(0 if ok_all else 1)
