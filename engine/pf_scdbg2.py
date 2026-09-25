# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7E2 no-probe end-state harness: run the REAL prefill_batch (SC path incl.
dfill + head + argmax) with ZERO mid-run instrumentation, then dump the full
world (all 48 rec/conv, kv slices, scratch, logits) to npz. fresh vs ref diff.
Env: SCDBG_DUMP=path; PF_SC_SKIPREF=1 for the fresh variant."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from trunk_w1c import TrunkEngineW1C, LS
import pf_prefill

E = TrunkEngineW1C(theta=1e7)
P, W, d = E.P, E.W, E.P.d

def t1_token(tok):
  P.win_up("tok_slot", 0, np.array([int(tok)], dtype=np.int32))
  if not hasattr(E, "_seq"): E._build_seqs()
  seq = E._seq[0]
  for n_, (p, a, g) in enumerate(seq):
    p(*a, global_size=(g[0],1,1) if isinstance(g, tuple) else (g,1,1),
      local_size=(1024,1,1) if "nw32" in getattr(p, "name", "") else LS,
      wait=(n_ == len(seq)-1))

rng = np.random.default_rng(99)
N = int(os.getenv("SCDBG2_N", "256"))
ids = rng.integers(1000, 200000, size=N).astype(np.int32)
P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
dev.synchronize()

if os.getenv("PF_SC_SKIPREF") != "1":
  import time as _time
  t0 = _time.perf_counter()
  ref_logits = np.zeros((N, 248320), dtype=np.float16)
  for t in range(N):
    t1_token(ids[t])
    ref_logits[t] = P.down("logits", (248320,), np.float16)
  print(f"[ref] T=1 trunk {N} tokens in {_time.perf_counter()-t0:.1f}s", flush=True)
  ref_rec = {i: P.down(f"rec{i}", (48*128*128,), np.float32).copy() for i in E.gdn_idx[:4]}
  ref_conv = {i: P.down(f"conv{i}_0", (3*10240,), np.float32).copy() for i in E.gdn_idx[:4]}
  ref_tok = int(P.down_at("tok_slot", 0, 1)[0])
  KVS = 2*4*256*256*8
  ref_kv = {i: P.down_at(f"kv{i}", 0, KVS, np.uint8).copy() for i in list(E.attn_idx)[:2]}

for i in E.attn_idx:
  P.win_up(f"kv{i}", 0, np.zeros(2*4*100352*256, dtype=np.uint8))
  P._keep.clear()
for i in E.gdn_idx:
  P.win_up(f"conv{i}_0", 0, np.zeros(3*10240, dtype=np.float32))
  P.win_up(f"conv{i}_1", 0, np.zeros(3*10240, dtype=np.float32))
  P.win_up(f"rec{i}", 0, np.zeros(48*128*128, dtype=np.float32))
dev.synchronize(); P._keep.clear()
P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
dev.synchronize()

class G: pass
ct = []
t0 = time.perf_counter()
pf_prefill.prefill_batch(E, G, ids, log=lambda s, **kw: print(f"[sc] {s} {kw}", flush=True), chunk_times=ct)
print(f"[sc] wall {time.perf_counter()-t0:.2f}s", flush=True)

dz = {}
for nm, shp, dt in [("xAsc",(N,5120),np.float32), ("xBsc",(N,5120),np.float32),
                    ("xhsc",(N,5120),np.float16), ("qkvsc",(N,10240),np.float16),
                    ("gatesc",(N,6144),np.float16), ("zsc",(N,6144),np.float16),
                    ("sco",(N,6144),np.float32), ("aosc",(N,6144),np.float16),
                    ("qwsc",(N,6144),np.float16), ("qrowsc",(N,12288),np.float16)]:
  try: dz[nm] = P.down(nm, shp, dt); P._keep.clear()
  except Exception as ex: print(nm, "ERR", ex)
for i in E.gdn_idx:
  dz[f"rec{i}"] = P.down(f"rec{i}", (48*128*128,), np.float32); P._keep.clear()
  dz[f"conv{i}_0"] = P.down(f"conv{i}_0", (3*10240,), np.float32); P._keep.clear()
KVS = 2*4*256*256*8
for i in list(E.attn_idx)[:4]:
  dz[f"kv{i}"] = P.down_at(f"kv{i}", 0, KVS, np.uint8)
dz["logits"] = P.down("logits", (248320,), np.float16)
dz["tok_slot"] = P.down_at("tok_slot", 0, 1, np.int32)
dz["pos_slot"] = P.down_at("pos_slot", 0, 1, np.int32)
np.savez(os.getenv("SCDBG_DUMP", "~/scdbg_end.npz"), **dz)
nanrec = [i for i in E.gdn_idx if np.isnan(dz[f"rec{i}"]).any()]
nanconv = [i for i in E.gdn_idx if np.isnan(dz[f"conv{i}_0"]).any()]
print(f"[end] rec NaN blocks {nanrec[:8]} (n={len(nanrec)}) conv {nanconv[:4]} (n={len(nanconv)}) "
      f"logits nan {int(np.isnan(dz['logits'].astype(np.float32)).sum())} tok {int(dz['tok_slot'][0])} pos {int(dz['pos_slot'][0])}", flush=True)
if os.getenv("PF_SC_SKIPREF") != "1":
  rr = [i for i in E.gdn_idx[:4] if np.isnan(ref_rec[i]).any()]
  print(f"[refchk] ref NaN rec blocks {rr} ref_tok {ref_tok} | sc tok {int(dz['tok_slot'][0])}", flush=True)
  fin = dz["logits"].astype(np.float32); r = ref_logits[N-1].astype(np.float32)
  act = np.abs(r) > 1e-6
  e = np.abs(fin[act] - r[act]) / np.abs(r[act])
  F = np.linalg.norm(fin - r) / max(np.linalg.norm(r), 1e-9)
  print(f"[gate] logits vs T=1: med {np.median(e):.3e} F {F:.3e}", flush=True)
  for i in E.gdn_idx[:4]:
    rm = dz[f"rec{i}"]
    dr = np.linalg.norm(rm - ref_rec[i]) / max(np.linalg.norm(ref_rec[i]), 1e-9)
    print(f"[drift] blk {i}: rec {dr:.3e}", flush=True)
