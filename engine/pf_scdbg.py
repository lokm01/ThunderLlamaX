# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7E2 milestone-diff harness: run the SC plan with buffer snapshots at chosen
launches, writing JSON lines; run once fresh (SKIPREF) + once post-reference;
diff the two to find the FIRST (launch, buffer) divergence = the clobber root.
Allocation history mirrors pf_fwdsc.py exactly (ref -> reset -> ensure -> ensure_sc).
Env: SCDBG_STEPS="n,n,..." explicit milestones; SCDBG_OUT=path; PF_SC_SKIPREF=1
"""
import os, sys, time, json, zlib
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from trunk_w1c import TrunkEngineW1C, LS
import pf_prefill

E = TrunkEngineW1C(theta=1e7)
P, W, d = E.P, E.W, E.P.d
pr = E.pr
N = int(os.getenv("PF_SC", "256"))
OUT = os.getenv("SCDBG_OUT", "~/scdbg.jsonl")
STEPS = sorted(set(int(x) for x in os.getenv("SCDBG_STEPS", "").split(",") if x))

def t1_token(tok):
  P.win_up("tok_slot", 0, np.array([int(tok)], dtype=np.int32))
  if not hasattr(E, "_seq"): E._build_seqs()
  seq = E._seq[0]
  for n_, (p, a, g) in enumerate(seq):
    p(*a, global_size=(g[0],1,1) if isinstance(g, tuple) else (g,1,1),
      local_size=(1024,1,1) if "nw32" in getattr(p, "name", "") else LS,
      wait=(n_ == len(seq)-1))

rng = np.random.default_rng(99)
ids = rng.integers(1000, 200000, size=N).astype(np.int32)
P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
dev.synchronize()

if os.getenv("PF_SC_SKIPREF") == "1":
  ref_logits = None
else:
  ref_logits = np.zeros((N, 248320), dtype=np.float16)
  for t in range(N):
    t1_token(ids[t])
    ref_logits[t] = P.down("logits", (248320,), np.float16)
  print(f"[ref] T=1 trunk {N} tokens", flush=True)
  ref_rec = {i: P.down(f"rec{i}", (48*128*128,), np.float32).copy() for i in E.gdn_idx[:4]}
  ref_conv = {i: P.down(f"conv{i}_0", (3*10240,), np.float32).copy() for i in E.gdn_idx[:4]}
  ref_tok = int(P.down_at("tok_slot", 0, 1)[0])
  KVS = 2*4*256*256*8
  ref_kv = {i: P.down_at(f"kv{i}", 0, KVS, np.uint8).copy() for i in list(E.attn_idx)[:2]}

# ---- reset world for the SC pass (verbatim pf_fwdsc) ----
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

# ---- build the SC plan (same alloc order as prefill_batch_sc) ----
pf_prefill.ensure(E)
pf_prefill.ensure_sc(E)
plan = E._pfsc_plan
NL = len(plan)
print(f"[plan] {NL} launches; milestones {STEPS[:8]}...{STEPS[-3:] if len(STEPS)>3 else ''}", flush=True)

# full launch-name dump (align launch index -> kernel)
with open(OUT + ".plan", "w") as f:
  for n_, (p, a, g, ls) in enumerate(plan):
    f.write(f"{n_+1}\t{getattr(p,'name','?')}\t{g}\n")

SMALL = os.getenv("SCDBG_SMALL", "1") == "1"
if SMALL:
  NAMED = [("xhsc",(N,5120),np.float16), ("hhxsc",(N,5120),np.float16),
           ("attn_outsc",(N,5120),np.float16), ("gatesc",(N,6144),np.float16),
           ("zsc",(N,6144),np.float16), ("aosc",(N,6144),np.float16),
           ("qwsc",(N,6144),np.float16), ("krowsc",(N,1024),np.float16),
           ("vrowsc",(N,1024),np.float16), ("arawsc",(N*48,),np.float32),
           ("brawsc",(N*48,),np.float32), ("sco",(N,6144),np.float32),
           ("ids_sc",(N,),np.int32), ("pos_arr",(pf_prefill.NC,),np.int32),
           ("pos_w",(16,),np.int32)]
else:
  NAMED = [("xAsc",(N,5120),np.float32), ("xBsc",(N,5120),np.float32),
           ("xhsc",(N,5120),np.float16), ("hhsc",(N,5120),np.float32),
           ("hhxsc",(N,5120),np.float16), ("attn_outsc",(N,5120),np.float16),
           ("qkvsc",(N,10240),np.float16), ("gatesc",(N,6144),np.float16),
           ("zsc",(N,6144),np.float16), ("gactsc",(N,17408),np.float16),
           ("arawsc",(N*48,),np.float32), ("brawsc",(N*48,),np.float32),
           ("qrowsc",(N,12288),np.float16), ("krowsc",(N,1024),np.float16),
           ("vrowsc",(N,1024),np.float16), ("qwsc",(N,6144),np.float16),
           ("aosc",(N,6144),np.float16), ("sco",(N,6144),np.float32),
           ("ids_sc",(N,),np.int32), ("pos_arr",(pf_prefill.NC,),np.int32),
           ("pos_w",(16,),np.int32)]
RECS = [0, 1, 2, 4]
KVS = 2*4*256*256*8

def snap(n, name):
  o = {"n": n, "k": name, "b": {}}
  for nm, shp, dt in NAMED:
    try:
      a = P.down(nm, shp, dt)
      raw = np.ascontiguousarray(a)
      ent = [zlib.crc32(raw.tobytes()) & 0xFFFFFFFF]
      if dt != np.int32:
        v = a.astype(np.float64)
        ent.append(int(np.isnan(v).sum()))
        am = float(np.abs(v).max()) if v.size else 0.0
        ent.append(am if np.isfinite(am) else None)
      o["b"][nm] = ent
    except Exception as ex:
      o["b"][nm] = ["ERR", str(ex)[:30]]
    P._keep.clear()
  for i in RECS:
    try:
      a = P.down(f"rec{i}", (48*128*128,), np.float32)
      o["b"][f"rec{i}"] = [zlib.crc32(np.ascontiguousarray(a).tobytes()) & 0xFFFFFFFF,
                           int(np.isnan(a).sum()), float(np.abs(a).max()) if np.isfinite(np.abs(a).max()) else None]
    except Exception as ex:
      o["b"][f"rec{i}"] = ["ERR", str(ex)[:30]]
    P._keep.clear()
    a = P.down(f"conv{i}_0", (3*10240,), np.float32)
    o["b"][f"conv{i}_0"] = [zlib.crc32(np.ascontiguousarray(a).tobytes()) & 0xFFFFFFFF,
                            int(np.isnan(a).sum()), None]
    P._keep.clear()
  for i in list(E.attn_idx)[:2]:
    a = P.down_at(f"kv{i}", 0, KVS, np.uint8)
    o["b"][f"kv{i}"] = [zlib.crc32(a.tobytes()) & 0xFFFFFFFF, -1, None]
  return o

# ---- instrumented plan run (sync pacing identical to _run_sc_chunk) ----
P.win_up("ids_sc", 0, np.array([int(t) for t in ids], dtype=np.int32))
P.win_up("pos_arr", 0, np.array([64*k for k in range(pf_prefill.NC)], dtype=np.int32))
P.win_up("pos_w", 0, np.array([16*k for k in range(16)], dtype=np.int32))
dev.synchronize()
fout = open(OUT, "w")
t0 = time.perf_counter()
TRUNC = int(os.getenv("SCDBG_TRUNC", "0"))
SKIP = set(int(x) for x in os.getenv("SCDBG_SKIP", "").split(",") if x)
n = 0
for p, a, g, ls in plan:
  if TRUNC and n >= TRUNC:
    break
  if (n + 1) in SKIP:
    print(f"[skip] launch {n+1} {getattr(p,'name','?')}", flush=True)
    n += 1
    continue
  p(*a, global_size=(g, 1, 1), local_size=ls)
  n += 1
  if n <= 8 or n % 32 == 0:
    dev.synchronize()
  if n in STEPS:
    dev.synchronize()
    rec = snap(n, getattr(p, "name", "?"))
    fout.write(json.dumps(rec) + "\n"); fout.flush()
  if n == int(os.getenv("SCDBG_DUMPN", "0")):
    dev.synchronize()
    dz = {}
    for nm, shp, dt in NAMED:
      dz[nm] = P.down(nm, shp, dt); P._keep.clear()
    k0 = list(E.attn_idx)[0]
    dz["kv_first"] = P.down_at(f"kv{k0}", 0, 2*4*1024*256*8, np.uint8)
    np.savez(os.getenv("SCDBG_DUMPZ", "~/scdbg_dump.npz"), **dz)
    print(f"[dump] saved buffers at n={n}", flush=True)
fout.close()
dev.synchronize()
if os.getenv("SCDBG_REMAP"):
  a1 = P.down("qwsc", (N, 6144), np.float16); P._keep.clear()
  pat = np.full(256*6144, 0.5, dtype=np.float16)
  P.win_up("qwsc", 0, pat); P._keep.clear(); dev.synchronize()
  a2 = P.down("qwsc", (N, 6144), np.float16); P._keep.clear()
  print(f"[remap] pre nan {int(np.isnan(a1.astype(np.float32)).sum())} | post-upload equal {bool((a2 == 0.5).all())} nan {int(np.isnan(a2.astype(np.float32)).sum())}", flush=True)

if os.getenv("SCDBG_ENDUMP"):
  dz = {}
  for nm, shp, dt in NAMED + [("xAsc",(N,5120),np.float32), ("xBsc",(N,5120),np.float32)]:
    if nm in dz: continue
    dz[nm] = P.down(nm, shp, dt); P._keep.clear()
  for i in E.gdn_idx[:6]:
    dz[f"rec{i}"] = P.down(f"rec{i}", (48*128*128,), np.float32); P._keep.clear()
    dz[f"conv{i}_0"] = P.down(f"conv{i}_0", (3*10240,), np.float32); P._keep.clear()
  if os.getenv("SCDBG_TRUNC"):
    for i in E.gdn_idx[6:18]:
      dz[f"rec{i}"] = P.down(f"rec{i}", (48*128*128,), np.float32); P._keep.clear()
  k0 = list(E.attn_idx)[0]
  dz["kv_first"] = P.down_at(f"kv{k0}", 0, 2*4*1024*256*8, np.uint8)
  np.savez(os.getenv("SCDBG_ENDUMP"), **dz)
  nanq = int(np.isnan(dz["qwsc"].astype(np.float32)).sum())
  nanx = (int(np.isnan(dz["xAsc"].astype(np.float32)).sum()) + int(np.isnan(dz["xBsc"].astype(np.float32)).sum())) if "xAsc" in dz else -1
  nanr = sum(int(np.isnan(dz[f"rec{i}"]).sum()) for i in E.gdn_idx[:4])
  print(f"[endump] qwsc nan {nanq} x nan {nanx} rec nan {nanr}", flush=True)
print(f"[done] plan {N} launches in {time.perf_counter()-t0:.1f}s; snapshots {len(STEPS)} -> {OUT}", flush=True)
