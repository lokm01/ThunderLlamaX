# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7E3 chunk-2 bisection: M32 ground truth @256 and @512 vs SC chunk-1 carry
and SC 2-chunk end state + stages. Phases run SC-FIRST (fresh-process safe),
truth M32 LAST (the after-M32 fault config only affects a LATER SC run).
Env: SKV=1 KV8=1 QH=1 SKV_CTXK=100352 PF_GEMM3=0 PF_SCANC=1."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from trunk_w1c import TrunkEngineW1C, LS
import pf_prefill

E = TrunkEngineW1C(theta=1e7)
P, d = E.P, E.P.d
N = int(os.getenv("SCDBG5_N", "512"))
assert N == 512
rng = np.random.default_rng(99)
import json as _j; ids = np.array(_j.load(open("~/ids8k.json"))[:N], dtype=np.int32)
GB = list(E.gdn_idx[:3]) + list(E.gdn_idx[-2:])
AB = list(E.attn_idx[:2])

def reset():
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
  dev.synchronize()
  for i in E.attn_idx:
    P.win_up(f"kv{i}", 0, np.zeros(2*4*100352*256, dtype=np.uint8))
    P._keep.clear()
  for i in E.gdn_idx:
    P.win_up(f"conv{i}_0", 0, np.load(f"~/snap100k/conv_{i}.npy", mmap_mode="r"))
    P.win_up(f"conv{i}_1", 0, np.zeros(3*10240, dtype=np.float32))
    P.win_up(f"rec{i}", 0, np.load(f"~/snap100k/rc_{i}.npy", mmap_mode="r"))
  dev.synchronize(); P._keep.clear()
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
  dev.synchronize()

def snap(tag):
  """state snapshot: rec/conv for GB, kv+scale slice rows [248..280), logits."""
  s = {}
  for i in GB:
    s[f"rec{i}"] = P.down(f"rec{i}", (48*128*128,), np.float32).copy(); P._keep.clear()
    s[f"conv{i}"] = P.down(f"conv{i}_0", (3*10240,), np.float32).copy(); P._keep.clear()
  for i in AB:
    s[f"kv{i}"] = P.down_at(f"kv{i}", 248*256, 32*256, np.uint8).copy(); P._keep.clear()
    s[f"kvs{i}"] = P.down_at(f"sc{i}", 248*8*2, 32*8*2, np.uint8).copy(); P._keep.clear()
  s["logits"] = P.down("logits", (248320,), np.float16).copy(); P._keep.clear()
  return s

def stages_m32():
  o = {}
  for nm, shp, dt in [("xh32",(32,5120),np.float16),("qrow32",(32,12288),np.float16),
                      ("qw32",(32,6144),np.float16),("ao32",(32,6144),np.float16),
                      ("attn_out32",(32,5120),np.float16),("qkv32",(32,10240),np.float16),
                      ("gate32",(32,6144),np.float16),("z32",(32,6144),np.float16),
                      ("gact32",(32,17408),np.float16)]:
    o[nm] = P.down(nm, shp, dt).copy(); P._keep.clear()
  return o

def stages_sc():
  o = {}
  for nm, shp, dt in [("xhsc",(256,5120),np.float16),("qrowsc",(256,12288),np.float16),
                      ("qwsc",(256,6144),np.float16),("aosc",(256,6144),np.float16),
                      ("attn_outsc",(256,5120),np.float16),("qkvsc",(256,10240),np.float16),
                      ("gatesc",(256,6144),np.float16),("zsc",(256,6144),np.float16),
                      ("gactsc",(256,17408),np.float16)]:
    o[nm] = P.down(nm, shp, dt).copy(); P._keep.clear()
  return o

class G: pass
out = {}

# ---- Phase S1: SC chunk 1 only, snapshot carry-out at 256 ----
os.environ["PF_SUPER"] = "1"
reset(); t0 = time.perf_counter()
ct = []
pf_prefill.prefill_batch(E, G, ids[:256], chunk_times=ct)
dev.synchronize()
print(f"[S1] SC chunk1 {ct} wall {time.perf_counter()-t0:.1f}s", flush=True)
out["S1"] = snap("S1"); P._keep.clear()

# ---- Phase S2: SC 2 chunks ----
reset(); t0 = time.perf_counter()
ct = []
pf_prefill.prefill_batch(E, G, ids, chunk_times=ct)
dev.synchronize()
print(f"[S2] SC 2-chunk {ct} wall {time.perf_counter()-t0:.1f}s", flush=True)
out["S2"] = snap("S2")
out["S2stg"] = stages_sc(); P._keep.clear()

# ---- Phase T: M32 truth @256 then @512 ----
os.environ["PF_SUPER"] = "0"
reset(); t0 = time.perf_counter()
pf_prefill.prefill_batch(E, G, ids[:256])
dev.synchronize()
print(f"[T] M32 0..256 wall {time.perf_counter()-t0:.1f}s", flush=True)
out["T256"] = snap("T256"); P._keep.clear()
t0 = time.perf_counter()
pf_prefill.prefill_batch(E, G, ids[256:])
dev.synchronize()
print(f"[T] M32 256..512 wall {time.perf_counter()-t0:.1f}s", flush=True)
out["T512"] = snap("T512")
out["T512stg"] = stages_m32(); P._keep.clear()

np.savez("~/scdbg5.npz", ids=ids, **{f"{k}_{kk}": vv for k, o in out.items() for kk, vv in o.items()})
print("[saved] ~/scdbg5.npz", flush=True)

def rel(a, b):
  a = a.astype(np.float64); b = b.astype(np.float64)
  dd = np.abs(a - b); sc = np.maximum(np.abs(a), 1e-9)
  return dd.max(), float(np.median(dd / sc)), int(np.isnan(b).sum())

print("\n===== STATE CARRY: SC-chunk1(S1) vs M32@256 (T256) =====", flush=True)
for k in [f"rec{i}" for i in GB] + [f"conv{i}" for i in GB] + [f"kv{i}" for i in AB] + [f"kvs{i}" for i in AB] + ["logits"]:
  mx, md, nn = rel(out["T256"][k], out["S1"][k])
  print(f"S1 {k:9s} maxabs {mx:.3e} medrel {md:.3e} nan {nn}", flush=True)
print("\n===== 2-CHUNK: SC(S2) vs M32@512 (T512) =====", flush=True)
for k in [f"rec{i}" for i in GB] + [f"conv{i}" for i in GB] + [f"kv{i}" for i in AB] + [f"kvs{i}" for i in AB] + ["logits"]:
  mx, md, nn = rel(out["T512"][k], out["S2"][k])
  print(f"S2 {k:9s} maxabs {mx:.3e} medrel {md:.3e} nan {nn}", flush=True)
print("\n===== STAGES: SC chunk2 rows 240:256 vs M32 rows 16:32 =====", flush=True)
for nm in out["T512stg"]:
  a = out["T512stg"][nm][16:32]; b = out["S2stg"][nm[:-2] + "sc"][240:256] if nm.endswith("32") else out["S2stg"][nm]
  b = out["S2stg"][nm.replace("32", "sc")][240:256]
  mx, md, nn = rel(a, b)
  print(f"stg {nm:12s} maxabs {mx:.3e} medrel {md:.3e} nan {nn}", flush=True)
