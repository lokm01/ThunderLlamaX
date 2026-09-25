# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7E2 stage-diff: run the P6 M32 path (ground truth) and the SC path on the
same ids/world, dump each stage's buffers, diff per stage -> first guilty kernel."""
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
N = int(os.getenv("SCDBG3_N", "256"))
rng = np.random.default_rng(99)
ids = rng.integers(1000, 200000, size=N).astype(np.int32)

def reset():
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
  dev.synchronize()
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

# ---- pass 1: M32 ground truth ----
os.environ["PF_SUPER"] = "0"
reset()
ct = []
pf_prefill.prefill_batch(E, G, ids, chunk_times=ct)
m32 = {}
for nm, shp, dt in [("xA32",(32,5120),np.float32), ("xh32",(32,5120),np.float16),
                    ("qrow32",(32,12288),np.float16), ("qw32",(32,6144),np.float16),
                    ("ao32",(32,6144),np.float16), ("attn_out32",(32,5120),np.float16),
                    ("hh32",(32,5120),np.float32), ("hhx32",(32,5120),np.float16),
                    ("qkv32",(32,10240),np.float16), ("gate32",(32,6144),np.float16),
                    ("z32",(32,6144),np.float16), ("gact32",(32,17408),np.float16),
                    ("araw32",(32,48),np.float32), ("braw32",(32,48),np.float32)]:
  m32[nm] = P.down(nm, shp, dt); P._keep.clear()
m32log = P.down("logits", (248320,), np.float16); P._keep.clear()
print("[m32] dumped", flush=True)

# ---- pass 2: SC on the same world ----
os.environ["PF_SUPER"] = "1"
reset()
ct2 = []
pf_prefill.prefill_batch(E, G, ids, chunk_times=ct2)
sc = {}
for nm, shp, dt in [("xAsc",(N,5120),np.float32), ("xBsc",(N,5120),np.float32),
                    ("xhsc",(N,5120),np.float16), ("qrowsc",(N,12288),np.float16),
                    ("qwsc",(N,6144),np.float16), ("aosc",(N,6144),np.float16),
                    ("attn_outsc",(N,5120),np.float16), ("hhsc",(N,5120),np.float32),
                    ("hhxsc",(N,5120),np.float16), ("qkvsc",(N,10240),np.float16),
                    ("gatesc",(N,6144),np.float16), ("zsc",(N,6144),np.float16),
                    ("gactsc",(N,17408),np.float16), ("arawsc",(N*48,),np.float32),
                    ("brawsc",(N*48,),np.float32)]:
  sc[nm] = P.down(nm, shp, dt); P._keep.clear()
sclog = P.down("logits", (248320,), np.float16); P._keep.clear()
np.savez("~/stagediff.npz", m32log=m32log, sclog=sclog, **{"sc_"+k: v for k, v in sc.items()},
         **{"m32_"+k: v for k, v in m32.items()})

def cmp(nm, key32, keysc, rows=None, col=None):
  a = m32[key32].astype(np.float64)  # last 32 rows hold tokens 224..255
  b = sc[keysc].astype(np.float64)
  # M32 ping-pong: the LAST chunk's rows live in xA32/xB32 depending on parity; we take what's there
  r = rows if rows is not None else range(max(0, a.shape[0]-8), a.shape[0])
  aa = a[list(r)]; bb = b[224 + np.array(list(r))]
  if aa.shape != bb.shape: return f"{nm}: shape {aa.shape} vs {bb.shape}"
  dd = np.abs(aa - bb)
  scale = np.maximum(np.abs(aa), 1e-9)
  return "%-10s maxabs %.3e medrel %.3e" % (nm, dd.max() if dd.size else 0,
         float(np.median(dd / scale)) if dd.size else 0)

print("[stage-diff] M32-last-8-rows vs SC-rows-232..239 (post-attention stages row-aligned):", flush=True)
for nm, k32, ksc in [("emb/x","xA32","xAsc"), ("xh","xh32","xhsc"), ("qrow","qrow32","qrowsc"),
                     ("qw","qw32","qwsc"), ("ao","ao32","aosc"), ("attn_out","attn_out32","attn_outsc"),
                     ("hh","hh32","hhsc"), ("hhx","hhx32","hhxsc"), ("qkv","qkv32","qkvsc"),
                     ("gate","gate32","gatesc"), ("z","z32","zsc"), ("gact","gact32","gactsc")]:
  print(" ", cmp(nm, k32, ksc), flush=True)
ml = m32log.astype(np.float32); sl = sclog.astype(np.float32)
act = np.abs(ml) > 1e-6
e = np.abs(sl[act] - ml[act]) / np.abs(ml[act])
print("[logits] SC vs M32: med %.3e F %.3e argmax m32 %d sc %d" % (
  np.median(e), np.linalg.norm(sl - ml) / max(np.linalg.norm(ml), 1e-9),
  int(np.argmax(ml)), int(np.argmax(sl))), flush=True)
