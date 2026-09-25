# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P15 debug: 7714-token M64-vs-M32 bit-identity A/B (the 8k-gate failure
adjudication: PG_SPLIT=4 vs the r%64 tail composition). Attr-world boot;
M32 is the truth. Env: FULL env + PF_M64=1 (PG_SPLIT from env)."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from mtp import MTPEngine, CBLK, SLICE
import pf_prefill

N = int(os.getenv("P15_N", "7714"))
E = MTPEngine(theta=1e7)
P, W, d = E.P, E.W, E.P.d
rng = np.random.default_rng(0)
IDS = [int(t) for t in rng.integers(1000, 60000, N)]

def fresh_world():
  E.reset_fresh(IDS[0])
  E.stload_trunk()
  for i in E.gdn_idx:
    E._mfill(f"conv{i}_1", 0, CBLK)
  dev.synchronize()

fresh_world()
_seen, sl = set(), []
for t in IDS:
  if t not in _seen:
    _seen.add(t); sl.append(t)
_base = sl[:]
while len(sl) < SLICE:
  sl += _base
E.init_draft(sl[:SLICE])
E.fill_draft(IDS, start_pos=0, seed_hd=None)
dev.synchronize()

GDN5 = [E.gdn_idx[0], E.gdn_idx[len(E.gdn_idx)//3], E.gdn_idx[2*len(E.gdn_idx)//3], E.gdn_idx[-2], E.gdn_idx[-1]]
ATTN3 = list(E.qtypes)[:3]

def snap():
  S = {"logits": P.down("logits", (248320,), np.float16).copy(),
       "tok_slot": int(P.down_at("tok_slot", 0, 1)[0]),
       "pos_slot": int(P.down_at("pos_slot", 0, 1)[0])}
  for i in GDN5:
    S[f"rec{i}"] = P.down(f"rec{i}", (48*128*128,), np.float32).copy()
    S[f"conv{i}"] = P.down(f"conv{i}_0", (3*10240,), np.float32).copy()
  for i in ATTN3:
    S[f"kv{i}"] = P.down_at(f"kv{i}", (N-96)*2048, 96*2048, np.int8).copy()
    S[f"sc{i}"] = P.down_at(f"sc{i}", (N-96)*64, 96*64, np.int8).copy()
  P._keep.clear()
  return S

CT = []
def run_prefill(mode):
  fresh_world()
  pf_prefill.m64_set(mode == "m64")
  CT.clear()
  t0 = time.perf_counter()
  dt = pf_prefill.prefill_batch(E, None, IDS, chunk_times=CT)
  dev.synchronize()
  wall = time.perf_counter() - t0
  return wall, dt, snap(), list(CT)

w64, dt64, S64, CT64 = run_prefill("m64")
c64 = sorted(t for _, t in CT64)
print(f"[ab] M64: wall {w64:.2f}s ret {dt64:.2f}s chunks={len(CT64)} med={c64[len(c64)//2]:.1f}ms", flush=True)
w32, dt32, S32, CT32 = run_prefill("m32")
c32 = sorted(t for _, t in CT32)
print(f"[ab] M32: wall {w32:.2f}s ret {dt32:.2f}s chunks={len(CT32)} med={c32[len(c32)//2]:.1f}ms", flush=True)
w64b, _, S64b, _ = run_prefill("m64")
bad_det = [k for k in S64 if not np.array_equal(np.asarray(S64[k]), np.asarray(S64b[k]))]
print(f"[ab] M64 det x2: {'OK' if not bad_det else 'MISMATCH ' + str(bad_det)} (wall {w64b:.2f}s)", flush=True)
bad = [k for k in S32 if not np.array_equal(np.asarray(S32[k]), np.asarray(S64[k]))]
print(f"[ab] M64 vs M32 (N={N}, split={os.getenv('PG_SPLIT','2')}): "
      f"{'BIT-IDENTICAL (all keys)' if not bad else 'MISMATCH ' + str(bad)}", flush=True)
lg64 = S64["logits"].astype(np.float64); lg32 = S32["logits"].astype(np.float64)
print(f"[ab] F-relerr: {np.linalg.norm(lg64-lg32)/max(np.linalg.norm(lg32),1e-9):.3e} "
      f"tok {S64['tok_slot']} vs {S32['tok_slot']} pos {S64['pos_slot']} vs {S32['pos_slot']}", flush=True)
print("[ab] DONE", flush=True)
