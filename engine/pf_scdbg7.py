# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7E3 scale ladder: M32 truth @7714 FIRST (tests post-M32 fault config when SC
runs after), then SC 30-chunk (r=0) and SC 30-chunk+34-tail. Clean world."""
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
NSC = int(os.getenv("SCDBG7_NSC", "30"))
N = 256 * NSC
rng = np.random.default_rng(99)
ids = rng.integers(1000, 200000, size=N + 64).astype(np.int32)
GB = list(E.gdn_idx[:3]) + list(E.gdn_idx[-2:])
AB = list(E.attn_idx[:2])

def reset():
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32)); dev.synchronize()
  for i in E.attn_idx:
    P.win_up(f"kv{i}", 0, np.zeros(2*4*100352*256, dtype=np.uint8)); P._keep.clear()
  for i in E.gdn_idx:
    P.win_up(f"conv{i}_0", 0, np.zeros(3*10240, dtype=np.float32))
    P.win_up(f"conv{i}_1", 0, np.zeros(3*10240, dtype=np.float32))
    P.win_up(f"rec{i}", 0, np.zeros(48*128*128, dtype=np.float32))
  dev.synchronize(); P._keep.clear()
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32)); dev.synchronize()

def snap():
  s = {"logits": P.down("logits", (248320,), np.float16).copy()}
  P._keep.clear()
  for i in GB:
    s[f"rec{i}"] = P.down(f"rec{i}", (48*128*128,), np.float32).copy(); P._keep.clear()
    s[f"conv{i}"] = P.down(f"conv{i}_0", (3*10240,), np.float32).copy(); P._keep.clear()
  return s

def rel(a, b):
  a = a.astype(np.float64); b = b.astype(np.float64)
  dd = np.abs(a - b); sc = np.maximum(np.abs(a), 1e-9)
  return dd.max(), float(np.median(dd / sc))

class G: pass
out = {}
import collections
seen, sl = set(), []
for t in ids.tolist()[:N]:
    if t not in seen: seen.add(t); sl.append(t)
base = sl[:]
while len(sl) < 40960: sl += base
E.init_draft(sl[:40960])
print("[draft] init done", flush=True)
# ---- Phase T: M32 truth (the post-M32 world for SC) ----
os.environ["PF_SUPER"] = "0"
reset(); t0 = time.perf_counter()
pf_prefill.prefill_batch(E, G, ids[:N])
dev.synchronize(); print(f"[T] M32 {N} wall {time.perf_counter()-t0:.1f}s", flush=True)
out["T"] = snap()
# extend truth by 34 tail tokens


# ---- Phase S30: SC 30 chunks r=0 (AFTER the M32 pass — the fault config) ----
os.environ["PF_SUPER"] = "1"
reset(); t0 = time.perf_counter()
ct = []
pf_prefill.prefill_batch(E, G, ids[:N], chunk_times=ct)
dev.synchronize(); print(f"[S] SC {NSC} chunks wall {time.perf_counter()-t0:.1f}s last3={ct[-3:]}", flush=True)
out["S"] = snap()



np.savez("~/scdbg7.npz", ids=ids, **{f"{k}_{kk}": vv for k, o in out.items() for kk, vv in o.items()})
for ph, tk in [("S", "T")]:
  print(f"\n===== {ph} vs {tk} =====", flush=True)
  for k in ["logits"] + [f"rec{i}" for i in GB] + [f"conv{i}" for i in GB]:
    mx, md = rel(out[tk][k], out[ph][k])
    print(f"{ph} {k:9s} maxabs {mx:.3e} medrel {md:.3e}", flush=True)
  la = out[tk]["logits"].astype(np.float64); lb = out[ph]["logits"].astype(np.float64)
  oa, ob = np.argsort(la)[::-1], np.argsort(lb)[::-1]
  print(f"{ph} top1 {oa[0]}({la[oa[0]]:.2f}) vs {ob[0]}({lb[ob[0]]:.2f})  top2 {oa[1]}({la[oa[1]]:.2f}) vs {ob[1]}({lb[ob[1]]:.2f})", flush=True)
