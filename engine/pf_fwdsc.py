# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7e super-chunk gate harness: 256-token PF_SC prefill vs the T=1 trunk.
Gates (READOUT-ORDER LAW: gates from the first clean run; timing after):
  - final-row logits relerr vs T=1 (med ~1e-3 / F <= 3e-3 class)
  - tok_slot == T=1 argmax at pos 255
  - GDN rec/conv drift vs the T=1 trunk states (P6 class <= ~3e-2)
  - kv first-256-row byte drift (tie-mine class: expect small nonzero)
Timing: per-super-chunk ms + projected tok/s (2k-class harness, pos 0).
Usage: env SKV=1 ... DEV=NV ~/tg311/bin/python -u pf_fwdsc.py
"""
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
pr = E.pr
N = int(os.getenv("PF_SC", "256"))

def t1_token(t, tok):
  P.win_up("tok_slot", 0, np.array([int(tok)], dtype=np.int32))
  if not hasattr(E, "_seq"): E._build_seqs()
  seq = E._seq[t & 1]   # P7E2 FIX: GDN live-state ping-pong parity (was [0] -> wrong ref)
  for n_, (p, a, g) in enumerate(seq):
    p(*a, global_size=(g[0],1,1) if isinstance(g, tuple) else (g,1,1),
      local_size=(1024,1,1) if "nw32" in getattr(p, "name", "") else LS,
      wait=(n_ == len(seq)-1))

rng = np.random.default_rng(99)
ids = rng.integers(1000, 200000, size=N).astype(np.int32)
P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
dev.synchronize()

SKIPREF = os.getenv("PF_SC_SKIPREF") == "1"
if SKIPREF:
  ref_logits = None
else:
  t0 = time.perf_counter()
  ref_logits = np.zeros((N, 248320), dtype=np.float16)
  for t in range(N):
    t1_token(t, ids[t])
    ref_logits[t] = P.down("logits", (248320,), np.float16)
  print(f"[ref] T=1 trunk {N} tokens in {time.perf_counter()-t0:.1f}s", flush=True)
  ref_rec = {i: P.down(f"rec{i}", (48*128*128,), np.float32).copy() for i in E.gdn_idx[:4]}
  ref_conv = {i: P.down(f"conv{i}_0", (3*10240,), np.float32).copy() for i in E.gdn_idx[:4]}
  ref_tok = int(P.down_at("tok_slot", 0, 1)[0])
  KVS = 2*4*256*256*8   # first 2048 rows of each slab (covers 256 tokens)
  ref_kv = {i: P.down_at(f"kv{i}", 0, KVS, np.uint8).copy() for i in list(E.attn_idx)[:2]}

# ---- reset world for the SC pass ----
# fixed-handle reset (M1-A law: P.up reallocs -> orphan growth + stale handles)
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
if os.getenv("PF_SC_PROBE") == "1":
  _r0 = P.down("rec0", (48*128*128,), np.float32); _c0 = P.down("conv0_0", (3*10240,), np.float32)
  print(f"[probe] post-reset rec0 nan {int(np.isnan(_r0).sum())} absmax {np.abs(_r0).max():.3e} | conv0 nan {int(np.isnan(_c0).sum())} absmax {np.abs(_c0).max():.3e}", flush=True)
  P._keep.clear()

if os.getenv("PF_SC_DIAG2") == "1":
  from tinygrad.device import TinyELF as _TE
  from tinygrad.runtime.ops_nv import NVProgram as _NP
  _lib = open("~/tinygrad-metal/engine0/pfk_emb16.cubin", "rb").read()
  _pk = _NP(dev, _TE(lib=_lib, name="pfk_emb16", target=dev.renderer.target, signature=tuple()))
  P.up("diag_emb_out", np.zeros(256*5120, dtype=np.float32))
  P.up("diag_ids", np.array(ids, dtype=np.int32))
  dev.synchronize()
  _pk(W[("emb",0)], d["grid512"], d["diag_ids"], d["diag_emb_out"], global_size=(256,1,1), local_size=LS)
  dev.synchronize()
  _a = P.down("diag_emb_out", (256, 5120), np.float32)
  print(f"[diag2] manual emb post-ref: nan {int(np.isnan(_a).sum())}/{_a.size} absmax {np.abs(_a).max():.3e}", flush=True)
  _e = np.frombuffer(bytes(P.down_at("grid512", 0, 16, np.float32)), dtype=np.float32) if False else None
  P._keep.clear()

# ---- ONE clean SC run; gates read immediately ----
class G:
  pass
ct = []
t0 = time.perf_counter()
wall = pf_prefill.prefill_batch(E, G, ids, log=lambda s, **kw: print(f"[sc] {s} {kw}", flush=True), chunk_times=ct)
sc_time = (ct[0][1] / 1e3) if ct else wall
print(f"[sc] {len(ct)} super-chunk(s), first-chunk {sc_time*1e3:.1f} ms, wall {wall:.2f}s -> {N/wall:.1f} tok/s", flush=True)

import os as _os
if _os.getenv("PF_SC_PROBE") == "1":
  P2 = P
  for nm, shape, dt in [("xAsc", (256, 5120), np.float32), ("xhsc", (256, 5120), np.float16),
                        ("qkvsc", (256, 10240), np.float16), ("arawsc", (256*48,), np.float32),
                        ("gatesc", (256, 6144), np.float16), ("sco", (256, 6144), np.float32),
                        ("zsc", (256, 6144), np.float16), ("gactsc", (256, 17408), np.float16)]:
    try:
      arr = P2.down(nm, shape, dt)
      print(f"[probe] {nm}: nan {int(np.isnan(arr.astype(np.float32)).sum())}/{arr.size} absmax {np.nanmax(np.abs(arr.astype(np.float32))) if arr.size else 0:.3e}", flush=True)
    except Exception as ex:
      print(f"[probe] {nm}: ERR {ex}", flush=True)
if not SKIPREF:
  fin = P.down("logits", (248320,), np.float16).astype(np.float32)
  r = ref_logits[N-1].astype(np.float32)
  act = np.abs(r) > 1e-6
  e = np.abs(fin[act] - r[act]) / np.abs(r[act])
  F = np.linalg.norm(fin - r) / max(np.linalg.norm(r), 1e-9)
  print(f"[gate] final logits vs T=1: med {np.median(e):.3e} F {F:.3e} -> {'PASS' if (np.median(e) <= 3e-3 or F <= 3e-3) else 'FAIL'}", flush=True)
  tok = int(P.down_at("tok_slot", 0, 1)[0])
  print(f"[gate] tok_slot {tok} vs T=1 argmax {ref_tok} -> {'PASS' if tok == ref_tok else 'CHECK (tie class)'}", flush=True)
  for i in E.gdn_idx[:4]:
    rm = P.down(f"rec{i}", (48*128*128,), np.float32); cm = P.down(f"conv{i}_0", (3*10240,), np.float32)
    dr = np.linalg.norm(rm - ref_rec[i]) / max(np.linalg.norm(ref_rec[i]), 1e-9)
    dc = np.linalg.norm(cm - ref_conv[i]) / max(np.linalg.norm(ref_conv[i]), 1e-9)
    print(f"[drift] blk {i}: rec {dr:.3e} conv {dc:.3e}", flush=True)
  for i in list(E.attn_idx)[:2]:
    kvm = P.down_at(f"kv{i}", 0, KVS, np.uint8)
    nz = int((kvm != ref_kv[i]).sum())
    print(f"[drift] kv blk {i}: byte mismatches {nz}/{kvm.size} (int8-KV tie-mine class)", flush=True)
  pos_end = int(P.down_at("pos_slot", 0, 1)[0])
  print(f"[post] pos_slot {pos_end} (expect {N})", flush=True)

# timing reps AFTER gates
best = sc_time
for _ in range(2):
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
  t0 = time.perf_counter(); ct2 = []
  pf_prefill.prefill_batch(E, G, ids, chunk_times=ct2)
  best = min(best, (ct2[0][1] / 1e3) if ct2 else 1e9)
print(f"[time] SC chunk ({N} tok): {best*1e3:.1f} ms -> projected prefill {N/best:.1f} tok/s (harness, pos 0)", flush=True)
