# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P3 100k gate module (PF_GATE100k=1 inside test_w100k.py — HOST-PROCESS BOOT
LAW): rebuild the 100k snapshot position via PF_BATCH batched prefill from
scratch, compare trunk GDN states vs the snapshot npys, then T=1-decode 60 and
compare vs the banked engine_t1_ref. Also the 100k prefill wall-time bench.
"""
import os, time
import numpy as np
from engine0 import dev

def run_gates(E, G, sess, ref, ids, CTXK):
  import pf_prefill, json
  SNAP = os.getenv("SNAPDIR", "~/snap100k")
  meta = json.load(open(f"{SNAP}/meta.json"))
  CUR0 = int(meta["cur0"])
  NTOK = 60
  CT = []
  toks = [int(t) for t in ids]
  print(f"[g100k] rebuilding {len(toks)}-token prompt via PF_BATCH from scratch", flush=True)

  E.reset_fresh(toks[0])
  E.stload_trunk()
  from mtp import CBLK
  for i in E.gdn_idx: E._mfill(f"conv{i}_1", 0, CBLK)
  dev.synchronize()
  t0 = time.perf_counter()
  E.fill_draft(toks, start_pos=0, seed_hd=None)
  tfd = time.perf_counter() - t0
  dt = pf_prefill.prefill_batch(E, G, toks, chunk_times=CT,
                                log=lambda s, **kw: print(f"[g100k] {s} {kw}", flush=True) if s != "prefill_batch_chunk" else None)
  print(f"[g100k] PF_BATCH 100k prefill: {dt:.1f}s = {len(toks)/dt:.1f} tok/s (fill_draft {tfd:.1f}s = {len(toks)/tfd:.1f} tok/s)", flush=True)
  np.save("~/pf_chunk_times_100k.npy", np.array(CT))

  cur = int(E.P.down_at("tok_slot", 0, 1)[0])
  pos = int(E.P.down_at("pos_slot", 0, 1)[0])
  print(f"[g100k] rebuilt cur={cur} (snapshot cur0={CUR0} match={cur==CUR0}) pos={pos}", flush=True)

  # trunk GDN state compare vs snapshot (rec + conv on a sample of blocks)
  rel = []
  for j, i in enumerate(E.gdn_idx):
    if j % 8: continue
    rm = E.P.down(f"rec{i}", (48*128*128,), np.float32)
    rs = np.load(f"{SNAP}/rc_{i}.npy", mmap_mode="r")
    r = np.linalg.norm(rm - rs) / max(np.linalg.norm(np.asarray(rs)), 1e-9)
    cm = E.P.down(f"conv{i}_0", (3*10240,), np.float32)
    cs = np.load(f"{SNAP}/conv_{i}.npy", mmap_mode="r")
    c = np.linalg.norm(cm - cs) / max(np.linalg.norm(np.asarray(cs)), 1e-9)
    rel.append((i, r, c))
    print(f"[g100k] blk {i}: rec relerr {r:.3e} conv relerr {c:.3e}", flush=True)
  rr = max(r for _, r, _ in rel); cc = max(c for _, _, c in rel)
  print(f"[GATE 100k-state] max rec relerr {rr:.3e} max conv relerr {cc:.3e} (gate <=1e-2)", flush=True)

  # T=1 decode vs banked ref
  G.run_tokens(NTOK, wait_each=True)
  h = E.P.down("tok_hist", (CTXK + 256,), np.int32)
  out = h[len(toks):len(toks)+NTOK].tolist()
  agree = sum(1 for a, b in zip(out, ref) if a == b)
  print(f"[GATE 100k-decode] rebuilt-state T1 decode vs engine_t1_ref: {agree}/{NTOK}", flush=True)
  print(f"[g100k] out[:20] {out[:20]}", flush=True)
  print(f"[g100k] ref[:20] {ref[:20]}", flush=True)
  ps = sorted(CT)
  if ps:
    print("[BENCH 100k] chunk ms sample:", flush=True)
    for k in sorted(set(list(range(0, len(ps), max(1, len(ps)//10))) + [len(ps)-1])):
      p, ms = ps[k]
      print(f"  pos {p:6d}: {ms:7.1f} ms ({16/(ms/1000):.1f} tok/s)", flush=True)
  print("[g100k] done", flush=True)
