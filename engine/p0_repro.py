# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P0 stale-feed repro: successive prefill_batch calls with DIFFERENT content
(the gates always prefill the same toks, so a stale ids feed is invisible to
them). For each probe length L and prompt pair (A,B):
  t1(A) ground truth -> pf(A) -> pf(B) [no t1 between] -> t1(B) ground truth
stale-feed verdicts: pf(B).cur == pf(A).cur != t1(B).cur  => STALE
                      pf(B).cur == t1(B).cur              => FRESH
pos verdicts: pf(X).pos must equal len(X) (the tail-runs-at-pos-0 class shows
pos == r or similar).
Env: P0_LENS="66,65,64,63,130,128,32,96" (csv), P0_MODE=pf|t1.
"""
import os, time
import numpy as np
from engine0 import dev
from mtp import CBLK
import pf_prefill

NTOK = 6

def fresh(E, toks):
  E.reset_fresh(toks[0])
  E.stload_trunk()
  for i in E.gdn_idx:
    E._mfill(f"conv{i}_1", 0, CBLK)
  dev.synchronize()

def postludes(E):
  E.stseed_spec(1)
  newcur = int(E.P.down_at("tok_slot", 0, 1)[0])
  E.P.win_up("cur_slot", 0, np.array([newcur], dtype=np.int32))
  E.P.win_up("h_seed", 0, np.zeros(5120, dtype=np.float32))
  E.P.win_up("dring0", 0, np.full(1, -1, dtype=np.int32))
  E.P.win_up("dring1", 0, np.full(1, -1, dtype=np.int32))
  dev.synchronize()

def pf_once(E, G, toks, tag):
  fresh(E, toks)
  t0 = time.perf_counter()
  pf_prefill.prefill_batch(E, G, toks)
  dt = time.perf_counter() - t0
  pos = int(E.P.down_at("pos_slot", 0, 1)[0])
  cur = int(E.P.down_at("tok_slot", 0, 1)[0])
  postludes(E)
  print(f"[p0] {tag}: n={len(toks)} pos={pos} cur={cur} ({dt:.2f}s)"
        f"{'  <-- POS WRONG' if pos != len(toks) else ''}", flush=True)
  E.P._keep.clear()
  return pos, cur

def t1_once(E, G, toks, tag):
  fresh(E, toks)
  t0 = time.perf_counter()
  E.prefill_t1(G, toks)
  dt = time.perf_counter() - t0
  pos = int(E.P.down_at("pos_slot", 0, 1)[0])
  cur = int(E.P.down_at("tok_slot", 0, 1)[0])
  postludes(E)
  print(f"[p0] {tag}: n={len(toks)} pos={pos} cur={cur} ({dt:.2f}s) [T1 ref]", flush=True)
  E.P._keep.clear()
  return pos, cur

def run(E, G, sess, ref, ids, CTXK):
  import json
  ids8 = [int(t) for t in json.load(open("~/ids8k.json"))]
  lens = [int(x) for x in os.getenv("P0_LENS", "66,65,64,63,130,96,32").split(",")]
  print(f"[p0] lens={lens} PG={os.getenv('PF_PG','1')} PG_WAIT={os.getenv('PG_WAIT','1')}", flush=True)
  nbad = 0
  for L in lens:
    A = ids8[0:L]
    B = ids8[4000:4000+L]
    sameAB = A == B
    print(f"== L={L} ==", flush=True)
    _, cA_t1 = t1_once(E, G, A, f"A-ref")
    pA, cA = pf_once(E, G, A, "A-pf ")
    _, cB_t1 = t1_once(E, G, B, f"B-ref")
    pB, cB = pf_once(E, G, B, "B-pf ")
    stale = (cB == cA) and not sameAB and (cA == cA_t1)
    verdict = "STALE!!" if stale else ("fresh" if cB == cB_t1 else f"DIFF(t1B) cB={cB} cB_t1={cB_t1}")
    posok = (pA == L) and (pB == L)
    print(f"[p0] L={L}: cA_t1={cA_t1} cA={cA} cB={cB} cB_t1={cB_t1} -> {verdict} | pos {'OK' if posok else 'WRONG'}", flush=True)
    if stale or not posok: nbad += 1
  print(f"[p0 done] bad={nbad}", flush=True)
