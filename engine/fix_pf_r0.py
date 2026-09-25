# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7a: fix the pre-existing prefill_batch_m64 r==0 (n%64==0) broken-head bug
— the m64-own final head (_pf_last64 row 63 -> pfk_n16/head8/h_argmax)
produced argmax=0 (tok_slot=0 -> instant im_end) on the n=64 gate prompt;
n=65/66 (M32-tail head) healthy. Never exercised by the 2k/8k/100k banks
(all have r>0 tails or run the M128 trunk). FIX: when r == 0, run nc-1 M64
chunks and delegate the FINAL 64 tokens to the M32 path (the proven head —
same delegation pattern as the r>0 tail, P15 laws: pos_slot upload + flags
cleared). Same guard for the M128 trunk (r==0 -> final 128 via M32)."""
BASE = "~/tinygrad-metal/engine0"
s = open(f"{BASE}/pf_prefill.py").read()

def rep(a, b, label):
  global s
  c = s.count(a)
  assert c == 1, f"{label}: found {c}"
  s = s.replace(a, b)
  print(f"[fix] {label:44s} ok")

# m64: r==0 -> keep the last 64 for the M32 delegation
rep('''  if r > 0:
    if log is not None: log("prefill_batch64_tail_m32", n=r)
    # P15 FIX: the M64 chunks track position in pos_arr64/pos_w64 and never
    # touch pos_slot -> upload the running pos BEFORE the M32 tail delegation
    # (the tail reads pos_slot as ITS pos0; a stale value re-runs the tail at
    # pos 0 and clobbers kv rows 0..31 -- the 8k-gate failure root cause).
    P.win_up("pos_slot", 0, np.array([pos0 + 64*nc], dtype=np.int32))
    _sav = _M64ON
    m64_set(False)   # P15 FIX 2: the M32 tail MUST run the M32 plan/graphs (_pf_graphs keys on the ambient flag -- with it on, the 32-token tail replayed the STALE M64 graph on ids64/pos_arr64 = deterministic garbage; the 8k-gate root cause)
    try:
      prefill_batch(E, G, ids[64*nc:], prog=(lambda k, n: prog(64*nc + k, N)) if prog is not None else None,
                    log=None, chunk_times=chunk_times)
    finally:
      m64_set(_sav)
    dev.synchronize()
    return time.perf_counter() - t0   # (the M32 tail did its own head/hlast/seed)''',
    '''  # R7a r==0 FIX: n%64==0 never has a tail, and the m64-own final head
  # (_pf_last64 row 63) is UNGATED/BUGGY (argmax=0 on the api-gate n=64
  # prompt; the M32 head is the proven one). Keep the LAST 64 tokens out of
  # the M64 chunk loop and delegate them to M32 exactly like the r>0 tail.
  if r == 0 and nc >= 1:
    r = 64
    nc -= 1
    if log is not None: log("prefill_batch64_r0_tail_m32", n=64)
  if r > 0:
    if log is not None: log("prefill_batch64_tail_m32", n=r)
    # P15 FIX: the M64 chunks track position in pos_arr64/pos_w64 and never
    # touch pos_slot -> upload the running pos BEFORE the M32 tail delegation
    # (the tail reads pos_slot as ITS pos0; a stale value re-runs the tail at
    # pos 0 and clobbers kv rows 0..31 -- the 8k-gate failure root cause).
    P.win_up("pos_slot", 0, np.array([pos0 + 64*nc], dtype=np.int32))
    _sav = _M64ON
    m64_set(False)   # P15 FIX 2: the M32 tail MUST run the M32 plan/graphs (_pf_graphs keys on the ambient flag -- with it on, the 32-token tail replayed the STALE M64 graph on ids64/pos_arr64 = deterministic garbage; the 8k-gate root cause)
    try:
      prefill_batch(E, G, ids[64*nc:], prog=(lambda k, n: prog(64*nc + k, N)) if prog is not None else None,
                    log=None, chunk_times=chunk_times)
    finally:
      m64_set(_sav)
    dev.synchronize()
    return time.perf_counter() - t0   # (the M32 tail did its own head/hlast/seed)''',
    "m64 r==0 tail delegation")

open(f"{BASE}/pf_prefill.py", "w").write(s)
print("[fix] pf_prefill r==0 guard written")
