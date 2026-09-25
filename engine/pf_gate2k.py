# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P3 gate module (invoked INSIDE test_w100k.py under PF_GATE=1 — the
HOST-PROCESS BOOT LAW: only a `python -u test_w100k.py` main is proven; a
hand-rolled host faults in fill_draft / collapses draft acceptance).

Gates:
  A  T1-prefill + T1-decode(60) vs PF_BATCH-prefill + T1-decode(60)  -> N/60
  D  PF_BATCH-prefill + spec-decode(60) vs PF_BATCH-prefill + T1-decode(60)
     (Tier-1: spec == greedy on the SAME batched-prefilled state class)
Bench: prefill wall times, per-chunk ms by position (the P3 table source).
"""
import os, time
import numpy as np
from engine0 import dev

def run_gates(E, G, sess, ref, ids, CTXK):
  import pf_prefill
  NP = int(os.getenv("PF_NPROMPT", "2048"))
  NTOK = int(os.getenv("NTOK", "60"))
  RBLK_C = None
  from mtp import CBLK
  CT = []

  def fresh(toks, mode):
    E.reset_fresh(toks[0])
    E.stload_trunk()
    for i in E.gdn_idx: E._mfill(f"conv{i}_1", 0, CBLK)
    dev.synchronize()
    dfill = (mode == "PF_BATCH") and os.getenv("PF_DFILL", "1") == "1"
    tf0 = time.perf_counter()
    if not dfill:
      E.fill_draft(toks, start_pos=0, seed_hd=None)
    tf = time.perf_counter() - tf0
    if mode == "PF_BATCH":
      dt = pf_prefill.prefill_batch(E, G, toks, chunk_times=CT)
    else:
      dt = E.prefill_t1(G, toks)
    E.stseed_spec(len(toks) & 1)
    newcur = int(E.P.down_at("tok_slot", 0, 1)[0])
    E.P.win_up("cur_slot", 0, np.array([newcur], dtype=np.int32))
    E.P.win_up("h_seed", 0, np.zeros(5120, dtype=np.float32))
    E.P.win_up("dring0", 0, np.full(1, -1, dtype=np.int32))
    E.P.win_up("dring1", 0, np.full(1, -1, dtype=np.int32))
    dev.synchronize()
    return dt, tf, newcur

  def t1_decode(n):
    G.run_tokens(n, wait_each=True)
    pos = int(E.P.down_at("pos_slot", 0, 1)[0])
    h = E.P.down("tok_hist", (CTXK + 256,), np.int32)
    return h[pos - n:pos].tolist()

  def spec_decode(n):
    sess.begin()
    emt = []
    r = None
    for _ in range(n):
      r = sess.step()
      emt += r["tokens"]
    return emt, r

  import json as _json
  try:
    ids8 = _json.load(open("~/ids8k.json"))
    toks = [int(t) for t in ids8]
    print(f"[gate] prompt8k.txt (pre-tokenized): {len(toks)} tokens, natural document end", flush=True)
  except Exception as e:
    print(f"[gate] prompt8k fallback ({e!r}); using 100k-prompt head NP={NP}", flush=True)
    toks = [int(t) for t in ids[:NP]]
  _tr = int(os.getenv("PF_TRUNC", "0"))
  if _tr and len(toks) > _tr:
    toks = toks[:_tr]
    print(f"[gate] PF_TRUNC={_tr} (ids8k head)", flush=True)
  print(f"[gate] prompt: {len(toks)} tokens", flush=True)

  def logits_stats(tag):
    lg = E.P.down("logits", (248320,), np.float16).astype(np.float64)
    o = np.argsort(lg)[::-1]
    print(f"[gate] {tag}: top1 {o[0]} ({lg[o[0]]:.3f}) top2 {o[1]} ({lg[o[1]]:.3f}) gap {lg[o[0]]-lg[o[1]]:.4f}", flush=True)
    return lg

  def t1_decode_gapped(n):
    out, gaps = [], []
    for _ in range(n):
      G.run_tokens(1, wait_each=True)
      lg = E.P.down("logits", (248320,), np.float16).astype(np.float64)
      o = np.argsort(lg)[::-1]
      out.append(int(o[0])); gaps.append((int(o[0]), int(o[1]), float(lg[o[0]]-lg[o[1]])))
      pos = int(E.P.down_at("pos_slot", 0, 1)[0])
      h = E.P.down("tok_hist", (CTXK + 256,), np.int32)
      if h[pos-1] != o[0]:
        # exact-tie mine (kernel argmax vs np.argsort order on EQUAL fp16 logits
        # — the P7F1 (4649,43614) class). PF_GATE_TIEOK=1 downgrades to a warn
        # so the WY numerics landscape (different tie set) can't eat the ladder.
        _tie = (lg[h[pos-1]] == lg[o[0]])
        if _tie and os.getenv("PF_GATE_TIEOK", "0") == "1":
          print(f"[gate] tie-mine pos {pos}: engine {h[pos-1]} vs argsort {o[0]} (equal logits)", flush=True)
        else:
          assert h[pos-1] == o[0], (h[pos-1], o[0])
    return out, gaps

  print("== A1: T1 prefill + T1 decode ==", flush=True)
  dt_t1, tfd1, cur1 = fresh(toks, "T1")
  lgA = logits_stats("T1-prefill end logits")
  if os.getenv("SCDBG_GATE_DUMP"):
    import numpy as _np
    _o = {"logits": _np.asarray(lgA, dtype=_np.float16)}
    for i in list(E.gdn_idx[:3]) + list(E.gdn_idx[-2:]):
      _o[f"rec{i}"] = E.P.down(f"rec{i}", (48*128*128,), _np.float32).copy(); E.P._keep.clear()
      _o[f"conv{i}"] = E.P.down(f"conv{i}_0", (3*10240,), _np.float32).copy(); E.P._keep.clear()
    _np.savez("~/gate_a1_state.npz", **_o)
    print("[gate] A1 state dumped", flush=True)
  tA0 = time.perf_counter(); tokA, gapsA = t1_decode_gapped(NTOK); dtA = time.perf_counter() - tA0
  print(f"[gate] T1 prefill {dt_t1:.1f}s (fill_draft {tfd1:.1f}s), T1 decode {dtA:.1f}s", flush=True)
  print(f"[gate] tokA[:20] {tokA[:20]}", flush=True)

  print("== A2: PF_BATCH prefill + T1 decode ==", flush=True)
  CT.clear()
  dt_pf, tfd2, cur2 = fresh(toks, "PF_BATCH")
  _ct = [t for _, t in CT[-max(1, len(CT)//2):]]
  print(f"[gate] PF_BATCH prefill {dt_pf:.1f}s = {len(toks)/dt_pf:.1f} tok/s "
        f"(last-half chunk ms: med {np.median(_ct):.1f} min {min(_ct):.1f} max {max(_ct):.1f})", flush=True)
  lgB = logits_stats("PF_BATCH-prefill end logits")
  fnlg = np.linalg.norm(lgB - lgA) / max(np.linalg.norm(lgA), 1e-9)
  print(f"[gate] end-state logits F-relerr T1 vs PF_BATCH: {fnlg:.3e}", flush=True)
  if os.getenv("SCDBG_GATE_DUMP"):
    import numpy as _np
    _o = {"logits": _np.asarray(lgB, dtype=_np.float16)}
    for i in list(E.gdn_idx[:3]) + list(E.gdn_idx[-2:]):
      _o[f"rec{i}"] = E.P.down(f"rec{i}", (48*128*128,), _np.float32).copy(); E.P._keep.clear()
      _o[f"conv{i}"] = E.P.down(f"conv{i}_0", (3*10240,), _np.float32).copy(); E.P._keep.clear()
    _np.savez("~/gate_a2_state.npz", **_o)
    print("[gate] A2 state dumped", flush=True)
  if os.getenv("SCDBG_GATE_B") == "1":
    print("== B: PF_BATCH prefill PF_DFILL=0 (same world) ==", flush=True)
    os.environ["PF_DFILL"] = "0"
    CT.clear()
    dt_pf2, tfd3, cur3 = fresh(toks, "PF_BATCH")
    lgC = logits_stats("PF_BATCH-nodfill end logits")
    fnlg2 = np.linalg.norm(lgC - lgA) / max(np.linalg.norm(lgA), 1e-9)
    print(f"[gate] end-state logits F-relerr T1 vs PF_BATCH-nodfill: {fnlg2:.3e}", flush=True)
    tC0 = time.perf_counter(); tokC2, gapsC = t1_decode_gapped(NTOK); dtC = time.perf_counter() - tC0
    ag2 = sum(1 for a, b in zip(tokA, tokC2) if a == b)
    print(f"[GATE B] PF_BATCH-nodfill vs T1: {ag2}/{NTOK}", flush=True)
    print(f"[gate] tokB2[:20] {tokC2[:20]}", flush=True)
    if os.getenv("SCDBG_GATE_DUMP"):
      import numpy as _np
      _o = {"logits": _np.asarray(lgC, dtype=_np.float16)}
      for i in list(E.gdn_idx[:3]) + list(E.gdn_idx[-2:]):
        _o[f"rec{i}"] = E.P.down(f"rec{i}", (48*128*128,), _np.float32).copy(); E.P._keep.clear()
        _o[f"conv{i}"] = E.P.down(f"conv{i}_0", (3*10240,), _np.float32).copy(); E.P._keep.clear()
      _np.savez("~/gate_b_state.npz", **_o)
      print("[gate] B state dumped", flush=True)
    os.environ["PF_DFILL"] = "1"
  if os.getenv("PF_PG_BIT", "0") == "1":
    print("== PG-BIT: eager chunk vs captured chunk (bit-identity, same world) ==", flush=True)
    def _snap():
      S = {"logits": E.P.down("logits", (248320,), np.float16).copy(),
           "tok_slot": int(E.P.down_at("tok_slot", 0, 1)[0]),
           "pos_slot": int(E.P.down_at("pos_slot", 0, 1)[0])}
      _g = sorted(E.gdn_idx)
      for i in (_g[0], _g[len(_g)//3], _g[2*len(_g)//3], _g[-1]):
        S[f"rec{i}"] = E.P.down(f"rec{i}", (48*128*128,), np.float32).copy()
        S[f"conv{i}"] = E.P.down(f"conv{i}_0", (3*10240,), np.float32).copy()
      for i in list(E.qtypes)[:3]:
        S[f"kv{i}"] = E.P.down_at(f"kv{i}", (len(toks)-64)*2048, 64*2048, np.int8).copy()
        S[f"sc{i}"] = E.P.down_at(f"sc{i}", (len(toks)-64)*64, 64*64, np.int8).copy()
      E.P._keep.clear()
      return S
    os.environ["PF_PG"] = "0"; CT.clear()
    dte, _, _ = fresh(toks, "PF_BATCH")
    cte = list(CT); Se = _snap()
    os.environ["PF_PG"] = "1"; os.environ["PG_WAIT"] = "1"; CT.clear()
    dtg, _, _ = fresh(toks, "PF_BATCH")
    ctg = list(CT); Sg = _snap()
    if os.getenv("PG_PIPE_BIT", "0") == "1":   # OFF by default: graph-graph back-to-back
      # on the compute channel = SKEDCHECK22_INVALIDATE_ACTIVE_QMD fault class (P7F-1 law)
      os.environ["PG_WAIT"] = "0"; CT.clear()
      dtg0, _, _ = fresh(toks, "PF_BATCH")
      ctg0 = list(CT); Sg0 = _snap()
    else:
      dtg0, ctg0, Sg0 = -1.0, [], Se
    os.environ["PG_WAIT"] = "1"
    os.environ["PF_PG"] = "1"
    bad = [k for k in Se if not np.array_equal(np.asarray(Se[k]), np.asarray(Sg[k]))]
    bad0 = [k for k in Se if not np.array_equal(np.asarray(Se[k]), np.asarray(Sg0[k]))]
    cte_a, ctg_a = np.array(cte)[:,1], np.array(ctg)[:,1]
    print(f"[PG-BIT] eager {len(cte)} chunks mean {cte_a.mean():.1f} ms (F {cte_a[0]:.1f}/{cte_a[-1]:.1f}) "
          f"wall {dte:.1f}s | graph {len(ctg)} chunks mean {ctg_a.mean():.1f} ms (F {ctg_a[0]:.1f}/{ctg_a[-1]:.1f}) "
          f"wall {dtg:.1f}s | chunk speedup {cte_a.mean()/max(ctg_a.mean(),1e-9):.2f}x", flush=True)
    print(f"[GATE PG-BIT] eager vs captured (PG_WAIT=1): {'BIT-IDENTICAL (all keys)' if not bad else 'MISMATCH ' + str(bad)}", flush=True)
    if dtg0 > 0:
      print(f"[GATE PG-BIT] eager vs pipelined (PG_WAIT=0): {'BIT-IDENTICAL (all keys)' if not bad0 else 'MISMATCH ' + str(bad0)} | pipelined wall {dtg0:.1f}s", flush=True)
  if os.getenv("SCDBG_EARLY_EXIT") == "1": raise SystemExit(0)
  tB0 = time.perf_counter(); tokB, gapsB = t1_decode_gapped(NTOK); dtB = time.perf_counter() - tB0
  agree_ab = sum(1 for a, b in zip(tokA, tokB) if a == b)
  print(f"[gate] PF_BATCH prefill {dt_pf:.1f}s (fill_draft {tfd2:.1f}s), T1 decode {dtB:.1f}s", flush=True)
  print(f"[GATE A] T1-prefill vs PF_BATCH-prefill (both T1-decoded): {agree_ab}/{NTOK}", flush=True)
  print(f"[gate] tokB[:20] {tokB[:20]}", flush=True)
  if agree_ab < NTOK:
    for k, (a, b) in enumerate(zip(tokA, tokB)):
      if a != b:
        print(f"[gate] first divergence at {k}: {a} vs {b}", flush=True)
        print(f"[gate]   T1 path top1/2/gap: {gapsA[k]}", flush=True)
        print(f"[gate]   PF  path top1/2/gap: {gapsB[k]}", flush=True)
        break

  if os.getenv("SCDBG_SKIP_CD") == "1":
    np.save("~/pf_chunk_times.npy", np.array(CT))
    print("[gate] done (SCDBG_SKIP_CD)", flush=True)
    raise SystemExit(0)
  print("== CTRL: T1 prefill + SPEC decode (Tier-1 baseline for this text) ==", flush=True)
  dt_c1, _, _ = fresh(toks, "T1")
  tD0 = time.perf_counter(); tokD, rD = spec_decode(NTOK); dtD = time.perf_counter() - tD0
  agree_da = sum(1 for c, a in zip(tokD[:len(tokA)], tokA) if c == a)
  print(f"[gate] spec-after-T1-prefill decode {dtD:.1f}s ({len(tokD)} tokens, alpha-class {len(tokD)/NTOK:.2f} tok/cyc)", flush=True)
  print(f"[CTRL GATE] spec-after-T1 vs greedy-T1 (BOTH on T1 prefill): {agree_da}/{min(len(tokD), len(tokA))}", flush=True)
  print(f"[gate] tokD[:20] {tokD[:20]}", flush=True)

  print("== D: PF_BATCH prefill + SPEC decode (Tier-1) ==", flush=True)
  dt_pf2, tfd3, cur3 = fresh(toks, "PF_BATCH")
  tC0 = time.perf_counter(); tokC, rlast = spec_decode(NTOK); dtC = time.perf_counter() - tC0
  agree_cb = sum(1 for c, b in zip(tokC[:len(tokB)], tokB) if c == b)
  print(f"[gate] spec decode {dtC:.1f}s ({len(tokC)} tokens, {NTOK} cycles)", flush=True)
  print(f"[GATE D] spec-on-batched vs greedy-T1-on-batched: {agree_cb}/{min(len(tokC), len(tokB))}", flush=True)
  agree_cd = sum(1 for c, dd in zip(tokC[:len(tokD)], tokD) if c == dd)
  print(f"[GATE D2] spec-on-batched vs spec-on-T1: {agree_cd}/{min(len(tokC), len(tokD))}", flush=True)
  print(f"[gate] tokC[:20] {tokC[:20]}", flush=True)

  print(f"[BENCH] PF_BATCH prefill: {dt_pf:.1f}s for {len(toks)} pos = {len(toks)/dt_pf:.1f} tok/s; "
        f"T1: {dt_t1:.1f}s = {len(toks)/dt_t1:.1f} tok/s; speedup {dt_t1/dt_pf:.2f}x", flush=True)
  if CT:
    ps = sorted(CT)
    print("[BENCH] chunk ms by position (sample):", flush=True)
    idxs = list(range(0, len(ps), max(1, len(ps)//12))) + [len(ps)-1]
    for k in sorted(set(idxs)):
      p, ms = ps[k]
      print(f"  pos {p:6d}: {ms:7.1f} ms ({16/(ms/1000):.1f} tok/s)", flush=True)
  np.save("~/pf_chunk_times.npy", np.array(CT))
  print("[gate] done", flush=True)
