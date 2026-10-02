#!/usr/bin/env python3

# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""TLX DRAFTER Phase 2 (anchor-scale Stage B) — the engine decode-anchor dump.

THE PHASE-1 UNLOCK: 51 r8 anchors memorized; 2-10k anchors at serve positions
(64k-100k) from FRESH novel-prose sessions force generalization. This harness
captures, per session (bookcorpusopen indie novels, disjoint from r8):

  <out>/s{NN}/cycles.json          per-cycle records (verbatim test_w100k
                                   TLX_TRACE_DUMP semantics: pos/cur/tokens/
                                   dring/amds/prose/deep/t1 flags + h_idx)
  <out>/s{NN}/cycles_h_seed.npy    [ncyc, 5120] fp32 — the EXACT chain anchor
                                   (h_seed captured BEFORE each cycle's draft)
  <out>/s{NN}/kvd_decode_start.npy int8 [2,4,upto,256] + scd fp16 (the draft
                                   KV base at decode start — same shape as r8)
  <out>/s{NN}/prompt_ids.npy       the fed session prompt
  <out>/s{NN}/manifest.json        sha256 manifest (trace_dump_lib)

Globals (emb/head/grid512/stab) are NOT re-dumped — the r8_prose trace's
globals are the same model+pack; stab is sha-verified per session instead.

Env (via ~/p0_run.sh + overrides): PF_W4A8=0 TLX_T1_MODE=0 TLX_EAGLE_K=4
TLX_EAGLE_PROSE_TRIG=0 PC_ENABLED=0 (canonical dense env otherwise; the r8
capture env IDENTICALLY — anchor semantics match the eval trace bit-for-bit).

Resume: sessions with manifest.json present are skipped (idempotent reruns
across the box's reset class). ~4.5min/session at 64k-99.4k + ~12min boot.
"""
import os, sys, time, json, collections

os.environ["SKV"] = "1"
os.environ["SKV_CTXK"] = "100352"
os.environ.setdefault("DEV", "NV")
os.environ.pop("M1A_SERVE", None)          # NEVER attach the daemon

BASE = "/Users/lokm/tinygrad-metal"
sys.path.insert(0, "/Users/lokm/tinygrad-src")
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, BASE)

import numpy as np
from mtp import MTPEngine, DecodeSession, CTXK, SLICE
import pcache as _pc_mod
import trace_dump_lib as tdl
from engine0 import dev
from gcycle import GCycleEngine

SNAP = os.getenv("SNAPDIR", "/Users/lokm/snap100k")
SESSIONS = os.getenv("ANCHOR_SESSIONS", os.path.expanduser("~/anchor_sessions"))
OUT = os.getenv("ANCHOR_OUT", os.path.expanduser("~/trace_dump/anchor_scale"))
NTOK = int(os.getenv("ANCHOR_NCYC", "250"))
GLOBAL_REBUILD_EVERY = int(os.getenv("TLX_GLOBAL_CYCLE_REBUILD_EVERY", "928"))
POS_HARD_CAP = CTXK - 64                   # decode safety stop


def maybe_rebuild(E, where):
    """The daemon's _gpu_rpc_entry discipline: fence-class rebuild at a
    quiescent point when the GLOBAL submit counter crossed the envelope
    (prefill chunk replays spend it — the R6 GRAPH-CLASS PREFILL BUDGET law)."""
    if GLOBAL_REBUILD_EVERY and getattr(dev, "global_cycle_ctr", 0) >= GLOBAL_REBUILD_EVERY:
        dev.synchronize()
        E.build_graphs()
        dev.global_cycle_ctr = 0
        print(f"[rebuild] {where} (budget reset)", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    sess_man = json.load(open(f"{SESSIONS}/sessions.json"))
    rows = sorted(sess_man["rows"], key=lambda r: r["n"])   # cheap sessions first
    # keep the slot ordering: sort by target then keep canaries interleaved
    rows = sorted(sess_man["rows"], key=lambda r: (r["target"], r["split"] != "canary"))

    done = {r["f"] for r in rows if os.path.exists(f"{OUT}/{r['f']}/manifest.json")}
    todo = [r for r in rows if r["f"] not in done]
    print(f"[dump] {len(rows)} sessions, {len(done)} already done, {len(todo)} to run", flush=True)
    if not todo:
        print("[dump] NOTHING TO DO", flush=True)
        return

    # ---------- boot (dump_gsm8k.py verbatim: snap load -> canonical slice
    # -> init_draft -> fill_draft -> build_graphs) ----------
    meta = json.load(open(f"{SNAP}/meta.json"))
    P0 = int(meta["P"]); assert CTXK == int(meta["CTXK"])
    snap_ids = np.load(f"{SNAP}/ids.npy").tolist()

    t0 = time.perf_counter()
    E = MTPEngine(theta=1e7)
    E.load_snapshot_kv(SNAP, progress=False)
    for j, i in enumerate(E.gdn_idx):
        E.P.win_up(f"conv{i}_0", 0, np.load(f"{SNAP}/conv_{i}.npy", mmap_mode="r"))
        E.P.win_up(f"rec{i}", 0, np.load(f"{SNAP}/rc_{i}.npy", mmap_mode="r"))
        if j % 8 == 0:
            E._flush()
    E.P.win_up("tok_slot", 0, np.array([int(meta["cur0"])], dtype=np.int32))
    E.P.win_up("pos_slot", 0, np.array([P0], dtype=np.int32))
    E._flush()
    dev.synchronize()
    print(f"[boot] engine loaded {time.perf_counter()-t0:.0f}s", flush=True)

    G = GCycleEngine(E)
    G.build(); dev.synchronize()
    refsuf = ("_kv8_qh" if os.getenv("QH", "0") == "1" else "_kv8") if os.getenv("KV8", "0") == "1" else ""
    ref = np.load(f"{SNAP}/engine_t1_ref{refsuf}.npy").tolist()
    # the production slice construction (test_w100k verbatim — every daemon
    # boot + the r8 trace share THIS slice; stab is sha-verified below)
    seen, sl = set(), []
    for t in ref + snap_ids:
        if t not in seen: seen.add(t); sl.append(t)
    for t, _ in collections.Counter(snap_ids).most_common():
        if t not in seen: seen.add(t); sl.append(t)
    base = sl[:]
    while len(sl) < SLICE:
        sl += base
    E.init_draft(sl[:SLICE])
    E.fill_draft(snap_ids)
    E.build_graphs()
    dev.synchronize()
    print(f"[boot] graphs built ({time.perf_counter()-t0:.0f}s) — boot complete", flush=True)

    # stab provenance guard: our slice must equal the r8 trace's slice
    stab = E.P.down("stab", (SLICE,), np.int32)
    r8_stab = np.load("/Users/lokm/trace_dump/r8_prose/stab.npy")
    assert (stab == r8_stab).all(), "boot slice drifted from the r8 trace slice!"
    import hashlib
    sha = hashlib.sha256(stab.tobytes()).hexdigest()
    print(f"[boot] stab == r8 trace slice ({len(np.unique(stab))} distinct ids; provenance OK)", flush=True)

    # ---------- session loop ----------
    summary_path = f"{OUT}/summary.json"
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else []
    lk_active = int(os.getenv("LOOKUP_K", "0") or 0) > 0

    for si, r in enumerate(todo):
        f = r["f"]
        sdir = f"{OUT}/{f}"
        os.makedirs(sdir, exist_ok=True)
        t0 = time.perf_counter()
        ids = np.load(f"{SESSIONS}/{f}_ids.npy")
        toks = [int(t) for t in ids]
        print(f"[sess {si+1}/{len(todo)}] {f} split={r['split']} n={len(toks)} "
              f"target={r['target']}", flush=True)

        # FRESH conversation (serve h_prefill body verbatim; ingest=None — no
        # pcache in the dump process; interleaved dfill via PF_DFILL=1)
        maybe_rebuild(E, "pre-prefill")
        newcur, posn, _ = _pc_mod.fresh_prefill(E, G, toks)
        maybe_rebuild(E, "post-prefill")     # the daemon's generate-entry rebuild
        if lk_active:                        # seed the lookup history (serve law)
            E.P.win_up("tok_hist", 0, np.asarray(toks, dtype=np.int32))
            dev.synchronize()
        print(f"[sess] prefill done pos={posn} cur={newcur} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

        # decode-start kv base (BEFORE the decode appends own rows)
        tdl.dump_kvd(sdir, E, "decode_start", min(posn, CTXK))
        np.save(f"{sdir}/prompt_ids.npy", ids.astype(np.int32))
        np.save(f"{sdir}/stab.npy", stab)    # per-session copy (trainer convenience)

        # the trace-capture decode loop (test_w100k TLX_TRACE_DUMP verbatim)
        sess = DecodeSession(E)
        sess.begin()
        recs, hseeds = [], []
        emitted = 0
        for ci in range(NTOK):
            # the daemon's generate-cycle discipline: rebuild at the quiescent
            # point between cycles when the global submit budget crossed (250
            # cycles x ~5 submits > 928; sess.begin() re-anchors after — the
            # proven h_generate fence path; costs ~2 anchors to prose=0 flags)
            if GLOBAL_REBUILD_EVERY and getattr(dev, "global_cycle_ctr", 0) >= GLOBAL_REBUILD_EVERY:
                maybe_rebuild(E, f"mid-decode c{ci}")
                sess.begin()
            pre = tdl.dump_state_scalars(E)
            pre_hs = tdl.dump_h_seed(E)
            ran_prose_entry = int(getattr(sess, "prose", 0))
            deep_at_entry = int(getattr(sess, "deep", 0))
            t1_at_entry = int(getattr(sess, "t1mode", 0))
            rr = sess.step()
            rec = tdl.dump_cycle_record(E, rr, ran_prose_entry, pre)
            rec["deep_at_entry"] = deep_at_entry
            rec["t1_at_entry"] = t1_at_entry
            rec["h_idx"] = len(hseeds)
            hseeds.append(pre_hs)
            recs.append(rec)
            emitted += len(rr["tokens"])
            if int(rr.get("pos_new", 0)) > POS_HARD_CAP or int(rr.get("stop", 0)):
                print(f"[sess] early stop at cycle {ci} (pos_new={rr.get('pos_new')} "
                      f"stop={rr.get('stop')})", flush=True)
                break

        np.save(f"{sdir}/cycles_h_seed.npy", np.stack(hseeds))
        json.dump(recs, open(f"{sdir}/cycles.json", "w"), indent=1)
        nprose = sum(1 for c in recs if c.get("prose") and not c.get("deep_at_entry")
                     and not c.get("t1_at_entry"))
        ndeep = sum(1 for c in recs if c.get("deep_at_entry"))
        ms = [c["m"] for c in recs]
        tdl.finish_trace(sdir, dict(
            kind="anchor_scale_decode", f=f, split=r["split"], title=r["title"],
            ntok_prompt=len(toks), ncyc=len(recs), n_prose_keep=nprose, n_deep=ndeep,
            emitted=emitted, pos_start=int(recs[0]["pos"]) if recs else -1,
            pos_end=int(recs[-1]["pos"]) if recs else -1,
            lookup_k=int(os.getenv("LOOKUP_K", "0") or 0),
            eagle_k=int(os.getenv("TLX_EAGLE_K", "0") or 0),
            prose_trig=int(os.getenv("TLX_EAGLE_PROSE_TRIG", "4")),
            pf_w4a8=int(os.getenv("PF_W4A8", "0") or 0),
            t1_mode=int(os.getenv("TLX_T1_MODE", "0") or 0),
            stab_sha=sha, snap=SNAP, m_hist={str(k): ms.count(k) for k in sorted(set(ms))}))
        dt = time.perf_counter() - t0
        entry = dict(f=f, split=r["split"], ncyc=len(recs), n_prose_keep=nprose,
                     n_deep=ndeep, emitted=emitted, wall_s=round(dt, 1),
                     m_mean=round(float(np.mean(ms)), 3) if ms else None)
        summary.append(entry)
        json.dump(summary, open(summary_path, "w"), indent=1)
        print(f"[sess] {f} DONE {len(recs)} cycles ({nprose} prose anchors, "
              f"{ndeep} deep) m~{entry['m_mean']} in {dt:.0f}s", flush=True)
        # leave the device clean before the next FRESH
        dev.synchronize()

    ntr = sum(1 for s in summary if s["split"] == "train")
    nca = sum(1 for s in summary if s["split"] == "canary")
    atr = sum(s["n_prose_keep"] for s in summary if s["split"] == "train")
    aca = sum(s["n_prose_keep"] for s in summary if s["split"] == "canary")
    print(f"[dump] ALL DONE: {len(summary)} sessions ({ntr} train / {nca} canary), "
          f"~{atr} train anchors + ~{aca} canary anchors -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
