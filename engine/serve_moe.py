"""TLX P8 MoE SERVING BRIDGE — the Qwen3.6-35B-A3B daemon (serve_moe).

The serve.py CONFORMANCE CONTRACT (the exact RPC surface api_server.py drives)
on the MM P5-P7 MoE harness (Rig7 / GraphRunner pair graphs / the K-mix
selector / the chunk-256 PrefillFeed):

  prefill   FRESH | FOLLOW_UP | AUTO_CACHE  (+progress events) -> {pos, cur, fed, mode?, cached_tokens?}
  generate  max_cycles/stop_token_ids       (+cycle/done/cancelled events) -> {tokens, cycles, stop?, usage}
  cancel    side-channel flag (no ack)
  status    the h_status field set (model_id/ctxk/pos/cur/fed/dirty/config_fp/pc/...)
  shutdown  admin-gated clean exit (staydown + pcache flush)
  snapshot_save/load: NOT SUPPORTED v1 (admin-only surface; a clear error)

ARCHITECTURE: serve.py is imported as a LIBRARY for the protocol primitives
(send/_send_lock/_mark_dead/validate_rpc/check_admin/_peer_uid/_emit_seq_
violation/watchdog patterns) — its module state (ST) is SWAPPED for this
daemon's so the shared failure paths (send-stall cancel arming, slog uptime)
read OUR state. The dense daemon (serve.run_daemon) is untouched; the MoE
host (test_moe36.py) attaches HERE after building the rig + graphs.

MoE specifics vs the dense daemon:
  - pos/cur are HOST-side (the rig's cpu-mapped control ids_view/pos_view);
    ST.pos_cache/ST.cur_cache are the truth.
  - The lookup is the P6 incremental 4-gram index (host-side; rebuilt from
    ST.fed at each prefill — the R5d seeding law is trivially satisfied).
  - The cycle: T1 (miss) / D2 (n>=4) / D8 (n>=8) graph selection, the folded
    readback (eb = (m, amds[m]) after one timeline wait), acc36+selc36
    in-graph commit — emit records shaped exactly like the dense engine's
    (pos_new = pos + m + 1, tokens = m + 1) so _emit_seq_violation applies.
  - P10 THE MTP MODE (MM_MTP=1, default): the miss path becomes the P5+MTP
    cycle — the K=4 chain (MM_P9_mtp.MtpRig: blk.40 nextn, its own KV, the
    one-cycle-late EAGLE protocol) drafts 4 tokens, the P=5 probe verifies.
    Drafts are verification-only: the committed stream stays bit-identical
    to T1 by construction (the probe's bnd = the T1 greedy). The chain goes
    STALE on every lookup cycle / prefill; the first miss after a streak
    pays ONE T1 cycle whose hA[0] re-anchors the seed. force_mode gates:
    "t1" | "spec" (the P8 K-mix) | "mtp". MTP does NOT change trunk
    numerics -> config_fp/pcache namespace UNCHANGED (MM_MTP is not in
    svc_fp._ENV_KEYS, same reasoning as MM_SPEC).
  - THE PF-ONLY CUR FIX (P10): a prompt with len % 256 == 0 runs zero T1
    steps, so the post-feed greedy reads PFB["hA"][255] (the bit-exact
    chunk-256 hidden), not rig.hA (stale decode hidden — the old code's
    latent wrong-cur class).
  - The ~950-cycle dext budget: the rig's GraphRunner fences (1024) + a
    daemon-level fence every MOE_FENCE_CYCLES cycles rebuilding ALL runners
    (the E.build_graphs() equivalent; P10: includes gr5 + the 3 MTP chain
    runners).
  - pcache (pcache_moe.MoePromptCache): per-model root, r1moe1 nodes, the
    same chain-hash / transactional-restore / quarantine discipline.
  - Positions are absolute; feeds take (delta, pos0). Chunk boundaries land
    at pos0 + 256k; pcache ingest fires at ABSOLUTE STRIDE boundaries that
    coincide with chunk boundaries (FRESH from 0 always does).

GPU-free importability (the mock battery): numpy imports live inside
run_daemon_moe, mirroring serve.py.
"""
import os, sys, time, json, socket, threading, queue

import serve as SD          # the dense daemon module — protocol primitives ONLY

MOE_LOGF = "/tmp/moe36_serve.log"
MOE_LOGF_PERSIST = os.path.join(SD.LOGS_DIR, "moe36_serve.log")
SD.LOGF = MOE_LOGF
SD.LOGF_PERSIST = MOE_LOGF_PERSIST

from serve import (log, slog, send, _mark_dead, _peer_uid, check_admin,
                   validate_rpc, _emit_seq_violation, _log_rotate,
                   ALLOWED_PEER_UIDS, MAX_LINE_BYTES, SEND_TIMEOUT_S,
                   KEEPALIVE_S, STEP_TIMEOUT_S, _PRIV_METHODS,
                   _GPU_RPC, MAX_Q_DEPTH, MAX_CONNS, LOGS_DIR, STAYDOWN,
                   _PrefillCancelled, _exit_now)

# The engine socket (testable: TLX_ENGINE_SOCK overrides; the mock battery
# must never touch the production socket while the dense daemon serves).
SOCK = os.getenv("TLX_ENGINE_SOCK", SD.SOCK)

# ---- MoE daemon knobs (env-gated; defaults = the proven P7 classes) ----
DEC_S = int(os.getenv("MM_DEC_S", "32"))      # decode split-S (the L64/L96 class)
PF_S = int(os.getenv("MM_PF_S", "8"))         # prefill chunk split-S (the F1b class)
SPEC_ON = os.getenv("MM_SPEC", "1") == "1"    # K-mix (off = pure T1; the gates)
MTP_ON = os.getenv("MM_MTP", "1") == "1"      # P10: miss -> P5+MTP chain (prose lever)
MOE_FENCE_CYCLES = int(os.getenv("MOE_FENCE_CYCLES", "512"))
PF_CHUNK = 256                                # the bit-exact chunk-256 class (fixed)
VOCAB = 248320


# ---- THE LOOKUP (vendored verbatim from MM_P6_run.Lookup — the P6
# incremental 4-gram index; deterministic: longest match, tie -> latest
# occurrence; n>=8 -> kmax drafts (D8), 4..7 -> 2 (D2)). Vendored because
# MM_P6_run imports the GPU harness at module top (the mock battery must
# run this daemon CPU-only). -----------------------------------------------
class Lookup:
    def __init__(self, hist=None):
        self.idx4 = {}; self.h = []
        for t in (hist or []): self.append(t)
    def append(self, t):
        self.h.append(t)
        L = len(self.h)
        if L >= 4:
            self.idx4.setdefault(tuple(self.h[-4:]), []).append(L - 4)
    def scan(self, kmax=8, maxn=16):
        L = len(self.h)
        if L < 9: return 0, []
        cands = self.idx4.get(tuple(self.h[-4:]), [])
        best = None
        for q0 in reversed(cands):
            if q0 + 4 > L - 4: continue
            n = 4
            qq = q0
            while n < maxn and qq > 0 and self.h[qq - 1] == self.h[L - n - 1]:
                qq -= 1; n += 1
            avail = L - (qq + n)
            if best is None or n > best[0]:
                best = (n, qq, avail)
            if n >= maxn: break
        if best is None or best[0] < 4 or best[2] < 2: return 0, []
        n, qq, avail = best
        if n >= 8 and avail >= kmax:
            return n, list(self.h[qq + n : qq + n + kmax])
        if n >= 8 and avail >= 2:
            return n, list(self.h[qq + n : qq + n + min(2, avail)])
        if n >= 4 and avail >= 2:
            return n, list(self.h[qq + n : qq + n + 2])
        return 0, []

ST = SD.State()
ST.ready = False; ST.busy = False; ST.cancel = False
ST.t0 = time.time(); ST.mode = None
ST.fed = []; ST.convo_id = None; ST.dirty = False
ST.last_rpc = time.time()
ST.rpc = None; ST.config_fp = None; ST.vocab = VOCAB
ST.cycles_since_rebuild = 0; ST.step_beat = time.time()
ST.last_keepalive_ok = time.time(); ST.rebuild_fails = 0
ST.cancel_armed_by = None
ST.pos_cache = 0; ST.cur_cache = 0
ST.inline_status = None; ST.cancel_hook = None; ST.active_conn = None
ST.pc = None; ST.pc_root = None
ST.pc_last_end = 0; ST.pc_last_hkey = None
SD.ST = ST        # the shared failure paths (send/slog) read OUR state


def _clean_exit(code, reason, flush=True, staydown=False):
    """The one exit path (V-36/V-37 shape): socket unlink, persistent staydown
    marker (fsync + dir-fsync — the L5 law), pcache flush."""
    try: os.unlink(SOCK)
    except Exception: pass
    if staydown:
        try:
            os.makedirs(LOGS_DIR, exist_ok=True)
            with open(STAYDOWN, "a") as f:
                f.write(f"{time.time()} {reason}\n")
                f.flush(); os.fsync(f.fileno())
            dfd = os.open(LOGS_DIR, os.O_RDONLY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
        except Exception: pass
    if ST.pc is not None and flush:
        try: ST.pc.flush()
        except Exception as e:
            slog(op="clean_exit_flush_error", error=repr(e))
    slog(op="clean_exit", code=code, reason=reason, staydown=staydown)
    _exit_now(code, reason)


# ============================================================================
def run_daemon_moe(eng, CTXK):
  """eng: the MoE engine facade (test_moe36.MoeServeEngine): rig, dev, gr1/
  gr2/gr8/gr_pf, feed/feed9/feed_step/pf_feed_chunk/eager_head_cur/
  fence_all/keepalive_probe."""
  import numpy as np   # noqa: F401  (parity with serve.py's lazy-import law)
  import pcache
  import pcache_moe
  import svc_fp

  # config fingerprint: shared env keys + model identity (TLX_MODEL_PATH) +
  # the packed-manifest sha of THE PACK THE RIG LOADS (MM_P2_ports.PACK —
  # a repack changes numerics with NO env change; hashing the env's MM_PACKED
  # alone missed the iq4_xs hardcode).
  try:
    svc_fp.clear_extras()
    import MM_P2_ports as _P2
    svc_fp.set_mm_pack_extra({"MM_PACKED": _P2.PACK})
    slog(op="mm_pack_fp", pack=_P2.PACK, sha=svc_fp.extra_fp().get("mm_pack"))
  except Exception as e:
    slog(op="mm_pack_fp_failed", error=repr(e))
  ST.config_fp = svc_fp.config_fp()

  lk = [None]           # the resident conversation's lookup index
  # P10: the MTP chain state (host-side; {"pos","cur","drafts"} — valid only
  # while pos/cur match the engine frontier: the drafts are a pure function
  # of the committed state + the MTP-KV conditioning, both untouched by a
  # pause) + the last-feed hidden source for the initial chain seed
  # ("hA",0) after a T1 tail / ("pf",255) after a PF-only feed / None).
  ST.mtp_chain = None
  feed_hidden = [None]

  def tl(): return getattr(eng.dev, "timeline_value", -1)

  def _beat():
    ST.step_beat = time.time()
    ST.last_keepalive_ok = ST.step_beat

  def _eager_cur():
    """P10 FIX (the PF-only latent class): the greedy `cur` after a feed must
    read the hidden the feed ACTUALLY produced. A T1 tail leaves it in
    rig.hA row 0; a PF-only feed (len % 256 == 0) leaves it in PFB["hA"]
    seat 255 — the OLD code always read rig.hA, so PF-only prompts derived
    cur from a STALE decode-graph hidden (G3's both-arms-same-garbage class:
    invisible to spec-vs-t1 gates, wrong vs any offline anchor)."""
    src = feed_hidden[0]
    if src is None:
      return eng.eager_head_cur()
    if src[0] == "pf":
      return eng.eager_head_cur(seat=src[1], pf=True)
    return eng.eager_head_cur()

  # ---- pcache boot ----
  if os.getenv("PC_ENABLED", "0") == "1":
    try:
      root = os.getenv("PC_ROOT", os.path.expanduser("~/prompt_cache/qwen3.6-35b-a3b-egpu"))
      if os.getenv("PC_QUOTA_GB"):
          pcache.QUOTA_BYTES = int(float(os.getenv("PC_QUOTA_GB")) * 1e9)
      ST.pc = pcache_moe.MoePromptCache(root)
      ST.pc_root = root
      slog(op="pc_boot", stage="loaded", entries=len(ST.pc.man["entries"]),
           bytes_gb=round(ST.pc.total_bytes() / 1e9, 2),
           quota_gb=round(pcache.QUOTA_BYTES / 1e9, 1), root=root)
    except Exception as e:
      slog(op="pc_boot_error", error=repr(e))
      ST.pc = None

  # ---- L7 abort-safety (the same protocol, MoE bodies) ----
  def _arm_cancel(src):
    ST.cancel = True
    ST.cancel_armed_by = (src, ST.rpc)

  def _gpu_rpc_entry():
    ST.cancel = False
    ST.cancel_armed_by = None

  def cancel_checkpoint(stage):
    """The ONLY legal raise site (L7 FIX 2): drain then raise."""
    if not ST.cancel:
      return
    _beat()
    eng.dev.synchronize()
    slog(op="abort", stage=stage, tl=tl(), armed_by=ST.cancel_armed_by,
         cycles_since_rebuild=ST.cycles_since_rebuild)
    raise _PrefillCancelled()

  def keepalive():
    eng.keepalive_probe()
    ST.last_keepalive_ok = time.time()

  def _pc_status():
    if ST.pc is None:
      return None
    try:
      return {"entries": len(ST.pc.man["entries"]), "total_bytes": ST.pc.total_bytes(),
              "quota_bytes": pcache.QUOTA_BYTES, "stats": dict(ST.pc.stats)}
    except Exception as e:
      return {"error": repr(e)}

  def h_status(p):
    return {"ready": ST.ready, "ctxk": CTXK, "busy": ST.busy, "rpc": ST.rpc,
            "pos": ST.pos_cache, "mode": ST.mode, "cur": ST.cur_cache,
            "fed_len": len(ST.fed), "fed_tail": ST.fed[-64:],
            "conversation_id": ST.convo_id, "queue": Q.qsize() if Q else 0,
            "dirty": ST.dirty, "keepalive_s": KEEPALIVE_S,
            "uptime_s": round(time.time() - ST.t0, 1),
            "config_fp": ST.config_fp,
            "model_id": SD._MODEL_ID,
            "cycles_since_rebuild": ST.cycles_since_rebuild,
            "cycle_cap": max(1, min(4096, CTXK - max(0, int(ST.pos_cache)))),
            "pc": _pc_status(),
            "spec": {"dec_s": DEC_S, "pf_s": PF_S, "spec_on": SPEC_ON,
                     "mtp": bool(getattr(eng, "mtp_on", False)) and MTP_ON}}

  # ---- pcache capture / restore ----
  def _pc_node_done(hk, B):
    ST.pc_last_end = B; ST.pc_last_hkey = hk
    ST.pc.add_protect(hk)

  def _pc_maybe_ingest(at_pos, full_toks):
    """At a chunk boundary of a feed: capture a node when at_pos is an
    ABSOLUTE STRIDE boundary (FRESH feeds from 0 always coincide) with a
    >= HASH_BLK window since the last node. cur = the eager head on the
    chunk's last seat (the token at at_pos)."""
    if ST.pc is None or full_toks is None:
      return
    A = ST.pc_last_end
    if at_pos % pcache.STRIDE != 0 or at_pos - A < pcache_moe.HASH_BLK:
      return
    if at_pos > len(full_toks):
      return
    t0 = time.perf_counter()
    try:
      cur = eng.eager_head_cur(seat=PF_CHUNK - 1, pf=True)
      node = pcache_moe.capture_node_moe(eng.rig, A, at_pos, fed_prefix=full_toks,
                                         parent=ST.pc_last_hkey, cur=cur)
      ST.pc.write_node(node)
      _pc_node_done(node["hkey"], at_pos)
      slog(op="pc_ingest", pos=at_pos, win=[A, at_pos],
           secs=round(time.perf_counter() - t0, 2), wq=ST.pc.wq.qsize())
    except Exception as e:
      slog(op="pc_ingest_error", pos=at_pos, error=repr(e))

  def _pc_turn_end(pos):
    if ST.pc is None: return
    if pos % pcache_moe.HASH_BLK != 0 or pos - ST.pc_last_end < pcache_moe.HASH_BLK:
      return
    if pos > len(ST.fed): return
    try:
      t0 = time.perf_counter()
      node = pcache_moe.capture_node_moe(eng.rig, ST.pc_last_end, pos,
                                         fed_prefix=ST.fed, parent=ST.pc_last_hkey,
                                         cur=ST.cur_cache)
      ST.pc.write_node(node)
      _pc_node_done(node["hkey"], pos)
      slog(op="pc_turn_end_ingest", pos=pos, win_secs=round(time.perf_counter() - t0, 2))
    except Exception as e:
      slog(op="pc_turn_end_error", error=repr(e))

  def _pc_prefill(conn, rid, p, toks):
    """AUTO_CACHE: deepest-chain restore + tail. None -> caller falls FRESH."""
    ttl = p.get("cache_ttl") or p.get("prompt_cache_ttl")
    min_cov = max(pcache.MIN_HIT, int(0.5 * len(toks)))
    t0 = time.perf_counter()
    B, chain = ST.pc.lookup(toks, min_hit=min_cov)
    if not chain:
      slog(op="pc_lookup", stage="miss", n=len(toks), secs=round(time.perf_counter() - t0, 2))
      return None
    slog(op="pc_lookup", stage="hit", B=B, nodes=len(chain),
         cached_gb=round(sum(int(e["bytes"]) for _, e in chain) / 1e9, 2))
    ST.pc.touch(chain)
    ck = p.get("cache_key") or p.get("prompt_cache_key")
    if ck: ST.pc.pin(chain, str(ck)[:128], ttl=ttl)
    ST.pc.set_protect(hk for hk, _ in chain)
    t0 = time.perf_counter()
    ST.mutated = True
    ST.mtp_chain = None; feed_hidden[0] = None   # P10: restore provides no hidden
    try:
      B2, cur = pcache_moe.restore_chain_moe(eng.rig, chain, ST.pc_root, beat=_beat)
      assert B2 == B
    except pcache_moe.NodeCorrupt as nc:
      slog(op="pc_corrupt", hkey=(nc.hkey or "")[:16], why=nc.why)
      ST.pc.quarantine(nc.hkey)
      eng.rig.reset_states(2048)
      return None
    if cur is None:
      # legacy/hand-made node without meta cur: re-forward the boundary token
      # (idempotent row rewrite; the GDN state lands exactly on the restored
      # boundary) and read the greedy.
      eng.feed(int(toks[B - 1]), B - 1)
      eng.gr1.step()
      cur = int(eng.rig.am_view[0])
    ST.fed = list(toks[:B]); ST.mode = "CACHE_HIT"; ST.convo_id = p.get("conversation_id")
    _pc_node_done(chain[-1][0], B)
    ST.pos_cache = B; ST.cur_cache = cur
    slog(op="pc_restore", stage="done", B=B, secs=round(time.perf_counter() - t0, 1))
    if len(toks) > B:
      t0 = time.perf_counter()
      _feed(toks[B:], B, full_toks=toks, conn=conn, rid=rid)
      cur = _eager_cur()
      ST.fed = list(toks)
      slog(op="pc_tail", stage="done", n=len(toks) - B, secs=round(time.perf_counter() - t0, 1))
    ST.pos_cache = len(toks); ST.cur_cache = cur
    return {"pos": ST.pos_cache, "cur": cur, "fed": len(ST.fed),
            "mode": "CACHE_HIT", "cached_tokens": B}

  # ---- the core feed path: (delta tokens, absolute pos0) ----
  def _feed(delta, pos0, full_toks, conn=None, rid=None):
    prog = _mk_prog(conn, rid) if conn is not None else None
    n = len(delta); i = 0; chunks = 0
    last_seat = PF_CHUNK - 1
    t0 = time.perf_counter()
    while i + PF_CHUNK <= n:
      cancel_checkpoint("prefill_pf_chunk")
      eng.pf_feed_chunk([int(x) for x in delta[i:i + PF_CHUNK]], pos0 + i)
      i += PF_CHUNK; chunks += 1
      _beat()
      if prog is not None and chunks % 2 == 0:
        prog(i, n, "prefill_pf")
      _pc_maybe_ingest(pos0 + i, full_toks)
      last_seat = PF_CHUNK - 1
    # SESSION B (L5): the 1..255 tail runs 64-seat PF chunks at chunk rate
    # (bit-exact vs the per-token T1 tail by the chunk-256 construction)
    # before the final <64 per-token tail.
    if getattr(eng, "pf64_on", False) and getattr(eng, "gr_pf64", None) is not None:
      while n - i >= 64:
        cancel_checkpoint("prefill_pf64_chunk")
        eng.pf64_feed_chunk([int(x) for x in delta[i:i + 64]], pos0 + i)
        i += 64; last_seat = 63
        _beat()
        if prog is not None:
          prog(i, n, "prefill_pf")
        if (pos0 + i) % PF_CHUNK == 0:
          _pc_maybe_ingest(pos0 + i, full_toks)
    pf_ms = (time.perf_counter() - t0) * 1e3
    for q in range(i, n):
      if (q - i) % 16 == 0:
        cancel_checkpoint("prefill_t1_tail")
      eng.feed_step(int(delta[q]), pos0 + q)
      _beat()
      if prog is not None and (q - i) % 16 == 0:
        prog(q - i + 1, n - i, "prefill_t1")
    if n > 0:
      # P10: where the LAST fed token's hidden lives (the bit-exact
      # chunk-256 class: PF seat 255 == the T1 hidden by construction).
      # SESSION B: the final chunk may be a 64 (seat 63) -- last_seat tracks
      # the final chunk's last seat; the T1 tail (i < n) anchors at hA[0].
      feed_hidden[0] = ("hA", 0) if i < n else ("pf", last_seat)
    return chunks, pf_ms

  def _mk_prog(conn, rid):
    def prog(done, total, stage="prefill_pf"):
      _beat()
      send(conn, {"id": rid, "event": "prefill_progress", "stage": stage,
                  "done": done, "total": total}, abort_on_fail=True,
           cancel_fn=lambda: _arm_cancel("send_stalled"))
      cancel_checkpoint(stage)
    return prog

  # ---- prefill handler ----
  def h_prefill(conn, rid, p):
    if "snapshot" in p:
      raise ValueError("snapshot prefill is not supported for the MoE model "
                       "(no base-snapshot boot path; use FRESH/AUTO_CACHE)")
    mode = p.get("mode", "FRESH"); toks = [int(t) for t in p["ids"]]
    ST.mutated = False      # FIRST: validation raises below touch no engine state
    if mode == "FOLLOW_UP":
      req_cid = p.get("conversation_id")
      if req_cid != ST.convo_id:
        raise ValueError(f"FOLLOW_UP conversation_id mismatch: engine resident="
                         f"{ST.convo_id!r} request={req_cid!r} fed_len={len(ST.fed)}")
      if ST.dirty:
        raise ValueError("engine state dirty (cancel/fault); FRESH required")
      cur_override = p.get("cur")
      if cur_override is not None and int(cur_override) != int(ST.cur_cache):
        raise ValueError(f"FOLLOW_UP cur override {cur_override} != resident cur "
                         f"{ST.cur_cache} (streams diverged; FRESH required)")
    ST.convo_id = p.get("conversation_id")
    if mode in ("FRESH", "AUTO_CACHE") and ST.pc is not None:
      r = _pc_prefill(conn, rid, p, toks)
      if r is not None:
        return r
    if mode == "FOLLOW_UP":
      _beat()
      ST.mutated = True
      ST.mtp_chain = None; feed_hidden[0] = None   # P10: positions shift; the chain re-seeds after the feed
      # the cur token sits at position len(fed) and is NOT re-fed (it is the
      # pending seat-0 token of the next generate — the dense contract);
      # the delta fills [len(fed)+1, ...)
      pos0 = len(ST.fed) + 1
      slog(op="follow_up", stage="begin", n=len(toks), pos0=pos0, cur=ST.cur_cache)
      _prefix_len = len(ST.fed)
      _feed(toks, pos0, full_toks=None, conn=conn, rid=rid)
      newcur = _eager_cur()
      ST.fed = ST.fed[:_prefix_len] + [ST.cur_cache] + toks
      ST.mode = "FOLLOW_UP"
      ST.pos_cache = len(ST.fed); ST.cur_cache = newcur
      r = {"pos": ST.pos_cache, "cur": newcur, "fed": len(toks), "cached_tokens": _prefix_len}
      slog(op="follow_up", stage="done", **r)
      return r
    # FRESH
    slog(op="prefill_fresh", stage="begin", n=len(toks))
    _beat()
    ST.mutated = True       # engine state changes from here on
    ST.mtp_chain = None; feed_hidden[0] = None
    eng.rig.reset_states(2048)
    if getattr(eng, "mtp_on", False):
      n_mtp = eng.mtp_reset_kv()      # zero exactly the chain-written prefix
      slog(op="mtp_kv_reset", rows=n_mtp)
    ST.pc_last_end = 0; ST.pc_last_hkey = None
    if ST.pc is not None: ST.pc.clear_protect()
    _feed(toks, 0, full_toks=toks, conn=conn, rid=rid)
    newcur = _eager_cur()
    ST.fed = list(toks); ST.mode = "FRESH"
    ST.pos_cache = len(toks); ST.cur_cache = newcur
    r = {"pos": ST.pos_cache, "cur": newcur, "fed": len(toks), "mode": "FRESH", "cached_tokens": 0}
    slog(op="prefill_fresh", stage="done", **r)
    return r

  # ---- generate handler (the K-mix folded cycle + the P10 MTP chain) ----
  def h_generate(conn, rid, p):
    if ST.mode is None:
      raise ValueError("no resident conversation: prefill before generate")
    mc = int(p.get("max_cycles", 60)); stops = set(int(t) for t in p.get("stop_token_ids", []))
    force = p.get("force_mode")          # gates: "t1" | "spec" | "mtp" (None = env default)
    if force not in (None, "t1", "spec", "mtp"):
      raise ValueError(f"force_mode must be one of t1|spec|mtp (got {force!r})")
    mtp = getattr(eng, "mtp_on", False) and (MTP_ON if force is None else force == "mtp")
    ST.cancel = False; ST.cancel_armed_by = None
    t0 = time.perf_counter()
    if lk[0] is None:
      lk[0] = Lookup(ST.fed)
    L = lk[0]
    slog(op="generate", stage="begin", max_cycles=mc, force=force, pos=ST.pos_cache, mtp=mtp)
    all_toks = []
    k = 0
    ST.mutated = True       # cycles write engine state immediately
    pos = ST.pos_cache; cur = ST.cur_cache
    _prev_pos = [pos]
    stats = {"cyc": 0, "cyc_t1": 0, "cyc_d2": 0, "cyc_d8": 0, "cyc_p5": 0,
             "hits": 0, "msum": 0, "m5_sum": 0, "stale_rec": 0, "chain_n": 0}
    # ---- P10: the chain (valid only while pos/cur match the frontier) ----
    chain = ST.mtp_chain
    if mtp and chain is not None and (chain["pos"] != pos or chain["cur"] != cur):
      chain = None
    if mtp and chain is None:
      # INITIAL SEED straight from the prefill's last hidden (the harness's
      # initial-chain path — no T1 re-anchor needed when a feed just ran).
      src = feed_hidden[0]
      if src is not None and pos > 0 and ST.fed:
        tc = time.perf_counter()
        try:
          drafts0 = eng.mtp_run_chain(src[0], src[1], ST.fed[-1], pos - 1, cur)
          chain = {"pos": pos, "cur": cur, "drafts": drafts0}
          stats["chain_n"] += 1
          slog(op="mtp_seed", src=src[0], row=src[1], pos=pos,
               ms=round((time.perf_counter() - tc) * 1e3, 1))
        except Exception as e:
          slog(op="mtp_seed_error", error=repr(e))
          chain = None
      # else: the first miss pays ONE T1 cycle (the stale-recovery path)
    feed_hidden[0] = None   # consumed; only a prefill re-arms it
    try:
      for k in range(1, mc + 1):
        _beat()
        if ST.cancel:
          send(conn, {"id": rid, "event": "cancelled", "tokens": all_toks, "cycles": k - 1}, abort_on_fail=True)
          ST.fed = ST.fed + all_toks
          ST.mtp_chain = chain
          slog(op="generate", stage="cancelled", cycles=k - 1, ntoks=len(all_toks))
          return {"cancelled": True, "tokens": all_toks, "cycles": k - 1}
        # --- the mode select (the scan-oracle policy + the MTP chain) ---
        n, drafts = (0, [])
        spec = SPEC_ON if force is None else force in ("spec", "mtp")
        if spec:
          n, drafts = L.scan(kmax=8)
        if n >= 8 and len(drafts) >= 8:
          K, gr, tag = 8, eng.gr8, "d8"
        elif n >= 4 and len(drafts) >= 2:
          K, gr, tag = 2, eng.gr2, "d2"
        else:
          K, gr, tag = 0, None, "t1"
        # --- the cycle ---
        if K == 0 and mtp and chain is not None:
          # ---- THE P5+MTP CYCLE (the prose lever): probe verifies
          # [cur + 4 MTP drafts]; emitted = accepted prefix + bnd; then the
          # one-cycle-late chain re-seeds from the probe's accepted seat. ----
          dr = chain["drafts"]
          ids9 = [int(cur)] + [int(d) for d in dr] + [0] * 4
          eng.feed9(ids9, pos)
          eng.gr5.step()
          m = int(eng.rig.eb_view[0]); bnd = int(eng.rig.eb_view[1])
          toks_out = [int(x) for x in dr[:m]] + [bnd]
          pos += m + 1; cur = bnd
          stats["cyc_p5"] += 1; stats["m5_sum"] += m
          tc = time.perf_counter()
          chain = {"pos": pos, "cur": cur,
                   "drafts": eng.mtp_run_chain("hA", m, ids9[m], pos - 1, bnd)}
          stats["chain_n"] += 1
          stats["ms_chain"] = stats.get("ms_chain", 0.0) + (time.perf_counter() - tc)
        elif K == 0 and mtp:
          # ---- STALE RECOVERY: one T1 cycle re-anchors hA[0] (a deferred
          # chain cannot seed — the prefill/PF graphs own the hidden). The
          # T1's token is the probe's own bnd class: exact by construction. ----
          eng.feed(int(cur), pos)
          eng.gr1.step()
          t2 = int(eng.rig.am_view[0])
          toks_out = [t2]; pos += 1
          tc = time.perf_counter()
          chain = {"pos": pos, "cur": t2,
                   "drafts": eng.mtp_run_chain("hA", 0, cur, pos - 1, t2)}
          stats["chain_n"] += 1
          stats["ms_chain"] = stats.get("ms_chain", 0.0) + (time.perf_counter() - tc)
          cur = t2
          stats["cyc_t1"] += 1; stats["stale_rec"] += 1
        elif K == 0:
          eng.feed(int(cur), pos)
          eng.gr1.step()
          t2 = int(eng.rig.am_view[0])
          toks_out = [t2]; pos += 1; cur = t2
          stats["cyc_t1"] += 1
        else:
          eng.feed9([int(cur)] + [int(d) for d in drafts[:K]] + [0] * (8 - K), pos)
          gr.step()
          m = int(eng.rig.eb_view[0]); bnd = int(eng.rig.eb_view[1])
          toks_out = [int(x) for x in drafts[:m]] + [bnd]
          pos += m + 1; cur = bnd
          stats[f"cyc_{tag}"] += 1; stats["hits"] += 1; stats["msum"] += m
          if mtp:
            chain = None            # STALE (the deferred-recovery discipline)
        stats["cyc"] += 1
        rec = {"m": len(toks_out) - 1, "pos_new": pos, "tokens": toks_out}
        _viol = _emit_seq_violation(_prev_pos[0], rec)
        _prev_pos[0] = pos
        if _viol is not None:
          ST.dirty = True
          slog(op="emit_seq_violation", violation=_viol, k=k, tl=tl())
          send(conn, {"id": rid, "event": "error", "error": f"emit_seq_violation: {_viol}"}, abort_on_fail=True)
          raise RuntimeError(f"emit_seq_violation: {_viol}")
        all_toks += toks_out
        ST.cycles_since_rebuild += 1
        ST.pos_cache = pos; ST.cur_cache = cur
        ST.mtp_chain = chain if mtp else None
        _beat()
        if ST.cycles_since_rebuild >= MOE_FENCE_CYCLES:
          try:
            eng.fence_all()
            ST.cycles_since_rebuild = 0; ST.rebuild_fails = 0
            slog(op="gen_rebuild", k=k, fences=eng.rig.fence_count, tl=tl())
          except Exception as e:
            ST.rebuild_fails += 1
            slog(op="gen_rebuild_failed", error=repr(e), fails=ST.rebuild_fails)
            if ST.rebuild_fails >= 3:
              _clean_exit(1, "rebuild_budget_exhausted", flush=True)
        send(conn, {"id": rid, "event": "cycle", "cycle": k, "pos": pos, "tokens": toks_out}, abort_on_fail=True)
        for t in toks_out:
          L.append(t)
        if stops & set(toks_out):
          ST.fed = ST.fed + all_toks
          ST.mtp_chain = chain if mtp else None
          _pc_turn_end(pos)
          send(conn, {"id": rid, "event": "done", "tokens": all_toks, "cycles": k, "pos": pos, "stop": True,
                      "usage": {"tokens": len(all_toks), "cycles": k, "secs": round(time.perf_counter() - t0, 2)}}, abort_on_fail=True)
          slog(op="generate", stage="done_stop", cycles=k, ntoks=len(all_toks), stop=True, **stats)
          return {"tokens": all_toks, "cycles": k, "stop": True}
      ST.fed = ST.fed + all_toks
      ST.mtp_chain = chain if mtp else None
      _pc_turn_end(pos)
      send(conn, {"id": rid, "event": "done", "tokens": all_toks, "cycles": mc, "pos": pos,
                  "usage": {"tokens": len(all_toks), "cycles": mc, "secs": round(time.perf_counter() - t0, 2)}}, abort_on_fail=True)
      slog(op="generate", stage="done_max", cycles=mc, ntoks=len(all_toks), **stats)
      return {"tokens": all_toks, "cycles": mc}
    except Exception:
      ST.fed = ST.fed + all_toks
      ST.mtp_chain = None
      ST.dirty = True
      slog(op="generate", stage="faulted", cycles_completed=k - 1, ntoks=len(all_toks))
      raise

  def _reseed_lookup():
    lk[0] = None      # lazy rebuild at the next generate (post-prefill state)

  # ---- the dispatcher ----
  def handle(conn, req):
    rid = req.get("id"); m = req.get("method"); p = req.get("params") or {}
    slog(op="rpc", method=m, tl=tl(), busy=ST.busy, rid=str((p or {}).get("rid") or "")[:40])
    if m in _PRIV_METHODS or (m == "prefill" and "snapshot" in p):
      ok, why = check_admin(p)
      if not ok:
        send(conn, {"id": rid, "ok": False, "error": f"admin required: {why}"})
        slog(op="admin_denied", method=m)
        return None
    if m in ("generate", "prefill"):
      err, p2 = validate_rpc(m, p, CTXK, ST.vocab or VOCAB, ST.pos_cache)
      if err is not None:
        send(conn, {"id": rid, "ok": False, "error": f"invalid params: {err}"})
        slog(op="rpc_rejected", method=m, error=err)
        return None
      p = p2
      if m == "prefill":
        n_ids = len(p.get("ids") or [])
        if ST.pos_cache + n_ids + 64 > CTXK:
          send(conn, {"id": rid, "ok": False,
                      "error": f"prompt too long: pos {ST.pos_cache} + {n_ids} > ctxk {CTXK} - 64"})
          slog(op="rpc_rejected", method=m, error="ctx overflow")
          return None
    gpu_rpc = m in _GPU_RPC
    if gpu_rpc:
      _gpu_rpc_entry()
      ST.busy = True; ST.rpc = m; _beat()
      ST.active_conn = conn
    try:
      if m == "generate":
        r = h_generate(conn, rid, p)
        ST.last_rpc = time.time()
        return r
      if m == "status":
        r = h_status(p)
      elif m == "prefill":
        r = h_prefill(conn, rid, p)
        ST.dirty = False
        _reseed_lookup()
      elif m in ("snapshot_save", "snapshot_load"):
        raise ValueError("snapshots are not supported for the MoE model v1")
      elif m == "cancel":
        return {"cancelled": True}
      elif m == "shutdown":
        send(conn, {"id": rid, "ok": True, "result": {"bye": True}})
        eng.dev.synchronize()
        slog(op="shutdown", stage="clean_exit")
        _clean_exit(0, "shutdown_rpc", flush=True, staydown=True)
      else:
        raise ValueError(f"unknown method {m}")
      send(conn, {"id": rid, "ok": True, "result": r})
      ST.last_rpc = time.time()
      return r
    except _PrefillCancelled:
      _beat()
      eng.dev.synchronize()
      ST.dirty = True
      ST.convo_id = None; ST.fed = []; ST.mode = None
      ST.pos_cache = 0; ST.cur_cache = 0
      ST.mtp_chain = None; feed_hidden[0] = None
      lk[0] = None
      try: send(conn, {"id": rid, "ok": False, "error": "cancelled"})
      except Exception: pass
      slog(op="prefill", stage="cancelled", armed_by=ST.cancel_armed_by)
      return None
    except Exception as e:
      import traceback; traceback.print_exc()
      try: eng.dev.synchronize()
      except Exception: pass
      slog(op="rpc_error", method=m, error=repr(e), tb=traceback.format_exc()[-2000:])
      try: send(conn, {"id": rid, "ok": False, "error": str(e)})
      except Exception: pass
      # a validation-class raise (pre-mutation) is a CLIENT error — the
      # engine state is trustworthy; only post-mutation faults dirty it.
      if m in ("prefill", "generate") and getattr(ST, "mutated", True):
        ST.dirty = True
      if "Device fault" in repr(e) or "device hang" in repr(e).lower():
        slog(op="device_fault", stage="exiting")
        _clean_exit(1, "device_fault", flush=True)
      return None
    finally:
      if gpu_rpc:
        ST.busy = False; ST.rpc = None
        if ST.active_conn is conn:
          ST.active_conn = None

  # ---- threads (the serve.py shapes) ----
  Q = queue.Queue(maxsize=MAX_Q_DEPTH)

  ST.inline_status = lambda req_id: {"id": req_id, "ok": True, "result": {
      "ready": ST.ready, "ctxk": CTXK, "busy": ST.busy, "pos": ST.pos_cache,
      "rpc": ST.rpc, "mode": ST.mode, "cur": ST.cur_cache, "fed_len": len(ST.fed),
      "fed_tail": ST.fed[-16:], "conversation_id": ST.convo_id,
      "dirty": ST.dirty, "queue": Q.qsize(),
      "keepalive_s": KEEPALIVE_S, "uptime_s": round(time.time() - ST.t0, 1),
      "config_fp": ST.config_fp, "model_id": SD._MODEL_ID,
      "cycles_since_rebuild": ST.cycles_since_rebuild,
      "cycle_cap": max(1, min(4096, CTXK - max(0, int(ST.pos_cache)))),
      "pc": _pc_status(),
      "spec": {"dec_s": DEC_S, "pf_s": PF_S, "spec_on": SPEC_ON,
               "mtp": bool(getattr(eng, "mtp_on", False)) and MTP_ON}}}

  def _cancel_hook(conn):
    if ST.active_conn is not None and conn is not ST.active_conn:
      slog(op="cancel_scoped_ignored")
      return
    ST.cancel = True
    ST.cancel_armed_by = ("cancel_rpc", ST.rpc)
  ST.cancel_hook = _cancel_hook

  _conn_count = {"n": 0}
  _conn_g = threading.Lock()

  def listener_conn(conn):
    with _conn_g:
      _conn_count["n"] += 1
      n_now = _conn_count["n"]
    if n_now > MAX_CONNS:
      try: conn.close()
      except Exception: pass
      with _conn_g: _conn_count["n"] -= 1
      slog(op="conn_cap_exceeded", active=n_now, cap=MAX_CONNS)
      return
    uid = _peer_uid(conn)
    if uid is not None and uid not in ALLOWED_PEER_UIDS:
      slog(op="socket_peer_denied", peer_uid=uid)
      try: conn.close()
      except Exception: pass
      return
    if uid is None and not SD.PEERCRED_LENIENT:
      slog(op="socket_peer_denied", note="LOCAL_PEERCRED unavailable; fail-closed")
      try: conn.close()
      except Exception: pass
      return
    buf = b""
    conn.settimeout(SEND_TIMEOUT_S)
    try:
      while True:
        try:
          chunk = conn.recv(65536)
        except socket.timeout:
          continue
        if not chunk: break
        buf += chunk
        if len(buf) > MAX_LINE_BYTES:
          slog(op="line_too_long", pending=len(buf), cap=MAX_LINE_BYTES, dropped=True)
          break
        while b"\n" in buf:
          line, buf = buf.split(b"\n", 1)
          if not line.strip(): continue
          if len(line) > MAX_LINE_BYTES:
            slog(op="line_too_long", line=len(line), cap=MAX_LINE_BYTES, dropped=True)
            return
          try: req = json.loads(line)
          except Exception: continue
          if req.get("method") == "cancel":
            ST.cancel_hook(conn)
          elif req.get("method") == "status" and ST.ready:
            try: send(conn, ST.inline_status(req.get("id")))
            except Exception: pass
          else:
            try:
              Q.put_nowait((conn, req))
            except queue.Full:
              slog(op="engine_queue_full", dropped=req.get("method"), q=Q.qsize())
              try:
                send(conn, {"id": req.get("id"), "ok": False,
                            "error": f"engine queue full ({MAX_Q_DEPTH}); retry shortly"})
              except Exception: pass
    except Exception: pass
    finally:
      with _conn_g: _conn_count["n"] -= 1
      _mark_dead(conn)

  def listener():
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if os.path.exists(SOCK): os.unlink(SOCK)
    srv.bind(SOCK)
    os.chmod(SOCK, 0o600)
    srv.listen(16)
    while True:
      try:
        conn, _ = srv.accept()
        threading.Thread(target=listener_conn, args=(conn,), daemon=True).start()
      except Exception as e:
        log("listener:", e); time.sleep(0.5)

  def heartbeat():
    n = 0
    while True:
      time.sleep(2)
      with SD._LOGL:
        line = json.dumps({"ts": round(time.time(), 3), "hb": n, "tl": tl(), "busy": ST.busy})
        for p in (MOE_LOGF, MOE_LOGF_PERSIST):
          try:
            with open(p, "a") as f: f.write(line + "\n")
          except Exception: pass
      n += 1
      if n % 300 == 0:
        _log_rotate((MOE_LOGF, MOE_LOGF_PERSIST), 20 * 1024 * 1024)

  def watchdog():
    iv = min(5.0, max(0.05, STEP_TIMEOUT_S / 10.0))
    while True:
      time.sleep(iv)
      try:
        now = time.time()
        if ST.busy and (now - ST.step_beat) > STEP_TIMEOUT_S:
          ST.dirty = True
          slog(op="watchdog", stage="step_timeout", rpc=ST.rpc,
               overdue_s=round(now - ST.step_beat, 1), timeout_s=STEP_TIMEOUT_S)
          _clean_exit(1, "watchdog_step_timeout", flush=True)
        if not ST.busy and (now - ST.last_keepalive_ok) > 3 * KEEPALIVE_S:
          slog(op="watchdog", stage="keepalive_stale",
               stale_s=round(now - ST.last_keepalive_ok, 1))
          _clean_exit(1, "keepalive_stale", flush=True)
      except Exception as e:
        slog(op="watchdog_error", error=repr(e))

  # ---- boot: watched probe, then serve ----
  threading.Thread(target=watchdog, daemon=True).start()
  threading.Thread(target=heartbeat, daemon=True).start()
  ST.busy = True; ST.rpc = "boot"; _beat()
  try:
    keepalive()
    _beat()
    eng.rig.reset_states(2048)
  finally:
    ST.busy = False; ST.rpc = None
  ST.ready = True
  ST.fed = []; ST.pos_cache = 0; ST.cur_cache = 0; ST.mode = None
  ST.mtp_chain = None; feed_hidden[0] = None
  slog(op="boot", stage="daemon_attached", ctxk=CTXK, dec_s=DEC_S, pf_s=PF_S,
       spec=SPEC_ON, mtp=bool(getattr(eng, "mtp_on", False)), pc=bool(ST.pc),
       fences=eng.rig.fence_count)
  log(f"[serve_moe] daemon attached; health probe ok; listening on {SOCK}")

  threading.Thread(target=listener, daemon=True).start()
  nka = 0
  while True:
    try:
      conn, req = Q.get(timeout=KEEPALIVE_S)
    except queue.Empty:
      try:
        keepalive(); nka += 1
        if nka % 50 == 1: slog(op="keepalive", n=nka, tl=tl())
      except Exception as e:
        slog(op="keepalive_error", error=repr(e), tl=tl())
        if "Device fault" in repr(e):
          slog(op="device_fault", stage="exiting_from_keepalive")
          _clean_exit(1, "device_fault_keepalive", flush=True)
        time.sleep(1)
      continue
    nka = 0
    handle(conn, req)
