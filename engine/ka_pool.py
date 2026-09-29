"""TLX P10 — the kernargs-slab pool (fixes P9 findings F2/F3).

ROOT CAUSE (one paragraph): every ParityGraph / PfGraph build carves a fresh
host-mapped kernargs slab through the dext's MAP_SYSMEM_FD RPC; the tinygpu
server (installer/Shared/server.c) caps live sysmem mappings at MAX_SYSMEM=128
and has NO unmap RPC — g_sysmem[] holds its own mmap + shm fd until process
death, so every slab alloc burns a server slot for the daemon's whole life.
The dense serving stack rebuilds graphs at every fence
(serve.py gen_rebuild / l7_entry_rebuild -> mtp.build_graphs() = ~6-8 fresh
ParityGraphs) and on every prefill graph-class churn (pf_prefill._pf_graphs:
new PfGraph sets on key change / PG_REBUILD / ATTN_THR twins), and the RETIRED
graphs' slabs were simply dropped — never returned. Long-prompt + sustained
dense traffic exhausts the pool: MAP_SYSMEM_FD returns no fd -> IndexError at
tinygrad/runtime/support/system.py:383 (F3, the soft face); under sustained
dense decode the exhaustion correlates with hard process-tree deaths and
machine resets (F2). The MoE path never trips this because its fence
(MG.rebuild) re-fills kernargs into the SAME slab — zero new mappings.

THE FIX (pool, because server slots cannot be returned): retirement — always
at a quiescent point; every rebuild site already dev.synchronize()s first —
zero-fills the slab and pushes it on a free list; a new build pops the
smallest slab that fits before mapping fresh. Steady state therefore NEVER
calls MAP_SYSMEM_FD again (bounded by the live-set + pool caps).

DETERMINISM CONTRACT: a fresh shm mapping is zero-filled; a pooled slab is
zeroed on release, so at hand-out it is byte-identical to a fresh mapping.
The GPU-visible bytes are then exactly what the new build's fill_kernargs
writes plus the per-submit QMD patches — the same bytes a fresh slab would
carry. Fresh NVComputeQueue objects are still built per rebuild (the
~950-cycle dext budget reset keeps its "fresh queue objects + kernargs"
character); only the underlying mapping recycles. The closest proven
precedent for slab-rewrite-after-traffic at a quiescent point is the MoE
MG.rebuild fence (P10 soak: 37 fences clean). The P9 KAPool SLAB COHERENCE
law (late CARVES from a traffic'd shared arena wedge) does not apply: we
recycle whole, previously-good slabs with a full rewrite, not fresh carves.

Env gates (project convention; kill switch restores EXACT legacy behavior):
  TLX_KA_POOL=1     default ON. 0 = fresh alloc every build, never pooled
                    (the documented leak — for A/B and rollback).
  TLX_KA_POOL_MAX=32  pool slab-count cap.
  TLX_KA_POOL_MB=128  pool total-bytes cap.
  TLX_KA_ZERO=1     default ON: zero-fill on release (fresh-mapping
                    equivalence). 0 skips zeroing (diagnostic only).
Overflow releases (pool full / zero failure) go through dev.allocator.free —
the host munmap reclaims VA space; the server slot is still burnt, but the
burn is bounded by the process high-water mark, which the pool pins.
"""
import os, bisect, ctypes, threading

_POOL_ON  = os.getenv("TLX_KA_POOL", "1") == "1"
_POOL_MAX = int(os.getenv("TLX_KA_POOL_MAX", "32"))
_POOL_MB  = int(os.getenv("TLX_KA_POOL_MB", "128"))
_ZERO     = os.getenv("TLX_KA_ZERO", "1") == "1"

_lock = threading.Lock()
_free: list = []          # sorted list of (size, id(buf), buf) — ascending size
_stats = dict(fresh=0, reused=0, released=0, freed_overflow=0, zero_fail=0,
              pool_bytes=0, live=0)

def _bspec():
  """The cpu_access/nolru BufferSpec (tinygrad-loaded contexts only; None in
  GPU-free unit tests)."""
  global _BS
  if _BS is None:
    try:
      from tinygrad.device import BufferSpec
      _BS = BufferSpec
    except Exception:
      _BS = False
  return _BS(cpu_access=True, nolru=True) if _BS else None

_BS = None

def ka_alloc(nbytes: int, dev):
  """Acquire an exclusive kernargs slab of at least `nbytes` bytes.
  Best-fit pool pop first, else a fresh allocator mapping (the legacy path)."""
  with _lock:
    if _POOL_ON:
      i = bisect.bisect_left(_free, (nbytes, -1))
      if i < len(_free):
        _sz, _id, buf = _free.pop(i)
        try: del buf._ka_pooled          # re-arm the idempotence marker
        except AttributeError: pass
        _stats["pool_bytes"] -= _sz
        _stats["reused"] += 1; _stats["live"] += 1
        return buf
    _stats["fresh"] += 1; _stats["live"] += 1
  return dev.allocator.alloc(nbytes, _bspec())

def ka_release(buf, dev, sync: bool = True) -> bool:
  """Retire a slab back to the pool. MUST only be called when the GPU is no
  longer reading it — every call site is a graph-teardown at a quiescent
  point; `sync=True` re-asserts dev.synchronize() defensively (the V-56
  close() discipline). Zero-fills the slab (fresh-mapping equivalence), then
  pools it under the caps or frees it. Idempotent per buffer."""
  if buf is None: return False
  if getattr(buf, "_ka_pooled", False): return False
  try:
    if sync: dev.synchronize()
  except Exception:
    pass  # teardown sites are already quiescent; never fail the serving path here
  ok_zero = True
  if _ZERO and buf.view is not None:
    try: ctypes.memset(buf.view.addr, 0, buf.size)
    except Exception: ok_zero = False
  with _lock:
    _stats["live"] -= 1; _stats["released"] += 1
    if not ok_zero: _stats["zero_fail"] += 1
    if (_POOL_ON and ok_zero and len(_free) < _POOL_MAX
        and (_stats["pool_bytes"] + buf.size) <= (_POOL_MB << 20)):
      buf._ka_pooled = True          # idempotence marker
      bisect.insort(_free, (buf.size, id(buf), buf))
      _stats["pool_bytes"] += buf.size
      return True
  # pool full (or zero failed): real free — host munmap reclaims VA; the
  # server-side slot stays burnt (no unmap RPC exists), bounded by high-water.
  try:
    dev.allocator.free(buf, buf.size, _bspec())
  except Exception:
    pass
  with _lock: _stats["freed_overflow"] += 1
  return False

def ka_stats() -> dict:
  """Live counters for /health + slog: pool traffic, and the TOTAL dext
  MAP_SYSMEM_FD count (the true server-slot burn — never decreases within a
  process life; with the pool on it must go FLAT after warm-up)."""
  with _lock:
    st = dict(_stats); st["pooled"] = len(_free); st["pool_cap"] = _POOL_MAX
    st["pool_on"] = _POOL_ON
  try:
    import tinygrad.runtime.support.system as _s
    st["mapfd_total"] = int(getattr(_s, "_MAPFD_TOTAL", 0))
  except Exception:
    st["mapfd_total"] = -1
  return st
