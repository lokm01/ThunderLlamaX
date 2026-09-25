# W0-E2: The Graph-Budget Wall — cmdq-Ring Hypothesis Test (2026-09-13)

HYPOTHESIS (W0_DOSSIER.md §4, ranked #1): the deterministic ~50-blocks-worth
piece-graph capture fault at 100k shapes is a silent wrap-clobber of the shared
2MB cmdq ring — capture enqueues >2MB of command stream ahead of GPU execution,
the wrap allocator overwrites unfetched pushbuffer, gpfifo entries then point at
garbage -> device fault.

## VERDICT: REFUTED (direct measurement). The ring never wraps before the fault.

### 1. Code analysis (fork; pre-patch baseline b8fda0b)
- ops_nv.py:627-629 (pre-patch): `self.cmdq_page = self.iface.alloc(0x200000, cpu_access=True)`;
  `BumpAllocator(..., wrap=True)`. Ring = 2MiB host sysmem, shared by compute+copy queues.
- ops_nv.py:114-125 `_submit_to_gpfifo`: EVERY unbound submit (MTP_GRAPH_NOBIND=1 -> all
  graph AND eager submits) allocates `len(self._q)*4` bytes at 16B alignment from the ring
  and adds ONE gpfifo entry `(addr//4)<<2 | len<<42 | 1<<41`. Bound submits (bind path) use
  a per-graph hw_page and never touch the ring.
- support/memory.py:5-12 BumpAllocator: on `round_up(ptr,align)+size > ring` with wrap=True
  it resets ptr=0 SILENTLY -> the next alloc overwrites the OLDEST ring bytes, which are
  exactly the most-likely-unfetched commands if the GPU lags. Mechanism is real.
- Volume estimate: eager single-kernel submits ~57B each (measured); piece-graph submits
  < 1KiB (max_single observed <1024B — per-piece graphs are small, a few kernels each).
  gpfifo entry count: 19,125 total submits << 65,536 entries — also refuted.

### 2. Instrumentation (fork commit e8e594d, env-gated, default bit-identical)
- MTP_CMDQ_MB=<MiB> (default 2): sizes the shared ring.
- MTP_CMDQ_DIAG=1: counts submits/bytes, detects + logs every wrap (new ring offset <
  pre-alloc offset), logs volume every 4096 submits, dumps state at on_device_hang.
- Sanity: 20k eager submits on a 16MiB ring — clean, ~57B/submit, monotonic ring offset.

### 3. Control run (fault must reproduce today): ~/run_w0_ctrl.sh
= run_100k_split13.sh (the deterministic 100k split repro: SKV + SCAN_SPLIT + merged
families + FRAGDBG + SYNCMID, ckpt resume) + MTP_CMDQ_DIAG=1, default 2MB ring.
Log: ~/mtp_w0_ctrl.log. RESULT: fault reproduced, EXIT=1, ~4.5 min wall:
- early-caps all CLEAN (draft 1.9s/0.2s, probe 17.3s/7.6s, commit 11.8s/6.1s, select_final 0.2s)
- then `mtp_v3.py:926 -> fwd3_split.py:198 -> merged_j:132 -> _bufs_for:60` — RuntimeError
  "Device fault detected" (err state caught at a later alloc->synchronize; same signature
  as split13/12/10).
- *** [CMDQ] HANG: wraps=0 total=1359KiB submits=19125 since_last_wrap=1359KiB max_single<1KiB ***
- Ring-offset trace monotonic 383->762->1060->1341 KiB — the bump pointer NEVER reset.
  Cumulative command bytes for the WHOLE PROCESS = 1359KiB < 2048KiB ring: the ring was
  ~2/3 virgin at fault. No overwrite ever occurred => the ring cannot be the corruption
  vector. (=> MTP_CMDQ_MB=16/64 rerun is moot — enlarging cannot change a never-wrapped
  ring; not run.)

### 4. Fallback discrimination (task step 5): pages-vs-handles
Probe: ~/tinygrad-metal/w0_budget_probe.py — N distinct TinyJit graph families (distinct
shapes => distinct HCQGraphs), trivial add kernel over one stable buffer per family,
3 capture passes + 2 replays each, everything kept alive (mirrors capture-held
graph_cache). NOBIND=1, BEAM=0, bare process (no model).
- TINY (0.25MiB x 150 families, ~300+ piece-graphs + eager kernels): ALL 150 FAMILIES
  CLEAN in 47s. => No per-graph-object / userspace-handle budget at this scale
  (graph_cache weakref, signals pool, gpfifo entries, kernargs pool all fine).
- BIG (64MiB x 150) and MID (8MiB x 150): HOST PANIC — the whole Mac went down (watchdog
  reboot, connection reset, /tmp wiped) within ~a minute, before ANY family line could
  flush. Two reboots total. => Large per-graph buffers move the failure EARLIER and into
  a DIFFERENT, harder class (host kernel panic — consistent with the documented DART/
  large-mapping hazard on this rig), NOT the production wall's catchable device fault.
  Pages move *something* but this arm is confounded by the known host-panic class; it
  does NOT reproduce the split fault mode.
- CAUTION for future probes: per-family buffers >= 8MiB in bare capture loops can panic
  the HOST (2 crashes today, ~16:05 and ~16:22). Reboot recovery is automatic (~10 min).

### 5. So what IS the budget? (narrowed by elimination)
Excluded by measurement: cmdq ring wrap (this doc), gpfifo entry count (19k<<65k),
kernargs pool (wrap=False -> loud RuntimeError, wrong signature), per-graph object count
(150 trivial families clean), MAP_SYSMEM_FD cap (43 flat, prior work), graph_cache
(weakref, unbounded), allocator age (exonerated in K4BEAM).
Remaining: the wall fires only under REAL model captures at 100k shapes (~50 blocks-worth,
merge-invariant, 2k-clean with the same family structure, host SURVIVES). With every
userspace cap now excluded, the budget lives GPU/GSP/dext-side, coupled to real capture
content — dependent-QMD chains / per-piece semaphore release volume / dext pushbuffer or
GSP per-channel queue accounting (dossier hypothesis #2). That is opaque from source.
=> PRIMARY ROUTE: the fork-maintainer question (is there a tunable GSP/dext-side
per-channel queue/graph budget?). PRAGMATIC ROUTES unchanged: piece reduction (drop
stable-buffer assigns) or split-at-2k-only (canonical 100k path = the 7.28 stock).

### 6. Recipes / env guidance
- NO canonical recipe should adopt MTP_CMDQ_MB — the fix hypothesis is refuted; the
  default 2MiB ring is sufficient (process lifetime peak = 1359KiB even in the faulting
  capture; steady decode cycles sync every cycle).
- Keep MTP_CMDQ_DIAG=1 available for future enqueue-lag forensics (e.g. if a route ever
  removes per-cycle syncs, watch for wraps).
- Fork commit: e8e594d (instrumentation + knob, default-identical).

Artifacts: ~/run_w0_ctrl.sh, ~/run_w0_cmdq16.sh (unused), ~/mtp_w0_ctrl.log,
~/tinygrad-metal/w0_budget_probe.py. Test env per AGENTS.md (PATH/DOCKER_HOST prefix,
~/tg311/bin/python, DEV=NV).
