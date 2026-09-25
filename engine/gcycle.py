# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W1-c G_CYCLE: capture the ENTIRE per-token program (embed -> 64 blocks -> head
-> argmax, 452 kernels) as ONE bound NVComputeQueue per conv-parity = ONE gpfifo
entry + doorbell per token. Kernels chain via QMD dependent pointers inside the
queue (the active_qmd path in ops_nv exec) -- no host work between kernels.
All data flow is device-resident (argmax -> tok_slot -> next embed; pos_slot).
Kernargs: one host-mapped buffer per graph (static args written once at build).
Timeline: graph starts with wait(timeline, v-1) and ends signal(timeline, v);
the two values are var-patched into the bound hw_page on each submit."""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from tinygrad.helpers import round_up
from tinygrad.device import BufferSpec
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.runtime.ops_nv import NVComputeQueue, nv_wait_timeline
from engine0 import dev

class ParityGraph:
  def __init__(self, seq, tag="g"):
    self.seq = seq
    self.dev = dev
    self.prev_var = UOp.variable(f"{tag}_tl_prev", 0, 0xffffffff, dtype=dtypes.uint32)
    self.cur_var = UOp.variable(f"{tag}_tl_cur", 0, 0xffffffff, dtype=dtypes.uint32)
    # one host-mapped kernargs slab for the whole graph
    per = max(round_up(p.kernargs_alloc_size, 8) for p, a, g in seq)
    self.ka = dev.allocator.alloc(per * len(seq), BufferSpec(cpu_access=True, nolru=True))
    self.ka_keep = self.ka  # hold refs (Bufs._keep discipline)
    q = NVComputeQueue()
    q.memory_barrier()
    q.wait(dev.timeline_signal, self.prev_var)
    off = 0
    for p, bufs, grid in seq:
      g0 = grid[0] if isinstance(grid, (tuple, list)) else grid
      argsbuf = self.ka.offset(offset=off, size=p.kernargs_alloc_size)
      st = p.fill_kernargs(tuple(bufs), (), kernargs=argsbuf)
      nm = getattr(p, "name", "")
      ls = ((1024,1,1) if "nw32" in nm else (768,1,1) if "nw24" in nm else (512,1,1) if "nw16" in nm else (256,1,1))
      q.exec(p, st, (g0, 1, 1), ls)
      off += round_up(p.kernargs_alloc_size, 8)
    q.signal(dev.timeline_signal, self.cur_var)
    self.q = q
    # M1-C forensics: ring bytes per submit of this graph (cmdq ring budget math)
    self.ring_bytes = len(q._q) * 4
    print(f"[gcycle] {tag}: kernels={len(seq)} _q_words={len(q._q)} ring_bytes={self.ring_bytes}", flush=True)
    # NOTE: no bind() -- the shared-cmdq-ring submit path is the proven one on this
    # dext (MTP_GRAPH_NOBIND convention). Per-submit cost is a tiny _q copy; the
    # 452 kernels chain inside QMDs, so _q is only a few dozen words.

  def submit(self, prev_v, cur_v):
    self.q.submit(self.dev, {self.prev_var.expr: int(prev_v), self.cur_var.expr: int(cur_v)})

class GCycleEngine:
  """Wraps TrunkEngineW1C: graph-replayed decode loop."""
  def __init__(self, E):
    self.E = E
    self.graphs = {}
    self._kick_q = None   # W4.2 kicker: a signal-only queue (private signal,
    # never the timeline — an extra timeline value would pass later waits early)

  def build(self):
    if not hasattr(self.E, "_seq"): self.E._build_seqs()
    self.graphs = {par: ParityGraph(self.E._seq[par], tag=f"w1c{par}") for par in (0, 1)}

  def _kicker(self):
    """W4.2 (V-59, the LONE-GRAPH law): a LONE-submitted graph never completes
    (every graph needs a follow-up submit in flight; proven W2). Signal-only
    queue submits are the proven boot pattern (_setup_gpfifos)."""
    if self._kick_q is None:
      sig = dev.new_signal()
      q = NVComputeQueue()
      q.signal(sig, 1)
      self._kick_q = q
    self._kick_q.submit(dev)

  def run_tokens(self, n, wait_each=False, sync_every=1):
    """sync_every: submit this many graphs before waiting (pipelining depth).
    The dext faults with too many kernels in flight (W1-b gotcha 8: ~300); a full
    graph is 452 kernels, so pipelining >1 graph needs care -- bisect K if ambitious.
    W4.2: sync_every > 2 is FORBIDDEN — submit() var-patches the SHARED host
    kernargs slab per call, so at it=2 the host re-writes parity-0's slab while
    parity-0 may still be in flight (same-parity re-patch race, kimi F2 / V-48:
    early timeline release -> waits pass prematurely). Depth 2 is safe because
    the it=2 submit happens only after the it=1 wait released parity-0."""
    assert self.graphs, "call build() first"
    assert isinstance(sync_every, int) and 1 <= sync_every <= 2, \
      f"sync_every={sync_every} violates the same-parity kernargs re-patch race law (V-48; max 2)"
    # W4.2 (V-58): the shared cmdq ring wraps SILENTLY — if the submits this
    # call can enqueue outrun one ring of GPU-unfetched pushbuffer, the wrap
    # overwrites pending commands (device-fault class). Budget-check at entry.
    max_ring = max(g.ring_bytes for g in self.graphs.values())
    assert (max_ring * sync_every) < dev.cmdq_page.size, \
      f"cmdq ring budget: {max_ring}B x {sync_every} >= ring {dev.cmdq_page.size}B (MTP_CMDQ_MB too small)"
    prev_val = dev.timeline_value - 1
    last_v = prev_val
    pending = 0
    for it in range(n):
      v = dev.next_timeline()
      self.graphs[it & 1].submit(prev_val, v)
      prev_val, last_v = v, v
      pending += 1
      if wait_each or ((it + 1) % sync_every == 0):
        if pending < 2: self._kicker()   # lone-graph law
        nv_wait_timeline(dev, v, what=f"gcycle.run_tokens(it={it})")
        pending = 0
    if n % sync_every != 0:
      if pending < 2: self._kicker()
      nv_wait_timeline(dev, last_v, what="gcycle.run_tokens(trailing)")
    dev.synchronize()
