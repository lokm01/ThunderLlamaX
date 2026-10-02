"""MM SESSION A shared graph plumbing -- THE UNCACHED-KA LAW.

Late-built graphs (built AFTER GPU traffic in the process) must NOT use the
KAPool cacheable slab: host kernargs writes race GPU reads (the MM_P9_mtp
MGUnc law -- a pool-ka graph built after replays faulted live in session A's
first attempt). Every session-A graph goes through the dedicated uncached ka
(sysmem, GPU_CACHEABLE_NO -- bidirectionally coherent, the cpu-fold control
class). Proven in production: the daemon's gr5 (P5 probe) runs GraphRunnerUnc
interleaved with the pool-based trunk runners.
"""
import os, sys
BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")

from MM_P56_lib import MG, GraphRunner

class MGUnc(MG):
    def __init__(self, rig, seq, tag):
        from tinygrad.device import BufferSpec
        dev = rig.dev
        self.rig = rig; self.tag = tag
        MG._seq += 1; self.uid = MG._seq
        self.prev = rig.UOp.variable(f"{tag}_p{self.uid}", 0, 0xffffffff, dtype=rig.dtypes.uint32)
        self.cur  = rig.UOp.variable(f"{tag}_c{self.uid}", 0, 0xffffffff, dtype=rig.dtypes.uint32)
        per = max(rig.round_up(p.kernargs_alloc_size, 8) for p, a, g, v in seq)
        self.kb = per * len(seq)
        self.ka = dev.allocator.alloc(self.kb + 8, BufferSpec(cpu_access=True, nolru=True, uncached=True))
        rig.keep.append(self.ka)
        self.seq = seq
        self._build()

class GraphRunnerUnc(GraphRunner):
    def __init__(self, rig, seq, tag, fence_every=1024):
        self.rig = rig
        self.ga = MGUnc(rig, seq, tag + "a")
        self.gb = MGUnc(rig, seq, tag + "b")
        self.turn = 0
        self.fence_every = fence_every
        self.n = 0

def mkgraph_unc(rig, seq, tag, fence_every=1024):
    """The uncached-ka mkgraph twin (all session-A graphs use this)."""
    return GraphRunnerUnc(rig, [(rig.K[n], b, g, v) for n, b, g, v in seq], tag,
                           fence_every=fence_every)
