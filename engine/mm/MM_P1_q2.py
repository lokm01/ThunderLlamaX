#!/usr/bin/env python3
"""MM P1 follow-up — NaN-aware determinism re-verification of the graph replays.

Run-1 (MM_P1_q.py) verdicts: 256/1024 replays eids-EXACT vs the numpy sim at
every checkpoint (the mutating chain IS deterministic); the ys det check used
plain np.array_equal, which FALSE-ALARMS on NaN rows (P0's random banks produce
NaN outputs; the proven compare is NaN-agreement). This rerun redoes T4/T5 det
with NaN-agreement semantics + records the finite-frac.
Run on a FRESH boot (one GPU process per boot).
"""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
sys.path.insert(0, "~/tinygrad-metal")
BASE = "~/tinygrad-metal"
SLAB, GATE_B, SHARD, NSHARD, NPAIR = 1458176, 450560, 256, 20, 88
NBANK = SHARD * NSHARD
PROG = os.path.expanduser("~/mm_p1_q2_progress.txt")

def record(tag, result):
    with open(PROG, "a") as f: f.write(f"{tag} {result}\n")
    print(f"[PROGRESS] {tag} {result}", flush=True)

def done(tag):
    if not os.path.exists(PROG): return False
    return any(l.split()[0] == tag for l in open(PROG).read().splitlines() if l.split())

def naneq(a, b):
    return np.array_equal(a, b) or ((a == b) | (np.isnan(a) & np.isnan(b))).all()

def main():
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram, NVComputeQueue, nv_wait_timeline
    from tinygrad.helpers import round_up
    from tinygrad.uop.ops import UOp
    from tinygrad.dtype import dtypes

    SYM = {"MM_P0_mut": "mm_eidmut", "MM_P0_hrot": "mm_hrot",
           "MM_P0_rt8e256poc": "rt8e256poc", "MM_P0_gx8e256nw32": "gx8e256nw32"}
    def prog(n):
        lib = open(f"{BASE}/{n}.cubin", "rb").read()
        return NVProgram(dev, TinyELF(lib=lib, name=SYM[n], target=dev.renderer.target, signature=tuple()))
    gx = prog("MM_P0_gx8e256nw32")
    mut, rt, hrot = prog("MM_P0_mut"), prog("MM_P0_rt8e256poc"), prog("MM_P0_hrot")
    dev.synchronize(); print("[S0] 4 programs loaded", flush=True)

    keep = []
    def up(a):
        a = np.ascontiguousarray(a); keep.append(a)
        b = dev.allocator.alloc(a.nbytes, BufferSpec())
        dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
    def dn(b, shape, dtype=np.float32):
        n = int(np.prod(shape))
        mv = memoryview(bytearray(int(n)*np.dtype(dtype).itemsize)).cast("B")
        dev.allocator._copyout(mv, b)
        return np.frombuffer(mv, dtype=dtype).reshape(shape).copy()

    from MM_P1_q import iq3s_grid_f32
    gridf = up(iq3s_grid_f32())
    xs_np = (np.random.default_rng(11).uniform(-0.5, 0.5, (NPAIR, 2048))).astype(np.float32)
    xs = up(xs_np)
    seed = (np.arange(88, dtype=np.uint16) % 8)
    eids_g = up(seed)
    ys_g = dev.allocator.alloc(88*512*4, BufferSpec())
    W_np = (np.random.default_rng(5).uniform(-0.05, 0.05, (256, 2048))).astype(np.float32)
    Wb = up(W_np)
    h0 = (np.random.default_rng(6).uniform(-0.5, 0.5, (88, 2048))).astype(np.float32)
    hb = up(h0)
    xs704 = up(np.tile(xs_np, (8, 1)))
    ys704 = dev.allocator.alloc(704*512*4, BufferSpec())
    rng = np.random.default_rng(7)
    banks = [dev.allocator.alloc(SLAB*SHARD, BufferSpec()) for _ in range(NSHARD)]
    CH = 64 << 20
    for bank in banks:
        for off in range(0, SLAB*SHARD, CH):
            n = min(CH, SLAB*SHARD - off)
            a = rng.integers(0, 256, n, dtype=np.uint8); keep.append(a)
            dev.allocator._copyin(bank.offset(offset=off, size=n), memoryview(a.data).cast("B"))
    ptbl = up(np.array([banks[e//SHARD].va_addr + (e % SHARD)*SLAB for e in range(NBANK)], dtype=np.uint64))
    dev.synchronize(); print("[S0] buffers up", flush=True)

    LS = {"gx8e256nw32": (1024,1,1), "rt8e256poc": (1024,1,1),
          "mm_eidmut": (128,1,1), "mm_hrot": (128,1,1)}
    class MG:
        def __init__(self, seq, tag):
            self.prev = UOp.variable(f"{tag}_p", 0, 0xffffffff, dtype=dtypes.uint32)
            self.cur  = UOp.variable(f"{tag}_c", 0, 0xffffffff, dtype=dtypes.uint32)
            per = max(round_up(p.kernargs_alloc_size, 8) for p, a, g in seq)
            self.ka = dev.allocator.alloc(per*len(seq), BufferSpec(cpu_access=True, nolru=True))
            keep.append(self.ka)
            q = NVComputeQueue(); q.memory_barrier()
            q.wait(dev.timeline_signal, self.prev)
            off = 0
            for p, bufs, grid in seq:
                ab = self.ka.offset(offset=off, size=p.kernargs_alloc_size)
                st = p.fill_kernargs(tuple(bufs), (), kernargs=ab)
                q.exec(p, st, (grid,1,1), LS[p.name])
                off += round_up(p.kernargs_alloc_size, 8)
            q.signal(dev.timeline_signal, self.cur)
            self.q = q
        def submit(self, pv, cv):
            self.q.submit(dev, {self.prev.expr: int(pv), self.cur.expr: int(cv)})
    def replay_wait_each(g0, g1, n, tag):
        # THE LAW (T4 run-1): wait EVERY graph (SKEDCHECK22 / P7F1); no pipelining.
        prev = dev.timeline_value - 1
        for i in range(n):
            v = dev.next_timeline()
            (g0 if (i & 1) == 0 else g1).submit(prev, v)
            nv_wait_timeline(dev, v, what=tag, timeout_s=20.0)
            prev = v
        dev.synchronize()

    # ---- T4b: 256 replays det with NaN-agreement ----
    if not done("T4b"):
        g0 = MG([(mut, (eids_g,), 1), (gx, (ptbl, eids_g, xs, gridf, ys_g), 88)], "t4ba")
        g1 = MG([(mut, (eids_g,), 1), (gx, (ptbl, eids_g, xs, gridf, ys_g), 88)], "t4bb")
        replay_wait_each(g0, g1, 256, "T4b")
        got1 = dn(eids_g, (88,), np.uint16); ys1 = dn(ys_g, (88, 512))
        dev.allocator._copyin(eids_g, memoryview(seed.tobytes()))
        replay_wait_each(g0, g1, 256, "T4b")
        got2 = dn(eids_g, (88,), np.uint16); ys2 = dn(ys_g, (88, 512))
        ff = float(np.isfinite(ys1).mean())
        record("T4b", f"256-replays det(NaN-aware): eids={np.array_equal(got1,got2)} ys={naneq(ys1,ys2)} "
                      f"finite-frac={ff:.3f}")

    # ---- T5b: chain det with NaN-aware ----
    if not done("T5b"):
        big704 = np.concatenate([np.random.default_rng(300+i).choice(256, 8, replace=False) for i in range(88)]).astype(np.uint16)
        eids704 = up(big704)
        c0 = MG([(hrot, (hb,), 1), (rt, (Wb, hb, eids704), 88), (gx, (ptbl, eids704, xs704, gridf, ys704), 704)], "t5ba")
        c1 = MG([(hrot, (hb,), 1), (rt, (Wb, hb, eids704), 88), (gx, (ptbl, eids704, xs704, gridf, ys704), 704)], "t5bb")
        replay_wait_each(c0, c1, 256, "T5b")
        h1 = dn(hb, (88, 2048)); got1 = dn(eids704, (88, 8), np.uint16); ys1 = dn(ys704, (704, 512))
        dev.allocator._copyin(hb, memoryview(h0.tobytes()))
        replay_wait_each(c0, c1, 256, "T5b")
        h2 = dn(hb, (88, 2048)); got2 = dn(eids704, (88, 8), np.uint16); ys2 = dn(ys704, (704, 512))
        ff = float(np.isfinite(ys1).mean())
        record("T5b", f"chain256 det(NaN-aware): h={np.array_equal(h1,h2)} eids={np.array_equal(got1,got2)} "
                      f"ys={naneq(ys1,ys2)} finite-frac={ff:.3f}")
    print("[ALL DONE]", flush=True)

if __name__ == "__main__":
    import threading
    def _wd():
        time.sleep(900); print("[WATCHDOG] deadline", flush=True); os._exit(4)
    threading.Thread(target=_wd, daemon=True).start()
    main()
