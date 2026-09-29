#!/usr/bin/env python3
"""MM P1 ITEM #1 — root-cause the D4 queue-replay fault.

CODE ARCHAEOLOGY (why this might already be solved):
  The wedge was ONLY ever observed in MM_P0_d3_gather.py, whose programs were
  loaded as TinyELF(name="MM_P0_gx8e256nw32") -- the FILE prefix, NOT the cubin
  symbol ("gx8e256nw32"). That is a direct NAME-LAW violation: the per-kernel
  .nv.info/.text/.nv.shared sections don't match -> prog_addr = whole-ELF base
  (executes header bytes = SM Illegal Instruction) -> the graph's QMD faults ->
  its release never fires -> the host timeline wait WEDGES -> device fault.
  Eager launches faulted in that era too (the 15-boot cascade). d3_final.py
  (name-law fixed) NEVER ran the queue path -- S3b silently went eager-batched.
  SECOND confound: d3_gather's run_block submits the SAME NVComputeQueue object
  twice in flight (parity-less) -- the same-parity kernargs re-patch race
  (V-48) gcycle forbids by construction.

Stages (one GPU process per boot; progress file survives reboots):
  T1 : gx-only graph, LONE submit, 15s wait  (name-law + lone-graph test)
  T1K: kicker arm (only if T1 wedged) -- gcycle signal-only flusher queue
  T2 : [mut->gx] graph, LONE submit          (chained-exec + QMD-release signal)
  T2K: kicker arm (only if T2 wedged)
  T3 : same-object double-submit x4 (the d3_gather run_block pattern) -- document
       the re-patch race behavior with CORRECT names (wedge/early-pass/clean)
  T4 : parity-pair 256 replays [mut->gx] + eids vs numpy affine sim + det x2
       + per-cycle timing (88-pair in-graph GB/s)
  T5 : parity-pair chain [hrot->rt->gx704] 256 replays x2 + rt exactness at
       checkpoints + det + 704-pair in-graph GB/s  (THE P0-unmeasurable number)
  T7 : endurance 1024 replays of T4 graph (the ~950-cycle budget probe)
On wedge: record + exit(3) (GPU-EXIT reboot); next boot resumes after it.
"""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("MM_SKIP_BUILD", "1")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
sys.path.insert(0, "~/tinygrad-metal")
BASE = "~/tinygrad-metal"
SLAB, GATE_B, SHARD, NSHARD, NPAIR = 1458176, 450560, 256, 20, 88
NBANK = SHARD * NSHARD
PROG = os.path.expanduser("~/mm_p1_q_progress.txt")

def record(tag, result):
    with open(PROG, "a") as f: f.write(f"{tag} {result}\n")
    print(f"[PROGRESS] {tag} {result}", flush=True)

def done(tag):
    if not os.path.exists(PROG): return False
    return any(l.split()[0] == tag for l in open(PROG).read().splitlines() if l.split())

from MM_P0_d2_repack import dq_iq3_s

def iq3s_grid_f32():
    from tinygrad.runtime.autogen.ggml_common import iq3s_grid
    v = np.array([(int(w) >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.uint8)
    assert v.size == 2048
    return v.view(np.int8).astype(np.float32).reshape(512, 4).copy()

def gx_ref(rows, x, gridf):
    W = dq_iq3_s(np.ascontiguousarray(rows), 2048)
    partial = np.zeros((512, 32), dtype=np.float32)
    Wb = W.reshape(512, 8, 32, 8)
    xk = x.reshape(8, 32, 8)
    prod = (Wb * xk[None, :, :, :]).astype(np.float32)
    for b in range(8):
        for j in range(8):
            partial = partial + prod[:, b, :, j]
    p = partial
    for o in (16, 8, 4, 2, 1):
        p = p + p[:, np.arange(32) ^ o]
    return np.ascontiguousarray(p[:, 0])

def rt_ref(W, h, positions):
    out = {}
    for p in positions:
        hr = h[p]
        lg = np.empty(256, dtype=np.float32)
        for e in range(256):
            parts = []
            for part in range(4):
                w = W[e, part*512:(part+1)*512]; hh = hr[part*512:(part+1)*512]
                s_ = np.float32(0)
                for i in range(0, 512, 4):
                    s_ = np.float32(s_ + np.float32(w[i]*hh[i]))
                    s_ = np.float32(s_ + np.float32(w[i+1]*hh[i+1]))
                    s_ = np.float32(s_ + np.float32(w[i+2]*hh[i+2]))
                    s_ = np.float32(s_ + np.float32(w[i+3]*hh[i+3]))
                parts.append(s_)
            lg[e] = np.float32(np.float32(np.float32(parts[0]+parts[1])+parts[2])+parts[3])
        ids = []
        for r in range(8):
            be = 0; bv = lg[0]
            for e2 in range(1, 256):
                if lg[e2] > bv: bv = lg[e2]; be = e2
            lg[be] = np.float32(-3.4e38); ids.append(be)
        out[p] = np.array(ids, dtype=np.uint16)
    return out

def main():
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram, NVComputeQueue, nv_wait_timeline
    from tinygrad.helpers import round_up
    from tinygrad.uop.ops import UOp
    from tinygrad.dtype import dtypes

    for n in ("MM_P0_gx8e256nw32", "MM_P0_mut", "MM_P0_rt8e256poc", "MM_P0_hrot"):
        assert os.path.exists(f"{BASE}/{n}.cubin"), f"missing {n}.cubin"
    # NAME LAW: TinyELF name == cubin kernel symbol (files keep the MM_P0_ prefix)
    SYM = {"MM_P0_mut": "mm_eidmut", "MM_P0_hrot": "mm_hrot",
           "MM_P0_rt8e256poc": "rt8e256poc", "MM_P0_gx8e256nw32": "gx8e256nw32"}
    def prog(n):
        lib = open(f"{BASE}/{n}.cubin", "rb").read()
        return NVProgram(dev, TinyELF(lib=lib, name=SYM[n], target=dev.renderer.target, signature=tuple()))
    gx = prog("MM_P0_gx8e256nw32")
    mut, rt, hrot = prog("MM_P0_mut"), prog("MM_P0_rt8e256poc"), prog("MM_P0_hrot")
    dev.synchronize(); print("[S0] 4 programs loaded (name-law compliant)", flush=True)

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

    # small buffers BEFORE banks (probe6 order law)
    gridf = up(iq3s_grid_f32())
    eids8 = up(np.arange(8, dtype=np.uint16))
    xs_np = (np.random.default_rng(11).uniform(-0.5, 0.5, (NPAIR, 2048))).astype(np.float32)
    xs = up(xs_np)
    ys = dev.allocator.alloc(NPAIR*512*4, BufferSpec())
    seed = (np.arange(88, dtype=np.uint16) % 8)
    eids_g = up(seed)
    ys_g = dev.allocator.alloc(88*512*4, BufferSpec())
    W_np = (np.random.default_rng(5).uniform(-0.05, 0.05, (256, 2048))).astype(np.float32)
    Wb = up(W_np)
    h0 = (np.random.default_rng(6).uniform(-0.5, 0.5, (88, 2048))).astype(np.float32)
    hb = up(h0)
    eids_c = up(np.zeros(704, dtype=np.uint16))
    xs704 = up(np.tile(xs_np, (8, 1)))
    ys704 = dev.allocator.alloc(704*512*4, BufferSpec())

    # 7.3GB bank: 20 x 373MB, pure-random fills, ptbl = absolute VAs (P0 laws)
    rng = np.random.default_rng(7)
    banks = [dev.allocator.alloc(SLAB*SHARD, BufferSpec()) for _ in range(NSHARD)]
    CH = 64 << 20
    for bank in banks:
        for off in range(0, SLAB*SHARD, CH):
            n = min(CH, SLAB*SHARD - off)
            a = rng.integers(0, 256, n, dtype=np.uint8); keep.append(a)
            dev.allocator._copyin(bank.offset(offset=off, size=n), memoryview(a.data).cast("B"))
    ptbl = up(np.array([banks[e//SHARD].va_addr + (e % SHARD)*SLAB for e in range(NBANK)], dtype=np.uint64))
    dev.synchronize(); print("[S0] buffers + 7.3GB bank up", flush=True)

    # eager sanity: the programs are clean in THIS boot (both kernels, waited)
    mut(eids_g, global_size=(1,1,1), local_size=(128,1,1), wait=True)
    gx(ptbl, eids_g, xs, gridf, ys_g, global_size=(88,1,1), local_size=(1024,1,1), wait=True)
    dev.synchronize(); print("[S0] eager sanity CLEAN (mut + gx waited)", flush=True)
    dev.allocator._copyin(eids_g, memoryview(seed.tobytes()))

    # ---- mini-graph builder: the gcycle ParityGraph pattern verbatim ----
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
            self.ring_bytes = len(q._q)*4
        def submit(self, pv, cv):
            self.q.submit(dev, {self.prev.expr: int(pv), self.cur.expr: int(cv)})
    _kick = {"q": None, "sig": None}
    def kicker():
        # gcycle W4.2 pattern: signal-only queue on a PRIVATE signal; the
        # signal() on a queue with active_qmd=None takes the NVM-semaphore +
        # NON_STALL_INTERRUPT path -- the eager path's proven completion class.
        if _kick["q"] is None:
            _kick["sig"] = dev.new_signal()
            q = NVComputeQueue(); q.signal(_kick["sig"], 1)
            _kick["q"] = q
        _kick["q"].submit(dev)
    def lone_wait(v, tag, to=15.0):
        try:
            nv_wait_timeline(dev, v, what=tag, timeout_s=to)
            return True
        except RuntimeError as e:
            print(f"[{tag}] WEDGE: {e!r}", flush=True)
            return False

    # ============================ T1 ============================
    if not done("T1") and not done("T1K"):
        g = MG([(gx, (ptbl, eids8, xs, gridf, ys), 8)], "t1")
        pv = dev.timeline_value - 1          # last SIGNALED value (capture BEFORE reserving v)
        v = dev.next_timeline()
        g.submit(pv, v)
        print("[T1] gx-only graph LONE submitted; waiting 5s...", flush=True)
        if lone_wait(v, "T1", to=5.0):
            record("T1", "OK-lone-clean")
        else:
            record("T1", "WEDGE-lone-qmd-release")
            print("[T1K] gcycle kicker (semaphore+interrupt flusher submit)...", flush=True)
            kicker()
            if lone_wait(v, "T1K", to=15.0):
                record("T1K", "OK-kicker-fixed")
                dev.synchronize()
                # reproducibility: a second lone submit + kicker
                pv2 = dev.timeline_value - 1
                v2 = dev.next_timeline()
                g.submit(pv2, v2)
                wedge2 = not lone_wait(v2, "T1c", to=5.0)
                kicker()
                ok2 = lone_wait(v2, "T1c", to=15.0)
                record("T1c", f"{'wedge-then-' if wedge2 else 'no-wedge-'}kicker={'OK' if ok2 else 'FAIL'}")
                if not ok2: sys.exit(3)
            else:
                record("T1K", "WEDGE-kicker-nohelp"); print("T1 wedge unfixable -> exit for reboot", flush=True); sys.exit(3)
        dev.synchronize()
    # ============================ T2 ============================
    if not done("T2") and not done("T2K"):
        g = MG([(mut, (eids_g,), 1), (gx, (ptbl, eids_g, xs, gridf, ys_g), 88)], "t2")
        pv = dev.timeline_value - 1          # last SIGNALED value (capture BEFORE reserving v)
        v = dev.next_timeline()
        g.submit(pv, v)
        print("[T2] [mut->gx] graph LONE submitted; waiting 15s...", flush=True)
        if lone_wait(v, "T2"):
            ev = seed.astype(np.int64); ev = (ev*7 + 3) % NBANK
            got = dn(eids_g, (88,), np.uint16)
            ok = np.array_equal(got.astype(np.int64), ev)
            record("T2", f"OK-lone-clean-eids{'EXACT' if ok else 'BAD'}")
        else:
            print("[T2K] kicker arm...", flush=True)
            kicker()
            if lone_wait(v, "T2K", to=15.0): record("T2K", "OK-kicker-fixed")
            else:
                record("T2K", "WEDGE-kicker-nohelp"); sys.exit(3)
        dev.synchronize()
        dev.allocator._copyin(eids_g, memoryview(seed.tobytes()))
    # ============================ T3 ============================
    if not done("T3"):
        # the d3_gather run_block pattern: SAME object submitted twice in flight.
        # V-48 predicts same-parity kernargs re-patch: early timeline release /
        # lost intermediate value -- possibly a WEDGE if the payload patch lands
        # mid-completion. Document the actual behavior.
        g = MG([(mut, (eids_g,), 1), (gx, (ptbl, eids_g, xs, gridf, ys_g), 88)], "t3")
        print("[T3] same-object double-submit x4 (the parity-less pattern)...", flush=True)
        prev = dev.timeline_value - 1
        ok = True
        try:
            for i in range(4):
                v = dev.next_timeline()
                g.submit(prev, v)
                prev = v
                if (i & 1) == 1: nv_wait_timeline(dev, v, what="T3", timeout_s=15.0)
            dev.synchronize()
        except RuntimeError as e:
            print(f"[T3] WEDGE/timeout: {e!r}", flush=True); ok = False
        if ok:
            got = dn(eids_g, (88,), np.uint16)
            ev = seed.astype(np.int64)
            for _ in range(4): ev = (ev*7 + 3) % NBANK
            exact = np.array_equal(got.astype(np.int64), ev)
            record("T3", f"{'COMPLETED-4-cycles' if ok else 'WEDGE'}-eids{'EXACT' if exact else 'RACY/EARLY-PASS'}")
        else:
            record("T3", "WEDGE-same-object-inflight")
            sys.exit(3)
        dev.allocator._copyin(eids_g, memoryview(seed.tobytes()))

    # ============================ T4 ============================
    if not done("T4"):
        g0 = MG([(mut, (eids_g,), 1), (gx, (ptbl, eids_g, xs, gridf, ys_g), 88)], "t4a")
        g1 = MG([(mut, (eids_g,), 1), (gx, (ptbl, eids_g, xs, gridf, ys_g), 88)], "t4b")
        def replay256(timed=False):
            # T4 run-1 verdict: depth-2 pipelining FAULTED (SKEDCHECK22_INVALIDATE_
            # ACTIVE_QMD -- the P7F1 law, timing-dependent; T3 survived 4 cycles by
            # luck). The PRODUCTION discipline = wait EVERY graph before the next
            # submit (gcycle run_tokens wait_each=True / PG_WAIT=1 semantics).
            prev = dev.timeline_value - 1
            blocks = []
            t0 = time.perf_counter()
            for i in range(256):
                v = dev.next_timeline()
                (g0 if (i & 1) == 0 else g1).submit(prev, v)
                nv_wait_timeline(dev, v, what="T4", timeout_s=20.0)
                prev = v
                if timed and (i & 63) == 63:
                    blocks.append((time.perf_counter()-t0)*1e3); t0 = time.perf_counter()
            dev.synchronize()
            return blocks
        print("[T4] parity-pair 256 replays [mut->gx] (timed)...", flush=True)
        blocks = replay256(timed=True)
        ev = seed.astype(np.int64)
        for _ in range(256): ev = (ev*7 + 3) % NBANK
        got1 = dn(eids_g, (88,), np.uint16)
        ok1 = np.array_equal(got1.astype(np.int64), ev)
        ys_run1 = dn(ys_g, (88, 512))
        dev.allocator._copyin(eids_g, memoryview(seed.tobytes()))
        replay256()
        got2 = dn(eids_g, (88,), np.uint16)
        ys_run2 = dn(ys_g, (88, 512))
        det = np.array_equal(got1, got2) and np.array_equal(ys_run1, ys_run2)
        med = sorted(blocks)[len(blocks)//2] if blocks else -1
        gbs = (64 * NPAIR * GATE_B) / (med/1e3) / 1e9 if med > 0 else -1
        record("T4", f"256-replays-eids{'EXACT' if ok1 else 'BAD'}-det{'OK' if det else 'FAIL'}-"
                     f"per64blk {med:.1f} ms -> {gbs:.0f} GB/s (88-pair, in-graph)")
        print(f"[T4] per-64-block median {med:.2f} ms -> {gbs:.0f} GB/s in-graph (eids exact={ok1}, det={det})", flush=True)
        dev.allocator._copyin(eids_g, memoryview(seed.tobytes()))

    # ============================ T5 ============================
    if not done("T5"):
        big704 = np.concatenate([np.random.default_rng(300+i).choice(256, 8, replace=False) for i in range(88)]).astype(np.uint16)
        eids704 = up(big704)
        def chain():
            return MG([(hrot, (hb,), 1), (rt, (Wb, hb, eids704), 88),
                       (gx, (ptbl, eids704, xs704, gridf, ys704), 704)], "t5")
        c0, c1 = chain(), chain()
        def chain256(timed=False):
            # wait-each (the SKEDCHECK22 law; see T4)
            prev = dev.timeline_value - 1
            blocks = []
            t0 = time.perf_counter()
            for i in range(256):
                v = dev.next_timeline()
                (c0 if (i & 1) == 0 else c1).submit(prev, v)
                nv_wait_timeline(dev, v, what="T5", timeout_s=20.0)
                prev = v
                if timed and (i & 63) == 63:
                    blocks.append((time.perf_counter()-t0)*1e3); t0 = time.perf_counter()
            dev.synchronize()
            return blocks
        print("[T5] chain [hrot->rt->gx704] 256 replays (timed)...", flush=True)
        blocks = chain256(timed=True)
        h_dev = dn(hb, (88, 2048))
        refs = rt_ref(W_np, h_dev, range(0, 88, 11))
        got = dn(eids704, (88, 8), np.uint16)
        ok_rt = all(np.array_equal(refs[p], got[p]) for p in refs)
        ys1 = dn(ys704, (704, 512))
        dev.allocator._copyin(hb, memoryview(h0.tobytes()))
        chain256()
        h_dev2 = dn(hb, (88, 2048))
        got2 = dn(eids704, (88, 8), np.uint16)
        ys2 = dn(ys704, (704, 512))
        det = np.array_equal(got, got2) and np.array_equal(ys1, ys2) and np.array_equal(h_dev, h_dev2)
        med = sorted(blocks)[len(blocks)//2] if blocks else -1
        gbs = (64 * 704 * GATE_B) / (med/1e3) / 1e9 if med > 0 else -1
        record("T5", f"chain256-rt{'EXACT' if ok_rt else 'BAD'}-det{'OK' if det else 'FAIL'}-"
                     f"per64blk {med:.1f} ms -> {gbs:.0f} GB/s (704-pair, in-graph)")
        print(f"[T5] per-64-block median {med:.2f} ms -> {gbs:.0f} GB/s in-graph (rt exact={ok_rt}, det={det})", flush=True)

    # ============================ T7 ============================
    if not done("T7"):
        g0 = MG([(mut, (eids_g,), 1), (gx, (ptbl, eids_g, xs, gridf, ys_g), 88)], "t7a")
        g1 = MG([(mut, (eids_g,), 1), (gx, (ptbl, eids_g, xs, gridf, ys_g), 88)], "t7b")
        print("[T7] endurance 1024 replays (the ~950-cycle budget probe)...", flush=True)
        prev = dev.timeline_value - 1
        ok = True
        try:
            t0 = time.perf_counter()
            for i in range(1024):
                v = dev.next_timeline()
                (g0 if (i & 1) == 0 else g1).submit(prev, v)
                nv_wait_timeline(dev, v, what="T7", timeout_s=20.0)   # wait-each (SKEDCHECK22 law)
                prev = v
                if (i & 255) == 255:
                    print(f"[T7] {i+1}/1024 replays, {(time.perf_counter()-t0)*1e3/(i+1):.0f} us/replay", flush=True)
            dev.synchronize()
        except RuntimeError as e:
            print(f"[T7] wedge/fault at replay log: {e!r}", flush=True); ok = False
        el = time.perf_counter()-t0
        ev = seed.astype(np.int64)
        for _ in range(1024): ev = (ev*7 + 3) % NBANK
        got = dn(eids_g, (88,), np.uint16)
        record("T7", f"1024-replays-{'CLEAN' if ok else 'FAULTED'}-eids{'EXACT' if np.array_equal(got.astype(np.int64), ev) else 'BAD'}-"
                     f"{el/1024*1e6:.0f}us/replay")
    print("[ALL DONE] see ~/mm_p1_q_progress.txt", flush=True)

if __name__ == "__main__":
    import threading
    def _wd():
        time.sleep(900)
        print("[WATCHDOG] total deadline exceeded — exiting for reboot", flush=True)
        os._exit(4)
    threading.Thread(target=_wd, daemon=True).start()
    main()
