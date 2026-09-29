#!/usr/bin/env python3
"""MM P1 PRODUCTION v0 A/B — rt8e256 (gold-router epilogue) + gx8e256up (SWIGLU).

Stages:
  S0 build + audit (cuobjdump: 0 spill, maxntid=launch bounds) + load
  S1 rt8e256 eager vs numpy: top-8 ids BIT-EXACT, gates/sg allclose (expf ULP),
     det x2 exact
  S2 gx8e256up eager vs numpy on a structured small bank (gate|up slabs):
     ys allclose (expf epilogue), det x2 exact
  S3 in-graph chain [hrot -> rt8e256 -> gx8e256up] x 256 replays, WAIT-EACH
     (the SKEDCHECK22 law), device-written eids: ids exact vs rt_ref(final h),
     det x2 NaN-aware, per-64-block GB/s at 704 pairs x (gate+up)
  S4 the production decode shape: P=9 (K=8) -> 72 pairs, timed 64 replays
Progress file survives reboots. One GPU process per boot.
"""
import os, sys, time, subprocess
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
sys.path.insert(0, "~/tinygrad-metal")
BASE = "~/tinygrad-metal"
SLAB, GATE_B, SHARD, NSHARD, NPAIR = 1458176, 450560, 256, 20, 88
NBANK = SHARD * NSHARD
PROG = os.path.expanduser("~/mm_p1_prod_progress.txt")

def record(tag, result):
    with open(PROG, "a") as f: f.write(f"{tag} {result}\n")
    print(f"[PROGRESS] {tag} {result}", flush=True)

def done(tag):
    if not os.path.exists(PROG): return False
    return any(l.split()[0] == tag for l in open(PROG).read().splitlines() if l.split())

def naneq(a, b):
    return np.array_equal(a, b) or ((a == b) | (np.isnan(a) & np.isnan(b))).all()

def sh(cmd, env=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)

def build_all():
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
               DOCKER_HOST="unix://~/.colima/default/docker.sock")
    for src, extra in (("MM_P1_rt8e256", ""), ("MM_P1_gx8e256up", "-fmad=false")):
        cu, cb = f"{BASE}/{src}.cu", f"{BASE}/{src}.cubin"
        r = sh(f"nvcc -arch=sm_86 -cubin {extra} --output-file={cb} {cu}", env=env)
        if r.returncode: print(r.stderr[-2000:]); sys.exit(1)
        print(f"[built] {src}.cubin", flush=True)
    for src in ("MM_P1_rt8e256", "MM_P1_gx8e256up"):
        r = sh(f"cuobjdump -res-usage {BASE}/{src}.cubin")
        lines = [l.strip() for l in r.stdout.splitlines() if "STACK" in l.upper() or "SPILL" in l.upper() or "REG" in l.upper()]
        print(f"[audit] {src}: " + (" | ".join(lines) if lines else "(clean)"), flush=True)

from MM_P0_d2_repack import dq_iq3_s

def iq3s_grid_f32():
    from tinygrad.runtime.autogen.ggml_common import iq3s_grid
    v = np.array([(int(w) >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.uint8)
    assert v.size == 2048
    return v.view(np.int8).astype(np.float32).reshape(512, 4).copy()

def logits_ref(W, h, p):
    """POC rt_ref logit computation (bit-exact kernel order)."""
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
    return lg

def rt_prod_ref(W, wsh, h, positions):
    """ids from RAW fp32 logits (bit-exact, tie->lower); gates/sg in float64 exp."""
    out = {}
    for p in positions:
        lg = logits_ref(W, h, p)
        l = lg.copy()
        ids, ex = [], []
        for r in range(8):
            be = 0; bv = l[0]
            for e2 in range(1, 256):
                if l[e2] > bv: bv = l[e2]; be = e2
            ids.append(be); ex.append(np.exp(np.float64(bv - lg.max()))); l[be] = np.float32(-3.4e38)
        s8 = np.sum(ex)
        out[p] = (np.array(ids, dtype=np.uint16), np.array(ex, dtype=np.float32)/np.float32(s8),
                  np.float32(1.0/(1.0+np.exp(-np.float64(np.dot(wsh.astype(np.float64), h[p].astype(np.float64)))))))
    return out

def gxup_ref(rows_g, rows_u, x, gridf):
    """Kernel-order reference for gx8e256up: per-lane running sums (b asc, j asc),
    xor tree, silu(g)*u epilogue."""
    Wg = dq_iq3_s(np.ascontiguousarray(rows_g), 2048)
    Wu = dq_iq3_s(np.ascontiguousarray(rows_u), 2048)
    def dot_ref(Wmat):
        partial = np.zeros((512, 32), dtype=np.float32)
        Wb = Wmat.reshape(512, 8, 32, 8)
        xk = x.reshape(8, 32, 8)
        prod = (Wb * xk[None, :, :, :]).astype(np.float32)
        for b in range(8):
            for j in range(8):
                partial = partial + prod[:, b, :, j]
        p = partial
        for o in (16, 8, 4, 2, 1):
            p = p + p[:, np.arange(32) ^ o]
        return p[:, 0]
    g = dot_ref(Wg); u = dot_ref(Wu)
    return (g / (1.0 + np.exp(-g.astype(np.float64))).astype(np.float32)) * u

def main():
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram, NVComputeQueue, nv_wait_timeline
    from tinygrad.helpers import round_up
    from tinygrad.uop.ops import UOp
    from tinygrad.dtype import dtypes

    build_all()
    def prog(sym):
        lib = open(f"{BASE}/MM_P1_{sym}.cubin", "rb").read()
        return NVProgram(dev, TinyELF(lib=lib, name=sym, target=dev.renderer.target, signature=tuple()))
    rt = prog("rt8e256")
    gxup = prog("gx8e256up")
    hrot_p = None
    try:
        lib = open(f"{BASE}/MM_P0_hrot.cubin", "rb").read()
        hrot_p = NVProgram(dev, TinyELF(lib=lib, name="mm_hrot", target=dev.renderer.target, signature=tuple()))
    except Exception as e:
        print(f"hrot load failed: {e}", flush=True)
    dev.synchronize(); print("[S0] rt8e256 + gx8e256up (+hrot) loaded", flush=True)

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

    gridf = up(iq3s_grid_f32())
    W_np = (np.random.default_rng(5).uniform(-0.05, 0.05, (256, 2048))).astype(np.float32)
    Wb = up(W_np)
    wsh_np = (np.random.default_rng(15).uniform(-0.05, 0.05, 2048)).astype(np.float32)
    wshb = up(wsh_np)
    h0 = (np.random.default_rng(6).uniform(-0.5, 0.5, (88, 2048))).astype(np.float32)
    hb = up(h0)
    eids_c = up(np.zeros(704, dtype=np.uint16))
    gates_c = dev.allocator.alloc(88*8*4, BufferSpec())
    sg_c = dev.allocator.alloc(88*4, BufferSpec())
    ys_c = dev.allocator.alloc(704*512*4, BufferSpec())

    # 7.3GB bank (gate+up live at slab+0 / slab+450560 -- same layout as pack36 banks)
    rng = np.random.default_rng(7)
    banks = [dev.allocator.alloc(SLAB*SHARD, BufferSpec()) for _ in range(NSHARD)]
    CH = 64 << 20
    for bank in banks:
        for off in range(0, SLAB*SHARD, CH):
            n = min(CH, SLAB*SHARD - off)
            a = rng.integers(0, 256, n, dtype=np.uint8); keep.append(a)
            dev.allocator._copyin(bank.offset(offset=off, size=n), memoryview(a.data).cast("B"))
    ptbl = up(np.array([banks[e//SHARD].va_addr + (e % SHARD)*SLAB for e in range(NBANK)], dtype=np.uint64))
    dev.synchronize(); print("[S0] buffers + bank up", flush=True)

    # ---------------- S1: rt8e256 eager vs numpy ----------------
    if not done("S1"):
        rt(Wb, wshb, hb, eids_c, gates_c, sg_c, global_size=(88,1,1), local_size=(1024,1,1), wait=True)
        got_ids = dn(eids_c, (88, 8), np.uint16)
        got_g = dn(gates_c, (88, 8))
        got_sg = dn(sg_c, (88,))
        refs = rt_prod_ref(W_np, wsh_np, h0, range(0, 88, 7))
        ids_ok = all(np.array_equal(refs[p][0], got_ids[p]) for p in refs)
        g_ok = all(np.allclose(refs[p][1], got_g[p], rtol=1e-5, atol=1e-7) for p in refs)
        sg_ok = all(abs(refs[p][2] - got_sg[p]) < 1e-5 for p in refs)
        gs_sum = np.abs(got_g.sum(axis=1) - 1.0).max()
        rt(Wb, wshb, hb, eids_c, gates_c, sg_c, global_size=(88,1,1), local_size=(1024,1,1), wait=True)
        det = (np.array_equal(got_ids, dn(eids_c, (88, 8), np.uint16)) and
               np.array_equal(got_g, dn(gates_c, (88, 8))) and np.array_equal(got_sg, dn(sg_c, (88,))))
        record("S1", f"rt8e256 ids{'EXACT' if ids_ok else 'BAD'} gates{'OK' if g_ok else 'BAD'} "
                     f"sg{'OK' if sg_ok else 'BAD'} renormmaxdev {gs_sum:.2e} det{'OK' if det else 'FAIL'}")
        print(f"[S1] ids exact={ids_ok} gates ok={g_ok} sg ok={sg_ok} renorm dev={gs_sum:.2e} det={det}", flush=True)

    # ---------------- S2: gx8e256up eager vs numpy (structured small bank) ----------------
    if not done("S2"):
        E_S = 8
        GU_SLAB = 2*GATE_B
        rngs = np.random.default_rng(21)
        sbank_np = rngs.integers(0, 256, GU_SLAB*E_S, dtype=np.uint8)
        # structured rows: force finite d16 in gate+up mats (P0 S1 class)
        for e in range(E_S):
            for m in range(2):
                rows = sbank_np[e*GU_SLAB + m*GATE_B : e*GU_SLAB + (m+1)*GATE_B].reshape(512, 880)
                d16 = np.random.default_rng(1000+e*2+m).integers(0x3800, 0x4000, 512).astype(np.uint16)
                rows[:, 0:2] = np.ascontiguousarray(d16).view(np.uint8).reshape(512, 2)
        sbank = up(sbank_np)
        ptbl_s = up(np.array([sbank.va_addr + e*GU_SLAB for e in range(E_S)], dtype=np.uint64))
        xs_np = (np.random.default_rng(11).uniform(-0.5, 0.5, (1, 2048))).astype(np.float32)  # P=1, 8 pairs
        xs1 = up(xs_np)
        eids8 = up(np.arange(8, dtype=np.uint16))
        ys8 = dev.allocator.alloc(8*512*4, BufferSpec())
        gxup(ptbl_s, eids8, xs1, gridf, ys8, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
        got = dn(ys8, (8, 512))
        ok = True
        for e in range(8):
            rg = sbank_np[e*GU_SLAB : e*GU_SLAB + GATE_B].reshape(512, 880)
            ru = sbank_np[e*GU_SLAB + GATE_B : e*GU_SLAB + 2*GATE_B].reshape(512, 880)
            yr = gxup_ref(rg, ru, xs_np[0], None)
            finite = np.isfinite(yr) & np.isfinite(got[e])
            if not np.allclose(got[e][finite], yr[finite], rtol=1e-5, atol=1e-6): ok = False
        gxup(ptbl_s, eids8, xs1, gridf, ys8, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
        det = np.array_equal(got, dn(ys8, (8, 512)))
        ff = float(np.isfinite(got).mean())
        record("S2", f"gx8e256up vs numpy allclose{'OK' if ok else 'BAD'} det{'OK' if det else 'FAIL'} finite-frac {ff:.3f}")
        print(f"[S2] allclose={ok} det={det} finite={ff:.3f}", flush=True)

    # ---------------- S3: chain [hrot -> rt -> gxup] 256 replays, wait-each ----------------
    if not done("S3"):
        LS = {"rt8e256": (1024,1,1), "gx8e256up": (1024,1,1), "mm_hrot": (128,1,1)}
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
        def run256(g0, g1, timed=False):
            prev = dev.timeline_value - 1
            blocks = []
            t0 = time.perf_counter()
            for i in range(256):
                v = dev.next_timeline()
                (g0 if (i & 1) == 0 else g1).submit(prev, v)
                nv_wait_timeline(dev, v, what="S3", timeout_s=30.0)   # WAIT-EACH law
                prev = v
                if timed and (i & 63) == 63:
                    blocks.append((time.perf_counter()-t0)*1e3); t0 = time.perf_counter()
            dev.synchronize()
            return blocks
        c0 = MG([(hrot_p, (hb,), 1), (rt, (Wb, wshb, hb, eids_c, gates_c, sg_c), 88),
                 (gxup, (ptbl, eids_c, hb, gridf, ys_c), 704)], "s3a")
        c1 = MG([(hrot_p, (hb,), 1), (rt, (Wb, wshb, hb, eids_c, gates_c, sg_c), 88),
                 (gxup, (ptbl, eids_c, hb, gridf, ys_c), 704)], "s3b")
        blocks = run256(c0, c1, timed=True)
        h1 = dn(hb, (88, 2048)); got1 = dn(eids_c, (88, 8), np.uint16); ys1 = dn(ys_c, (704, 512))
        refs = rt_prod_ref(W_np, wsh_np, h1, range(0, 88, 7))
        ids_ok = all(np.array_equal(refs[p][0], got1[p]) for p in refs)
        dev.allocator._copyin(hb, memoryview(h0.tobytes()))
        run256(c0, c1)
        h2 = dn(hb, (88, 2048)); got2 = dn(eids_c, (88, 8), np.uint16); ys2 = dn(ys_c, (704, 512))
        det = np.array_equal(h1, h2) and np.array_equal(got1, got2) and naneq(ys1, ys2)
        med = sorted(blocks)[len(blocks)//2] if blocks else -1
        bts = 704 * (2*GATE_B)
        gbs = (64 * bts) / (med/1e3) / 1e9 if med > 0 else -1
        record("S3", f"chain256-ids{'EXACT' if ids_ok else 'BAD'}-det{'OK' if det else 'FAIL'}-"
                     f"per64blk {med:.1f} ms -> {gbs:.0f} GB/s (704-pair gate+up in-graph)")
        print(f"[S3] ids exact={ids_ok} det={det} per64 {med:.2f}ms -> {gbs:.0f} GB/s", flush=True)
        dev.allocator._copyin(hb, memoryview(h0.tobytes()))

    # ---------------- S4: production decode shape P=9 (K=8) ----------------
    if not done("S4"):
        h9 = up(h0[:9].copy())
        e9 = up(np.zeros(72, dtype=np.uint16))
        y9 = dev.allocator.alloc(72*512*4, BufferSpec())
        LS = {"rt8e256": (1024,1,1), "gx8e256up": (1024,1,1)}
        class MG2:
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
        d0 = MG2([(rt, (Wb, wshb, h9, e9, gates_c, sg_c), 9),
                  (gxup, (ptbl, e9, h9, gridf, y9), 72)], "s4a")
        d1 = MG2([(rt, (Wb, wshb, h9, e9, gates_c, sg_c), 9),
                  (gxup, (ptbl, e9, h9, gridf, y9), 72)], "s4b")
        prev = dev.timeline_value - 1
        blocks = []
        t0 = time.perf_counter()
        for i in range(256):
            v = dev.next_timeline()
            (d0 if (i & 1) == 0 else d1).submit(prev, v)
            nv_wait_timeline(dev, v, what="S4", timeout_s=30.0)
            prev = v
            if (i & 63) == 63:
                blocks.append((time.perf_counter()-t0)*1e3); t0 = time.perf_counter()
        dev.synchronize()
        med = sorted(blocks)[len(blocks)//2]
        bts = 72 * (2*GATE_B) + 256*2048*4 + 9*2048*4
        gbs = (64 * bts) / (med/1e3) / 1e9
        us = med*1e3/64
        record("S4", f"K8-shape 256 replays: {us:.0f} us/cycle -> {gbs:.0f} GB/s effective "
                     f"({bts/1e6:.1f} MB/cycle, wait-each)")
        print(f"[S4] {us:.0f} us/cycle -> {gbs:.0f} GB/s effective", flush=True)
    print("[ALL DONE]", flush=True)

if __name__ == "__main__":
    import threading
    def _wd():
        time.sleep(900); print("[WATCHDOG] deadline", flush=True); os._exit(4)
    threading.Thread(target=_wd, daemon=True).start()
    main()
