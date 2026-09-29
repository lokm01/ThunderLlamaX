#!/usr/bin/env python3
"""MM P0 D3+D4 — THE GATHER-GEMV POC bench + mutating-ids-under-graphs.

D3: gx8e256nw32 (one cubin, nw32, flat sequential pair walk, expert-pointer
table, IQ3_S dequant GEMV over an ~8GB packed bank):
  (a) grouped BW   (all pairs -> 8 experts)
  (b) scattered BW (~75 distinct experts of 256 — the expected decode case)
  (c) bit-exact vs a numpy reference with EXACT op order (+ det x2)
  (d) graph capture + 256 replays with device-MUTATED pair lists
  (+) contiguous-64 + full-bank scatter arms -> the TLB/scatter tax
D4: router-write -> gemv-read chain (eids written ONLY by kernels; 16 x 256
  replays with changing ids; fresh-graph rebuild each 256 (the cadence law);
  determinism x2; fault watch).

Usage: ~/tg311/bin/python MM_P0_d3_gather.py [--quick]
"""
import os, sys, time, subprocess
import numpy as np

os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")

BASE = "~/tinygrad-metal"
SLAB = 1458176
GATE_B = 450560
SHARD = 256             # experts per bank (256*1.458MB = 373MB — under the ~457MB arg cap)
NSHARD = 20             # 20 banks = 5120 experts = 7.3GB total bank
NBANK = SHARD * NSHARD  # 5120
NPAIR = 88
QUICK = "--quick" in sys.argv

def sh(cmd, env=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)

def build_all():
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
               DOCKER_HOST="unix://~/.colima/default/docker.sock")
    for src, extra in (("MM_P0_gx8e256nw32", "-fmad=false"), ("MM_P0_mut", ""),
                       ("MM_P0_rt8e256poc", "-fmad=false"), ("MM_P0_hrot", ""),
                       ("MM_P0_k2s36", "")):
        cu, cb = f"{BASE}/{src}.cu", f"{BASE}/{src}.cubin"
        r = sh(f"nvcc -arch=sm_86 -cubin {extra} --output-file={cb} {cu}", env=env)
        if r.returncode:
            print(r.stderr[-2000:]); sys.exit(1)
        print(f"[built] {src}.cubin", flush=True)
    for src in ("MM_P0_gx8e256nw32", "MM_P0_rt8e256poc", "MM_P0_k2s36", "MM_P0_mut", "MM_P0_hrot"):
        r = sh(f"cuobjdump -res-usage {BASE}/{src}.cubin")
        lines = [l.strip() for l in r.stdout.splitlines() if "STACK" in l.upper() or "SPILL" in l.upper()]
        print(f"[audit] {src}: " + (" | ".join(lines) if lines else "(clean)"), flush=True)

def iq3s_grid_f32():
    from tinygrad.runtime.autogen.ggml_common import iq3s_grid
    v = np.array([(int(w) >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.uint8)
    assert v.size == 2048
    return v.view(np.int8).astype(np.float32).reshape(512, 4).copy()

# ---------------- numpy reference: EXACT op order of gx8e256nw32 ----------------
def gx_ref(rows, x, gridf):
    """rows [512,880] u8 one expert gate mat; x [2048] f32 -> y [512] f32."""
    lane = np.arange(32, dtype=np.int64)
    koff = lane << 3
    partial = np.zeros((512, 32), dtype=np.float32)
    for b in range(8):
        blk = rows[:, b*110:(b+1)*110]
        d = np.ascontiguousarray(blk[:, 0:2]).view(np.float16).astype(np.float32)[:, 0]
        sraw = lane >> 2
        nib = (np.take(blk, 106 + (sraw >> 1), axis=1) >> ((sraw & 1) << 2)) & 0xF
        sc = 1.0 + 2.0*nib.astype(np.float32)
        sg = np.take(blk, 74 + lane, axis=1)
        qlo = np.take(blk, 2 + 2*lane, axis=1).astype(np.uint16)
        qhi = np.take(blk, 3 + 2*lane, axis=1).astype(np.uint16)
        q16 = qlo | (qhi << 8)
        g0i, g1i = lane*2, lane*2 + 1
        bit0 = (np.take(blk, 66 + (g0i >> 3), axis=1) >> (g0i & 7)) & 1
        bit1 = (np.take(blk, 66 + (g1i >> 3), axis=1) >> (g1i & 7)) & 1
        qb0 = (q16 & 0xFF).astype(np.int64) + (bit0.astype(np.int64) << 8)
        qb1 = (q16 >> 8).astype(np.int64) + (bit1.astype(np.int64) << 8)
        t1 = d[:, None] * sc
        w = np.empty((512, 32, 8), dtype=np.float32)
        w[:, :, 0:4] = t1[:, :, None] * gridf[qb0]
        w[:, :, 4:8] = t1[:, :, None] * gridf[qb1]
        for j in range(8):
            wj = w[:, :, j].copy()
            m = (sg & (1 << j)) != 0
            wj[m] = -wj[m]
            partial = partial + (wj * x[koff + j][None, :]).astype(np.float32)
    p = partial
    for o in (16, 8, 4, 2, 1):
        p = p + p[:, np.arange(32) ^ o]
    return np.ascontiguousarray(p[:, 0])

def rt_ref(W, h, positions):
    """router top-8 for the given positions (the rt8e256poc exact order)."""
    out = {}
    for p in positions:
        wr_, hr_ = W[p], h[p]
        lg = np.empty(256, dtype=np.float32)
        for e in range(256):
            parts = []
            for part in range(4):
                w = wr_[e, part*512:(part+1)*512]; hh = hr_[part*512:(part+1)*512]
                s = np.float32(0)
                for i in range(0, 512, 4):
                    s = np.float32(s + np.float32(w[i]*hh[i]))
                    s = np.float32(s + np.float32(w[i+1]*hh[i+1]))
                    s = np.float32(s + np.float32(w[i+2]*hh[i+2]))
                    s = np.float32(s + np.float32(w[i+3]*hh[i+3]))
                parts.append(s)
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

    build_all()
    def prog(n):
        lib = open(f"{BASE}/{n}.cubin", "rb").read()
        return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
    gx, mut = prog("MM_P0_gx8e256nw32"), prog("MM_P0_mut")
    rt, hrot = prog("MM_P0_rt8e256poc"), prog("MM_P0_hrot")
    print("[progs loaded]", flush=True)

    keep = []
    def up(arr):
        a = np.ascontiguousarray(arr); keep.append(a)
        b = dev.allocator.alloc(a.nbytes, BufferSpec())
        dev.allocator._copyin(b, memoryview(a.data).cast("B"))
        return b
    def dn(b, shape, dtype=np.float32):
        n = int(np.prod(shape))
        mv = memoryview(bytearray(int(n)*np.dtype(dtype).itemsize)).cast("B")
        dev.allocator._copyout(mv, b)
        return np.frombuffer(mv, dtype=dtype).reshape(shape).copy()
    def win(b, off, arr):
        a = np.ascontiguousarray(arr); keep.append(a)
        dev.allocator._copyin(b.offset(offset=off, size=a.nbytes), memoryview(a.data).cast("B"))

    # ---- bank: 20 x 373MB (the SINGLE-ARG CAP law: >~457MB-1.16GB per arg faults),
    #      expert table = ABSOLUTE 64-bit VAs (probe5-proven legal) ----
    rng = np.random.default_rng(7)
    banks = []
    for s in range(NSHARD):
        banks.append(dev.allocator.alloc(SLAB*SHARD, BufferSpec()))
    CH = 64 << 20
    ref_rows = {}
    for si, bank in enumerate(banks):
        for off in range(0, SLAB*SHARD, CH):
            n = min(CH, SLAB*SHARD - off)
            a = rng.integers(0, 256, n, dtype=np.uint8)
            if si == 0 and off == 0:
                # structured finite experts 0..7 embedded host-side (no windowed copyin)
                a = a.reshape(-1)
                for e in range(8):
                    r2 = np.random.default_rng(1000+e)
                    rows = r2.integers(0, 256, GATE_B, dtype=np.uint8).reshape(512, 880).copy()
                    dexp = r2.integers(0x3800, 0x4000, 512).astype(np.uint16)
                    dsign = (r2.integers(0, 2, 512).astype(np.uint16) << 15)
                    d16 = dexp | dsign
                    rows[:, 0:2] = np.ascontiguousarray(d16).view(np.uint8).reshape(512, 2)
                    ref_rows[e] = rows
                    a[e*SLAB:e*SLAB+GATE_B] = rows.reshape(-1)
                a = a.reshape(n)
            keep.append(a)
            dev.allocator._copyin(bank.offset(offset=off, size=n), memoryview(a.data).cast("B"))
    print("[bank] fill done (20 x 373MB, experts 0..7 structured in banks[0])", flush=True)
    ptbl_np = np.array([banks[e // SHARD].va_addr + (e % SHARD) * SLAB for e in range(NBANK)], dtype=np.uint64)
    ptbl = up(ptbl_np)
    gridf_np = iq3s_grid_f32(); gridf = up(gridf_np)
    xs_np = (np.random.default_rng(11).uniform(-0.5, 0.5, (NPAIR, 2048))).astype(np.float32)
    xs = up(xs_np)
    ys = dev.allocator.alloc(NPAIR*512*4, BufferSpec())
    dev.synchronize(); print("[buffers up]", flush=True)

    # ---------------- (c) bit-exact + det x2 ----------------
    print("\n== D3(c) bit-exact vs numpy reference (8 structured experts) ==")
    eids = up(np.arange(8, dtype=np.uint16))
    y_exp = np.stack([gx_ref(ref_rows[e], xs_np[e], gridf_np) for e in range(8)])
    outs = []
    for _ in range(2):
        gx(ptbl, eids, xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
        outs.append(dn(ys, (8, 512)))
    eq0 = np.array_equal(outs[0], y_exp)
    eqd = np.array_equal(outs[0], outs[1])
    print(f"  vs-ref: {'BIT-EXACT' if eq0 else f'DIFF nz={int((outs[0]!=y_exp).sum())}/{y_exp.size}'}; det-x2: {'OK' if eqd else 'FAIL'}")
    assert eq0 and eqd, "D3(c) FAILED"

    # ---------------- (a/b/+) BW arms ----------------
    r3 = np.random.default_rng(33)
    scat = np.concatenate([np.random.default_rng(100+i).choice(256, 8, replace=False) for i in range(11)]).astype(np.uint16)
    scat256 = np.concatenate([np.random.default_rng(200+i).choice(256, 8, replace=False) for i in range(11)]).astype(np.uint16)
    arms = {
        "grouped-8":  np.tile(np.arange(8, dtype=np.uint16), 11),
        "contig-64":  (np.arange(88, dtype=np.uint16) % 64 + 100),
        "scattered":  scat256,   # ~75 distinct of 256 (the E[dist|M=11] decode case)
        "bankscatter": r3.integers(0, NBANK, 88).astype(np.uint16),  # ~87 distinct over 7.3GB
    }
    byts = NPAIR*GATE_B
    print(f"\n== D3(a/b/+) BW arms (grid 88, {byts/1e6:.2f} MB/launch) ==")
    res = {}
    for name, ev in arms.items():
        eb = up(np.ascontiguousarray(ev))
        gx(ptbl, eb, xs, gridf, ys, global_size=(NPAIR,1,1), local_size=(1024,1,1), wait=True)
        ts = []
        for _ in range(5 if QUICK else 20):
            t0 = time.perf_counter()
            gx(ptbl, eb, xs, gridf, ys, global_size=(NPAIR,1,1), local_size=(1024,1,1), wait=True)
            ts.append((time.perf_counter()-t0)*1e3)
        res[name] = (min(ts), sorted(ts)[len(ts)//2], byts/min(ts)/1e3)
        print(f"  [{name:12s}] min {res[name][0]:7.3f} ms med {res[name][1]:7.3f} ms -> {res[name][2]:7.1f} GB/s (distinct={len(set(ev.tolist()))})")
    print(f"  SCATTER TAX: grouped {res['grouped-8'][2]:.0f} | contig-64 {res['contig-64'][2]:.0f} | scattered-256 {res['scattered'][2]:.0f} | bank-wide {res['bankscatter'][2]:.0f} GB/s")

    # ---------------- mini-graph helper (ParityGraph pattern, NOBIND) ----------------
    LS = {"MM_P0_gx8e256nw32": (1024,1,1), "MM_P0_rt8e256poc": (1024,1,1),
          "MM_P0_mut": (128,1,1), "MM_P0_hrot": (128,1,1)}
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
                q.exec(p, st, (grid,1,1), LS[getattr(p, "name", "")])
                off += round_up(p.kernargs_alloc_size, 8)
            q.signal(dev.timeline_signal, self.cur)
            self.q = q
            self.ring_bytes = len(q._q)*4
        def submit(self, pv, cv):
            self.q.submit(dev, {self.prev.expr: int(pv), self.cur.expr: int(cv)})
    def run_block(g, n, tag):
        """submit n replays at depth<=2 (the V-48 kernargs re-patch race law)."""
        prev = dev.timeline_value - 1
        t0 = time.perf_counter()
        for i in range(n):
            v = dev.next_timeline()
            g.submit(prev, v)
            prev = v
            if (i & 1) == 1:
                nv_wait_timeline(dev, v, what=tag)
        if n % 2 == 1:
            nv_wait_timeline(dev, prev, what=tag+"-tail")
        el = time.perf_counter()-t0
        dev.synchronize()
        return el

    # ---------------- (d) graph + 256 replays, device-mutated eids ----------------
    print("\n== D3(d) graph [mm_eidmut -> gx] x 256 replays, eids mutated ON DEVICE ==")
    seed = (np.arange(88, dtype=np.uint16) % 8)
    eids_g = up(seed)
    ys_g = dev.allocator.alloc(88*512*4, BufferSpec())
    g1 = MG([(mut, (eids_g,), 1), (gx, (ptbl, eids_g, xs, gridf, ys_g), 88)], "d3d")
    el = run_block(g1, 256, "d3d")
    print(f"  256 replays OK in {el*1e3:.1f} ms ({el/256*1e6:.0f} us/replay; {256*byts/el/1e9:.1f} GB/s pipelined incl. mutator)")
    ev = seed.astype(np.int64)
    for _ in range(256): ev = (ev*7 + 3) % NBANK
    got1 = dn(eids_g, (88,), np.uint16)
    print(f"  eids == numpy sim after 256 affine mutations: {np.array_equal(got1.astype(np.int64), ev)}")
    # det x2: reseed, replay, compare
    win(eids_g, 0, seed)
    run_block(g1, 256, "d3d-again")
    got2 = dn(eids_g, (88,), np.uint16)
    print(f"  rerun deterministic: {np.array_equal(got1, got2)}")

    # ---------------- D4 router -> topk -> gather chain ----------------
    print("\n== D4 chain [mm_hrot -> rt8e256poc -> gx(704)] — eids written ONLY by kernels ==")
    W_np = (np.random.default_rng(5).uniform(-0.05, 0.05, (256, 2048))).astype(np.float32)
    Wb = up(W_np)
    h0 = (np.random.default_rng(6).uniform(-0.5, 0.5, (88, 2048))).astype(np.float32)
    hb = up(h0)
    eids_c = up(np.zeros(704, dtype=np.uint16))
    xs704 = up(np.tile(xs_np, (8, 1)))
    ys_c = dev.allocator.alloc(704*512*4, BufferSpec())
    CHAIN = lambda tag: MG([(hrot, (hb,), 1), (rt, (Wb, hb, eids_c), 88),
                            (gx, (ptbl, eids_c, xs704, gridf, ys_c), 704)], tag)
    # one-shot rt verify before the graph (eids_c from current h)
    rt(Wb, hb, eids_c, global_size=(88,1,1), local_size=(1024,1,1), wait=True)
    got = dn(eids_c, (88, 8), np.uint16)
    refs = rt_ref(W_np, h0, range(0, 88, 11))
    ok_rt = all(np.array_equal(refs[p], got[p]) for p in refs)
    print(f"  rt8e256poc vs numpy ref (8 spot positions): {'EXACT' if ok_rt else 'MISMATCH'}")
    # round 1 (save for determinism)
    win(hb, 0, h0)
    g2 = CHAIN("d4r0")
    run_block(g2, 256, "d4-r0")
    eids_r1 = dn(eids_c, (88, 8), np.uint16)
    h_dev = dn(hb, (88, 2048))
    ref0 = rt_ref(W_np, h_dev, (0,))
    print(f"  after 256: eids[0] == ref(h_dev): {np.array_equal(ref0[0], eids_r1[0])}")
    rounds = 2 if QUICK else 16
    ok_all = True
    t0 = time.perf_counter()
    for rnd in range(1, rounds):
        g2 = CHAIN(f"d4r{rnd}")          # fresh graph = the 256-rebuild cadence
        run_block(g2, 256, f"d4-r{rnd}")
        if rnd % 4 == 0 or rnd == rounds-1:
            got_e = dn(eids_c, (88, 8), np.uint16)
            h_now = dn(hb, (88, 2048))
            refc = rt_ref(W_np, h_now, (0, 44, 87))
            okc = all(np.array_equal(refc[p], got_e[p]) for p in refc)
            ok_all &= okc
            print(f"  [r{rnd:2d}] cycles={rnd*256+256} eids-vs-ref({'OK' if okc else 'MISMATCH'}) distinct={len(set(got_e.flatten().tolist()))}")
    el = time.perf_counter()-t0
    print(f"  D4 total {rounds*256} mixed cycles, {el:.1f}s, no faults; ref checks {'ALL OK' if ok_all else 'FAILED'}")
    # determinism x2: reset h, run 256 (same as r0), compare eids
    win(hb, 0, h0)
    g2 = CHAIN("d4det")
    run_block(g2, 256, "d4-det")
    eids_det = dn(eids_c, (88, 8), np.uint16)
    print(f"  determinism x2 (fresh 256 from same seed): {np.array_equal(eids_r1, eids_det)}")

    print("\n== D3/D4 SUMMARY ==")
    print(f"  grouped {res['grouped-8'][2]:.0f} GB/s | contig64 {res['contig-64'][2]:.0f} | "
          f"scattered {res['scattered'][2]:.0f} | bankscatter {res['bankscatter'][2]:.0f}")
    print(f"  GO bar: >=180 grouped / >=120 scattered; KILL bar: <100 scattered")

if __name__ == "__main__":
    main()
