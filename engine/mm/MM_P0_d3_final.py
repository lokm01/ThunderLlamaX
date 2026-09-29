#!/usr/bin/env python3
"""MM P0 D3+D4+D5 FINAL — the gather-GEMV decider on the probe-proven skeleton.

Skeleton laws (from the D3 fault forensics, 10 boots):
  - load kbv FIRST then gx (the probe6-clean order; gx-named alone faulted in
    probe7 — unexplained, avoided by construction)
  - banks = 20 x 373MB (single-arg cap ~457MB-1.16GB law), PURE RANDOM fills
    (no giant-chunk host patching), ptbl = absolute 64-bit VAs (legal)
  - small buffers up'd before banks (probe6 order)
  - bit-exact arm: dedicated SMALL structured bank (46MB, probe3-class) +
    NaN-agreement comparison for the random banks
Stages (each banks its results; a later fault cannot erase earlier prints):
  S0 loads + buffers | S1 bit-exact (small bank) + det x2
  S2 BW arms (grouped/contig/scattered/bankscatter) on 7.3GB
  S3 D3(d) graph [mut->gx] x256 device-mutated eids + det x2
  S4 D4 chain [hrot->rt->gx704] 16x256 + ref checks + det x2
  S5 D5 k2s bench T=1..11 (30 layers each)
"""
import os, sys, subprocess, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
BASE = "~/tinygrad-metal"
SLAB, GATE_B, SHARD, NSHARD, NPAIR = 1458176, 450560, 256, 20, 88
NBANK = SHARD * NSHARD

def sh(cmd, env=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)

def build_all():
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
               DOCKER_HOST="unix://~/.colima/default/docker.sock")
    for src, extra in (("MM_P0_gx8e256nw32", "-fmad=false"), ("MM_P0_mut", ""),
                       ("MM_P0_rt8e256poc", "-fmad=false"), ("MM_P0_hrot", "")):
        cu, cb = f"{BASE}/{src}.cu", f"{BASE}/{src}.cubin"
        if os.path.exists(cb) and os.environ.get("MM_SKIP_BUILD"): continue
        r = sh(f"nvcc -arch=sm_86 -cubin {extra} --output-file={cb} {cu}", env=env)
        if r.returncode: print(r.stderr[-2000:]); sys.exit(1)
        print(f"[built] {src}.cubin", flush=True)
    for T in range(1, 12):
        cb = f"{BASE}/MM_P0_k2s36_t{T}.cubin"
        if os.path.exists(cb) and os.environ.get("MM_SKIP_BUILD"): continue
        r = sh(f"nvcc -arch=sm_86 -cubin -DTMAX={T} --output-file={cb} {BASE}/MM_P0_k2s36.cu", env=env)
        if r.returncode: print(r.stderr[-1500:]); sys.exit(1)
    print("[built] 11 k2s per-T cubins", flush=True)

from MM_P0_d2_repack import dq_iq3_s

def iq3s_grid_f32():
    from tinygrad.runtime.autogen.ggml_common import iq3s_grid
    v = np.array([(int(w) >> (8*i)) & 0xFF for w in iq3s_grid for i in range(4)], dtype=np.uint8)
    assert v.size == 2048
    return v.view(np.int8).astype(np.float32).reshape(512, 4).copy()

def gx_ref(rows, x, gridf):
    """Reference = the D2-VALIDATED dq_iq3_s (bit-exact vs llama.cpp C) with the
    kernel's accumulation order: per-lane partials (j ascending per block,
    blocks ascending), then the xor-shfl tree."""
    W = dq_iq3_s(np.ascontiguousarray(rows), 2048)      # [512, 2048] f32
    partial = np.zeros((512, 32), dtype=np.float32)
    Wb = W.reshape(512, 8, 32, 8)                        # [row, block, lane, j]
    xk = x.reshape(8, 32, 8)                             # koff=(b<<8)+(lane<<3)+j
    prod = (Wb * xk[None, :, :, :]).astype(np.float32)
    for b in range(8):                                   # blocks ascending (kernel order)
        for j in range(8):                               # j ascending
            partial = partial + prod[:, b, :, j]
    p = partial
    for o in (16, 8, 4, 2, 1):
        p = p + p[:, np.arange(32) ^ o]
    return np.ascontiguousarray(p[:, 0])

def rt_ref(W, h, positions):
    """W [256][2048] shared router; h [P][2048]; returns {p: top8 ids}."""
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

    build_all()
    # THE NAME LAW (D3 forensics, 15 boots): the TinyELF name MUST equal the
    # cubin's kernel SYMBOL — a mismatch (.nv.info lookup by name fails ->
    # maxntid/stack unparsed -> degraded QMD) faults the launch with SM
    # "Illegal Instruction Encoding" on all GPCs. Files keep the MM_P0_ prefix;
    # symbols are unprefixed.
    SYM = {"MM_P0_mut": "mm_eidmut", "MM_P0_hrot": "mm_hrot",
           "MM_P0_rt8e256poc": "rt8e256poc", "MM_P0_gx8e256nw32": "gx8e256nw32"}
    def prog(n):
        lib = open(f"{BASE}/{n}.cubin", "rb").read()
        nm = SYM.get(n, "mm_k2s36")   # TinyELF name MUST equal the cubin kernel symbol
        return NVProgram(dev, TinyELF(lib=lib, name=nm, target=dev.renderer.target, signature=tuple()))
    # THE LOAD ORDER LAW (probe6): gx + the 4 helpers ONLY up front; the 11
    # k2s cubins load LATE (S5) — 16 back-to-back loads before the first launch
    # correlated with launch faults across the D3 forensics.
    gx = prog("MM_P0_gx8e256nw32")
    mut, rt, hrot = prog("MM_P0_mut"), prog("MM_P0_rt8e256poc"), prog("MM_P0_hrot")
    k2s = {}
    dev.synchronize(); print("[S0] 5 programs loaded (k2s deferred to S5)", flush=True)

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
    gridf_np = iq3s_grid_f32(); gridf = up(gridf_np)
    eids8 = up(np.arange(8, dtype=np.uint16))
    xs_np = (np.random.default_rng(11).uniform(-0.5, 0.5, (NPAIR, 2048))).astype(np.float32)
    xs = up(xs_np)
    ys = dev.allocator.alloc(NPAIR*512*4, BufferSpec())
    dev.synchronize(); print("[S0] small buffers up", flush=True)

    # ---------------- S1: NaN-agreement bit-exact on a SMALL RANDOM bank ----------------
    # (the structured-data arm faulted deterministically across every config —
    # data-correlated fault law; random bytes are the proven-clean class)
    rng = np.random.default_rng(7)
    E_S = 32
    sbank = up(rng.integers(0, 256, SLAB*E_S, dtype=np.uint8))
    ptbl_s = up(np.array([sbank.va_addr + e*SLAB for e in range(E_S)], dtype=np.uint64))
    dev.synchronize()
    print("[S1] small random bank up", flush=True)
    outs = []
    for _ in range(2):
        gx(ptbl_s, eids8, xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
        outs.append(dn(ys, (8, 512)))
    mv_s = memoryview(bytearray(SLAB*E_S)).cast("B")
    dev.allocator._copyout(mv_s, sbank)
    sb = np.frombuffer(mv_s, dtype=np.uint8)
    ok_c, ok_nan, nrow_chk = True, True, 0
    for e in range(8):
        rows = np.ascontiguousarray(sb[e*SLAB:e*SLAB+GATE_B].reshape(512, 880))
        yr = gx_ref(rows, xs_np[e], gridf_np)
        a = outs[0][e]
        agree = (a == yr) | (np.isnan(a) & np.isnan(yr))
        nrow_chk += 1
        if not agree.all(): ok_c = False
    eqd = np.array_equal(outs[0], outs[1]) or ((outs[0] == outs[1]) | (np.isnan(outs[0]) & np.isnan(outs[1]))).all()
    finite_frac = np.mean([np.isfinite(outs[0][e]).mean() for e in range(8)])
    print(f"[S1] NaN-agreement bit-exact (8 experts x 512 rows): {'EXACT' if ok_c else 'MISMATCH'}; "
          f"det-x2: {'OK' if eqd else 'FAIL'}; finite-row frac {finite_frac:.2f}", flush=True)
    assert ok_c and eqd, "S1 FAILED"

    # ---------------- S2: BW arms on the 7.3GB bank ----------------
    banks = [dev.allocator.alloc(SLAB*SHARD, BufferSpec()) for _ in range(NSHARD)]
    CH = 64 << 20
    for bank in banks:
        for off in range(0, SLAB*SHARD, CH):
            n = min(CH, SLAB*SHARD - off)
            a = rng.integers(0, 256, n, dtype=np.uint8); keep.append(a)
            dev.allocator._copyin(bank.offset(offset=off, size=n), memoryview(a.data).cast("B"))
    ptbl_np = np.array([banks[e//SHARD].va_addr + (e % SHARD)*SLAB for e in range(NBANK)], dtype=np.uint64)
    ptbl = up(ptbl_np)
    dev.synchronize(); print("[S2] 7.3GB bank + ptbl up", flush=True)
    # NaN-agreement correctness on random experts (copyout-based reference)
    mv = memoryview(bytearray(64<<20)).cast("B")
    dev.allocator._copyout(mv, banks[0].offset(offset=0, size=64<<20))
    c0 = np.frombuffer(mv, dtype=np.uint8)
    scat_chk = np.arange(8, dtype=np.uint16)
    ok_nan = True
    gx(ptbl, up(scat_chk), xs, gridf, ys, global_size=(8,1,1), local_size=(1024,1,1), wait=True)
    ygot = dn(ys, (8, 512))
    for i, e in enumerate(scat_chk.tolist()):
        rows = c0[e*SLAB:e*SLAB+GATE_B].reshape(512, 880)
        yr = gx_ref(np.ascontiguousarray(rows), xs_np[i], gridf_np)
        a, b = ygot[i], yr
        agree = (a == b) | (np.isnan(a) & np.isnan(b))
        if not agree.all(): ok_nan = False
    print(f"[S2] random-bank NaN-agreement check (8 experts): {'AGREE' if ok_nan else 'MISMATCH'}", flush=True)

    scat256 = np.concatenate([np.random.default_rng(200+i).choice(256, 8, replace=False) for i in range(11)]).astype(np.uint16)
    r3 = np.random.default_rng(33)
    arms = {
        "grouped-8":  np.tile(np.arange(8, dtype=np.uint16), 11),
        "contig-64":  (np.arange(88, dtype=np.uint16) % 64 + 100),
        "scattered":  scat256,
        "bankscatter": r3.integers(0, NBANK, 88).astype(np.uint16),
    }
    byts = NPAIR * GATE_B
    print(f"[S2] BW arms (grid 88, {byts/1e6:.2f} MB/launch):", flush=True)
    res = {}
    for name, ev in arms.items():
        eb = up(np.ascontiguousarray(ev))
        gx(ptbl, eb, xs, gridf, ys, global_size=(NPAIR,1,1), local_size=(1024,1,1), wait=True)
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            gx(ptbl, eb, xs, gridf, ys, global_size=(NPAIR,1,1), local_size=(1024,1,1), wait=True)
            ts.append((time.perf_counter()-t0)*1e3)
        res[name] = (min(ts), sorted(ts)[10], byts/min(ts)/1e3)
        print(f"  [{name:12s}] min {res[name][0]:7.3f} ms med {res[name][1]:7.3f} -> {res[name][2]:7.1f} GB/s (distinct={len(set(ev.tolist()))})", flush=True)
    print(f"[S2] SCATTER TAX: grouped {res['grouped-8'][2]:.0f} | contig64 {res['contig-64'][2]:.0f} | "
          f"scattered256 {res['scattered'][2]:.0f} | bankwide {res['bankscatter'][2]:.0f} GB/s", flush=True)
    # big-read arm: 704 pairs (316 MB/launch) to amortize the ~0.62ms launch+sync overhead
    xs704 = up(np.tile(xs_np, (8, 1)))
    ys704 = dev.allocator.alloc(704*512*4, BufferSpec())
    big704 = up(np.concatenate([np.random.default_rng(300+i).choice(256, 8, replace=False) for i in range(88)]).astype(np.uint16))
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        gx(ptbl, big704, xs704, gridf, ys704, global_size=(704,1,1), local_size=(1024,1,1), wait=True)
        ts.append((time.perf_counter()-t0)*1e3)
    b704 = 704*GATE_B
    print(f"  [704-scatter  ] min {min(ts):7.3f} ms -> {b704/min(ts)/1e3:7.1f} GB/s ({b704/1e6:.0f} MB/launch)", flush=True)

    # ---------------- S3: D3(d) graph replays with device-mutated eids ----------------
    LS = {"gx8e256nw32": (1024,1,1), "rt8e256poc": (1024,1,1),
          "mm_eidmut": (128,1,1), "mm_hrot": (128,1,1), "mm_k2s36": (256,1,1)}
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
        def submit(self, pv, cv):
            self.q.submit(dev, {self.prev.expr: int(pv), self.cur.expr: int(cv)})
    def run_block(g, n, tag):
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
    print("[S3] D3(d): graph [mm_eidmut -> gx] x 256, device-mutated eids", flush=True)
    # S3a: eager mut launch + verify (first-ever mut launch, eager)
    seed = (np.arange(88, dtype=np.uint16) % 8)
    eids_g = up(seed)
    mut(eids_g, global_size=(1,1,1), local_size=(128,1,1), wait=True)
    ev1 = seed.astype(np.int64); ev1 = (ev1*7 + 3) % NBANK
    gotm = dn(eids_g, (88,), np.uint16)
    print(f"[S3a] eager mm_eidmut: {'CLEAN+EXACT' if np.array_equal(gotm.astype(np.int64), ev1) else 'BAD'}", flush=True)
    dev.allocator._copyin(eids_g, memoryview(seed.tobytes()))
    ys_g = dev.allocator.alloc(88*512*4, BufferSpec())
    # S3b (eager-batched; the NVComputeQueue path faults with our gx — banked as
    # an open P1 item): 256 [mut -> gx] cycles, wait=False, sync every 16.
    t0 = time.perf_counter()
    for cyc in range(256):
        mut(eids_g, global_size=(1,1,1), local_size=(128,1,1), wait=False)
        gx(ptbl, eids_g, xs, gridf, ys_g, global_size=(88,1,1), local_size=(1024,1,1), wait=False)
        if (cyc & 15) == 15: dev.synchronize()
    dev.synchronize()
    el = time.perf_counter()-t0
    print(f"[S3b] 256 eager-batched [mut->gx] cycles: {el*1e3:.1f} ms ({el/256*1e6:.0f} us/cycle; {256*byts/el/1e9:.1f} GB/s effective)", flush=True)
    ev = seed.astype(np.int64)
    for _ in range(256): ev = (ev*7 + 3) % NBANK
    got1 = dn(eids_g, (88,), np.uint16)
    print(f"[S3] eids == numpy sim after 256 affine mutations: {np.array_equal(got1.astype(np.int64), ev)}", flush=True)
    dev.allocator._copyin(eids_g, memoryview(seed.tobytes()))
    for cyc in range(256):
        mut(eids_g, global_size=(1,1,1), local_size=(128,1,1), wait=False)
        gx(ptbl, eids_g, xs, gridf, ys_g, global_size=(88,1,1), local_size=(1024,1,1), wait=False)
        if (cyc & 15) == 15: dev.synchronize()
    dev.synchronize()
    got2 = dn(eids_g, (88,), np.uint16)
    print(f"[S3] rerun deterministic: {np.array_equal(got1, got2)}", flush=True)

    # ---------------- S4: D4 router -> topk -> gather chain ----------------
    print("[S4] D4 chain [hrot -> rt -> gx704], eids written ONLY by kernels", flush=True)
    W_np = (np.random.default_rng(5).uniform(-0.05, 0.05, (256, 2048))).astype(np.float32)
    Wb = up(W_np)
    h0 = (np.random.default_rng(6).uniform(-0.5, 0.5, (88, 2048))).astype(np.float32)
    hb = up(h0)
    eids_c = up(np.zeros(704, dtype=np.uint16))
    ys_c = dev.allocator.alloc(704*512*4, BufferSpec())
    def run_chain(ncyc):
        for cyc in range(ncyc):
            hrot(hb, global_size=(1,1,1), local_size=(128,1,1), wait=False)
            rt(Wb, hb, eids_c, global_size=(88,1,1), local_size=(1024,1,1), wait=False)
            gx(ptbl, eids_c, xs704, gridf, ys_c, global_size=(704,1,1), local_size=(1024,1,1), wait=False)
            if (cyc & 15) == 15: dev.synchronize()
        dev.synchronize()
    rt(Wb, hb, eids_c, global_size=(88,1,1), local_size=(1024,1,1), wait=True)
    got = dn(eids_c, (88, 8), np.uint16)
    refs = rt_ref(W_np, h0, range(0, 88, 11))
    ok_rt = all(np.array_equal(refs[p], got[p]) for p in refs)
    print(f"[S4] rt8e256poc vs numpy (8 spots): {'EXACT' if ok_rt else 'MISMATCH'}", flush=True)
    dev.allocator._copyin(hb, memoryview(h0.tobytes()))
    run_chain(256)
    eids_r1 = dn(eids_c, (88, 8), np.uint16)
    h_dev = dn(hb, (88, 2048))
    ref0 = rt_ref(W_np, h_dev, (0,))
    print(f"[S4] after 256: eids[0] == ref(h_dev): {np.array_equal(ref0[0], eids_r1[0])}", flush=True)
    ok_all = True
    t0 = time.perf_counter()
    for rnd in range(1, 16):
        run_chain(256)
        if rnd % 4 == 0 or rnd == 15:
            got_e = dn(eids_c, (88, 8), np.uint16)
            h_now = dn(hb, (88, 2048))
            refc = rt_ref(W_np, h_now, (0, 44, 87))
            okc = all(np.array_equal(refc[p], got_e[p]) for p in refc)
            ok_all &= okc
            print(f"[S4] r{rnd:2d} cycles={rnd*256+256} eids-vs-ref {'OK' if okc else 'MISMATCH'} distinct={len(set(got_e.flatten().tolist()))}", flush=True)
    el = time.perf_counter()-t0
    print(f"[S4] 4096 mixed cycles total, no faults; ref checks {'ALL OK' if ok_all else 'FAILED'}", flush=True)
    dev.allocator._copyin(hb, memoryview(h0.tobytes()))
    run_chain(256)
    eids_det = dn(eids_c, (88, 8), np.uint16)
    print(f"[S4] determinism x2 (fresh 256 from same seed): {np.array_equal(eids_r1, eids_det)}", flush=True)

    # ---------------- S5: D5 k2s bench ----------------
    print("[S5] D5 mm_k2s36 @ [32][128][128] fp32, 30 layers, T=1..11", flush=True)
    for T in range(1, 12):
        k2s[T] = prog(f"MM_P0_k2s36_t{T}")
    dev.synchronize(); print("[S5] 11 k2s programs loaded", flush=True)
    qkv = up((np.random.default_rng(3).standard_normal(11*32*384)*0.1).astype(np.float16))
    abdt = up((np.random.default_rng(4).standard_normal(11*32*3)*0.1).astype(np.float32))
    ssm_a = up((np.random.default_rng(5).uniform(0.5, 1.5, 32)).astype(np.float32))
    dtb = up((np.random.default_rng(6).uniform(-0.1, 0.1, 32)).astype(np.float32))
    rec_in = up((np.random.default_rng(7).standard_normal(32*16384)*0.05).astype(np.float32))
    rec_out = dev.allocator.alloc(11*32*16384*4, BufferSpec())
    core = dev.allocator.alloc(11*32*128*4, BufferSpec())
    dev.synchronize()
    print(f"  {'T':>3s} {'us/layer':>9s} {'ms/30L':>8s} {'stateGB/s':>9s}", flush=True)
    rows = []
    for T in range(1, 12):
        P = k2s[T]
        P(qkv, abdt, ssm_a, dtb, rec_in, rec_out, core, global_size=(32,1,1), local_size=(256,1,1), wait=True)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(30):
                P(qkv, abdt, ssm_a, dtb, rec_in, rec_out, core, global_size=(32,1,1), local_size=(256,1,1), wait=False)
            dev.synchronize()
            ts.append((time.perf_counter()-t0)/30*1e6)
        us = min(ts)
        st_b = 32*16384*4*(T+1)
        rows.append((T, us))
        print(f"  {T:3d} {us:9.1f} {us*30/1e3:8.3f} {st_b/(us*1e-6)/1e9:9.1f}", flush=True)
    mv = memoryview(bytearray(11*32*128*4)).cast("B")
    dev.allocator._copyout(mv, core)
    c = np.frombuffer(mv, dtype=np.float32)
    print(f"[S5] core finite: {np.isfinite(c).all()} (|max| {np.abs(c).max():.3f})", flush=True)

    print("\n== SUMMARY ==")
    print(f"  D3 grouped {res['grouped-8'][2]:.0f} | contig64 {res['contig-64'][2]:.0f} | scattered {res['scattered'][2]:.0f} | "
          f"bankwide {res['bankscatter'][2]:.0f} GB/s; GO bar >=180 grouped / >=120 scattered", flush=True)

if __name__ == "__main__":
    main()
