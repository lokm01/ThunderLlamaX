#!/usr/bin/env python3
"""MM P2 RUN — the one-GPU-process harness (everything in one boot).

Stages (progress file ~/mm_p2_progress.txt; resumable after GPU-EXIT reboots):
  S1  gx8e256dn  (IQ4_XS down lane) A/B vs gxdn4_ref on REAL L00 down bytes
      (experts 0..7, one launch, P=1) -> fp16-bit-exact; det x2.
  S1b gx8e256dn6 (Q6_K down lane)    A/B vs gxdn6_ref on REAL L34 down bytes -> fp16-bit-exact; det.
  S1c gx8e256up4 (IQ4_XS gate+up)    A/B vs gxup4_ref on REAL L39 gu bytes -> allclose (expf ULP); det.
  S2  shexp8 A/B vs shexp_ref on REAL L00 shared tensors (P=1); det x2.
  S2b rmsz2048 A/B vs rmsz_ref (P=4, seeded) -> BIT-EXACT; det.
  S2c mx8e256cmb A/B vs cmb_ref (deterministic inputs) -> fp16-bit-exact; det.
  S3  THE SLICE (layer 0, real everything, P=3):
      eager A/B vs the numpy slice ref -> ids EXACT + fp16-tolerance outputs;
      det x2; in-graph [hrot->rmsz->rt->shexp->gxup->gxdn->cmb] x256 wait-each,
      ids EXACT vs rt ref on the final drifted h, full-output A/B, det x2,
      per-cycle ms + GB/s.
  S3b the exception lanes in-graph: L34 [.. gxup -> gxdn6] x256 and
      L39 [.. gxup4 -> gxdn6(split dn bank)] x256; det; ms.
  S4  trunk families on REAL tensors: gv8k2048 (attn_qkv L0 + DOUBLED attn_q L3),
      gv8k4096 (ssm_out L0 + attn_output L3), gv8k512 (shexp down L0 + attn_k L3),
      embg248 (real embed rows + clamp), h8i2048 (synthetic g128 rows) -- each
      A/B bit-exact vs the numpy ports + det.
"""
import os, sys, time, threading
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
sys.path.insert(0, "~/tinygrad-metal")
BASE = "~/tinygrad-metal"
PROG = os.path.expanduser("~/mm_p2_progress.txt")
P = 3                      # slice positions (the K=2 spec decode shape)

import json
from MM_P2_ports import (dq_q8_0, dq_q6_k, dq_iq4_xs, gxdn4_ref, gxdn6_ref,
    gxup4_ref, gxup_ref, shexp_ref, rmsz_ref, cmb_ref, rt_prod_ref, logits_ref,
    h8i_quant, h8i_ref, q8_dot_ref, load_router, load_wsh, load_norm,
    load_shexp_raw, routed_rows, iq4nl_f32, iq3s_grid_f32, bank_reader, PACK, RB)

def record(tag, result):
    with open(PROG, "a") as f: f.write(f"{tag} {result}\n"); f.flush(); os.fsync(f.fileno())
    print(f"[PROGRESS] {tag} {result}", flush=True)

def done(tag):
    if not os.path.exists(PROG): return False
    return any(l.split()[0] == tag for l in open(PROG).read().splitlines() if l.split())

def naneq(a, b):
    return np.array_equal(a, b) or ((a == b) | (np.isnan(a) & np.isnan(b))).all()

CUBINS = ["gx8e256dn", "gx8e256dn6", "gx8e256up4", "mx8e256cmb", "shexp8",
          "gv8k2048", "gv8k4096", "gv8k512", "rmsz2048", "embg248", "h8i2048"]
LS = {k: ((256,1,1) if k in ("mx8e256cmb", "rmsz2048") else (1024,1,1)) for k in CUBINS}
LS["rt8e256"] = (1024,1,1); LS["gx8e256up"] = (1024,1,1); LS["mm_hrot"] = (128,1,1)

def main():
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram, NVComputeQueue, nv_wait_timeline
    from tinygrad.helpers import round_up
    from tinygrad.uop.ops import UOp
    from tinygrad.dtype import dtypes

    from tinygrad.dtype import dtypes as _dt
    INT_SIG = (None, 4, _dt.int32, ())
    SCALAR_TAIL = {"gv8k2048", "gv8k4096", "gv8k512", "embg248", "h8i2048"}
    def prog(stem, sym):
        lib = open(f"{BASE}/{stem}.cubin", "rb").read()
        return NVProgram(dev, TinyELF(lib=lib, name=sym, target=dev.renderer.target,
                                      signature=(INT_SIG,) if sym in SCALAR_TAIL else tuple()))
    K = {k: prog(f"MM_P2_{k}", k) for k in CUBINS}
    K["rt"] = prog("MM_P1_rt8e256", "rt8e256")
    K["gxup"] = prog("MM_P1_gx8e256up", "gx8e256up")
    K["hrot"] = prog("MM_P0_hrot", "mm_hrot")
    dev.synchronize(); print("[S0] 14 programs loaded", flush=True)

    keep = []
    def up(a):
        a = np.ascontiguousarray(a); keep.append(a)
        b = dev.allocator.alloc(a.nbytes, BufferSpec())
        dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
    def up_big(np_getter, nbytes):
        b = dev.allocator.alloc(nbytes, BufferSpec())
        CH = 64 << 20; off = 0
        while off < nbytes:
            n = min(CH, nbytes - off)
            a = np.ascontiguousarray(np_getter(off, n)); keep.append(a)
            dev.allocator._copyin(b.offset(offset=off, size=n), memoryview(a.data).cast("B"))
            off += n
        return b
    def dn(b, shape, dtype=np.float32):
        n = int(np.prod(shape))
        mv = memoryview(bytearray(int(n)*np.dtype(dtype).itemsize)).cast("B")
        dev.allocator._copyout(mv, b)
        return np.frombuffer(mv, dtype=dtype).reshape(shape).copy()

    gridf = up(iq3s_grid_f32())
    iq4nl = up(iq4nl_f32())

    # ---------------- S1: gx8e256dn (IQ4_XS) on real L00 down bytes ----------------
    if not done("S1"):
        E = 8
        rows = routed_rows(0, "down", list(range(E)))          # [8, 2048, 272] real
        slab = 557056
        sb_np = np.zeros(slab*E, dtype=np.uint8)
        for e in range(E): sb_np[e*slab:(e+1)*slab] = rows[e].reshape(-1)
        sbank = up(sb_np)
        ptbl = up(np.array([sbank.va_addr + e*slab for e in range(E)], dtype=np.uint64))
        act = (np.random.default_rng(31).uniform(-0.5, 0.5, (1, 512))).astype(np.float32)
        actb = up(np.tile(act, (E, 1)))          # PER-PAIR rows (dn is pair-addressed)
        eids8 = up(np.arange(E, dtype=np.uint16))
        parts8 = dev.allocator.alloc(E*2048*2, BufferSpec())
        K["gx8e256dn"](ptbl, eids8, actb, iq4nl, parts8, global_size=(E,1,1), local_size=LS["gx8e256dn"], wait=True)
        got = dn(parts8, (E, 2048), np.float16)
        ok = True
        for e in range(E):
            ref = gxdn4_ref(rows[e], act[0])
            if not naneq(got[e], ref): ok = False; print(f"  S1 e{e} MISMATCH nz={int((got[e]!=ref).sum())}", flush=True)
        K["gx8e256dn"](ptbl, eids8, actb, iq4nl, parts8, global_size=(E,1,1), local_size=LS["gx8e256dn"], wait=True)
        det = naneq(got, dn(parts8, (E, 2048), np.float16))
        nz_tot = int((got.astype(np.float32) == 0).all(axis=1).sum())
        record("S1", f"gx8e256dn vs gxdn4_ref(L00 real) {'FP16-BIT-EXACT' if ok else 'BAD'} det{'OK' if det else 'FAIL'} allzero-rows {nz_tot}/8")
        print(f"[S1] bit-exact={ok} det={det}", flush=True)

    # ---------------- S1b: gx8e256dn6 (Q6_K) on real L34 down bytes ----------------
    if not done("S1b"):
        E = 8
        rows = routed_rows(34, "down", list(range(E)))          # [8, 2048, 420]
        slab = 860160
        sb_np = np.zeros(slab*E, dtype=np.uint8)
        for e in range(E): sb_np[e*slab:(e+1)*slab] = rows[e].reshape(-1)
        sbank = up(sb_np)
        ptbl = up(np.array([sbank.va_addr + e*slab for e in range(E)], dtype=np.uint64))
        act = (np.random.default_rng(32).uniform(-0.5, 0.5, (1, 512))).astype(np.float32)
        actb = up(np.tile(act, (E, 1)))          # PER-PAIR rows
        eids8 = up(np.arange(E, dtype=np.uint16))
        parts8 = dev.allocator.alloc(E*2048*2, BufferSpec())
        K["gx8e256dn6"](ptbl, eids8, actb, parts8, global_size=(E,1,1), local_size=LS["gx8e256dn6"], wait=True)
        got = dn(parts8, (E, 2048), np.float16)
        ok = True
        for e in range(E):
            ref = gxdn6_ref(rows[e], act[0])
            if not naneq(got[e], ref): ok = False; print(f"  S1b e{e} MISMATCH nz={int((got[e]!=ref).sum())}", flush=True)
        K["gx8e256dn6"](ptbl, eids8, actb, parts8, global_size=(E,1,1), local_size=LS["gx8e256dn6"], wait=True)
        det = naneq(got, dn(parts8, (E, 2048), np.float16))
        record("S1b", f"gx8e256dn6 vs gxdn6_ref(L34 real Q6_K) {'FP16-BIT-EXACT' if ok else 'BAD'} det{'OK' if det else 'FAIL'}")
        print(f"[S1b] bit-exact={ok} det={det}", flush=True)

    # ---------------- S1c: gx8e256up4 (IQ4_XS gate+up) on real L39 gu bytes ----------------
    if not done("S1c"):
        E = 8
        rg = routed_rows(39, "gate", list(range(E)))
        ru = routed_rows(39, "up", list(range(E)))
        slab = 1114112
        sb_np = np.zeros(slab*E, dtype=np.uint8)
        for e in range(E):
            sb_np[e*slab:e*slab+557056] = rg[e].reshape(-1)
            sb_np[e*slab+557056:(e+1)*slab] = ru[e].reshape(-1)
        sbank = up(sb_np)
        ptbl = up(np.array([sbank.va_addr + e*slab for e in range(E)], dtype=np.uint64))
        x = (np.random.default_rng(33).uniform(-0.5, 0.5, (1, 2048))).astype(np.float32)
        xb = up(x)
        eids8 = up(np.arange(E, dtype=np.uint16))
        ys8 = dev.allocator.alloc(E*512*4, BufferSpec())
        K["gx8e256up4"](ptbl, eids8, xb, iq4nl, ys8, global_size=(E,1,1), local_size=LS["gx8e256up4"], wait=True)
        got = dn(ys8, (E, 512))
        ok = True
        for e in range(E):
            ref = gxup4_ref(rg[e], ru[e], x[0])
            if not np.allclose(got[e], ref, rtol=1e-5, atol=1e-6): ok = False
            if not np.array_equal(np.isnan(got[e]), np.isnan(ref)): ok = False
        K["gx8e256up4"](ptbl, eids8, xb, iq4nl, ys8, global_size=(E,1,1), local_size=LS["gx8e256up4"], wait=True)
        det = naneq(got, dn(ys8, (E, 512)))
        record("S1c", f"gx8e256up4 vs gxup4_ref(L39 real IQ4_XS) allclose{'OK' if ok else 'BAD'} det{'OK' if det else 'FAIL'}")
        print(f"[S1c] allclose={ok} det={det}", flush=True)

    # ---------------- S2: shexp8 on real L00 shared tensors ----------------
    if not done("S2"):
        wg, wu, wd = load_shexp_raw(0)
        wgb, wub, wdb = up(wg), up(wu), up(wd)
        x = (np.random.default_rng(34).uniform(-0.5, 0.5, (1, 2048))).astype(np.float32)
        xb = up(x)
        yb = dev.allocator.alloc(2048*4, BufferSpec())
        K["shexp8"](wgb, wub, wdb, xb, yb, global_size=(1,1,1), local_size=LS["shexp8"], wait=True)
        got = dn(yb, (2048,))
        ref = shexp_ref(wg, wu, wd, x[0])
        ok = np.allclose(got, ref, rtol=1e-5, atol=1e-6)
        K["shexp8"](wgb, wub, wdb, xb, yb, global_size=(1,1,1), local_size=LS["shexp8"], wait=True)
        det = np.array_equal(got, dn(yb, (2048,)))
        record("S2", f"shexp8 vs shexp_ref(L00 real Q8_0) allclose{'OK' if ok else 'BAD'} det{'OK' if det else 'FAIL'} maxdev {np.abs(got-ref).max():.2e}")
        print(f"[S2] allclose={ok} det={det}", flush=True)

    # ---------------- S2b: rmsz2048 vs rmsz_ref ----------------
    if not done("S2b"):
        x = (np.random.default_rng(35).uniform(-1.0, 1.0, (4, 2048))).astype(np.float32)
        w = load_norm(0)
        xb, wb = up(x), up(w)
        yb = dev.allocator.alloc(4*2048*4, BufferSpec())
        K["rmsz2048"](xb, wb, yb, global_size=(4,1,1), local_size=LS["rmsz2048"], wait=True)
        got = dn(yb, (4, 2048))
        ref = rmsz_ref(x, w)
        ok = np.array_equal(got, ref)
        K["rmsz2048"](xb, wb, yb, global_size=(4,1,1), local_size=LS["rmsz2048"], wait=True)
        det = np.array_equal(got, dn(yb, (4, 2048)))
        record("S2b", f"rmsz2048 vs rmsz_ref {'BIT-EXACT' if ok else f'close maxdev={np.abs(got-ref).max():.2e}'} det{'OK' if det else 'FAIL'}")
        print(f"[S2b] exact={ok} det={det}", flush=True)

    # ---------------- S2c: mx8e256cmb vs cmb_ref ----------------
    if not done("S2c"):
        g = (np.random.default_rng(36).uniform(0.01, 0.3, (4, 8))).astype(np.float32)
        g = (g / g.sum(axis=1, keepdims=True)).astype(np.float32)
        sg = (np.random.default_rng(37).uniform(0.1, 0.9, 4)).astype(np.float32)
        pt = (np.random.default_rng(38).uniform(-2, 2, (4, 8, 2048))).astype(np.float16)
        sh = (np.random.default_rng(39).uniform(-1, 1, (4, 2048))).astype(np.float32)
        gb, sgb, ptb, shb = up(g), up(sg), up(pt), up(sh)
        yb = dev.allocator.alloc(4*2048*2, BufferSpec())
        K["mx8e256cmb"](ptb, gb, sgb, shb, yb, global_size=(4,1,1), local_size=LS["mx8e256cmb"], wait=True)
        got = dn(yb, (4, 2048), np.float16)
        ref = cmb_ref(g, sg, pt, sh)
        ok = naneq(got, ref)
        K["mx8e256cmb"](ptb, gb, sgb, shb, yb, global_size=(4,1,1), local_size=LS["mx8e256cmb"], wait=True)
        det = naneq(got, dn(yb, (4, 2048), np.float16))
        record("S2c", f"mx8e256cmb vs cmb_ref {'FP16-BIT-EXACT' if ok else 'BAD'} det{'OK' if det else 'FAIL'}")
        print(f"[S2c] bit-exact={ok} det={det}", flush=True)

    # ---------------- S3: THE SLICE (layer 0, real everything, P=3) ----------------
    man = bank_reader()
    SLAB00, DOWN_OFF, SLAB34, SLAB39GU, SLAB39DN = 1458176, 901120, 1761280, 1114112, 860160

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

    def upload_bank(fname, nbytes):
        f = open(os.path.join(PACK, fname), "rb")
        def getter(off, n):
            f.seek(off); return np.frombuffer(f.read(n), dtype=np.uint8)
        return up_big(getter, nbytes)

    def slice_np(h3, Wrt, wsh, wn, layer, rows_cache):
        """the FULL numpy slice reference on h [P][2048] (fp16-tolerance class)."""
        hn = rmsz_ref(h3, wn)
        refs = rt_prod_ref(Wrt, wsh, hn, range(h3.shape[0]))
        act = np.empty((h3.shape[0]*8, 512), dtype=np.float32)
        parts = np.empty((h3.shape[0]*8, 2048), dtype=np.float16)
        shared = np.empty((h3.shape[0], 2048), dtype=np.float32)
        for p in range(h3.shape[0]):
            ids = refs[p][0]
            for r in range(8):
                e = int(ids[r]); pr = p*8 + r
                rg, ru, rd = rows_cache[e]
                act[pr] = gxup_ref(rg, ru, hn[p])
                parts[pr] = gxdn4_ref(rd, act[pr])
            shared[p] = shexp_ref(rows_cache["wg"], rows_cache["wu"], rows_cache["wd"], hn[p])
        gates = np.stack([refs[p][1] for p in range(h3.shape[0])])
        sg = np.array([refs[p][2] for p in range(h3.shape[0])])
        y = cmb_ref(gates, sg, parts.reshape(h3.shape[0], 8, 2048), shared)
        return hn, refs, gates, sg, act, parts, shared, y

    if not done("S3"):
        t0 = time.time()
        bank00 = upload_bank("routed/L00.bank", SLAB00*256)
        ptbl_up00  = up(np.array([bank00.va_addr + e*SLAB00 for e in range(256)], dtype=np.uint64))
        ptbl_dn00  = up(np.array([bank00.va_addr + e*SLAB00 + DOWN_OFF for e in range(256)], dtype=np.uint64))
        Wrt, wsh, wn = load_router(0), load_wsh(0), load_norm(0)
        Wrtb, wshb, wnb = up(Wrt), up(wsh), up(wn)
        wg, wu, wd = load_shexp_raw(0)
        wgb, wub, wdb = up(wg), up(wu), up(wd)
        # real expert rows cache for the ref (experts actually routed + a fixed set)
        # h lives at [88][2048]: mm_hrot (P1) writes ALL 88 rows -- a [P][2048]
        # buffer gets 680KB stamped OOB per replay (the S3g fault root cause)
        h0 = (np.random.default_rng(40).uniform(-0.5, 0.5, (88, 2048))).astype(np.float32)
        hb = up(h0)
        hnb = dev.allocator.alloc(P*2048*4, BufferSpec())
        eidsb = dev.allocator.alloc(P*8*2, BufferSpec())
        gatesb = dev.allocator.alloc(P*8*4, BufferSpec())
        sgb = dev.allocator.alloc(P*4, BufferSpec())
        actb = dev.allocator.alloc(P*8*512*4, BufferSpec())
        partsb = dev.allocator.alloc(P*8*2048*2, BufferSpec())
        shb = dev.allocator.alloc(P*2048*4, BufferSpec())
        yb = dev.allocator.alloc(P*2048*2, BufferSpec())
        print(f"[S3] bank+trunk up in {time.time()-t0:.0f}s", flush=True)

        def run_eager():
            K["rmsz2048"](hb, wnb, hnb, global_size=(P,1,1), local_size=LS["rmsz2048"], wait=True)
            K["rt"](Wrtb, wshb, hnb, eidsb, gatesb, sgb, global_size=(P,1,1), local_size=LS["rt8e256"], wait=True)
            K["shexp8"](wgb, wub, wdb, hnb, shb, global_size=(P,1,1), local_size=LS["shexp8"], wait=True)
            K["gxup"](ptbl_up00, eidsb, hnb, gridf, actb, global_size=(P*8,1,1), local_size=LS["gx8e256up"], wait=True)
            K["gx8e256dn"](ptbl_dn00, eidsb, actb, iq4nl, partsb, global_size=(P*8,1,1), local_size=LS["gx8e256dn"], wait=True)
            K["mx8e256cmb"](partsb, gatesb, sgb, shb, yb, global_size=(P,1,1), local_size=LS["mx8e256cmb"], wait=True)
        run_eager()
        hn_g = dn(hnb, (P, 2048)); ids_g = dn(eidsb, (P, 8), np.uint16)
        gates_g = dn(gatesb, (P, 8)); sg_g = dn(sgb, (P,))
        act_g = dn(actb, (P*8, 512)); parts_g = dn(partsb, (P*8, 2048), np.float16)
        sh_g = dn(shb, (P, 2048)); y_g = dn(yb, (P, 2048), np.float16)
        # numpy ref (needs the routed experts' real rows)
        all_e = sorted(set(int(e) for e in ids_g.reshape(-1)))
        rows_cache = {}
        rg_all = routed_rows(0, "gate", all_e); ru_all = routed_rows(0, "up", all_e); rd_all = routed_rows(0, "down", all_e)
        for i, e in enumerate(all_e):
            rows_cache[e] = (rg_all[i], ru_all[i], rd_all[i])
        rows_cache["wg"], rows_cache["wu"], rows_cache["wd"] = wg, wu, wd
        hn_r, refs, gates_r, sg_r, act_r, parts_r, sh_r, y_r = slice_np(np.ascontiguousarray(h0[:P]), Wrt, wsh, wn, 0, rows_cache)
        ids_ok = all(np.array_equal(refs[p][0], ids_g[p]) for p in range(P))
        hn_ok = np.array_equal(hn_g, hn_r)
        gates_ok = np.allclose(gates_g, gates_r, rtol=1e-5, atol=1e-7)
        sg_ok = np.allclose(sg_g, sg_r, rtol=1e-5, atol=1e-7)
        act_ok = np.allclose(act_g, act_r, rtol=1e-4, atol=1e-5)
        sh_ok = np.allclose(sh_g, sh_r, rtol=1e-5, atol=1e-6)
        parts_ok = np.allclose(parts_g.astype(np.float32), parts_r.astype(np.float32), rtol=2e-3, atol=2e-3)
        y_ok = np.allclose(y_g.astype(np.float32), y_r.astype(np.float32), rtol=2e-3, atol=2e-3)
        if not (parts_ok and y_ok):
            pv = parts_g.astype(np.float32); rv = parts_r.astype(np.float32)
            dd = np.abs(pv - rv); rel = dd / np.maximum(np.abs(rv), 1e-6)
            nz = int((pv != rv).sum())
            wi = np.unravel_index(np.argmax(dd), dd.shape)
            print(f"[DIAG] parts: nz={nz}/{pv.size} maxabs={dd.max():.3e} maxrel={rel.max():.3e} worst(pair={wi[0]},row={wi[1]}) got={pv[wi]:.6f} ref={rv[wi]:.6f}", flush=True)
            yv = y_g.astype(np.float32); yr2 = y_r.astype(np.float32)
            print(f"[DIAG] y: nz={int((yv!=yr2).sum())}/{yv.size} maxabs={np.abs(yv-yr2).max():.3e}", flush=True)
            print(f"[DIAG] parts mismatches per pair: {(pv != rv).sum(axis=1).tolist()[:24]}", flush=True)
            print(f"[DIAG] act dev: maxabs={np.abs(act_g-act_r).max():.3e} maxrel={(np.abs(act_g-act_r)/np.maximum(np.abs(act_r),1e-6)).max():.3e}", flush=True)
        run_eager()
        det = (np.array_equal(ids_g, dn(eidsb, (P,8), np.uint16)) and
               naneq(y_g, dn(yb, (P, 2048), np.float16)) and
               naneq(parts_g, dn(partsb, (P*8, 2048), np.float16)) and
               np.array_equal(sh_g, dn(shb, (P, 2048))))
        record("S3e", f"slice EAGER: ids{'EXACT' if ids_ok else 'BAD'} hn{'EXACT' if hn_ok else 'BAD'} gates{'OK' if gates_ok else 'BAD'} "
                       f"sg{'OK' if sg_ok else 'BAD'} act{'OK' if act_ok else 'BAD'} shared{'OK' if sh_ok else 'BAD'} "
                       f"parts{'OK' if parts_ok else 'BAD'} y{'OK' if y_ok else 'BAD'} det{'OK' if det else 'FAIL'}")
        print(f"[S3e] ids={ids_ok} hn={hn_ok} gates={gates_ok} sg={sg_ok} act={act_ok} sh={sh_ok} parts={parts_ok} y={y_ok} det={det}", flush=True)

        # in-graph x256 (drift h each replay via hrot; wait-each law)
        seq = [(K["hrot"], (hb,), 1),
               (K["rmsz2048"], (hb, wnb, hnb), P),
               (K["rt"], (Wrtb, wshb, hnb, eidsb, gatesb, sgb), P),
               (K["shexp8"], (wgb, wub, wdb, hnb, shb), P),
               (K["gxup"], (ptbl_up00, eidsb, hnb, gridf, actb), P*8),
               (K["gx8e256dn"], (ptbl_dn00, eidsb, actb, iq4nl, partsb), P*8),
               (K["mx8e256cmb"], (partsb, gatesb, sgb, shb, yb), P)]
        c0 = MG(seq, "s3a"); c1 = MG(seq, "s3b")
        def run256(timed=False):
            prev = dev.timeline_value - 1     # PREV BEFORE NEXT (the off-by-one law)
            blocks = []; t0 = time.perf_counter()
            for i in range(256):
                v = dev.next_timeline()
                (c0 if (i & 1) == 0 else c1).submit(prev, v)
                nv_wait_timeline(dev, v, what="S3", timeout_s=30.0)
                prev = v
                if (i & 63) == 63:
                    blocks.append((time.perf_counter()-t0)*1e3); t0 = time.perf_counter()
                    print(f"[S3g] replay {i+1}/256 blk {blocks[-1]:.1f}ms", flush=True)
            dev.synchronize()
            return blocks
        blocks = run256(timed=True)
        h_dr = dn(hb, (88, 2048))[:P]; ids_dr = dn(eidsb, (P, 8), np.uint16)
        y_dr = dn(yb, (P, 2048), np.float16)
        refs_dr = rt_prod_ref(Wrt, wsh, rmsz_ref(h_dr, wn), range(P))
        ids_ok2 = all(np.array_equal(refs_dr[p][0], ids_dr[p]) for p in range(P))
        all_e2 = sorted(set(int(e) for e in ids_dr.reshape(-1)))
        rc2 = {}
        rg2 = routed_rows(0, "gate", all_e2); ru2 = routed_rows(0, "up", all_e2); rd2 = routed_rows(0, "down", all_e2)
        for i, e in enumerate(all_e2): rc2[e] = (rg2[i], ru2[i], rd2[i])
        rc2["wg"], rc2["wu"], rc2["wd"] = wg, wu, wd
        _, _, _, _, _, _, _, y_r2 = slice_np(h_dr, Wrt, wsh, wn, 0, rc2)
        y_ok2 = np.allclose(y_dr.astype(np.float32), y_r2.astype(np.float32), rtol=2e-3, atol=2e-3)
        dev.allocator._copyin(hb, memoryview(h0.tobytes()))
        run256()
        det2 = (np.array_equal(ids_dr, dn(eidsb, (P,8), np.uint16)) and
                naneq(y_dr, dn(yb, (P, 2048), np.float16)))
        med = sorted(blocks)[len(blocks)//2]
        bts = P*((256*2048*4) + (2*512*2176 + 2048*544)) + P*8*(2*450560 + 557056)
        gbs = (64*bts)/(med/1e3)/1e9
        record("S3g", f"slice GRAPH x256 wait-each: ids-vs-drifted-h{'EXACT' if ids_ok2 else 'BAD'} y{'OK' if y_ok2 else 'BAD'} "
                       f"det{'OK' if det2 else 'FAIL'} | {med*1e3/64:.0f} us/cycle = {med/64:.3f} ms/layer-cycle -> {gbs:.0f} GB/s ({bts/1e6:.1f} MB/cyc)")
        print(f"[S3g] ids={ids_ok2} y={y_ok2} det={det2} {med/64:.3f} ms/cyc {gbs:.0f} GB/s", flush=True)
        dev.allocator._copyin(hb, memoryview(h0.tobytes()))

    # ---------------- S3b: exception lanes in-graph (L34 Q6_K, L39 IQ4_XS+split) ----------------
    if not done("S3b"):
        bank34 = upload_bank("routed/L34.bank", SLAB34*256)
        ptbl_up34 = up(np.array([bank34.va_addr + e*SLAB34 for e in range(256)], dtype=np.uint64))
        ptbl_dn34 = up(np.array([bank34.va_addr + e*SLAB34 + DOWN_OFF for e in range(256)], dtype=np.uint64))
        bank39gu = upload_bank("routed/L39gu.bank", SLAB39GU*256)
        bank39dn = upload_bank("routed/L39dn.bank", SLAB39DN*256)
        ptbl_up39 = up(np.array([bank39gu.va_addr + e*SLAB39GU for e in range(256)], dtype=np.uint64))
        ptbl_dn39 = up(np.array([bank39dn.va_addr + e*SLAB39DN for e in range(256)], dtype=np.uint64))
        Wrt34, wsh34 = load_router(34), load_wsh(34)
        Wrt39, wsh39 = load_router(39), load_wsh(39)
        hb2 = up((np.random.default_rng(41).uniform(-0.5, 0.5, (P, 2048))).astype(np.float32))
        hnb2 = dev.allocator.alloc(P*2048*4, BufferSpec())
        eids2 = dev.allocator.alloc(P*8*2, BufferSpec())
        gates2 = dev.allocator.alloc(P*8*4, BufferSpec())
        sg2 = dev.allocator.alloc(P*4, BufferSpec())
        act2 = dev.allocator.alloc(P*8*512*4, BufferSpec())
        parts2 = dev.allocator.alloc(P*8*2048*2, BufferSpec())
        yb2 = dev.allocator.alloc(P*2048*2, BufferSpec())
        wn34b = up(load_norm(34)); wn39b = up(load_norm(39))
        def mkchain(wrtb, wshb, wnb_, ptbl_up, upk, ptbl_dn, dnk):
            return [(K["rmsz2048"], (hb2, wnb_, hnb2), P),
                    (K["rt"], (wrtb, wshb, hnb2, eids2, gates2, sg2), P),
                    (upk, (ptbl_up, eids2, hnb2, gridf, act2) if upk.name != "gx8e256up4" else (ptbl_up, eids2, hnb2, iq4nl, act2), P*8),
                    (dnk, (ptbl_dn, eids2, act2, parts2) if dnk.name != "gx8e256dn" else (ptbl_dn, eids2, act2, iq4nl, parts2), P*8)]
        W34b, wsh34b = up(Wrt34), up(wsh34)
        W39b, wsh39b = up(Wrt39), up(wsh39)
        ch34 = mkchain(W34b, wsh34b, wn34b, ptbl_up34, K["gxup"], ptbl_dn34, K["gx8e256dn6"])
        ch39 = mkchain(W39b, wsh39b, wn39b, ptbl_up39, K["gx8e256up4"], ptbl_dn39, K["gx8e256dn6"])
        g34a, g34b = MG(ch34, "l34a"), MG(ch34, "l34b")
        g39a, g39b = MG(ch39, "l39a"), MG(ch39, "l39b")
        def run2(ga, gb):
            prev = dev.timeline_value - 1
            t0 = time.perf_counter()
            for i in range(256):
                v = dev.next_timeline()
                (ga if (i & 1) == 0 else gb).submit(prev, v)
                nv_wait_timeline(dev, v, what="S3b", timeout_s=30.0)
                prev = v
            dev.synchronize()
            return (time.perf_counter()-t0)*1e3/256
        ms34 = run2(g34a, g34b)
        p1 = dn(parts2, (P*8, 2048), np.float16); e1 = dn(eids2, (P,8), np.uint16).copy()
        ms39 = run2(g39a, g39b)
        ms34b = run2(g34a, g34b)
        det34 = np.array_equal(e1, dn(eids2, (P,8), np.uint16)) and naneq(p1, dn(parts2, (P*8, 2048), np.float16))
        record("S3b", f"exception lanes x256 wait-each CLEAN: L34[iq3s->q6k] {ms34:.2f} ms/cyc det{'OK' if det34 else 'FAIL'} | L39[iq4xs->q6k split banks] {ms39:.2f} ms/cyc")
        print(f"[S3b] L34 {ms34:.2f} ms | L39 {ms39:.2f} ms (in-graph, no wedge)", flush=True)

    # ---------------- S4: trunk families on real tensors ----------------
    if not done("S4"):
        res = []
        rng = np.random.default_rng(50)
        x2048 = rng.uniform(-0.3, 0.3, 2048).astype(np.float32)
        x4096 = rng.uniform(-0.3, 0.3, 4096).astype(np.float32)
        x512  = rng.uniform(-0.3, 0.3, 512).astype(np.float32)
        def gv8_gate(sym, path, rows, kd, x, nsamp=24):
            raw = np.fromfile(path, dtype=np.uint8)
            rowb = (kd//32)*34
            assert raw.size == rows*rowb, (path, raw.size, rows*rowb)
            wb = up_big(lambda off, n: raw[off:off+n], raw.size)
            xb = up(x)
            yb = dev.allocator.alloc(rows*4, BufferSpec())
            K[sym](wb, xb, yb, vals=(rows,), global_size=(rows//32,1,1), local_size=LS[sym], wait=True)
            got = dn(yb, (rows,))
            sel = np.linspace(0, rows-1, nsamp).astype(int)
            ref = q8_dot_ref(raw.reshape(rows, rowb)[sel], x, kd)
            ok = np.array_equal(got[sel], ref)
            K[sym](wb, xb, yb, vals=(rows,), global_size=(rows//32,1,1), local_size=LS[sym], wait=True)
            det = np.array_equal(got, dn(yb, (rows,)))
            res.append(f"{sym}[{os.path.basename(path)[:26]}] {'EXACT' if ok else 'BAD'} det{'OK' if det else 'FAIL'}")
            return ok, det
        gv8_gate("gv8k2048", f"{PACK}/trunk/blk_0_attn_qkv_weight.bin", 8192, 2048, x2048)
        gv8_gate("gv8k2048", f"{PACK}/trunk/blk_3_attn_q_weight.bin", 8192, 2048, x2048)   # DOUBLED Q+gate
        gv8_gate("gv8k4096", f"{PACK}/trunk/blk_0_ssm_out_weight.bin", 2048, 4096, x4096)
        gv8_gate("gv8k4096", f"{PACK}/trunk/blk_3_attn_output_weight.bin", 2048, 4096, x4096)
        gv8_gate("gv8k512",  f"{PACK}/trunk/blk_0_ffn_down_shexp_weight.bin", 2048, 512, x512)
        gv8_gate("gv8k2048", f"{PACK}/trunk/blk_3_attn_k_weight.bin", 512, 2048, x2048)   # k/v: ne0=2048
        # embg248: real embed rows + clamp
        emb_path = os.path.join(PACK, "trunk", "token_embd_weight.bin")
        emb_sz = 248320*2176
        embbuf = upload_bank("trunk/token_embd_weight.bin", emb_sz)   # the REAL 540MB table
        ids_t = np.array([0, 1, 12345, 248319, 300000, 999999, -7, 4242], dtype=np.int32)
        idsb = up(ids_t); yeb = dev.allocator.alloc(8*2048*4, BufferSpec())
        K["embg248"](embbuf, idsb, yeb, vals=(8,), global_size=(1,1,1), local_size=LS["embg248"], wait=True)
        got_e = dn(yeb, (8, 2048))
        ef = open(emb_path, "rb"); ok_e = True
        for i in ids_t:
            ii = min(max(int(i), 0), 248319)
            ef.seek(ii*2176)
            ref_e = dq_q8_0(np.frombuffer(ef.read(2176), dtype=np.uint8).reshape(1, 2176), 2048)[0]
            k = int(np.where(ids_t == i)[0][0])
            if not np.array_equal(got_e[k], ref_e): ok_e = False
        K["embg248"](embbuf, idsb, yeb, vals=(8,), global_size=(1,1,1), local_size=LS["embg248"], wait=True)
        det_e = np.array_equal(got_e, dn(yeb, (8, 2048)))
        res.append(f"embg248[REAL 540MB embed, ids 0/1/12345/248319/300000/999999/-7/4242 + clamp] {'EXACT' if ok_e else 'BAD'} det{'OK' if det_e else 'FAIL'}")
        # h8i2048: synthetic g128 rows + det
        wrows = (np.random.default_rng(51).uniform(-0.05, 0.05, (256, 2048))).astype(np.float32)
        q8, s8 = h8i_quant(wrows)
        q8b, s8b = up(q8), up(s8)
        xh = x2048
        xhb = up(xh)
        yhb = dev.allocator.alloc(256*4, BufferSpec())
        K["h8i2048"](q8b, s8b, xhb, yhb, vals=(256,), global_size=(8,1,1), local_size=LS["h8i2048"], wait=True)
        got_h = dn(yhb, (256,))
        ref_h = h8i_ref(q8, s8, xh)
        ok_h = np.array_equal(got_h, ref_h)
        K["h8i2048"](q8b, s8b, xhb, yhb, vals=(256,), global_size=(8,1,1), local_size=LS["h8i2048"], wait=True)
        det_h = np.array_equal(got_h, dn(yhb, (256,)))
        res.append(f"h8i2048[int8-g128 synthetic 256r] {'EXACT' if ok_h else 'BAD'} det{'OK' if det_h else 'FAIL'}")
        record("S4", " | ".join(res))
        print("[S4] " + " | ".join(res), flush=True)

    print("[ALL DONE]", flush=True)

if __name__ == "__main__":
    def _wd():
        time.sleep(3000); print("[WATCHDOG] deadline", flush=True); os._exit(4)
    threading.Thread(target=_wd, daemon=True).start()
    main()
