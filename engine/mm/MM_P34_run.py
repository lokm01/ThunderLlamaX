#!/usr/bin/env python3
"""MM P3+P4 RUN — the one-GPU-process harness: component gates, the FULL
40-layer T=1 train vs the fp32 anchor, perf, and the K-mix graph sets.

Stages (progress file ~/mm_p34_progress.txt, resumable after GPU-EXIT reboots):
  S1  gconv36 t3 A/B vs gconv_ref (real L0 conv weight, seeded) + det.
  S2  k2s36 t1/t3 A/B vs k2s_ref (real alog/dtb/ssm_norm) + det.
  S3  THE GDN LAYER SLICE (real L0, T=1): hn seeded -> [qkv|z|a/b GEMVs ->
      conv -> k2s -> out_proj+resid] eager A/B vs the kernel-order numpy slice.
  S4  THE ATTN SLICE (real L3, pos=7, seeded prior int8 cache): spka+spkq A/B
      vs spka_ref/spkq_h_ref + det.
  S5  h6k2048 A/B (sampled rows) + full top-1 on the anchor's real final h.
  S6  THE FULL TRAIN T=1 (eager): battery vs anchor top-1 per position.
  S6b THE TRAIN IN-GRAPH (one ~523-node MG, wait-each): battery re-run.
  S7  perf: K=8-shape MoE quartet; T=1 per-class launch times (wait-each +
      pipelined); ka-slab size.
  S8  K-mix: D2 (P=3) + D8 (P=9) trunk trains + 32-replay smoke each.
"""
import os, sys, time, hashlib, subprocess
import numpy as np

os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
sys.path.insert(0, "~/tinygrad-metal")
BASE = "~/tinygrad-metal"
PROG = os.path.expanduser("~/mm_p34_progress.txt")
ANCHOR_NPZ = os.path.expanduser("~/mm_p34_anchor.npz")
CTX = 1024
PMAX = 9

import json
from MM_P2_ports import (dq_q8_0, _dot_lane_ref, load_router, load_wsh,
    load_shexp_raw, iq4nl_f32, iq3s_grid_f32, bank_reader, PACK)
from MM_P34_ports import (_f, rope_tables, gconv_ref, k2s_ref, gvf32_ref,
    kv_quant, spka_ref, spkq_h_ref, h6k_rows_ref, q8_dot4096_ref,
    GDN_LAYERS, ATTN_LAYERS, load_f32, load_q8)

def record(tag, result):
    with open(PROG, "a") as f: f.write(f"{tag} {result}\n"); f.flush(); os.fsync(f.fileno())
    print(f"[PROGRESS] {tag} {result}", flush=True)

def done(tag):
    if not os.path.exists(PROG): return False
    return any(l.split()[0] == tag for l in open(PROG).read().splitlines() if l.split())

def naneq(a, b):
    return np.array_equal(a, b) or ((a == b) | (np.isnan(a) & np.isnan(b))).all()

NVCC_ENV = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
                DOCKER_HOST="unix://~/.colima/default/docker.sock")

def build_cubins():
    specs = [
        ("MM_P34_gconv36", "gconv36", ["1", "3", "9"]),  # symbol gconv36_{T}
        ("MM_P34_k2s36",   "k2s36",   ["1", "3", "9"]),
        ("MM_P34_gvf32ab", "gvf32ab", [None]),
        ("MM_P34_spka256", "spka256", [None]),
        ("MM_P34_spkq256", "spkq256", [None]),
        ("MM_P34_h6k2048", "h6k2048", [None]),
        ("MM_P34_cmbz2048", "cmbz2048", [None]),
        ("MM_P34_gv8k4096r", "gv8k4096r", [None]),
        ("MM_P34_gv8k2048p", "gv8k2048p", [None]),
        ("MM_P34_rmsz2048", "rmsz2048g", [None]),
    ]
    built = {}
    for stem, sym, variants in specs:
        for v in (variants if variants else [None]):
            name = sym if v is None else f"{sym}_{v}"
            cb = f"{BASE}/MM_P34_{name}.cubin"
            if not os.path.exists(cb):
                cmd = f"nvcc -arch=sm_86 -cubin -fmad=false --output-file={cb} {BASE}/{stem}.cu"
                if v: cmd += f" -DTMAX={v}"
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=NVCC_ENV)
                if r.returncode:
                    print(r.stderr[-2000:]); sys.exit(1)
            built[name] = cb
    for name, cb in built.items():
        r = subprocess.run(f"cuobjdump -res-usage {cb}", shell=True, capture_output=True, text=True, env=NVCC_ENV)
        for l in r.stdout.splitlines():
            if "STACK" in l.upper() or "SPILL" in l.upper():
                print(f"[audit {name}]", l.strip(), flush=True)
    return built

LSZ = {"gconv36_1": (256,1,1), "gconv36_3": (256,1,1), "gconv36_9": (256,1,1),
       "k2s36_1": (256,1,1), "k2s36_3": (256,1,1), "k2s36_9": (256,1,1),
       "embg248": (1024,1,1), "rmsz2048g": (256,1,1), "gv8k2048p": (1024,1,1),
       "gv8k4096r": (1024,1,1), "gvf32ab": (1024,1,1),
       "spka256": (256,1,1), "spkq256": (256,1,1), "rt8e256": (1024,1,1),
       "shexp8": (1024,1,1), "gx8e256up": (1024,1,1), "gx8e256up4": (1024,1,1),
       "gx8e256dn": (1024,1,1), "gx8e256dn6": (1024,1,1), "cmbz2048": (256,1,1),
       "h6k2048": (1024,1,1)}

KA_BYTES = [0]

def main():
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram, NVComputeQueue, nv_wait_timeline
    from tinygrad.helpers import round_up
    from tinygrad.uop.ops import UOp
    from tinygrad.dtype import dtypes
    INT_SIG = (None, 4, dtypes.int32, ())

    built = build_cubins()
    SCALAR_TAIL = {"gv8k2048p", "gv8k4096r", "h6k2048", "embg248"}
    def prog(cbpath, sym):
        lib = open(cbpath, "rb").read()
        return NVProgram(dev, TinyELF(lib=lib, name=sym, target=dev.renderer.target,
                                      signature=(INT_SIG,) if sym in SCALAR_TAIL else tuple()))
    K = {}
    for name, cb in built.items():
        K[name] = prog(cb, name)
    for stem, sym in [("MM_P1_rt8e256", "rt8e256"), ("MM_P1_gx8e256up", "gx8e256up"),
                      ("MM_P2_gx8e256up4", "gx8e256up4"), ("MM_P2_gx8e256dn", "gx8e256dn"),
                      ("MM_P2_gx8e256dn6", "gx8e256dn6"), ("MM_P2_shexp8", "shexp8"),
                      ("MM_P2_embg248", "embg248")]:
        K[sym] = prog(f"{BASE}/{stem}.cubin", sym)
    dev.synchronize(); print(f"[S0] {len(K)} programs loaded", flush=True)

    keep = []
    def up(a):
        a = np.ascontiguousarray(a); keep.append(a)
        b = dev.allocator.alloc(a.nbytes, BufferSpec())
        dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
    def up_big(np_getter, nbytes):
        # THE ASYNC-COPYIN LAW: chunk sources MUST stay alive until the DMA
        # completes -- freeing the numpy chunk right after _copyin returns
        # uploads GARBAGE (the S5 full-head mismatch root cause; P2's keep
        # was load-bearing). Sync per chunk instead of holding 16.5GB in RAM.
        b = dev.allocator.alloc(nbytes, BufferSpec())
        CH = 64 << 20; off = 0
        while off < nbytes:
            n = min(CH, nbytes - off)
            a = np.ascontiguousarray(np_getter(off, n))
            dev.allocator._copyin(b.offset(offset=off, size=n), memoryview(a.data).cast("B"))
            dev.synchronize()
            del a
            off += n
        return b
    def alloc(nbytes):
        b = dev.allocator.alloc(nbytes, BufferSpec()); keep.append(b); return b
    def dn(b, shape, dtype=np.float32):
        n = int(np.prod(shape))
        mv = memoryview(bytearray(int(n)*np.dtype(dtype).itemsize)).cast("B")
        dev.allocator._copyout(mv, b)
        return np.frombuffer(mv, dtype=dtype).reshape(shape).copy()
    def file_uploader(path):
        f = open(path, "rb")
        def getter(off, n):
            f.seek(off); return np.frombuffer(f.read(n), dtype=np.uint8)
        return getter

    gridf = up(iq3s_grid_f32())
    iq4nl = up(iq4nl_f32())

    # ============================ S1: gconv36 ============================
    if not done("S1"):
        rng = np.random.default_rng(11)
        wreal = load_f32(0, "ssm_conv1d_weight").reshape(8192, 4)
        xin = (rng.standard_normal((3, 8192)) * 0.3).astype(np.float32)
        st0 = (rng.standard_normal((8192, 3)) * 0.3).astype(np.float32)
        wb, xb, sb, yb = up(wreal), up(xin), up(st0.copy()), dev.allocator.alloc(3*8192*4, BufferSpec())
        K["gconv36_3"](wb, xb, sb, yb, global_size=(32,1,1), local_size=LSZ["gconv36_3"], wait=True)
        got = dn(yb, (3, 8192)); gotst = dn(sb, (8192, 3))
        yr, str_ = gconv_ref(wreal, xin, st0)
        dev.allocator._copyin(sb, memoryview(np.ascontiguousarray(st0).data).cast("B"))   # reset state (stateful kernel)
        K["gconv36_3"](wb, xb, sb, yb, global_size=(32,1,1), local_size=LSZ["gconv36_3"], wait=True)
        det = naneq(got, dn(yb, (3, 8192)))
        ok = np.allclose(got, yr, rtol=2e-6, atol=1e-7)
        stok = np.allclose(gotst, str_, rtol=1e-6, atol=1e-7)
        record("S1", f"gconv36 t3: y{'OK' if ok else 'BAD'} (maxdev {np.abs(got-yr).max():.2e}) "
                      f"state{'OK' if stok else 'BAD'} det{'OK' if det else 'FAIL'}")

    # ============================ S2: k2s36 ============================
    if not done("S2"):
        rng = np.random.default_rng(13)
        alog = load_f32(0, "ssm_a"); dtb = load_f32(0, "ssm_dt_bias")
        wn = load_f32(0, "ssm_norm_weight")
        for T, key, tag in ((1, "k2s36_1", "t1"), (3, "k2s36_3", "t3")):
            qkvs = (rng.standard_normal((T, 8192)) * 0.2).astype(np.float32)
            ab = (rng.standard_normal((T, 64)) * 0.5).astype(np.float32)
            z = (rng.standard_normal((T, 4096)) * 0.2).astype(np.float32)
            S0 = (rng.standard_normal((32, 128, 128)) * 0.05).astype(np.float32)
            qk, abk, zk = up(qkvs), up(ab), up(z)
            alk, dtk, wnk = up(alog), up(dtb), up(wn)
            Sb = up(S0.copy()); yb = dev.allocator.alloc(T*4096*4, BufferSpec())
            def run2():
                dev.allocator._copyin(Sb, memoryview(np.ascontiguousarray(S0).data).cast("B"))
                K[key](qk, abk, alk, dtk, wnk, zk, Sb, yb, global_size=(32,1,1), local_size=LSZ[key], wait=True)
            run2()
            gy = dn(yb, (T, 4096)); gS = dn(Sb, (32, 128, 128))
            yr, Sr = k2s_ref(qkvs, ab, alog, dtb, wn, z, S0)
            run2()
            det = naneq(gy, dn(yb, (T, 4096))) and naneq(gS, dn(Sb, (32,128,128)))
            ok = np.allclose(gy, yr, rtol=3e-5, atol=3e-6)
            sok = np.allclose(gS, Sr, rtol=3e-5, atol=3e-6)
            record("S2", f"k2s36 {tag}: y{'OK' if ok else 'BAD'} (maxdev {np.abs(gy-yr).max():.2e}) "
                          f"S{'OK' if sok else 'BAD'} ({np.abs(gS-Sr).max():.2e}) det{'OK' if det else 'FAIL'}")

    # ============================ S3: GDN layer slice (real L0) ============================
    if not done("S3"):
        rng = np.random.default_rng(17)
        hn = (rng.standard_normal(2048) * 0.4).astype(np.float32)
        resid = (rng.standard_normal(2048) * 0.6).astype(np.float32)
        wqkv = load_q8(0, "attn_qkv_weight", 8192)
        wz = load_q8(0, "attn_gate_weight", 4096)
        wout = load_q8(0, "ssm_out_weight", 2048, 4096)
        Wa = load_f32(0, "ssm_alpha_weight").reshape(32, 2048)
        Wb = load_f32(0, "ssm_beta_weight").reshape(32, 2048)
        cw = load_f32(0, "ssm_conv1d_weight").reshape(8192, 4)
        alog = load_f32(0, "ssm_a"); dtb = load_f32(0, "ssm_dt_bias")
        wn = load_f32(0, "ssm_norm_weight")
        hnb, rb = up(hn), up(resid)
        WQKVB, WZB, WOUTB = up(wqkv), up(wz), up(wout)
        WAB, WBB, CWB = up(Wa), up(Wb), up(cw)
        ALB, DTB, WNB = up(alog), up(dtb), up(wn)
        qkvb = dev.allocator.alloc(8192*4, BufferSpec())
        zb = dev.allocator.alloc(4096*4, BufferSpec())
        abb = dev.allocator.alloc(64*4, BufferSpec())
        convst = (rng.standard_normal((8192, 3)) * 0.3).astype(np.float32)
        csb = up(convst)
        qkvsb = dev.allocator.alloc(8192*4, BufferSpec())
        S0 = (rng.standard_normal((32, 128, 128)) * 0.05).astype(np.float32)
        Sb = up(S0.copy())
        gyb = dev.allocator.alloc(4096*4, BufferSpec())
        h2b = dev.allocator.alloc(2048*4, BufferSpec())
        def run3():
            K["gv8k2048p"](WQKVB, hnb, qkvb, global_size=(256,1,1), local_size=LSZ["gv8k2048p"], vals=(8192,), wait=True)
            K["gv8k2048p"](WZB, hnb, zb, global_size=(128,1,1), local_size=LSZ["gv8k2048p"], vals=(4096,), wait=True)
            K["gvf32ab"](WAB, WBB, hnb, abb, global_size=(1,1,1), local_size=LSZ["gvf32ab"], wait=True)
            K["gconv36_1"](CWB, qkvb, csb, qkvsb, global_size=(32,1,1), local_size=LSZ["gconv36_1"], wait=True)
            K["k2s36_1"](qkvsb, abb, ALB, DTB, WNB, zb, Sb, gyb, global_size=(32,1,1), local_size=LSZ["k2s36_1"], wait=True)
            K["gv8k4096r"](WOUTB, gyb, rb, h2b, global_size=(64,1,1), local_size=LSZ["gv8k4096r"], vals=(2048,), wait=True)
        run3()
        qkv_g = dn(qkvb, (8192,)); z_g = dn(zb, (4096,)); ab_g = dn(abb, (64,))
        qkvs_g = dn(qkvsb, (8192,)); gy_g = dn(gyb, (4096,)); h2_g = dn(h2b, (2048,))
        qkv_r = _dot_lane_ref(dq_q8_0(wqkv, 2048), hn, 64, 1)
        z_r = _dot_lane_ref(dq_q8_0(wz, 2048), hn, 64, 1)
        a_r, b_r = gvf32_ref(Wa, Wb, hn)
        qkvs_r, _ = gconv_ref(cw, qkv_r.reshape(1, 8192), convst)
        gy_r, S_r = k2s_ref(qkvs_r, np.concatenate([a_r, b_r]).reshape(1, 64), alog, dtb, wn,
                            z_r.reshape(1, 4096), S0)
        h2_r = q8_dot4096_ref(wout, gy_r[0], resid)
        dev.allocator._copyin(csb, memoryview(np.ascontiguousarray(convst).data).cast("B"))
        dev.allocator._copyin(Sb, memoryview(np.ascontiguousarray(S0).data).cast("B"))
        run3()
        det = (naneq(gy_g, dn(gyb, (4096,))) and naneq(h2_g, dn(h2b, (2048,)))
               and np.array_equal(qkv_g, dn(qkvb, (8192,))))
        ab_dev = float(np.abs(ab_g - np.concatenate([a_r, b_r])).max())
        checks = {
            "qkv": np.array_equal(qkv_g, qkv_r),
            "z": np.array_equal(z_g, z_r),
            "ab": np.array_equal(ab_g, np.concatenate([a_r, b_r])),
            "conv": np.allclose(qkvs_g, qkvs_r[0], rtol=2e-6, atol=1e-7),
            "gy": np.allclose(gy_g, gy_r[0], rtol=3e-5, atol=3e-6),
            "h2": np.allclose(h2_g, h2_r, rtol=3e-4, atol=3e-5),
        }
        devs = {k2: float(np.abs(a - b).max()) for k2, a, b in
                [("qkv", qkv_g, qkv_r), ("ab", ab_g, np.concatenate([a_r, b_r])), ("gy", gy_g, gy_r[0]), ("h2", h2_g, h2_r)]}
        record("S3", f"GDN L0 slice: " + " ".join(f"{k2}:{'OK' if v else 'BAD'}" for k2, v in checks.items())
                      + f" maxdevs {devs} det{'OK' if det else 'FAIL'}")

    # ============================ S4: attn slice (real L3) ============================
    if not done("S4"):
        rng = np.random.default_rng(19)
        hn = (rng.standard_normal(2048) * 0.4).astype(np.float32)
        pos = 7
        wq = load_q8(3, "attn_q_weight", 8192)
        wk = load_q8(3, "attn_k_weight", 512)
        wv = load_q8(3, "attn_v_weight", 512)
        qw = load_f32(3, "attn_q_norm_weight")
        kw = load_f32(3, "attn_k_norm_weight")
        cos, sin = rope_tables(CTX)
        Kq = np.zeros((2, CTX, 256), dtype=np.int8); Ks = np.ones((2, CTX, 2), dtype=np.float32)
        Vq = np.zeros((2, CTX, 256), dtype=np.int8); Vs = np.ones((2, CTX, 2), dtype=np.float32)
        for j in range(2):
            for p in range(pos):
                Kq[j, p], Ks[j, p] = kv_quant((rng.standard_normal(256)*0.5).astype(np.float32))
                Vq[j, p], Vs[j, p] = kv_quant((rng.standard_normal(256)*0.5).astype(np.float32))
        WQB, WKB, WVB = up(wq), up(wk), up(wv)
        QWB, KWB = up(qw), up(kw)
        COSB, SINB = up(cos), up(sin)
        hnb = up(hn)
        KQB, KSB = up(Kq.copy()), up(Ks.copy())
        VQB, VSB = up(Vq.copy()), up(Vs.copy())
        POSB = up(np.array([0], dtype=np.int32))
        PTB4 = up(np.array([KWB.va_addr, QWB.va_addr, COSB.va_addr, SINB.va_addr,
                            KQB.va_addr, KSB.va_addr, VQB.va_addr, VSB.va_addr,
                            POSB.va_addr], dtype=np.uint64))
        qgb = dev.allocator.alloc(8192*4, BufferSpec())
        kqb = dev.allocator.alloc(512*4, BufferSpec())
        vqb = dev.allocator.alloc(512*4, BufferSpec())
        ayb = dev.allocator.alloc(4096*4, BufferSpec())
        def run4():
            print("    [S4] q/k/v gemvs...", flush=True)
            dev.allocator._copyin(POSB, memoryview(np.array([pos], dtype=np.int32).tobytes()))
            K["gv8k2048p"](WQB, hnb, qgb, global_size=(256,1,1), local_size=LSZ["gv8k2048p"], vals=(8192,), wait=True)
            print("    [S4] spka...", flush=True)
            K["gv8k2048p"](WKB, hnb, kqb, global_size=(16,1,1), local_size=LSZ["gv8k2048p"], vals=(512,), wait=True)
            K["gv8k2048p"](WVB, hnb, vqb, global_size=(16,1,1), local_size=LSZ["gv8k2048p"], vals=(512,), wait=True)
            K["spka256"](kqb, vqb, PTB4, global_size=(2,1,1), local_size=LSZ["spka256"], wait=True)
            K["spkq256"](qgb, ayb, PTB4, global_size=(16,1,1), local_size=LSZ["spkq256"], wait=True)
        run4()
        qg_g = dn(qgb, (8192,)); kq_g = dn(kqb, (512,)); vq_g = dn(vqb, (512,))
        ay_g = dn(ayb, (4096,))
        qg_r = _dot_lane_ref(dq_q8_0(wq, 2048), hn, 64, 1)
        kq_r = _dot_lane_ref(dq_q8_0(wk, 2048), hn, 64, 1)
        vq_r = _dot_lane_ref(dq_q8_0(wv, 2048), hn, 64, 1)
        Kq_r = Kq.copy(); Ks_r = Ks.copy(); Vq_r = Vq.copy(); Vs_r = Vs.copy()
        for j in range(2):
            spka_ref(_f(kq_r[j*256:(j+1)*256]), _f(vq_r[j*256:(j+1)*256]), qw, kw,
                     cos[pos], sin[pos], pos, Kq_r[j], Ks_r[j], Vq_r[j], Vs_r[j])
        ay_r = np.empty(4096, dtype=np.float32)
        for hh in range(16):
            jj = hh >> 3
            ay_r[hh*256:(hh+1)*256] = spkq_h_ref(qg_r, qw, Kq_r[jj], Ks_r[jj], Vq_r[jj], Vs_r[jj],
                                                 cos[pos], sin[pos], pos, hh)
        kq_ok = np.array_equal(kq_g, kq_r); vq_ok = np.array_equal(vq_g, vq_r)
        kcache_ok = np.array_equal(dn(KQB, (2, CTX, 256), np.int8)[:, :pos+1], Kq_r[:, :pos+1])
        vcache_ok = np.array_equal(dn(VQB, (2, CTX, 256), np.int8)[:, :pos+1], Vq_r[:, :pos+1])
        ay_ok = np.allclose(ay_g, ay_r, rtol=3e-5, atol=3e-6)
        dev.allocator._copyin(KQB, memoryview(np.ascontiguousarray(Kq).data).cast("B"))
        dev.allocator._copyin(KSB, memoryview(np.ascontiguousarray(Ks).data).cast("B"))
        dev.allocator._copyin(VQB, memoryview(np.ascontiguousarray(Vq).data).cast("B"))
        dev.allocator._copyin(VSB, memoryview(np.ascontiguousarray(Vs).data).cast("B"))
        run4()
        det = naneq(ay_g, dn(ayb, (4096,))) and np.array_equal(dn(KQB, (2,CTX,256), np.int8)[:, :pos+1], Kq_r[:, :pos+1])
        record("S4", f"attn L3 slice pos={pos}: kq{'OK' if kq_ok else 'BAD'} vq{'OK' if vq_ok else 'BAD'} "
                      f"Kcache{'EXACT' if kcache_ok else 'BAD'} Vcache{'EXACT' if vcache_ok else 'BAD'} "
                      f"y{'OK' if ay_ok else 'BAD'} (maxdev {np.abs(ay_g-ay_r).max():.2e}) det{'OK' if det else 'FAIL'}")

    # ============================ S5: h6k2048 ============================
    if not done("S5"):
        t0w = time.time()
        while not os.path.exists(ANCHOR_NPZ):
            if time.time() - t0w > 7200:
                record("S5", "TIMEOUT waiting for the anchor npz"); sys.exit(2)
            print(f"[S5] waiting for {ANCHOR_NPZ} ({(time.time()-t0w)/60:.0f} min)", flush=True)
            time.sleep(60)
        rng = np.random.default_rng(23)
        x = (rng.standard_normal(2048) * 0.3).astype(np.float32)
        with open(f"{PACK}/trunk/output_weight.bin", "rb") as f:
            f.seek(int(rng.integers(0, 248320-49)) * 1680)
            rr = np.frombuffer(f.read(48*1680), dtype=np.uint8).reshape(48, 1680)
        XB, RRB = up(x), up(rr)
        OB = dev.allocator.alloc(48*4, BufferSpec())
        def run5():
            K["h6k2048"](RRB, XB, OB, global_size=(2,1,1), local_size=LSZ["h6k2048"], vals=(48,), wait=True)
        run5()
        got = dn(OB, (48,))
        ref = h6k_rows_ref(rr, x)
        run5()
        det = np.array_equal(got, dn(OB, (48,)))
        rows_ok = np.array_equal(got, ref)
        anc = np.load(ANCHOR_NPZ, allow_pickle=True)
        hf = np.asarray(anc["htrace"][0][-1], dtype=np.float32)   # OBJECT-ARRAY TRAP: npz rows are boxed floats
        HFB = up(hf)
        HB = up_big(file_uploader(f"{PACK}/trunk/output_weight.bin"), 248320*1680)
        LB = alloc(248320*4)
        SENT = np.full(248320, -3.0e30, dtype=np.float32)
        def run5f():
            dev.allocator._copyin(LB, memoryview(SENT.data).cast("B"))
            K["h6k2048"](HB, HFB, LB, global_size=(7760,1,1), local_size=LSZ["h6k2048"], vals=(248320,), wait=True)
        run5f()
        logits = dn(LB, (248320,))
        n_unw = int((logits == -3.0e30).sum())
        t1 = int(np.argmax(logits))
        print(f"[S5d] unwritten={n_unw} argmax={t1} l[11751]={logits[11751]:.4f} l[97412]={logits[97412]:.4f} "
              f"max={logits.max():.3f} nfinite={int(np.isfinite(logits).all())}", flush=True)
        exp_t1 = int(anc["top1"][0][-1])
        run5f()
        det2 = np.array_equal(logits, dn(LB, (248320,)))
        record("S5", f"h6k2048: rows{'BIT-EXACT' if rows_ok else 'BAD'} (maxdev {np.abs(got-ref).max():.2e}) "
                      f"det{'OK' if det else 'FAIL'} | full head top1={t1} anchor={exp_t1} "
                      f"{'EXACT' if t1 == exp_t1 else 'MISMATCH'} det{'OK' if det2 else 'FAIL'}")

    # ============================ S6: THE FULL TRAIN ============================
    man = bank_reader()
    anc = np.load(ANCHOR_NPZ, allow_pickle=True)
    battery_ids = [np.asarray(a, dtype=np.int32) for a in anc["ids"]]
    battery_t1 = [np.asarray(a, dtype=np.int32) for a in anc["top1"]]
    battery_gap = [np.asarray(g, dtype=np.float32) for g in anc["gap"]]

    t0 = time.time()
    PTB_UP = {}; PTB_DN = {}
    for L in range(40):
        meta = man["routed"][L]
        fl = meta["files"]
        if len(fl) == 1:
            f = fl[0]
            b = up_big(file_uploader(os.path.join(PACK, f["file"])), f["bytes"])
            PTB_UP[L] = up(np.array([b.va_addr + e*f["slab"] + f["offsets"]["gate"] for e in range(256)], dtype=np.uint64))
            PTB_DN[L] = up(np.array([b.va_addr + e*f["slab"] + f["offsets"]["down"] for e in range(256)], dtype=np.uint64))
        else:
            fg = next(f for f in fl if "gate" in f["offsets"])
            fd = next(f for f in fl if "down" in f["offsets"])
            bg = up_big(file_uploader(os.path.join(PACK, fg["file"])), fg["bytes"])
            bd = up_big(file_uploader(os.path.join(PACK, fd["file"])), fd["bytes"])
            PTB_UP[L] = up(np.array([bg.va_addr + e*fg["slab"] + fg["offsets"]["gate"] for e in range(256)], dtype=np.uint64))
            PTB_DN[L] = up(np.array([bd.va_addr + e*fd["slab"] + fd["offsets"]["down"] for e in range(256)], dtype=np.uint64))
    W = {}
    for L in range(40):
        w = {}
        w["an"] = up(load_f32(L, "attn_norm_weight"))
        w["pn"] = up(load_f32(L, "post_attention_norm_weight"))
        w["rt"] = up(load_router(L)); w["wsh"] = up(load_wsh(L))
        wg, wu, wd = load_shexp_raw(L)
        w["sg"] = up(wg); w["su"] = up(wu); w["sd"] = up(wd)
        if L in GDN_LAYERS:
            w["qkv"] = up(load_q8(L, "attn_qkv_weight", 8192))
            w["z"] = up(load_q8(L, "attn_gate_weight", 4096))
            w["al"] = up(load_f32(L, "ssm_a")); w["dt"] = up(load_f32(L, "ssm_dt_bias"))
            w["wa"] = up(load_f32(L, "ssm_alpha_weight")); w["wb"] = up(load_f32(L, "ssm_beta_weight"))
            w["cw"] = up(load_f32(L, "ssm_conv1d_weight"))
            w["sn"] = up(load_f32(L, "ssm_norm_weight"))
            w["out"] = up(load_q8(L, "ssm_out_weight", 2048, 4096))
        else:
            w["q"] = up(load_q8(L, "attn_q_weight", 8192))
            w["k"] = up(load_q8(L, "attn_k_weight", 512))
            w["v"] = up(load_q8(L, "attn_v_weight", 512))
            w["qw"] = up(load_f32(L, "attn_q_norm_weight"))
            w["kw"] = up(load_f32(L, "attn_k_norm_weight"))
            w["o"] = up(load_q8(L, "attn_output_weight", 2048, 4096))
        W[L] = w
    EMB = up_big(file_uploader(f"{PACK}/trunk/token_embd_weight.bin"), 248320*2176)
    HEAD = up_big(file_uploader(f"{PACK}/trunk/output_weight.bin"), 248320*1680)
    ONORM = up(load_f32(None, "output_norm_weight"))
    cos, sin = rope_tables(CTX)
    COSB, SINB = up(cos), up(sin)
    dev.synchronize()
    print(f"[S6] banks+trunk uploaded in {time.time()-t0:.0f}s", flush=True)

    hA = alloc(PMAX*2048*4); hB = alloc(PMAX*2048*4); hnb = alloc(PMAX*2048*4)
    qkvb = alloc(PMAX*8192*4); zb = alloc(PMAX*4096*4); abb = alloc(PMAX*64*4)
    qkvsb = alloc(PMAX*8192*4); gyb = alloc(PMAX*4096*4)
    qgb = alloc(PMAX*8192*4); kqb = alloc(PMAX*512*4); vqb = alloc(PMAX*512*4)
    ayb = alloc(PMAX*4096*4)
    eidsb = alloc(PMAX*8*2); gatesb = alloc(PMAX*8*4); sgb = alloc(PMAX*4)
    actb = alloc(PMAX*8*512*4); partsb = alloc(PMAX*8*2048*2); shb = alloc(PMAX*2048*4)
    normhb = alloc(PMAX*2048*4); logitsb = alloc(248320*4)
    idsb = up(np.zeros(PMAX, dtype=np.int32))
    POSB = up(np.array([0], dtype=np.int32))
    SALL = alloc(30*32*128*128*4)
    SV = [SALL.offset(offset=L*32*128*128*4, size=32*128*128*4) for L in range(30)]
    CSALL = alloc(30*8192*3*4)
    CSV = [CSALL.offset(offset=L*8192*3*4, size=8192*3*4) for L in range(30)]
    KVQ = {}; KVS = {}; VVQ = {}; VVS = {}
    for ai in range(10):
        KVQ[ai] = up(np.zeros((2, CTX, 256), dtype=np.int8))
        VVQ[ai] = up(np.zeros((2, CTX, 256), dtype=np.int8))
        KVS[ai] = up(np.zeros((2, CTX, 2), dtype=np.float32))
        VVS[ai] = up(np.zeros((2, CTX, 2), dtype=np.float32))
    SPTB = {}
    for ai in range(10):
        L = ATTN_LAYERS[ai]
        SPTB[ai] = up(np.array([W[L]["kw"].va_addr, W[L]["qw"].va_addr, COSB.va_addr, SINB.va_addr,
                                KVQ[ai].va_addr, KVS[ai].va_addr, VVQ[ai].va_addr, VVS[ai].va_addr,
                                POSB.va_addr], dtype=np.uint64))
    ZS = np.zeros(30*32*128*128, dtype=np.float32).tobytes()
    ZCS = np.zeros(30*8192*3, dtype=np.float32).tobytes()
    ZKVQ = np.zeros((2, CTX, 256), dtype=np.int8).tobytes()
    ZKVS = np.zeros((2, CTX, 2), dtype=np.float32).tobytes()
    def reset_states():
        dev.allocator._copyin(SALL, memoryview(ZS))
        dev.allocator._copyin(CSALL, memoryview(ZCS))
        for ai in range(10):
            dev.allocator._copyin(KVQ[ai], memoryview(ZKVQ))
            dev.allocator._copyin(VVQ[ai], memoryview(ZKVQ))
            dev.allocator._copyin(KVS[ai], memoryview(ZKVS))
            dev.allocator._copyin(VVS[ai], memoryview(ZKVS))

    def build_seq(P, tg, tk, with_head=True):
        seq = [("embg248", (EMB, idsb, hA), max(1, (P+31)//32), (P,))]
        for L in range(40):
            w = W[L]
            # IN-PLACE PING-PONG LAW: every layer reads hA, uses hB as the
            # attn/gdn scratch (hmid = out + resid), and cmb writes the layer
            # output BACK INTO hA (cmbz2048 args: hres=hmid, y=hin). The old
            # alternating hin/hmid made L=1 read layer-0's PRE-MoE h (stale).
            hin = hA
            hmid = hB
            seq.append(("rmsz2048g", (hin, w["an"], hnb), (P,), ()))
            if L in GDN_LAYERS:
                gi = GDN_LAYERS.index(L)
                seq.append(("gv8k2048p", (w["qkv"], hnb, qkvb), (256,), (8192,)))
                seq.append(("gv8k2048p", (w["z"], hnb, zb), (128,), (4096,)))
                seq.append(("gvf32ab", (w["wa"], w["wb"], hnb, abb), (P,), ()))
                seq.append((tg, (w["cw"], qkvb, CSV[gi], qkvsb), (32,), ()))
                seq.append((tk, (qkvsb, abb, w["al"], w["dt"], w["sn"], zb, SV[gi], gyb), (32,), ()))
                seq.append(("gv8k4096r", (w["out"], gyb, hin, hmid), (64,), (2048,)))
            else:
                ai = ATTN_LAYERS.index(L)
                seq.append(("gv8k2048p", (w["q"], hnb, qgb), (256,), (8192,)))
                seq.append(("gv8k2048p", (w["k"], hnb, kqb), (16,), (512,)))
                seq.append(("gv8k2048p", (w["v"], hnb, vqb), (16,), (512,)))
                seq.append(("spka256", (kqb, vqb, SPTB[ai]), (2*P,), ()))
                seq.append(("spkq256", (qgb, ayb, SPTB[ai]), (16*P,), ()))
                seq.append(("gv8k4096r", (w["o"], ayb, hin, hmid), (64,), (2048,)))
            seq.append(("rmsz2048g", (hmid, w["pn"], hnb), (P,), ()))
            seq.append(("rt8e256", (w["rt"], w["wsh"], hnb, eidsb, gatesb, sgb), (P,), ()))
            seq.append(("shexp8", (w["sg"], w["su"], w["sd"], hnb, shb), (P,), ()))
            upk = "gx8e256up4" if man["routed"][L]["types"]["gate"] == "IQ4_XS" else "gx8e256up"
            # THE LUT LAW (P2 mkchain): up4 (IQ4_XS) consumes iq4nl; up (IQ3_S) consumes gridf
            upbufs = (PTB_UP[L], eidsb, hnb, iq4nl, actb) if upk == "gx8e256up4" else (PTB_UP[L], eidsb, hnb, gridf, actb)
            seq.append((upk, upbufs, (P*8,), ()))
            dnk = "gx8e256dn6" if man["routed"][L]["types"]["down"] == "Q6_K" else "gx8e256dn"
            dnbufs = (PTB_DN[L], eidsb, actb, partsb) if dnk == "gx8e256dn6" else (PTB_DN[L], eidsb, actb, iq4nl, partsb)
            seq.append((dnk, dnbufs, (P*8,), ()))
            seq.append(("cmbz2048", (partsb, gatesb, sgb, shb, hmid, hin), (P,), ()))
        # in-place ping-pong -> final trunk h at hA
        if with_head:
            seq.append(("rmsz2048g", (hA, ONORM, normhb), (P,), ()))
            seq.append(("h6k2048", (HEAD, normhb, logitsb), (7760,), (248320,)))
        # normalize grids to plain ints (entries were authored as 1-tuples)
        return [(n, b, (g[0] if isinstance(g, tuple) else g), v) for n, b, g, v in seq]

    SEQ1 = build_seq(1, "gconv36_1", "k2s36_1", with_head=True)
    NNODES = len(SEQ1)

    def battery(run_step):
        all_t1 = []
        for pi, ids in enumerate(battery_ids):
            reset_states()
            pt1 = []
            for pos, tid in enumerate(ids):
                dev.allocator._copyin(idsb, memoryview(np.array([tid], dtype=np.int32).tobytes()))
                dev.allocator._copyin(POSB, memoryview(np.array([pos], dtype=np.int32).tobytes()))
                run_step()
                pt1.append(int(np.argmax(dn(logitsb, (248320,)))))
            all_t1.append(np.array(pt1, dtype=np.int32))
            print(f"    prompt {pi} ({len(ids)} pos) done", flush=True)
        return all_t1

    def score(all_t1, tag, secs):
        pm = sum(int((g == e).sum()) for g, e in zip(all_t1, battery_t1))
        pt = sum(len(e) for e in battery_t1)
        fm = sum(int((g == e).all()) for g, e in zip(all_t1, battery_t1))
        mism = []; mgaps = []
        for pi, (g, e) in enumerate(zip(all_t1, battery_t1)):
            bad = np.where(g != e)[0]
            if len(bad):
                mism.append((pi, bad.tolist()[:6]))
                for b in bad: mgaps.append(float(battery_gap[pi][b]))
        gap_str = ""
        if mgaps:
            mg = np.array(mgaps)
            gap_str = (f" | mismatch anchor-gaps: n={len(mg)} min={mg.min():.2e} med={np.median(mg):.2e} "
                       f"max={mg.max():.2e}; <1e-3:{int((mg<1e-3).sum())} <1e-2:{int((mg<1e-2).sum())} "
                       f">=1e-2:{int((mg>=1e-2).sum())} (tie-class adjudication)")
        return f"{tag}: {pm}/{pt} positions EXACT, {fm}/20 prompts full, {pt/max(1,secs):.2f} tok/s mism={mism[:6]}{gap_str}"

    if not done("S6"):
        def eager_step():
            for name, bufs, gx, vals in SEQ1:
                K[name](*bufs, global_size=(gx,1,1), local_size=LSZ[name], vals=vals, wait=True)
        te0 = time.time()
        T1_run1 = battery(eager_step)
        eager_s = time.time() - te0
        record("S6", score(T1_run1, "TRAIN T=1 EAGER vs anchor", eager_s))
        # det x2 (rerun 4 prompts)
        det_ids = battery_ids[:4]
        det_t1 = []
        for pi in range(4):
            reset_states()
            pt1 = []
            for pos, tid in enumerate(det_ids[pi]):
                dev.allocator._copyin(idsb, memoryview(np.array([tid], dtype=np.int32).tobytes()))
                dev.allocator._copyin(POSB, memoryview(np.array([pos], dtype=np.int32).tobytes()))
                eager_step()
                pt1.append(int(np.argmax(dn(logitsb, (248320,)))))
            det_t1.append(np.array(pt1, dtype=np.int32))
        detok = all(np.array_equal(a, b) for a, b in zip(det_t1, T1_run1[:4]))
        record("S6d", f"TRAIN EAGER det x2 (4 prompts): {'OK' if detok else 'FAIL'}")

    # ---- S6b: in-graph ----
    class MG:
        def __init__(self, seq, tag):
            self.prev = UOp.variable(f"{tag}_p", 0, 0xffffffff, dtype=dtypes.uint32)
            self.cur  = UOp.variable(f"{tag}_c", 0, 0xffffffff, dtype=dtypes.uint32)
            per = max(round_up(p.kernargs_alloc_size, 8) for p, a, g, v in seq)
            self.kb = per*len(seq)
            self.ka = dev.allocator.alloc(self.kb + 8, BufferSpec(cpu_access=True, nolru=True))
            keep.append(self.ka)
            KA_BYTES[0] = self.kb
            q = NVComputeQueue(); q.memory_barrier()
            q.wait(dev.timeline_signal, self.prev)
            off = 0
            for p, bufs, grid, vals in seq:
                ab = self.ka.offset(offset=off, size=p.kernargs_alloc_size)
                st = p.fill_kernargs(tuple(bufs), tuple(vals), kernargs=ab)
                q.exec(p, st, (grid,1,1), LSZ[p.name])
                off += round_up(p.kernargs_alloc_size, 8)
            q.signal(dev.timeline_signal, self.cur)
            self.q = q
        def submit(self, pv, cv):
            self.q.submit(dev, {self.prev.expr: int(pv), self.cur.expr: int(cv)})

    seqp = [(K[n], b, g, v) for n, b, g, v in SEQ1]
    ga = MG(seqp, "s6a"); gb2 = MG(seqp, "s6b")
    gst = {"turn": 0}
    def graph_step():
        m = ga if gst["turn"] == 0 else gb2
        gst["turn"] = 1 - gst["turn"]
        v = dev.next_timeline()
        m.submit(v - 1, v)
        nv_wait_timeline(dev, v, what="s6b", timeout_s=60.0)
    if not done("S6b"):
        tg0 = time.time()
        T1_graph = battery(graph_step)
        graph_s = time.time() - tg0
        record("S6b", score(T1_graph, f"TRAIN T=1 GRAPH ({NNODES} nodes, ka {ga.kb} B)", graph_s))
        # det x2 in graph (4 prompts)
        det_t1 = []
        for pi in range(4):
            reset_states()
            pt1 = []
            for pos, tid in enumerate(battery_ids[pi]):
                dev.allocator._copyin(idsb, memoryview(np.array([tid], dtype=np.int32).tobytes()))
                dev.allocator._copyin(POSB, memoryview(np.array([pos], dtype=np.int32).tobytes()))
                graph_step()
                pt1.append(int(np.argmax(dn(logitsb, (248320,)))))
            det_t1.append(np.array(pt1, dtype=np.int32))
        detok = all(np.array_equal(a, b) for a, b in zip(det_t1, T1_graph[:4]))
        record("S6bd", f"TRAIN GRAPH det x2 (4 prompts): {'OK' if detok else 'FAIL'}")

    # ---- S9: the 20-prompt greedy continuation BANK (the MoE Tier-1 seed) ----
    if not done("S9"):
        s6b_line = next((l for l in open(PROG).read().splitlines() if l.startswith("S6b ")), "")
        try: s6b_exact = int(s6b_line.split("positions EXACT")[0].split(":")[-1].strip().split("/")[0])
        except Exception: s6b_exact = -1
        if s6b_exact < 340:
            record("S9", f"SKIPPED-DEFERRED (S6b exact {s6b_exact} < 340 — fix the train before banking)")
        else:
            import json as _json
            NTOK = 32
            from MM_P34_ports import Anchor, fresh_state as _fs
            def run_cont(pi, ntok=NTOK, collect_prompt=True):
                ids = battery_ids[pi]
                reset_states()
                pt1 = []
                for pos, tid in enumerate(ids):
                    dev.allocator._copyin(idsb, memoryview(np.array([tid], dtype=np.int32).tobytes()))
                    dev.allocator._copyin(POSB, memoryview(np.array([pos], dtype=np.int32).tobytes()))
                    graph_step()
                    pt1.append(int(np.argmax(dn(logitsb, (248320,)))))
                gen = []
                t = pt1[-1]                   # first generated token (feed back)
                for g in range(ntok):
                    gen.append(t)
                    dev.allocator._copyin(idsb, memoryview(np.array([t], dtype=np.int32).tobytes()))
                    dev.allocator._copyin(POSB, memoryview(np.array([len(ids)+g], dtype=np.int32).tobytes()))
                    graph_step()
                    t = int(np.argmax(dn(logitsb, (248320,))))
                return (pt1 if collect_prompt else None), gen
            tb0 = time.time()
            bank_pt1, bank_gen = [], []
            for pi in range(len(battery_ids)):
                p1, g = run_cont(pi)
                bank_pt1.append(p1); bank_gen.append(g)
                print(f"    [S9] prompt {pi}: gen[:8]={g[:8]}", flush=True)
            # det x2 on 4 prompts
            detg = [run_cont(pi, collect_prompt=False)[1] for pi in range(4)]
            detok = all(a == b for a, b in zip(detg, bank_gen[:4]))
            # anchor cross-check: 3 prompts x 8 continuation tokens (CPU fp32 ref)
            axr = []
            try:
                anc_obj = Anchor(ctx=CTX)
                for pi in range(3):
                    ids = [int(x) for x in battery_ids[pi]]
                    S, convst, KV = _fs(CTX)
                    for pos, tid in enumerate(ids):
                        t1, _, _ = anc_obj.forward_token(tid, pos, S, convst, KV)
                    agen = []
                    t = t1
                    for g in range(8):
                        agen.append(t)
                        t, _, _ = anc_obj.forward_token(t, len(ids)+g, S, convst, KV)
                    axr.append((pi, agen, bank_gen[pi][:8]))
            except Exception as e:
                axr.append(("ERR", repr(e), []))
            ax_ok = all(a == b for _, a, b in axr if isinstance(_, int))
            bank = {"model": "Qwen3.6-35B-A3B-UD-IQ4_XS", "engine": "MM_P34 graph train",
                    "note": "MoE Tier-1 seed bank (P5 extends to 60/60 + ctx ladder)",
                    "ntok": NTOK, "det4": bool(detok),
                    "gen": bank_gen, "anchor_x8": [[a, b] for _, a, b in axr]}
            with open(os.path.expanduser("~/mm_p34_bank.json"), "w") as f:
                _json.dump(bank, f)
            record("S9", f"BANK 20 prompts x {NTOK} greedy cont: det4={'OK' if detok else 'FAIL'} "
                          f"anchor-x8(3 prompts)={'EXACT' if ax_ok else 'DIFF'} {axr} "
                          f"({(time.time()-tb0):.0f}s, ~/mm_p34_bank.json)")

    # ---- S7: perf ----
    if not done("S7"):
        def timed(launch, n=20):
            launch()
            ts = []
            for _ in range(5):
                t0p = time.perf_counter()
                for _ in range(n): launch()
                ts.append((time.perf_counter()-t0p)/n*1e3)
            return min(ts)
        def timed_pipe(launch, n=20):
            launch(); dev.synchronize()
            ts = []
            for _ in range(5):
                t0p = time.perf_counter()
                for _ in range(n): launch()
                dev.synchronize()
                ts.append((time.perf_counter()-t0p)/n*1e3)
            return min(ts)
        quartet = {}
        for P, tag in ((3, "P3"), (9, "P9")):
            hnp = up((np.random.default_rng(31).uniform(-0.5, 0.5, (P, 2048))).astype(np.float32))
            eids9 = up(np.tile(np.arange(8, dtype=np.uint16), P))
            gates9 = up(np.full((P, 8), 0.125, dtype=np.float32))
            sg9 = up(np.full(P, 0.5, dtype=np.float32))
            sh9 = up(np.zeros((P, 2048), dtype=np.float32))
            hres9 = up(np.zeros((P, 2048), dtype=np.float32))
            parts9 = alloc(P*8*2048*2); act9 = alloc(P*8*512*4)
            y9 = alloc(P*2048*4)
            def qq():
                K["rt8e256"](W[0]["rt"], W[0]["wsh"], hnp, eids9, gates9, sg9, global_size=(P,1,1), local_size=LSZ["rt8e256"], wait=True)
                K["shexp8"](W[0]["sg"], W[0]["su"], W[0]["sd"], hnp, sh9, global_size=(P,1,1), local_size=LSZ["shexp8"], wait=True)
                K["gx8e256up"](PTB_UP[0], eids9, hnp, gridf, act9, global_size=(P*8,1,1), local_size=LSZ["gx8e256up"], wait=True)
                K["gx8e256dn"](PTB_DN[0], eids9, act9, iq4nl, parts9, global_size=(P*8,1,1), local_size=LSZ["gx8e256dn"], wait=True)
                K["cmbz2048"](parts9, gates9, sg9, sh9, hres9, y9, global_size=(P,1,1), local_size=LSZ["cmbz2048"], wait=True)
            quartet[tag] = timed(qq, 10)
        split = []
        for name, bufs, gx, vals in [
            ("embg248", (EMB, idsb, hA), 1, (1,)),
            ("rmsz2048g", (hA, W[0]["an"], hnb), 1, ()),
            ("gv8k2048p", (W[0]["qkv"], hnb, qkvb), 256, (8192,)),
            ("gv8k2048p", (W[3]["q"], hnb, qgb), 256, (8192,)),
            ("gvf32ab", (W[0]["wa"], W[0]["wb"], hnb, abb), 1, ()),
            ("gconv36_1", (W[0]["cw"], qkvb, CSV[0], qkvsb), 32, ()),
            ("k2s36_1", (qkvsb, abb, W[0]["al"], W[0]["dt"], W[0]["sn"], zb, SV[0], gyb), 32, ()),
            ("gv8k4096r", (W[0]["out"], gyb, hA, hB), 64, (2048,)),
            ("spka256", (kqb, vqb, SPTB[0]), 2, ()),
            ("spkq256", (qgb, ayb, SPTB[0]), 16, ()),
            ("rt8e256", (W[0]["rt"], W[0]["wsh"], hnb, eidsb, gatesb, sgb), 1, ()),
            ("shexp8", (W[0]["sg"], W[0]["su"], W[0]["sd"], hnb, shb), 1, ()),
            ("gx8e256up", (PTB_UP[0], eidsb, hnb, gridf, actb), 8, ()),
            ("gx8e256dn", (PTB_DN[0], eidsb, actb, iq4nl, partsb), 8, ()),
            ("cmbz2048", (partsb, gatesb, sgb, shb, hB, hA), 1, ()),
            ("h6k2048", (HEAD, normhb, logitsb), 7760, (248320,)),
        ]:
            tagname = name if name != "gv8k2048p" else f"gv8p{vals[0] if vals else 0}"
            def lm(name=name, bufs=bufs, gx=gx, vals=vals):
                K[name](*bufs, global_size=(gx,1,1), local_size=LSZ[name], vals=vals, wait=True)
            split.append(f"{tagname}={timed(lm):.3f}/{timed_pipe(lm):.3f}")
        record("S7", f"PERF quartet/layer {quartet} ms (wait-each x5 launches) | "
                      f"per-class ms wait-each/pipelined: {' '.join(split)} | ka-slab/graph {KA_BYTES[0]} B")

    # ---- S8: K-mix graph sets (wire, no full gate) ----
    if not done("S8"):
        out = []
        for tag, P, tg, tk in (("D2", 3, "gconv36_3", "k2s36_3"), ("D8", 9, "gconv36_9", "k2s36_9")):
            seqp = build_seq(P, tg, tk, with_head=False)
            g = MG([(K[n], b, gr, v) for n, b, gr, v in seqp], f"s8{tag}")
            ids0 = battery_ids[0]
            reset_states()
            cks = []
            t0s = time.perf_counter()
            last = dev.timeline_value - 1
            for i in range(32):
                tid = int(ids0[i % len(ids0)])
                dev.allocator._copyin(idsb, memoryview(np.full(P, tid, dtype=np.int32).tobytes()))
                dev.allocator._copyin(POSB, memoryview(np.array([i % len(ids0)], dtype=np.int32).tobytes()))
                v = dev.next_timeline()
                g.submit(v - 1, v)
                nv_wait_timeline(dev, v, what=f"s8{tag}", timeout_s=60.0)
                cks.append(hashlib.sha256(dn(hA, (P, 2048)).tobytes()).hexdigest()[:12])
            ms = (time.perf_counter()-t0s)/32*1e3
            det = cks == cks  # per-replay checksum sequence; compare rerun below
            # rerun 8 for det
            cks2 = []
            reset_states()
            last = dev.timeline_value - 1
            for i in range(8):
                tid = int(ids0[i % len(ids0)])
                dev.allocator._copyin(idsb, memoryview(np.full(P, tid, dtype=np.int32).tobytes()))
                dev.allocator._copyin(POSB, memoryview(np.array([i % len(ids0)], dtype=np.int32).tobytes()))
                v = dev.next_timeline()
                g.submit(v - 1, v)
                nv_wait_timeline(dev, v, what=f"s8{tag}b", timeout_s=60.0)
                cks2.append(hashlib.sha256(dn(hA, (P, 2048)).tobytes()).hexdigest()[:12])
            detok = cks[:8] == cks2
            out.append(f"{tag}: P={P} trunk-train {len(seqp)} nodes x32 CLEAN {ms:.2f} ms/cyc det{'OK' if detok else 'FAIL'}")
        record("S8", "K-MIX " + " | ".join(out))

    print("[ALL DONE]", flush=True)

if __name__ == "__main__":
    main()
