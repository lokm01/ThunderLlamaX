#!/usr/bin/env python3
"""MM P34 DISCRIMINATOR — the 2 S6b mismatch positions (prompt 2 pos 18,
prompt 16 pos 15): per-layer h GPU-vs-anchor + top-5 logits at the flip.

Triage rule:
  - smooth growth (each layer ~fp16-partial noise, no jumps) + small engine
    top1-top2 gap  -> accumulated numerics drift (tie-class), accept + log;
  - one layer JUMPS (dev >> running max) -> real bug localized to that layer
    (then sub-localize by dumping that layer's internals).
Run after MM_P34_run.py finishes (one GPU process).
"""
import os, sys, time
import numpy as np

os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
sys.path.insert(0, "~/tinygrad-metal")
BASE = "~/tinygrad-metal"
ANCHOR_NPZ = os.path.expanduser("~/mm_p34_anchor.npz")
CTX = 1024
TARGETS = [(2, 18), (16, 15)]

import json
from MM_P2_ports import (load_router, load_wsh, load_shexp_raw, iq4nl_f32,
    iq3s_grid_f32, bank_reader, PACK)
from MM_P34_ports import (GDN_LAYERS, ATTN_LAYERS, load_f32, load_q8,
    Anchor, fresh_state)

def main():
    anc = np.load(ANCHOR_NPZ, allow_pickle=True)
    battery_ids = [np.asarray(a, dtype=np.int32) for a in anc["ids"]]

    # ---------- CPU anchor side first (state + per-layer h at target pos) ----------
    an = Anchor(ctx=CTX)
    anchor_dump = {}
    for pi, pos in TARGETS:
        ids = [int(x) for x in battery_ids[pi]]
        S, convst, KV = fresh_state(CTX)
        for p in range(pos + 1):
            tr = [] if p == pos else None
            t1, lg, _ = an.forward_token(ids[p], p, S, convst, KV, want_logits=True, trace=tr)
        anchor_dump[(pi, pos)] = (np.stack(tr), lg, t1)
        print(f"[anchor] p{pi}@{pos}: top1={t1} top5={np.argsort(-lg)[:5].tolist()}", flush=True)
    np.save(os.path.expanduser("~/mm_p34_disc_anchor.npy"),
            {f"{pi}_{pos}": v for (pi, pos), v in anchor_dump.items()}, allow_pickle=True)

    # ---------- GPU side ----------
    from engine0 import dev
    from tinygrad.device import BufferSpec, TinyELF
    from tinygrad.runtime.ops_nv import NVProgram
    from tinygrad.dtype import dtypes
    INT_SIG = (None, 4, dtypes.int32, ())
    LSZ = {"gconv36_1": (256,1,1), "k2s36_1": (256,1,1), "embg248": (1024,1,1),
           "rmsz2048g": (256,1,1), "gv8k2048p": (1024,1,1), "gv8k4096r": (1024,1,1),
           "gvf32ab": (1024,1,1), "spka256": (256,1,1), "spkq256": (256,1,1),
           "rt8e256": (1024,1,1), "shexp8": (1024,1,1), "gx8e256up": (1024,1,1),
           "gx8e256up4": (1024,1,1), "gx8e256dn": (1024,1,1), "gx8e256dn6": (1024,1,1),
           "cmbz2048": (256,1,1), "h6k2048": (1024,1,1)}
    SCALAR_TAIL = {"gv8k2048p", "gv8k4096r", "h6k2048", "embg248"}
    keep = []
    def prog(cbpath, sym):
        lib = open(cbpath, "rb").read()
        return NVProgram(dev, TinyELF(lib=lib, name=sym, target=dev.renderer.target,
                                      signature=(INT_SIG,) if sym in SCALAR_TAIL else tuple()))
    K = {}
    for cbname, sym in [("MM_P34_gconv36_1", "gconv36_1"), ("MM_P34_k2s36_1", "k2s36_1"),
                        ("MM_P34_gvf32ab", "gvf32ab"), ("MM_P34_gv8k2048p", "gv8k2048p"),
                        ("MM_P34_gv8k4096r", "gv8k4096r"), ("MM_P34_rmsz2048g", "rmsz2048g"),
                        ("MM_P34_h6k2048", "h6k2048"), ("MM_P34_cmbz2048", "cmbz2048"),
                        ("MM_P34_spka256", "spka256"), ("MM_P34_spkq256", "spkq256"),
                        ("MM_P1_rt8e256", "rt8e256"), ("MM_P1_gx8e256up", "gx8e256up"),
                        ("MM_P2_gx8e256up4", "gx8e256up4"), ("MM_P2_gx8e256dn", "gx8e256dn"),
                        ("MM_P2_gx8e256dn6", "gx8e256dn6"), ("MM_P2_shexp8", "shexp8"),
                        ("MM_P2_embg248", "embg248")]:
        K[sym] = prog(f"{BASE}/{cbname}.cubin", sym)
    dev.synchronize(); print(f"[disc] {len(K)} programs", flush=True)

    def up(a):
        a = np.ascontiguousarray(a); keep.append(a)
        b = dev.allocator.alloc(a.nbytes, BufferSpec())
        dev.allocator._copyin(b, memoryview(a.data).cast("B")); return b
    def up_big(np_getter, nbytes):
        b = dev.allocator.alloc(nbytes, BufferSpec())
        CH = 64 << 20; off = 0
        while off < nbytes:
            n = min(CH, nbytes - off)
            a = np.ascontiguousarray(np_getter(off, n))
            dev.allocator._copyin(b.offset(offset=off, size=n), memoryview(a.data).cast("B"))
            dev.synchronize(); del a; off += n
        return b
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

    man = bank_reader()
    PTB_UP = {}; PTB_DN = {}
    for L in range(40):
        fl = man["routed"][L]["files"]
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
    dev.synchronize(); print("[disc] uploaded", flush=True)

    def alloc(nbytes):
        b = dev.allocator.alloc(nbytes, BufferSpec()); keep.append(b); return b
    P = 1
    hA = alloc(P*2048*4); hB = alloc(P*2048*4); hnb = alloc(P*2048*4)
    qkvb = alloc(P*8192*4); zb = alloc(P*4096*4); abb = alloc(P*64*4)
    qkvsb = alloc(P*8192*4); gyb = alloc(P*4096*4)
    qgb = alloc(P*8192*4); kqb = alloc(P*512*4); vqb = alloc(P*512*4)
    ayb = alloc(P*4096*4)
    eidsb = alloc(P*8*2); gatesb = alloc(P*8*4); sgb = alloc(P*4)
    actb = alloc(P*8*512*4); partsb = alloc(P*8*2048*2); shb = alloc(P*2048*4)
    normhb = alloc(P*2048*4); logitsb = alloc(248320*4)
    idsb = up(np.zeros(P, dtype=np.int32))
    POSB = up(np.array([0], dtype=np.int32))
    SALL = alloc(30*32*128*128*4)
    SV = [SALL.offset(offset=L*32*128*128*4, size=32*128*128*4) for L in range(30)]
    CSALL = alloc(30*8192*3*4)
    CSV = [CSALL.offset(offset=L*8192*3*4, size=8192*3*4) for L in range(30)]
    KVQ = {}; KVS = {}; VVQ = {}; VVS = {}; SPTB = {}
    from MM_P34_ports import rope_tables
    cos, sin = rope_tables(CTX)
    COSB, SINB = up(cos), up(sin)
    for ai in range(10):
        L = ATTN_LAYERS[ai]
        KVQ[ai] = up(np.zeros((2, CTX, 256), dtype=np.int8))
        VVQ[ai] = up(np.zeros((2, CTX, 256), dtype=np.int8))
        KVS[ai] = up(np.zeros((2, CTX, 2), dtype=np.float32))
        VVS[ai] = up(np.zeros((2, CTX, 2), dtype=np.float32))
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

    gridf = up(iq3s_grid_f32()); iq4nl = up(iq4nl_f32())
    def layer_seq(L):
        """returns the eager step list for layer L (reads hA, writes hA via cmb)"""
        w = W[L]
        seq = [("rmsz2048g", (hA, w["an"], hnb), P, ())]
        if L in GDN_LAYERS:
            gi = GDN_LAYERS.index(L)
            seq += [("gv8k2048p", (w["qkv"], hnb, qkvb), 256, (8192,)),
                    ("gv8k2048p", (w["z"], hnb, zb), 128, (4096,)),
                    ("gvf32ab", (w["wa"], w["wb"], hnb, abb), P, ()),
                    ("gconv36_1", (w["cw"], qkvb, CSV[gi], qkvsb), 32, ()),
                    ("k2s36_1", (qkvsb, abb, w["al"], w["dt"], w["sn"], zb, SV[gi], gyb), 32, ()),
                    ("gv8k4096r", (w["out"], gyb, hA, hB), 64, (2048,))]
        else:
            ai = ATTN_LAYERS.index(L)
            seq += [("gv8k2048p", (w["q"], hnb, qgb), 256, (8192,)),
                    ("gv8k2048p", (w["k"], hnb, kqb), 16, (512,)),
                    ("gv8k2048p", (w["v"], hnb, vqb), 16, (512,)),
                    ("spka256", (kqb, vqb, SPTB[ai]), 2, ()),
                    ("spkq256", (qgb, ayb, SPTB[ai]), 16, ()),
                    ("gv8k4096r", (w["o"], ayb, hA, hB), 64, (2048,))]
        seq += [("rmsz2048g", (hB, w["pn"], hnb), P, ()),
                ("rt8e256", (w["rt"], w["wsh"], hnb, eidsb, gatesb, sgb), P, ()),
                ("shexp8", (w["sg"], w["su"], w["sd"], hnb, shb), P, ())]
        upk = "gx8e256up4" if man["routed"][L]["types"]["gate"] == "IQ4_XS" else "gx8e256up"
        upbufs = (PTB_UP[L], eidsb, hnb, iq4nl, actb) if upk == "gx8e256up4" else (PTB_UP[L], eidsb, hnb, gridf, actb)
        seq.append((upk, upbufs, P*8, ()))
        dnk = "gx8e256dn6" if man["routed"][L]["types"]["down"] == "Q6_K" else "gx8e256dn"
        dnbufs = (PTB_DN[L], eidsb, actb, partsb) if dnk == "gx8e256dn6" else (PTB_DN[L], eidsb, actb, iq4nl, partsb)
        seq.append((dnk, dnbufs, P*8, ()))
        seq.append(("cmbz2048", (partsb, gatesb, sgb, shb, hB, hA), P, ()))
        return seq
    LSEQS = [layer_seq(L) for L in range(40)]

    def run_L(L):
        for name, bufs, gx, vals in LSEQS[L]:
            K[name](*bufs, global_size=(gx,1,1), local_size=LSZ[name], vals=vals, wait=True)

    report = []
    for pi, pos in TARGETS:
        ids = [int(x) for x in battery_ids[pi]]
        reset_states()
        for p in range(pos + 1):
            dev.allocator._copyin(idsb, memoryview(np.array([ids[p]], dtype=np.int32).tobytes()))
            dev.allocator._copyin(POSB, memoryview(np.array([p], dtype=np.int32).tobytes()))
            K["embg248"](EMB, idsb, hA, global_size=(1,1,1), local_size=LSZ["embg248"], vals=(1,), wait=True)
            hs = [dn(hA, (2048,)).copy()]
            for L in range(40):
                run_L(L)
                hs.append(dn(hA, (2048,)).copy())
            K["rmsz2048g"](hA, ONORM, normhb, global_size=(1,1,1), local_size=LSZ["rmsz2048g"], wait=True)
            hs.append(dn(normhb, (2048,)).copy())
            if p == pos:
                K["h6k2048"](HEAD, normhb, logitsb, global_size=(7760,1,1), local_size=LSZ["h6k2048"], vals=(248320,), wait=True)
                lg = dn(logitsb, (248320,))
        htr_a, lg_a, t1_a = anchor_dump[(pi, pos)]
        devs = [float(np.abs(g - a).max()) for g, a in zip(hs, htr_a)]
        srt = np.sort(lg)[::-1]
        srt_a = np.sort(lg_a)[::-1]
        t5 = np.argsort(-lg)[:5].tolist()
        t5a = np.argsort(-lg_a)[:5].tolist()
        print(f"\n[disc] prompt {pi} pos {pos}:", flush=True)
        print(f"  per-layer maxdev: embed={devs[0]:.2e} " +
              " ".join(f"L{L}={devs[L+1]:.1e}" for L in range(0, 40, 4)) +
              f" finalnorm={devs[41]:.1e}", flush=True)
        jump = max(range(41), key=lambda i: devs[i+1] / max(devs[i], 1e-9))
        print(f"  max jump ratio at entry {jump} (x{devs[jump+1]/max(devs[jump],1e-9):.1f})", flush=True)
        print(f"  engine top1={int(np.argmax(lg))} gap={srt[0]-srt[1]:.4f} top5={t5}", flush=True)
        print(f"  anchor top1={int(t1_a)} gap={srt_a[0]-srt_a[1]:.4f} top5={t5a}", flush=True)
        print(f"  logits maxdev={np.abs(lg-lg_a).max():.3f} (scale {np.abs(lg).max():.1f})", flush=True)
        report.append((pi, pos, devs, t5, t5a))
    print("[disc] DONE", flush=True)

if __name__ == "__main__":
    main()
