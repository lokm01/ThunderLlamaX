#!/usr/bin/env python3
"""MM P6 RUN — the speculative decode path (n-gram LOOKUP + D2/D8 + accept +
partial-state commit) + THE Tier-1 gate at depth.

  G1  kernel gates: k2s36s/gconv36s chains + SLOT snapshots vs the numpy
      refs; selc36 commit; h6kam+amred36 vs per-row h6k2048+np.argmax;
      spkq256m bit-exact vs spkq256/spkq_h_ref at overlapping L.
  G2  P-batch exactness core: same state, T1-step top1/state vs the D8-slot
      probe's seat-0 top1 + slot-0 state.
  G3  THE SPEC ENGINE: lookup-driven K-mix (n>=8 -> D8, 4<=n<8 -> D2, else
      T1) + accept(m) + selc36 commit. GATE: spec continuations == pure-T1
      continuations on the battery, det x2.
  G4  E[m|hit] + sel-mode tok/s on quote-class + prose-class.
Progress ~/mm_p6_progress.txt.
"""
import os, sys, time, json
import numpy as np

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE); sys.path.insert(0, BASE + "/engine0")
PROG = os.path.expanduser("~/mm_p6_progress.txt")
from MM_P56_lib import feed_token, t1_argmax, amds_read, LSZ, GraphRunner

def record(tag, result):
    with open(PROG, "a") as f: f.write(f"{tag} {result}\n"); f.flush(); os.fsync(f.fileno())
    print(f"[PROGRESS] {tag} {result}", flush=True)
def done(tag):
    if not os.path.exists(PROG): return False
    return any(l.split()[0] == tag for l in open(PROG).read().splitlines() if l.split())

# ---------------- THE HOST LOOKUP (the 27B n-gram machinery, model-agnostic) ----------------
class Lookup:
    """Longest-suffix n-gram scan of the token history; drafts = the tokens
    that followed the earlier occurrence. LMIN=8 full-K; 4..7 -> the D2 class
    (the K-mix). Deterministic (latest occurrence, longest match first)."""
    def __init__(self):
        self.idx = {}
    def extend(self, hist):
        for p in range(max(0, len(hist) - 17), len(hist) - 7):
            self.idx.setdefault(tuple(hist[p:p+8]), []).append(p)
    def scan(self, hist, kmax=8):
        n_h = len(hist)
        for n in range(min(16, n_h - 8), 7, -1):        # the >=8 class
            suf = tuple(hist[n_h-n:n_h])
            for pos in reversed(self.idx.get(suf, [])):
                if pos + n >= n_h: continue             # the tail occurrence itself
                drafts = hist[pos+n:pos+n+kmax]
                if len(drafts) >= kmax: return n, list(drafts[:kmax])
                if len(drafts) >= 2: return n, list(drafts)
                break
        for n in range(min(7, n_h - 8), 3, -1):          # the 4..7 class
            suf = tuple(hist[n_h-n:n_h])
            for p2 in range(n_h - n - 1, -1, -1):
                if tuple(hist[p2:p2+n]) == suf:
                    drafts = hist[p2+n:p2+n+2]
                    if len(drafts) == 2: return n, list(drafts)
                    break
        return 0, []

def g1_gates(rig):
    from MM_P56_lib import Rig, GraphRunner, MG, feed_token, t1_argmax, amds_read, LSZ
    from MM_P34_ports import (Anchor, fresh_state, gconv_ref, k2s_ref, GDN_LAYERS,
        ATTN_LAYERS, load_f32, load_q8, rope_tables, kv_quant, spka_ref, spkq_h_ref, _f)
    from MM_P2_ports import _dot_lane_ref, dq_q8_0

    # ================= G1: kernel gates =================
    if not done("G1a"):
        rng = np.random.default_rng(101)
        alog = load_f32(0, "ssm_a"); dtb = load_f32(0, "ssm_dt_bias")
        wn = load_f32(0, "ssm_norm_weight")
        adt = np.concatenate([alog, dtb])
        for T, key in ((3, "k2s36s_3"), (9, "k2s36s_9")):
            qkvs = (rng.standard_normal((T, 8192)) * 0.2).astype(np.float32)
            ab = (rng.standard_normal((T, 64)) * 0.5).astype(np.float32)
            z = (rng.standard_normal((T, 4096)) * 0.2).astype(np.float32)
            S0 = (rng.standard_normal((32, 128, 128)) * 0.05).astype(np.float32)
            qk, abk, zk = rig.up(qkvs), rig.up(ab), rig.up(z)
            adtk, wnk = rig.up(adt), rig.up(wn)
            Sb = rig.up(S0.copy()); yb = rig.alloc(T*4096*4)
            SLOTS = rig.alloc(9*30*32*128*128*4)   # the REAL [9][30] t-major engine layout
            slo = SLOTS.offset(offset=0, size=32*128*128*4)  # layer-0 slice
            rig.K[key](qk, abk, adtk, wnk, zk, Sb, yb, slo, global_size=(32,1,1), local_size=LSZ[key], wait=True)
            gy = rig.dn(yb, (T, 4096)); gS = rig.dn(Sb, (32,128,128))
            slots = rig.dn(SLOTS, (9, 30, 32*128*128))[:, 0, :].reshape(9, 32, 128, 128)
            # numpy chain with per-t states
            St = S0.copy(); devs = []; sdevs = []
            for t in range(T):
                yr, St = k2s_ref(qkvs[t:t+1], ab[t:t+1], alog, dtb, wn, z[t:t+1], St)
                devs.append(float(np.abs(gy[t] - yr[0]).max()))
                sdevs.append(float(np.abs(slots[t] - St).max()))
            ok_all = all(sd < 3e-5 for sd in sdevs)
            # the kernel-internal copy: slot T-1 must be BIT-IDENTICAL to live S
            copy_ok = np.array_equal(slots[T-1].ravel(), gS.ravel())
            record("G1a", f"{key}: y maxdev {max(devs):.2e} slot-chain maxdev {max(sdevs):.2e} {'OK' if ok_all else 'BAD'} slot[T-1]==live {'BIT-EXACT' if copy_ok else 'BAD'}")
    if not done("G1b"):
        rng = np.random.default_rng(103)
        wreal = load_f32(0, "ssm_conv1d_weight").reshape(8192, 4)
        for T, key in ((3, "gconv36s_3"), (9, "gconv36s_9")):
            xin = (rng.standard_normal((T, 8192)) * 0.3).astype(np.float32)
            st0 = (rng.standard_normal((8192, 3)) * 0.3).astype(np.float32)
            wb, xb, sb = rig.up(wreal), rig.up(xin), rig.up(st0.copy())
            yb = rig.alloc(T*8192*4)
            CS = rig.alloc(9*30*8192*3*4)           # the REAL [9][30] t-major engine layout
            clo = CS.offset(offset=0, size=8192*3*4)
            rig.K[key](wb, xb, sb, yb, clo, global_size=(32,1,1), local_size=LSZ[key], wait=True)
            gy = rig.dn(yb, (T, 8192)); gs = rig.dn(sb, (8192,3))
            yr, stf = gconv_ref(wreal, xin, st0)
            cs = rig.dn(CS, (9, 30, 8192*3))[:, 0, :].reshape(9, 8192, 3)
            St = st0.copy(); ok = True
            for t in range(T):
                _, St = gconv_ref(wreal, xin[t:t+1], St)
                ok &= np.abs(cs[t] - St).max() == 0.0
            record("G1b", f"{key}: y maxdev {np.abs(gy-yr).max():.2e} state {np.abs(gs-stf).max():.2e} conv-slots {'BIT-EXACT' if ok else 'BAD'}")
    if not done("G1c"):
        rng = np.random.default_rng(107)
        SLOTS = rig.alloc(9*30*32*128*128*4); CS = rig.alloc(9*30*8192*3*4)
        base = rng.standard_normal(9*30*524288).astype(np.float32) * 0.01
        cbase = rng.standard_normal(9*30*24576).astype(np.float32) * 0.01
        rig.dev.allocator._copyin(SLOTS, memoryview(np.ascontiguousarray(base).data.cast("B")))
        rig.dev.allocator._copyin(CS, memoryview(np.ascontiguousarray(cbase).data.cast("B")))
        m = 3
        rig.dev.allocator._copyin(rig.MB, memoryview(np.array([m], dtype=np.int32).tobytes()))
        rig.K["selc36"](SLOTS, CS, rig.MB, rig.SALL, rig.CSALL, global_size=(8040,1,1), local_size=LSZ["selc36"], wait=True)
        gS = rig.dn(rig.SALL, (30*32*128*128,)); gC = rig.dn(rig.CSALL, (30*24576,))
        okS = np.array_equal(gS, base[m*15728640:(m+1)*15728640])
        okC = np.array_equal(gC, cbase[m*737280:(m+1)*737280])
        record("G1c", f"selc36 m=3: GDN {'BIT-EXACT' if okS else 'BAD'} conv {'BIT-EXACT' if okC else 'BAD'}")
    if not done("G1d"):
        rng = np.random.default_rng(109)
        X = (rng.standard_normal((9, 2048)) * 0.3).astype(np.float32)
        XB = rig.up(X)
        OB = rig.alloc(9*248320*4)   # full logits per seat for the ref path
        # ref: per-seat h6k2048 + np.argmax
        ref_amds = []
        for p in range(9):
            dst = OB.offset(offset=p*248320*4, size=248320*4)
            rig.K["h6k2048"](rig.HEAD, XB.offset(offset=p*2048*4, size=2048*4), dst,
                             global_size=(7760,1,1), local_size=LSZ["h6k2048"], vals=(248320,), wait=True)
            ref_amds.append(int(np.argmax(rig.dn(dst, (248320,)))))
        for P, key in ((3, "h6kam_3"), (9, "h6kam_9")):
            NB = rig.up(X[:P].copy())
            PART = rig.alloc(P*7760*8)
            rig.dev.allocator._copyin(rig.AMDB, memoryview(np.zeros(9, dtype=np.int32).tobytes()))
            rig.K[key](rig.HEAD, NB, PART, global_size=(7760,1,1), local_size=LSZ[key], wait=True)
            rig.K["amred36"](PART, rig.AMDB, global_size=(1,1,1), local_size=LSZ["amred36"], vals=(P,), wait=True)
            got = amds_read(rig, P)
            record("G1d", f"{key}+amred36: amds {got} ref {ref_amds[:P]} {'EXACT' if got == ref_amds[:P] else 'MISMATCH'}")
        # logits identity: h6kam row logits (unpacked best) vs h6k2048 — implied by amds match + the packed max;
        # tie-class check on synthetic near-tie rows:
        X2 = np.zeros((3, 2048), dtype=np.float32)   # all-zero h -> exact ties
        X2B = rig.up(X2)
        PART2 = rig.alloc(3*7760*8)
        rig.K["h6kam_3"](rig.HEAD, X2B, PART2, global_size=(7760,1,1), local_size=LSZ["h6kam_3"], wait=True)
        rig.K["amred36"](PART2, rig.AMDB, global_size=(1,1,1), local_size=LSZ["amred36"], vals=(3,), wait=True)
        zt = rig.up(X2[0].copy()); zdst = rig.alloc(248320*4)
        rig.K["h6k2048"](rig.HEAD, zt, zdst, global_size=(7760,1,1), local_size=LSZ["h6k2048"], vals=(248320,), wait=True)
        record("G1d2", f"h6kam tie-class: amds {amds_read(rig,3)} vs np.argmax {int(np.argmax(rig.dn(zdst, (248320,))))} (all-zero h = exact ties; lower-row wins)")
    if not done("G1e"):
        rng = np.random.default_rng(113)
        L = 900
        cos, sin = rope_tables(4096)
        qw = load_f32(3, "attn_q_norm_weight"); kw = load_f32(3, "attn_k_norm_weight")
        hn = (rng.standard_normal(2048) * 0.4).astype(np.float32)
        wq = load_q8(3, "attn_q_weight", 8192); wk = load_q8(3, "attn_k_weight", 512); wv = load_q8(3, "attn_v_weight", 512)
        qg = _dot_lane_ref(dq_q8_0(wq, 2048), hn, 64, 1)
        kq = _dot_lane_ref(dq_q8_0(wk, 2048), hn, 64, 1)
        vq = _dot_lane_ref(dq_q8_0(wv, 2048), hn, 64, 1)
        pos = L - 1
        # old-form buffers (CTXS=1024) — L=900 <= 1024 so both kernels legal
        Kq1 = np.zeros((2, 1024, 256), dtype=np.int8); Ks1 = np.ones((2, 1024, 2), dtype=np.float32)
        Vq1 = np.zeros((2, 1024, 256), dtype=np.int8); Vs1 = np.ones((2, 1024, 2), dtype=np.float32)
        for j in range(2):
            for p in range(pos+1):
                kr = kq[j*256:(j+1)*256].astype(np.float32) * (0.9 ** (p*0.01))
                vr = vq[j*256:(j+1)*256].astype(np.float32) * (0.9 ** (p*0.01))
                kn = _f((lambda x: x)(kr))
                Kq1[j, p], Ks1[j, p] = kv_quant(kn)
                Vq1[j, p], Vs1[j, p] = kv_quant(vr)
        qgb = rig.up(qg); ayb1 = rig.alloc(4096*4); ayb2 = rig.alloc(4096*4)
        POSB2 = rig.up(np.array([pos], dtype=np.int32))
        # path 1: the P34 spkq256 (CTXS=1024, smem form)
        KVQ1 = rig.up(Kq1.copy()); KVS1 = rig.up(Ks1.copy()); VVQ1 = rig.up(Vq1.copy()); VVS1 = rig.up(Vs1.copy())
        T1 = rig.up(np.array([rig.W[3]["kw"].va_addr, rig.W[3]["qw"].va_addr, rig.COSB.va_addr, rig.SINB.va_addr,
                              KVQ1.va_addr, KVS1.va_addr, VVQ1.va_addr, VVS1.va_addr, POSB2.va_addr], dtype=np.uint64))
        rig.K["spkq256"](qgb, ayb1, T1, global_size=(16,1,1), local_size=LSZ["spkq256"], wait=True)
        y1 = rig.dn(ayb1, (4096,))
        # path 2: spkq256m_4096 (CTXS=4096) on the same cache content
        Kq2 = np.zeros((2, 4096, 256), dtype=np.int8); Ks2 = np.ones((2, 4096, 2), dtype=np.float32)
        Vq2 = np.zeros((2, 4096, 256), dtype=np.int8); Vs2 = np.ones((2, 4096, 2), dtype=np.float32)
        Kq2[:, :1024] = Kq1; Ks2[:, :1024] = Ks1; Vq2[:, :1024] = Vq1; Vs2[:, :1024] = Vs1
        KVQ2 = rig.up(Kq2.copy()); KVS2 = rig.up(Ks2.copy()); VVQ2 = rig.up(Vq2.copy()); VVS2 = rig.up(Vs2.copy())
        T2 = rig.up(np.array([rig.W[3]["kw"].va_addr, rig.W[3]["qw"].va_addr, rig.COSB.va_addr, rig.SINB.va_addr,
                              KVQ2.va_addr, KVS2.va_addr, VVQ2.va_addr, VVS2.va_addr, POSB2.va_addr, rig.SPSCR.va_addr], dtype=np.uint64))
        rig.K["spkq256m_4096"](qgb, ayb2, T2, global_size=(16,1,1), local_size=LSZ["spkq256m_4096"], wait=True)
        y2 = rig.dn(ayb2, (4096,))
        # path 3: numpy ref
        yr = np.empty(4096, dtype=np.float32)
        for hh in range(16):
            jj = hh >> 3
            yr[hh*256:(hh+1)*256] = spkq_h_ref(qg, qw, Kq2[jj], Ks2[jj], Vq2[jj], Vs2[jj],
                                               cos[pos], sin[pos], pos, hh)
        # the PRE-FIX spkq256 (backup cubin) at L=200: the fix must be
        # BIT-IDENTICAL for L <= 256 (same expf/div/sum order)
        from tinygrad.device import TinyELF
        from tinygrad.runtime.ops_nv import NVProgram
        lib_old = open(f"{BASE}/MM_P34_spkq256.cubin.prefillfix.bak", "rb").read()
        Kold = NVProgram(rig.dev, TinyELF(lib=lib_old, name="spkq256", target=rig.dev.renderer.target, signature=tuple()))
        ayb3 = rig.alloc(4096*4)
        Kold(qgb, ayb3, T1, global_size=(16,1,1), local_size=LSZ["spkq256"], wait=True)
        y3 = rig.dn(ayb3, (4096,))
        # L=200 subtest: refill caches to 200 positions
        pos2 = 199
        for j in range(2):
            for p in range(pos, pos2, -1) if False else []: pass
        Kq1b = Kq1.copy(); Vq1b = Vq1.copy()
        POSB3 = rig.up(np.array([pos2], dtype=np.int32))
        T1b = rig.up(np.array([rig.W[3]["kw"].va_addr, rig.W[3]["qw"].va_addr, rig.COSB.va_addr, rig.SINB.va_addr,
                               KVQ1.va_addr, KVS1.va_addr, VVQ1.va_addr, VVS1.va_addr, POSB3.va_addr], dtype=np.uint64))
        Kold(qgb, ayb3, T1b, global_size=(16,1,1), local_size=LSZ["spkq256"], wait=True)
        y3b = rig.dn(ayb3, (4096,))
        rig.K["spkq256"](qgb, ayb1, T1b, global_size=(16,1,1), local_size=LSZ["spkq256"], wait=True)
        y1b = rig.dn(ayb1, (4096,))
        bit200 = np.array_equal(y1b, y3b)
        # the pre-fix kernel at L=900 must DIFFER (the bug demo) -- y3 holds it
        record("G1e", f"spkq256m vs spkq256(FIXED) @L=900: {'BIT-EXACT' if np.array_equal(y1, y2) else 'DIFF ' + str(float(np.abs(y1-y2).max()))} | both vs ref: old {np.abs(y1-yr).max():.2e} m {np.abs(y2-yr).max():.2e} | FIX bit-identical @L=200: {'YES' if bit200 else 'NO'} | pre-fix @L=900 vs ref: {np.abs(y3-yr).max():.2e} (the L>256 latent bug demo)")
    print("[G1 done]", flush=True)

# ================= G2: P-batch exactness core =================
def g2(rig, gr1, gr8, gr2, ids, npos_check=6):
    """At npos_check positions of a real prompt: T1-step vs the D8/D2 probe
    seat-0 (top1 AND slot-0 state) -- the spec==T1 core discriminator.
    Full-state snapshot/restore around every arm (S, CS, all 4 KV arrays)."""
    def snap_all():
        d = {"S": rig.dn(rig.SALL, (30*32*128*128,)), "CS": rig.dn(rig.CSALL, (30*24576,))}
        for ai in range(10):
            d[f"KQ{ai}"] = rig.dn(rig.KVQ[ai], (2*1024*256,), np.int8)
            d[f"KS{ai}"] = rig.dn(rig.KVS[ai], (2*1024*2,),)
            d[f"VQ{ai}"] = rig.dn(rig.VVQ[ai], (2*1024*256,), np.int8)
            d[f"VS{ai}"] = rig.dn(rig.VVS[ai], (2*1024*2,),)
        return d
    def restore(d):
        rig.dev.allocator._copyin(rig.SALL, memoryview(d["S"].data.cast("B")))
        rig.dev.allocator._copyin(rig.CSALL, memoryview(d["CS"].data.cast("B")))
        for ai in range(10):
            rig.dev.allocator._copyin(rig.KVQ[ai], memoryview(d[f"KQ{ai}"].data.cast("B")))
            rig.dev.allocator._copyin(rig.KVS[ai], memoryview(d[f"KS{ai}"].data.cast("B")))
            rig.dev.allocator._copyin(rig.VVQ[ai], memoryview(d[f"VQ{ai}"].data.cast("B")))
            rig.dev.allocator._copyin(rig.VVS[ai], memoryview(d[f"VS{ai}"].data.cast("B")))
    rig.reset_states(1024)
    res = {"t1": [], "d8": [], "d2": []}
    slot_seq_all = True
    for pos, tid in enumerate(ids):
        if pos >= npos_check: break
        pre = snap_all()
        # -- T1 arm --
        feed_token(rig, int(tid), pos)
        gr1.step()
        t1 = t1_argmax(rig)
        post = snap_all()
        # -- D8 arm --
        restore(pre)
        rig.dev.allocator._copyin(rig.idsb, memoryview(np.array([int(tid)] + [0]*8, dtype=np.int32).tobytes()))
        rig.dev.allocator._copyin(rig.POSB, memoryview(np.array([pos], dtype=np.int32).tobytes()))
        gr8.step()
        a9 = amds_read(rig, 9)
        slot0 = rig.dn(rig.SLOTS, (9, 30*32*128*128))[0]
        seq8 = np.array_equal(slot0, post["S"])
        slot_seq_all &= seq8
        # -- D2 arm --
        restore(pre)
        rig.dev.allocator._copyin(rig.idsb, memoryview(np.array([int(tid), 0, 0], dtype=np.int32).tobytes()))
        rig.dev.allocator._copyin(rig.POSB, memoryview(np.array([pos], dtype=np.int32).tobytes()))
        gr2.step()
        a3 = amds_read(rig, 3)
        res["t1"].append(t1); res["d8"].append(a9[0]); res["d2"].append(a3[0])
        print(f"    [G2] pos {pos}: t1={t1} d8={a9[0]} d2={a3[0]} slot0 {'EXACT' if seq8 else 'DIFF ' + str(float(np.abs(slot0 - post['S']).max()))}", flush=True)
        # -- committed advance = the T1 result --
        restore(post)
    ok9 = sum(1 for a, b in zip(res["t1"], res["d8"]) if a == b)
    ok3 = sum(1 for a, b in zip(res["t1"], res["d2"]) if a == b)
    return ok9, ok3, len(res["t1"]), slot_seq_all

# ================= THE LOOKUP (incremental 4-gram index) =================
class Lookup:
    """Longest-suffix n-gram scan (4..16) of the token history via an
    incremental 4-gram index; drafts = what followed the earlier occurrence.
    Deterministic: longest match; tie -> latest occurrence. kmax drafts for
    n>=8 (the D8 class); 2 for n in 4..7 (the D2 class)."""
    def __init__(self, hist=None):
        self.idx4 = {}; self.h = []
        for t in (hist or []): self.append(t)
    def append(self, t):
        self.h.append(t)
        L = len(self.h)
        if L >= 4:
            self.idx4.setdefault(tuple(self.h[-4:]), []).append(L - 4)
    def scan(self, kmax=8, maxn=16):
        L = len(self.h)
        if L < 9: return 0, []
        cands = self.idx4.get(tuple(self.h[-4:]), [])
        best = None
        for q0 in reversed(cands):
            if q0 + 4 > L - 4: continue
            n = 4
            qq = q0
            while n < maxn and qq > 0 and self.h[qq - 1] == self.h[L - n - 1]:
                qq -= 1; n += 1
            # the match window is [qq, qq+n) (qq+n == q0+4 invariant);
            # drafts = what followed THE EXTENDED MATCH at qq+n
            avail = L - (qq + n)
            if best is None or n > best[0]:
                best = (n, qq, avail)
            if n >= maxn: break
        if best is None or best[0] < 4 or best[2] < 2: return 0, []
        n, qq, avail = best
        if n >= 8 and avail >= kmax:
            return n, list(self.h[qq + n : qq + n + kmax])
        if n >= 8 and avail >= 2:
            return n, list(self.h[qq + n : qq + n + min(2, avail)])
        if n >= 4 and avail >= 2:
            return n, list(self.h[qq + n : qq + n + 2])
        return 0, []

# ================= G3+G4: THE SPEC ENGINE =================
class SpecEngine:
    """The K-mix selector: T1 (no hit) / D2 (4..7) / D8 (>=8 + hysteresis),
    n-gram lookup drafting, greedy accept + selc36 partial-state commit.
    mode: 'spec' = the full selector; 't1' = pure T1 baseline."""
    def __init__(self, rig, gr1, gr2, gr8):
        self.rig = rig; self.gr1 = gr1; self.gr2 = gr2; self.gr8 = gr8
        self.stats = None
    def reset_stats(self):
        self.stats = {"cyc": 0, "cyc_t1": 0, "cyc_d2": 0, "cyc_d8": 0, "tok": 0,
                      "ms": 0.0, "ms_t1": 0.0, "ms_d2": 0.0, "ms_d8": 0.0,
                      "hits": 0, "msum": 0, "m_hist": [], "n_hist": [], "commit_ms": 0.0}
    def feed_prompt(self, ids):
        rig = self.rig
        rig.reset_states(1024)
        lk = Lookup()
        for pos, tid in enumerate(ids):
            feed_token(rig, int(tid), pos)
            self.gr1.step()
            lk.append(int(tid))
        t = t1_argmax(rig)
        lk.append(t)
        return t, lk
    def generate(self, t, lk, pos0, ntok, mode="spec"):
        """Generate ntok tokens (t = the first, already committed to lk)."""
        rig = self.rig
        self.reset_stats()
        gen = [t]
        pos = pos0
        t0 = time.perf_counter()
        while len(gen) < ntok:
            cur = lk.h[-1]
            n, drafts = (0, [])
            if mode == "spec":
                n, drafts = lk.scan(kmax=8)
            if n >= 8 and len(drafts) >= 8:
                K = 8; gr = self.gr8; P = 9; var = "d8"
            elif n >= 4 and len(drafts) >= 2:
                K = 2; gr = self.gr2; P = 3; var = "d2"
            else:
                K = 0; var = "t1"
            if var == "t1":
                tc = time.perf_counter()
                feed_token(rig, cur, pos)
                self.gr1.step()
                t2 = t1_argmax(rig)
                self.stats["ms_t1"] += time.perf_counter() - tc
                self.stats["cyc_t1"] += 1
                gen.append(t2); lk.append(t2); pos += 1
            else:
                tc = time.perf_counter()
                rig.dev.allocator._copyin(rig.idsb, memoryview(np.array([cur] + drafts[:K], dtype=np.int32).tobytes()))
                rig.dev.allocator._copyin(rig.POSB, memoryview(np.array([pos], dtype=np.int32).tobytes()))
                gr.step()
                amds = amds_read(rig, P)
                m = 0
                while m < K and amds[m] == drafts[m]:
                    m += 1
                emitted = drafts[:m] + [amds[m]]
                gen.extend(emitted)
                for e in emitted: lk.append(e)
                pos += m + 1
                # partial-state commit from slot m
                tcc = time.perf_counter()
                rig.dev.allocator._copyin(rig.MB, memoryview(np.array([m], dtype=np.int32).tobytes()))
                rig.K["selc36"](rig.SLOTS, rig.CSLOTS, rig.MB, rig.SALL, rig.CSALL,
                                global_size=(8040,1,1), local_size=LSZ["selc36"], wait=True)
                self.stats["commit_ms"] += time.perf_counter() - tcc
                self.stats[f"ms_{var}"] += time.perf_counter() - tc
                self.stats[f"cyc_{var}"] += 1
                self.stats["hits"] += 1
                self.stats["msum"] += m
                self.stats["m_hist"].append(m)
                self.stats["n_hist"].append(n)
            self.stats["cyc"] += 1
        dt = time.perf_counter() - t0
        self.stats["tok"] = len(gen)
        self.stats["ms"] = dt * 1e3
        return gen

def spec_gates_and_perf(rig, prompts_ids, tok, quote_prompts, prog_tags):
    from MM_P56_lib import GraphRunner, LSZ
    seq1 = rig.build_seq(1, "gconv36_1", "k2s36_1", with_head=True, head_mode="full")
    seq2 = rig.build_seq(3, "gconv36s_3", "k2s36s_3", with_head=True, head_mode="am", slots=True)
    seq9 = rig.build_seq(9, "gconv36s_9", "k2s36s_9", with_head=True, head_mode="am", slots=True)
    gr1 = GraphRunner(rig, [(rig.K[n], b, g, v) for n, b, g, v in seq1], "g3t1")
    gr2 = GraphRunner(rig, [(rig.K[n], b, g, v) for n, b, g, v in seq2], "g3d2")
    gr8 = GraphRunner(rig, [(rig.K[n], b, g, v) for n, b, g, v in seq9], "g3d8")
    eng = SpecEngine(rig, gr1, gr2, gr8)
    # ---- G2 ----
    if not done(prog_tags[0]):
        ids0 = prompts_ids[0]
        ok9, ok3, ntot, slot_ok = g2(rig, gr1, gr8, gr2, ids0, npos_check=6)
        record(prog_tags[0], f"P-BATCH CORE: D8 seat0 top1 {ok9}/{ntot}, D2 {ok3}/{ntot}, slot0-state {'BIT-EXACT' if slot_ok else 'DIFF'}")
    # ---- G3: spec == T1 on the battery ----
    if not done(prog_tags[1]):
        NTOK = 32
        mism = []
        for pi, ids in enumerate(prompts_ids):
            t1, lk = eng.feed_prompt(ids)
            g_t1 = eng.generate(t1, lk, len(ids), NTOK, mode="t1")
            t1b, lkb = eng.feed_prompt(ids)
            g_sp = eng.generate(t1b, lkb, len(ids), NTOK, mode="spec")
            if g_t1 != g_sp[:len(g_t1)]:
                bad = next(i for i, (a, b) in enumerate(zip(g_t1, g_sp)) if a != b)
                mism.append((pi, bad, g_t1[max(0,bad-2):bad+3], g_sp[max(0,bad-2):bad+3]))
            print(f"    [G3] prompt {pi}: {'EXACT' if g_t1 == g_sp else 'MISMATCH'} spec-modes cyc={eng.stats['cyc']} d8={eng.stats['cyc_d8']} d2={eng.stats['cyc_d2']} t1={eng.stats['cyc_t1']}", flush=True)
        n_ok = len(prompts_ids) - len(mism)
        # det x2 on the spec path (first 6 prompts)
        detok = True
        for pi in range(min(6, len(prompts_ids))):
            t1b, lkb = eng.feed_prompt(prompts_ids[pi])
            g1 = eng.generate(t1b, lkb, len(prompts_ids[pi]), NTOK, mode="spec")
            t1b, lkb = eng.feed_prompt(prompts_ids[pi])
            g2v = eng.generate(t1b, lkb, len(prompts_ids[pi]), NTOK, mode="spec")
            detok &= (g1 == g2v)
        record(prog_tags[1], f"SPEC==T1: {n_ok}/{len(prompts_ids)} prompts exact (32-tok) det-x2 {'OK' if detok else 'FAIL'} mism={mism[:4]}")
    # ---- G3b: the 60-prompt battery spec==T1 (the mission's 60/60 gate) ----
    if not done("G3b"):
        from MM_P34_anchor import PROMPTS as P20A
        from MM_P5_anchor import PROMPTS40 as P40A
        P60ALL = P20A + P40A
        from MM_P34_tok import parse_gguf_kv as _pgk, SimpleTokenizer as _ST
        _tok = _ST.from_gguf_kv(_pgk(os.path.expanduser("~/models36/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")))
        ids60 = [_tok.encode(p) for p in P60ALL]
        mism = []
        for pi, ids in enumerate(ids60):
            t1v, lk = eng.feed_prompt(ids)
            g_t1 = eng.generate(t1v, lk, len(ids), 32, mode="t1")
            t1b, lkb = eng.feed_prompt(ids)
            g_sp = eng.generate(t1b, lkb, len(ids), 32, mode="spec")
            if g_t1 != g_sp[:len(g_t1)]:
                bad = next(i for i, (a, b) in enumerate(zip(g_t1, g_sp)) if a != b)
                mism.append((pi, bad))
            if pi % 10 == 0:
                print(f"    [G3b] prompt {pi}: {'EXACT' if g_t1 == g_sp[:len(g_t1)] else 'MISMATCH'} d8={eng.stats['cyc_d8']} d2={eng.stats['cyc_d2']}", flush=True)
        detok = True
        for pi in range(0, 60, 12):
            t1b, lkb = eng.feed_prompt(ids60[pi])
            g1 = eng.generate(t1b, lkb, len(ids60[pi]), 32, mode="spec")
            t1b, lkb = eng.feed_prompt(ids60[pi])
            g2v = eng.generate(t1b, lkb, len(ids60[pi]), 32, mode="spec")
            detok &= (g1 == g2v)
        record("G3b", f"SPEC==T1 60-PROMPT: {60 - len(mism)}/60 exact, det-x2 {'OK' if detok else 'FAIL'} mism={mism[:6]}")

    # ---- G4: E[m|hit] + the sel-mode tok/s table ----
    if not done(prog_tags[2]):
        rows = []
        for tag, texts, ntok in quote_prompts:
            ids = texts if isinstance(texts, list) else tok.encode(texts)
            t1b, lkb = eng.feed_prompt(ids)
            g = eng.generate(t1b, lkb, len(ids), ntok, mode="spec")
            st = eng.stats
            em = st["msum"] / max(1, st["hits"])
            gen_ms = st["ms"] - st["ms_t1"] - 0  # total incl t1 cycles
            row = (tag, st["tok"], st["cyc"], st["cyc_d8"], st["cyc_d2"], st["cyc_t1"],
                   round(em, 2), round(st["ms"]/max(1,st["tok"])*1e3, 2),
                   round(st["tok"]/max(1e-9, st["ms"]/1e3), 2),
                   g[:16])
            rows.append(row)
            print(f"    [G4] {tag}: {row}", flush=True)
        json.dump(rows, open(os.path.expanduser("~/mm_p6_perf.json"), "w"))
        rec = " | ".join(f"{r[0]}: {r[8]} tok/s (D8 {r[3]}/D2 {r[4]}/T1 {r[5]} cyc, E[m|hit] {r[6]})" for r in rows)
        record(prog_tags[2], f"PERF: {rec}")

def main(rig=None):
    from MM_P34_tok import parse_gguf_kv, SimpleTokenizer
    from MM_P34_anchor import PROMPTS as P20
    A16 = os.path.expanduser("~/mm_p5_anchor16.npz")
    A60 = os.path.expanduser("~/mm_p5_anchor60.npz")
    AOLD = os.path.expanduser("~/mm_p34_anchor.npz")
    if rig is None: rig = Rig()
    g1_gates(rig)
    apath = A60 if os.path.exists(A60) else (A16 if os.path.exists(A16) else AOLD)
    a = np.load(apath, allow_pickle=True)
    ids_list = [np.asarray(x, dtype=np.int32) for x in a["ids"]]
    print(f"[P6] battery: {len(ids_list)} prompts from {apath}", flush=True)
    kv = parse_gguf_kv(os.path.expanduser("~/models36/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"))
    tok = SimpleTokenizer.from_gguf_kv(kv)
    doc_txt = open(os.path.expanduser("~/prompt100k.txt"), encoding="utf-8", errors="replace").read()
    passage = doc_txt[3000:3900]
    half = passage[:len(passage)//2]
    qp_docx2 = f"Here is a passage:\n{passage}\nNow repeat the passage exactly, word for word:\n{half}"
    code = "def process(items):\n    out = []\n    for it in items:\n        if it is not None:\n            out.append(it.strip())\n    return out\n"
    qp_code = f"{code}\nThe same function again:\ndef process(items):"
    quote_prompts = [
        ("quote-docx2", qp_docx2, 48),
        ("quote-alpha", P20[7], 48),
        ("quote-code", qp_code, 48),
        ("prose-0", P20[0], 32),
        ("prose-2", P20[2], 32),
        ("prose-9", P20[9], 32),
    ]
    spec_gates_and_perf(rig, ids_list, tok, quote_prompts, ["G2", "G3", "G4"])
    print("[P6 ALL DONE]", flush=True)

if __name__ == "__main__":
    main()
