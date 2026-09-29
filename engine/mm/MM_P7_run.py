#!/usr/bin/env python3
"""MM P7 RUN — the three-part session (resumable, ~/mm_p7_progress.txt):

  H   THE 140 CROSS (host fold): H0 cpu-view coherence + acc36 unit;
      H2 spec==T1 re-gate (20 + 60 prompts) on the FOLDED engine; H3 the
      quote/prose perf table + the cycle split (before: 66ms = 49.6 trunk
      + ~14 host / 118.6 quote-alpha).
  S   SPLIT-KV: S1 spkq256s+spkc256 vs the split numpy anchor (L=900,S=4)
      + det x2; S2 split-vs-serial magnitude; S3 the O(L) timing evidence.
  F   PREFILL: F1 chunk-consistency (PF-256 + T1-split vs per-token T1-split,
      bit-exact by construction); F2 doc2048 continuation + anchor spot;
      F3 the tok/s ladder (2k/8k/16k, split S=8).
  L   THE CTX LADDER: L64 (64440) + L96 (97700): PF-chunk feed + T1-split
      cont32 det x2 + rebase16 vs the SPLIT anchor + first-64 + GDN |S| +
      decode tok/s at depth.
  D   audit.
"""
import os, sys, time, json
import numpy as np

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE); sys.path.insert(0, BASE + "/engine0")
PROG = os.path.expanduser("~/mm_p7_progress.txt")
DOC = os.path.expanduser("~/mm_p5_doc100k_ids.npy")
A60 = os.path.expanduser("~/mm_p5_anchor60.npz")
A16 = os.path.expanduser("~/mm_p5_anchor16.npz")
AOLD = os.path.expanduser("~/mm_p34_anchor.npz")

def record(tag, result):
    with open(PROG, "a") as f: f.write(f"{tag} {result}\n"); f.flush(); os.fsync(f.fileno())
    print(f"[PROGRESS] {tag} {result}", flush=True)
def done(tag):
    if not os.path.exists(PROG): return False
    return any(l.split()[0] == tag for l in open(PROG).read().splitlines() if l.split())

def load_anchor(path):
    a = np.load(path, allow_pickle=True)
    return ([np.asarray(x, dtype=np.int32) for x in a["ids"]],
            [np.asarray(x, dtype=np.int32) for x in a["top1"]])

# ============================================================================
def stage_h(rig, doc):
    from MM_P7_lib import build_seq7, mkgraph, SpecEngine7
    seq1 = build_seq7(rig, 1, "gconv36_1", "k2s36_1", with_head=True, head_mode="full")
    gr1 = mkgraph(rig, seq1, "h_t1")
    ids20, _ = load_anchor(A16 if os.path.exists(A16) else AOLD)

    # -- H0a: cpu-view write visibility vs the anchor bank (end-to-end) --
    if not done("H0a"):
        ids60a, t60a = load_anchor(A60)
        rig.reset_states(1024)
        rig.feed(int(ids60a[0][0]), 0)
        gr1.step()
        top = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
        exp = int(t60a[0][0])
        record("H0a", f"view-fed bank-prompt-0 token {int(ids60a[0][0])} -> top1 {top} (anchor {exp}) {'OK' if top == exp else 'FAIL'}")
    # -- H0b: GPU->host cpu-view read coherence (acc36 + eb) --
    # NOTE the ids layout: ids[0] = cur (unused), ids[1+i] = drafts[i].
    if not done("H0b"):
        rng = np.random.default_rng(77)
        bad = 0; cases = 0; edge = ""
        for trial in range(120):
            K = 8 if trial % 2 == 0 else 2
            amds = rng.integers(0, 248320, size=9).astype(np.int32)
            drafts = rng.integers(0, 248320, size=8).astype(np.int32)
            if trial % 3 == 0:  # force full-accept m=K
                amds[:K] = drafts[:K]
            if trial % 5 == 0:  # force m=0
                amds[0] = (drafts[0] + 1) % 248320
            rig.am_view[:] = memoryview(np.ascontiguousarray(amds, dtype=np.int32).data)
            rig.ids_view[0] = 12345
            for i in range(8): rig.ids_view[1 + i] = int(drafts[i])
            dv = rig.idsb.offset(offset=4, size=8*4)
            rig.K["acc36"](rig.AMDB, dv, rig.MB, rig.EB, global_size=(1,1,1),
                           local_size=(256,1,1), vals=(K,), wait=True)
            m = 0
            while m < K and int(amds[m]) == int(drafts[m]): m += 1
            gm, gb = int(rig.eb_view[0]), int(rig.eb_view[1])
            cases += 1
            if gm != m or gb != int(amds[m]): bad += 1; edge = f"trial {trial}: got ({gm},{gb}) want ({m},{amds[m]})"
        record("H0b", f"acc36+EB cpu-view coherence: {cases-bad}/{cases} OK (m, amds[m]) {'PASS' if bad==0 else 'FAIL '+edge}")

    # -- H2: the folded engine, spec == T1 re-gate --
    seq2 = build_seq7(rig, 3, "gconv36s_3", "k2s36s_3", with_head=True, head_mode="am", slots=True, tail_acc_K=2)
    seq8 = build_seq7(rig, 9, "gconv36s_9", "k2s36s_9", with_head=True, head_mode="am", slots=True, tail_acc_K=8)
    gr2 = mkgraph(rig, seq2, "h_d2")
    gr8 = mkgraph(rig, seq8, "h_d8")
    eng = SpecEngine7(rig, gr1, gr2, gr8)
    if not done("H2"):
        NTOK = 32
        mism = []
        for pi, ids in enumerate(ids20[:20]):
            t1, lk = eng.feed_prompt(ids)
            g_t1 = eng.generate(t1, lk, len(ids), NTOK, mode="t1")
            t1b, lkb = eng.feed_prompt(ids)
            g_sp = eng.generate(t1b, lkb, len(ids), NTOK, mode="spec")
            if g_t1 != g_sp[:len(g_t1)]:
                bad = next(i for i, (a, b) in enumerate(zip(g_t1, g_sp)) if a != b)
                mism.append((pi, bad))
            print(f"    [H2] prompt {pi}: {'EXACT' if g_t1 == g_sp[:len(g_t1)] else 'MISMATCH'} d8={eng.stats['cyc_d8']} d2={eng.stats['cyc_d2']}", flush=True)
        record("H2", f"FOLDED spec==T1 20-prompt: {20-len(mism)}/20 exact mism={mism[:4]}")
    if not done("H2b"):
        ids60, _ = load_anchor(A60)
        mism = []
        for pi, ids in enumerate(ids60):
            t1, lk = eng.feed_prompt(ids)
            g_t1 = eng.generate(t1, lk, len(ids), 32, mode="t1")
            t1b, lkb = eng.feed_prompt(ids)
            g_sp = eng.generate(t1b, lkb, len(ids), 32, mode="spec")
            if g_t1 != g_sp[:len(g_t1)]: mism.append((pi, next(i for i,(a,b) in enumerate(zip(g_t1,g_sp)) if a != b)))
            if pi % 10 == 0: print(f"    [H2b] {pi}: {'EXACT' if not mism or mism[-1][0]!=pi else 'MISMATCH'}", flush=True)
        detok = True
        for pi in range(0, 60, 12):
            t1b, lkb = eng.feed_prompt(ids60[pi])
            g1 = eng.generate(t1b, lkb, len(ids60[pi]), 32, mode="spec")
            t1b, lkb = eng.feed_prompt(ids60[pi])
            g2 = eng.generate(t1b, lkb, len(ids60[pi]), 32, mode="spec")
            detok &= (g1 == g2)
        record("H2b", f"FOLDED spec==T1 60-prompt: {60-len(mism)}/60 exact, det-x2 {'OK' if detok else 'FAIL'} mism={mism[:6]}")

    # -- H3: the perf table (the 140 cross) --
    if not done("H3"):
        from MM_P34_tok import parse_gguf_kv, SimpleTokenizer
        from MM_P34_anchor import PROMPTS as P20
        tok = SimpleTokenizer.from_gguf_kv(parse_gguf_kv(os.path.expanduser("~/models36/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")))
        doc_txt = open(os.path.expanduser("~/prompt100k.txt"), encoding="utf-8", errors="replace").read()
        passage = doc_txt[3000:3900]; half = passage[:len(passage)//2]
        qp_docx2 = f"Here is a passage:\n{passage}\nNow repeat the passage exactly, word for word:\n{half}"
        code = "def process(items):\n    out = []\n    for it in items:\n        if it is not None:\n            out.append(it.strip())\n    return out\n"
        qp_code = f"{code}\nThe same function again:\ndef process(items):"
        batteries = [("quote-alpha", P20[7], 48), ("quote-code", qp_code, 48),
                     ("quote-docx2", qp_docx2, 48), ("prose-0", P20[0], 32), ("prose-9", P20[9], 32)]
        rows = []
        for tag, txt, ntok in batteries:
            ids = tok.encode(txt)
            t1b, lkb = eng.feed_prompt(ids)
            g = eng.generate(t1b, lkb, len(ids), ntok, mode="spec")
            st = eng.stats
            em = st["msum"] / max(1, st["hits"])
            row = dict(tag=tag, tok=st["tok"], cyc=st["cyc"], d8=st["cyc_d8"], d2=st["cyc_d2"], t1=st["cyc_t1"],
                       em=round(em,2), ms_per_tok=round(st["ms"]/max(1,st["tok"]),3),
                       tps=round(st["tok"]/(st["ms"]/1e3),2),
                       cyc_d8_ms=round(st["ms_d8"]/max(1,st["cyc_d8"]),3) if st["cyc_d8"] else 0,
                       wait_d8_ms=round(st["wait_ms"]/max(1,st["cyc_d8"]),3) if st["cyc_d8"] else 0)
            rows.append(row)
            print(f"    [H3] {row}", flush=True)
        json.dump(rows, open(os.path.expanduser("~/mm_p7_perf.json"), "w"))
        rec = " | ".join(f"{r['tag']}: {r['tps']} tok/s (E[m|hit] {r['em']}, D8cyc {r['d8']} @ {r['cyc_d8_ms']}ms, wait {r['wait_d8_ms']}ms)" for r in rows)
        record("H3", f"THE 140 CROSS: {rec}")

# ============================================================================
def stage_s(rig):
    from MM_P34_ports import load_f32, load_q8, rope_tables, kv_quant, spkq_h_ref, _f
    from MM_P2_ports import _dot_lane_ref, dq_q8_0
    from MM_P7_lib import spkq_h_split_ref
    rng = np.random.default_rng(113)
    L = 900; R = 16384
    cos, sin = rope_tables(R)
    qw = load_f32(3, "attn_q_norm_weight"); kw = load_f32(3, "attn_k_norm_weight")
    hn = (rng.standard_normal(2048) * 0.4).astype(np.float32)
    wq = load_q8(3, "attn_q_weight", 8192); wk = load_q8(3, "attn_k_weight", 512); wv = load_q8(3, "attn_v_weight", 512)
    qg = _dot_lane_ref(dq_q8_0(wq, 2048), hn, 64, 1)
    kq = _dot_lane_ref(dq_q8_0(wk, 2048), hn, 64, 1)
    vq = _dot_lane_ref(dq_q8_0(wv, 2048), hn, 64, 1)
    pos = L - 1
    Kq2 = np.zeros((2, R, 256), dtype=np.int8); Ks2 = np.ones((2, R, 2), dtype=np.float32)
    Vq2 = np.zeros((2, R, 256), dtype=np.int8); Vs2 = np.ones((2, R, 2), dtype=np.float32)
    for j in range(2):
        for p in range(pos+1):
            kr = kq[j*256:(j+1)*256].astype(np.float32) * (0.9 ** (p*0.01))
            vr = vq[j*256:(j+1)*256].astype(np.float32) * (0.9 ** (p*0.01))
            from MM_P34_ports import rmszc_ref
            kn = rmszc_ref(kr if False else _f(kr), kw)
            Kq2[j, p], Ks2[j, p] = kv_quant(kn)
            Vq2[j, p], Vs2[j, p] = kv_quant(_f(vr))
    KVQ2 = rig.up(Kq2.copy()); KVS2 = rig.up(Ks2.copy()); VVQ2 = rig.up(Vq2.copy()); VVS2 = rig.up(Vs2.copy())
    qgb = rig.up(qg)
    POSB2 = rig.up(np.array([pos], dtype=np.int32))
    TB = rig.up(np.array([rig.W[3]["kw"].va_addr, rig.W[3]["qw"].va_addr, rig.COSB.va_addr, rig.SINB.va_addr,
                          KVQ2.va_addr, KVS2.va_addr, VVQ2.va_addr, VVS2.va_addr, POSB2.va_addr], dtype=np.uint64))
    if not done("S1"):
        S = 4; NP = 8*S
        SCR = rig.alloc(16*S*8*258*4)
        ysplit = rig.alloc(4096*4)
        outs = []
        for rep in range(2):
            rig.K[f"spkq256s_{R}"](qgb, SCR, TB, global_size=(16,S,1), local_size=(256,1,1), vals=(S,), wait=True)
            rig.K["spkc256"](qgb, ysplit, SCR, global_size=(16,1,1), local_size=(256,1,1), vals=(NP,), wait=True)
            outs.append(rig.dn(ysplit, (4096,)))
        det = np.array_equal(outs[0], outs[1])
        devs = []
        for hh in range(16):
            jj = hh >> 3
            yr = spkq_h_split_ref(qg, qw, Kq2[jj], Ks2[jj], Vq2[jj], Vs2[jj], cos[pos], sin[pos], pos, hh, S)
            devs.append(float(np.abs(outs[0][hh*256:(hh+1)*256] - yr).max()))
        record("S1", f"spkq256s_16384+spkc256 S=4 L=900: vs split-anchor maxdev {max(devs):.2e} det-x2 {'BIT-EXACT' if det else 'FAIL'}")
    if not done("S2"):
        S = 4; NP = 8*S
        SCR = rig.alloc(16*S*8*258*4)
        ysplit = rig.alloc(4096*4)
        rig.K[f"spkq256s_{R}"](qgb, SCR, TB, global_size=(16,S,1), local_size=(256,1,1), vals=(S,), wait=True)
        rig.K["spkc256"](qgb, ysplit, SCR, global_size=(16,1,1), local_size=(256,1,1), vals=(NP,), wait=True)
        yser = rig.alloc(4096*4)
        TBM = rig.up(np.array([rig.W[3]["kw"].va_addr, rig.W[3]["qw"].va_addr, rig.COSB.va_addr, rig.SINB.va_addr,
                               KVQ2.va_addr, KVS2.va_addr, VVQ2.va_addr, VVS2.va_addr, POSB2.va_addr, rig.SPSCR.va_addr], dtype=np.uint64))
        rig.K[f"spkq256m_{R}"](qgb, yser, TBM, global_size=(16,), local_size=(256,1,1), wait=True)
        d = float(np.abs(rig.dn(ysplit, (4096,)) - rig.dn(yser, (4096,))).max())
        record("S2", f"split(S=4) vs serial(spkq256m) @L=900: maxdev {d:.2e} (the expected reassociation class, NOT bit-exact by design)")
    if not done("S3"):
        import time as _t
        # extend KV to full 16384 for the O(L) timing (fill the tail cheaply)
        Kq2[:, pos+1:] = 0; Vq2[:, pos+1:] = 1
        rig.dev.allocator._copyin(KVQ2, memoryview(np.ascontiguousarray(Kq2).data.cast("B")))
        rig.dev.allocator._copyin(VVQ2, memoryview(np.ascontiguousarray(Vq2).data.cast("B")))
        POSF = rig.up(np.array([R-1], dtype=np.int32))
        TBF = rig.up(np.array([rig.W[3]["kw"].va_addr, rig.W[3]["qw"].va_addr, rig.COSB.va_addr, rig.SINB.va_addr,
                               KVQ2.va_addr, KVS2.va_addr, VVQ2.va_addr, VVS2.va_addr, POSF.va_addr], dtype=np.uint64))
        yb = rig.alloc(4096*4)
        TBMF = rig.up(np.array([rig.W[3]["kw"].va_addr, rig.W[3]["qw"].va_addr, rig.COSB.va_addr, rig.SINB.va_addr,
                                KVQ2.va_addr, KVS2.va_addr, VVQ2.va_addr, VVS2.va_addr, POSF.va_addr, rig.SPSCR.va_addr], dtype=np.uint64))
        def t_serial():
            t0 = _t.perf_counter()
            rig.K[f"spkq256m_{R}"](qgb, yb, TBMF, global_size=(16,), local_size=(256,1,1), wait=True)
            return (_t.perf_counter()-t0)*1e3
        def t_split(S):
            SCR = rig.alloc(16*S*8*258*4)
            def run():
                rig.K[f"spkq256s_{R}"](qgb, SCR, TBF, global_size=(16,S,1), local_size=(256,1,1), vals=(S,), wait=True)
                rig.K["spkc256"](qgb, yb, SCR, global_size=(16,), local_size=(256,1,1), vals=(8*S,), wait=True)
            run()
            ts = []
            for _ in range(5):
                t0 = _t.perf_counter(); run(); ts.append((_t.perf_counter()-t0)*1e3)
            return min(ts)
        ser = min(t_serial() for _ in range(3))
        sp = {S: t_split(S) for S in (4, 8, 16)}
        record("S3", f"O(L) EVIDENCE @L=16384 P=1 (1 attn layer, min-of-N ms): serial {ser:.2f} | split S=4 {sp[4]:.2f} S=8 {sp[8]:.2f} S=16 {sp[16]:.2f} -> speedup {ser/sp[8]:.1f}x @S=8")

# ============================================================================
def pf_state_snap(rig, R, N):
    d = {"S": rig.dn(rig.SALL, (30,32,128,128)), "CS": rig.dn(rig.CSALL, (30,8192,3))}
    for ai in range(10):
        d[f"KQ{ai}"] = rig.dn(rig.KVQ[ai], (2*R*256,), np.int8)[:2*N*256].reshape(2,N,256)
        d[f"KS{ai}"] = rig.dn(rig.KVS[ai], (2*R*2,))[:2*N*2].reshape(2,N,2)
        d[f"VQ{ai}"] = rig.dn(rig.VVQ[ai], (2*R*256,), np.int8)[:2*N*256].reshape(2,N,256)
        d[f"VS{ai}"] = rig.dn(rig.VVS[ai], (2*R*2,))[:2*N*2].reshape(2,N,2)
    return d

def stage_f(rig, doc):
    from MM_P7_lib import build_seq7, mkgraph, PrefillFeed, install_split_anchor
    # split-class T1 + PF graphs at rung 16384 (both arms same class -> bit-comparable)
    seq1s = build_seq7(rig, 1, "gconv36_1", "k2s36_1", with_head=True, head_mode="full", spk="s16384", S=32)
    gr1s = mkgraph(rig, seq1s, "f_t1s")
    seqpf = build_seq7(rig, 256, "gconv36_256", "k2s36_256", with_head=False, spk="s16384", S=8, pf=True)
    grpf = mkgraph(rig, seqpf, "f_pf")
    pf = PrefillFeed(rig, grpf, gr1s)
    R = 16384
    if not done("F1"):
        N = 768
        # arm A: per-token T1-split
        rig.reset_states(R)
        for p in range(N):
            rig.feed(int(doc[p]), p)
            gr1s.step()
        topA = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
        snapA = pf_state_snap(rig, R, N)
        # arm B: 3 PF chunks
        rig.reset_states(R)
        chunks, pf_ms = pf.feed(doc[:N], 0)
        topB = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
        snapB = pf_state_snap(rig, R, N)
        eqs = [np.array_equal(snapA[k], snapB[k]) for k in snapA]
        # continuations from the SAME live state (arm B state in place now)
        genB = []
        t = topB
        for gix in range(16):
            genB.append(t)
            rig.feed(t, N + gix)
            gr1s.step()
            t = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
        # redo arm A continuation for compare
        rig.reset_states(R)
        for p in range(N):
            rig.feed(int(doc[p]), p); gr1s.step()
        genA = []
        t = topA
        for gix in range(16):
            genA.append(t)
            rig.feed(t, N + gix); gr1s.step()
            t = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
        record("F1", f"CHUNK CONSISTENCY doc768 @16384-split: state arrays {'ALL BIT-EXACT' if all(eqs) else 'DIFF '+str([k for k,e in zip(snapA,eqs) if not e])} | top1 {topA}=={topB} {'OK' if topA==topB else 'DIFF'} | cont16 {'IDENTICAL' if genA==genB else 'DIFF'} | pf {chunks} chunks {pf_ms:.0f}ms ({N/(pf_ms/1e3):.1f} tok/s)")
    if not done("F2"):
        from MM_P34_ports import Anchor, fresh_state
        N = 2048
        rig.reset_states(R)
        chunks, pf_ms = pf.feed(doc[:N], 0)
        tpf = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
        gen = []; t = tpf
        for gix in range(32):
            gen.append(t)
            rig.feed(t, N + gix); gr1s.step()
            t = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
        # det x2
        rig.reset_states(R)
        pf.feed(doc[:N], 0)
        gen2 = []; t = tpf
        for gix in range(32):
            gen2.append(t)
            rig.feed(t, N + gix); gr1s.step()
            t = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
        det = (gen == gen2)
        # anchor spot: rebase 8 from the snapshot (split anchor)
        snap = pf_state_snap(rig, R, N)
        install_split_anchor(32)
        anc = Anchor(ctx=R, fp16_partials=True)
        S2 = snap["S"].copy(); CS2 = snap["CS"].copy()
        KV2 = []
        for ai in range(10):
            lay = []
            for j in range(2):
                kqf = np.zeros((R, 256), np.int8); kqf[:N] = snap[f"KQ{ai}"][j]
                ksf = np.ones((R, 2), np.float32); ksf[:N] = snap[f"KS{ai}"][j]
                vqf = np.zeros((R, 256), np.int8); vqf[:N] = snap[f"VQ{ai}"][j]
                vsf = np.ones((R, 2), np.float32); vsf[:N] = snap[f"VS{ai}"][j]
                lay.append((kqf, ksf, vqf, vsf))
            KV2.append(lay)
        agen = []; t = tpf
        for gix in range(8):
            t, _, _ = anc.forward_token(t, N + gix, S2, CS2, KV2)
            agen.append(int(t))
        reb = sum(1 for a, b in zip(agen, gen[1:9]) if a == b)
        record("F2", f"PREFILL BATTERY doc2048: cont32 det-x2 {'OK' if det else 'FAIL'} | rebase8-vs-split-anchor {reb}/8 | top1 {tpf} | feed {chunks} chunks {pf_ms:.0f}ms ({N/(pf_ms/1e3):.1f} tok/s)")
        install_split_anchor(0)
    if not done("F3"):
        rows = []
        for N in (2048, 8192, 16384):
            rig.reset_states(R)
            t0 = time.perf_counter()
            chunks, pf_ms = pf.feed(doc[:N], 0)
            wall = time.perf_counter() - t0
            rows.append((N, chunks, round(pf_ms,1), round(N/(pf_ms/1e3),1)))
            print(f"    [F3] N={N}: {chunks} chunks, pf {pf_ms:.0f}ms -> {N/(pf_ms/1e3):.1f} tok/s (wall {wall:.1f}s)", flush=True)
        record("F3", "PREFILL LADDER (split S=8, chunk 256): " + " | ".join(f"N={n}: {tps} tok/s ({cms:.0f}ms, {c} chunks)" for n,c,cms,tps in rows))

# ============================================================================
def stage_l(rig, doc, R, N, tag):
    from MM_P7_lib import build_seq7, mkgraph, PrefillFeed, install_split_anchor
    from MM_P34_ports import Anchor, fresh_state
    seq1s = build_seq7(rig, 1, "gconv36_1", "k2s36_1", with_head=True, head_mode="full", spk=f"s{R}", S=32)
    gr1s = mkgraph(rig, seq1s, tag + "t1s")
    seqpf = build_seq7(rig, 256, "gconv36_256", "k2s36_256", with_head=False, spk=f"s{R}", S=8, pf=True)
    grpf = mkgraph(rig, seqpf, tag + "pf")
    pf = PrefillFeed(rig, grpf, gr1s)
    if done(tag): return
    t0 = time.time()
    rig.reset_states(R)
    # -- first-64 T1-split spot vs the split anchor --
    install_split_anchor(32)
    anc = Anchor(ctx=R, fp16_partials=True)
    S0, CS0, KV0 = fresh_state(R)
    aspot = []
    for pos in range(64):
        t1, _, _ = anc.forward_token(int(doc[pos]), pos, S0, CS0, KV0)
        aspot.append(int(t1))
    spot = []
    for pos in range(64):
        rig.feed(int(doc[pos]), pos)
        gr1s.step()
        spot.append(int(np.argmax(rig.dn(rig.logitsb, (248320,)))))
    spot_ok = sum(1 for a, b in zip(spot, aspot) if a == b)
    # -- PF bulk feed 64..N --
    tf = time.perf_counter()
    chunks, pf_ms = pf.feed(doc[64:N], 64)
    feed_wall = time.perf_counter() - tf
    rig.K["rmsz2048g"](rig.hA, rig.ONORM, rig.normhb, global_size=(1,1,1), local_size=(256,1,1), wait=True)
    rig.K["h6k2048"](rig.HEAD, rig.normhb, rig.logitsb, global_size=(7760,1,1), local_size=(1024,1,1), vals=(248320,), wait=True)
    t_next = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
    # -- cont32 + det x2 --
    snap = pf_state_snap(rig, R, N)
    def cont32():
        gen = [t_next]; t = t_next
        for gix in range(32):
            rig.feed(t, N + gix); gr1s.step()
            t = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
            gen.append(t)
        return gen
    gen1 = cont32()
    rig.dev.allocator._copyin(rig.SALL, memoryview(np.ascontiguousarray(snap["S"]).data.cast("B")))
    rig.dev.allocator._copyin(rig.CSALL, memoryview(np.ascontiguousarray(snap["CS"]).data.cast("B")))
    for ai in range(10):
        for nm, buf in (("KQ", rig.KVQ[ai]), ("VQ", rig.VVQ[ai]), ("KS", rig.KVS[ai]), ("VS", rig.VVS[ai])):
            rig.dev.allocator._copyin(buf, memoryview(np.ascontiguousarray(snap[f"{nm}{ai}"]).data.cast("B")))
    gen2 = cont32()
    detok = (gen1 == gen2)
    # -- decode perf at depth (T1-split cycles) --
    td = time.perf_counter()
    for gix in range(16):
        rig.feed(gen1[-1], N + 32 + gix); gr1s.step()
        int(np.argmax(rig.dn(rig.logitsb, (248320,))))
    dec_cycle_ms = (time.perf_counter() - td) / 16 * 1e3
    # -- rebase16 vs the split anchor --
    S2 = snap["S"].copy(); CS2 = snap["CS"].copy()
    KV2 = []
    for ai in range(10):
        lay = []
        for j in range(2):
            kqf = np.zeros((R, 256), np.int8); kqf[:N] = snap[f"KQ{ai}"][j]
            ksf = np.ones((R, 2), np.float32); ksf[:N] = snap[f"KS{ai}"][j]
            vqf = np.zeros((R, 256), np.int8); vqf[:N] = snap[f"VQ{ai}"][j]
            vsf = np.ones((R, 2), np.float32); vsf[:N] = snap[f"VS{ai}"][j]
            lay.append((kqf, ksf, vqf, vsf))
        KV2.append(lay)
    agen = []; t = t_next
    for gix in range(16):
        t, _, _ = anc.forward_token(t, N + gix, S2, CS2, KV2)
        agen.append(int(t))
    reb = sum(1 for a, b in zip(agen, gen1[1:17]) if a == b)
    snorm = float(np.linalg.norm(snap["S"])); smax = float(np.abs(snap["S"]).max())
    install_split_anchor(0)
    record(tag, f"rung {R}: feed N={N} ({chunks} chunks + tail) wall {feed_wall:.0f}s [{N/feed_wall:.0f} tok/s PF] | first64 {spot_ok}/64 | cont32 det x2 {'OK' if detok else 'FAIL'} | rebase16 {reb}/16 | GDN |S|={snorm:.3e} max={smax:.3e} | T1-split decode {dec_cycle_ms:.1f}ms/cyc (~{1e3/dec_cycle_ms:.1f} tok/s) | cont[:16] {gen1[:16]}")

# ============================================================================
def main():
    from MM_P7_lib import Rig7
    rig = Rig7(ctx_alloc=98304, load_p6=True)
    doc = np.load(DOC).astype(np.int32)
    stages = os.environ.get("MM_P7_STAGES", "H,S,F,L,D").split(",")
    if "H" in stages: stage_h(rig, doc)
    if "S" in stages: stage_s(rig)
    if "F" in stages: stage_f(rig, doc)
    if "L" in stages:
        stage_l(rig, doc, 65536, 64440, "L64")
        stage_l(rig, doc, 98304, 97700, "L96")
    if "D" in stages:
        record("D", f"AUDIT: ka-pool {rig.pool.slabs} slabs / {rig.pool.carves} carves / {rig.pool.off} B | fences {rig.fence_count}")
    print("[P7 ALL DONE]", flush=True)

if __name__ == "__main__":
    main()
