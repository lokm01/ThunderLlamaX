#!/usr/bin/env python3
"""MM P5 RUN — stages:
  A  battery20 graph replay -> engine top1s + THE P5.1 FP16-PARTIAL DECISION
     (engine vs old f32 anchor [repro 347/349] vs new engine-order fp16 anchor).
  B  the 60/60 Tier-1 bank (prompt pass + 32-tok greedy cont, det x2 full).
  C  the ctx ladder: rungs 4096/16384/65536 on the doc100k corpus -- D8-slot
     bulk feed (P=9) + T1 first-64 spot + T1 continuation det x2 + the
     anchor-rebase drift measurement + GDN-state health stats.
  D  audit: ka-slab pool, fences, perf.
Progress ~/mm_p5_progress.txt (resumable). Anchor files gate B (and A16 gates A).
"""
import os, sys, time, json, hashlib
import numpy as np

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE); sys.path.insert(0, BASE + "/engine0")
PROG = os.path.expanduser("~/mm_p5_progress.txt")
A16 = os.path.expanduser("~/mm_p5_anchor16.npz")
A60 = os.path.expanduser("~/mm_p5_anchor60.npz")
AOLD = os.path.expanduser("~/mm_p34_anchor.npz")
DOC = os.path.expanduser("~/mm_p5_doc100k_ids.npy")

def record(tag, result):
    with open(PROG, "a") as f: f.write(f"{tag} {result}\n"); f.flush(); os.fsync(f.fileno())
    print(f"[PROGRESS] {tag} {result}", flush=True)
def done(tag):
    if not os.path.exists(PROG): return False
    return any(l.split()[0] == tag for l in open(PROG).read().splitlines() if l.split())

def score(engine_t1, battery_t1, battery_gap, tag, secs, nprompts=20):
    pm = sum(int((g == e).sum()) for g, e in zip(engine_t1, battery_t1))
    pt = sum(len(e) for e in battery_t1)
    fm = sum(int((g == e).all()) for g, e in zip(engine_t1, battery_t1))
    mism = []; mgaps = []
    for pi, (g, e) in enumerate(zip(engine_t1, battery_t1)):
        bad = np.where(g != e)[0]
        if len(bad):
            mism.append((pi, bad.tolist()[:6]))
            for b in bad: mgaps.append(float(battery_gap[pi][b]))
    gs = ""
    if mgaps:
        mg = np.array(mgaps)
        gs = f" | gaps n={len(mg)} min={mg.min():.2e} med={np.median(mg):.2e} max={mg.max():.2e}; <1e-2:{int((mg<1e-2).sum())} >=1e-2:{int((mg>=1e-2).sum())}"
    return f"{tag}: {pm}/{pt} EXACT, {fm}/{nprompts} prompts full, {pt/max(1,secs):.2f} tok/s mism={mism[:6]}{gs}", (pm, pt, fm)

def load_anchor(path):
    a = np.load(path, allow_pickle=True)
    return ([np.asarray(x, dtype=np.int32) for x in a["ids"]],
            [np.asarray(x, dtype=np.int32) for x in a["top1"]],
            [np.asarray(x, dtype=np.float32) for x in a["gap"]])

def main():
    from MM_P56_lib import Rig, GraphRunner, feed_token, t1_argmax, LSZ
    from MM_P34_ports import Anchor, fresh_state

    need = set()
    rung_files = [f"{BASE}/MM_P6_spkq256m_{v}.cubin" for v in ("4096", "16384", "65536")] + \
                 [f"{BASE}/MM_P34_spka256_{v}.cubin" for v in ("4096", "16384", "65536")] + \
                 [f"{BASE}/MM_P6_k2s36s_9.cubin", f"{BASE}/MM_P6_gconv36s_9.cubin"]
    have = all(os.path.exists(p) for p in rung_files)
    rig = Rig(build_only=None)
    ids20, t1_old, gap_old = load_anchor(AOLD)

    # ================= STAGE A: battery20 + P5.1 =================
    if not done("A") and os.path.exists(A16):
        ids16, t1_16, gap16 = load_anchor(A16)
        seq1 = rig.build_seq(1, "gconv36_1", "k2s36_1", with_head=True, head_mode="full")
        gr = GraphRunner(rig, [(rig.K[n], b, g, v) for n, b, g, v in seq1], "t1")
        def battery(ids):
            out = []
            for pi, idl in enumerate(ids):
                rig.reset_states(1024)
                pt1 = []
                for pos, tid in enumerate(idl):
                    feed_token(rig, int(tid), pos)
                    gr.step()
                    pt1.append(t1_argmax(rig))
                out.append(np.array(pt1, dtype=np.int32))
                print(f"    [A] prompt {pi} ({len(idl)} pos)", flush=True)
            return out
        t0 = time.time()
        eng = battery(ids20)
        secs = time.time() - t0
        s_old, r_old = score(eng, t1_old, gap_old, "engine vs OLD f32 anchor", secs)
        s_16, r16 = score(eng, t1_16, gap16, "engine vs fp16-partial anchor", secs)
        json.dump({"engine": [x.tolist() for x in eng]}, open(os.path.expanduser("~/mm_p5_engine20.json"), "w"))
        verdict = "ADOPT-A" if r16[2] == 20 else ("KEEP-B" if r16[0] > r_old[0] else "NO-GAIN")
        record("A", f"{s_old} || {s_16} || P5.1 VERDICT: {verdict} (engine20 saved)")
        json.dump({"decision": verdict, "exact16": r16[0], "tot": r16[1], "full16": r16[2]},
                  open(os.path.expanduser("~/mm_p5_partial_decision.json"), "w"))

    # ================= STAGE B: bank60 =================
    if not done("B") and os.path.exists(A60):
        ids60, t1_60, gap60 = load_anchor(A60)
        seq1 = rig.build_seq(1, "gconv36_1", "k2s36_1", with_head=True, head_mode="full")
        gr = GraphRunner(rig, [(rig.K[n], b, g, v) for n, b, g, v in seq1], "t1b")
        def run_cont(idl, ntok=32):
            rig.reset_states(1024)
            pt1 = []
            for pos, tid in enumerate(idl):
                feed_token(rig, int(tid), pos)
                gr.step()
                pt1.append(t1_argmax(rig))
            gen = []
            t = pt1[-1]
            for gix in range(ntok):
                gen.append(t)
                feed_token(rig, t, len(idl) + gix)
                gr.step()
                t = t1_argmax(rig)
            return pt1, gen
        t0 = time.time()
        bank_pt1, bank_gen = [], []
        for pi in range(len(ids60)):
            p1, g = run_cont(ids60[pi])
            bank_pt1.append(p1); bank_gen.append(g)
        det_gen = []
        for pi in range(len(ids60)):
            _, g = run_cont(ids60[pi])
            det_gen.append(g)
        detok = all(a == b for a, b in zip(det_gen, bank_gen))
        detok_pt = all(np.array_equal(np.array(a), np.array(b)) for a, b in zip(det_gen, bank_gen))
        s60, r60 = score([np.array(x) for x in bank_pt1], t1_60, gap60, "bank60 prompt-pass vs anchor60", time.time()-t0, nprompts=60)
        # spot anchor continuations: 12 prompts x 8 tokens
        axr = []
        try:
            anc = Anchor(ctx=1024, fp16_partials=True)
            for pi in range(0, 60, 5):
                idl = [int(x) for x in ids60[pi]]
                S, convst, KV = fresh_state(1024)
                for pos, tid in enumerate(idl):
                    t1, _, _ = anc.forward_token(tid, pos, S, convst, KV)
                agen = []; t = t1
                for gix in range(8):
                    agen.append(t)
                    t, _, _ = anc.forward_token(t, len(idl) + gix, S, convst, KV)
                axr.append((pi, agen, bank_gen[pi][:8]))
        except Exception as e:
            axr.append(("ERR", repr(e), []))
        ax_ok = all(a == b for _, a, b in axr if isinstance(_, int))
        json.dump({"model": "Qwen3.6-35B-A3B-UD-IQ4_XS", "engine": "MM_P5 graph train",
                   "ntok": 32, "det_full_rerun": bool(detok),
                   "gen": bank_gen, "anchor_x8_12prompts": [[a, b] for _, a, b in axr]},
                  open(os.path.expanduser("~/mm_p5_bank60.json"), "w"))
        record("B", f"BANK60: det(full x2)={'OK' if detok else 'FAIL'} | {s60} | anchor-x8(12)={'EXACT' if ax_ok else 'DIFF'} {[x[:2] for x in axr if x[0]=='ERR'] or ''}")

    # ================= STAGE C: the ctx ladder =================
    if have and not done("C64k"):
        doc = np.load(DOC).astype(np.int32)
        from MM_P34_tok import parse_gguf_kv, SimpleTokenizer
        tok = SimpleTokenizer.from_gguf_kv(parse_gguf_kv(os.path.expanduser("~/models36/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")))
        rungs_env = os.environ.get("MM_LADDER_RUNGS", "4096,16384,65536")
        rungs_map = {4096: (4096, 4000), 16384: (16384, 16288), 65536: (65536, 64440)}
        for R, N in [rungs_map[int(x)] for x in rungs_env.split(",")]:
            tag = {4096: "C4k", 16384: "C16k", 65536: "C64k"}[R]
            if done(tag): continue
            t0 = time.time()
            seqt1 = rig.build_seq(1, "gconv36_1", "k2s36_1", with_head=True, head_mode="full", spk=str(R))
            gr1 = GraphRunner(rig, [(rig.K[n], b, g, v) for n, b, g, v in seqt1], tag + "t1")
            seqd8 = rig.build_seq(9, "gconv36s_9", "k2s36s_9", with_head=False, spk=str(R), slots=True)
            gr8 = GraphRunner(rig, [(rig.K[n], b, g, v) for n, b, g, v in seqd8], tag + "d8")
            rig.reset_states(R)
            # -- first-64 T1 spot --
            spot = []
            for pos in range(64):
                feed_token(rig, int(doc[pos]), pos)
                gr1.step()
                spot.append(t1_argmax(rig))
            # -- bulk D8 feed 64..N --
            p = 64
            feed_cycles = 0
            while p + 9 <= N:
                for s in range(9):
                    rig.dev.allocator._copyin(rig.idsb, memoryview(np.array(doc[p:p+9], dtype=np.int32).tobytes()))
                rig.dev.allocator._copyin(rig.POSB, memoryview(np.array([p], dtype=np.int32).tobytes()))
                gr8.step(); feed_cycles += 1
                p += 9
            while p < N:
                feed_token(rig, int(doc[p]), p)
                gr1.step(); p += 1; feed_cycles += 1
            # -- head on the final seat -> t_next --
            rig.K["rmsz2048g"](rig.hA.offset(offset=8*2048*4, size=2048*4), rig.ONORM, rig.normhb,
                                global_size=(1,1,1), local_size=LSZ["rmsz2048g"], wait=True)
            rig.K["h6k2048"](rig.HEAD, rig.normhb, rig.logitsb, global_size=(7760,1,1), local_size=LSZ["h6k2048"], vals=(248320,), wait=True)
            t_next = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
            # -- snapshot --
            snap = {"S": rig.dn(rig.SALL, (30,32,128,128)), "CS": rig.dn(rig.CSALL, (30,8192,3))}
            for ai in range(10):
                snap[f"KQ{ai}"] = rig.dn(rig.KVQ[ai], (2,R,256), np.int8)[:, :N]
                snap[f"KS{ai}"] = rig.dn(rig.KVS[ai], (2,R,2))[:, :N]
                snap[f"VQ{ai}"] = rig.dn(rig.VVQ[ai], (2,R,256), np.int8)[:, :N]
                snap[f"VS{ai}"] = rig.dn(rig.VVS[ai], (2,R,2))[:, :N]
            def restore():
                rig.dev.allocator._copyin(rig.SALL, memoryview(np.ascontiguousarray(snap["S"]).data.cast("B")))
                rig.dev.allocator._copyin(rig.CSALL, memoryview(np.ascontiguousarray(snap["CS"]).data.cast("B")))
                for ai in range(10):
                    for nm, buf in (("KQ", rig.KVQ[ai]), ("VQ", rig.VVQ[ai])):
                        rig.dev.allocator._copyin(buf, memoryview(np.ascontiguousarray(snap[f"{nm}{ai}"]).data.cast("B")))
                    for nm, buf in (("KS", rig.KVS[ai]), ("VS", rig.VVS[ai])):
                        rig.dev.allocator._copyin(buf, memoryview(np.ascontiguousarray(snap[f"{nm}{ai}"]).data.cast("B")))
            def cont32():
                gen = [t_next]
                t = t_next
                for gix in range(32):
                    feed_token(rig, t, N + gix)
                    gr1.step()
                    t = t1_argmax(rig)
                    gen.append(t)
                return gen
            gen1 = cont32()
            restore()
            gen2 = cont32()
            detok = gen1 == gen2
            # -- anchor first-64 spot (CPU) --
            anc = Anchor(ctx=R, fp16_partials=True)
            S, convst, KV = fresh_state(R)
            aspot = []
            for pos in range(64):
                t1, _, _ = anc.forward_token(int(doc[pos]), pos, S, convst, KV)
                aspot.append(int(t1))
            spot_ok = sum(1 for a, b in zip(spot, aspot) if a == b)
            # -- anchor rebase: 16 greedy tokens from the engine snapshot --
            S2 = snap["S"].copy(); CS2 = snap["CS"].copy()
            KV2 = []
            for ai in range(10):
                lay = []
                for j in range(2):
                    # full-R arrays (the rebase continuation WRITES KV[pos] for
                    # pos N..N+15 -- the sliced [:N] snapshot crashed here)
                    kqf = np.zeros((R, 256), dtype=np.int8); kqf[:N] = snap[f"KQ{ai}"][j]
                    ksf = np.ones((R, 2), dtype=np.float32); ksf[:N] = snap[f"KS{ai}"][j]
                    vqf = np.zeros((R, 256), dtype=np.int8); vqf[:N] = snap[f"VQ{ai}"][j]
                    vsf = np.ones((R, 2), dtype=np.float32); vsf[:N] = snap[f"VS{ai}"][j]
                    lay.append((kqf, ksf, vqf, vsf))
                KV2.append(lay)
            agen = []; t = t_next
            for gix in range(16):
                t, _, _ = anc.forward_token(t, N + gix, S2, CS2, KV2)
                agen.append(int(t))
            reb_ok = sum(1 for a, b in zip(agen, gen1[1:17]) if a == b)
            snorm = float(np.linalg.norm(snap["S"]))
            smax = float(np.abs(snap["S"]).max())
            ctxt = tok.decode(gen1[:24]).replace(chr(10), " ")[:120]
            t_wall = time.time() - t0
            record(tag, f"rung {R}: feed N={N} ({feed_cycles} D8 + tail) {t_wall:.0f}s | first64 engine-vs-anchor {spot_ok}/64 | cont32 det x2 {'OK' if detok else 'FAIL'} | rebase16 {reb_ok}/16 | GDN |S|={snorm:.3e} max={smax:.3e} | cont[:24]: {ctxt!r}")
            del snap

    # ================= STAGE E: the doc first-64 flip adjudication =================
    if not done("E"):
        doc = np.load(DOC).astype(np.int32)
        from MM_P34_ports import Anchor as _An, fresh_state as _fs
        seq1 = rig.build_seq(1, "gconv36_1", "k2s36_1", with_head=True, head_mode="full")
        gr = GraphRunner(rig, [(rig.K[n], b, g, v) for n, b, g, v in seq1], "et1")
        rig.reset_states(1024)
        eng64, logits64 = [], []
        for pos in range(64):
            feed_token(rig, int(doc[pos]), pos)
            gr.step()
            lg = rig.dn(rig.logitsb, (248320,))
            eng64.append(int(np.argmax(lg)))
            logits64.append(lg)
        anc = _An(ctx=1024, fp16_partials=True)
        S, convst, KV = _fs(1024)
        a64, agaps, alogits = [], [], []
        for pos in range(64):
            t1, lg, _ = anc.forward_token(int(doc[pos]), pos, S, convst, KV, want_logits=True)
            a64.append(int(t1))
            srt = np.sort(lg)[::-1]
            agaps.append(float(srt[0] - srt[1]))
            alogits.append(lg)
        bad = [i for i in range(64) if eng64[i] != a64[i]]
        detail = []
        for b in bad:
            e5 = np.argsort(-logits64[b])[:5].tolist()
            a5 = np.argsort(-alogits[b])[:5].tolist()
            detail.append(f"pos {b}: eng {eng64[b]} anc {a64[b]} gap {agaps[b]:.3e} top5 eng {e5} anc {a5}")
        record("E", f"DOC first-64 (std T1 train, ctx 1024): {64 - len(bad)}/64 EXACT vs fp16 anchor | {'; '.join(detail) if detail else 'ALL EXACT'}")

    # ================= STAGE D: audit =================
    if not done("D"):
        record("D", f"AUDIT: ka-pool {rig.pool.slabs} slabs / {rig.pool.carves} carves / {rig.pool.off} B | fences {rig.fence_count} | replays {rig.ga.n + rig.gb.n if hasattr(rig,'ga') else 'n/a'}")
    print("[ALL DONE]", flush=True)

if __name__ == "__main__":
    main()
