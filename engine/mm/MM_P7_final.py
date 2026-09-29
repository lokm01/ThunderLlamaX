#!/usr/bin/env python3
"""MM P7 FINAL — the corrected F-gates (same-S arms) + a second H3 sample.
F1b: PF-chunk-256 vs per-token T1 at the SAME split S=8 -> BIT-EXACT states
     + identical 16-tok continuations (the prefill-exactness contract).
F2b: doc2048 PF + 8-tok continuation vs the S=8 split-anchor rebase.
H3b: the quote/prose battery perf sample #2."""
import os, sys, time, json
import numpy as np
BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE); sys.path.insert(0, BASE + "/engine0")
from MM_P7_lib import Rig7, build_seq7, mkgraph, SpecEngine7, install_split_anchor

PROG = os.path.expanduser("~/mm_p7_progress.txt")
def record(tag, result):
    with open(PROG, "a") as f: f.write(f"{tag} {result}\n"); f.flush(); os.fsync(f.fileno())
    print(f"[PROGRESS] {tag} {result}", flush=True)

doc = np.load(os.path.expanduser("~/mm_p5_doc100k_ids.npy")).astype(np.int32)
rig = Rig7(ctx_alloc=98304, load_p6=True)
R = 16384

# ---------- F1b: same-S chunk consistency ----------
seq1s8 = build_seq7(rig, 1, "gconv36_1", "k2s36_1", with_head=True, head_mode="full", spk="s16384", S=8)
gr1s8 = mkgraph(rig, seq1s8, "f_t1s8")
seqpf = build_seq7(rig, 256, "gconv36_256", "k2s36_256", with_head=False, spk="s16384", S=8, pf=True)
grpf = mkgraph(rig, seqpf, "f_pf")
N = 768
rig.reset_states(R)
for p in range(N):
    rig.feed(int(doc[p]), p); gr1s8.step()
topA = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
snapA = {"S": rig.dn(rig.SALL, (30,32,128,128)), "CS": rig.dn(rig.CSALL, (30,8192,3))}
for ai in range(10):
    for nm, buf, shp, dt in (("KQ", rig.KVQ[ai], (2*N*256,), np.int8), ("VQ", rig.VVQ[ai], (2*N*256,), np.int8),
                             ("KS", rig.KVS[ai], (2*N*2,), np.float32), ("VS", rig.VVS[ai], (2*N*2,), np.float32)):
        snapA[f"{nm}{ai}"] = rig.dn(buf, shp, dt)
rig.reset_states(R)
for c in range(N // 256):
    rig.pf_ids_view[:] = memoryview(np.ascontiguousarray(doc[c*256:(c+1)*256], dtype=np.int32).data)
    rig.pos_view[0] = c * 256
    grpf.step()
topB = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
snapB = {"S": rig.dn(rig.SALL, (30,32,128,128)), "CS": rig.dn(rig.CSALL, (30,8192,3))}
for ai in range(10):
    for nm, buf, shp, dt in (("KQ", rig.KVQ[ai], (2*N*256,), np.int8), ("VQ", rig.VVQ[ai], (2*N*256,), np.int8),
                             ("KS", rig.KVS[ai], (2*N*2,), np.float32), ("VS", rig.VVS[ai], (2*N*2,), np.float32)):
        snapB[f"{nm}{ai}"] = rig.dn(buf, shp, dt)
alleq = all(np.array_equal(snapA[k], snapB[k]) for k in snapA)
# continuations (state = arm B live now)
genB = []; t = topB
for gix in range(16):
    genB.append(t); rig.feed(t, N + gix); gr1s8.step()
    t = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
rig.reset_states(R)
for p in range(N):
    rig.feed(int(doc[p]), p); gr1s8.step()
genA = []; t = topA
for gix in range(16):
    genA.append(t); rig.feed(t, N + gix); gr1s8.step()
    t = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
record("F1b", f"SAME-S=8 CHUNK CONSISTENCY doc768: state arrays {'ALL BIT-EXACT' if alleq else 'DIFF'} | top1 {topA}=={topB} {'OK' if topA==topB else 'DIFF'} | cont16 {'IDENTICAL' if genA==genB else 'DIFF'}")

# ---------- F2b: doc2048 rebase vs the S=8 split anchor ----------
N2 = 2048
rig.reset_states(R)
t0 = time.perf_counter()
for c in range(N2 // 256):
    rig.pf_ids_view[:] = memoryview(np.ascontiguousarray(doc[c*256:(c+1)*256], dtype=np.int32).data)
    rig.pos_view[0] = c * 256
    grpf.step()
pf_ms = (time.perf_counter() - t0) * 1e3
tpf = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
gen = []; t = tpf
for gix in range(8):
    gen.append(t); rig.feed(t, N2 + gix); gr1s8.step()
    t = int(np.argmax(rig.dn(rig.logitsb, (248320,))))
from MM_P34_ports import Anchor, fresh_state
snap = {"S": rig.dn(rig.SALL, (30,32,128,128)), "CS": rig.dn(rig.CSALL, (30,8192,3))}
for ai in range(10):
    for nm, buf, shp, dt in (("KQ", rig.KVQ[ai], (2*N2*256,), np.int8), ("VQ", rig.VVQ[ai], (2*N2*256,), np.int8),
                             ("KS", rig.KVS[ai], (2*N2*2,), np.float32), ("VS", rig.VVS[ai], (2*N2*2,), np.float32)):
        snap[f"{nm}{ai}"] = rig.dn(buf, shp, dt)
install_split_anchor(8)
anc = Anchor(ctx=R, fp16_partials=True)
S2_ = snap["S"].copy(); CS2_ = snap["CS"].copy()
KV2 = []
for ai in range(10):
    lay = []
    for j in range(2):
        kqf = np.zeros((R, 256), dtype=np.int8); kqf[:N2] = snap[f"KQ{ai}"].reshape(2, N2, 256)[j]
        ksf = np.ones((R, 2), dtype=np.float32); ksf[:N2] = snap[f"KS{ai}"].reshape(2, N2, 2)[j]
        vqf = np.zeros((R, 256), dtype=np.int8); vqf[:N2] = snap[f"VQ{ai}"].reshape(2, N2, 256)[j]
        vsf = np.ones((R, 2), dtype=np.float32); vsf[:N2] = snap[f"VS{ai}"].reshape(2, N2, 2)[j]
        lay.append((kqf, ksf, vqf, vsf))
    KV2.append(lay)
agen = []; t = tpf
for gix in range(8):
    t, _, _ = anc.forward_token(t, N2 + gix, S2_, CS2_, KV2)
    agen.append(int(t))
reb = sum(1 for a, b in zip(agen, gen[1:9]) if a == b)
install_split_anchor(0)
record("F2b", f"PREFILL doc2048 SAME-S rebase8-vs-anchor {reb}/8 | top1 {tpf} | PF {pf_ms:.0f}ms ({N2/(pf_ms/1e3):.1f} tok/s)")

# ---------- H3b: second perf sample ----------
seq1 = build_seq7(rig, 1, "gconv36_1", "k2s36_1", with_head=True, head_mode="full")
gr1 = mkgraph(rig, seq1, "h3b_t1")
seq2 = build_seq7(rig, 3, "gconv36s_3", "k2s36s_3", with_head=True, head_mode="am", slots=True, tail_acc_K=2)
gr2 = mkgraph(rig, seq2, "h3b_d2")
seq8 = build_seq7(rig, 9, "gconv36s_9", "k2s36s_9", with_head=True, head_mode="am", slots=True, tail_acc_K=8)
gr8 = mkgraph(rig, seq8, "h3b_d8")
eng = SpecEngine7(rig, gr1, gr2, gr8)
from MM_P34_tok import parse_gguf_kv, SimpleTokenizer
from MM_P34_anchor import PROMPTS as P20
tok = SimpleTokenizer.from_gguf_kv(parse_gguf_kv(os.path.expanduser("~/models36/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")))
doc_txt = open(os.path.expanduser("~/prompt100k.txt"), encoding="utf-8", errors="replace").read()
code = "def process(items):\n    out = []\n    for it in items:\n        if it is not None:\n            out.append(it.strip())\n    return out\n"
qp_code = f"{code}\nThe same function again:\ndef process(items):"
rows = []
for tag, txt, ntok in (("quote-alpha", P20[7], 48), ("quote-code", qp_code, 48), ("prose-0", P20[0], 32)):
    ids = tok.encode(txt)
    t1b, lkb = eng.feed_prompt(ids)
    g = eng.generate(t1b, lkb, len(ids), ntok, mode="spec")
    st = eng.stats
    rows.append((tag, round(st["tok"]/(st["ms"]/1e3), 2), round(st["wait_ms"]/max(1, st["cyc_d8"]), 3), st["cyc_d8"], round(st["ms_d8"]/max(1, st["cyc_d8"]), 3)))
    print(f"    [H3b] {tag}: {rows[-1]}", flush=True)
record("H3b", "PERF SAMPLE 2: " + " | ".join(f"{r[0]}: {r[1]} tok/s (d8cyc {r[3]} @ graph {r[4]*1e3:.1f}ms, wait {r[2]*1e3:.2f}ms)" for r in rows))
print("[FINAL DONE]", flush=True)
