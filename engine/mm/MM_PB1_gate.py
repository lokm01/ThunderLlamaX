#!/usr/bin/env python3
"""MM PB1 GATE — the pairwise MoE fusion (rtsh8 + gxdn8) on the T=1 graph.

  B1  BIT-EXACT A/B: the SAME doc slice fed 32 T1 steps through the UNFUSED
      graph vs the FUSED graph (MM_FUSE2 flips between builds): the greedy
      stream AND the full state arrays (S, CS, all 40 KV tensors) must match
      BIT-FOR-BIT (the F1b discipline).
  B2  PERF: warm gr1.step() timings, unfused vs fused; the kernel-count
      delta; the projected T=1 tok/s.

One GPU process. Usage: ~/tg311/bin/python MM_PB1_gate.py
"""
import os, sys, time
import numpy as np

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE); sys.path.insert(0, BASE + "/engine0")
os.environ.setdefault("DEV", "NV")

CTXK = 98304; DEC_S = 32
N = 32
PROG = os.path.expanduser("~/mm_pb1_progress.txt")

def record(tag, result):
    with open(PROG, "a") as f: f.write(f"{tag} {result}\n"); f.flush(); os.fsync(f.fileno())
    print(f"[PROGRESS] {tag} {result}", flush=True)

print("[pb1gate] booting rig...", flush=True)
from MM_P7_lib import Rig7, build_seq7, mkgraph, load_pb1_fused
rig = Rig7(ctx_alloc=CTXK, load_p6=True)
load_pb1_fused(rig)
spk = f"s{CTXK}"

def build_t1():
    seq1 = build_seq7(rig, 1, "gconv36_1", "k2s36_1", with_head=True,
                      head_mode="am", spk=spk, S=DEC_S)
    return seq1, mkgraph(rig, seq1, "pb1_t1")

os.environ["MM_FUSE2"] = "0"
seq_u, gr_u = build_t1()
os.environ["MM_FUSE2"] = "1"
seq_f, gr_f = build_t1()
os.environ["MM_FUSE2"] = "0"
print(f"[pb1gate] graphs built: unfused {len(seq_u)} kernels vs fused {len(seq_f)} kernels", flush=True)

doc = np.load("~/mm_p5_doc100k_ids.npy").astype(np.int32)
toks = [int(x) for x in doc[4000:4000+256+N]]

def dump_state():
    out = {"S": rig.dn(rig.SALL, (30, 32*128*128)), "CS": rig.dn(rig.CSALL, (30, 8192*3))}
    kv = {}
    for ai in range(10):
        kv[f"kq{ai}"] = rig.dn(rig.KVQ[ai], (2*CTXK*256,), np.int8)
        kv[f"ks{ai}"] = rig.dn(rig.KVS[ai], (2*CTXK*2,))
        kv[f"vq{ai}"] = rig.dn(rig.VVQ[ai], (2*CTXK*256,), np.int8)
        kv[f"vs{ai}"] = rig.dn(rig.VVS[ai], (2*CTXK*2,))
    out.update(kv)
    return out

def run_arm(gr):
    rig.reset_states(2048)
    stream = []
    for i, t in enumerate(toks):
        rig.feed(t, i)
        gr.step()
        stream.append(int(rig.am_view[0]))
    return stream, dump_state()

t0 = time.perf_counter()
su, Su = run_arm(gr_u)
t_u = time.perf_counter() - t0
print(f"[pb1gate] unfused arm done ({t_u:.1f}s)", flush=True)
t0 = time.perf_counter()
sf, Sf = run_arm(gr_f)
t_f = time.perf_counter() - t0
print(f"[pb1gate] fused arm done ({t_f:.1f}s)", flush=True)

stream_ok = su == sf
state_bad = []
for k in Su:
    if not np.array_equal(Su[k], Sf[k]):
        d = np.argwhere(Su[k] != Sf[k])
        state_bad.append((k, len(d), d[0].tolist()))
record("B1", f"FUSED T1 BIT-EXACT: stream {'IDENTICAL' if stream_ok else 'DIVERGED at ' + str(next((i for i,(a,b) in enumerate(zip(su,sf)) if a!=b), -1))} | "
             f"state {'ALL BIT-IDENTICAL' if not state_bad else str(state_bad[:4])} | "
             f"first8={su[:8]}")

# ---- B2: warm step timing ----
def time_steps(gr, n=60):
    cur = su[-1]; pos = len(toks)
    for i in range(8):
        rig.feed(cur, pos + i); gr.step(); cur = int(rig.am_view[0])
    t0 = time.perf_counter()
    for i in range(n):
        rig.feed(cur, pos + 100 + i); gr.step(); cur = int(rig.am_view[0])
    return (time.perf_counter() - t0) / n * 1e3

ms_u = time_steps(gr_u)
ms_f = time_steps(gr_f)
record("B2", f"T1 CYCLE: unfused {ms_u:.2f} ms vs fused {ms_f:.2f} ms ({(ms_u/ms_f - 1)*100:+.1f}%) | "
             f"kernels {len(seq_u)} -> {len(seq_f)} (-{len(seq_u)-len(seq_f)}) | "
             f"tok/s {1000/ms_u:.1f} -> {1000/ms_f:.1f}")
print("[pb1gate] DONE", flush=True)
