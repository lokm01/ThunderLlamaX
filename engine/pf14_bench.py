# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P14: the in-plan persistent-FFN measure (P13 kernel B, benched inside the
PROVEN pf host process). Post-boot law: ZERO fresh device allocations after
the warm boot (the P13/P14 allocation-history fault class) — only win_up into
EXISTING plan buffers (hhx32/gact32) + PfGraph kernargs slabs (the pf12_attr
pattern). Boots the full trunk with PF_PERSIST=1 (persistent ffn launches
already live in the plan+graphs — the boot prefill itself is a smoke test),
then:
  1) BIT-IDENTITY: pf13ffn_w{2,4,8}_n1 (grid 82) vs shipped
     pfg3_ffn_r7_m32_nw8k128 (grid 272) on REAL W7 weights, poison via
     re-launch on the same gact32 buffer + host copies; determinism x2.
  2) BENCH: synced min-of-10 over 8 covered blocks (per-block launch pattern
     = the chunk plan). Metrics: us/blk, w GB/s, amort (2x), DRAM (x512/392).
  3) GRAPH: captured PfGraph of 8 persist launches vs 8 r7 launches.
GATE: amort >= 350 GB/s AND >= 1.15x the same-session r7 control -> GO.
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from mtp import MTPEngine, CBLK
import pf_prefill
from pf_prefill import PfGraph

WB = 2 * 17408 * 1960          # original weight bytes per block (fg+fu)
R7F = 512.0 / 392.0
LS = (256, 1, 1)

E = MTPEngine(theta=1e7)
P, d, pr = E.P, E.P.d, E.pr
rng = np.random.default_rng(0)
ids = [int(t) for t in rng.integers(1000, 60000, 2048)]

E.reset_fresh(ids[0])
E.stload_trunk()
for i in E.gdn_idx:
  E._mfill(f"conv{i}_1", 0, CBLK)
dev.synchronize()
from mtp import SLICE
_seen, sl = set(), []
for t in ids:
  if t not in _seen:
    _seen.add(t); sl.append(t)
_base = sl[:]
while len(sl) < SLICE:
  sl += _base
E.init_draft(sl[:SLICE])
E.fill_draft(ids, start_pos=0, seed_hd=None)
dev.synchronize()

CT = []
t0 = time.perf_counter()
dt = pf_prefill.prefill_batch(E, None, ids, chunk_times=CT)
dev.synchronize()
cts = sorted(ms for _, ms in CT)
print(f"[boot] prefill 2048: {dt:.2f}s chunks={len(cts)} med={cts[len(cts)//2]:.1f}ms", flush=True)

W7 = E._pf_W7
blocks = [i for i in range(64) if ("fg", i) in W7]
KB = blocks[:8]
assert len(KB) == 8, f"need 8 covered blocks, got {len(blocks)}"
print(f"[boot] W7 fg coverage {len(blocks)}/64; bench blocks {KB}", flush=True)

R7 = pr["pfg3_ffn_r7_m32_nw8k128"]
PERS = [pr[n] for n in ["pf13ffn_w2_n1", "pf13ffn_w4_n1", "pf13ffn_w8_n1"]]

# deterministic x (32 rows) into the PLAN buffer (win_up = no new allocation)
P.win_up("hhx32", 0, (np.random.default_rng(7).standard_normal((32, 5120)) * 0.8).astype(np.float16).reshape(-1))
dev.synchronize()

def run_r7(i):
  R7(W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx32"], d["gact32"], global_size=(272, 1, 1), local_size=LS)
def run_pers(pp, i):
  pp(W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx32"], d["gact32"], global_size=(82, 1, 1), local_size=LS)
def grab():
  return P.down("gact32", (32, 17408), np.float16).copy()

# ---- 1) bit-identity + determinism x2 (first clean run = the gate) ----
ALL_OK = True
for pp in PERS:
  run_pers(pp, KB[0]); dev.synchronize()
  m1 = grab()
  run_pers(pp, KB[0]); dev.synchronize()
  m2 = grab()
  det = int((m1 != m2).sum())
  run_r7(KB[0]); dev.synchronize()
  ref = grab()
  nz = int((m1 != ref).sum())
  ok = (nz == 0) and (det == 0)
  ALL_OK &= ok
  print(f"[gate] {pp.name}: vs r7 nz={nz}/{m1.size} det-x2 nz={det} -> {'BIT-IDENTICAL' if ok else 'DIFF'}", flush=True)
pp = PERS[2]
bad = 0
for i in KB:
  run_pers(pp, i); dev.synchronize()
  m1 = grab()
  run_r7(i); dev.synchronize()
  r = grab()
  bad += int((m1 != r).sum())
print(f"[gate] w8 all {len(KB)} bench blocks: nz={bad} -> {'BIT-IDENTICAL' if bad == 0 else 'DIFF'}", flush=True)

# ---- 2) bench: synced min-of-10 over KB blocks ----
def bench(fn, n=10):
  fn(KB[0]); dev.synchronize()
  best = 1e9
  for _ in range(n):
    t0 = time.perf_counter()
    for i in KB: fn(i)
    dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  return best / len(KB)

t_r7 = bench(lambda i: run_r7(i))
print(f"[bench] R7-CONTROL (grid 272): {t_r7*1e6:7.1f} us/blk | w {WB/t_r7/1e9:6.1f} | "
      f"amort {2*WB/t_r7/1e9:6.1f} | dram {R7F*WB/t_r7/1e9:6.1f} GB/s", flush=True)
best_amort, best_name, best_t = 0.0, None, None
for pp in PERS:
  t = bench(lambda i, pp=pp: run_pers(pp, i))
  am = 2 * WB / t / 1e9
  print(f"[bench] PERSIST {pp.name} (grid 82): {t*1e6:7.1f} us/blk | x{t_r7/t:4.2f} vs r7 | "
        f"w {WB/t/1e9:6.1f} | amort {am:6.1f} | dram {R7F*WB/t/1e9:6.1f} GB/s"
        f" | 350-gate {'PASS' if am >= 350 else 'FAIL'} | 1.15x-gate {'PASS' if t_r7/t >= 1.15 else 'FAIL'}", flush=True)
  if am > best_amort:
    best_amort, best_name, best_t = am, pp.name, t

# ---- 3) the graph question: captured QMD chain of persistent launches ----
seq_r7 = [(R7, (W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx32"], d["gact32"]), 272, LS) for i in KB]
bpp = next(p for p in PERS if p.name == best_name)
seq_p = [(bpp, (W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx32"], d["gact32"]), 82, LS) for i in KB]
def time_graph(g, n=10):
  best = 1e9
  for _ in range(n):
    prev = dev.timeline_value - 1
    t1 = time.perf_counter()
    v = dev.next_timeline(); g.submit(prev, v)
    dev.timeline_signal.wait(v)
    best = min(best, time.perf_counter() - t1)
  return best * 1e3 / len(KB)
g_r7 = PfGraph(seq_r7, "p14r7")
g_p = PfGraph(seq_p, "p14pw")
print(f"[graph] r7   captured {time_graph(g_r7):7.1f} us/blk", flush=True)
print(f"[graph] pers captured {time_graph(g_p):7.1f} us/blk", flush=True)
# captured bit-identity re-check (the QMD path must write the same bytes)
run_r7(KB[0]); dev.synchronize()
ref = grab()
v = dev.next_timeline(); g_p.submit(dev.timeline_value - 1, v); dev.timeline_signal.wait(v)
mine = grab()
print(f"[graph] captured {best_name} vs r7 eager: nz={int((mine != ref).sum())} (block {KB[0]})", flush=True)

verdict = "GO" if (best_amort >= 350 and t_r7 / best_t >= 1.15) else ("PARTIAL-265-350" if best_amort >= 265 else "KILL-lt265")
print(f"[done] gates={'ALL-OK' if ALL_OK else 'FAILED'} best={best_name} amort={best_amort:.1f} -> verdict {verdict}", flush=True)
