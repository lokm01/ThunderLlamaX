# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P16 DISCRIMINATOR A (in-plan, the pf14 law): the ws ping-pong vs the
shipped r7 iq3d. Cubins loaded at the proven-stable window (post-trunk-load,
pre-prefill); bench = win_up into EXISTING plan buffers only.
Arms (readout-order law — first clean run is the gate):
  G1 BIT-IDENTITY: pp1-nw8 (grid 80, 2-bar decode-off-path) and pp2-nw4
     (grid 160, ONE barrier/chunk) vs shipped pfg3_iq3d_r7_m32_nw8k128
     (grid 80) on REAL W7 fd weights; determinism x2.
  G2 BENCH (synced min-of-10, 8 fd-covered blocks): shipped m32 nw8,
     shipped nt32 nw4 control, shipped m64 nw8 (the gemm_fd class),
     pp1-nw8, pp2-nw4, pp1-m64-nw4. Per-64-token cost = m32-class x2.
Verdict: best-pp / shipped-m32 >= 1.35 GO | < 1.2 DEAD.
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
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

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

BASE = os.path.dirname(os.path.abspath(pf_prefill.__file__))
MINE = ["pfg3_iq3d_r7_m32_nt32_nw4k128", "pfg3_iq3d_r7pp1_m32_nw8k128", "pfg3_iq3d_r7pp2_m32_nw4k128",
        "pfg3_iq3d_r7pp1_m64_nw4k128", "pfg3_ffn_r7pp1_m32_nw4k128"]
for n in MINE:
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
  print(f"[load] {n}: shmem {pr[n].shmem_usage} regs {pr[n].regs_usage}", flush=True)
dev.synchronize(); P._keep.clear()

CT = []
t0 = time.perf_counter()
dt = pf_prefill.prefill_batch(E, None, ids, chunk_times=CT)
dev.synchronize()
cts = sorted(ms for _, ms in CT)
print(f"[boot] prefill 2048: {dt:.2f}s chunks={len(cts)} med={cts[len(cts)//2]:.1f}ms", flush=True)

W7 = E._pf_W7
blocks = [i for i in range(64) if ("fd", i) in W7]
assert len(blocks) >= 8, f"fd coverage {len(blocks)}/64"
KB = blocks[:8]
print(f"[boot] W7 fd coverage {len(blocks)}/64; bench blocks {KB}", flush=True)

# deterministic inputs into PLAN buffers (win_up law)
P.win_up("gact32", 0, (np.random.default_rng(11).standard_normal((32, 17408)) * 0.7).astype(np.float16).reshape(-1))
g64 = np.zeros((64, 17408), np.float16); g64[:32] = P.down("gact32", (32, 17408), np.float16)
g64[32:] = (np.random.default_rng(12).standard_normal((32, 17408)) * 0.7).astype(np.float16)
P.win_up("gact64", 0, g64.reshape(-1))
dev.synchronize()


SH_M32 = pr["pfg3_iq3d_r7_m32_nw8k128"]
SH_N32 = pr["pfg3_iq3d_r7_m32_nt32_nw4k128"]
SH_M64 = pr["pfg3_iq3d_r7_m64_nw8k128"]
PP1    = pr["pfg3_iq3d_r7pp1_m32_nw8k128"]
PP2    = pr["pfg3_iq3d_r7pp2_m32_nw4k128"]
PP1M64 = pr["pfg3_iq3d_r7pp1_m64_nw4k128"]
LS = (256, 1, 1); LS4 = (128, 1, 1)

def run_m32(p, i, out):
  p(W7[("fd", i)], d["gridf"], d["gact32"], d["hh32"], out, global_size=(80, 1, 1), local_size=LS)
def run_n32(p, i, out):
  p(W7[("fd", i)], d["gridf"], d["gact32"], d["hh32"], out, global_size=(160, 1, 1), local_size=LS4)
def run_m64(p, i, out):
  p(W7[("fd", i)], d["gridf"], d["gact64"], d["hh64"], out, global_size=(80, 1, 1), local_size=LS)
def run_p64(p, i, out):
  p(W7[("fd", i)], d["gridf"], d["gact64"], d["hh64"], out, global_size=(160, 1, 1), local_size=LS4)
def grab32():
  return P.down("xA32", (32, 5120), np.float32).copy()
def grab64():
  return P.down("xA64", (64, 5120), np.float32).copy()

print("== G1: bit-identity (first clean run) ==", flush=True)
# shipped m32 ref (rows 0..31 of the m64 inputs must MATCH gact32/hh32 pairing:
# hh32 is the live post-prefill residual; m64 uses hh64 — compare each class to ITSELF)
run_m32(SH_M32, KB[0], d["xB32"]); dev.synchronize()
ref32 = P.down("xB32", (32, 5120), np.float32).copy()
ALL_OK = True
for nm, fn in [("pp1-nw8", lambda p, i, o: run_m32(p, i, o)), ("pp2-nw4", lambda p, i, o: run_n32(p, i, o))]:
  pp = PP1 if nm == "pp1-nw8" else PP2
  fn(pp, KB[0], d["xA32"]); dev.synchronize()
  m1 = grab32()
  fn(pp, KB[0], d["xA32"]); dev.synchronize()
  m2 = grab32()
  nz = int((m1 != ref32).sum()); det = int((m1 != m2).sum())
  ok = nz == 0 and det == 0
  ALL_OK &= ok
  print(f"[gate] {nm} vs shipped-m32: nz={nz}/{m1.size} det-x2 nz={det} -> {'BIT-IDENTICAL' if ok else 'DIFF'}", flush=True)
run_n32(SH_N32, KB[0], d["xB32"]); dev.synchronize()
refn = P.down("xB32", (32, 5120), np.float32).copy()
nz = int((refn != ref32).sum())
print(f"[gate] shipped-nt32 vs shipped-nw8 control: nz={nz} (geometry A/B)", flush=True)
# m64 pair: shipped m64 rows 0..31 must equal shipped m32 (P15 seam law, same inputs rows0..31)
run_m64(SH_M64, KB[0], d["xB64"]); dev.synchronize()
ref64 = P.down("xB64", (64, 5120), np.float32).copy()
run_p64(PP1M64, KB[0], d["xA64"]); dev.synchronize()
mine64 = grab64()
nz64 = int((mine64 != ref64).sum())
print(f"[gate] pp1-m64-nw4 vs shipped-m64: nz={nz64}/{mine64.size} -> {'BIT-IDENTICAL' if nz64 == 0 else 'DIFF'}", flush=True)
ALL_OK &= (nz64 == 0)
print(f"[G1] {'ALL BIT-IDENTICAL' if ALL_OK else 'IDENTITY FAILED'}", flush=True)

print("== G2: bench (synced min-of-10 over 8 blocks) ==", flush=True)
def bench(fn, n=10):
  fn(KB[0]); dev.synchronize()
  best = 1e9
  for _ in range(n):
    t0 = time.perf_counter()
    for i in KB: fn(i)
    dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  return best / len(KB)

WB = 17408 * 98 / 8 * 5120 / 5120 * (98 * (17408 >> 8))  # placeholder, printed per-class below
WBYTES = 98 * (17408 // 256) * 5120  # fd original bytes per block
res = {}
for nm, fn in [("shipped-m32-nw8", lambda i: run_m32(SH_M32, i, d["xB32"])),
               ("shipped-nt32-nw4", lambda i: run_n32(SH_N32, i, d["xB32"])),
               ("pp1-nw8", lambda i: run_m32(PP1, i, d["xB32"])),
               ("pp2-nw4", lambda i: run_n32(PP2, i, d["xB32"]))]:
  t = bench(fn)
  res[nm] = t
  print(f"[bench] {nm:18s} {t*1e6:7.1f} us/blk | w {WBYTES/t/1e9:6.1f} GB/s | x{res['shipped-m32-nw8']/t:4.2f}", flush=True)
for nm, fn in [("shipped-m64-nw8", lambda i: run_m64(SH_M64, i, d["xB64"])),
               ("pp1-m64-nw4", lambda i: run_p64(PP1M64, i, d["xB64"]))]:
  t = bench(fn)
  res[nm] = t
  print(f"[bench] {nm:18s} {t*1e6:7.1f} us/blk (64-row) | per-64 equiv vs 2xshipped-m32 x{(2*res['shipped-m32-nw8'])/t:4.2f}", flush=True)

t32 = res["shipped-m32-nw8"]
bestpp = min(res["pp1-nw8"], res["pp2-nw4"])
r = t32 / bestpp
print(f"[G2] per-32: shipped {t32*1e6:.1f}us best-pp {bestpp*1e6:.1f}us -> RATIO {r:.3f}", flush=True)
verdict = "GO (>=1.35)" if r >= 1.35 else ("DEAD (<1.2)" if r < 1.2 else "GRAY-ZONE (1.2-1.35)")
print(f"[VERDICT] {verdict}", flush=True)

print("== G3: in-graph check (pp2 34816B <= 36.8KB law; pp1 43520B document) ==", flush=True)
seq_sh = [(SH_M32, (W7[("fd", i)], d["gridf"], d["gact32"], d["hh32"], d["xB32"]), 80, LS) for i in KB[:4]]
seq_pp2 = [(PP2, (W7[("fd", i)], d["gridf"], d["gact32"], d["hh32"], d["xB32"]), 160, LS4) for i in KB[:4]]
seq_pp1 = [(PP1, (W7[("fd", i)], d["gridf"], d["gact32"], d["hh32"], d["xB32"]), 80, LS) for i in KB[:4]]
def time_graph(g, n=6):
  best = 1e9
  for _ in range(n):
    prev = dev.timeline_value - 1
    t1 = time.perf_counter()
    v = dev.next_timeline(); g.submit(prev, v)
    dev.timeline_signal.wait(v)
    best = min(best, time.perf_counter() - t1)
  return best * 1e3 / len(KB[:4])
try:
  g_sh = PfGraph(seq_sh, "p16sh"); print(f"[graph] shipped-m32 {time_graph(g_sh):7.1f} us/blk", flush=True)
except Exception as ex:
  print(f"[graph] shipped-m32 capture FAIL: {ex}", flush=True)
for nm, sq in [("pp2-nw4", seq_pp2), ("pp1-nw8", seq_pp1)]:
  try:
    g = PfGraph(sq, "p16" + nm[:3]); print(f"[graph] {nm:10s} {time_graph(g):7.1f} us/blk", flush=True)
  except Exception as ex:
    print(f"[graph] {nm} capture FAIL: {type(ex).__name__} {ex}", flush=True)
print("[p16] DONE", flush=True)
