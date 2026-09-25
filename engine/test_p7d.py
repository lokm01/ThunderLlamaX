# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-D validate+bench: the widened attention (pfaW/pfcW) vs the SHIPPED pfa16
pair. READOUT-ORDER LAW: gates from the first clean run; timing after.

Gates:
  1. pfk_pre64 bit-identity: one 64-row launch (grid (24,1), pos_arr) vs 4x
     shipped pfk_pre16 16-row launches — kv/sc/qw bytes IDENTICAL.
  2. attention class gate: pfa32 (R192, 32 tok, one launch) + pfc32 vs the
     shipped pfa16 x2 halves + pfc16 x2 on identical kv/sc/qw inputs —
     final ao16 relerr med <= 3e-3, F <= 1e-2 (online-softmax re-batching
     class); pm/ps spot-check via the row-index map; pfa24 (R144) same gate.
  3. determinism: pfa32+pfc32 twice -> byte-identical.
Bench (synced min-of-10): per L in {2048, 32768, 100352}: shipped pair (2x
  pfa16+pfc per 32 tok) vs pfa32 s{8,32,64,128,256} + pfa24 — ms per 32-tok
  layer, effective KV GB/s (honest byte counts), TFLOPS.
Usage: ~/tg311/bin/python -u test_p7d.py [gate|bench] (default all)
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
LS = (256, 1, 1)
CTXK = 100352
P = Bufs()
_cache = {}
KSYM = {"pfk_pre16_100k": "pfk_pre16", "pfa16nw32_s32_100k": "pfa16", "pfc16_s32": "pfc16",
        "pfk_pre64_100k": "pfk_pre64"}  # LAW: split("_")[0] mangles multi-underscore symbols
def prog(n):
  if n not in _cache:
    lib = open(f"{BASE}/{n}.cubin", "rb").read()
    _cache[n] = NVProgram(dev, TinyELF(lib=lib, name=KSYM.get(n, n.split("_")[0]), target=dev.renderer.target, signature=tuple()))
  return _cache[n]

def relerr(mine, ref, floor=1e-3):
  ref = np.asarray(ref, dtype=np.float64); mine = np.asarray(mine, dtype=np.float64)
  act = np.abs(ref) > floor * max(np.abs(ref).max(), 1e-9)
  e = np.abs(mine[act] - ref[act]) / np.abs(ref[act])
  return (float(np.median(e)) if e.size else 0.0,
          float(np.linalg.norm(mine - ref) / max(np.linalg.norm(ref), 1e-9)))

# ---------------- inputs ----------------
rng = np.random.default_rng(7)
QROW, KROW, VROW = 64, 64, 64
P.up("qrow64", (rng.standard_normal((QROW, 12288)) * 0.6).astype(np.float16))
P.up("krow64", (rng.standard_normal((KROW, 1024)) * 0.6).astype(np.float16))
P.up("vrow64", (rng.standard_normal((VROW, 1024)) * 0.6).astype(np.float16))
P.up("qnw", (1.0 + rng.standard_normal(256) * 0.1).astype(np.float32))
P.up("knw", (1.0 + rng.standard_normal(256) * 0.1).astype(np.float32))
P.up("freqs", np.exp(-np.arange(32, dtype=np.float64) / 32.0 * np.log(1e7)).astype(np.float32))
for tag in ("a", "b"):
  P.up(f"kv{tag}", np.zeros(2 * 4 * CTXK * 256, dtype=np.uint8))   # engine recipe (Dext-law: poison'd kv/sc = launch hang)
  P.up(f"sc{tag}", np.zeros(2 * 4 * CTXK * 8, dtype=np.float16))  # NOTE: count-style = 2x nbytes, same as trunk.py
P.poison("qw64a", 64 * 6144 * 2, np.float16, 7.7)   # qw stride = 6144 halves/token (24h x 256)
P.poison("qw64b", 64 * 6144 * 2, np.float16, 7.7)
P.up("pos0", np.zeros(1, dtype=np.int32))
P.up("pos16", np.array([16], dtype=np.int32))
P.up("pos32", np.array([32], dtype=np.int32))
P.up("pos48", np.array([48], dtype=np.int32))
dev.synchronize(); P._keep.clear()
print("[harness] inputs ready", flush=True)

def kpre16(dst_kv, dst_sc, dst_qw, row0, posbuf):
  off_q = row0 * 12288 * 2; off_k = row0 * 1024 * 2; off_v = row0 * 1024 * 2
  q4 = P.d["qrow64"].offset(offset=off_q, size=16 * 12288 * 2)
  k4 = P.d["krow64"].offset(offset=off_k, size=16 * 1024 * 2)
  v4 = P.d["vrow64"].offset(offset=off_v, size=16 * 1024 * 2)
  off_w = row0 * 6144 * 2
  qw4 = P.d[dst_qw].offset(offset=off_w, size=16 * 6144 * 2)
  prog("pfk_pre16_100k")(q4, k4, v4, P.d["qnw"], P.d["knw"], P.d["freqs"],
                         P.d[dst_kv], P.d[dst_sc], posbuf, qw4,
                         global_size=(24, 1, 1), local_size=LS)

# ============ GATE 1: pfk_pre64 bit-identity ============
def gate1():
  for row0, pb in ((0, "pos0"), (16, "pos16"), (32, "pos32"), (48, "pos48")):
    kpre16("kva", "sca", "qw64a", row0, P.d[pb])
  prog("pfk_pre64_100k")(P.d["qrow64"], P.d["krow64"], P.d["vrow64"], P.d["qnw"], P.d["knw"], P.d["freqs"],
                         P.d["kvb"], P.d["scb"], P.d["pos0"], P.d["qw64b"],
                         global_size=(24, 1, 1), local_size=LS)
  dev.synchronize()
  nrow = 64 * 256
  same_kv = same_sc = same_qw = 0
  for gsel in range(8):
    off = gsel * CTXK * 256
    a = P.down_at("kva", off, nrow, np.uint8); b = P.down_at("kvb", off, nrow, np.uint8)
    same_kv += int((a != b).sum())
  for gsel in range(8):
    off = gsel * CTXK * 8 * 2
    a = P.down_at("sca", off, 64 * 8, np.float16); b = P.down_at("scb", off, 64 * 8, np.float16)
    same_sc += int((a != b).sum())
  qa = P.down("qw64a", (64, 6144), np.float16); qb = P.down("qw64b", (64, 6144), np.float16)
  same_qw = int((qa != qb).sum())
  ok = same_kv == 0 and same_sc == 0 and same_qw == 0
  print(f"[g1] pfk_pre64 vs 4x pfk_pre16: kv {same_kv} sc {same_sc} qw {same_qw} mismatches -> {'PASS (BIT-IDENTICAL)' if ok else 'FAIL'}", flush=True)
  return ok

# ============ GATE 2/3: attention class + determinism ============
def gate23():
  S = 32
  # inputs: 32 tokens via the shipped kpre (bit-identical into BOTH kv sets)
  kpre16("kva", "sca", "qw64a", 0, P.d["pos0"])
  kpre16("kva", "sca", "qw64a", 16, P.d["pos16"])
  dev.synchronize()
  # partial/acc buffers
  P.poison("pmA", 4*S*96*4, np.float32, 7.7e31); P.poison("psA", 4*S*96*4, np.float32, 7.7e31)
  P.poison("pAA", 4*S*96*256*4, np.float32, 7.7e31)
  P.poison("pmB", 4*S*96*4, np.float32, 7.7e31); P.poison("psB", 4*S*96*4, np.float32, 7.7e31)
  P.poison("pAB", 4*S*96*256*4, np.float32, 7.7e31)
  P.poison("pm32", 4*S*192*4, np.float32, 7.7e31); P.poison("ps32", 4*S*192*4, np.float32, 7.7e31)
  P.poison("pA32", 4*S*192*256*4, np.float32, 7.7e31)
  P.poison("aoA", 16*6144*2, np.float16, 7.7); P.poison("aoB", 16*6144*2, np.float16, 7.7)
  P.poison("ao32", 32*6144*2, np.float16, 7.7)
  dev.synchronize(); P._keep.clear()
  qwA = P.d["qw64a"].offset(offset=0, size=16*6144*2)
  qwB = P.d["qw64a"].offset(offset=16*6144*2, size=16*6144*2)
  qrowA = P.d["qrow64"].offset(offset=0, size=16*12288*2)
  qrowB = P.d["qrow64"].offset(offset=16*12288*2, size=16*12288*2)
  # shipped: half A then half B (pos 0 / 16)
  prog("pfa16nw32_s32_100k")(P.d["kva"], P.d["sca"], qwA, P.d["pos0"], P.d["pmA"], P.d["psA"], P.d["pAA"],
                             global_size=(4*S,1,1), local_size=(1024,1,1))
  prog("pfa16nw32_s32_100k")(P.d["kva"], P.d["sca"], qwB, P.d["pos16"], P.d["pmB"], P.d["psB"], P.d["pAB"],
                             global_size=(4*S,1,1), local_size=(1024,1,1))
  prog("pfc16_s32")(P.d["pmA"], P.d["psA"], P.d["pAA"], qrowA, P.d["aoA"], global_size=(24,1,1), local_size=LS)
  prog("pfc16_s32")(P.d["pmB"], P.d["psB"], P.d["pAB"], qrowB, P.d["aoB"], global_size=(24,1,1), local_size=LS)
  dev.synchronize()
  aoA = P.down("aoA", (16, 6144), np.float16).copy(); aoB = P.down("aoB", (16, 6144), np.float16).copy()
  ref = np.concatenate([aoA, aoB], axis=0)
  # mine: pfa32 (R192) one launch, pos 0
  for rep in range(2):
    prog("pfa32nw16_s32_100k")(P.d["kva"], P.d["sca"], P.d["qw64a"], P.d["pos0"], P.d["pm32"], P.d["ps32"], P.d["pA32"],
                               global_size=(4*S,1,1), local_size=(512,1,1))
    prog("pfc32_s32")(P.d["pm32"], P.d["ps32"], P.d["pA32"], P.d["qrow64"], P.d["ao32b"], global_size=(24,1,1), local_size=LS)
    dev.synchronize()
    if rep == 0:
      mine = P.down("ao32", (32, 6144), np.float16).copy()
      pm32 = P.down("pm32", (4*S*192,), np.float32).copy()
      ps32 = P.down("ps32", (4*S*192,), np.float32).copy()
      pA32 = P.down("pA32", (4*S*192, 256), np.float32).copy()
    else:
      mine2 = P.down("ao32", (32, 6144), np.float16).copy()
  m, F = relerr(mine, ref, floor=0.02)
  bitdet = int((mine != mine2).sum())
  print(f"[g2] pfa32 ao vs shipped-pair ao: med {m:.3e} F {F:.3e} maxdiff {np.abs(mine.astype(np.float32)-ref.astype(np.float32)).max():.3e} -> "
        f"{'PASS' if m <= 3e-3 and F <= 1e-2 else 'FAIL'}", flush=True)
  print(f"[g3] pfa32 determinism: {bitdet} byte mismatches -> {'PASS' if bitdet == 0 else 'FAIL'}", flush=True)
  # pm/ps spot check via the row map (pfa32 row r = tok + 32*hh; pfa16 r' = tok + 16*hh)
  pmA_full = P.down("pmA", (4*S, 96), np.float32)
  psA_full = P.down("psA", (4*S, 96), np.float32)
  pA32v = pA32.reshape(4*S, 192, 256)
  pAA = P.down("pAA", (4*S, 96, 256), np.float32)
  em = es = epa = 0.0
  for hh, tok in ((0, 3), (3, 7), (5, 15)):
    for gsel in range(4):
      for s2 in range(S):
        r32 = tok + 32*hh; r16 = tok + 16*hh
        em = max(em, abs(pm32[(gsel*S+s2)*192 + r32] - pmA_full[gsel*S+s2, r16]) / max(abs(pmA_full[gsel*S+s2, r16]), 1e-9))
        es = max(es, abs(ps32[(gsel*S+s2)*192 + r32] - psA_full[gsel*S+s2, r16]) / max(abs(psA_full[gsel*S+s2, r16]), 1e-9))
      d = pA32v[gsel*S:(gsel*S+S), r32, :] - pAA[gsel*S:(gsel*S+S), r16, :]
      epa = max(epa, float(np.linalg.norm(d) / max(np.linalg.norm(pAA[gsel*S:(gsel*S+S), r16, :]), 1e-9)))
  print(f"[g2] partials spot (pm maxrel {em:.2e} ps maxrel {es:.2e} pA F {epa:.2e})", flush=True)
  return m <= 3e-3 and F <= 1e-2 and bitdet == 0

# ============ BENCH ============
def bench():
  P.up("kvc", np.zeros(2 * 4 * CTXK * 256, dtype=np.uint8))
  scv = (rng.uniform(0.01, 0.08, 2 * 4 * CTXK * 8)).astype(np.float16)
  P.up("scc", scv)
  for s in (8, 32, 64, 128, 256):
    P.poison(f"pmw{s}", 4*s*192*4, np.float32, 7.7e31); P.poison(f"psw{s}", 4*s*192*4, np.float32, 7.7e31)
    P.poison(f"pAw{s}", 4*s*192*256*4, np.float32, 7.7e31)
    P.poison(f"pmh{s}", 4*s*96*4, np.float32, 7.7e31); P.poison(f"psh{s}", 4*s*96*4, np.float32, 7.7e31)
    P.poison(f"pAh{s}", 4*s*96*256*4, np.float32, 7.7e31)
  P.poison("ao32b", 32*6144*2, np.float16, 7.7)
  P.poison("pm24", 4*32*144*4, np.float32, 7.7e31); P.poison("ps24", 4*32*144*4, np.float32, 7.7e31)
  P.poison("pA24", 4*32*144*256*4, np.float32, 7.7e31)
  dev.synchronize(); P._keep.clear()
  print(f"[bench] {'L':>6} {'shipped-pair':>13} {'pfa32 s32':>10} {'s64':>8} {'s128':>8} {'s256':>8} {'s8':>8} {'pfa24':>8} {'ctl16nw16':>10}", flush=True)
  for L in (2048, 32768, 100352):
    P.win_up("pos0", 0, np.array([L - 32], dtype=np.int32))
    P.win_up("pos16", 0, np.array([L - 16], dtype=np.int32))
    dev.synchronize()
    def tmin(fn, n=10):
      best = 1e9
      for _ in range(n):
        t0 = time.perf_counter(); fn(); dev.synchronize(); best = min(best, time.perf_counter() - t0)
      return best
    qwA = P.d["qw64a"].offset(offset=0, size=16*12288*2)
    qwB = P.d["qw64a"].offset(offset=16*12288*2, size=16*12288*2)
    def shipped():
      prog("pfa16nw32_s32_100k")(P.d["kvc"], P.d["scc"], qwA, P.d["pos0"], P.d["pmh32"], P.d["psh32"], P.d["pAh32"], global_size=(128,1,1), local_size=(1024,1,1))
      prog("pfa16nw32_s32_100k")(P.d["kvc"], P.d["scc"], qwB, P.d["pos16"], P.d["pmh32"], P.d["psh32"], P.d["pAh32"], global_size=(128,1,1), local_size=(1024,1,1))
      prog("pfc16_s32")(P.d["pmh32"], P.d["psh32"], P.d["pAh32"], P.d["qrow64"], P.d["ao32b"], global_size=(24,1,1), local_size=LS)
      prog("pfc16_s32")(P.d["pmh32"], P.d["psh32"], P.d["pAh32"], P.d["qrow64"], P.d["ao32b"], global_size=(24,1,1), local_size=LS)
    ts = tmin(shipped)
    res = {}
    for s in (8, 32, 64, 128, 256):
      def mine(s=s):
        prog(f"pfa32nw16_s{s}_100k")(P.d["kvc"], P.d["scc"], P.d["qw64a"], P.d["pos0"], P.d[f"pmw{s}"], P.d[f"psw{s}"], P.d[f"pAw{s}"],
                                    global_size=(4*s,1,1), local_size=(512,1,1))
        prog(f"pfc32_s{s}")(P.d[f"pmw{s}"], P.d[f"psw{s}"], P.d[f"pAw{s}"], P.d["qrow64"], P.d["ao32b"], global_size=(24,1,1), local_size=LS)
      res[s] = tmin(mine)
    def mine24():
      prog("pfa24nw16_s32_100k")(P.d["kvc"], P.d["scc"], P.d["qw64a"], P.d["pos0"], P.d["pm24"], P.d["ps24"], P.d["pA24"],
                                 global_size=(128,1,1), local_size=(512,1,1))
      prog("pfc24_s32")(P.d["pm24"], P.d["ps24"], P.d["pA24"], P.d["qrow64"], P.d["ao32b"], global_size=(24,1,1), local_size=LS)
    res24 = tmin(mine24)
    def ctl():
      prog("pfa16ctl_nw16_s32_100k")(P.d["kvc"], P.d["scc"], qwA, P.d["pos0"], P.d["pmh32"], P.d["psh32"], P.d["pAh32"], global_size=(128,1,1), local_size=(512,1,1))
      prog("pfa16ctl_nw16_s32_100k")(P.d["kvc"], P.d["scc"], qwB, P.d["pos16"], P.d["pmh32"], P.d["psh32"], P.d["pAh32"], global_size=(128,1,1), local_size=(512,1,1))
      prog("pfc16ctl_s32")(P.d["pmh32"], P.d["psh32"], P.d["pAh32"], P.d["qrow64"], P.d["ao32b"], global_size=(24,1,1), local_size=LS)
    tc = tmin(ctl)
    t32 = res[32]
    kvb = 2048 * L            # bytes one 32-tok window reads once (K+V, 4 groups)
    flops = 786432 * L        # QK+PV over R=192 rows (2x for the shipped pair reads)
    print(f"[bench] {L:>6} {ts*1e3:>10.2f}ms {res[32]*1e3:>7.2f}ms {res[64]*1e3:>6.2f} {res[128]*1e3:>6.2f} {res[256]*1e3:>6.2f} {res[8]*1e3:>6.2f} {res24*1e3:>6.2f} {tc*1e3:>8.2f}", flush=True)
    print(f"[bench]   pfa32-s32: {kvb/t32/1e9:.0f} GB/s eff (1x KV) | {flops/t32/1e12:.1f} TFLOPS | shipped-pair {2*kvb/ts/1e9:.0f} GB/s (2x KV reads) {2*flops/ts/1e12:.1f} TFLOPS | speedup {ts/t32:.2f}x", flush=True)

if __name__ == "__main__":
  arg = next((a for a in sys.argv[1:] if a in ("gate", "bench")), None)
  ok = True
  if arg in (None, "gate"):
    ok &= gate1(); ok &= gate23()
  if arg in (None, "bench"):
    bench()
  print(f"[test_p7d] {'ALL PASS' if ok else 'FAIL'}", flush=True)
