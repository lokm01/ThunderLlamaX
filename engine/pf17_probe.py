# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P17 probe: the wide-M attention standalone validation + bench.
corr100k/corr2k = per-row BIT-IDENTITY vs the shipped t32 pair (pfa32c_t32 +
pfc16t) on real-scale synthetic kv8: the wide window at pos P covers rows
[P, P+ROWS); rows [P+16k, P+16(k+1)) must equal the t32 run at pos P+16k
(the causal mask ka <= pos + t_ makes the extra KV stream rows contribute
exactly zero to earlier rows: fully-masked tiles evolve ms/ss/acc by *1.0
and +0.0). Determinism x2, poison-first everywhere.
bench = synced min-of-5 (eager law) @pos {2032, 48000, 100288}: per-shape
per-64-chunk-equivalent attention ms vs the shipped 4x t32+pfc16t pair.
Run: cd ~/tinygrad-metal/engine0 && <FULL env> ~/tg311/bin/python -u pf17_probe.py corr100k|corr2k|bench
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from tinygrad.device import Device, TinyELF
from tinygrad.runtime.ops_nv import NVProgram
from engine0 import Bufs

BASE = "~/tinygrad-metal/engine0"
dev = Device["NV"]
P = Bufs()
mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
CTXK = 2048 if mode == "corr2k" else 100352
rng = np.random.default_rng(11)
print(f"[env] AUTO={os.getenv('NV_SMEM_CFG_AUTO','-')} ANAMES={os.getenv('NV_SMEM_CFG_AUTO_NAMES','-')} "
      f"mode={mode} CTXK={CTXK}", flush=True)

_pc = {}
def prog(cub, ent):
  if (cub, ent) in _pc: return _pc[(cub, ent)]
  lib = open(f"{BASE}/{cub}.cubin", "rb").read()
  _pc[(cub, ent)] = NVProgram(dev, TinyELF(lib=lib, name=ent, target=dev.renderer.target, signature=tuple()))
  return _pc[(cub, ent)]

SH = "2k" if CTXK == 2048 else "100k"
SUF = "_s13_2k" if CTXK == 2048 else "_s13_100k"
# name -> (attn cubin, attn entry, comb cubin, comb entry, rows, hrp, nw, s)
WIDE = {
  "w32":  ("pfaw_w32"  + SUF, "pfaw32",  "pfcw32_s13_2k" if CTXK == 2048 else "pfcw32_s13", "pfcw32",  32, 2, 16, 13),
  "w64":  ("pfaw_w64_s13_100k",  "pfaw64",  "pfcw64_s13",  "pfcw64",  64, 2, 16, 13),
  "w64q": ("pfaw_w64q_s13_100k", "pfaw64q", "pfcw64_s13",  "pfcw64",  64, 2, 32, 13),
  "w64h": ("pfaw_w64h_s13_100k", "pfaw64h", "pfcw64h_s13", "pfcw64h", 64, 1, 16, 13),
}
if mode == "corr2k": WIDE = {"w32": WIDE["w32"]}   # 2k corr twins built for w32 only

# ---- world buffers (real-scale synthetic, P7F2 distributions; the P17
# allocation-pattern law: every buffer poison+up paired, pf10-shaped, and a
# warm-up pair before any corr sequence — the bare-world first-pair fault) ----
kv = np.clip(rng.integers(0, 256, (2*4*CTXK*256,)).astype(np.uint8), 0, 255)
P.poison("kv", 2*4*CTXK*256, np.uint8, 200); P.up("kv", kv)
sc = (rng.uniform(0.001, 0.02, (2*4*CTXK*8))).astype(np.float16)
P.poison("sc", 2*4*CTXK*8*2, np.float16, np.float16(7.7)); P.up("sc", sc)
qw32 = (rng.standard_normal(32*24*256) * 0.5).astype(np.float16)
P.poison("qw", 32*24*256*2, np.float16, np.float16(7.7)); P.up("qw", qw32)
qrow16 = (rng.standard_normal(16*12288) * 0.3).astype(np.float16)
P.poison("qrow16", 16*12288*2, np.float16, np.float16(7.7)); P.up("qrow16", qrow16)
P.poison("ao16", 16*6144*2, np.float16, np.float16(7.7))
P.poison("atc_ctr", 4, np.uint32, 0xFF); P.up("atc_ctr", np.zeros(1, dtype=np.uint32))
P.poison("pos_slot", 4, np.int32, -1)
P.poison("pm", 4*32*96*4, np.float32, 7.7e31)
P.poison("ps", 4*32*96*4, np.float32, 7.7e31)
P.poison("pA", 4*32*96*256*4, np.float32, 7.7e31)
P._keep.clear(); dev.synchronize()

def set_pos(pos):
  P.win_up("pos_slot", 0, np.array([pos], dtype=np.int32)); dev.synchronize()

# ---- warm-up control pair (the P17 bare-world first-pair fault law) ----
S13, NHP3, R32 = 13, 3, 32
_wu_pos = 100336 if CTXK == 100352 else 2032
P.win_up("pos_slot", 0, np.array([_wu_pos], dtype=np.int32)); dev.synchronize()
prog(f"pfa32c_t32_s13_{SH}", "pfa32ct")(P.d["kv"], P.d["sc"], P.d["qw"], P.d["pos_slot"],
  P.d["pm"], P.d["ps"], P.d["pA"], global_size=(156,1,1), local_size=(512,1,1))
dev.synchronize()
prog("pfc16t_s13", "pfc16t")(P.d["pm"], P.d["ps"], P.d["pA"], P.d["qrow16"], P.d["ao16"],
  global_size=(24,1,1), local_size=(256,1,1))
dev.synchronize(); print(f"[wu] warm-up pair ok @pos{_wu_pos}", flush=True)

# ---- the shipped t32 pair (reference): 16 rows at pos, qw/qrow window k ----
P.poison("pmR", 4*S13*NHP3*R32*4, np.float32, 7.7e31)
P.poison("psR", 4*S13*NHP3*R32*4, np.float32, 7.7e31)
P.poison("pAR", 4*S13*NHP3*R32*256*4, np.float32, 7.7e31)
P.poison("qrowR", 16*12288*2, np.float16, np.float16(7.7))
dev.synchronize(); P._keep.clear()
qw = (rng.standard_normal(64*24*256) * 0.5).astype(np.float16)   # 64 rows max
P.poison("qw64", 64*24*256*2, np.float16, np.float16(7.7)); P.up("qw64", qw)
qrow64 = (rng.standard_normal(64*12288) * 0.3).astype(np.float16)
P.poison("qrow64", 64*12288*2, np.float16, np.float16(7.7)); P.up("qrow64", qrow64)
P.poison("aoW", 64*6144*2, np.float16, np.float16(7.7))
MAXSLOT = 4*26*6*64   # 39936 slots covers every corr/bench shape we launch
P.poison("pmW", MAXSLOT*4, np.float32, 7.7e31)
P.poison("psW", MAXSLOT*4, np.float32, 7.7e31)
P.poison("pAW", MAXSLOT*256*4, np.float32, 7.7e31)
P.poison("qrowW", 64*12288*2, np.float16, np.float16(7.7))
P._keep.clear(); dev.synchronize()

def run_t32_pair(pos, k):
  gs = 4*S13*NHP3
  P.win_up("pmR", 0, np.full(gs*R32, 7.7e31, np.float32))
  P.win_up("psR", 0, np.full(gs*R32, 7.7e31, np.float32))
  P.win_up("pAR", 0, np.full(gs*R32*256, 7.7e31, np.float32))
  P.win_up("qrowR", 0, qrow64[k*16:(k+1)*16].copy())
  P.win_up("ao16", 0, np.full(16*6144, 7.7, dtype=np.float16))
  set_pos(pos)
  dev.synchronize()
  qwb = P.d["qw64"].offset(offset=k*16*12288, size=16*12288)
  prog(f"pfa32c_t32_s13_{SH}", "pfa32ct")(P.d["kv"], P.d["sc"], qwb, P.d["pos_slot"],
    P.d["pmR"], P.d["psR"], P.d["pAR"], global_size=(gs,1,1), local_size=(512,1,1))
  dev.synchronize(); print(f"[dbg] t32 attn k={k} pos={pos} ok", flush=True)
  prog("pfc16t_s13", "pfc16t")(P.d["pmR"], P.d["psR"], P.d["pAR"], P.d["qrowR"], P.d["ao16"],
    global_size=(24,1,1), local_size=(256,1,1))
  dev.synchronize(); print(f"[dbg] t32 comb k={k} ok", flush=True)
  o = P.down("ao16", (16, 6144), np.float16).astype(np.float32).copy()
  P._keep.clear()
  return o

# ---- the wide pair: ROWS rows at pos (poison-first, det via double call) ----
def run_wide(pos, nm):
  acub, aent, ccub, cent, rows, hrp, nw, s = WIDE[nm]
  nhp = 6 // hrp; gs = 4*s*nhp; rmax = hrp*rows
  slots = gs*rmax
  assert slots <= MAXSLOT, slots
  P.win_up("pmW", 0, np.full(slots, 7.7e31, np.float32))
  P.win_up("psW", 0, np.full(slots, 7.7e31, np.float32))
  P.win_up("pAW", 0, np.full(slots*256, 7.7e31, np.float32))
  P.win_up("aoW", 0, np.full(rows*6144, 7.7, dtype=np.float16))
  P.win_up("qrowW", 0, qrow64[:rows].copy())
  set_pos(pos); dev.synchronize()
  pmb = P.d["pmW"].offset(offset=0, size=slots*4)
  psb = P.d["psW"].offset(offset=0, size=slots*4)
  pab = P.d["pAW"].offset(offset=0, size=slots*256*4)
  aob = P.d["aoW"].offset(offset=0, size=rows*6144*2)
  qrb = P.d["qrowW"].offset(offset=0, size=rows*12288*2)
  prog(acub, aent)(P.d["kv"], P.d["sc"], P.d["qw64"], P.d["pos_slot"], pmb, psb, pab,
    global_size=(gs,1,1), local_size=(nw*32,1,1))
  prog(ccub, cent)(pmb, psb, pab, qrb, aob, global_size=(24,1,1), local_size=(256,1,1))
  dev.synchronize()
  o = P.down("aoW", (rows, 6144), np.float16).astype(np.float32).copy()
  P._keep.clear()
  return o

if mode.startswith("corr"):
  pos0 = int(os.getenv("POS", "1984" if CTXK == 2048 else "100224"))
  refs = [run_t32_pair(pos0 + 16*k, k) for k in range(4)]
  ref = np.concatenate(refs, axis=0)   # (64, 6144) — the 4-window truth
  for nm in WIDE:
    try:
      o1 = run_wide(pos0, nm); o2 = run_wide(pos0, nm)
    except Exception as e:
      print(f"[corr] {nm}: LAUNCH FAIL {type(e).__name__}: {e}", flush=True); continue
    rows = WIDE[nm][4]
    det = np.array_equal(o1, o2)
    nz = int((o1 != ref[:rows]).sum())
    mad = float(np.abs(o1 - ref[:rows]).max())
    e = np.abs(o1 - ref[:rows]) / np.maximum(np.abs(ref[:rows]), 1e-3)
    print(f"[corr] {nm} @pos{pos0} rows={rows}: BIT-IDENTICAL nz={nz}/{o1.size} "
          f"maxabsdiff {mad:.3e} relerr med {np.median(e):.3e} max {np.max(e):.3e} | det x2 {det} "
          f"| poison {(o1 == 7.7).sum()}", flush=True)
else:
  # bench: per-64-chunk-equivalent attention cost per shape (eager, synced min-of-5)
  res = {}
  def bench(name, fn):
    for _ in range(2): fn()
    dev.synchronize()
    best = 1e9
    for _ in range(5):
      t0 = time.perf_counter(); fn(); dev.synchronize()
      best = min(best, time.perf_counter() - t0)
    print(f"[bench] {name}: {best*1e3:8.2f} ms", flush=True)
    return best*1e3
  for pos in (2032, 48000, 100288):
    print(f"---- pos {pos} ----", flush=True)
    set_pos(pos)
    pk = prog("pfa32c_t32_s13_100k", "pfa32ct"); pc = prog("pfc16t_s13", "pfc16t")
    pmb, psb, pab = P.d["pmW"], P.d["psW"], P.d["pAW"]
    def t32_pair():
      pk(P.d["kv"], P.d["sc"], P.d["qw64"], P.d["pos_slot"], pmb, psb, pab,
         global_size=(156,1,1), local_size=(512,1,1))
      pc(pmb, psb, pab, P.d["qrow64"].offset(offset=0, size=16*12288*2),
         P.d["aoW"].offset(offset=0, size=16*6144*2), global_size=(24,1,1), local_size=(256,1,1))
    bench("4x t32+pfc pair (shipped/64chunk)", lambda: [t32_pair() for _ in range(4)])
    for nm, nlaunch in (("w32", 2), ("w64", 1), ("w64q", 1), ("w64h", 1)):
      acub, aent, ccub, cent, rows, hrp, nw, s = WIDE[nm]
      nhp = 6 // hrp; gs = 4*s*nhp; slots = gs*hrp*rows
      pm2 = P.d["pmW"].offset(offset=0, size=slots*4)
      ps2 = P.d["psW"].offset(offset=0, size=slots*4)
      pa2 = P.d["pAW"].offset(offset=0, size=slots*256*4)
      qr2 = P.d["qrow64"].offset(offset=0, size=rows*12288*2)
      ao2 = P.d["aoW"].offset(offset=0, size=rows*6144*2)
      pk2 = prog(acub, aent); pc2 = prog(ccub, cent)
      def pair():
        pk2(P.d["kv"], P.d["sc"], P.d["qw64"], P.d["pos_slot"], pm2, ps2, pa2,
            global_size=(gs,1,1), local_size=(nw*32,1,1))
        pc2(pm2, ps2, pa2, qr2, ao2, global_size=(24,1,1), local_size=(256,1,1))
      bench(f"{nlaunch}x {nm}+pfc pair /64chunk", lambda: [pair() for _ in range(nlaunch)])
    if pos >= 8192:
      slots = 312*64
      pm3 = P.d["pmW"].offset(offset=0, size=slots*4)
      ps3 = P.d["psW"].offset(offset=0, size=slots*4)
      pa3 = P.d["pAW"].offset(offset=0, size=slots*256*4)
      qr3 = P.d["qrow64"].offset(offset=0, size=32*12288*2)
      ao3 = P.d["aoW"].offset(offset=0, size=32*6144*2)
      pk3 = prog("pfaw_w32_s26_100k", "pfaw32"); pc3 = prog("pfcw32_s26", "pfcw32")
      def pair26():
        pk3(P.d["kv"], P.d["sc"], P.d["qw64"], P.d["pos_slot"], pm3, ps3, pa3,
            global_size=(312,1,1), local_size=(512,1,1))
        pc3(pm3, ps3, pa3, qr3, ao3, global_size=(24,1,1), local_size=(256,1,1))
      bench("2x w32s26+pfc pair /64chunk", lambda: [pair26() for _ in range(2)])
      pkt = prog("pfa32c_t32_s26_100k", "pfa32ct"); pct = prog("pfc16t_s26", "pfc16t")
      def t32_pair26():
        pkt(P.d["kv"], P.d["sc"], P.d["qw64"], P.d["pos_slot"], pm3, ps3, pa3,
            global_size=(312,1,1), local_size=(512,1,1))
        pct(pm3, ps3, pa3, P.d["qrow64"].offset(offset=0, size=16*12288*2),
            P.d["aoW"].offset(offset=0, size=16*6144*2), global_size=(24,1,1), local_size=(256,1,1))
      bench("4x t32s26+pfc pair /64chunk", lambda: [t32_pair26() for _ in range(4)])
P._keep.clear()
print("[pf17_probe done]", flush=True)
