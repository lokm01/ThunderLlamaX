# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P1 shape sweep: bench each class across all built (NTHR,NTILE,KCH) combos,
synced timing, min-of-10. Reports best shape per class + best-total projection."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev, parse_gguf, read_raw, iq3_grid_f32
from trunk import iq3s_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
PACKED = f"{BASE}/packed"
P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

ds, infos = parse_gguf()
attn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" in infos]
gdn_idx = [i for i in range(64) if i not in set(attn_idx)]
qtype = {i: infos[f"blk.{i}.attn_q.weight"][0] for i in attn_idx}
oq_type = {i: infos[f"blk.{i}.ssm_out.weight"][0] for i in gdn_idx}
G0 = gdn_idx[0]
A14 = next(i for i in attn_idx if qtype[i] == 14)
A18 = next(i for i in attn_idx if qtype[i] == 18)
O18 = next(i for i in gdn_idx if oq_type[i] == 18)
O8 = next(i for i in gdn_idx if oq_type[i] == 8)

P.up("gridf", iq3_grid_f32())
P.up("grid512", iq3s_grid_f32())
P.up("w_gate", np.load(f"{PACKED}/gate{G0}.npy"))
P.up("w_fd", np.load(f"{PACKED}/fd{G0}.npy"))
P.up("w_o18", np.load(f"{PACKED}/out{O18}.npy"))
P.up("w_fg", np.load(f"{PACKED}/fg{G0}.npy"))
P.up("w_fu", np.load(f"{PACKED}/fu{G0}.npy"))
P.up("w_qkv", np.frombuffer(read_raw(infos[f"blk.{G0}.attn_qkv.weight"], ds), dtype=np.uint8))
P.up("w_q6", np.load(f"{PACKED}/q{A14}.npy"))
P.up("w_q3", np.load(f"{PACKED}/q{A18}.npy"))
P.up("w_k3b", np.load(f"{PACKED}/k{A18}.npy"))
P.up("w_v4b", np.frombuffer(read_raw(infos[f"blk.{A18}.attn_v.weight"], ds), dtype=np.uint8))
P.up("w_os", np.frombuffer(read_raw(infos[f"blk.{A14}.attn_output.weight"], ds), dtype=np.uint8))
P.up("w_q8", np.frombuffer(read_raw(infos[f"blk.{O8}.ssm_out.weight"], ds), dtype=np.uint8))
dev.synchronize()

# (tag, namestem, nd, kd, rowb, inst, wnames, gridname, xscale, fused2x)
CLASSES = [
  ("ffn_gu_fg", "pfg_ffn", 17408, 5120, 1960, 64, ("w_fg", "w_fu"), "gridf", 0.8, True),
  ("qkv_q5",    "pfg_q5kv", 10240, 5120, 3520, 48, ("w_qkv",), "gridf", 0.8, False),
  ("gate_iq3",  "pfg_iq3g", 6144, 5120, 1960, 48, ("w_gate",), "gridf", 0.8, False),
  ("down_iq3",  "pfg_iq3d", 5120, 17408, 6664, 64, ("w_fd",), "gridf", 0.4, False),
  ("o18_iq3",   "pfg_iq3o", 5120, 6144, 2352, 24, ("w_o18",), "gridf", 0.8, False),
  ("o8_q8",     "pfg_q8o", 5120, 6144, 6528, 24, ("w_q8",), "gridf", 0.8, False),
  ("aq_q6",     "pfg_q6q", 12288, 5120, 4240, 8, ("w_q6",), "gridf", 0.8, False),
  ("aq_iq3",    "pfg_iq3q", 12288, 5120, 1960, 8, ("w_q3",), "gridf", 0.8, False),
  ("ak_iq3",    "pfg_iq3k", 1024, 5120, 1960, 16, ("w_k3b",), "gridf", 0.8, False),
  ("av_q4k",    "pfg_q4v", 1024, 5120, 2880, 16, ("w_v4b",), "gridf", 0.8, False),
  ("ao_iq3s",   "pfg_iq3s", 5120, 6144, 2640, 16, ("w_os",), "grid512", 0.8, False),
]
SHAPES = [(256, 64, 64), (256, 64, 128), (512, 128, 64), (512, 128, 128), (1024, 256, 64)]  # NTILE must be NWARP*8 (warp tile hardcoded 8)
rng = np.random.default_rng(5)

def bench_one(name, nd, kd, ntile, nthr, wargs, gn, reps=10):
  if not os.path.exists(f"{BASE}/{name}.cubin"): return None
  x = (rng.standard_normal((16, kd)) * 0.8).astype(np.float16)
  P.up("x16buf", x.reshape(-1))
  P.poison("out16", 16*nd*2, np.float16, 7.7)
  dev.synchronize()
  args = wargs + (P.d[gn], P.d["x16buf"], P.d["out16"])
  pr = prog(name)
  grid = nd // ntile
  try:
    for _ in range(2): pr(*args, global_size=(grid, 1, 1), local_size=(nthr, 1, 1)); dev.synchronize()
    best = 1e9
    for _ in range(reps):
      t0 = time.perf_counter(); pr(*args, global_size=(grid, 1, 1), local_size=(nthr, 1, 1)); dev.synchronize()
      best = min(best, time.perf_counter() - t0)
    return best
  except Exception as e:
    print(f"  {name}: FAULT {type(e).__name__}", flush=True)
    return None

best_total = {"hm": 0.0, "hf": 0.0}
head_done = False
print(f"{'class':<12} {'mode':<3} {'shape':<14} {'ms':>8} {'GB/s':>7} {'TFLOP':>6} {'chunk':>8}")
for tag, stem, nd, kd, rowb, inst, wn, gn, xs, f2 in CLASSES:
  for mode in ("hm", "hf"):
    rows = []
    for (nthr, ntile, kch) in SHAPES:
      name = f"{stem}_{mode}_nw{nthr//32}k{kch}"
      dt = bench_one(name, nd, kd, ntile, nthr, tuple(P.d[b] for b in wn), gn)
      if dt is None: continue
      mult = 2 if f2 else 1
      gbs = rowb*nd*mult/dt/1e9; tfs = 2*16*nd*kd*mult/dt/1e12
      rows.append((dt, name, nthr, gbs, tfs))
      print(f"{tag:<12} {mode:<3} nw{nthr//32}k{kch:<8} {dt*1000:8.3f} {gbs:7.1f} {tfs:6.2f} {dt*1000*inst:8.2f}", flush=True)
    if rows:
      b = min(rows)
      best_total[mode] += b[0]*1000*inst
      print(f"  BEST {tag} {mode}: {b[1]} {b[0]*1000:.3f} ms", flush=True)

# head (load 2.5GB once, bench all shapes)
wh = np.frombuffer(read_raw(infos["output.weight"], ds), dtype=np.uint8)
P.up("w_head", wh); dev.synchronize(); del wh
for mode in ("hm", "hf"):
  rows = []
  for (nthr, ntile, kch) in SHAPES:
    name = f"pfg_q5h_{mode}_nw{nthr//32}k{kch}"
    dt = bench_one(name, 248320, 5120, ntile, nthr, (P.d["w_head"],), "gridf", reps=6)
    if dt is None: continue
    rows.append((dt, name))
    print(f"{'head_q5':<12} {mode:<3} nw{nthr//32}k{kch:<8} {dt*1000:8.3f} {3520*248320/dt/1e9:7.1f} {2*16*248320*5120/dt/1e12:6.2f} {dt*1000:8.2f}", flush=True)
  if rows:
    b = min(rows)
    best_total[mode] += b[0]*1000
    print(f"  BEST head {mode}: {b[1]} {b[0]*1000:.3f} ms", flush=True)

for mode in ("hm", "hf"):
  t = best_total[mode]
  tf = 16*49e9/(t/1000)/1e12
  print(f"[proj-best] {mode}: GEMM chunk {t:.1f} ms -> {16*1000/t:.1f} tok/s | {tf:.1f} TFLOPS eff | MFU {tf/71*100:.1f}% (71T) / {tf/35.6*100:.1f}% (35.6T)", flush=True)
