# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7-E4: gemm3-m64 AT M=256 standalone validation vs classic m32 x8 — the
EXACT SC call pattern (ONE launch per kernel per chunk, grid = NGRID*MP64,
flat M-grid over full 256-row buffers, NO per-pass offsets). The P7E TODO:
the P7E2 launch-226 wrong-VA corruption blamed m64-at-M256; if it was the
dirty-world machine class this must be BIT-IDENTICAL on the clean machine.
Plus 64KB guard canaries (overrun detection) + synced bench (m64@256 single
launch vs m32 x8 = the ~150ms/chunk question).
Usage: ~/tg311/bin/python -u test_p7b256.py [filter ...]
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev, parse_gguf, read_raw, iq3_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
PACKED = f"{BASE}/packed"
P7 = f"{BASE}/packed7"
LS = (256, 1, 1)
M = 256
MP64 = M // 64
MP32 = M // 32
GUARD = 65536  # guard canary bytes after every mine/ref output
P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

P.up("gridf", iq3_grid_f32())
dev.synchronize()

ds, infos = parse_gguf()
attn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" in infos]
gdn_idx = [i for i in range(64) if i not in set(attn_idx)]
G0 = gdn_idx[0]
A18 = next(i for i in attn_idx if infos[f"blk.{i}.attn_q.weight"][0] != 14)
print(f"[blocks] gdn0={G0} attn-iq3={A18}  M={M} MP64={MP64} MP32={MP32}", flush=True)

P.up("w_fg", np.load(f"{PACKED}/fg{G0}.npy"))
P.up("w_fu", np.load(f"{PACKED}/fu{G0}.npy"))
P.up("w_fd", np.load(f"{PACKED}/fd{G0}.npy"))
OQ = next(i for i in gdn_idx if os.path.exists(f"{PACKED}/out{i}.npy"))
P.up("w_o18", np.load(f"{PACKED}/out{OQ}.npy"))
P.up("w_qkv5", np.frombuffer(read_raw(infos[f"blk.{G0}.attn_qkv.weight"], ds), dtype=np.uint8))
P.up("w_v4", np.frombuffer(read_raw(infos[f"blk.{A18}.attn_v.weight"], ds), dtype=np.uint8))
P.up("r_fg", np.load(f"{P7}/fg{G0}.npy"))
P.up("r_fu", np.load(f"{P7}/fu{G0}.npy"))
P.up("r_fd", np.load(f"{P7}/fd{G0}.npy"))
P.up("r_o", np.load(f"{P7}/out{OQ}.npy"))
P.up("r_gate", np.load(f"{P7}/gate{G0}.npy"))
P.up("r_q", np.load(f"{P7}/q{A18}.npy"))
P.up("r_k", np.load(f"{P7}/k{A18}.npy"))
rng = np.random.default_rng(7)
P.up("res256", (rng.standard_normal((M, 5120)) * 4.0).astype(np.float32).reshape(-1))
P.up("w_gate_o", np.load(f"{PACKED}/gate{G0}.npy"))
P.up("w_q18_o", np.load(f"{PACKED}/q{A18}.npy"))
P.up("w_k18_o", np.load(f"{PACKED}/k{A18}.npy"))
dev.synchronize()
print("[weights] up", flush=True)

only = [a for a in sys.argv[1:]] or None
ALL_OK = True
GUARD_VAL = np.float32(7.7e31)

def poison_guarded(name, nbytes, dtype, val):
  P.poison(name, nbytes + GUARD, dtype, val)

# ---- singles: (tag, kd, nd, res, (refkernel, refweights, g32), (m64kernel, m64weights, g64, ls64))
SINGLES = [
  ("ffn", 5120, 17408, False,
   ("pfg_ffn_m32_hm_nw8k128", ("w_fg", "w_fu"), 272),
   ("pfg3_ffn_r7_m64_nw4k128", ("r_fg", "r_fu"), 544, (128, 1, 1))),
  ("iq3d", 17408, 5120, True,
   ("pfg_iq3d_m32_res_hm_nw8k128", ("w_fd",), 80),
   ("pfg3_iq3d_r7_m64_nw8k128", ("r_fd",), 80, LS)),
  ("iq3o", 6144, 5120, False,
   ("pfg_iq3o_m32_hm_nw8k128", ("w_o18",), 80),
   ("pfg3_iq3o_r7_m64_nw8k128", ("r_o",), 80, LS)),
]
for tag, kd, nd, res, (rn, rw, g32), (mn, mw, g64, ls64) in SINGLES:
  if only and not any(o in tag for o in only): continue
  P.up(f"x{tag}", (rng.standard_normal((M, kd)) * 0.8).astype(np.float16).reshape(-1))
  elt = 4 if res else 2
  dt = np.float32 if res else np.float16
  pval = 7.7e31 if res else 7.7
  P.poison(f"ref{tag}", M*nd*elt, dt, pval)
  poison_guarded(f"mine{tag}", M*nd*elt, dt, pval)
  dev.synchronize()
  # ref: classic m32 x8, 32-row offsets (the pf_prefill fallback pattern)
  pr = prog(rn)
  for p in range(MP32):
    x = P.d[f"x{tag}"] if p == 0 else P.d[f"x{tag}"].offset(offset=p*32*kd*2, size=32*kd*2)
    o = P.d[f"ref{tag}"] if p == 0 else P.d[f"ref{tag}"].offset(offset=p*32*nd*elt, size=32*nd*elt)
    rr = P.d["res256"] if p == 0 else P.d["res256"].offset(offset=p*32*5120*4, size=32*5120*4)
    a = tuple(P.d[w] for w in rw) + (P.d["gridf"], x)
    if res: a = a + (rr, o)
    else: a = a + (o,)
    pr(*a, global_size=(g32, 1, 1), local_size=LS)
  dev.synchronize()
  ref = P.down(f"ref{tag}", (M, nd), dt).astype(np.float32)
  # mine: ONE m64 launch, grid = NGRID*MP64, full 256-row buffers (the SC pattern)
  pr2 = prog(mn)
  P.poison(f"mine{tag}", M*nd*elt + GUARD, dt, pval)
  dev.synchronize()
  a = tuple(P.d[w] for w in mw) + (P.d["gridf"], P.d[f"x{tag}"])
  if res: a = a + (P.d["res256"], P.d[f"mine{tag}"])
  else: a = a + (P.d[f"mine{tag}"],)
  pr2(*a, global_size=(g64*MP64, 1, 1), local_size=ls64)
  dev.synchronize()
  mine = P.down(f"mine{tag}", (M, nd), dt).astype(np.float32)
  nz = int((mine != ref).sum())
  ok = nz == 0
  ALL_OK &= ok
  msg = "BIT-IDENTICAL" if ok else f"DIFF nz={nz}/{mine.size}"
  if not ok:
    d = np.abs(mine - ref); rel = d / np.maximum(np.abs(ref), 1e-6)
    bad = np.argwhere(mine != ref)
    msg += f" maxrel {rel.max():.3e} first-bad {tuple(bad[0]) if len(bad) else None}"
  # guard canary check
  graw = P.down(f"mine{tag}", (int((M*nd*elt + GUARD)/np.dtype(dt).itemsize),), dt)
  gv = np.asarray([pval], dtype=dt)[0]
  gbad = int((graw[M*nd*elt // np.dtype(dt).itemsize:] != gv).sum())
  print(f"[gate256] {tag:<5} m64@M256 vs m32x8: {msg}; guard bad {gbad}", flush=True)
  if gbad: ALL_OK = False
  if not ok: continue
  # bench: m64 single launch vs m32 x8 over the same 256 rows
  def one_m64():
    a = tuple(P.d[w] for w in mw) + (P.d["gridf"], P.d[f"x{tag}"])
    if res: a = a + (P.d["res256"], P.d[f"mine{tag}"])
    else: a = a + (P.d[f"mine{tag}"],)
    pr2(*a, global_size=(g64*MP64, 1, 1), local_size=ls64)
  def one_m32():
    for p in range(MP32):
      x = P.d[f"x{tag}"] if p == 0 else P.d[f"x{tag}"].offset(offset=p*32*kd*2, size=32*kd*2)
      o = P.d[f"mine{tag}"] if p == 0 else P.d[f"mine{tag}"].offset(offset=p*32*nd*elt, size=32*nd*elt)
      rr = P.d["res256"] if p == 0 else P.d["res256"].offset(offset=p*32*5120*4, size=32*5120*4)
      a = tuple(P.d[w] for w in rw) + (P.d["gridf"], x)
      if res: a = a + (rr, o)
      else: a = a + (o,)
      pr(*a, global_size=(g32, 1, 1), local_size=LS)
  for f, nm in ((one_m64, "m64@256"), (one_m32, "m32x8")):
    f(); dev.synchronize()
    best = 1e9
    for _ in range(10):
      t0 = time.perf_counter(); f(); dev.synchronize()
      best = min(best, time.perf_counter() - t0)
    print(f"[bench256] {tag:<5} {nm}: {best*1e3:7.3f} ms/256r", flush=True)

# ---- twins: (tag, kd, outs, refkernel/refweights/g32/ls32, m64kernel/m64weights/g64/ls64)
TWINS = [
  ("gdnqg", 5120, [("qkv32", 10240), ("gate32", 6144)],
   ("pfg2_gdnqg_m32_hm_nw16k128", ("w_qkv5", "w_gate_o"), 128, (512, 1, 1)),
   ("pfg3m_gdnqg_r7_m64_nw8k128", ("w_qkv5", "r_gate"), 256, LS)),
  ("attnqkvi3", 5120, [("qrow", 12288), ("krow", 1024), ("vrow", 1024)],
   ("pfg2_attnqkvi3_m32_hm_nw8k128", ("w_q18_o", "w_k18_o", "w_v4"), 224, LS),
   ("pfg3m_attnqkvi3_r7_m64_nw8k128", ("r_q", "r_k", "w_v4"), 224, LS)),
]
for tag, kd, outs, (rn, rw, g32, ls32), (mn, mw, g64, ls64) in TWINS:
  if only and not any(o in tag for o in only): continue
  osum = sum(n for _, n in outs)
  P.up(f"xt_{tag}", (rng.standard_normal((M, kd)) * 0.5).astype(np.float16).reshape(-1))
  for on, ondim in outs:
    P.poison(f"{on}_r", M*ondim*2, np.float16, 7.7)
    P.poison(f"{on}_m", M*ondim*2 + GUARD, np.float16, 7.7)
  dev.synchronize()
  pr = prog(rn)
  for p in range(MP32):
    x = P.d[f"xt_{tag}"] if p == 0 else P.d[f"xt_{tag}"].offset(offset=p*32*kd*2, size=32*kd*2)
    oas = tuple(P.d[f"{on}_r"] if p == 0 else P.d[f"{on}_r"].offset(offset=p*32*n*2, size=32*n*2) for on, n in outs)
    pr(*(tuple(P.d[w] for w in rw) + (P.d["gridf"], x) + oas), global_size=(g32, 1, 1), local_size=ls32)
  dev.synchronize()
  refs = [P.down(f"{on}_r", (M, n), np.float16).astype(np.float32) for on, n in outs]
  pr2 = prog(mn)
  for on, n in outs: P.poison(f"{on}_m", M*n*2 + GUARD, np.float16, 7.7)
  dev.synchronize()
  oas = tuple(P.d[f"{on}_m"] for on, n in outs)
  pr2(*(tuple(P.d[w] for w in mw) + (P.d["gridf"], P.d[f"xt_{tag}"]) + oas),
      global_size=(g64*MP64, 1, 1), local_size=ls64)
  dev.synchronize()
  nz = tot = 0; firstbad = None
  for (on, n), rr in zip(outs, refs):
    mm = P.down(f"{on}_m", (M, n), np.float16).astype(np.float32)
    d = mm != rr
    nz += int(d.sum()); tot += mm.size
    if d.any() and firstbad is None:
      firstbad = (on, tuple(np.argwhere(d)[0]))
  gbad = 0
  for on, n in outs:
    graw = P.down(f"{on}_m", (int((M*n*2 + GUARD)/2),), np.float16)
    gbad += int((graw[M*n:] != np.float16(7.7)).sum())
  ok = nz == 0 and gbad == 0
  ALL_OK &= ok
  print(f"[gate256] {tag:<9} m64@M256 vs m32x8: {'BIT-IDENTICAL' if nz==0 else f'DIFF nz={nz}/{tot} first-bad {firstbad}'}; guard bad {gbad}", flush=True)
  if nz: continue
  def one_m64():
    oas = tuple(P.d[f"{on}_m"] for on, n in outs)
    pr2(*(tuple(P.d[w] for w in mw) + (P.d["gridf"], P.d[f"xt_{tag}"]) + oas),
        global_size=(g64*MP64, 1, 1), local_size=ls64)
  def one_m32():
    for p in range(MP32):
      x = P.d[f"xt_{tag}"] if p == 0 else P.d[f"xt_{tag}"].offset(offset=p*32*kd*2, size=32*kd*2)
      oas = tuple(P.d[f"{on}_m"] if p == 0 else P.d[f"{on}_m"].offset(offset=p*32*n*2, size=32*n*2) for on, n in outs)
      pr(*(tuple(P.d[w] for w in rw) + (P.d["gridf"], x) + oas), global_size=(g32, 1, 1), local_size=ls32)
  for f, nm in ((one_m64, "m64@256"), (one_m32, "m32x8")):
    f(); dev.synchronize()
    best = 1e9
    for _ in range(10):
      t0 = time.perf_counter(); f(); dev.synchronize()
      best = min(best, time.perf_counter() - t0)
    print(f"[bench256] {tag:<9} {nm}: {best*1e3:7.3f} ms/256r", flush=True)

print("[p7b256] ALL BIT-IDENTICAL" if ALL_OK else "[p7b256] DIFFERENCES FOUND", flush=True)
sys.exit(0 if ALL_OK else 1)
