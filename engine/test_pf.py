# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P1 pGEMM validation + bench: differential vs the T=1 bit-proven reference
kernels (w1c.cu family) on REAL packed weights, 16 random fp16 input rows.
Poison-first, distinct output buffers, args built AFTER the last P.up.
Bench: synced timing only (launch -> dev.synchronize per rep), min-of-N.
Usage: python test_pf.py [validate|bench] [name-filter ...]"""
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
DPACK = f"{BASE}/draft_pack"
LS = (256, 1, 1)
M = 16

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
A14 = next(i for i in attn_idx if qtype[i] == 14)   # Q6_K q block
A18 = next(i for i in attn_idx if qtype[i] == 18)   # IQ3 q block
O18 = next(i for i in gdn_idx if oq_type[i] == 18)  # IQ3 ssm_out block
O8 = next(i for i in gdn_idx if oq_type[i] == 8)    # Q8_0 ssm_out block
print(f"[blocks] gdn0={G0} a14={A14} a18={A18} o18={O18} o8={O8}", flush=True)

# ---- shared tables + scratch ----
P.up("gridf", iq3_grid_f32())
P.up("grid512", iq3s_grid_f32())
for nm, nb, dt in [("xh", 17408*2, np.float16), ("qkv_row", 10240*2, np.float16),
                   ("gate_row", 6144*2, np.float16), ("qrow", 12288*2, np.float16),
                   ("k_row", 1024*2, np.float16), ("v_row", 1024*2, np.float16),
                   ("ao_row", 6144*2, np.float16), ("attn_out", 5120*2, np.float16),
                   ("gact1", 17408*2, np.float16), ("z16", 6144*2, np.float16),
                   ("logits1", 248320*2, np.float16)]:
  P.poison(nm, nb, dt, 7.7)
P.up("hh0", np.zeros(5120, dtype=np.float32))   # down8 residual = 0
P.poison("yf32", 5120*4, np.float32, 7.7e31)    # down8 output is FP32 (trunk: x0/x1)
dev.synchronize()

rng = np.random.default_rng(11)
X16 = {}   # class -> (16,K) fp16 inputs

def gen_x(k, scale=0.8):
  a = (rng.standard_normal((M, k)) * scale).astype(np.float16)
  X16[k] = a
  return a

def relerr(mine, ref):
  mine = mine.astype(np.float32); ref = ref.astype(np.float32)
  act = np.abs(ref) > 1e-6
  if act.sum() == 0: return 0.0, 0.0, 0.0
  e = np.abs(mine[act] - ref[act]) / np.abs(ref[act])
  fn = np.linalg.norm(mine - ref) / max(np.linalg.norm(ref), 1e-9)
  return float(e.max()), float(np.median(e)), float(fn)

def check(tag, mine, ref, gate=3e-3):
  mx, med, fn = relerr(mine, ref)
  pz = int((np.abs(mine.astype(np.float32)) > 1e30).sum())
  ok = (mx <= gate or fn <= gate) and pz == 0
  print(f"[val] {tag}: relerr max {mx:.3e} med {med:.3e} F {fn:.3e} poison {pz} -> {'PASS' if ok else 'FAIL'}", flush=True)
  return ok

def run_ref_row(prname, args, grid, outname, outshape, npdt=np.float16, wait=True):
  prog(prname)(*args, global_size=(grid, 1, 1), local_size=LS, wait=wait)
  return P.down(outname, outshape, npdt)

def myrun(name, wargs, nd, kd, ntile=64, nthr=256, reps=1, xbuf="x16buf"):
  pr = prog(name)
  args = wargs + (P.d["gridf"], P.d[xbuf], P.d["out16"])
  grid = nd // ntile
  for _ in range(2): pr(*args, global_size=(grid, 1, 1), local_size=(nthr, 1, 1)); dev.synchronize()
  t0 = time.perf_counter()
  for _ in range(reps): pr(*args, global_size=(grid, 1, 1), local_size=(nthr, 1, 1)); dev.synchronize()
  dt = (time.perf_counter() - t0) / reps
  return dt

def prep_x(k, scale=0.8):
  a = gen_x(k, scale)
  P.up("x16buf", a.reshape(-1))
  dev.synchronize()
  return a

def prep_out(nd):
  P.poison("out16", M*nd*2, np.float16, 7.7)
  dev.synchronize()

def get_out(nd):
  return P.down("out16", (M, nd), np.float16)

ALL_OK = True
def validate():
  global ALL_OK
  # ---------- IQ3_XXS: gate (q5g8 gate_row) ----------
  wg = np.load(f"{PACKED}/gate{G0}.npy")
  wqkvr = np.frombuffer(read_raw(infos[f"blk.{G0}.attn_qkv.weight"], ds), dtype=np.uint8)
  P.up("w_gate", wg); P.up("w_qkv", wqkvr); dev.synchronize()
  x = prep_x(5120)
  refs = np.zeros((M, 6144), np.float32)
  for m in range(M):
    P.win_up("xh", 0, x[m]); dev.synchronize()
    run_ref_row("q5g8", (P.d["w_qkv"], P.d["w_gate"], P.d["gridf"], P.d["xh"], P.d["qkv_row"], P.d["gate_row"]), 2048, "gate_row", (6144,))
    refs[m] = P.down("gate_row", (6144,), np.float16).astype(np.float32)
  for mode in ("hm", "hf"):
    prep_out(6144)
    dt = myrun(f"pfg_iq3g_{mode}_nw8k128", (P.d["w_gate"],), 6144, 5120)
    ALL_OK &= check(f"iq3g_{mode}", get_out(6144).astype(np.float32), refs)
  print(f"[bench] iq3g hm/hf @nw8k128 above; ref 1-row q5g8 included", flush=True)

  # ---------- IQ3_XXS: fd (down8, hh=0) ----------
  wd = np.load(f"{PACKED}/fd{G0}.npy")
  P.up("w_fd", wd); dev.synchronize()
  x = prep_x(17408, 0.4)
  refs = np.zeros((M, 5120), np.float32)
  for m in range(M):
    P.win_up("gact1", 0, x[m]); dev.synchronize()
    run_ref_row("down8", (P.d["w_fd"], P.d["gridf"], P.d["gact1"], P.d["hh0"], P.d["yf32"]), 640, "yf32", (5120,), npdt=np.float32)
    refs[m] = P.down("yf32", (5120,), np.float32)
  for mode in ("hm", "hf"):
    prep_out(5120)
    myrun(f"pfg_iq3d_{mode}_nw8k128", (P.d["w_fd"],), 5120, 17408)
    ALL_OK &= check(f"iq3d_{mode}", get_out(5120).astype(np.float32), refs)

  # ---------- IQ3_XXS: ssm_out type-18 (op38) ----------
  wo = np.load(f"{PACKED}/out{O18}.npy")
  P.up("w_o18", wo); dev.synchronize()
  x = prep_x(6144)
  refs = np.zeros((M, 5120), np.float32)
  for m in range(M):
    P.win_up("z16", 0, x[m]); dev.synchronize()
    run_ref_row("op38", (P.d["w_o18"], P.d["gridf"], P.d["z16"], P.d["attn_out"]), 640, "attn_out", (5120,))
    refs[m] = P.down("attn_out", (5120,), np.float16).astype(np.float32)
  for mode in ("hm", "hf"):
    prep_out(5120)
    myrun(f"pfg_iq3o_{mode}_nw8k128", (P.d["w_o18"],), 5120, 6144)
    ALL_OK &= check(f"iq3o_{mode}", get_out(5120).astype(np.float32), refs)

  # ---------- fused FFN gate+up (ffn8) ----------
  wfg = np.load(f"{PACKED}/fg{G0}.npy"); wfu = np.load(f"{PACKED}/fu{G0}.npy")
  P.up("w_fg", wfg); P.up("w_fu", wfu); dev.synchronize()
  x = prep_x(5120)
  refs = np.zeros((M, 17408), np.float32)
  for m in range(M):
    P.win_up("xh", 0, x[m]); dev.synchronize()
    run_ref_row("ffn8", (P.d["w_fg"], P.d["w_fu"], P.d["gridf"], P.d["xh"], P.d["gact1"]), 2176, "gact1", (17408,))
    refs[m] = P.down("gact1", (17408,), np.float16).astype(np.float32)
  for mode in ("hm", "hf"):
    prep_out(17408)
    pr = prog(f"pfg_ffn_{mode}_nw8k128")
    args = (P.d["w_fg"], P.d["w_fu"], P.d["gridf"], P.d["x16buf"], P.d["out16"])
    pr(*args, global_size=(17408//64, 1, 1), local_size=LS); dev.synchronize()
    ALL_OK &= check(f"ffn_{mode}", get_out(17408).astype(np.float32), refs)

  # ---------- Q5_K: qkv (q5g8 qkv_row; gate=zeros) ----------
  P.up("w_gate0", np.zeros(wg.shape, np.uint8)); dev.synchronize()
  x = prep_x(5120)
  refs = np.zeros((M, 10240), np.float32)
  for m in range(M):
    P.win_up("xh", 0, x[m]); dev.synchronize()
    run_ref_row("q5g8", (P.d["w_qkv"], P.d["w_gate0"], P.d["gridf"], P.d["xh"], P.d["qkv_row"], P.d["gate_row"]), 2048, "qkv_row", (10240,))
    refs[m] = P.down("qkv_row", (10240,), np.float16).astype(np.float32)
  for mode in ("hm", "hf"):
    prep_out(10240)
    myrun(f"pfg_q5kv_{mode}_nw8k128", (P.d["w_qkv"],), 10240, 5120)
    ALL_OK &= check(f"q5kv_{mode}", get_out(10240).astype(np.float32), refs)

  # ---------- Q5_K head (head8; rows 0-2 only for time) ----------
  wh = np.frombuffer(read_raw(infos["output.weight"], ds), dtype=np.uint8)
  P.up("w_head", wh); dev.synchronize()
  del wh
  x = prep_x(5120)
  for m in range(2):
    P.win_up("xh", 0, x[m]); dev.synchronize()
    run_ref_row("head8", (P.d["w_head"], P.d["xh"], P.d["logits1"]), 248320//8, "logits1", (248320,))
    ref = P.down("logits1", (248320,), np.float16).astype(np.float32)
    if m == 0: refs = np.zeros((2, 248320), np.float32)
    refs[m] = ref
  for mode in ("hm", "hf"):
    prep_out(248320)
    myrun(f"pfg_q5h_{mode}_nw8k128", (P.d["w_head"],), 248320, 5120)
    out = get_out(248320).astype(np.float32)
    ALL_OK &= check(f"q5h_{mode}", out[:2], refs)
  P.d.pop("w_head", None); P._keep.clear(); dev.synchronize()

  # ---------- Q6_K q (aq6k8 qrow) ----------
  wq6 = np.load(f"{PACKED}/q{A14}.npy"); wk3 = np.load(f"{PACKED}/k{A14}.npy")
  wv4 = np.frombuffer(read_raw(infos[f"blk.{A14}.attn_v.weight"], ds), dtype=np.uint8)
  P.up("w_q6", wq6); P.up("w_k3", wk3); P.up("w_v4", wv4); dev.synchronize()
  x = prep_x(5120)
  refs = np.zeros((M, 12288), np.float32)
  for m in range(M):
    P.win_up("xh", 0, x[m]); dev.synchronize()
    run_ref_row("aq6k8", (P.d["w_q6"], P.d["w_k3"], P.d["w_v4"], P.d["gridf"], P.d["xh"], P.d["qrow"], P.d["k_row"], P.d["v_row"]), 1792, "qrow", (12288,))
    refs[m] = P.down("qrow", (12288,), np.float16).astype(np.float32)
  for mode in ("hm", "hf"):
    prep_out(12288)
    myrun(f"pfg_q6q_{mode}_nw8k128", (P.d["w_q6"],), 12288, 5120)
    ALL_OK &= check(f"q6q_{mode}", get_out(12288).astype(np.float32), refs)

  # ---------- IQ3 q/k (aq3k8) ----------
  wq3 = np.load(f"{PACKED}/q{A18}.npy"); wk3b = np.load(f"{PACKED}/k{A18}.npy")
  wv4b = np.frombuffer(read_raw(infos[f"blk.{A18}.attn_v.weight"], ds), dtype=np.uint8)
  P.up("w_q3", wq3); P.up("w_k3b", wk3b); P.up("w_v4b", wv4b); dev.synchronize()
  x = prep_x(5120)
  refs_q = np.zeros((M, 12288), np.float32); refs_k = np.zeros((M, 1024), np.float32)
  for m in range(M):
    P.win_up("xh", 0, x[m]); dev.synchronize()
    run_ref_row("aq3k8", (P.d["w_q3"], P.d["w_k3b"], P.d["w_v4b"], P.d["gridf"], P.d["xh"], P.d["qrow"], P.d["k_row"], P.d["v_row"]), 1792, "qrow", (12288,))
    refs_q[m] = P.down("qrow", (12288,), np.float16).astype(np.float32)
    refs_k[m] = P.down("k_row", (1024,), np.float16).astype(np.float32)
  for mode in ("hm", "hf"):
    prep_out(12288)
    myrun(f"pfg_iq3q_{mode}_nw8k128", (P.d["w_q3"],), 12288, 5120)
    ALL_OK &= check(f"iq3q_{mode}", get_out(12288).astype(np.float32), refs_q)
    prep_out(1024)
    myrun(f"pfg_iq3k_{mode}_nw8k128", (P.d["w_k3b"],), 1024, 5120)
    ALL_OK &= check(f"iq3k_{mode}", get_out(1024).astype(np.float32), refs_k)

  # ---------- Q4_K v (aq3k8 vrow) ----------
  refs_v = np.zeros((M, 1024), np.float32)
  for m in range(M):
    P.win_up("xh", 0, x[m]); dev.synchronize()
    run_ref_row("aq3k8", (P.d["w_q3"], P.d["w_k3b"], P.d["w_v4b"], P.d["gridf"], P.d["xh"], P.d["qrow"], P.d["k_row"], P.d["v_row"]), 1792, "v_row", (1024,))
    refs_v[m] = P.down("v_row", (1024,), np.float16).astype(np.float32)
  for mode in ("hm", "hf"):
    prep_out(1024)
    myrun(f"pfg_q4v_{mode}_nw8k128", (P.d["w_v4b"],), 1024, 5120)
    ALL_OK &= check(f"q4v_{mode}", get_out(1024).astype(np.float32), refs_v)

  # ---------- IQ3_S o (ao8) ----------
  wos = np.frombuffer(read_raw(infos[f"blk.{A14}.attn_output.weight"], ds), dtype=np.uint8)
  P.up("w_os", wos); dev.synchronize()
  x = prep_x(6144)
  refs = np.zeros((M, 5120), np.float32)
  for m in range(M):
    P.win_up("ao_row", 0, x[m]); dev.synchronize()
    run_ref_row("ao8", (P.d["w_os"], P.d["grid512"], P.d["ao_row"], P.d["attn_out"]), 640, "attn_out", (5120,))
    refs[m] = P.down("attn_out", (5120,), np.float16).astype(np.float32)
  for mode in ("hm", "hf"):
    prep_out(5120)
    pr = prog(f"pfg_iq3s_{mode}_nw8k128")
    args = (P.d["w_os"], P.d["grid512"], P.d["x16buf"], P.d["out16"])
    pr(*args, global_size=(5120//64, 1, 1), local_size=LS); dev.synchronize()
    ALL_OK &= check(f"iq3s_{mode}", get_out(5120).astype(np.float32), refs)

  # ---------- Q8_0 ssm_out (k3a_oproj) ----------
  wq8 = np.frombuffer(read_raw(infos[f"blk.{O8}.ssm_out.weight"], ds), dtype=np.uint8)
  P.up("w_q8", wq8); dev.synchronize()
  x = prep_x(6144)
  refs = np.zeros((M, 5120), np.float32)
  for m in range(M):
    P.win_up("z16", 0, x[m]); dev.synchronize()
    run_ref_row("k3a_oproj", (P.d["w_q8"], P.d["z16"], P.d["attn_out"]), 640, "attn_out", (5120,))
    refs[m] = P.down("attn_out", (5120,), np.float16).astype(np.float32)
  for mode in ("hm", "hf"):
    prep_out(5120)
    myrun(f"pfg_q8o_{mode}_nw8k128", (P.d["w_q8"],), 5120, 6144)
    ALL_OK &= check(f"q8o_{mode}", get_out(5120).astype(np.float32), refs)

  # ---------- Q4_0 draft fg (dfgu) ----------
  dfg = np.load(f"{DPACK}/d_fg.npy"); dfu = np.load(f"{DPACK}/d_fu.npy")
  P.up("d_fg", dfg); P.up("d_fu", dfu); dev.synchronize()
  x = prep_x(5120)
  refs = np.zeros((M, 17408), np.float32)
  for m in range(M):
    P.win_up("xh", 0, x[m]); dev.synchronize()
    run_ref_row("dfgu", (P.d["d_fg"], P.d["d_fu"], P.d["xh"], P.d["gact1"]), 2176, "gact1", (17408,))
    refs[m] = P.down("gact1", (17408,), np.float16).astype(np.float32)
  # NOTE: plain q4dn (no epilogue) cannot use dfgu directly; validate via fg-side
  # reconstruction is not possible with silu-mul — so q4dn validated by bench only
  # after numpy cross-check (below, run in validate_q4z path if needed).
  for mode in ("hm", "hf"):
    prep_out(17408)
    myrun(f"pfg_q4dn_{mode}_nw8k128", (P.d["d_fg"],), 17408, 5120)
    out = get_out(17408).astype(np.float32)
    # numpy dequant cross-check of the plain GEMM (fg side, fp32 dot)
    k = 5120
    rows = rng.integers(0, 17408, 256)
    ref_np = np.zeros((M, len(rows)), np.float32)
    for ri, r in enumerate(rows):
      row = dfg[r]
      b32 = np.arange(k) >> 5; o = np.arange(k) & 31
      qb = row[(b32 << 4) + ((o >> 4) << 3) + (o & 7)]
      qv = np.where(o >= 16, qb >> 4, qb) & 0xF
      d = np.repeat(row[k//2:k//2 + (k//32)*2].view(np.float16).astype(np.float32), 32)
      w = d * (qv.astype(np.float32) - 8.0)
      ref_np[:, ri] = x.astype(np.float32) @ w
    e = np.abs(out[:, rows] - ref_np) / np.maximum(np.abs(ref_np), 1e-3)
    print(f"[val] q4dn_{mode} numpy-xcheck relerr max {e.max():.3e} med {np.median(e):.3e} -> {'PASS' if e.max() < 3e-2 else 'FAIL'} (numpy ref = fp32, looser gate)", flush=True)
    ALL_OK &= e.max() < 3e-2

  print(f"[validate] ALL {'PASS' if ALL_OK else 'FAIL'}", flush=True)

# per-class instance counts per 16-token chunk + weight MB
# (tag, namefmt, nd, kd, rowbytes, instances, wnames, gridname, xscale, fused2x)
CLASSES_BENCH = [
  ("ffn_gu_fg", "pfg_ffn_{m}_nw8k128", 17408, 5120, 1960, 64, ("w_fg", "w_fu"), "gridf", 0.8, True),
  ("qkv_q5",    "pfg_q5kv_{m}_nw8k128", 10240, 5120, 3520, 48, ("w_qkv",), "gridf", 0.8, False),
  ("gate_iq3",  "pfg_iq3g_{m}_nw8k128", 6144, 5120, 1960, 48, ("w_gate",), "gridf", 0.8, False),
  ("down_iq3",  "pfg_iq3d_{m}_nw8k128", 5120, 17408, 6664, 64, ("w_fd",), "gridf", 0.4, False),
  ("o18_iq3",   "pfg_iq3o_{m}_nw8k128", 5120, 6144, 2352, 24, ("w_o18",), "gridf", 0.8, False),
  ("o8_q8",     "pfg_q8o_{m}_nw8k128", 5120, 6144, 6528, 24, ("w_q8",), "gridf", 0.8, False),
  ("aq_q6",     "pfg_q6q_{m}_nw8k128", 12288, 5120, 4240, 8, ("w_q6",), "gridf", 0.8, False),
  ("aq_iq3",    "pfg_iq3q_{m}_nw8k128", 12288, 5120, 1960, 8, ("w_q3",), "gridf", 0.8, False),
  ("ak_iq3",    "pfg_iq3k_{m}_nw8k128", 1024, 5120, 1960, 16, ("w_k3b",), "gridf", 0.8, False),
  ("av_q4k",    "pfg_q4v_{m}_nw8k128", 1024, 5120, 2880, 16, ("w_v4b",), "gridf", 0.8, False),
  ("ao_iq3s",   "pfg_iq3s_{m}_nw8k128", 5120, 6144, 2640, 16, ("w_os",), "grid512", 0.8, False),
  ("head_q5",   "pfg_q5h_{m}_nw8k128", 248320, 5120, 3520, 1, ("w_head",), "gridf", 0.8, False),
]

def bench():
  print("[bench] NOTE: fused ffn kernel covers fg+fu in one launch (2x weight bytes/FLOPs)", flush=True)
  # head weight only if validate() already ran in this process; else load
  if "w_head" not in P.d:
    wh = np.frombuffer(read_raw(infos["output.weight"], ds), dtype=np.uint8)
    P.up("w_head", wh); dev.synchronize(); del wh
  total_ms = {"hm": 0.0, "hf": 0.0}
  print(f"{'class':<12} {'mode':<3} {'ms':>9} {'GB/s':>7} {'TFLOPS':>7} {'chunk_ms':>9}", flush=True)
  for tag, kn, nd, kd, rowb, inst, wn, gn, xs, f2 in CLASSES_BENCH:
    prep_x(kd, xs)   # size x16buf for this class's K
    for mode in ("hm", "hf"):
      name = kn.format(m=mode)
      pr = prog(name)
      wargs = tuple(P.d[b] for b in wn)
      prep_out(nd)   # REALLOC FIRST (args-before-up trap, W2G law)
      args = wargs + (P.d[gn], P.d["x16buf"], P.d["out16"])
      grid = nd // 64
      for _ in range(2): pr(*args, global_size=(grid, 1, 1), local_size=LS); dev.synchronize()
      t0 = time.perf_counter()
      for _ in range(10): pr(*args, global_size=(grid, 1, 1), local_size=LS); dev.synchronize()
      dt = (time.perf_counter() - t0) / 10
      mult = 2 if f2 else 1
      nbytes = rowb * nd * mult
      gbs = nbytes / dt / 1e9
      fl = 2 * M * nd * kd * mult
      tfs = fl / dt / 1e12
      total_ms[mode] += dt * 1000 * inst
      print(f"{tag:<12} {mode:<3} {dt*1000:9.3f} {gbs:7.1f} {tfs:7.2f} {dt*1000*inst:9.2f}", flush=True)
  for mode in ("hm", "hf"):
    t = total_ms[mode]
    print(f"[proj] {mode}: GEMM-only chunk {t:.1f} ms -> {16*1000/t:.1f} tok/s | MFU(vs 71T tensor) {16*49e9/t/71e12*100:.1f}% | (vs 35.6T) {16*49e9/t/35.6e12*100:.1f}%", flush=True)

if __name__ == "__main__":
  mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
  if mode == "validate": validate()
  elif mode == "bench": bench()
  elif mode == "all": validate(); bench()
