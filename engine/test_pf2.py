# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P2 differential validation: every new M=16 piece vs the T=1 canonical path.
  1. pfk_emb16  vs h_embed x16            (bit-class gate 1e-3)
  2. pfk_n16/ab16/hh16 vs k0_norm/k0ab/k3m_hh x16  (1e-3)
  3. pfk_pre16  vs spk_pre1qh_100k x16 T=1 (BIT-EXACT gate: kv bytes + qw16)
  4. pfa16+pfc16 vs spk_g4nw32qh1_100k + spk_c1_100k x16 T=1 (3e-3, HMMA class)
     at pos=0 AND pos=512 (multi-tile splits)
  5. pfs16 vs trunk k2s x16 T=1 (1e-3; identical op order -> expect ~1e-6)
Plus synced per-kernel timing (min-of-10). Usage: ~/tg311/bin/python test_pf2.py [piece...]"""
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
LS = (256, 1, 1)
LS1K = (1024, 1, 1)
M = 16
CTXK = 100352
S = 32       # my pfa16 splits
SR = 256     # the T=1 reference cubins (spk_g4nw32qh1_100k/spk_c1_100k) bake S=256

P = Bufs()
KSYM = {"pfk_pre16_2k": "pfk_pre16", "pfk_pre16_100k": "pfk_pre16",
        "pfa16nw32_s32_100k": "pfa16", "pfa16nw32_s8_2k": "pfa16",
        "pfc16_s32": "pfc16", "pfc16_s8": "pfc16", "pfs16": "pfs16"}
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=KSYM.get(n, n), target=dev.renderer.target, signature=tuple()))

ds, infos = parse_gguf()
attn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" in infos]
gdn_idx = [i for i in range(64) if i not in set(attn_idx)]
G0 = gdn_idx[0]
A0 = attn_idx[0]

def relerr(mine, ref):
  mine = np.asarray(mine, dtype=np.float32); ref = np.asarray(ref, dtype=np.float32)
  act = np.abs(ref) > 1e-6
  if act.sum() == 0: return 0.0, 0.0, 0.0
  e = np.abs(mine[act] - ref[act]) / np.abs(ref[act])
  fn = np.linalg.norm(mine - ref) / max(np.linalg.norm(ref), 1e-9)
  return float(e.max()), float(np.median(e)), float(fn)

def check(tag, mine, ref, gate=3e-3):
  mx, med, fn = relerr(mine, ref)
  pz = int((np.abs(np.asarray(mine, dtype=np.float32)) > 1e30).sum()) if np.issubdtype(np.asarray(mine).dtype, np.floating) else 0
  ok = (mx <= gate or fn <= gate) and pz == 0
  print(f"[val] {tag}: relerr max {mx:.3e} med {med:.3e} F {fn:.3e} poison {pz} -> {'PASS' if ok else 'FAIL'}", flush=True)
  return ok

def timing(tag, fn, reps=10):
  fn()  # warm
  dev.synchronize()
  best = 1e30
  for _ in range(reps):
    t0 = time.perf_counter(); fn(); dev.synchronize()
    best = min(best, time.perf_counter() - t0)
  print(f"[time] {tag}: {best*1e3:.3f} ms", flush=True)
  return best

# ---- programs ----
pr = {}
for n in ["pfk_emb16","pfk_n16","pfk_ab16","pfk_hh16","pfk_pre16_100k","pfa16nw32_s32_100k","pfc16_s32","pfs16",
          "h_embed","k0_norm","k0ab","k3m_hh","k2s","spk_pre1qh_100k","spk_g4nw32qh1_100k","spk_c1g_100k"]:
  pr[n] = prog(n)

# ---- tables + GDN block G0 f32 weights ----
P.up("gridf", iq3_grid_f32())
P.up("grid512", iq3s_grid_f32())
freqs = (1.0 / (1e7 ** (np.arange(0, 64, 2, dtype=np.float64) / 64.0))).astype(np.float32)
P.up("freqs", freqs)
pre = f"blk.{G0}."
for nm, arr in [("convw", np.frombuffer(read_raw(infos[pre+"ssm_conv1d.weight"], ds), dtype="<f4").reshape(10240,4)),
                ("dtb", np.frombuffer(read_raw(infos[pre+"ssm_dt.bias"], ds), dtype="<f4")),
                ("ssma", np.frombuffer(read_raw(infos[pre+"ssm_a"], ds), dtype="<f4")),
                ("snw", np.frombuffer(read_raw(infos[pre+"ssm_norm.weight"], ds), dtype="<f4")),
                ("alpha", np.frombuffer(read_raw(infos[pre+"ssm_alpha.weight"], ds), dtype="<f4").reshape(48,5120)),
                ("beta", np.frombuffer(read_raw(infos[pre+"ssm_beta.weight"], ds), dtype="<f4").reshape(48,5120)),
                ("nw1", np.frombuffer(read_raw(infos[pre+"attn_norm.weight"], ds), dtype="<f4")),
                ("nw2", np.frombuffer(read_raw(infos[pre+"post_attention_norm.weight"], ds), dtype="<f4"))]:
  P.up(nm, np.ascontiguousarray(arr))
# attn block A0 q/k norms
preA = f"blk.{A0}."
P.up("qnw", np.frombuffer(read_raw(infos[preA+"attn_q_norm.weight"], ds), dtype="<f4"))
P.up("knw", np.frombuffer(read_raw(infos[preA+"attn_k_norm.weight"], ds), dtype="<f4"))
dev.synchronize()

# ---- scratch (poisoned) ----
def pois(nm, nb, dt, v):
  P.poison(nm, nb, dt, v)
for nm, nb, dt, v in [
    ("xin16", 16*5120*4, np.float32, 7.7e31), ("xh16", 16*5120*2, np.float16, 7.7),
    ("hh16", 16*5120*4, np.float32, 7.7e31), ("hhx16", 16*5120*2, np.float16, 7.7),
    ("attn_out16", 16*5120*2, np.float16, 7.7), ("araw16", 16*48*4, np.float32, 7.7e31),
    ("braw16", 16*48*4, np.float32, 7.7e31), ("alpharaw", 48*4, np.float32, 7.7e31),
    ("betaraw", 48*4, np.float32, 7.7e31), ("qkv16", 16*10240*2, np.float16, 7.7),
    ("gate16", 16*6144*2, np.float16, 7.7), ("z16", 16*6144*2, np.float16, 7.7),
    ("z_t1", 6144*2, np.float16, 7.7), ("q", 48*128*4, np.float32, 7.7e31),
    ("k", 48*128*4, np.float32, 7.7e31), ("v", 48*128*4, np.float32, 7.7e31),
    ("core", 48*128*4, np.float32, 7.7e31), ("convA", 3*10240*4, np.float32, 7.7e31),
    ("convB", 3*10240*4, np.float32, 7.7e31), ("conv_live", 3*10240*4, np.float32, 7.7e31),
    ("rec_ref", 48*128*128*4, np.float32, 7.7e31), ("rec_mine", 48*128*128*4, np.float32, 7.7e31),
    ("x16f", 16*5120*4, np.float32, 7.7e31), ("x_t1", 5120*4, np.float32, 7.7e31),
    ("qrow16", 16*12288*2, np.float16, 7.7), ("krow16", 16*1024*2, np.float16, 7.7),
    ("vrow16", 16*1024*2, np.float16, 7.7), ("qw_t1", 24*256*4, np.float32, 7.7e31),
    ("qw16_t1", 24*256*2, np.float16, 7.7), ("qw16", 16*24*256*2, np.float16, 7.7),
    ("pm_t1", 4*SR*6*4, np.float32, 7.7e31), ("ps_t1", 4*SR*6*4, np.float32, 7.7e31),
    ("pA_t1", 4*SR*6*256*4, np.float32, 7.7e31), ("pm16", 4*S*96*4, np.float32, 7.7e31),
    ("ps16", 4*S*96*4, np.float32, 7.7e31), ("pA16", 4*S*96*256*4, np.float32, 7.7e31),
    ("ao16", 16*6144*2, np.float16, 7.7), ("ao_t1", 6144*2, np.float16, 7.7),
]:
  pois(nm, nb, dt, v)
P.up("pos_slot", np.zeros(1, dtype=np.int32))
P.up("tok_slot", np.zeros(1, dtype=np.int32))
P.up("ids16", np.zeros(16, dtype=np.int32))
# kv: int8 canonical at CTXK=100352 (zeros -> deterministic poison-free base)
P.up("kv", np.zeros(2*4*CTXK*256, dtype=np.uint8))
P.up("sc", np.zeros(2*4*CTXK*8, dtype=np.float16))
dev.synchronize()
print("[setup] buffers ready", flush=True)

rng = np.random.default_rng(42)
d = P.d
RUN = sys.argv[1:] if len(sys.argv) > 1 else None
def want(x): return RUN is None or x in RUN

# =========== piece 1+2: embed + norms ===========
if want("norms"):
  embs = np.frombuffer(read_raw(infos["token_embd.weight"], ds), dtype=np.uint8)
  P.up("embw", embs); del embs
  dev.synchronize(); P._keep.clear()
  ids = rng.integers(0, 248320, size=M).astype(np.int32)
  P.win_up("ids16", 0, ids)
  pr["pfk_emb16"](d["embw"], d["grid512"], d["ids16"], d["x16f"], global_size=(16,1,1), local_size=LS, wait=True)
  x16 = P.down("x16f", (M, 5120), np.float32)
  xref = np.zeros((M, 5120), dtype=np.float32)
  for t in range(M):
    P.win_up("tok_slot", 0, np.array([int(ids[t])], dtype=np.int32))
    pr["h_embed"](d["embw"], d["grid512"], d["tok_slot"], d["x_t1"], global_size=(1,1,1), local_size=LS, wait=True)
    xref[t] = P.down("x_t1", (5120,), np.float32)
  check("pfk_emb16 vs h_embed", x16, xref, 1e-3)

  P.win_up("xin16", 0, (rng.standard_normal((M, 5120)) * 0.7).astype(np.float32))
  pr["pfk_n16"](d["xin16"], d["nw1"], d["xh16"], global_size=(16,1,1), local_size=LS, wait=True)
  xh16 = P.down("xh16", (M, 5120), np.float16)
  ref = np.zeros((M, 5120), dtype=np.float16)
  for t in range(M):
    xr = P.d["xin16"].offset(offset=t*5120*4, size=5120*4)
    pr["k0_norm"](xr, d["nw1"], d["x_t1"], global_size=(1,1,1), local_size=LS, wait=True)
    ref[t] = P.down("x_t1", (5120,), np.float16)
  check("pfk_n16 vs k0_norm", xh16, ref, 1e-3)

  pr["pfk_ab16"](d["xin16"], d["nw1"], d["alpha"], d["beta"], d["xh16"], d["araw16"], d["braw16"],
                 global_size=(16*13,1,1), local_size=LS, wait=True)
  a16 = P.down("araw16", (M, 48), np.float32); b16 = P.down("braw16", (M, 48), np.float32)
  aref = np.zeros((M, 48), np.float32); bref = np.zeros((M, 48), np.float32)
  for t in range(M):
    xr = P.d["xin16"].offset(offset=t*5120*4, size=5120*4)
    pr["k0ab"](xr, d["nw1"], d["alpha"], d["beta"], d["x_t1"], d["alpharaw"], d["betaraw"],
               global_size=(13,1,1), local_size=LS, wait=True)
    aref[t] = P.down("alpharaw", (48,), np.float32); bref[t] = P.down("betaraw", (48,), np.float32)
  check("pfk_ab16 araw vs k0ab", a16, aref, 1e-3)
  check("pfk_ab16 braw vs k0ab", b16, bref, 1e-3)

  ao_in = (rng.standard_normal((M, 5120)) * 0.5).astype(np.float16)
  P.win_up("attn_out16", 0, ao_in)
  pr["pfk_hh16"](d["xin16"], d["attn_out16"], d["nw2"], d["hh16"], d["hhx16"], global_size=(16,1,1), local_size=LS, wait=True)
  hh16 = P.down("hh16", (M, 5120), np.float32); hhx16 = P.down("hhx16", (M, 5120), np.float16)
  href = np.zeros((M, 5120), dtype=np.float32); hxref = np.zeros((M, 5120), dtype=np.float16)
  P.up("hh_t1", np.zeros(5120, dtype=np.float32)); P.up("hhx_t1", np.zeros(5120, dtype=np.float16))
  for t in range(M):
    xr = P.d["xin16"].offset(offset=t*5120*4, size=5120*4)
    ar = P.d["attn_out16"].offset(offset=t*5120*2, size=5120*2)
    pr["k3m_hh"](xr, ar, d["nw2"], d["hh_t1"], d["hhx_t1"], global_size=(1,1,1), local_size=LS, wait=True)
    href[t] = P.down("hh_t1", (5120,), np.float32); hxref[t] = P.down("hhx_t1", (5120,), np.float16)
  check("pfk_hh16 hh vs k3m_hh", hh16, href, 1e-3)
  check("pfk_hh16 hhx vs k3m_hh", hhx16, hxref, 1e-3)
  timing("pfk_n16", lambda: pr["pfk_n16"](d["xin16"], d["nw1"], d["xh16"], global_size=(16,1,1), local_size=LS))
  timing("pfk_ab16", lambda: pr["pfk_ab16"](d["xin16"], d["nw1"], d["alpha"], d["beta"], d["xh16"], d["araw16"], d["braw16"], global_size=(16*13,1,1), local_size=LS))
  timing("pfk_hh16", lambda: pr["pfk_hh16"](d["xin16"], d["attn_out16"], d["nw2"], d["hh16"], d["hhx16"], global_size=(16,1,1), local_size=LS))

# =========== piece 3+4: KPRE + attention ===========
if want("attn"):
  def rand_qkv(seed):
    r = np.random.default_rng(seed)
    P.win_up("qrow16", 0, (r.standard_normal((M, 12288)) * 0.35).astype(np.float16))
    P.win_up("krow16", 0, (r.standard_normal((M, 1024)) * 0.35).astype(np.float16))
    P.win_up("vrow16", 0, (r.standard_normal((M, 1024)) * 0.5).astype(np.float16))

  def run_chunk_pair(pos):
    """mine: pfk_pre16 + pfa16 + pfc16 (kv state advanced); returns (ao16, kv_snapshot_before)."""
    rand_qkv(1000 + pos)
    kv_before = P.down("kv", (2*4*CTXK*256,), np.uint8).copy() if pos else None
    P.win_up("pos_slot", 0, np.array([pos], dtype=np.int32))
    pr["pfk_pre16_100k"](d["qrow16"], d["krow16"], d["vrow16"], d["qnw"], d["knw"], d["freqs"],
                         d["kv"], d["sc"], d["pos_slot"], d["qw16"], global_size=(24,1,1), local_size=LS)
    pr["pfa16nw32_s32_100k"](d["kv"], d["sc"], d["qw16"], d["pos_slot"], d["pm16"], d["ps16"], d["pA16"],
                             global_size=(4*S,1,1), local_size=LS1K)
    pr["pfc16_s32"](d["pm16"], d["ps16"], d["pA16"], d["qrow16"], d["ao16"], global_size=(24,1,1), local_size=LS, wait=True)
    return P.down("ao16", (M, 6144), np.float16)

  def run_t1(pos, kv_before):
    """T=1 reference: restore kv, loop 16 positions through spk_pre1qh/a1/c1."""
    if kv_before is not None: P.win_up("kv", 0, kv_before)
    out = np.zeros((M, 6144), dtype=np.float16)
    kv_ref = None
    for t in range(M):
      P.win_up("pos_slot", 0, np.array([pos + t], dtype=np.int32))
      qr = P.d["qrow16"].offset(offset=t*12288*2, size=12288*2)
      kr = P.d["krow16"].offset(offset=t*1024*2, size=1024*2)
      vr = P.d["vrow16"].offset(offset=t*1024*2, size=1024*2)
      pr["spk_pre1qh_100k"](qr, kr, vr, d["qnw"], d["knw"], d["freqs"], d["kv"], d["sc"], d["pos_slot"],
                            d["qw_t1"], d["qw16_t1"], global_size=(24,1,1), local_size=LS)
      pr["spk_g4nw32qh1_100k"](d["kv"], d["sc"], d["qw16_t1"], d["pos_slot"], d["pm_t1"], d["ps_t1"], d["pA_t1"],
                               global_size=(4*SR,1,1), local_size=LS1K)
      pr["spk_c1g_100k"](d["pm_t1"], d["ps_t1"], d["pA_t1"], qr, d["ao_t1"], global_size=(24,1,1), local_size=LS, wait=True)
      out[t] = P.down("ao_t1", (6144,), np.float16)
      if t == M - 1: kv_ref = P.down("kv", (2*4*CTXK*256,), np.uint8).copy()
    return out, kv_ref

  for pos in (0, 512):
    # pre-fill kv rows [0, pos) with earlier chunks (mine == bit-exact path)
    for c in range(0, pos, M):
      rand_qkv(1000 + c)
      P.win_up("pos_slot", 0, np.array([c], dtype=np.int32))
      pr["pfk_pre16_100k"](d["qrow16"], d["krow16"], d["vrow16"], d["qnw"], d["knw"], d["freqs"],
                           d["kv"], d["sc"], d["pos_slot"], d["qw16"], global_size=(24,1,1), local_size=LS)
    kv_before = P.down("kv", (2*4*CTXK*256,), np.uint8).copy()
    ao_mine = run_chunk_pair(pos)
    kv_mine = P.down("kv", (2*4*CTXK*256,), np.uint8).copy()
    qw16_mine = P.down("qw16", (M, 24*256), np.float16).copy()
    ao_ref, kv_ref = run_t1(pos, kv_before)
    qw16_ref = np.zeros((M, 24*256), dtype=np.float16)
    # T=1 qw16 rows were overwritten each step; re-derive via one clean pass per row
    P.win_up("kv", 0, kv_before)
    for t in range(M):
      P.win_up("pos_slot", 0, np.array([pos + t], dtype=np.int32))
      qr = P.d["qrow16"].offset(offset=t*12288*2, size=12288*2)
      kr = P.d["krow16"].offset(offset=t*1024*2, size=1024*2)
      vr = P.d["vrow16"].offset(offset=t*1024*2, size=1024*2)
      pr["spk_pre1qh_100k"](qr, kr, vr, d["qnw"], d["knw"], d["freqs"], d["kv"], d["sc"], d["pos_slot"],
                            d["qw_t1"], d["qw16_t1"], global_size=(24,1,1), local_size=LS, wait=True)
      qw16_ref[t] = P.down("qw16_t1", (24*256,), np.float16)
    nb = int((kv_mine != kv_ref).sum())
    nq = int((qw16_mine != qw16_ref).sum())
    print(f"[val] pfk_pre16 @pos={pos}: kv byte mismatches {nb}/{kv_mine.size}  qw16 mismatches {nq}/{qw16_mine.size} -> {'PASS' if nb==0 and nq==0 else 'FAIL'}", flush=True)
    check(f"pfa16+pfc16 ao @pos={pos}", ao_mine, ao_ref, 3e-3)

  # timing at pos=100320 (full splits; garbage kv beyond is fine for timing)
  P.win_up("pos_slot", 0, np.array([100320], dtype=np.int32))
  timing("pfk_pre16_100k", lambda: pr["pfk_pre16_100k"](d["qrow16"], d["krow16"], d["vrow16"], d["qnw"], d["knw"], d["freqs"], d["kv"], d["sc"], d["pos_slot"], d["qw16"], global_size=(24,1,1), local_size=LS))
  timing("pfa16 s32 @pos100320", lambda: pr["pfa16nw32_s32_100k"](d["kv"], d["sc"], d["qw16"], d["pos_slot"], d["pm16"], d["ps16"], d["pA16"], global_size=(4*S,1,1), local_size=LS1K))
  timing("pfc16 s32", lambda: pr["pfc16_s32"](d["pm16"], d["ps16"], d["pA16"], d["qrow16"], d["ao16"], global_size=(24,1,1), local_size=LS))

# =========== piece 5: pfs16 scan ===========
if want("scan"):
  P.win_up("conv_live", 0, (rng.standard_normal(3*10240) * 0.4).astype(np.float32))
  P.win_up("convA", 0, P.down("conv_live", (3*10240,), np.float32))
  P.win_up("rec_mine", 0, (rng.standard_normal(48*128*128) * 0.25).astype(np.float32))
  P.win_up("rec_ref", 0, P.down("rec_mine", (48*128*128,), np.float32))
  P.win_up("qkv16", 0, (rng.standard_normal((M, 10240)) * 0.45).astype(np.float16))
  P.win_up("gate16", 0, (rng.standard_normal((M, 6144)) * 0.4).astype(np.float16))
  P.win_up("xin16", 0, (rng.standard_normal((M, 5120)) * 0.7).astype(np.float32))
  dev.synchronize()
  # araw16/braw16 (mine, all rows) == alpharaw/betaraw per row (ref)
  pr["pfk_ab16"](d["xin16"], d["nw1"], d["alpha"], d["beta"], d["xh16"], d["araw16"], d["braw16"],
                 global_size=(16*13,1,1), local_size=LS, wait=True)
  # mine
  pr["pfs16"](d["conv_live"], d["rec_mine"], d["qkv16"], d["gate16"], d["convw"], d["dtb"], d["ssma"],
              d["araw16"], d["braw16"], d["q"], d["k"], d["v"], d["core"], d["snw"], d["z16"],
              global_size=(48,1,1), local_size=LS, wait=True)
  z16 = P.down("z16", (M, 6144), np.float16)
  conv_mine = P.down("conv_live", (3*10240,), np.float32)
  rec_mine = P.down("rec_mine", (48*128*128,), np.float32)
  # reference: 16x T=1 k2s (k0ab per row for alpha/beta)
  zref = np.zeros((M, 6144), dtype=np.float16)
  par = 0
  for t in range(M):
    xr = P.d["xin16"].offset(offset=t*5120*4, size=5120*4)
    pr["k0ab"](xr, d["nw1"], d["alpha"], d["beta"], d["x_t1"], d["alpharaw"], d["betaraw"],
               global_size=(13,1,1), local_size=LS)
    qr = P.d["qkv16"].offset(offset=t*10240*2, size=10240*2)
    gr = P.d["gate16"].offset(offset=t*6144*2, size=6144*2)
    csrc, cdst = (d["convA"], d["convB"]) if par == 0 else (d["convB"], d["convA"])
    pr["k2s"](csrc, cdst, qr, gr, d["convw"], d["dtb"], d["ssma"], d["alpharaw"], d["betaraw"],
              d["q"], d["k"], d["v"], d["rec_ref"], d["core"], d["snw"], d["z_t1"],
              global_size=(48,1,1), local_size=LS, wait=True)
    zref[t] = P.down("z_t1", (6144,), np.float16)
    par ^= 1
  conv_ref = P.down("convB" if par == 1 else "convA", (3*10240,), np.float32)
  rec_ref = P.down("rec_ref", (48*128*128,), np.float32)
  check("pfs16 z16 vs k2s x16", z16, zref, 1e-3)
  check("pfs16 conv final vs k2s", conv_mine, conv_ref, 1e-3)
  check("pfs16 rec final vs k2s", rec_mine, rec_ref, 1e-3)
  timing("pfs16", lambda: pr["pfs16"](d["conv_live"], d["rec_mine"], d["qkv16"], d["gate16"], d["convw"], d["dtb"], d["ssma"], d["araw16"], d["braw16"], d["q"], d["k"], d["v"], d["core"], d["snw"], d["z16"], global_size=(48,1,1), local_size=LS))

print("[test_pf2] done", flush=True)
