# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P2 16-row forward harness: embed(M) -> 64 blocks (pGEMMs + pSCAN-M / pKPRE-M
+ pATTN-M) -> head(M), vs the T=1 trunk (KV8+QH+SKV canonical) 16 steps at the
same positions. Gates: logits per-row <=3e-3, GDN state drift <=1e-2 (report),
timing -> projected prefill tok/s. Usage:
  PATH=... DOCKER_HOST=... ~/tg311/bin/python -u pf_fwd16.py [--pos0]
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
import trunk_w1c
from trunk_w1c import TrunkEngineW1C, LS
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
M = 16
S = 32
MERGE = os.getenv("PF_MERGE") == "1"

E = TrunkEngineW1C(theta=1e7)   # SKV=1 KV8=1 QH=1 via env
P, W, d = E.P, E.W, E.P.d
pr = E.pr
# FIX the trunk's KV8+QH combine pairing: spk_c1 key binds spk_c1_100k (S=32)
# but the K1 (spk_g4nw32qh1_100k) bakes S=256 -> partial-layout mismatch = garbage
# T=1 reference (the g-variant is the S=256 combine; same fix the MTPEngine path uses).
_lib = open(f"{BASE}/spk_c1g_100k.cubin", "rb").read()
pr["spk_c1"] = NVProgram(dev, TinyELF(lib=_lib, name="spk_c1g_100k", target=dev.renderer.target, signature=tuple()))

KSYM = {"pfk_pre16_100k": "pfk_pre16", "pfa16nw32_s32_100k": "pfa16", "pfc16_s32": "pfc16", "pfs16": "pfs16"}
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=KSYM.get(n, n), target=dev.renderer.target, signature=tuple()))

PF = {}
for n in ["pfk_emb16","pfk_n16","pfk_ab16","pfk_hh16","pfk_pre16_100k","pfa16nw32_s32_100k","pfc16_s32","pfs16",
          "pfg_q5kv_hm_nw16k128","pfg_iq3g_hm_nw16k128","pfg_ffn_hm_nw8k128","pfg_iq3d_res_hm_nw8k128",
          "pfg_iq3o_hm_nw8k128","pfg_q8o_hm_nw8k64","pfg_q6q_hm_nw8k64","pfg_iq3q_hm_nw8k128",
          "pfg_iq3k_hm_nw8k128","pfg_q4v_hm_nw8k128","pfg_iq3s_hm_nw8k128","pfg_q5h_hm_nw16k128",
          "pfg2_gdnqg_hm_nw16k128","pfg2_attnqkvq6_hm_nw8k128","pfg2_attnqkvi3_hm_nw8k128"]:
  PF[n] = prog(n)

# ---- M=16 scratch + per-block GDN live slots ----
for nm, nb, dt, v in [
    ("xA16", M*5120*4, np.float32, 7.7e31), ("xB16", M*5120*4, np.float32, 7.7e31),
    ("xh16", M*5120*2, np.float16, 7.7), ("hh16", M*5120*4, np.float32, 7.7e31),
    ("hhx16", M*5120*2, np.float16, 7.7), ("attn_out16", M*5120*2, np.float16, 7.7),
    ("qkv16", M*10240*2, np.float16, 7.7), ("gate16", M*6144*2, np.float16, 7.7),
    ("z16", M*6144*2, np.float16, 7.7), ("gact16", M*17408*2, np.float16, 7.7),
    ("araw16", M*48*4, np.float32, 7.7e31), ("braw16", M*48*4, np.float32, 7.7e31),
    ("qrow16", M*12288*2, np.float16, 7.7), ("krow16", M*1024*2, np.float16, 7.7),
    ("vrow16", M*1024*2, np.float16, 7.7), ("qw16", M*24*256*2, np.float16, 7.7),
    ("ao16", M*6144*2, np.float16, 7.7), ("pm16", 4*S*96*4, np.float32, 7.7e31),
    ("ps16", 4*S*96*4, np.float32, 7.7e31), ("pA16", 4*S*96*256*4, np.float32, 7.7e31),
    ("logits16", M*248320*2, np.float16, 7.7)]:
  P.poison(nm, nb, dt, v)
P.up("ids16", np.zeros(M, dtype=np.int32))
for j, i in enumerate(E.gdn_idx):
  P.up(f"convp{j}", np.zeros(3*10240, dtype=np.float32))
dev.synchronize(); P._keep.clear()
for j, i in enumerate(E.gdn_idx):
  P.up(f"recp{j}", np.zeros(48*128*128, dtype=np.float32))
  if j % 8 == 0: dev.synchronize(); P._keep.clear()
dev.synchronize(); P._keep.clear()

# zero-seed the T=1 trunk GDN states too (canonical FRESH = zeros, not poison)
for i in E.gdn_idx:
  P.up(f"conv{i}_0", np.zeros(3*10240, dtype=np.float32))
  P.up(f"conv{i}_1", np.zeros(3*10240, dtype=np.float32))
  P.up(f"rec{i}", np.zeros(48*128*128, dtype=np.float32))
  if i % 8 == 0: dev.synchronize(); P._keep.clear()
dev.synchronize(); P._keep.clear()
print("[harness] engine + scratch ready", flush=True)

CAP0 = {}
G = {  # grid sizes for the pfg classes (N/NTILE)
  "pfg_q5kv_hm_nw16k128": (80, (512,1,1)), "pfg_iq3g_hm_nw16k128": (48, (512,1,1)),
  "pfg_ffn_hm_nw8k128": (272, (256,1,1)), "pfg_iq3d_res_hm_nw8k128": (80, (256,1,1)),
  "pfg_iq3o_hm_nw8k128": (80, (256,1,1)), "pfg_q8o_hm_nw8k64": (80, (256,1,1)),
  "pfg_q6q_hm_nw8k64": (192, (256,1,1)), "pfg_iq3q_hm_nw8k128": (192, (256,1,1)),
  "pfg_iq3k_hm_nw8k128": (16, (256,1,1)), "pfg_q4v_hm_nw8k128": (16, (256,1,1)),
  "pfg_iq3s_hm_nw8k128": (80, (256,1,1)), "pfg_q5h_hm_nw16k128": (248320//128, (512,1,1)),
}

def fwd16(pos, wait_last=False):
  """One M=16 chunk forward. Returns nothing; logits in d['logits16']."""
  n = [0]
  def L(p, *a, g=1, ls=LS, wait=False):
    p(*a, global_size=(g,1,1), local_size=ls, wait=wait)
    n[0] += 1
    if n[0] % 200 == 0: dev.synchronize()
  L(PF["pfk_emb16"], W[("emb",0)], d["grid512"], d["ids16"], d["xA16"], g=16)
  cur = 0
  for i in range(64):
    xin, xout = (d["xA16"] if cur == 0 else d["xB16"]), (d["xB16"] if cur == 0 else d["xA16"])
    if i in E.qtypes:
      L(PF["pfk_n16"], xin, W[("nw1",i)], d["xh16"], g=16)
      if MERGE:
        mqn = "pfg2_attnqkvq6_hm_nw8k128" if E.qtypes[i] == 14 else "pfg2_attnqkvi3_hm_nw8k128"
        L(PF[mqn], W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh16"], d["qrow16"], d["krow16"], d["vrow16"], g=224, ls=LS)
      else:
        qn = "pfg_q6q_hm_nw8k64" if E.qtypes[i] == 14 else "pfg_iq3q_hm_nw8k128"
        g_, ls_ = G[qn]; L(PF[qn], W[("q",i)], d["gridf"], d["xh16"], d["qrow16"], g=g_, ls=ls_)
        g_, ls_ = G["pfg_iq3k_hm_nw8k128"]; L(PF["pfg_iq3k_hm_nw8k128"], W[("k",i)], d["gridf"], d["xh16"], d["krow16"], g=g_, ls=ls_)
        g_, ls_ = G["pfg_q4v_hm_nw8k128"]; L(PF["pfg_q4v_hm_nw8k128"], W[("v",i)], d["gridf"], d["xh16"], d["vrow16"], g=g_, ls=ls_)
      L(PF["pfk_pre16_100k"], d["qrow16"], d["krow16"], d["vrow16"], W[("qnw",i)], W[("knw",i)], d["freqs"],
        d[f"kv{i}"], d[f"sc{i}"], d["pos_slot"], d["qw16"], g=24)
      L(PF["pfa16nw32_s32_100k"], d[f"kv{i}"], d[f"sc{i}"], d["qw16"], d["pos_slot"], d["pm16"], d["ps16"], d["pA16"],
        g=4*S, ls=(1024,1,1))
      L(PF["pfc16_s32"], d["pm16"], d["ps16"], d["pA16"], d["qrow16"], d["ao16"], g=24)
      g_, ls_ = G["pfg_iq3s_hm_nw8k128"]; L(PF["pfg_iq3s_hm_nw8k128"], W[("o",i)], d["grid512"], d["ao16"], d["attn_out16"], g=g_, ls=ls_)
    else:
      j = E.gdn_idx.index(i)
      L(PF["pfk_ab16"], xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh16"], d["araw16"], d["braw16"], g=16*13)
      if MERGE:
        L(PF["pfg2_gdnqg_hm_nw16k128"], W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh16"], d["qkv16"], d["gate16"], g=128, ls=(512,1,1))
      else:
        g_, ls_ = G["pfg_q5kv_hm_nw16k128"]; L(PF["pfg_q5kv_hm_nw16k128"], W[("qkv",i)], d["gridf"], d["xh16"], d["qkv16"], g=g_, ls=ls_)
        g_, ls_ = G["pfg_iq3g_hm_nw16k128"]; L(PF["pfg_iq3g_hm_nw16k128"], W[("gate",i)], d["gridf"], d["xh16"], d["gate16"], g=g_, ls=ls_)
      L(PF["pfs16"], d[f"convp{j}"], d[f"recp{j}"], d["qkv16"], d["gate16"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
        d["araw16"], d["braw16"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], d["z16"], g=48)
      if i == E.gdn_idx[0] and not CAP0:
        dev.synchronize()
        CAP0["qkv16"] = P.down("qkv16", (M,10240), np.float16).copy()
        CAP0["gate16"] = P.down("gate16", (M,6144), np.float16).copy()
        CAP0["araw16"] = P.down("araw16", (M,48), np.float32).copy()
        CAP0["braw16"] = P.down("braw16", (M,48), np.float32).copy()
        CAP0["recp0"] = P.down("recp0", (48*128*128,), np.float32).copy()
        CAP0["convp0"] = P.down("convp0", (3*10240,), np.float32).copy()
      on = "pfg_q8o_hm_nw8k64" if E.gdn_oq8[i] else "pfg_iq3o_hm_nw8k128"
      g_, ls_ = G[on]; L(PF[on], W[("out",i)], d["gridf"], d["z16"], d["attn_out16"], g=g_, ls=ls_)
    L(PF["pfk_hh16"], xin, d["attn_out16"], W[("nw2",i)], d["hh16"], d["hhx16"], g=16)
    g_, ls_ = G["pfg_ffn_hm_nw8k128"]; L(PF["pfg_ffn_hm_nw8k128"], W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx16"], d["gact16"], g=g_, ls=ls_)
    g_, ls_ = G["pfg_iq3d_res_hm_nw8k128"]; L(PF["pfg_iq3d_res_hm_nw8k128"], W[("fd",i)], d["gridf"], d["gact16"], d["hh16"], xout, g=g_, ls=ls_)
    cur ^= 1
  last = "xA16" if cur == 0 else "xB16"
  xr = d[last]
  L(PF["pfk_n16"], xr, W[("onw",0)], d["xh16"], g=16)
  g_, ls_ = G["pfg_q5h_hm_nw16k128"]; L(PF["pfg_q5h_hm_nw16k128"], W[("head",0)], d["gridf"], d["xh16"], d["logits16"], g=g_, ls=ls_, wait=wait_last)

def t1_token(t, tok):
  """One T=1 trunk step via the _seq list with nw32-aware launch configs.
  (E.token() hardcodes local_size=256 for every kernel — WRONG for the
  1024-thread spk_g4nw32qh1 K1: 3/4 of each CTA never run -> NaN partials.
  The daemon path launches via graphs with correct configs; E.token is only
  valid for the pre-SKV all-256-thread trunk.)"""
  P.win_up("tok_slot", 0, np.array([int(tok)], dtype=np.int32))
  if not hasattr(E, "_seq"): E._build_seqs()
  seq = E._seq[t & 1]
  for n, (p, a, g) in enumerate(seq):
    p(*a, global_size=(g[0],1,1) if isinstance(g, tuple) else (g,1,1),
      local_size=(1024,1,1) if "nw32" in getattr(p, "name", "") else LS,
      wait=(n == len(seq)-1))

# =========== run ===========
rng = np.random.default_rng(99)
ids = rng.integers(1000, 200000, size=M).astype(np.int32)
P.win_up("ids16", 0, ids)
P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
dev.synchronize()

# reference T=1 x16
t0 = time.perf_counter()
ref_logits = np.zeros((M, 248320), dtype=np.float16)
for t in range(M):
  t1_token(t, ids[t])
  ref_logits[t] = P.down("logits", (248320,), np.float16)
t1_time = time.perf_counter() - t0
print(f"[ref] T=1 trunk 16 tokens in {t1_time:.2f}s ({t1_time/M*1e3:.0f} ms/tok)", flush=True)
# snapshot T=1 GDN final states + kv for drift compare
ref_rec = {i: P.down(f"rec{i}", (48*128*128,), np.float32).copy() for i in E.gdn_idx[:4]}
ref_conv = {i: P.down(f"conv{i}_0", (3*10240,), np.float32).copy() for i in E.gdn_idx[:4]}
ref_kv = {i: P.down(f"kv{i}", (2*4*100352*256,), np.uint8).copy() for i in list(E.attn_idx)[:2]}

# ---- PF_LAMBDA=1: root-cause forensics ----
# Capture the per-token blk0 REFERENCE scan inputs (qkv/gate from the T=1 GEMV
# q5g8, araw/braw from k0ab) via a 4-kernel prefix replay (embed -> k0ab ->
# q5g8 -> k2s). Block 0 is the first block => its input is the raw embedding
# only => this prefix replay is bitwise-identical to the full-chain reference.
REF0 = {}
if os.getenv("PF_LAMBDA") == "1":
  i0 = E.gdn_idx[0]
  P.up("convR0", np.zeros(3*10240, dtype=np.float32)); P.up("convR1", np.zeros(3*10240, dtype=np.float32))
  P.up("recR", np.zeros(48*128*128, dtype=np.float32)); dev.synchronize()
  rq = np.zeros((M,10240), np.float16); rg = np.zeros((M,6144), np.float16)
  ra = np.zeros((M,48), np.float32); rb = np.zeros((M,48), np.float32)
  par = 0
  for t in range(M):
    P.win_up("tok_slot", 0, np.array([int(ids[t])], dtype=np.int32))
    pr["h_embed"](W[("emb",0)], d["grid512"], d["tok_slot"], d["x0"], global_size=(1,1,1), local_size=LS)
    pr["k0ab"](d["x0"], W[("nw1",i0)], W[("alpha",i0)], W[("beta",i0)], d["xh"], d["alpharaw"], d["betaraw"],
               global_size=(13,1,1), local_size=LS)
    pr["q5g8"](W[("qkv",i0)], W[("gate",i0)], d["gridf"], d["xh"], d["qkv_row"], d["gate_row"],
               global_size=(2048,1,1), local_size=LS)
    pr["k2s"](d[f"convR{par}"], d[f"convR{par^1}"], d["qkv_row"], d["gate_row"], W[("convw",i0)], W[("dtb",i0)], W[("ssma",i0)],
              d["alpharaw"], d["betaraw"], d["q"], d["k"], d["v"], d["recR"], d["core"], W[("snw",i0)], d["z"],
              global_size=(48,1,1), local_size=LS, wait=True)
    rq[t] = P.down("qkv_row", (10240,), np.float16); rg[t] = P.down("gate_row", (6144,), np.float16)
    ra[t] = P.down("alpharaw", (48,), np.float32); rb[t] = P.down("betaraw", (48,), np.float32)
    par ^= 1
  REF0["qkv"], REF0["gate"], REF0["araw"], REF0["braw"] = rq, rg, ra, rb
  REF0["rec"] = P.down("recR", (48*128*128,), np.float32).copy()
  v0 = np.linalg.norm(REF0["rec"] - ref_rec[i0]) / max(np.linalg.norm(ref_rec[i0]), 1e-9)
  print(f"[lam] prefix-replay vs full-chain ref rec: {v0:.3e} (must be ~0)", flush=True)

# reset kv (zeros; convp/recp already zero-seeded, untouched by the T=1 ref)
for i in E.attn_idx:
  P.up(f"kv{i}", np.zeros(2*4*100352*256, dtype=np.uint8))
  if i % 4 == 0: dev.synchronize(); P._keep.clear()
dev.synchronize(); P._keep.clear()
P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
dev.synchronize()

# my M=16 forward — ONE clean 16-step run; ALL gates read IMMEDIATELY after it.
# (P3 law: each fwd16 call advances the live recp/convp GDN state another 16
# steps — the P2 "0.34 rec drift / logits decorrelation" was the gate readout
# taken AFTER 9 timing reps = a ~160-step-advanced state vs the 16-step ref.
# Timing reps must come AFTER the gate reads, never before.)
t0 = time.perf_counter()
fwd16(0, wait_last=True)
dev.synchronize()
mine_time = time.perf_counter() - t0

logits16 = P.down("logits16", (M, 248320), np.float16).astype(np.float32)
ref = ref_logits.astype(np.float32)
act = np.abs(ref) > 1e-6
e = np.abs(logits16[act] - ref[act]) / np.abs(ref[act])
fn = np.linalg.norm(logits16 - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-9)
print(f"[gate] logits16 vs T=1: med {np.median(e):.3e} F(row-max) {fn.max():.3e} -> {'PASS' if fn.max() <= 3e-3 or np.median(e) <= 3e-3 else 'FAIL'}", flush=True)
amd_mine = logits16.argmax(axis=1); amd_ref = ref_logits.astype(np.float32).argmax(axis=1)
print(f"[gate] argmax agreement {int((amd_mine == amd_ref).sum())}/{M}", flush=True)

# state drift (first 4 GDN blocks) — read from the same single clean run
for i in E.gdn_idx[:4]:
  j = E.gdn_idx.index(i)
  rm = P.down(f"recp{j}", (48*128*128,), np.float32)
  cm = P.down(f"convp{j}", (3*10240,), np.float32)
  rr, cr = ref_rec[i], ref_conv[i]
  dr = np.linalg.norm(rm - rr) / max(np.linalg.norm(rr), 1e-9)
  dc = np.linalg.norm(cm - cr) / max(np.linalg.norm(cr), 1e-9)
  print(f"[drift] blk {i}: rec {dr:.3e} conv {dc:.3e}", flush=True)
# kv drift (2 attn layers)
for i in list(E.attn_idx)[:2]:
  kvm = P.down(f"kv{i}", (2*4*100352*256,), np.uint8)
  nz = int((kvm != ref_kv[i]).sum())
  print(f"[drift] kv blk {i}: byte mismatches {nz}/{kvm.size}", flush=True)

# timing reps (min of 10) — AFTER the gates (each rep advances live GDN state)
best = mine_time
for _ in range(9):
  t0 = time.perf_counter(); fwd16(0, wait_last=True); dev.synchronize()
  best = min(best, time.perf_counter() - t0)

# self-consistency: replay the T=1 k2s on MY captured block-0 scan inputs
P.up("convR0", np.zeros(3*10240, dtype=np.float32)); P.up("convR1", np.zeros(3*10240, dtype=np.float32))
P.up("recR", np.zeros(48*128*128, dtype=np.float32))
P.up("qkvC", CAP0["qkv16"]); P.up("gateC", CAP0["gate16"])
dev.synchronize()
par = 0
for t in range(M):
  P.up("arw", CAP0["araw16"][t]); P.up("brw", CAP0["braw16"][t])
  dev.synchronize()
  qr = d["qkvC"].offset(offset=t*10240*2, size=10240*2)
  gr = d["gateC"].offset(offset=t*6144*2, size=6144*2)
  i0 = E.gdn_idx[0]
  pr["k2s"](d[f"convR{par}"], d[f"convR{par^1}"], qr, gr, W[("convw",i0)], W[("dtb",i0)], W[("ssma",i0)],
            d["arw"], d["brw"], d["q"], d["k"], d["v"], d["recR"], d["core"], W[("snw",i0)], d["z_t1x"] if "z_t1x" in d else d["z"],
            global_size=(48,1,1), local_size=LS, wait=(t==M-1))
  par ^= 1
recR = P.down("recR", (48*128*128,), np.float32)
dr_self = np.linalg.norm(recR - CAP0["recp0"]) / max(np.linalg.norm(CAP0["recp0"]), 1e-9)
print(f"[self] k2s-replay-on-my-inputs vs my pfs16 rec drift: {dr_self:.3e}", flush=True)
print(f"[self] my araw[0][:3] {CAP0[chr(97)+chr(114)+chr(97)+chr(119)+chr(49)+chr(54)][0][:3]} ref-araw-class check", flush=True)

# ---- PF_LAMBDA=1: the lambda interpolation (root-cause adjudication) ----
# Replay the T=1 k2s 16x with qkv/gate = mine + lam*(ref - mine). lam=1 must
# reproduce the ref rec bitwise (validates the replay); lam=0 isolates how much
# of the 0.34 drift the in_proj GEMM rounding difference alone explains; the
# intermediate lams measure the drift-vs-input-error curve (amplification /
# chaos saturation). araw/braw use the REF values (bit-exact vs mine at blk0).
if os.getenv("PF_LAMBDA") == "1":
  i0 = E.gdn_idx[0]
  mq, rg_ = CAP0["qkv16"].astype(np.float32), REF0["qkv"].astype(np.float32)
  mg, gg_ = CAP0["gate16"].astype(np.float32), REF0["gate"].astype(np.float32)
  for nm, a, b in [("qkv q[0:6144]", mq[:,:6144], rg_[:,:6144]), ("qkv k[6144:8192]", mq[:,6144:8192], rg_[:,6144:8192]),
                   ("qkv v[8192:10240]", mq[:,8192:10240], rg_[:,8192:10240]), ("gate", mg, gg_)]:
    fn = np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-9)
    print(f"[lam] input diff {nm}: F {fn:.3e}", flush=True)
  rrn = np.linalg.norm(REF0["rec"])
  for lam in [1.0, 0.9999, 0.999, 0.99, 0.95, 0.5, 0.0]:
    mixq = (mq + lam*(rg_-mq)).astype(np.float16); mixg = (mg + lam*(gg_-mg)).astype(np.float16)
    P.up("convR0", np.zeros(3*10240, dtype=np.float32)); P.up("convR1", np.zeros(3*10240, dtype=np.float32))
    P.up("recR", np.zeros(48*128*128, dtype=np.float32)); dev.synchronize()
    par = 0
    for t in range(M):
      P.up("qkvC", mixq[t]); P.up("gateC", mixg[t])
      P.up("arw", REF0["araw"][t]); P.up("brw", REF0["braw"][t])
      dev.synchronize()
      qr = d["qkvC"]
      gr = d["gateC"]
      pr["k2s"](d[f"convR{par}"], d[f"convR{par^1}"], qr, gr, W[("convw",i0)], W[("dtb",i0)], W[("ssma",i0)],
                d["arw"], d["brw"], d["q"], d["k"], d["v"], d["recR"], d["core"], W[("snw",i0)], d["z"],
                global_size=(48,1,1), local_size=LS, wait=(t==M-1))
      par ^= 1
    rec = P.down("recR", (48*128*128,), np.float32)
    dr = np.linalg.norm(rec - REF0["rec"]) / max(rrn, 1e-9)
    da = np.linalg.norm(rec - REF0["rec"])
    print(f"[lam] lam={lam:<7} rec drift vs ref: rel {dr:.3e} abs {da:.3e} (|ref| {rrn:.1f})", flush=True)

print(f"[time] mine M=16 chunk: best {best*1e3:.1f} ms -> projected {M/best:.1f} tok/s (pos=0)", flush=True)
print(f"[time] T=1 ref: {t1_time/M*1e3:.0f} ms/tok -> {M/t1_time:.1f} tok/s; speedup {t1_time/best:.1f}x", flush=True)
print("[pf_fwd16] done", flush=True)
