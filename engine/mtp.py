# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2-MTP: speculative decoding (K=2) on the engine. Cycle order per graph chain:
  draft_g  (2 steps: eh_proj -> blk.64 -> slice head -> argmax, writes dring)
  probe_g  (T=3 trunk: embed[cur,p1,p2] -> 64 blocks M=3 -> head -> amds)
  accept_g (m calc, emit to tok_hist, pos+=m+1, select rec/conv slot m, h_seed)
All device-resident; state via block-major rec4/conv4 [48][5][...] (slot 4 live).
No scalar kernel args anywhere (empty-sig vals unwritten gotcha, hcq.py:350)."""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from trunk_w1c import TrunkEngineW1C, DR7
from trunk import CTX, VOCAB
from engine0 import dev, parse_gguf, read_raw
from engine0 import Bufs
from gcycle import ParityGraph
from tinygrad.helpers import round_up
from tinygrad.device import TinyELF, BufferSpec
from tinygrad.runtime.ops_nv import NVComputeQueue, NVProgram, nv_wait_timeline
from rung_manifest import assert_rung_wiring, emit_offsets
from tinygrad.uop.ops import UOp
from tinygrad.dtype import dtypes

BASE = "~/tinygrad-metal/engine0"
DPACK = f"{BASE}/draft_pack"
LS = (256, 1, 1)
import os as _os2
TLX_MANIFEST = bool(int(_os2.getenv("TLX_MANIFEST", "1")))   # W4.4 kill-switch
CTXK = int(_os2.getenv("SKV_CTXK", "2304"))
SKV = bool(int(_os2.getenv("SKV", "0")))
SKV_S = int(_os2.getenv("SKV_S", "32"))
KV8 = bool(int(_os2.getenv("KV8", "0")))
QH = bool(int(_os2.getenv("QH", "0")))   # W2F: half2-QK K1 + KPRE qw16 (int8-KV path)
PVH = bool(int(_os2.getenv("PVH", "0")))  # W2F L2: partial-half2 PV in the QH K1
HM = bool(int(_os2.getenv("HM", "0")))    # W2G: HMMA tensor-core K1 dots (a3/probe build only; a1 stays scalar)
SG = bool(int(_os2.getenv("SG", "0")))    # W2F L4: k2s3v grid-split scan + k2z3 norm
SLICE = 40960
RBLK = 48*128*128
CBLK = 3*10240

M3_CUBINS = ["h_embed3","k0n3","k0ab3","q5g8_3","k2s3","op38_3","k3ao3","hh3","ffn8_3","down8_3",
             "aq6k8_3","aq3k8_3","aattn3","ao8_3","head8_3","amx3"]
# W2D L1: bit-identical restructured GEMVs (half2 cores; fat 1024-thr CTAs where
# the 640-CTA grids were warps-in-flight-bound). GEMVV=1 swaps them into the probe.
GEMVV = bool(int(_os2.getenv("GEMVV", "0")))
V2_CUBINS = ["q5g8v_3","ffn8v_3","down8nw32_3","op38nw32_3","k3aonw32_3","ao8nw32_3","aq3k8v_3","head8v_3"]
# W2D L2: K=3 speculative depth (probe T=4). K3=1 selects the M=4 probe trunk
# (v2 half2 cores + fat CTAs built in), 3-step draft, accept m<=3.
K3 = bool(int(_os2.getenv("K3", "0")))
RM = 4 if K3 else 3
M4_CUBINS = ["h_embed4","k0n4","k0ab4","q5g8v4","k2s4","op38nw32_4","k3aonw32_4","ao8nw32_4",
             "hh4","ffn8v4","down8nw32_4","aq3k8v4","aq6k8v4","head8v4","accept4"]
# R4 LOOKUP_K: the deep-K lookup graph set (4 = K=4, probe T=5). Per-cycle
# host-side selection between the K=2 graphs and the DEEP graphs via the emit
# hit flag (readout-order law: the decision reads the PREVIOUS cycle's emit).
# RM=5 scratch serves BOTH sets (K2 kernels touch rows 0..2 only; cycles are
# timeline-serialized so shared scratch is safe). rec4/conv4 stay [48][5]:
# k2s5 writes per-step slots 0..4 (slot 4 = live; read at t=0..2, written at
# t=4 — sequential in-CTA, no hazard). emit record widened to 16 int32:
# {pos_new, m, tok0..4, stop, cyc, hit, rsv}.
LOOKUP_K = int(_os2.getenv("LOOKUP_K", "0"))
DEEP_MODE = _os2.getenv("LOOKUP_DEEP_MODE", "sel")   # R4 diag: sel | off | on
assert not (LOOKUP_K and K3), "LOOKUP_K and K3 are exclusive"
assert LOOKUP_K in (0, 4, 5, 6, 7, 8, 9, 10), "LOOKUP_K: 0 (off), 4/5/6/7/8/9/10 (deep K, R5/R5d/R7a/R8)"
if LOOKUP_K:
  RM = {4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11}[LOOKUP_K]
  assert KV8 and QH and HM and SKV, "LOOKUP_K requires the canonical W2H env (SKV KV8 QH HM)"
M5_CUBINS = ["h_embed5","k0n5","k0ab5","q5g8v5","k2s5","op38nw32_5","k3aonw32_5","ao8nw32_5",
             "hh5","ffn8v5","down8nw32_5","aq3k8v5","aq6k8v5","head8v5","lookup5_nw32","acceptk","accept5k","acceptsel5k"]
# R5 K=5: the M=6/T=6 deep set. k2s6 keeps the [48][5] rec4/conv4 layout with
# live=slot-4 (t=5 state -> rec6x/conv6x per-block scratch; acceptsel6k copies
# on m==5) — ZERO trunk/serve layout surgery. emit widened again for K=5 deep
# cycles: 17-word {pos_new, m, tok0..5, stop, cyc, hit} (16-word for K2/K=4).
M6_CUBINS = ["h_embed6","k0n6","k0ab6","q5g8v6","k2s6","op38nw32_6","k3aonw32_6","ao8nw32_6",
             "hh6","ffn8v6","down8nw32_6","aq3k8v6","aq6k8v6","head8v6","lookup6_nw32","acceptk","accept6k","acceptsel6k"]
# R5 K=6: the M=7/T=7 deep set; 18-word emit {pos_new, m, tok0..6, stop, cyc, hit}.
M7_CUBINS = ["h_embed7","k0n7","k0ab7","q5g8v7","k2s7","op38nw32_7","k3aonw32_7","ao8nw32_7",
             "hh7","ffn8v7","down8nw32_7","aq3k8v7","aq6k8v7","head8v7","lookup7_nw32","acceptk","accept7k","acceptsel7k"]
# R5d K=7: the M=8/T=8 deep set; 19-word emit {pos_new, m, tok0..7, stop, cyc, hit}.
# k2s8 slots 0..4 + rec6x/7x/8x (t=5/6/7) + conv5x/6x/7x/8x; [48][5] layout +
# live=slot-4 PRESERVED — zero trunk/serve surgery.
M8_CUBINS = ["h_embed8","k0n8","k0ab8","q5g8v8","k2s8","op38nw32_8","k3aonw32_8","ao8nw32_8",
             "hh8","ffn8v8","down8nw32_8","aq3k8v8","aq6k8v8","head8v8","lookup8_nw32","acceptk","accept8k","acceptsel8k"]
M9_CUBINS = ["h_embed9","k0n9","k0ab9","q5g8v9","k2s9","op38nw32_9","k3aonw32_9","ao8nw32_9",
             "hh9","ffn8v9r7","down8nw32v9r7","aq3k8v9","aq6k8v9","head8v9","lookup9_nw32","acceptk","accept9k","acceptsel9k"]
# R8 K=9: the M=10/T=10 deep set; 21-word emit {pos_new, m, tok0..9, stop, cyc, hit}.
# k2s10 slots 0..4 + rec6x..rec10x (t=5..9) + conv5x..conv10x; [48][5] layout +
# live=slot-4 PRESERVED — zero trunk/serve surgery.
M10_CUBINS = ["h_embed10","k0n10","k0ab10","q5g8v10","k2s10","op38nw32_10","k3aonw32_10","ao8nw32_10",
              "hh10","ffn8v10r7","down8nw32v10r7","aq3k8v10","aq6k8v10","head8v10","lookup10_nw32","acceptk","accept10k","acceptsel10k"]
# R8 K=10: the M=11/T=11 deep set; 22-word emit {pos_new, m, tok0..10, stop, cyc, hit}.
# k2s11 slots 0..4 + rec6x..rec11x (t=5..10) + conv5x..conv11x; [48][5] layout +
# live=slot-4 PRESERVED — zero trunk/serve surgery.
M11_CUBINS = ["h_embed11","k0n11","k0ab11","q5g8v11","k2s11","op38nw32_11","k3aonw32_11","ao8nw32_11",
              "hh11","ffn8v11r7","down8nw32v11r7","aq3k8v11","aq6k8v11","head8v11","lookup11_nw32","acceptk","accept11k","acceptsel11k"]
DEEP_TRIG = int(_os2.getenv("LOOKUP_TRIG", "1"))   # 1 = prev-cycle hit (R4); 2 = TWO consecutive hits (HyperQwen refinement; lossless filter)
# R3 LOOKUP: in-graph n-gram drafter, appended to the END of the draft graph.
# Overwrites dring0/dring1 when an 8-gram FULL-window suffix match is found in
# tok_hist (LMIN=8: alpha1=1.000 offline on a/c; l in [4,7] carries spurious matches).
# The probe verifies the target -> Tier-1 exactness preserved by construction.
LOOKUP = bool(int(_os2.getenv("LOOKUP", "0")))
LUT_CUBINS = ["lookup_nw32"]
MTpd_CUBINS = ["dnorm2","dfgu","dkv","aattn_d","shead","samx","dposadd","accept","acceptsel",
               "ehproj","dq","doproj","ddown","mfill","stxrec","stxconv"]

class MTPEngine(TrunkEngineW1C):
  def __init__(self, theta=10000.0):
    super().__init__(theta)
    P = self.P
    for n in M3_CUBINS + (V2_CUBINS if GEMVV else []) + (M4_CUBINS if K3 else []) + (M5_CUBINS if LOOKUP_K == 4 else []) + (M6_CUBINS if LOOKUP_K == 5 else []) + (M7_CUBINS if LOOKUP_K == 6 else []) + (M8_CUBINS if LOOKUP_K == 7 else []) + (M9_CUBINS if LOOKUP_K == 8 else []) + (M10_CUBINS if LOOKUP_K == 9 else []) + (M11_CUBINS if LOOKUP_K == 10 else []) + (["k2s3v","k2z3"] if SG else []) + (LUT_CUBINS if (LOOKUP and not LOOKUP_K) else []) + MTpd_CUBINS:
      lib = open(f"{BASE}/{n}.cubin", "rb").read()
      self.pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
    # ---- T=3 probe scratch ----
    for nm, nb, dt, pv in [("xA", RM*5120*4, np.float32, 7.7e31), ("xB", RM*5120*4, np.float32, 7.7e31),
                           ("xh3", RM*5120*2, np.float16, 7.7), ("hh3b", RM*5120*4, np.float32, 7.7e31),
                           ("hhx3", RM*5120*2, np.float16, 7.7), ("qkv3", RM*10240*2, np.float16, 7.7),
                           ("gate3", RM*6144*2, np.float16, 7.7), ("araw3", RM*48*4, np.float32, 7.7e31),
                           ("braw3", RM*48*4, np.float32, 7.7e31), ("z3", RM*6144*2, np.float16, 7.7),
                           ("attn_out3", RM*5120*2, np.float16, 7.7), ("gact3", RM*17408*2, np.float16, 7.7),
                           ("qrow3", RM*12288*2, np.float16, 7.7), ("krow3", RM*1024*2, np.float16, 7.7),
                           ("vrow3", RM*1024*2, np.float16, 7.7), ("ao_row3", RM*6144*2, np.float16, 7.7),
                           ("logits3", RM*VOCAB*2, np.float16, 7.7)]:
      P.poison(nm, nb, dt, pv)
    P.up("amds", np.full(RM, -1, dtype=np.int32))
    # block-major per-step GDN states: [48][5][RBLK]
    P.poison("rec4", 48*5*RBLK*4, np.float32, 7.7e31)
    P.poison("conv4", 48*5*CBLK*4, np.float32, 7.7e31)
    if LOOKUP_K:
      # R4 RACE FIX scratch: k2s5's t=4 conv window lands here (writing slot 4
      # directly races other CTAs' live reads); acceptsel5k copies it on m==4.
      P.poison("conv5x", 48*CBLK*4, np.float32, 7.7e31)
    if LOOKUP_K == 5:
      # R5 K=5: k2s6's t=5 window (conv6x) + t=5 rec state (rec6x) scratch.
      P.poison("conv6x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("rec6x", 48*RBLK*4, np.float32, 7.7e31)
    if LOOKUP_K == 6:
      # R5 K=6: k2s7 scratch — conv5x/6x/7x (t=4/5/6 windows) + rec6x/7x (t=5/6).
      P.poison("conv5x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("conv6x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("conv7x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("rec6x", 48*RBLK*4, np.float32, 7.7e31)
      P.poison("rec7x", 48*RBLK*4, np.float32, 7.7e31)
    if LOOKUP_K == 8:
      # R7a K=8: k2s9 scratch — conv5x..9x (t=4..8 windows) + rec6x..9x (t=5..8).
      P.poison("conv5x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("conv6x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("conv7x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("conv8x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("conv9x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("rec6x", 48*RBLK*4, np.float32, 7.7e31)
      P.poison("rec7x", 48*RBLK*4, np.float32, 7.7e31)
      P.poison("rec8x", 48*RBLK*4, np.float32, 7.7e31)
      P.poison("rec9x", 48*RBLK*4, np.float32, 7.7e31)
    if LOOKUP_K == 9:
      # R8 K=9: k2s10 scratch — conv5x..10x (t=4..9 windows) + rec6x..10x (t=5..9).
      P.poison("conv5x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("conv6x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("conv7x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("conv8x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("conv9x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("conv10x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("rec6x", 48*RBLK*4, np.float32, 7.7e31)
      P.poison("rec7x", 48*RBLK*4, np.float32, 7.7e31)
      P.poison("rec8x", 48*RBLK*4, np.float32, 7.7e31)
      P.poison("rec9x", 48*RBLK*4, np.float32, 7.7e31)
      P.poison("rec10x", 48*RBLK*4, np.float32, 7.7e31)
    if LOOKUP_K == 10:
      # R8 K=10: k2s11 scratch — conv5x..11x (t=4..10 windows) + rec6x..11x (t=5..10).
      for _n in ("conv5x","conv6x","conv7x","conv8x","conv9x","conv10x","conv11x"):
        P.poison(_n, 48*CBLK*4, np.float32, 7.7e31)
      for _n in ("rec6x","rec7x","rec8x","rec9x","rec10x","rec11x"):
        P.poison(_n, 48*RBLK*4, np.float32, 7.7e31)
    if LOOKUP_K == 7:
      # R5d K=7: k2s8 scratch — conv5x/6x/7x/8x (t=4..7 windows) + rec6x/7x/8x (t=5..7).
      P.poison("conv5x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("conv6x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("conv7x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("conv8x", 48*CBLK*4, np.float32, 7.7e31)
      P.poison("rec6x", 48*RBLK*4, np.float32, 7.7e31)
      P.poison("rec7x", 48*RBLK*4, np.float32, 7.7e31)
      P.poison("rec8x", 48*RBLK*4, np.float32, 7.7e31)
    # ---- draft scratch ----
    for nm, nb, dt, pv in [("e_buf", 5120*4, np.float32, 7.7e31), ("cat", 10240*2, np.float16, 7.7),
                           ("xin_d", 5120*4, np.float32, 7.7e31), ("xh_d", 5120*2, np.float16, 7.7),
                           ("qrow_d", 12288*2, np.float16, 7.7), ("krow_d", 1024*2, np.float16, 7.7),
                           ("vrow_d", 1024*2, np.float16, 7.7), ("ao_row_d", 6144*2, np.float16, 7.7),
                           ("attn_out_d", 5120*2, np.float16, 7.7), ("hh_d", 5120*4, np.float32, 7.7e31),
                           ("hhx_d", 5120*2, np.float16, 7.7), ("gact_d", 17408*2, np.float16, 7.7),
                           ("hd_d0", 5120*4, np.float32, 7.7e31), ("hd_d1", 5120*4, np.float32, 7.7e31),
                           ("slogits", SLICE*2, np.float16, 7.7)]:
      P.poison(nm, nb, dt, pv)
    if KV8:
      P.up("kv_d", np.zeros(2*4*CTXK*256, dtype=np.uint8))       # biased int8
      P.poison("sc_d", 2*4*CTXK*8*2, np.float16, 0.0)
    else:
      P.poison("kv_d", 2*4*CTXK*256*2, np.float16, 7.7)
    P.up("h_seed", np.zeros(5120, dtype=np.float32))
    if SKV:
      suf = "2k" if CTXK <= 2304 else "100k"
      skv_k = _os2.getenv("SKV_K", "a")  # "a" = W3 K1S | "g2" = SKV-G (spk_g.cu)
      if K3:
        k1n = f"spk_g4nw32a4_{suf}"
        cn = "spk_c4" if suf == "2k" else "spk_c4g_100k"
        pren = "spk_pre4_" + suf
      elif KV8:
        assert suf == "100k" and not K3, "KV8 int8-KV kernels built for 100k K=2 only"
        if QH:
          k1n = "spk_g4nw32hm3_100k" if HM else ("spk_g4nw32qh3p_100k" if PVH else "spk_g4nw32qh3_100k")
          k1n = _os2.getenv("MTP_K1N_A3", k1n)  # W2H debug: force a3 K1 cubin (smem A/B)
          pren = "spk_pre3qh_100k"
        else:
          k1n, pren = "spk_g4nw32qa3_100k", "spk_pre3q_100k"
        cn = "spk_c3g_100k"
      else:
        k1n = (f"spk_{skv_k}a3_" if skv_k != "a" else "spk_a3_") + suf
        if suf == "2k": cn = "spk_c3"
        elif skv_k == "a": cn = "spk_c3_100k"
        else: cn = f"spk_c3g{0 if SKV_S == 256 else SKV_S}_100k".replace("g0", "g")
        pren = "spk_pre3_" + suf
      for key, n in (("spk_pre3", pren), ("spk_a3", k1n), ("spk_c3", cn)):
        lib = open(f"{BASE}/{n}.cubin", "rb").read()
        self.pr[key] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
      if LOOKUP_K:
        # R4: the DEEP (ROWS=5) attention set under its own keys (canonical QH+HMMA path)
        for key, n in (("spk_pre5", "spk_pre5qh_100k"), ("spk_a5", "spk_g4nw32hm5_100k"), ("spk_c5", "spk_c5g_100k")):
          lib = open(f"{BASE}/{n}.cubin", "rb").read()
          self.pr[key] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
      if LOOKUP_K == 5:
        # R5: the DEEP (ROWS=6) attention set (RMAX=36; hm6 built from the ks4-unroll-8 variant for the 64-reg/1024-thr budget)
        for key, n in (("spk_pre6", "spk_pre6qh_100k"), ("spk_a6", "spk_g4nw32hm6_100k"), ("spk_c6", "spk_c6g_100k")):
          lib = open(f"{BASE}/{n}.cubin", "rb").read()
          self.pr[key] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
      if LOOKUP_K == 6:
        # R5: the DEEP (ROWS=7) attention set (RMAX=42, RP=48 — the MAXOWN=2 owner path)
        for key, n in (("spk_pre7", "spk_pre7qh_100k"), ("spk_a7", "spk_g4nw32hm7_100k"), ("spk_c7", "spk_c7g_100k")):
          lib = open(f"{BASE}/{n}.cubin", "rb").read()
          self.pr[key] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
      if LOOKUP_K == 8:
        # R7a: the DEEP (ROWS=9) attention set (RMAX=54, RP=64 — 10 pad rows, MAXOWN=2)
        for key, n in (("spk_pre9", "spk_pre9qh_100k"), ("spk_a9", "spk_g4nw32hm9_100k"), ("spk_c9", "spk_c9g_100k")):
          lib = open(f"{BASE}/{n}.cubin", "rb").read()
          self.pr[key] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
      if LOOKUP_K == 9:
        # R8: the DEEP (ROWS=10) attention set (RMAX=60, RP=64 — 4 pad rows, MAXOWN=2)
        for key, n in (("spk_pre10", "spk_pre10qh_100k"), ("spk_a10", "spk_g4nw32hm10_100k"), ("spk_c10", "spk_c10g_100k")):
          lib = open(f"{BASE}/{n}.cubin", "rb").read()
          self.pr[key] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
      if LOOKUP_K == 10:
        # R8: the DEEP (ROWS=11) attention set (RMAX=66, RP=80 — 14 pad rows, MAXOWN=3;
        # hm11 carries ONE SASS-audited cold 4B index spill — far below the P18 class)
        for key, n in (("spk_pre11", "spk_pre11qh_100k"), ("spk_a11", "spk_g4nw32hm11_100k"), ("spk_c11", "spk_c11g_100k")):
          lib = open(f"{BASE}/{n}.cubin", "rb").read()
          self.pr[key] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
      if LOOKUP_K == 7:
        # R5d: the DEEP (ROWS=8) attention set (RMAX=48, RP=48 — NO padding rows, MAXOWN=2)
        for key, n in (("spk_pre8", "spk_pre8qh_100k"), ("spk_a8", "spk_g4nw32hm8_100k"), ("spk_c8", "spk_c8g_100k")):
          lib = open(f"{BASE}/{n}.cubin", "rb").read()
          self.pr[key] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
      P.poison("qw3", RM*24*256*4, np.float32, 7.7e31)
      if QH:
        P.poison("qw16_3", RM*24*256*2, np.float16, 7.7)
        P.poison("qw16_1", 24*256*2, np.float16, 7.7)
      P.poison("pm3", 4*SKV_S*6*RM*4, np.float32, 7.7e31)
      P.poison("ps3", 4*SKV_S*6*RM*4, np.float32, 7.7e31)
      P.poison("pA3", 4*SKV_S*6*RM*256*4, np.float32, 7.7e31)
      self._flush()
    for nm in ("cur_slot", "dring0", "dring1", "dring2", "dring3", "dring4", "dring5", "dring6", "dring7", "dring8", "dring9", "m_slot", "cyc_slot", "dpos1", "dpos2", "fillpos", "dtokf"):
      P.up(nm, np.zeros(1, dtype=np.int32))
    P.up("m_hist", np.zeros(1 << 20, dtype=np.int32))   # M1-C: 1024 OOB-d at cyc>=1024 (accept writes m_hist[cyc_slot])
    if LOOKUP or LOOKUP_K: P.up("l_hist", np.zeros(1 << 20, dtype=np.int32))   # R3/R4: per-cycle lookup match len (0=miss; 9=hit)
    P.up("zed5k", np.zeros(5120, dtype=np.float32))
    # M1-A fixed handles: per-cycle emit outbox (16 int32, R4-wide: {pos_new, m,
    # tok0..4, stop, cyc, hit, rsv}; written by accept/acceptk/accept5k, read by
    # DecodeSession.step via down_at — NEVER a tok_hist down), committed-position
    # draft hidden (accept.cu), and the mfill value/count staging slots.
    P.up("emit", np.zeros(26, dtype=np.int32))   # R8: 26-word alloc (22-word K=10 deep layout fits)
    P.up("dhd_seed", np.zeros(5120, dtype=np.float32))
    P.up("fillval", np.zeros(1, dtype=np.int32))
    P.up("filln", np.zeros(1, dtype=np.int32))
    self._flush()

  # ---------- draft weights + slice ----------
  def init_draft(self, slice_ids):
    assert len(slice_ids) == SLICE, len(slice_ids)
    P = self.P
    for nm in ("d_eh","d_q","d_k","d_v","d_o","d_fg","d_fu","d_fd","d_enw","d_hnw","d_shnw","d_nw1","d_nw2","d_qnw","d_knw"):
      P.up(nm, np.load(f"{DPACK}/{nm}.npy"))
      if nm == "d_o": self._flush()
    self._flush()
    ds, infos = parse_gguf()
    head_raw = np.frombuffer(read_raw(infos["output.weight"], ds), dtype=np.uint8).reshape(VOCAB, 3520)
    rows = np.ascontiguousarray(head_raw[np.asarray(slice_ids, dtype=np.int64)])
    del head_raw
    P.up("slice_w", rows.reshape(-1))
    P.up("stab", np.asarray(slice_ids, dtype=np.int32))
    self._flush()

  # ---------- state restore (MTP view; pads KV to 2304) ----------
  def restore_mtp(self, snap, kv=True):
    P = self.P
    rec4 = np.full((48, 5, RBLK), 7.7e31, dtype=np.float32)
    conv4 = np.full((48, 5, CBLK), 7.7e31, dtype=np.float32)
    for j, i in enumerate(self.gdn_idx):
      rec4[j, 4] = snap[f"rec{j}"].astype(np.float32).reshape(RBLK)
      conv4[j, 4] = snap[f"conv{j}"].astype(np.float32).reshape(CBLK)
      if j % 8 == 0: self._flush()
    P.up("rec4", rec4.reshape(-1))
    P.up("conv4", conv4.reshape(-1))
    self._flush()
    if kv:
      for j, i in enumerate(self.attn_idx):
        kvs = snap[f"kv{j}"].astype(np.float16).reshape(2, 4, 2048, 256)
        kvbig = np.full((2, 4, CTXK, 256), 7.7, dtype=np.float16)
        kvbig[:, :, :2048] = kvs
        if KV8:
          gg = kvbig.astype(np.float32).reshape(2, 4, CTXK, 8, 32)
          am = np.abs(gg).max(axis=-1)
          scq = (np.maximum(am, 1e-8) * (1.0/127.0)).astype(np.float16)
          qq = (np.clip(np.rint(gg / scq.astype(np.float32)[..., None]), -127, 127) + 128).astype(np.uint8)
          P.up(f"kv{i}", qq.reshape(-1))
          P.up(f"sc{i}", scq.reshape(-1))
          del gg, am, scq, qq
        else:
          P.up(f"kv{i}", kvbig.reshape(-1))
        if j % 4 == 0: self._flush()
    P.up("cur_slot", np.array([int(snap["ids"].reshape(-1)[-1])], dtype=np.int32))
    P.up("pos_slot", np.array([int(snap["P"].reshape(-1)[0]) - 1], dtype=np.int32))
    P.up("tok_hist", np.full(CTX+128, -1, dtype=np.int32))
    P.up("h_seed", np.zeros(5120, dtype=np.float32))
    P.up("m_hist", np.zeros(1 << 20, dtype=np.int32))   # M1-C: 1024 OOB-d at cyc>=1024 (accept writes m_hist[cyc_slot])
    if LOOKUP or LOOKUP_K: P.up("l_hist", np.zeros(1 << 20, dtype=np.int32))   # R3/R4: per-cycle lookup match len (0=miss)
    P.up("cyc_slot", np.zeros(1, dtype=np.int32))
    P.up("m_slot", np.zeros(1, dtype=np.int32))
    P.up("dring0", np.full(1, -1, dtype=np.int32))
    P.up("dring1", np.full(1, -1, dtype=np.int32))
    P.up("dring2", np.full(1, 0, dtype=np.int32))
    P.up("dring3", np.full(1, 0, dtype=np.int32))
    P.up("dring4", np.full(1, 0, dtype=np.int32))
    P.up("dring5", np.full(1, 0, dtype=np.int32))
    P.up("dring6", np.full(1, 0, dtype=np.int32))
    P.up("dring7", np.full(1, 0, dtype=np.int32))
    P.up("dring8", np.full(1, 0, dtype=np.int32))
    P.up("dring9", np.full(1, 0, dtype=np.int32))
    self._flush()
    self.graphs = None   # buffer handles changed

  # ---------- draft chain step (13 kernels; buffer handles in, launch list out) ----------
  def _draft_entries(self, tokbuf, posbuf, hmbuf, hdbuf, dringbuf, dd=None):
    # R6: dd = per-stream buffer-dict override (batched draft chains); default
    # path identical (dd=None -> self.P.d). Only kv_d/sc_d are persistent state
    # read here; all other d[] refs are intra-step transient (serialized-safe).
    d, pr = (dd if dd is not None else self.P.d), self.pr
    e = []
    e.append((pr["h_embed"], (self.W[("emb",0)], d["grid512"], tokbuf, d["e_buf"]), 1))
    e.append((pr["dnorm2"], (d["e_buf"], hmbuf, d["d_enw"], d["d_hnw"], d["cat"]), 1))
    e.append((pr["ehproj"], (d["d_eh"], d["cat"], d["zed5k"], d["xin_d"]), 640))   # ADDHH build: float out = 0 + acc
    e.append((pr["k0_norm"], (d["xin_d"], d["d_nw1"], d["xh_d"]), 1))
    e.append((pr["dq"], (d["d_q"], d["xh_d"], d["hh_d"], d["qrow_d"]), 1536))
    e.append((pr["dkv"], (d["d_k"], d["d_v"], d["xh_d"], d["krow_d"], d["vrow_d"]), 256))
    if SKV:
      scd = (d["sc_d"],) if KV8 else ()
      e.append((pr["spk_pre1"], (d["qrow_d"], d["krow_d"], d["vrow_d"], d["d_qnw"], d["d_knw"], d["freqs"], d["kv_d"], *scd, posbuf, d["qw1"], *((d["qw16_1"],) if QH else ())), 24))
      e.append((pr["spk_a1"], (d["kv_d"], *scd, *((d["qw16_1"],) if QH else d["qw1"]), posbuf, d["pm1"], d["ps1"], d["pA1"]), 4*SKV_S))
      e.append((pr["spk_c1"], (d["pm1"], d["ps1"], d["pA1"], d["qrow_d"], d["ao_row_d"]), 24))
    else:
      e.append((pr["aattn_d"], (d["qrow_d"], d["krow_d"], d["vrow_d"], d["d_qnw"], d["d_knw"], d["freqs"], d["kv_d"], posbuf, d["ao_row_d"]), 24))
    e.append((pr["doproj"], (d["d_o"], d["ao_row_d"], d["hh_d"], d["attn_out_d"]), 640))
    e.append((pr["k3m_hh"], (d["xin_d"], d["attn_out_d"], d["d_nw2"], d["hh_d"], d["hhx_d"]), 1))
    e.append((pr["dfgu"], (d["d_fg"], d["d_fu"], d["hhx_d"], d["gact_d"]), 2176))
    e.append((pr["ddown"], (d["d_fd"], d["gact_d"], d["hh_d"], hdbuf), 640))
    e.append((pr["k0_norm"], (hdbuf, d["d_shnw"], d["xh_d"]), 1))
    e.append((pr["shead"], (d["slice_w"], d["xh_d"], d["slogits"]), SLICE//8))
    e.append((pr["samx"], (d["slogits"], d["stab"], dringbuf), 1))
    return e

  def fill_draft(self, ids, start_pos=0, seed_hd=None, prog=None):
    """Prompt fill of the draft KV: chain draft hiddens, xin at each position q
    from (ids[q], hm_{q-1}); KV[q] = K/V(xin_q). Eager, sync/16 steps.
    M1-A: FIXED-HANDLE (win_up only — safe after build_graphs); start_pos + seed_hd
    support the FOLLOW-UP delta fill (chain resumes from the committed draft hidden).
    M1-B: prog(done,total) optional progress hook (None = canonical)."""
    d, pr = self.P.d, self.pr
    P = self.P
    P.win_up("fillpos", 0, np.array([int(start_pos)], dtype=np.int32))
    P.win_up("hd_d1", 0, seed_hd if seed_hd is not None else np.zeros(5120, dtype=np.float32))
    dev.synchronize()
    t0 = time.perf_counter()
    for q, tid in enumerate(ids):
      P.win_up("dtokf", 0, np.array([int(tid)], dtype=np.int32))
      for p, a, g in self._draft_entries(P.d["dtokf"], P.d["fillpos"], P.d["hd_d1"], P.d["hd_d1"], P.d["dring0"]):
        _nm = getattr(p, "name", "")
        p(*a, global_size=(g, 1, 1), local_size=((1024,1,1) if "nw32" in _nm else (768,1,1) if "nw24" in _nm else (512,1,1) if "nw16" in _nm else LS))
      pr["dposadd"](P.d["fillpos"], P.d["fillpos"], global_size=(1,1,1), local_size=LS)
      if (q+1) % 2000 == 0: print(f'[fill_draft] {q+1}/{len(ids)} ({time.perf_counter()-t0:.0f}s)', flush=True)
      if prog is not None and (q+1) % 64 == 0: prog(q+1, len(ids))
      if (q+1) % 16 == 0: dev.synchronize()
    dev.synchronize()
    print(f"[fill_draft] {len(ids)} positions in {time.perf_counter()-t0:.1f}s", flush=True)

  # ---------- graphs ----------
  def build_graphs(self):
    d, W, pr = self.P.d, self.W, self.pr
    # TLX W4.4 (V-52): the K-suffix manifest assert — every trunk/stateful cubin,
    # scratch buffer, and the emit layout must match the rung manifest EXACTLY
    # (catches accept/acceptsel/lookup mispairing = the R4/R7 rec4-OOB class).
    # The W4.1 exec tripwires additionally verify each kernel's launch size vs
    # its cubin maxntid DURING this capture (ParityGraph execs happen here).
    if TLX_MANIFEST:
      # TLX W5: pr carries ROLE keys for the spk set (pr["spk_pre11"] -> name
      # spk_pre11qh_100k) — pass the union of dict keys and program NAMES so the
      # manifest matches either form.
      assert_rung_wiring(LOOKUP_K, LOOKUP, set(pr.keys()) | {q.name for q in pr.values()}, RM, set(d.keys()), dr7=DR7)
      _eo = emit_offsets(LOOKUP_K)
      assert self.P.d["emit"].size >= 4 * _eo["words"], \
        f"emit buffer {self.P.d['emit'].size}B < manifest {_eo['words']} words"
    if DR7:
      # r7d.cu ports exist ONLY for the live families: K2-probe GEMVV (ffn8v_3 /
      # down8nw32_3) and the K=7 deep probe (ffn8v8 / down8nw32_8). Anything else
      # would read packed7 bytes through packed-layout kernels = silent garbage.
      assert GEMVV and not K3 and LOOKUP_K in (0, 7, 8, 9, 10), \
        "PF_DR7 requires GEMVV=1, K3 off, LOOKUP_K in (0,7,8,9,10)"
    # probe
    if K3:
      seq = [(pr["h_embed4"], (W[("emb",0)], d["grid512"], d["cur_slot"], d["dring0"], d["dring1"], d["dring2"], d["xA"]), 1)]
    else:
      seq = [(pr["h_embed3"], (W[("emb",0)], d["grid512"], d["cur_slot"], d["dring0"], d["dring1"], d["xA"]), 3)]
    cur = 0
    for i in range(64):
      xin, xout = (d["xA"] if cur == 0 else d["xB"]), (d["xB"] if cur == 0 else d["xA"])
      if i in self.qtypes:
        if K3: qkname = "aq6k8v4" if self.qtypes[i] == 14 else "aq3k8v4"
        elif self.qtypes[i] == 14: qkname = "aq6k8_3"
        else: qkname = "aq3k8v_3" if GEMVV else "aq3k8_3"
        a = [(pr["k0n4" if K3 else "k0n3"], (xin, W[("nw1",i)], d["xh3"]), 3 if not K3 else 1),
             (pr[qkname], (W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh3"], d["qrow3"], d["krow3"], d["vrow3"]), 1792),
             ]
        if SKV:
          sca = (d[f"sc{i}"],) if KV8 else ()
          a += [(pr["spk_pre3"], (d["qrow3"], d["krow3"], d["vrow3"], W[("qnw",i)], W[("knw",i)], d["freqs"], d[f"kv{i}"], *sca, d["pos_slot"], d["qw3"], *((d["qw16_3"],) if QH else ())), 24),
                (pr["spk_a3"], (d[f"kv{i}"], *sca, *((d["qw16_3"],) if QH else d["qw3"]), d["pos_slot"], d["pm3"], d["ps3"], d["pA3"]), 4*SKV_S),
                (pr["spk_c3"], (d["pm3"], d["ps3"], d["pA3"], d["qrow3"], d["ao_row3"]), 24)]
        else:
          a += [(pr["aattn3"], (d["qrow3"], d["krow3"], d["vrow3"], W[("qnw",i)], W[("knw",i)], d["freqs"], d[f"kv{i}"], d["pos_slot"], d["ao_row3"]), 24)]
        a += [
             (pr["ao8nw32_4" if K3 else ("ao8nw32_3" if GEMVV else "ao8_3")], (W[("o",i)], d["grid512"], d["ao_row3"], d["attn_out3"]), 160 if (K3 or GEMVV) else 640),
             (pr["hh4" if K3 else "hh3"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 3 if not K3 else 1),
             (pr["ffn8v4" if K3 else (("ffn8v3r7" if DR7 else "ffn8v_3") if GEMVV else "ffn8_3")], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176),
             (pr["down8nw32_4" if K3 else (("down8nw32v3r7" if DR7 else "down8nw32_3") if GEMVV else "down8_3")], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160 if (K3 or GEMVV) else 640)]
      else:
        conv_b = d["conv4"].offset(offset=(i if False else self.gdn_idx.index(i))*5*CBLK*4, size=5*CBLK*4)
        rec_b = d["rec4"].offset(offset=self.gdn_idx.index(i)*5*RBLK*4, size=5*RBLK*4)
        a = [(pr["k0ab4" if K3 else "k0ab3"], (xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh3"], d["araw3"], d["braw3"]), 39 if not K3 else 13),
             (pr["q5g8v4" if K3 else "q5g8_3"], (W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh3"], d["qkv3"], d["gate3"]), 2048),
             *(((pr["k2s3v"], (conv_b, rec_b, d["qkv3"], d["gate3"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
                            d["araw3"], d["braw3"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], d["z3"]), 384),
                (pr["k2z3"], (d["core"], d["gate3"], W[("snw",i)], d["z3"]), 48)) if SG else
              ((pr["k2s3"], (conv_b, rec_b, d["qkv3"], d["gate3"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
                           d["araw3"], d["braw3"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], d["z3"]), 48),))]
        if self.gdn_oq8[i]:
          a.append((pr["k3aonw32_4" if K3 else ("k3aonw32_3" if GEMVV else "k3ao3")], (W[("out",i)], d["z3"], d["attn_out3"]), 160 if (K3 or GEMVV) else 640))
        else:
          a.append((pr["op38nw32_4" if K3 else ("op38nw32_3" if GEMVV else "op38_3")], (W[("out",i)], d["gridf"], d["z3"], d["attn_out3"]), 160 if (K3 or GEMVV) else 640))
        a.append((pr["hh4" if K3 else "hh3"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 3 if not K3 else 1))
        a.append((pr["ffn8v4" if K3 else (("ffn8v3r7" if DR7 else "ffn8v_3") if GEMVV else "ffn8_3")], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176))
        a.append((pr["down8nw32_4" if K3 else (("down8nw32v3r7" if DR7 else "down8nw32_3") if GEMVV else "down8_3")], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160 if (K3 or GEMVV) else 640))
      seq += a
      cur ^= 1
    seq.append((pr["k0n4" if K3 else "k0n3"], (d["xA"], W[("onw",0)], d["xh3"]), 3 if not K3 else 1))
    seq.append((pr["head8v4" if K3 else ("head8v_3" if GEMVV else "head8_3")], (W[("head",0)], d["xh3"], d["logits3"]), VOCAB//8))
    seq.append((pr["amx3"], (d["logits3"], d["amds"]), RM))
    probe_g = ParityGraph(seq, tag="mtpP")
    # draft: step0 (cur, h_seed @ pos) -> dring0; step1 (dring0, hd0 @ pos+1) -> dring1
    dseq = self._draft_entries(d["cur_slot"], d["pos_slot"], d["h_seed"], d["hd_d0"], d["dring0"])
    dseq += [(pr["dposadd"], (d["pos_slot"], d["dpos1"]), 1)]
    dseq += self._draft_entries(d["dring0"], d["dpos1"], d["hd_d0"], d["hd_d1"], d["dring1"])
    if K3:
      dseq += [(pr["dposadd"], (d["dpos1"], d["dpos2"]), 1)]
      dseq += self._draft_entries(d["dring1"], d["dpos2"], d["hd_d1"], d["hd_d0"], d["dring2"])
    if LOOKUP_K == 4:
      # R4: 4-proposal n-gram overwrite (deep-K scan range i in [0, pos-12]).
      dseq += [(pr["lookup5_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"],
                                     d["dring2"], d["dring3"], d["l_hist"], d["cyc_slot"]), 1)]
    elif LOOKUP_K == 5:
      # R5: 5-proposal overwrite (scan range i in [0, pos-13] — the deep-K law at K=5).
      dseq += [(pr["lookup6_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"],
                                     d["dring2"], d["dring3"], d["dring4"], d["l_hist"], d["cyc_slot"]), 1)]
    elif LOOKUP_K == 6:
      # R5: 6-proposal overwrite (scan range i in [0, pos-14] — the deep-K law at K=6).
      dseq += [(pr["lookup7_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"],
                                     d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["l_hist"], d["cyc_slot"]), 1)]
    elif LOOKUP_K == 7:
      # R5d: 7-proposal overwrite (scan range i in [0, pos-15] — the deep-K law at K=7).
      dseq += [(pr["lookup8_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"],
                                     d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["l_hist"], d["cyc_slot"]), 1)]
    elif LOOKUP_K == 8:
      # R7a: 8-proposal overwrite (scan range i in [0, pos-17] — the deep-K law at K=8).
      lu8 = (pr["lookup9_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"],
                                  d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["dring7"], d["l_hist"], d["cyc_slot"]), 1)
      dseq += [lu8]
    elif LOOKUP_K == 9:
      # R8: 9-proposal overwrite (scan range i in [0, pos-18] — the deep-K law at K=9).
      lu9 = (pr["lookup10_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"],
                                   d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["dring7"],
                                   d["dring8"], d["l_hist"], d["cyc_slot"]), 1)
      dseq += [lu9]
    elif LOOKUP_K == 10:
      # R8: 10-proposal overwrite (scan range i in [0, pos-19] — the deep-K law at K=10).
      lu10 = (pr["lookup11_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"],
                                    d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["dring7"],
                                    d["dring8"], d["dring9"], d["l_hist"], d["cyc_slot"]), 1)
      dseq += [lu10]
    elif LOOKUP:
      # R3: n-gram proposal overwrite AFTER the draft chain wrote dring0/dring1,
      # BEFORE probe_g's h_embed3 reads them (graphs chain in order).
      dseq += [(pr["lookup_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"], d["l_hist"], d["cyc_slot"]), 1)]
    draft_g = ParityGraph(dseq, tag="mtpD")
    if LOOKUP_K == 8:
      # R7a draft-skip: deep cycles (prev emit hit) run LOOKUP ONLY — the draft
      # chain's dring0/1 would be overwritten by the lookup; skipping it is
      # output-lossless (probe-verifies amds; draft is only the K2/miss
      # proposal source, and those cycles keep the full draft graph). On a deep
      # cycle whose lookup MISSES, dring0/1 hold stale-but-valid ids (the
      # deep-K Tier-1-safety law). Saves ~5.3ms x deep-fraction per cycle.
      self.draft_lu_g = ParityGraph([lu8], tag="mtpDL")
    if LOOKUP_K == 9:
      # R8: same draft-skip at K=9 (the lookup-only deep draft graph).
      self.draft_lu_g = ParityGraph([lu9], tag="mtpDL")
    if LOOKUP_K == 10:
      # R8: same draft-skip at K=10.
      self.draft_lu_g = ParityGraph([lu10], tag="mtpDL")
    # accept
    if K3:
      a0 = (pr["accept4"], (d["amds"], d["dring0"], d["dring1"], d["dring2"], d["xA"], d["m_slot"], d["m_hist"],
                            d["cyc_slot"], d["pos_slot"], d["cur_slot"], d["tok_hist"], d["h_seed"]), 1)
    elif LOOKUP_K:
      a0 = (pr["acceptk"], (d["amds"], d["dring0"], d["dring1"], d["xA"], d["m_slot"], d["m_hist"],
                            d["cyc_slot"], d["pos_slot"], d["cur_slot"], d["tok_hist"], d["h_seed"],
                            d["hd_d0"], d["hd_d1"], d["dhd_seed"], d["emit"], d["l_hist"]), 1)
    else:
      a0 = (pr["accept"], (d["amds"], d["dring0"], d["dring1"], d["xA"], d["m_slot"], d["m_hist"],
                           d["cyc_slot"], d["pos_slot"], d["cur_slot"], d["tok_hist"], d["h_seed"],
                           d["hd_d0"], d["hd_d1"], d["dhd_seed"], d["emit"]), 1)
    aseq = [a0,
            (pr["acceptsel"], (d["rec4"], d["conv4"], d["m_slot"]), 48)]
    accept_g = ParityGraph(aseq, tag="mtpA")
    # DEXT QUIRK (W2): a LONE-submitted graph never completes -- every graph needs a
    # follow-up submit in flight before its signal fires (proven: chained g0->g1 waits
    # OK, single-graph waits hang at value-1, any length). 1-kernel flusher after accept.
    fseq = [(pr["dposadd"], (d["fillpos"], d["dpos1"]), 1)]
    flush_g = ParityGraph(fseq, tag="mtpF")
    self.graphs = (draft_g, probe_g, accept_g, flush_g)
    if LOOKUP_K == 4:
      # R4 deep set: probe T=5 (M=5 trunk) + accept5k (m-ladder to 4 + the
      # 16-word emit). draft_g/flush_g SHARED with the K2 set (the draft chain
      # writes dring0/1; lookup5 overwrites all four on hit; a deep-cycle MISS
      # leaves dring2/3 stale-but-valid ids — probe-verifiable, Tier-1-safe).
      p5seq = self._probe5_seq()
      probe5_g = ParityGraph(p5seq, tag="mtpP5")
      aseq5 = [(pr["accept5k"], (d["amds"], d["dring0"], d["dring1"], d["dring2"], d["dring3"], d["xA"],
                                 d["m_slot"], d["m_hist"], d["cyc_slot"], d["pos_slot"], d["cur_slot"],
                                 d["tok_hist"], d["h_seed"], d["hd_d0"], d["hd_d1"], d["dhd_seed"],
                                 d["emit"], d["l_hist"]), 1),
               (pr["acceptsel5k"], (d["rec4"], d["conv4"], d["m_slot"], d["conv5x"]), 48)]
      accept5_g = ParityGraph(aseq5, tag="mtpA5")
      self.graphs5 = (draft_g, probe5_g, accept5_g, flush_g)
      print(f"[mtp] deep graphs built: probe5 {len(p5seq)}k, accept5 {len(aseq5)}k", flush=True)
    if LOOKUP_K >= 5:
      # R5/R5d deep K=5/6/7 set: probe T=RM (M=RM trunk) + acceptNK (m-ladder,
      # 17/18/19-word emit). draft_g/flush_g SHARED with the K2 set (stale dring2..
      # on a deep MISS are valid ids — probe-verifiable, Tier-1-safe).
      p6seq = self._probe6_seq() if LOOKUP_K == 5 else (self._probe7_seq() if LOOKUP_K == 6 else (self._probe9_seq() if LOOKUP_K == 8 else (self._probe10_seq() if LOOKUP_K == 9 else (self._probe11_seq() if LOOKUP_K == 10 else self._probe8_seq()))))
      probe6_g = ParityGraph(p6seq, tag="mtpP6")
      if LOOKUP_K == 5:
        aseqN = [(pr["accept6k"], (d["amds"], d["dring0"], d["dring1"], d["dring2"], d["dring3"], d["dring4"], d["xA"],
                                   d["m_slot"], d["m_hist"], d["cyc_slot"], d["pos_slot"], d["cur_slot"],
                                   d["tok_hist"], d["h_seed"], d["hd_d0"], d["hd_d1"], d["dhd_seed"],
                                   d["emit"], d["l_hist"]), 1),
                 (pr["acceptsel6k"], (d["rec4"], d["conv4"], d["m_slot"], d["conv5x"], d["conv6x"], d["rec6x"]), 48)]
      elif LOOKUP_K == 6:
        aseqN = [(pr["accept7k"], (d["amds"], d["dring0"], d["dring1"], d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["xA"],
                                   d["m_slot"], d["m_hist"], d["cyc_slot"], d["pos_slot"], d["cur_slot"],
                                   d["tok_hist"], d["h_seed"], d["hd_d0"], d["hd_d1"], d["dhd_seed"],
                                   d["emit"], d["l_hist"]), 1),
                 (pr["acceptsel7k"], (d["rec4"], d["conv4"], d["m_slot"], d["conv5x"], d["conv6x"], d["conv7x"], d["rec6x"], d["rec7x"]), 48)]
      elif LOOKUP_K == 8:
        aseqN = [(pr["accept9k"], (d["amds"], d["dring0"], d["dring1"], d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["dring7"], d["xA"],
                                   d["m_slot"], d["m_hist"], d["cyc_slot"], d["pos_slot"], d["cur_slot"],
                                   d["tok_hist"], d["h_seed"], d["hd_d0"], d["hd_d1"], d["dhd_seed"],
                                   d["emit"], d["l_hist"]), 1),
                 (pr["acceptsel9k"], (d["rec4"], d["conv4"], d["m_slot"], d["conv5x"], d["conv6x"], d["conv7x"], d["conv8x"], d["conv9x"], d["rec6x"], d["rec7x"], d["rec8x"], d["rec9x"]), 48)]
      elif LOOKUP_K == 9:
        aseqN = [(pr["accept10k"], (d["amds"], d["dring0"], d["dring1"], d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["dring7"], d["dring8"], d["xA"],
                                    d["m_slot"], d["m_hist"], d["cyc_slot"], d["pos_slot"], d["cur_slot"],
                                    d["tok_hist"], d["h_seed"], d["hd_d0"], d["hd_d1"], d["dhd_seed"],
                                    d["emit"], d["l_hist"]), 1),
                 (pr["acceptsel10k"], (d["rec4"], d["conv4"], d["m_slot"], d["conv5x"], d["conv6x"], d["conv7x"], d["conv8x"], d["conv9x"], d["conv10x"], d["rec6x"], d["rec7x"], d["rec8x"], d["rec9x"], d["rec10x"]), 48)]
      elif LOOKUP_K == 10:
        aseqN = [(pr["accept11k"], (d["amds"], d["dring0"], d["dring1"], d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["dring7"], d["dring8"], d["dring9"], d["xA"],
                                    d["m_slot"], d["m_hist"], d["cyc_slot"], d["pos_slot"], d["cur_slot"],
                                    d["tok_hist"], d["h_seed"], d["hd_d0"], d["hd_d1"], d["dhd_seed"],
                                    d["emit"], d["l_hist"]), 1),
                 (pr["acceptsel11k"], (d["rec4"], d["conv4"], d["m_slot"], d["conv5x"], d["conv6x"], d["conv7x"], d["conv8x"], d["conv9x"], d["conv10x"], d["conv11x"], d["rec6x"], d["rec7x"], d["rec8x"], d["rec9x"], d["rec10x"], d["rec11x"]), 48)]
      else:
        aseqN = [(pr["accept8k"], (d["amds"], d["dring0"], d["dring1"], d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["xA"],
                                   d["m_slot"], d["m_hist"], d["cyc_slot"], d["pos_slot"], d["cur_slot"],
                                   d["tok_hist"], d["h_seed"], d["hd_d0"], d["hd_d1"], d["dhd_seed"],
                                   d["emit"], d["l_hist"]), 1),
                 (pr["acceptsel8k"], (d["rec4"], d["conv4"], d["m_slot"], d["conv5x"], d["conv6x"], d["conv7x"], d["conv8x"], d["rec6x"], d["rec7x"], d["rec8x"]), 48)]
      accept6_g = ParityGraph(aseqN, tag="mtpA6")
      self.graphs6 = (draft_g, probe6_g, accept6_g, flush_g)
      print(f"[mtp] deep graphs built: probe6 {len(p6seq)}k, accept6 {len(aseqN)}k", flush=True)
    print(f"[mtp] graphs built: probe {len(seq)}k, draft {len(dseq)}k, accept {len(aseq)}k, flush {len(fseq)}k", flush=True)

  def _probe5_seq(self):
    """R4: the DEEP (T=5) probe seq — mirrors the K2 probe with the M=5 kernel
    set (RM=5 shared scratch; attention via the ROWS=5 spk keys)."""
    d, W, pr = self.P.d, self.W, self.pr
    seq = [(pr["h_embed5"], (W[("emb",0)], d["grid512"], d["cur_slot"], d["dring0"], d["dring1"],
                             d["dring2"], d["dring3"], d["xA"]), 1)]
    cur = 0
    for i in range(64):
      xin, xout = (d["xA"] if cur == 0 else d["xB"]), (d["xB"] if cur == 0 else d["xA"])
      if i in self.qtypes:
        qkname = "aq6k8v5" if self.qtypes[i] == 14 else "aq3k8v5"
        a = [(pr["k0n5"], (xin, W[("nw1",i)], d["xh3"]), 1),
             (pr[qkname], (W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh3"], d["qrow3"], d["krow3"], d["vrow3"]), 1792)]
        if SKV:
          sca = (d[f"sc{i}"],) if KV8 else ()
          a += [(pr["spk_pre5"], (d["qrow3"], d["krow3"], d["vrow3"], W[("qnw",i)], W[("knw",i)], d["freqs"], d[f"kv{i}"], *sca, d["pos_slot"], d["qw3"], *((d["qw16_3"],) if QH else ())), 24),
                (pr["spk_a5"], (d[f"kv{i}"], *sca, *((d["qw16_3"],) if QH else d["qw3"]), d["pos_slot"], d["pm3"], d["ps3"], d["pA3"]), 4*SKV_S),
                (pr["spk_c5"], (d["pm3"], d["ps3"], d["pA3"], d["qrow3"], d["ao_row3"]), 24)]
        a += [(pr["ao8nw32_5"], (W[("o",i)], d["grid512"], d["ao_row3"], d["attn_out3"]), 160),
              (pr["hh5"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 1),
              (pr["ffn8v5"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176),
              (pr["down8nw32_5"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160)]
      else:
        conv_b = d["conv4"].offset(offset=self.gdn_idx.index(i)*5*CBLK*4, size=5*CBLK*4)
        rec_b = d["rec4"].offset(offset=self.gdn_idx.index(i)*5*RBLK*4, size=5*RBLK*4)
        a = [(pr["k0ab5"], (xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh3"], d["araw3"], d["braw3"]), 13),
             (pr["q5g8v5"], (W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh3"], d["qkv3"], d["gate3"]), 2048),
             (pr["k2s5"], (conv_b, rec_b, d["conv5x"].offset(offset=self.gdn_idx.index(i)*CBLK*4, size=CBLK*4),
                          d["qkv3"], d["gate3"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
                          d["araw3"], d["braw3"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], d["z3"]), 48)]
        if self.gdn_oq8[i]:
          a.append((pr["k3aonw32_5"], (W[("out",i)], d["z3"], d["attn_out3"]), 160))
        else:
          a.append((pr["op38nw32_5"], (W[("out",i)], d["gridf"], d["z3"], d["attn_out3"]), 160))
        a.append((pr["hh5"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 1))
        a.append((pr["ffn8v5"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176))
        a.append((pr["down8nw32_5"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160))
      seq += a
      cur ^= 1
    seq.append((pr["k0n5"], (d["xA"], W[("onw",0)], d["xh3"]), 1))
    seq.append((pr["head8v5"], (W[("head",0)], d["xh3"], d["logits3"]), VOCAB//8))
    seq.append((pr["amx3"], (d["logits3"], d["amds"]), RM))
    return seq

  def _probe6_seq(self):
    """R5: the DEEP K=5 (T=6) probe seq — mirrors the K2 probe with the M=6 kernel
    set (RM=6 shared scratch; attention via the ROWS=6 spk keys; k2s6 t=5 state to
    rec6x/conv6x scratch — [48][5] layout + live=slot-4 unchanged)."""
    d, W, pr = self.P.d, self.W, self.pr
    seq = [(pr["h_embed6"], (W[("emb",0)], d["grid512"], d["cur_slot"], d["dring0"], d["dring1"],
                             d["dring2"], d["dring3"], d["dring4"], d["xA"]), 1)]
    cur = 0
    for i in range(64):
      xin, xout = (d["xA"] if cur == 0 else d["xB"]), (d["xB"] if cur == 0 else d["xA"])
      if i in self.qtypes:
        qkname = "aq6k8v6" if self.qtypes[i] == 14 else "aq3k8v6"
        a = [(pr["k0n6"], (xin, W[("nw1",i)], d["xh3"]), 1),
             (pr[qkname], (W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh3"], d["qrow3"], d["krow3"], d["vrow3"]), 1792)]
        if SKV:
          sca = (d[f"sc{i}"],) if KV8 else ()
          a += [(pr["spk_pre6"], (d["qrow3"], d["krow3"], d["vrow3"], W[("qnw",i)], W[("knw",i)], d["freqs"], d[f"kv{i}"], *sca, d["pos_slot"], d["qw3"], *((d["qw16_3"],) if QH else ())), 24),
                (pr["spk_a6"], (d[f"kv{i}"], *sca, *((d["qw16_3"],) if QH else d["qw3"]), d["pos_slot"], d["pm3"], d["ps3"], d["pA3"]), 4*SKV_S),
                (pr["spk_c6"], (d["pm3"], d["ps3"], d["pA3"], d["qrow3"], d["ao_row3"]), 24)]
        a += [(pr["ao8nw32_6"], (W[("o",i)], d["grid512"], d["ao_row3"], d["attn_out3"]), 160),
              (pr["hh6"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 1),
              (pr["ffn8v6"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176),
              (pr["down8nw32_6"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160)]
      else:
        gi = self.gdn_idx.index(i)
        conv_b = d["conv4"].offset(offset=gi*5*CBLK*4, size=5*CBLK*4)
        rec_b = d["rec4"].offset(offset=gi*5*RBLK*4, size=5*RBLK*4)
        a = [(pr["k0ab6"], (xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh3"], d["araw3"], d["braw3"]), 13),
             (pr["q5g8v6"], (W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh3"], d["qkv3"], d["gate3"]), 2048),
             (pr["k2s6"], (conv_b, rec_b, d["conv5x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                          d["conv6x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                          d["rec6x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                          d["qkv3"], d["gate3"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
                          d["araw3"], d["braw3"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], d["z3"]), 48)]
        if self.gdn_oq8[i]:
          a.append((pr["k3aonw32_6"], (W[("out",i)], d["z3"], d["attn_out3"]), 160))
        else:
          a.append((pr["op38nw32_6"], (W[("out",i)], d["gridf"], d["z3"], d["attn_out3"]), 160))
        a.append((pr["hh6"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 1))
        a.append((pr["ffn8v6"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176))
        a.append((pr["down8nw32_6"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160))
      seq += a
      cur ^= 1
    seq.append((pr["k0n6"], (d["xA"], W[("onw",0)], d["xh3"]), 1))
    seq.append((pr["head8v6"], (W[("head",0)], d["xh3"], d["logits3"]), VOCAB//8))
    seq.append((pr["amx3"], (d["logits3"], d["amds"]), RM))
    return seq

  def _probe7_seq(self):
    """R5: the DEEP K=6 (T=7) probe seq — M=7 kernels; k2s7 t=5/6 state to
    rec6x/rec7x + conv6x/conv7x (layout + live=4 unchanged)."""
    d, W, pr = self.P.d, self.W, self.pr
    seq = [(pr["h_embed7"], (W[("emb",0)], d["grid512"], d["cur_slot"], d["dring0"], d["dring1"],
                             d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["xA"]), 1)]
    cur = 0
    for i in range(64):
      xin, xout = (d["xA"] if cur == 0 else d["xB"]), (d["xB"] if cur == 0 else d["xA"])
      if i in self.qtypes:
        qkname = "aq6k8v7" if self.qtypes[i] == 14 else "aq3k8v7"
        a = [(pr["k0n7"], (xin, W[("nw1",i)], d["xh3"]), 1),
             (pr[qkname], (W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh3"], d["qrow3"], d["krow3"], d["vrow3"]), 1792)]
        if SKV:
          sca = (d[f"sc{i}"],) if KV8 else ()
          a += [(pr["spk_pre7"], (d["qrow3"], d["krow3"], d["vrow3"], W[("qnw",i)], W[("knw",i)], d["freqs"], d[f"kv{i}"], *sca, d["pos_slot"], d["qw3"], *((d["qw16_3"],) if QH else ())), 24),
                (pr["spk_a7"], (d[f"kv{i}"], *sca, *((d["qw16_3"],) if QH else d["qw3"]), d["pos_slot"], d["pm3"], d["ps3"], d["pA3"]), 4*SKV_S),
                (pr["spk_c7"], (d["pm3"], d["ps3"], d["pA3"], d["qrow3"], d["ao_row3"]), 24)]
        a += [(pr["ao8nw32_7"], (W[("o",i)], d["grid512"], d["ao_row3"], d["attn_out3"]), 160),
              (pr["hh7"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 1),
              (pr["ffn8v7"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176),
              (pr["down8nw32_7"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160)]
      else:
        gi = self.gdn_idx.index(i)
        conv_b = d["conv4"].offset(offset=gi*5*CBLK*4, size=5*CBLK*4)
        rec_b = d["rec4"].offset(offset=gi*5*RBLK*4, size=5*RBLK*4)
        a = [(pr["k0ab7"], (xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh3"], d["araw3"], d["braw3"]), 13),
             (pr["q5g8v7"], (W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh3"], d["qkv3"], d["gate3"]), 2048),
             (pr["k2s7"], (conv_b, rec_b, d["conv5x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                          d["conv6x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                          d["conv7x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                          d["rec6x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                          d["rec7x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                          d["qkv3"], d["gate3"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
                          d["araw3"], d["braw3"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], d["z3"]), 48)]
        if self.gdn_oq8[i]:
          a.append((pr["k3aonw32_7"], (W[("out",i)], d["z3"], d["attn_out3"]), 160))
        else:
          a.append((pr["op38nw32_7"], (W[("out",i)], d["gridf"], d["z3"], d["attn_out3"]), 160))
        a.append((pr["hh7"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 1))
        a.append((pr["ffn8v7"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176))
        a.append((pr["down8nw32_7"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160))
      seq += a
      cur ^= 1
    seq.append((pr["k0n7"], (d["xA"], W[("onw",0)], d["xh3"]), 1))
    seq.append((pr["head8v7"], (W[("head",0)], d["xh3"], d["logits3"]), VOCAB//8))
    seq.append((pr["amx3"], (d["logits3"], d["amds"]), RM))
    return seq

  def _probe8_seq(self):
    """R5d: the DEEP K=7 (T=8) probe seq — M=8 kernels; k2s8 t=5/6/7 state to
    rec6x/rec7x/rec8x + conv6x/conv7x/conv8x (layout + live=4 unchanged;
    t=7 rec_in reads rec7x per the REC-CHAIN SLOT LAW)."""
    d, W, pr = self.P.d, self.W, self.pr
    seq = [(pr["h_embed8"], (W[("emb",0)], d["grid512"], d["cur_slot"], d["dring0"], d["dring1"],
                             d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["xA"]), 8)]
    cur = 0
    for i in range(64):
      xin, xout = (d["xA"] if cur == 0 else d["xB"]), (d["xB"] if cur == 0 else d["xA"])
      if i in self.qtypes:
        qkname = "aq6k8v8" if self.qtypes[i] == 14 else "aq3k8v8"
        a = [(pr["k0n8"], (xin, W[("nw1",i)], d["xh3"]), 8),
             (pr[qkname], (W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh3"], d["qrow3"], d["krow3"], d["vrow3"]), 1792)]
        if SKV:
          sca = (d[f"sc{i}"],) if KV8 else ()
          a += [(pr["spk_pre8"], (d["qrow3"], d["krow3"], d["vrow3"], W[("qnw",i)], W[("knw",i)], d["freqs"], d[f"kv{i}"], *sca, d["pos_slot"], d["qw3"], *((d["qw16_3"],) if QH else ())), 24),
                (pr["spk_a8"], (d[f"kv{i}"], *sca, *((d["qw16_3"],) if QH else d["qw3"]), d["pos_slot"], d["pm3"], d["ps3"], d["pA3"]), 4*SKV_S),
                (pr["spk_c8"], (d["pm3"], d["ps3"], d["pA3"], d["qrow3"], d["ao_row3"]), 24)]
        a += [(pr["ao8nw32_8"], (W[("o",i)], d["grid512"], d["ao_row3"], d["attn_out3"]), 160),
              (pr["hh8"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 8),
              (pr["ffn8v8r7" if DR7 else "ffn8v8"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176),
              (pr["down8nw32v8r7" if DR7 else "down8nw32_8"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160)]
      else:
        gi = self.gdn_idx.index(i)
        conv_b = d["conv4"].offset(offset=gi*5*CBLK*4, size=5*CBLK*4)
        rec_b = d["rec4"].offset(offset=gi*5*RBLK*4, size=5*RBLK*4)
        a = [(pr["k0ab8"], (xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh3"], d["araw3"], d["braw3"]), 104),
             (pr["q5g8v8"], (W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh3"], d["qkv3"], d["gate3"]), 2048),
             (pr["k2s8"], (conv_b, rec_b, d["conv5x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                          d["conv6x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                          d["conv7x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                          d["conv8x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                          d["rec6x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                          d["rec7x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                          d["rec8x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                          d["qkv3"], d["gate3"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
                          d["araw3"], d["braw3"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], d["z3"]), 48)]
        if self.gdn_oq8[i]:
          a.append((pr["k3aonw32_8"], (W[("out",i)], d["z3"], d["attn_out3"]), 160))
        else:
          a.append((pr["op38nw32_8"], (W[("out",i)], d["gridf"], d["z3"], d["attn_out3"]), 160))
        a.append((pr["hh8"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 8))
        a.append((pr["ffn8v8r7" if DR7 else "ffn8v8"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176))
        a.append((pr["down8nw32v8r7" if DR7 else "down8nw32_8"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160))
      seq += a
      cur ^= 1
    seq.append((pr["k0n8"], (d["xA"], W[("onw",0)], d["xh3"]), 8))
    seq.append((pr["head8v8"], (W[("head",0)], d["xh3"], d["logits3"]), VOCAB//8))
    seq.append((pr["amx3"], (d["logits3"], d["amds"]), RM))
    return seq

  def _probe9_seq(self):
    """R7a K=8: the DEEP K=8 (T=9) probe seq — M=9 kernels; k2s9 t=5..8 state
    to rec6x..rec9x + conv6x..conv9x (layout + live=4 unchanged; t=8 rec_in
    reads rec8x per the REC-CHAIN SLOT LAW). Norms are the R7a per-row-CTA
    set (k0n9/hh9 grid 9, k0ab9 grid 117 = 13*9, h_embed9 grid 9)."""
    d, W, pr = self.P.d, self.W, self.pr
    seq = [(pr["h_embed9"], (W[("emb",0)], d["grid512"], d["cur_slot"], d["dring0"], d["dring1"],
                             d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["dring7"], d["xA"]), 9)]
    cur = 0
    for i in range(64):
      xin, xout = (d["xA"] if cur == 0 else d["xB"]), (d["xB"] if cur == 0 else d["xA"])
      if i in self.qtypes:
        qkname = "aq6k8v9" if self.qtypes[i] == 14 else "aq3k8v9"
        a = [(pr["k0n9"], (xin, W[("nw1",i)], d["xh3"]), 9),
             (pr[qkname], (W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh3"], d["qrow3"], d["krow3"], d["vrow3"]), 1792)]
        if SKV:
          sca = (d[f"sc{i}"],) if KV8 else ()
          a += [(pr["spk_pre9"], (d["qrow3"], d["krow3"], d["vrow3"], W[("qnw",i)], W[("knw",i)], d["freqs"], d[f"kv{i}"], *sca, d["pos_slot"], d["qw3"], *((d["qw16_3"],) if QH else ())), 24),
                (pr["spk_a9"], (d[f"kv{i}"], *sca, *((d["qw16_3"],) if QH else d["qw3"]), d["pos_slot"], d["pm3"], d["ps3"], d["pA3"]), 4*SKV_S),
                (pr["spk_c9"], (d["pm3"], d["ps3"], d["pA3"], d["qrow3"], d["ao_row3"]), 24)]
        a += [(pr["ao8nw32_9"], (W[("o",i)], d["grid512"], d["ao_row3"], d["attn_out3"]), 160),
              (pr["hh9"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 9),
              (pr["ffn8v9r7" if DR7 else "ffn8v9"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176),
              (pr["down8nw32v9r7" if DR7 else "down8nw32_9"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160)]
      else:
        gi = self.gdn_idx.index(i)
        conv_b = d["conv4"].offset(offset=gi*5*CBLK*4, size=5*CBLK*4)
        rec_b = d["rec4"].offset(offset=gi*5*RBLK*4, size=5*RBLK*4)
        a = [(pr["k0ab9"], (xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh3"], d["araw3"], d["braw3"]), 117),
             (pr["q5g8v9"], (W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh3"], d["qkv3"], d["gate3"]), 2048),
             (pr["k2s9"], (conv_b, rec_b, d["conv5x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                          d["conv6x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                          d["conv7x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                          d["conv8x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                          d["conv9x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                          d["rec6x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                          d["rec7x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                          d["rec8x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                          d["rec9x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                          d["qkv3"], d["gate3"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
                          d["araw3"], d["braw3"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], d["z3"]), 48)]
        if self.gdn_oq8[i]:
          a.append((pr["k3aonw32_9"], (W[("out",i)], d["z3"], d["attn_out3"]), 160))
        else:
          a.append((pr["op38nw32_9"], (W[("out",i)], d["gridf"], d["z3"], d["attn_out3"]), 160))
        a.append((pr["hh9"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 9))
        a.append((pr["ffn8v9r7" if DR7 else "ffn8v9"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176))
        a.append((pr["down8nw32v9r7" if DR7 else "down8nw32_9"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160))
      seq += a
      cur ^= 1
    seq.append((pr["k0n9"], (d["xA"], W[("onw",0)], d["xh3"]), 9))
    seq.append((pr["head8v9"], (W[("head",0)], d["xh3"], d["logits3"]), VOCAB//8))
    seq.append((pr["amx3"], (d["logits3"], d["amds"]), RM))
    return seq


  def _probe10_seq(self):
    """R8 K=9: the DEEP K=9 (T=10) probe seq — M=10 kernels; k2s10 t=5..9 state
    to rec6x..rec10x + conv6x..conv10x (layout + live=4 unchanged; t=9 rec_in
    reads rec9x per the REC-CHAIN SLOT LAW). Norms are the R7a per-row-CTA
    set (k0n10/hh10 grid 10, k0ab10 grid 130 = 13*10, h_embed10 grid 10)."""
    d, W, pr = self.P.d, self.W, self.pr
    seq = [(pr["h_embed10"], (W[("emb",0)], d["grid512"], d["cur_slot"], d["dring0"], d["dring1"],
                              d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["dring7"], d["dring8"], d["xA"]), 10)]
    cur = 0
    for i in range(64):
      xin, xout = (d["xA"] if cur == 0 else d["xB"]), (d["xB"] if cur == 0 else d["xA"])
      if i in self.qtypes:
        qkname = "aq6k8v10" if self.qtypes[i] == 14 else "aq3k8v10"
        a = [(pr["k0n10"], (xin, W[("nw1",i)], d["xh3"]), 10),
             (pr[qkname], (W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh3"], d["qrow3"], d["krow3"], d["vrow3"]), 1792)]
        if SKV:
          sca = (d[f"sc{i}"],) if KV8 else ()
          a += [(pr["spk_pre10"], (d["qrow3"], d["krow3"], d["vrow3"], W[("qnw",i)], W[("knw",i)], d["freqs"], d[f"kv{i}"], *sca, d["pos_slot"], d["qw3"], *((d["qw16_3"],) if QH else ())), 24),
                (pr["spk_a10"], (d[f"kv{i}"], *sca, *((d["qw16_3"],) if QH else d["qw3"]), d["pos_slot"], d["pm3"], d["ps3"], d["pA3"]), 4*SKV_S),
                (pr["spk_c10"], (d["pm3"], d["ps3"], d["pA3"], d["qrow3"], d["ao_row3"]), 24)]
        a += [(pr["ao8nw32_10"], (W[("o",i)], d["grid512"], d["ao_row3"], d["attn_out3"]), 160),
              (pr["hh10"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 10),
              (pr["ffn8v10r7" if DR7 else "ffn8v10"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176),
              (pr["down8nw32v10r7" if DR7 else "down8nw32_10"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160)]
      else:
        gi = self.gdn_idx.index(i)
        conv_b = d["conv4"].offset(offset=gi*5*CBLK*4, size=5*CBLK*4)
        rec_b = d["rec4"].offset(offset=gi*5*RBLK*4, size=5*RBLK*4)
        a = [(pr["k0ab10"], (xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh3"], d["araw3"], d["braw3"]), 130),
             (pr["q5g8v10"], (W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh3"], d["qkv3"], d["gate3"]), 2048),
             (pr["k2s10"], (conv_b, rec_b, d["conv5x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                            d["conv6x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                            d["conv7x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                            d["conv8x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                            d["conv9x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                            d["conv10x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                            d["rec6x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                            d["rec7x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                            d["rec8x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                            d["rec9x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                            d["rec10x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                            d["qkv3"], d["gate3"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
                            d["araw3"], d["braw3"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], d["z3"]), 48)]
        if self.gdn_oq8[i]:
          a.append((pr["k3aonw32_10"], (W[("out",i)], d["z3"], d["attn_out3"]), 160))
        else:
          a.append((pr["op38nw32_10"], (W[("out",i)], d["gridf"], d["z3"], d["attn_out3"]), 160))
        a.append((pr["hh10"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 10))
        a.append((pr["ffn8v10r7" if DR7 else "ffn8v10"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176))
        a.append((pr["down8nw32v10r7" if DR7 else "down8nw32_10"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160))
      seq += a
      cur ^= 1
    seq.append((pr["k0n10"], (d["xA"], W[("onw",0)], d["xh3"]), 10))
    seq.append((pr["head8v10"], (W[("head",0)], d["xh3"], d["logits3"]), VOCAB//8))
    seq.append((pr["amx3"], (d["logits3"], d["amds"]), RM))
    return seq


  def _probe11_seq(self):
    """R8 K=10: the DEEP K=10 (T=11) probe seq — M=11 kernels; k2s11 t=5..10 state
    to rec6x..rec11x + conv6x..conv11x (layout + live=4 unchanged; t=10 rec_in
    reads rec10x per the REC-CHAIN SLOT LAW). Norms per-row-CTA (k0n11/hh11
    grid 11, k0ab11 grid 143 = 13*11, h_embed11 grid 11)."""
    d, W, pr = self.P.d, self.W, self.pr
    seq = [(pr["h_embed11"], (W[("emb",0)], d["grid512"], d["cur_slot"], d["dring0"], d["dring1"],
                              d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["dring7"], d["dring8"], d["dring9"], d["xA"]), 11)]
    cur = 0
    for i in range(64):
      xin, xout = (d["xA"] if cur == 0 else d["xB"]), (d["xB"] if cur == 0 else d["xA"])
      if i in self.qtypes:
        qkname = "aq6k8v11" if self.qtypes[i] == 14 else "aq3k8v11"
        a = [(pr["k0n11"], (xin, W[("nw1",i)], d["xh3"]), 11),
             (pr[qkname], (W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh3"], d["qrow3"], d["krow3"], d["vrow3"]), 1792)]
        if SKV:
          sca = (d[f"sc{i}"],) if KV8 else ()
          a += [(pr["spk_pre11"], (d["qrow3"], d["krow3"], d["vrow3"], W[("qnw",i)], W[("knw",i)], d["freqs"], d[f"kv{i}"], *sca, d["pos_slot"], d["qw3"], *((d["qw16_3"],) if QH else ())), 24),
                (pr["spk_a11"], (d[f"kv{i}"], *sca, *((d["qw16_3"],) if QH else d["qw3"]), d["pos_slot"], d["pm3"], d["ps3"], d["pA3"]), 4*SKV_S),
                (pr["spk_c11"], (d["pm3"], d["ps3"], d["pA3"], d["qrow3"], d["ao_row3"]), 24)]
        a += [(pr["ao8nw32_11"], (W[("o",i)], d["grid512"], d["ao_row3"], d["attn_out3"]), 160),
              (pr["hh11"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 11),
              (pr["ffn8v11r7" if DR7 else "ffn8v11"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176),
              (pr["down8nw32v11r7" if DR7 else "down8nw32_11"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160)]
      else:
        gi = self.gdn_idx.index(i)
        conv_b = d["conv4"].offset(offset=gi*5*CBLK*4, size=5*CBLK*4)
        rec_b = d["rec4"].offset(offset=gi*5*RBLK*4, size=5*RBLK*4)
        a = [(pr["k0ab11"], (xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh3"], d["araw3"], d["braw3"]), 143),
             (pr["q5g8v11"], (W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh3"], d["qkv3"], d["gate3"]), 2048),
             (pr["k2s11"], (conv_b, rec_b, d["conv5x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                            d["conv6x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                            d["conv7x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                            d["conv8x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                            d["conv9x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                            d["conv10x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                            d["conv11x"].offset(offset=gi*CBLK*4, size=CBLK*4),
                            d["rec6x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                            d["rec7x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                            d["rec8x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                            d["rec9x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                            d["rec10x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                            d["rec11x"].offset(offset=gi*RBLK*4, size=RBLK*4),
                            d["qkv3"], d["gate3"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
                            d["araw3"], d["braw3"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], d["z3"]), 48)]
        if self.gdn_oq8[i]:
          a.append((pr["k3aonw32_11"], (W[("out",i)], d["z3"], d["attn_out3"]), 160))
        else:
          a.append((pr["op38nw32_11"], (W[("out",i)], d["gridf"], d["z3"], d["attn_out3"]), 160))
        a.append((pr["hh11"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 11))
        a.append((pr["ffn8v11r7" if DR7 else "ffn8v11"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176))
        a.append((pr["down8nw32v11r7" if DR7 else "down8nw32_11"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 160))
      seq += a
      cur ^= 1
    seq.append((pr["k0n11"], (d["xA"], W[("onw",0)], d["xh3"]), 11))
    seq.append((pr["head8v11"], (W[("head",0)], d["xh3"], d["logits3"]), VOCAB//8))
    seq.append((pr["amx3"], (d["logits3"], d["amds"]), RM))
    return seq

  # ================= M1-A: fixed-handle state machine =================
  # LAWS obeyed here: no P.up past boot (realloc = stale graph kernargs + orphan
  # VRAM); mfill carries value/count in BUFFERS (no scalar kernel args); eager
  # mfill/stxrec/stxconv launches only ever run device-quiescent (post-wait), with
  # dev.synchronize() before returning to graph submits; graphs built ONCE.
  def _mfill(self, name, val, n, off=0):
    """Device memset of n int32 words (val = int bit pattern) via the mfill kernel.
    Eager; caller syncs. off in WORDS."""
    P = self.P
    P.win_up("fillval", 0, np.array([int(val)], dtype=np.int32))
    P.win_up("filln", 0, np.array([int(n)], dtype=np.int32))
    dst = P.d[name].offset(offset=off*4, size=n*4)
    self.pr["mfill"](dst, P.d["fillval"], P.d["filln"], global_size=((n+2047)//2048, 1, 1), local_size=LS)

  def _reset_slots(self, cur0, p0):
    P = self.P
    P.win_up("cur_slot", 0, np.array([int(cur0)], dtype=np.int32))
    P.win_up("pos_slot", 0, np.array([int(p0)], dtype=np.int32))
    P.win_up("tok_slot", 0, np.array([int(cur0)], dtype=np.int32))
    P.win_up("m_slot", 0, np.zeros(1, dtype=np.int32))
    P.win_up("cyc_slot", 0, np.zeros(1, dtype=np.int32))
    # W4.5 (kimi F4): drings zeroed to IN-VOCAB 0 (not -1) — a stale first read
    # feeds the clamped h_embed a valid row instead of a (size_t)(-1)*2200 wild
    # pointer (the garbage-row fault class at conversation starts).
    P.win_up("dring0", 0, np.zeros(1, dtype=np.int32))
    P.win_up("dring1", 0, np.zeros(1, dtype=np.int32))
    P.win_up("dring2", 0, np.full(1, 0, dtype=np.int32))
    P.win_up("dring3", 0, np.full(1, 0, dtype=np.int32))   # R4: 4th proposal slot (0 = valid token id; NEVER poison)
    P.win_up("dring4", 0, np.full(1, 0, dtype=np.int32))   # R5: 5th proposal slot (same law)
    P.win_up("dring5", 0, np.full(1, 0, dtype=np.int32))   # R5: 6th proposal slot (same law)
    P.win_up("dring6", 0, np.full(1, 0, dtype=np.int32))   # R5d: 7th proposal slot (same law)
    P.win_up("h_seed", 0, np.zeros(5120, dtype=np.float32))
    P.win_up("dhd_seed", 0, np.zeros(5120, dtype=np.float32))
    self._mfill("m_hist", 0, 1 << 20)   # M1-C: match the 1Mi-entry alloc
    if LOOKUP or LOOKUP_K: self._mfill("l_hist", 0, 1 << 20)
    self._mfill("tok_hist", -1, CTXK + 256)

  def reset_fresh(self, cur0, sync=True):
    """FRESH: whole engine to conversation start at pos 0. GDN state zeros
    (fresh GDN state IS zeros — NOT poison). kv/kv_d/sc untouched: a FRESH is
    always followed by prefill, which overwrites every row it later reads."""
    P = self.P
    z = np.float32(0.0).view(np.int32).item()
    self._mfill("rec4", z, 48*5*RBLK)
    self._mfill("conv4", z, 48*5*CBLK)
    self._reset_slots(cur0, 0)
    if sync: dev.synchronize()

  def reset_snapshot(self, snapdir, cur0, p0, sync=True):
    """Snapshot-load reset, FIXED-HANDLE (graphs stay valid — no realloc, no
    rebuild): poison rec4/conv4 (7.7e31, the proven tier-1 semantics), window-
    upload slot 4 from the bootstrap-layout npys, reset small slots. kv must have
    been loaded at boot (load_snapshot_kv); decode overwrites rows >= p0."""
    P = self.P
    pv = np.float32(7.7e31).view(np.int32).item()
    self._mfill("rec4", pv, 48*5*RBLK)
    self._mfill("conv4", pv, 48*5*CBLK)
    dev.synchronize()
    for j, i in enumerate(self.gdn_idx):
      P.win_up("rec4", (j*5+4)*RBLK*4, np.load(f"{snapdir}/rc_{i}.npy", mmap_mode="r"))
      P.win_up("conv4", (j*5+4)*CBLK*4, np.load(f"{snapdir}/conv_{i}.npy", mmap_mode="r"))
      if j % 16 == 0: dev.synchronize()
    self._reset_slots(cur0, p0)
    if sync: dev.synchronize()
    self.P._keep.clear()   # snapshot host refs released only after final sync

  def load_snapshot_kv(self, snapdir, kv8=True, progress=False):
    """One-time boot load of the prompt KV (fp16 bootstrap npys -> engine slabs,
    int8-quantized when KV8). FIXED-HANDLE win_up into the boot-time slabs."""
    P = self.P
    for j, i in enumerate(self.attn_idx):
      a = np.load(f"{snapdir}/kv_{i}.npy", mmap_mode="r")
      assert a.shape == (2, 4, CTXK, 256) and a.dtype == np.float16, (i, a.shape)
      if KV8:
        gg = np.asarray(a, dtype=np.float32).reshape(2, 4, CTXK, 8, 32)
        am = np.abs(gg).max(axis=-1)
        scq = (np.maximum(am, 1e-8) * (1.0/127.0)).astype(np.float16)
        qq = (np.clip(np.rint(gg / scq.astype(np.float32)[..., None]), -127, 127) + 128).astype(np.uint8)
        P.win_up(f"kv{i}", 0, qq.reshape(-1))
        P.win_up(f"sc{i}", 0, scq.reshape(-1))
        del gg, am, scq, qq
      else:
        P.win_up(f"kv{i}", 0, np.asarray(a))
      if progress: print(f"[load] kv block {i} ({j+1}/16){' int8' if KV8 else ''}", flush=True)
      self._flush()

  # ---- GDN state xfers between the spec world (rec4/conv4 slot 4) and the
  # T=1 trunk world (rec{i}, conv{i}_par) for prefill/FOLLOW-UP ----
  def stload_trunk(self):
    """spec live GDN state -> trunk rec{i}/conv{i}_0 (before a T=1 prefill)."""
    P, pr = self.P, self.pr
    for j, i in enumerate(self.gdn_idx):
      pr["stxrec"](P.d["rec4"].offset(offset=(j*5+4)*RBLK*4, size=RBLK*4), P.d[f"rec{i}"],
                   global_size=(1,1,1), local_size=LS)
      pr["stxconv"](P.d["conv4"].offset(offset=(j*5+4)*CBLK*4, size=CBLK*4), P.d[f"conv{i}_0"],
                    global_size=(1,1,1), local_size=LS)
      if j % 64 == 63: dev.synchronize()
    dev.synchronize()

  def stseed_spec(self, par):
    """trunk GDN state (rec{i}, conv{i}_par) -> spec rec4/conv4 slot 4.
    par = ndelta & 1: the conv ping-pong parity holding live state after ndelta
    T=1 tokens (token k reads conv_{k&1}, writes conv_{k&1^1})."""
    P, pr = self.P, self.pr
    for j, i in enumerate(self.gdn_idx):
      pr["stxrec"](P.d[f"rec{i}"], P.d["rec4"].offset(offset=(j*5+4)*RBLK*4, size=RBLK*4),
                   global_size=(1,1,1), local_size=LS)
      pr["stxconv"](P.d[f"conv{i}_{par}"], P.d["conv4"].offset(offset=(j*5+4)*CBLK*4, size=CBLK*4),
                    global_size=(1,1,1), local_size=LS)
      if j % 64 == 63: dev.synchronize()
    dev.synchronize()

  def prefill_t1(self, G, ids, log_every=2000, log=None, prog=None):
    """T=1 trunk prefill of ids from the CURRENT pos_slot (device tok_slot fed
    per token; h_argmax auto-advances pos_slot). Exact GCycle run_tokens pattern
    (submit one parity graph + wait — the proven T=1 wait_each path). Leaves
    trunk GDN live in conv{i}_{len(ids)&1} and tok_slot = argmax-after-last."""
    P = self.P
    prev = dev.timeline_value - 1
    t0 = time.perf_counter()
    for k, t in enumerate(ids):
      P.win_up("tok_slot", 0, np.array([int(t)], dtype=np.int32))
      v = dev.next_timeline(); G.graphs[k & 1].submit(prev, v); prev = v
      nv_wait_timeline(dev, v, what="prefill_t1(tok)")
      if log is not None and ((k+1) % 16 == 0 or k == 0 or k+1 == len(ids)): log(k+1, len(ids))
      if prog is not None and ((k+1) % 16 == 0 or k+1 == len(ids)): prog(k+1, len(ids))
      if (k+1) % log_every == 0:
        dev.synchronize(); print(f'[prefill_t1] {k+1}/{len(ids)} ({time.perf_counter()-t0:.1f}s)', flush=True)
    dev.synchronize()
    return time.perf_counter() - t0

  def follow_up(self, G, delta_ids, log=None, prog=None, batch=None):
    """Conversation reuse (FOLLOW-UP): resident spec state + delta user tokens.
    Feeds [cur_slot] + delta_ids via the T=1 trunk from resident pos_slot (incl.
    the delta fill_draft seeded from dhd_seed), then re-seeds spec state.
    Returns (new_cur, pos_new, n_fed)."""
    P = self.P
    pos0 = int(P.down_at("pos_slot", 0, 1)[0])
    cur = int(P.down_at("cur_slot", 0, 1)[0])
    feed = [cur] + [int(t) for t in delta_ids]
    dhd = P.down_at("dhd_seed", 0, 5120, np.float32)   # committed-pos draft hidden
    if batch is None: batch = os.getenv("FU_BATCH", "1") == "1"   # R1 GAP-2: M64 chunk path
    if batch and len(feed) >= 16:
      # R1 GAP-2: trunk delta via the P-series M64 chunk path (~4-8ms/tok vs
      # prefill_t1's ~46ms/tok). ORDER: trunk first, then the standalone
      # fill_draft — the batch path's interleaved dfill also writes kv_d rows,
      # but the standalone (dhd-seeded) values win as the LAST writer, exactly
      # matching the M1-proven T=1 FOLLOW_UP draft KV.
      if log is not None: log("stload_trunk_begin")
      self.stload_trunk()                              # spec live GDN -> trunk buffers
      if log is not None: log("prefill_batch_begin", n=len(feed))
      import pf_prefill
      pf_prefill.prefill_batch(self, G, feed,
                               prog=(lambda k, n: prog(k, n, "prefill_batch")) if prog is not None else None)
      if log is not None: log("fill_draft_begin", pos0=pos0, n=len(feed))
      self.fill_draft(feed, start_pos=pos0, seed_hd=dhd,
                      prog=(lambda d, t: prog(d, t, "fill_draft")) if prog is not None else None)
    else:
      if log is not None: log("fill_draft_begin", pos0=pos0, n=len(feed))
      self.fill_draft(feed, start_pos=pos0, seed_hd=dhd,
                      prog=(lambda d, t: prog(d, t, "fill_draft")) if prog is not None else None)  # draft KV delta (kv_d rows pos0..)
      if log is not None: log("stload_trunk_begin")
      self.stload_trunk()                              # spec live GDN -> trunk buffers
      if log is not None: log("prefill_t1_begin", n=len(feed))
      self.prefill_t1(G, feed,
                      log=(lambda k, n: log("prefill_t1_tok", k=k, n=n)) if log is not None else None,
                      prog=(lambda k, n: prog(k, n, "prefill_t1")) if prog is not None else None)  # trunk KV rows pos0.. + GDN advance
    nd = len(feed)
    if log is not None: log("stseed_spec_begin", par=nd & 1)
    self.stseed_spec(nd & 1)                           # trunk live GDN -> rec4/conv4 slot 4
    if log is not None: log("stseed_spec_done")
    newcur = int(P.down_at("tok_slot", 0, 1)[0])       # argmax after the last fed token
    posn = int(P.down_at("pos_slot", 0, 1)[0])
    P.win_up("cur_slot", 0, np.array([newcur], dtype=np.int32))
    P.win_up("h_seed", 0, np.zeros(5120, dtype=np.float32))
    # W4.5 (kimi F4): drings zeroed to IN-VOCAB 0 (not -1) — a stale first read
    # feeds the clamped h_embed a valid row instead of a (size_t)(-1)*2200 wild
    # pointer (the garbage-row fault class at conversation starts).
    P.win_up("dring0", 0, np.zeros(1, dtype=np.int32))
    P.win_up("dring1", 0, np.zeros(1, dtype=np.int32))
    dev.synchronize()
    return newcur, posn, nd

  # ---------- cycle loop ----------
  def run_cycles(self, ncyc, sync_every=1, phase_times=False):
    assert self.graphs, "build_graphs first"
    if not phase_times:
      s = DecodeSession(self)
      for _ in range(ncyc): s.step()
      return None
    draft_g, probe_g, accept_g, flush_g = self.graphs
    prev = dev.timeline_value - 1
    t_d = t_p = t_a = 0.0
    import time as _t
    for c in range(ncyc):
      t0 = _t.perf_counter()
      vd = dev.next_timeline(); draft_g.submit(prev, vd); prev = vd
      vp = dev.next_timeline(); probe_g.submit(vd, vp); prev = vp
      if phase_times: dev.timeline_signal.wait(vd)   # probe still in flight (>=2 submits rule)
      t1 = _t.perf_counter(); t_d += t1 - t0
      va = dev.next_timeline(); accept_g.submit(vp, va); prev = va
      if phase_times: dev.timeline_signal.wait(vp)   # accept in flight
      t2 = _t.perf_counter(); t_p += t2 - t1
      vf = dev.next_timeline(); flush_g.submit(va, vf); prev = vf
      if phase_times: dev.timeline_signal.wait(va)   # flusher in flight
      t3 = _t.perf_counter(); t_a += t3 - t2
      if phase_times: nv_wait_timeline(dev, vf, what="run_cycles(phase)")
    nv_wait_timeline(dev, prev, what="run_cycles(trailing)")
    dev.synchronize()
    if phase_times: return t_d, t_p, t_a


class DecodeSession:
  """M1-A serving core: one speculative cycle per step(). Submits the 4 graphs
  timeline-chained on a carried `prev` (draft -> probe -> accept -> flush; the
  flusher obeys the LONE-GRAPH law for accept), waits the cycle signal, then
  reads the 32B emit record (down_at) — never the 400KB tok_hist down.
  begin() re-anchors prev after any eager work (prefill/fill/reset).
  R4 (LOOKUP_K=4): per-cycle graph-set selection — after each cycle the emit's
  hit flag (emit[9] = l_hist[cyc]; 9 = 8-gram hit) picks the NEXT cycle's set
  (deep on hit, K2 on miss). Readout-order law: the decision reads the
  PREVIOUS cycle's completed emit; zero extra syncs."""
  def __init__(self, E):
    self.E = E
    self.prev = None
    self.deep = 0
    self.hitrun = 0
    self.ndeep = 0
    self.ncyc = 0

  def begin(self):
    assert self.E.graphs, "build_graphs first"
    self.prev = dev.timeline_value - 1
    self.deep = 0
    self.hitrun = 0

  def step(self):
    E = self.E
    if self.prev is None: self.begin()
    if LOOKUP_K >= 5: g = E.graphs6 if self.deep else E.graphs
    elif LOOKUP_K: g = E.graphs5 if self.deep else E.graphs
    else: g = E.graphs
    ran_deep5 = (LOOKUP_K >= 5 and self.deep)   # the set that ran THIS cycle (emit layout)
    draft_g, probe_g, accept_g, flush_g = g
    if LOOKUP_K >= 8 and self.deep and getattr(E, "draft_lu_g", None) is not None:
      draft_g = E.draft_lu_g   # R7a/R8 draft-skip on deep cycles
    prev = self.prev
    vd = dev.next_timeline(); draft_g.submit(prev, vd)
    vp = dev.next_timeline(); probe_g.submit(vd, vp)
    va = dev.next_timeline(); accept_g.submit(vp, va)
    vf = dev.next_timeline(); flush_g.submit(va, vf)
    self.prev = vf
    nv_wait_timeline(dev, vf, what="DecodeSession.step")   # device quiescent at return (cancel-safe point)
    e = E.P.down_at("emit", 0, 26, np.int32)
    m = int(e[1])
    if ran_deep5 and LOOKUP_K == 10:  # R8: accept11k 22-word {pos,m,tok0..10,stop,cyc,hit}
      stop, cyc, hit = int(e[13]), int(e[14]), int(e[15])
    elif ran_deep5 and LOOKUP_K == 9:  # R8: accept10k 21-word {pos,m,tok0..9,stop,cyc,hit}
      stop, cyc, hit = int(e[12]), int(e[13]), int(e[14])
    elif ran_deep5 and LOOKUP_K == 8:  # R7a: accept9k 20-word {pos,m,tok0..8,stop,cyc,hit}
      stop, cyc, hit = int(e[11]), int(e[12]), int(e[13])
    elif ran_deep5 and LOOKUP_K == 7:  # R5d: accept8k 19-word {pos,m,tok0..7,stop,cyc,hit}
      stop, cyc, hit = int(e[10]), int(e[11]), int(e[12])
    elif ran_deep5 and LOOKUP_K == 6:  # R5: accept7k 18-word {pos,m,tok0..6,stop,cyc,hit}
      stop, cyc, hit = int(e[9]), int(e[10]), int(e[11])
    elif ran_deep5:                   # R5: accept6k 17-word {pos,m,tok0..5,stop,cyc,hit}
      stop, cyc, hit = int(e[8]), int(e[9]), int(e[10])
    elif LOOKUP_K: # acceptk/accept5k 16-word layout
      stop, cyc, hit = int(e[7]), int(e[8]), int(e[9])
    else:          # legacy accept 8-word layout (canonical, unchanged)
      stop, cyc, hit = int(e[5]), int(e[6]), 0
    if LOOKUP_K:
      self.ndeep += self.deep; self.ncyc += 1
      if DEEP_MODE == "on": self.deep = 1
      elif DEEP_MODE == "off": self.deep = 0
      else:   # R5: LOOKUP_TRIG (HyperQwen refinement) — deep needs N consecutive hits
        self.hitrun = self.hitrun + 1 if hit >= 9 else 0
        self.deep = 1 if self.hitrun >= DEEP_TRIG else 0
    return {"pos_new": int(e[0]), "m": m, "cycle": cyc, "stop": stop, "hit": hit,
            "tokens": [int(t) for t in e[2:3+m]][:m+1]}
