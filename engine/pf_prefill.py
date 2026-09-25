# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P3: batched M=16 prefill assembly (pf_prefill.py).

Chunked prefill over arbitrary prompt lengths on the P2 kernel set
(pfk_emb16/n16/ab16/hh16 + pKPRE-M + pATTN-M + pSCAN-M + P1 pGEMMs), writing
the SAME state world the T=1 trunk prefill leaves behind:
  - trunk KV8 rows [pos0, pos0+N) appended (kv{i}/sc{i}) via pfk_pre16
  - trunk GDN live state in conv{i}_0 (full chunks) then the T=1 ping-pong for
    the tail -> overall live parity = N & 1 (SAME formula serve.py already uses
    for stseed_spec(len(toks) & 1))
  - pos_slot = pos0 + N, tok_slot = argmax-after-last (h_argmax on the last
    row's head8 logits), tok_hist last row written by h_argmax
  - tok_hist rows [pos0, pos0+N-1): the T=1 path writes argmax-after-p per row
    (never read downstream — decode/accept only touch rows >= pos); we fill the
    fed ids as a benign placeholder (documented divergence, zero consumers).

TAIL POLICY (v1, documented): full 16-row chunks batched; the r = N % 16
remainder rows run through the existing T=1 graph path (prefill_t1) — exact by
construction, cheap because tails are short.

Laws kept: per-kernel cubins + warp-token names; cached launch plan (fixed
handles, built once); args reference handles fetched AFTER the last up;
launch-pipeline ceiling (sync every 200 launches, chunk-end sync); no
grid-stride; single dev; eager launches only outside graphs.
"""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from engine0 import dev
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
LS = (256, 1, 1)
MERGE = os.getenv("PF_MERGE") == "1"
DFILL = os.getenv("PF_DFILL", "1") == "1"
M32 = os.getenv("PF_M32", "1") == "1"   # P6: 32-row chunks (GEMMs M32, rest 2x16)
G3M = os.getenv("PF_G3M", os.getenv("PF_GEMM3", "1")) == "1"  # P7F3: r7 m32 DBUF GEMM tier on the M32 trunk
NT32 = os.getenv("PF_NT32", "1") == "1"  # P8: ffn NTILE=32 grid-growth twin (160/544-CTA grids; CTASM unlock)
ATTN32 = os.getenv("PF_ATTN32", "0") == "1"  # P10: pfa32c-t32 co-resident attention (S=13, Tier-2; CFG=100 via AUTO_NAMES=pfa32c)
A4 = os.getenv("PF_A4", "0") == "1"          # P10-A4: fused self-resetting last-CTA combine (needs ATTN32)
ATTNW = os.getenv("PF_ATTNW", "0") == "1"       # P17: wide-M attention (w64h<8k / w64>=8k; needs M64)
S13 = 13
assert not A4 or ATTN32, "PF_A4=1 requires PF_ATTN32=1"
# P11: S13/S26 dual-graph hybrid. S26 dominates standalone at every pos but its
# ~300 empty CTAs x identity partial writes lose in-plan at low pos (P10 8k law);
# threshold picks S26 only once splits stay full. Both plans + graph sets resident.
HYB = os.getenv("PF_ATTN_HYB", "1") == "1" and ATTN32 and not A4
ATTN_THR = int(os.getenv("PF_ATTN_THR", "8192"))    # pos >= THR -> S26 graph (P11 sweep: S26 wins from ~9.7k, loses +35ms at pos 0; 8192 keeps every <=8k prompt on the P10-verified S13 arm)
# P12 diet: merge the 2x16-half seam launches on the M32 trunk. The norm/emb
# kernels are row-per-CTA (blockIdx.x = row, no hardcoded M) -> a single g=32
# launch writes identical bytes (bit-identical). pfk_pre32/pfs32 are TROWS=32
# builds of the same sources (per-row/per-step math verbatim).
N32 = os.getenv("PF_N32", "0") == "1"          # emb + n16/ab16/hh16 one-launch M32
PRE32 = os.getenv("PF_PRE32", "0") == "1"      # one pfk_pre32 launch per attn block (was 2x pre16)
SCAN32 = os.getenv("PF_SCAN32", "0") == "1"    # one pfs32 launch per GDN block (was 2x pfs16)
assert not (N32 or PRE32 or SCAN32) or (M32 and not A4), "P12 twins require PF_M32=1 (A4 off)"
# P14: persistent-CTA FFN (pf13ffn grid 82, NBLK=1 drop-in; bit-identity by
# construction — decode/mma verbatim from the r7 m32 body; the P13 stream law)
PERSIST = os.getenv("PF_PERSIST", "0") == "1"
PERSIST_NAME = os.getenv("PF_PERSIST_NAME", "pf13ffn_w8_n1")
PERSIST_MB = float(os.getenv("PF_PERSIST_MB", "6000"))
assert not PERSIST or M32, "PF_PERSIST requires PF_M32=1 (the 32-row chunk shape)"

# ---- P15: THE M=64 TRUNK (PF_M64=1). 64-row chunks, bit-identical by
# construction at every stage (the numerics-free M-scaling rung): the 4 clean
# m64 GEMM twins (P7E4/P7E7 in-plan bit-exact) on W7-covered classes; classic
# m32 x2 on 32-row views elsewhere (the attnqkv m64 twin stays QUARANTINED --
# P7E4 in-plan corrupt); attention = 4x t32 16-row windows (grid-doubling is
# ILLEGAL: pfa32c decodes g = KV-group with ROWS=16 hardcoded); scan = pfs64
# (TROWS=64 twin of pfs32, per-row op order verbatim) or pfs32 x2
# (PF_M64_SCAN32=1); norms/emb row-per-CTA g=64; pre = pfk_pre64 (one flat
# 64-row append; PF_M64_PRE32=1 -> pre32 x2 fallback). Tails r = N % 64
# delegate to the M32 path (bit-identical composition). ----
M64 = os.getenv("PF_M64", "0") == "1"
M64_SCAN32 = os.getenv("PF_M64_SCAN32", "0") == "1"
ABW = os.getenv("PF_ABW", "0") == "1"   # R4: pfk_ab16w fat-CTA reshape (bit-exact)
M64_PRE32 = os.getenv("PF_M64_PRE32", "0") == "1"
assert not M64 or (M32 and ATTN32 and not A4), "PF_M64=1 requires PF_M32=1 PF_ATTN32=1 (A4 off)"
assert not ATTNW or M64, "PF_ATTNW=1 requires PF_M64=1"
_M64ON = M64

def m64_set(on):
  global _M64ON
  _M64ON = bool(on)   # runtime toggle for same-process A/B (graphs re-key on it)

M64_CUBINS = ["pfs64", "pfk_pre64_100k",
              "pfg3_ffn_r7_m64_nw4k128", "pfg3_iq3d_r7_m64_nw8k128",
              "pfg3_iq3o_r7_m64_nw8k128", "pfg3m_gdnqg_r7_m64_nw8k128"]

CH = 32 if M32 else 16
S = 32            # pfa16 split count (s32 cubins)
VOCAB = 248320

KSYM = {"pfk_pre16_100k": "pfk_pre16", "pfa16nw32_s32_100k": "pfa16", "pfc16_s32": "pfc16", "pfs16": "pfs16",
        "pfk_pre32_100k": "pfk_pre32"}   # P12 LAW: entry symbol != cubin filename
PF_CUBINS = ["pfk_emb16", "pfk_n16", "pfk_ab16", "pfk_hh16", "pfk_pre16_100k",
             "pfa16nw32_s32_100k", "pfc16_s32", "pfs16",
             "pfg_q5kv_hm_nw16k128", "pfg_iq3g_hm_nw16k128", "pfg_ffn_hm_nw8k128",
             "pfg_iq3d_res_hm_nw8k128", "pfg_iq3o_hm_nw8k128", "pfg_q8o_hm_nw8k64",
             "pfg_q6q_hm_nw8k64", "pfg_iq3q_hm_nw8k128", "pfg_iq3k_hm_nw8k128",
             "pfg_q4v_hm_nw8k128", "pfg_iq3s_hm_nw8k128",
             "pfg2_gdnqg_hm_nw16k128", "pfg2_attnqkvq6_hm_nw8k128", "pfg2_attnqkvi3_hm_nw8k128",
             "pfk_rec16", "pfd_dnorm16", "pfg_ehd_res_hm_nw8k128", "pfg2_dqkv_hm_nw8k128",
             # P6 M32 family (loaded always; only used when PF_M32=1)
             "pfg_ffn_m32_hm_nw8k128", "pfg_iq3d_m32_res_hm_nw8k128", "pfg_iq3s_m32_hm_nw8k128",
             "pfg_iq3o_m32_hm_nw8k128", "pfg_q8o_m32_hm_nw8k64",
             "pfg2_gdnqg_m32_hm_nw16k128", "pfg2_attnqkvq6_m32_hm_nw8k128", "pfg2_attnqkvi3_m32_hm_nw8k128",
             # P12 M32-merge twins
             "pfk_pre32_100k", "pfs32",
             # P7F3 G3M tier (P7B-validated bit-identical r7 m32 cubins)
             "pfg3_ffn_r7_m32_nw8k128", "pfg3_iq3d_r7_m32_nw8k128", "pfg3_iq3o_r7_m32_nw8k128",
             "pfg3m_gdnqg_r7_m32_nw16k128", "pfg3m_attnqkvq6_r7_m32_nw8k128", "pfg3m_attnqkvi3_r7_m32_nw8k128"]

GRIDS = {  # N/NTILE per pfg class (P1 shape law)
  "pfg_q5kv_hm_nw16k128": (80, (512, 1, 1)), "pfg_iq3g_hm_nw16k128": (48, (512, 1, 1)),
  "pfg_ffn_hm_nw8k128": (272, (256, 1, 1)), "pfg_iq3d_res_hm_nw8k128": (80, (256, 1, 1)),
  "pfg_iq3o_hm_nw8k128": (80, (256, 1, 1)), "pfg_q8o_hm_nw8k64": (80, (256, 1, 1)),
  "pfg_q6q_hm_nw8k64": (192, (256, 1, 1)), "pfg_iq3q_hm_nw8k128": (192, (256, 1, 1)),
  "pfg_iq3k_hm_nw8k128": (16, (256, 1, 1)), "pfg_q4v_hm_nw8k128": (16, (256, 1, 1)),
  "pfg_iq3s_hm_nw8k128": (80, (256, 1, 1)),
}

def ensure(E):
  """One-time attach: cubins + M=16 scratch (FIXED handles, before any
  build_graphs concerns — prefill is eager and graphs never reference these)."""
  if getattr(E, "_pf_plan", None) is not None:
    return
  if getattr(E, "_r7native", None) and not G3M:
    raise RuntimeError("PF_DR7 requires the G3M tier (PF_G3M=1): the packed originals are not resident")
  assert os.getenv("SKV") == "1" and os.getenv("KV8") == "1" and os.getenv("QH") == "1", \
      "prefill_batch requires SKV=1 KV8=1 QH=1 (the 100k canonical env)"
  assert int(os.getenv("SKV_CTXK", "0")) == 100352, "pfk_pre16/pfa16 built for CTXK=100352"
  P, d, W, pr = E.P, E.P.d, E.W, E.pr
  for n in PF_CUBINS:
    lib = open(f"{BASE}/{n}.cubin", "rb").read()
    pr[n] = NVProgram(dev, TinyELF(lib=lib, name=KSYM.get(n, n), target=dev.renderer.target, signature=tuple()))
  if PERSIST:
    for pn in ["pf13ffn_w2_n1", "pf13ffn_w4_n1", "pf13ffn_w8_n1"]:   # all ring depths (bench access)
      lib = open(f"{BASE}/{pn}.cubin", "rb").read()
      pr[pn] = NVProgram(dev, TinyELF(lib=lib, name=pn, target=dev.renderer.target, signature=tuple()))
    print(f"[pf14] persistent FFN cubins live (grid 82), plan kernel = {PERSIST_NAME}", flush=True)
  if M32 and NT32:  # P8: NTILE=32 grid-growth twins (BIT-IDENTICAL, test_p8-gated)
    for n in ["pfg_ffn_m32_nt32_hm_nw4k128"]:
      lib = open(f"{BASE}/{n}.cubin", "rb").read()
      pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
  M = 16
  if not M32:
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
      ("ps16", 4*S*96*4, np.float32, 7.7e31), ("pA16", 4*S*96*256*4, np.float32, 7.7e31)]:
      P.poison(nm, nb, dt, v)
    P.up("ids16", np.zeros(M, dtype=np.int32))
  else:
    # ---- P6 M32 trunk scratch (32 rows) + fixed 16-row half views ----
    H = 16
    for nm, nb, dt, v in [
      ("xA32", 32*5120*4, np.float32, 7.7e31), ("xB32", 32*5120*4, np.float32, 7.7e31),
      ("xh32", 32*5120*2, np.float16, 7.7), ("hh32", 32*5120*4, np.float32, 7.7e31),
      ("hhx32", 32*5120*2, np.float16, 7.7), ("attn_out32", 32*5120*2, np.float16, 7.7),
      ("qkv32", 32*10240*2, np.float16, 7.7), ("gate32", 32*6144*2, np.float16, 7.7),
      ("z32", 32*6144*2, np.float16, 7.7), ("gact32", 32*17408*2, np.float16, 7.7),
      ("araw32", 32*48*4, np.float32, 7.7e31), ("braw32", 32*48*4, np.float32, 7.7e31),
      ("qrow32", 32*12288*2, np.float16, 7.7), ("krow32", 32*1024*2, np.float16, 7.7),
      ("vrow32", 32*1024*2, np.float16, 7.7), ("qw32", 32*24*256*2, np.float16, 7.7),
      ("ao32", 32*6144*2, np.float16, 7.7), ("pmA", 4*S*96*4, np.float32, 7.7e31),
      ("psA", 4*S*96*4, np.float32, 7.7e31), ("pAA", 4*S*96*256*4, np.float32, 7.7e31),
      ("pmB", 4*S*96*4, np.float32, 7.7e31), ("psB", 4*S*96*4, np.float32, 7.7e31),
      ("pAB", 4*S*96*256*4, np.float32, 7.7e31)]:
      P.poison(nm, nb, dt, v)
    P.up("ids16a", np.zeros(H, dtype=np.int32))
    P.up("ids16b", np.zeros(H, dtype=np.int32))
    P.up("ids32", np.zeros(2 * H, dtype=np.int32))   # P12 N32: one 32-id upload
    P.up("pos_slot_b", np.zeros(1, dtype=np.int32))
    dev.synchronize()
    # half views (rows 16..31 of each 32-row buffer; byte offsets), stashed on E
    E._pf_hv = {nm + "b": d[nm].offset(offset=H*nb, size=H*nb) for nm, nb in
                [("xA32", 5120*4), ("xB32", 5120*4), ("xh32", 5120*2), ("hh32", 5120*4),
                 ("hhx32", 5120*2), ("attn_out32", 5120*2), ("qkv32", 10240*2),
                 ("gate32", 6144*2), ("z32", 6144*2), ("gact32", 17408*2),
                 ("araw32", 48*4), ("braw32", 48*4), ("qrow32", 12288*2),
                 ("krow32", 1024*2), ("vrow32", 1024*2), ("qw32", 24*256*2), ("ao32", 6144*2)]}
    dev.synchronize()
  if DFILL:
    for nm, nb, dt, v in [
        ("e16f_d", M*5120*4, np.float32, 7.7e31), ("cat16_d", M*10240*2, np.float16, 7.7),
        ("xin_d16", M*5120*4, np.float32, 7.7e31), ("xh_d16", M*5120*2, np.float16, 7.7),
        ("qrow_d16", M*12288*2, np.float16, 7.7), ("krow_d16", M*1024*2, np.float16, 7.7),
        ("vrow_d16", M*1024*2, np.float16, 7.7), ("qw16_d", M*24*256*2, np.float16, 7.7)]:
      P.poison(nm, nb, dt, v)
    P.up("zed5k16", np.zeros(M*5120, dtype=np.float32))
    P.up("REC0", np.zeros(17*5120, dtype=np.float32))   # FRESH seed row0 = zeros
    P.up("REC1", np.zeros(17*5120, dtype=np.float32))
  dev.synchronize(); P._keep.clear()

  # ---- P7F3 G3M: budget-ordered both-live r7 swap. The ORIGINAL weight buffers
  # stay resident (the decode G_CYCLE graphs hold their fixed handles), so packed7
  # coverage is capped by free VRAM (PF_G3M_MB). Priority: fd (down-proj, biggest
  # covered class per byte), gate, out, q/k, then fg/fu pairs only if budget
  # remains. Kernels = the P7B bit-identical r7 m32 cubins; per-chunk math
  # unchanged (same decode words, same k-order). ----
  W7 = {}
  if G3M:
    P7D = f"{BASE}/packed7"
    budget = (PERSIST_MB if PERSIST else float(os.getenv("PF_G3M_MB", "4300"))) * 1e6
    cands = []
    for i in range(64):
      if PERSIST:   # P14: the persistent tier consumes ONLY fg/fu (packed7 unit-stream layout)
        pf_, pu_ = f"{P7D}/fg{i}.npy", f"{P7D}/fu{i}.npy"
        if os.path.exists(pf_) and os.path.exists(pu_):
          cands.append((2, os.path.getsize(pf_) + os.path.getsize(pu_), [("fg", i, pf_), ("fu", i, pu_)]))
        continue
      for t in ("fd", "gate", "out", "q", "k"):
        p = f"{P7D}/{t}{i}.npy"
        if os.path.exists(p):
          cands.append((0 if t == "fd" else 1, os.path.getsize(p), [(t, i, p)]))
      pf_, pu_ = f"{P7D}/fg{i}.npy", f"{P7D}/fu{i}.npy"
      if os.path.exists(pf_) and os.path.exists(pu_):
        cands.append((2, os.path.getsize(pf_) + os.path.getsize(pu_), [("fg", i, pf_), ("fu", i, pu_)]))
    if os.getenv("PF_G3M_ORDER", "") == "ffn":   # P15: fg/fu-first (the m64 ffn rate probe)
      cands.sort(key=lambda c: (0 if c[0] == 2 else 1, -c[1]))
    else:
      cands.sort(key=lambda c: (c[0], -c[1]))
    # R2c decode-r7: (t,i) the trunk ALREADY uploaded as packed7 (PF_DR7 — decode
    # reads the same plane) — alias the SAME buffers into W7: zero extra VRAM, so
    # the budget admits every remaining class => FULL m64 coverage.
    nat = getattr(E, "_r7native", set())
    for (t, i) in sorted(nat):
      W7[(t, i)] = E.W[(t, i)]
    used = 0; nk = len(W7)
    for prio, sz, keys in cands:
      if all((t, i) in W7 for (t, i, p) in keys):
        continue   # already native (or already uploaded)
      if used + sz > budget:
        continue
      for t, i, p in keys:
        if (t, i) in W7: continue
        W7[(t, i)] = P.up(f"r7_{t}_{i}", np.load(p)); nk += 1
      used += sz
      if nk % 16 == 15:
        dev.synchronize(); P._keep.clear()
    dev.synchronize(); P._keep.clear()
    cov = {}
    for (t, i) in W7:
      cov[t] = cov.get(t, 0) + 1
    print(f"[g3m] r7 swap: {nk} tensors, {used/1e9:.2f} GB (budget {budget/1e9:.2f} GB) cov {cov}", flush=True)
    E._pf_W7 = W7   # P14: expose packed7 handles for in-plan persistent-FFN benches
  # R8: packed5 qkv upload (PF_P5) — the gdnqg qkv seg true-16B plane. Raw Q5_K
  # stays resident (the decode q5g8v family reads it) => net +1.9GB VRAM.
  W5 = {}
  if P5:
    nk5 = 0
    for i in E.gdn_idx:
      p5f = f"{BASE}/packed5/qkv{i}.npy"
      assert os.path.exists(p5f), f"PF_P5: missing {p5f} (run pack_w5.py)"
      W5[i] = P.up(f"p5_qkv_{i}", np.load(p5f, mmap_mode="r")); nk5 += 1
      if nk5 % 8 == 7:
        dev.synchronize(); P._keep.clear()
    dev.synchronize(); P._keep.clear()
    print(f"[p5] packed5 swap: {nk5} tensors, {nk5*39.32/1000:.2f} GB", flush=True)
  E._pf_W5 = W5

  # P8w4-v2: the W4A8 ffn reads the EXISTING packed7 planes (zero new VRAM) —
  # only the 1KB linearized codebook LUT is uploaded (see p8_w4ffn7.cu).
  E._pf_W4 = {}

  # ---- P10: co-resident t32 attention cubins + the A4 ticket counter ----
  if ATTN32:
    _a10 = [("pfa32ctl_s13_100k", "pfa32ctl")] if A4 else \
           [("pfa32c_t32_s13_100k", "pfa32ct"), ("pfc16t_s13", "pfc16t")]
    if HYB:
      _a10 += [("pfa32c_t32_s26_100k", "pfa32ct"), ("pfc16t_s26", "pfc16t")]
    for n, sym in _a10:
      lib = open(f"{BASE}/{n}.cubin", "rb").read()
      pr[n] = NVProgram(dev, TinyELF(lib=lib, name=sym, target=dev.renderer.target, signature=tuple()))
    if A4:
      P.up("atc_ctr", np.zeros(1, dtype=np.uint32))   # self-resetting (stays 0 across launches)
    dev.synchronize(); P._keep.clear()
    print(f"[pf10] attn32={'A4-fused' if A4 else 't32+pfc16t'} cubins live", flush=True)

  # ---- cached launch plan (fixed handles; ids16/pos_slot mutated per chunk) ----
  plan = []
  def A(p, *a, g=1, ls=LS):
    plan.append((p, a, g, ls))
  HV = getattr(E, "_pf_hv", {})
  if M32:
    # P6 M32 plan: the 6 GEMM launches per block run M=32 (one 32-row pass);
    # norms/pre/attn/scan run as 2x16 halves on offset views (the seam).
    # P12: N32/PRE32/SCAN32 merge the halves where a 32-row launch is
    # bit-identical (row-per-CTA norms/emb; TROWS=32 pre/scan twins).
    if N32:
      A(pr["pfk_emb16"], W[("emb", 0)], d["grid512"], d["ids32"], d["xA32"], g=32)
    else:
      A(pr["pfk_emb16"], W[("emb", 0)], d["grid512"], d["ids16a"], d["xA32"], g=16)
      A(pr["pfk_emb16"], W[("emb", 0)], d["grid512"], d["ids16b"], HV["xA32b"], g=16)
    cur = 0
    for i in range(64):
      xin = d["xA32"] if cur == 0 else d["xB32"]
      xout = d["xB32"] if cur == 0 else d["xA32"]
      xinb, xoutb = HV["xA32b"] if cur == 0 else HV["xB32b"], HV["xB32b"] if cur == 0 else HV["xA32b"]
      if i in E.qtypes:
        if N32:
          A(pr["pfk_n16"], xin, W[("nw1", i)], d["xh32"], g=32)
        else:
          A(pr["pfk_n16"], xin, W[("nw1", i)], d["xh32"], g=16)
          A(pr["pfk_n16"], xinb, W[("nw1", i)], HV["xh32b"], g=16)
        if ("k", i) in W7 and (E.qtypes[i] == 14 or ("q", i) in W7):
          mqn = "pfg3m_attnqkvq6_r7_m32_nw8k128" if E.qtypes[i] == 14 else "pfg3m_attnqkvi3_r7_m32_nw8k128"
          qw = W[("q", i)] if E.qtypes[i] == 14 else W7[("q", i)]
          A(pr[mqn], qw, W7[("k", i)], W[("v", i)], d["gridf"], d["xh32"],
            d["qrow32"], d["krow32"], d["vrow32"], g=224, ls=LS)
        else:
          mqn = "pfg2_attnqkvq6_m32_hm_nw8k128" if E.qtypes[i] == 14 else "pfg2_attnqkvi3_m32_hm_nw8k128"
          A(pr[mqn], W[("q", i)], W[("k", i)], W[("v", i)], d["gridf"], d["xh32"],
            d["qrow32"], d["krow32"], d["vrow32"], g=224, ls=LS)
        if PRE32:
          A(pr["pfk_pre32_100k"], d["qrow32"], d["krow32"], d["vrow32"], W[("qnw", i)], W[("knw", i)], d["freqs"],
            d[f"kv{i}"], d[f"sc{i}"], d["pos_slot"], d["qw32"], g=24)
        else:
          A(pr["pfk_pre16_100k"], d["qrow32"], d["krow32"], d["vrow32"], W[("qnw", i)], W[("knw", i)], d["freqs"],
            d[f"kv{i}"], d[f"sc{i}"], d["pos_slot"], d["qw32"], g=24)
          A(pr["pfk_pre16_100k"], HV["qrow32b"], HV["krow32b"], HV["vrow32b"], W[("qnw", i)], W[("knw", i)], d["freqs"],
            d[f"kv{i}"], d[f"sc{i}"], d["pos_slot_b"], HV["qw32b"], g=24)
        if ATTN32:
          # P10: t32 co-resident attention (grid 4*13*3 = 156 x 512thr, 2/SM at CFG=100)
          if A4:
            A(pr["pfa32ctl_s13_100k"], d[f"kv{i}"], d[f"sc{i}"], d["qw32"], d["pos_slot"],
              d["pmA"], d["psA"], d["pAA"], d["qrow32"], d["ao32"], d["atc_ctr"],
              g=4*S13*3, ls=(512, 1, 1))
            A(pr["pfa32ctl_s13_100k"], d[f"kv{i}"], d[f"sc{i}"], HV["qw32b"], d["pos_slot_b"],
              d["pmB"], d["psB"], d["pAB"], HV["qrow32b"], HV["ao32b"], d["atc_ctr"],
              g=4*S13*3, ls=(512, 1, 1))
          else:
            A(pr["pfa32c_t32_s13_100k"], d[f"kv{i}"], d[f"sc{i}"], d["qw32"], d["pos_slot"],
              d["pmA"], d["psA"], d["pAA"], g=4*S13*3, ls=(512, 1, 1))
            A(pr["pfa32c_t32_s13_100k"], d[f"kv{i}"], d[f"sc{i}"], HV["qw32b"], d["pos_slot_b"],
              d["pmB"], d["psB"], d["pAB"], g=4*S13*3, ls=(512, 1, 1))
            A(pr["pfc16t_s13"], d["pmA"], d["psA"], d["pAA"], d["qrow32"], d["ao32"], g=24)
            A(pr["pfc16t_s13"], d["pmB"], d["psB"], d["pAB"], HV["qrow32b"], HV["ao32b"], g=24)
        else:
          A(pr["pfa16nw32_s32_100k"], d[f"kv{i}"], d[f"sc{i}"], d["qw32"], d["pos_slot"],
            d["pmA"], d["psA"], d["pAA"], g=4*S, ls=(1024, 1, 1))
          A(pr["pfa16nw32_s32_100k"], d[f"kv{i}"], d[f"sc{i}"], HV["qw32b"], d["pos_slot_b"],
            d["pmB"], d["psB"], d["pAB"], g=4*S, ls=(1024, 1, 1))
          A(pr["pfc16_s32"], d["pmA"], d["psA"], d["pAA"], d["qrow32"], d["ao32"], g=24)
          A(pr["pfc16_s32"], d["pmB"], d["psB"], d["pAB"], HV["qrow32b"], HV["ao32b"], g=24)
        A(pr["pfg_iq3s_m32_hm_nw8k128"], W[("o", i)], d["grid512"], d["ao32"], d["attn_out32"], g=80, ls=LS)
      else:
        if N32:
          A(pr["pfk_ab16"], xin, W[("nw1", i)], W[("alpha", i)], W[("beta", i)],
            d["xh32"], d["araw32"], d["braw32"], g=32*13)
        else:
          A(pr["pfk_ab16"], xin, W[("nw1", i)], W[("alpha", i)], W[("beta", i)],
            d["xh32"], d["araw32"], d["braw32"], g=16*13)
          A(pr["pfk_ab16"], xinb, W[("nw1", i)], W[("alpha", i)], W[("beta", i)],
            HV["xh32b"], HV["araw32b"], HV["braw32b"], g=16*13)
        if ("gate", i) in W7:
          A(pr["pfg3m_gdnqg_r7_m32_nw16k128"], W[("qkv", i)], W7[("gate", i)], d["gridf"], d["xh32"],
            d["qkv32"], d["gate32"], g=128, ls=(512, 1, 1))
        else:
          A(pr["pfg2_gdnqg_m32_hm_nw16k128"], W[("qkv", i)], W[("gate", i)], d["gridf"], d["xh32"],
            d["qkv32"], d["gate32"], g=128, ls=(512, 1, 1))
        if SCAN32:
          A(pr["pfs32"], d[f"conv{i}_0"], d[f"rec{i}"], d["qkv32"], d["gate32"], W[("convw", i)], W[("dtb", i)], W[("ssma", i)],
            d["araw32"], d["braw32"], d["q"], d["k"], d["v"], d["core"], W[("snw", i)], d["z32"], g=48)
        else:
          A(pr["pfs16"], d[f"conv{i}_0"], d[f"rec{i}"], d["qkv32"], d["gate32"], W[("convw", i)], W[("dtb", i)], W[("ssma", i)],
            d["araw32"], d["braw32"], d["q"], d["k"], d["v"], d["core"], W[("snw", i)], d["z32"], g=48)
          A(pr["pfs16"], d[f"conv{i}_0"], d[f"rec{i}"], HV["qkv32b"], HV["gate32b"], W[("convw", i)], W[("dtb", i)], W[("ssma", i)],
            HV["araw32b"], HV["braw32b"], d["q"], d["k"], d["v"], d["core"], W[("snw", i)], HV["z32b"], g=48)
        if (not E.gdn_oq8[i]) and ("out", i) in W7:
          A(pr["pfg3_iq3o_r7_m32_nw8k128"], W7[("out", i)], d["gridf"], d["z32"], d["attn_out32"], g=80, ls=LS)
        else:
          on = "pfg_q8o_m32_hm_nw8k64" if E.gdn_oq8[i] else "pfg_iq3o_m32_hm_nw8k128"
          A(pr[on], W[("out", i)], d["gridf"], d["z32"], d["attn_out32"], g=80, ls=LS)
      if N32:
        A(pr["pfk_hh16"], xin, d["attn_out32"], W[("nw2", i)], d["hh32"], d["hhx32"], g=32)
      else:
        A(pr["pfk_hh16"], xin, d["attn_out32"], W[("nw2", i)], d["hh32"], d["hhx32"], g=16)
        A(pr["pfk_hh16"], xinb, HV["attn_out32b"], W[("nw2", i)], HV["hh32b"], HV["hhx32b"], g=16)
      if PERSIST and ("fg", i) in W7:
        A(pr[PERSIST_NAME], W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx32"], d["gact32"], g=82, ls=LS)
      elif ("fg", i) in W7:
        A(pr["pfg3_ffn_r7_m32_nw8k128"], W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx32"], d["gact32"], g=272, ls=LS)
      else:
        if NT32:
          A(pr["pfg_ffn_m32_nt32_hm_nw4k128"], W[("fg", i)], W[("fu", i)], d["gridf"], d["hhx32"], d["gact32"], g=544, ls=(128, 1, 1))
        else:
          A(pr["pfg_ffn_m32_hm_nw8k128"], W[("fg", i)], W[("fu", i)], d["gridf"], d["hhx32"], d["gact32"], g=272, ls=LS)
      if ("fd", i) in W7:
        A(pr["pfg3_iq3d_r7_m32_nw8k128"], W7[("fd", i)], d["gridf"], d["gact32"], d["hh32"], xout, g=80, ls=LS)
      else:
        A(pr["pfg_iq3d_m32_res_hm_nw8k128"], W[("fd", i)], d["gridf"], d["gact32"], d["hh32"], xout, g=80, ls=LS)
      cur ^= 1
    E._pf_plan = plan
    if HYB:
      # P11 hybrid: same plan with the attention arm swapped S13->S26 (grid
      # 4*26*3 = 312; combine identical shape). Partials pm/ps/pA are allocated
      # at S=32 slots >= 4*26*96 -- the S26 workspace FITS the existing buffers.
      _p13a, _p13c = pr["pfa32c_t32_s13_100k"], pr["pfc16t_s13"]
      _p26a, _p26c = pr["pfa32c_t32_s26_100k"], pr["pfc16t_s26"]
      E._pf_plan26 = [(_p26a, a, 4 * 26 * 3, ls) if p is _p13a else
                      (_p26c, a, g, ls) if p is _p13c else (p, a, g, ls)
                      for (p, a, g, ls) in plan]
      print(f"[pf11] hybrid plan26 ready ({sum(1 for t in E._pf_plan26 if t[0] is _p26a)} attn launches, THR={ATTN_THR})", flush=True)
    E._pf_last = d["xA32"] if cur == 0 else d["xB32"]   # final-hidden buffer (row 31 = last token)
    dev.synchronize()
    return
  A(pr["pfk_emb16"], W[("emb", 0)], d["grid512"], d["ids16"], d["xA16"], g=16)
  cur = 0
  for i in range(64):
    xin, xout = (d["xA16"] if cur == 0 else d["xB16"]), (d["xB16"] if cur == 0 else d["xA16"])
    if i in E.qtypes:
      A(pr["pfk_n16"], xin, W[("nw1", i)], d["xh16"], g=16)
      if MERGE:
        mqn = "pfg2_attnqkvq6_hm_nw8k128" if E.qtypes[i] == 14 else "pfg2_attnqkvi3_hm_nw8k128"
        A(pr[mqn], W[("q", i)], W[("k", i)], W[("v", i)], d["gridf"], d["xh16"], d["qrow16"], d["krow16"], d["vrow16"], g=224, ls=LS)
      else:
        qn = "pfg_q6q_hm_nw8k64" if E.qtypes[i] == 14 else "pfg_iq3q_hm_nw8k128"
        g_, ls_ = GRIDS[qn]; A(pr[qn], W[("q", i)], d["gridf"], d["xh16"], d["qrow16"], g=g_, ls=ls_)
        g_, ls_ = GRIDS["pfg_iq3k_hm_nw8k128"]; A(pr["pfg_iq3k_hm_nw8k128"], W[("k", i)], d["gridf"], d["xh16"], d["krow16"], g=g_, ls=ls_)
        g_, ls_ = GRIDS["pfg_q4v_hm_nw8k128"]; A(pr["pfg_q4v_hm_nw8k128"], W[("v", i)], d["gridf"], d["xh16"], d["vrow16"], g=g_, ls=ls_)
      A(pr["pfk_pre16_100k"], d["qrow16"], d["krow16"], d["vrow16"], W[("qnw", i)], W[("knw", i)], d["freqs"],
        d[f"kv{i}"], d[f"sc{i}"], d["pos_slot"], d["qw16"], g=24)
      A(pr["pfa16nw32_s32_100k"], d[f"kv{i}"], d[f"sc{i}"], d["qw16"], d["pos_slot"], d["pm16"], d["ps16"], d["pA16"],
        g=4*S, ls=(1024, 1, 1))
      A(pr["pfc16_s32"], d["pm16"], d["ps16"], d["pA16"], d["qrow16"], d["ao16"], g=24)
      g_, ls_ = GRIDS["pfg_iq3s_hm_nw8k128"]; A(pr["pfg_iq3s_hm_nw8k128"], W[("o", i)], d["grid512"], d["ao16"], d["attn_out16"], g=g_, ls=ls_)
    else:
      A(pr["pfk_ab16"], xin, W[("nw1", i)], W[("alpha", i)], W[("beta", i)], d["xh16"], d["araw16"], d["braw16"], g=16*13)
      if MERGE:
        A(pr["pfg2_gdnqg_hm_nw16k128"], W[("qkv", i)], W[("gate", i)], d["gridf"], d["xh16"], d["qkv16"], d["gate16"], g=128, ls=(512, 1, 1))
      else:
        g_, ls_ = GRIDS["pfg_q5kv_hm_nw16k128"]; A(pr["pfg_q5kv_hm_nw16k128"], W[("qkv", i)], d["gridf"], d["xh16"], d["qkv16"], g=g_, ls=ls_)
        g_, ls_ = GRIDS["pfg_iq3g_hm_nw16k128"]; A(pr["pfg_iq3g_hm_nw16k128"], W[("gate", i)], d["gridf"], d["xh16"], d["gate16"], g=g_, ls=ls_)
      A(pr["pfs16"], d[f"conv{i}_0"], d[f"rec{i}"], d["qkv16"], d["gate16"], W[("convw", i)], W[("dtb", i)], W[("ssma", i)],
        d["araw16"], d["braw16"], d["q"], d["k"], d["v"], d["core"], W[("snw", i)], d["z16"], g=48)
      on = "pfg_q8o_hm_nw8k64" if E.gdn_oq8[i] else "pfg_iq3o_hm_nw8k128"
      g_, ls_ = GRIDS[on]; A(pr[on], W[("out", i)], d["gridf"], d["z16"], d["attn_out16"], g=g_, ls=ls_)
    A(pr["pfk_hh16"], xin, d["attn_out16"], W[("nw2", i)], d["hh16"], d["hhx16"], g=16)
    g_, ls_ = GRIDS["pfg_ffn_hm_nw8k128"]; A(pr["pfg_ffn_hm_nw8k128"], W[("fg", i)], W[("fu", i)], d["gridf"], d["hhx16"], d["gact16"], g=g_, ls=ls_)
    g_, ls_ = GRIDS["pfg_iq3d_res_hm_nw8k128"]; A(pr["pfg_iq3d_res_hm_nw8k128"], W[("fd", i)], d["gridf"], d["gact16"], d["hh16"], xout, g=g_, ls=ls_)
    cur ^= 1
  E._pf_plan = plan
  E._pf_last = d["xA16"] if cur == 0 else d["xB16"]   # final-hidden buffer (row 15 = last token)
  dev.synchronize()

def _run_plan(plan):
  n = 0
  for p, a, g, ls in plan:
    p(*a, global_size=(g, 1, 1), local_size=ls)
    n += 1
    if n % 200 == 0:
      dev.synchronize()
  dev.synchronize()


# ====================== P7F: CHUNK-GRAPH CAPTURE (PF_PG=1) ======================
# The M32 chunk program is a FIXED launch sequence (same kernels/shapes/handles,
# ~706 launches/chunk) paying ~0.34 ms/launch enqueue = the dominant cost
# (P7E7: SC chunk 80% launch-bound). Capture it gcycle-style: fixed-handle
# kernargs slab + QMD-chained execs in ONE NVComputeQueue per split, timeline
# wait/signal chaining. Local sizes come from the PLAN tuples (explicit ls --
# immune to the name-encoded warp-token trap). Per-chunk host work stays the
# win_up DMAs (ids16a/b, pos_slot, pos_slot_b) which chain on the SAME global
# timeline (copy-queue waits value-1 = our graph's signal; graph waits the
# copies' signals) -- zero extra syncs, no ids race.
PG_SPLIT = int(os.getenv("PG_SPLIT", "2"))       # sub-queues per chunk (~353 each, decode-proven 452-484 class)
PG_REBUILD = int(os.getenv("PG_REBUILD", "256")) # M1A_GEN_REBUILD_EVERY-class discipline (~950-replay dext budget)

class PfGraph:
  def __init__(self, seq, tag):
    from tinygrad.helpers import round_up as _ru
    from tinygrad.uop.ops import UOp
    from tinygrad.dtype import dtypes
    from tinygrad.device import BufferSpec
    from tinygrad.runtime.ops_nv import NVComputeQueue
    self.prev_var = UOp.variable(f"{tag}_prev", 0, 0xffffffff, dtype=dtypes.uint32)
    self.cur_var = UOp.variable(f"{tag}_cur", 0, 0xffffffff, dtype=dtypes.uint32)
    per = max(_ru(p.kernargs_alloc_size, 8) for p, a, g, ls in seq)
    self.ka = dev.allocator.alloc(per * len(seq), BufferSpec(cpu_access=True, nolru=True))
    q = NVComputeQueue()
    # P7F-1 law: the timeline WAIT must precede the shader-cache INVALIDATE.
    # Barrier-first + pipelined submits = GSP "SKEDCHECK22_INVALIDATE_ACTIVE_QMD"
    # fault (the invalidate fires while the prior chunk's QMDs are still active;
    # 8k arm-3 fault, diagnosed 09-17). Wait-first stalls the pushbuffer, so the
    # invalidate only runs after the previous chunk signaled completion.
    q.wait(dev.timeline_signal, self.prev_var)
    q.memory_barrier()
    off = 0
    for p, a, g, ls in seq:
      st = p.fill_kernargs(tuple(a), (), kernargs=self.ka.offset(offset=off, size=p.kernargs_alloc_size))
      q.exec(p, st, (g, 1, 1), ls)
      off += _ru(p.kernargs_alloc_size, 8)
    q.signal(dev.timeline_signal, self.cur_var)
    self.q = q
    print(f"[pf-pg] {tag}: launches={len(seq)} _q_words={len(q._q)} ring_bytes={len(q._q)*4}", flush=True)

  def submit(self, prev_v, cur_v):
    self.q.submit(dev, {self.prev_var.expr: int(prev_v), self.cur_var.expr: int(cur_v)})

# P0 fix (R8): per-path graph sets, keyed by the EXPLICIT caller path. The
# old ambient-flag inference (m64 = _M64ON and not _M128ON) submitted the
# M32/M128 graphs inside M64-context calls whenever PF_M128=1 — the graphs
# then read ids32/pos_slot that the M64 loop never writes (one-request-stale
# feeds + wrong tail positions; R7B_DECIDERS §7). Every loop now passes
# which= explicitly; None falls back to the legacy inference.
_PG = {"sets": {}}

def _pf_dfill_seq(E):
  """The two per-half draft-fill windows as plan tuples (M32: FIXED args every
  chunk -- ring alternates per HALF, half A always REC0->REC1)."""
  d, W, pr = E.P.d, E.W, E.pr
  HV = E._pf_hv
  # P12 N32: windows slice ids32 (16-int views) so ONE upload feeds everything.
  ida = d["ids32"] if N32 else d["ids16a"]
  idb = d["ids32"].offset(offset=16 * 4, size=16 * 4) if N32 else d["ids16b"]
  seq = []
  for (w, wn, xbuf, idsb, posslot) in ((d["REC0"], d["REC1"], d["xA32"], ida, d["pos_slot"]),
                                       (d["REC1"], d["REC0"], HV["xA32b"], idb, d["pos_slot_b"])):
    seq.append((pr["pfk_rec16"], (xbuf, w, wn), 17, LS))
    seq.append((pr["pfk_emb16"], (W[("emb", 0)], d["grid512"], idsb, d["e16f_d"]), 16, LS))
    seq.append((pr["pfd_dnorm16"], (d["e16f_d"], w, d["d_enw"], d["d_hnw"], d["cat16_d"]), 16, LS))
    seq.append((pr["pfg_ehd_res_hm_nw8k128"], (d["d_eh"], d["gridf"], d["cat16_d"], d["zed5k16"], d["xin_d16"]), 80, LS))
    seq.append((pr["pfk_n16"], (d["xin_d16"], d["d_nw1"], d["xh_d16"]), 16, LS))
    seq.append((pr["pfg2_dqkv_hm_nw8k128"], (d["d_q"], d["d_k"], d["d_v"], d["gridf"], d["xh_d16"],
                                             d["qrow_d16"], d["krow_d16"], d["vrow_d16"]), 224, LS))
    seq.append((pr["pfk_pre16_100k"], (d["qrow_d16"], d["krow_d16"], d["vrow_d16"], d["d_qnw"], d["d_knw"],
                                        d["freqs"], d["kv_d"], d["sc_d"], posslot, d["qw16_d"]), 24, LS))
  return seq

def _pf_graphs(E, which=None):
  m64 = bool(_M64ON) and not _M128ON and getattr(E, "_pf_plan64", None) is not None
  m128 = bool(_M128ON) and M128 and getattr(E, "_pf_plan128", None) is not None
  if which is None:
    which = "m128" if m128 else ("m64" if getattr(E, "_pf_plan64", None) is not None else "m32")
  assert which in ("m32", "m64", "m128"), which
  use64 = which == "m64" and getattr(E, "_pf_plan64", None) is not None
  use128 = which == "m128" and getattr(E, "_pf_plan128", None) is not None
  key = (which, use64, use128, M32, DFILL, G3M, NT32, ATTN32, A4, HYB, ATTN_THR, N32, PRE32, SCAN32, PERSIST, PERSIST_NAME, ATTNW, SCANC, SCANC_N2, M64QKV, ABW, FFNSPLIT, RING4, QKV1, PRE32X4, W4FFN)
  sets = _PG["sets"]
  st = sets.get(which)
  if st is None or st["key"] != key or st["n"] >= PG_REBUILD:
    if st is not None:
      dev.synchronize()   # rebuild ONLY at a quiescent point
    if use128:
      full = list(E._pf_plan128) + (_pf_dfill_seq128(E) if (M32 and DFILL) else [])
    elif use64:
      full = list(E._pf_plan64) + (_pf_dfill_seq64(E) if (M32 and DFILL) else [])
    else:
      full = list(E._pf_plan) + (_pf_dfill_seq(E) if (M32 and DFILL) else [])
    n = max(1, PG_SPLIT)
    k = (len(full) + n - 1) // n
    subs = [full[i:i + k] for i in range(0, len(full), k)]
    graphs = [PfGraph(s, f"pfc{which[1]}{j}") for j, s in enumerate(subs)]
    _p26 = None if use128 else (getattr(E, "_pf_plan26_64", None) if use64 else getattr(E, "_pf_plan26", None))
    graphs26 = None
    if HYB and _p26 is not None:
      # P11: the S26 twin graph set (identical except the attention arm)
      _d26 = (_pf_dfill_seq64(E) if (use64 and M32 and DFILL) else (_pf_dfill_seq(E) if (M32 and DFILL) else []))
      full26 = list(_p26) + _d26
      subs26 = [full26[i:i + k] for i in range(0, len(full26), k)]
      graphs26 = [PfGraph(s, f"pfw{j}") for j, s in enumerate(subs26)]
    sets[which] = st = {"graphs": graphs, "graphs26": graphs26, "n": 0, "key": key}
    print(f"[pf-pg] {which} chunk graph ready: {len(subs)} queues x ~{k} launches ({len(full)} total, dfill={DFILL}"
          f"{', +S26 twin' if graphs26 is not None else ''})", flush=True)
  return st["graphs"]

def _pf_submit_chunk(E, pos0=None, which=None):
  """Submit one captured chunk; returns the final timeline value.
  P0 fix: which = the calling path ("m32"/"m64"/"m128") — the graph set MUST
  match the loop's win_up buffers. P11 hybrid: pos0 >= ATTN_THR submits the
  S26 graph set (if resident)."""
  gs26 = _PG["sets"].get(which, {}).get("graphs26")
  gs = gs26 if (pos0 is not None and gs26 is not None and pos0 >= ATTN_THR) else _pf_graphs(E, which)
  _PG["sets"][which or "m32"]["n"] += 1
  prev = dev.timeline_value - 1
  for g in gs[:-1]:
    v = dev.next_timeline(); g.submit(prev, v); prev = v
  v = dev.next_timeline(); gs[-1].submit(prev, v)
  return v



# ====================== P7e: SUPER-CHUNK (PF_SC=256) ======================
# Per super-chunk of SC tokens: emb/norms M-wide -> merged GEMMs (gemm3-m64 on
# packed blocks x (SC/64) m-passes; classic m32 x (SC/32) on the rest) ->
# [attn: pfk_pre64 (ONE flat launch, 4x64-row kv/qw appends) -> per 32-tok
#  window: the SHIPPED pfa16 pair + pfc16 (P7d verdict: widening falsified
#  per-token; keep the pair) -> o GEMM] / [GDN: qg GEMM -> pfca/pfcb/pfcz
#  c64_nc4 chunked WY scan (LIVE trunk rec/conv, in-place-safe disjoint
#  slices) -> o GEMM] -> FFN. Head on the last row. Tails delegate to the
# existing M32 path. dfill: 16 interleaved windows (pos_w = pos0+16k).
SC = int(os.getenv("PF_SC", "256"))
SCANC = os.getenv("PF_SCANC", "0") == "1"   # R2b: M64-trunk WY scan (kernel FIXED 256ec3c-era: staging race + Xf upper-triangle + accM passes)
SCANC_N2 = os.getenv("PF_SCANC_N2", "0") == "1"  # R2b: NC=2 single-launch tier (3 launches/chunk vs 6; numerically identical by construction)
M64QKV = os.getenv("PF_M64QKV", "0") == "1"   # R2b rung-6 retry: the attnqkv m64 twins (P7E4-quarantined on the SC path; the M64 trunk is a different world)
FFNSPLIT = os.getenv("PF_FFNSPLIT", "0") == "1"  # R2c: split-plane FFN m64 (fg/fu as single-plane nw8 GEMMs + pfk_smul64; bit-identical)
# P7E3: PF_G3SC = the SC-side gemm3 control (default: follow PF_GEMM3). The m64-at-M256
# wrong-VA class (P7E2 bug 2) must stay OFF for SC while the M32 serving path keeps gemm3.
G3SCN = int(os.getenv("PF_G3_BLOCKS", "16"))
NC = 4 if SC == 256 else (SC // 64)
assert SC % 64 == 0 and SC >= 64
HC64 = {64: 263696}  # P7E5: +u_lo/vr_lo/m_lo/t_lo/dz hi-lo regions  # scratch bytes per (block? no) head-chunk at C=64

SC_CUBINS = ["pfk_pre64_100k", "pfca_c64_nc4_nw16", "pfcb_c64_nc4_nw8", "pfcz_c64_nc4_nw8"]
SC_G3 = {  # gemm3-m64: (cubin, NGRID/TGRID, ls)
  "ffn": ("pfg3_ffn_r7_m64_nw4k128", 544, (128, 1, 1)),
  "iq3d": ("pfg3_iq3d_r7_m64_nw8k128", 80, (256, 1, 1)),
  "iq3o": ("pfg3_iq3o_r7_m64_nw8k128", 80, (256, 1, 1)),
  "gdnqg": ("pfg3m_gdnqg_r7_m64_nw8k128", 256, (256, 1, 1)),
  "attnqkvi3": ("pfg3m_attnqkvi3_r7_m64_nw8k128", 224, (256, 1, 1)),
  "attnqkvq6": ("pfg3m_attnqkvq6_r7_m64_nw8k128", 224, (256, 1, 1)),
}

# P7E7: PF_G3SC per-class knob. "1" = all classes (legacy), "0" = none,
# else comma list of SC_G3 keys, e.g. "ffn,iq3d,iq3o,gdnqg" (the P7E4-clean
# twins; attnqkvi3/attnqkvq6 m64 corrupt in-plan per P7E4).
_g3v = os.getenv("PF_G3SC", os.getenv("PF_GEMM3", "1"))
if _g3v == "1": G3C = frozenset(SC_G3.keys())
elif _g3v in ("0", ""): G3C = frozenset()
else: G3C = frozenset(x.strip() for x in _g3v.split(",") if x.strip() in SC_G3)
G3SC = bool(G3C)
_G3W = {"ffn": ("fg", "fu"), "iq3d": ("fd",), "iq3o": ("out",), "gdnqg": ("gate",),
        "attnqkvi3": ("q", "k"), "attnqkvq6": ("q", "k")}

def ensure_sc(E):
  if getattr(E, "_pfsc_plan", None) is not None:
    return
  if getattr(E, "_r7native", None):
    raise RuntimeError("PF_DR7 is incompatible with the SC path (PF_SUPER)")
  P, d, W, pr = E.P, E.P.d, E.W, E.pr
  for n in SC_CUBINS + ([SC_G3[c][0] for c in sorted(G3C)] if G3SC else []):
    if n in pr: continue
    lib = open(f"{BASE}/{n}.cubin", "rb").read()
    pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n.split("_100k")[0] if n.endswith("_100k") else n,
                                   target=dev.renderer.target, signature=tuple()))
  M = SC
  for nm, nb, dt, v in [
      ("xAsc", M*5120*4, np.float32, 7.7e31), ("xBsc", M*5120*4, np.float32, 7.7e31),
      ("xhsc", M*5120*2, np.float16, 7.7), ("hhsc", M*5120*4, np.float32, 7.7e31),
      ("hhxsc", M*5120*2, np.float16, 7.7), ("attn_outsc", M*5120*2, np.float16, 7.7),
      ("qkvsc", M*10240*2, np.float16, 7.7), ("gatesc", M*6144*2, np.float16, 7.7),
      ("zsc", M*6144*2, np.float16, 7.7), ("gactsc", M*17408*2, np.float16, 7.7),
      ("arawsc", M*48*4, np.float32, 7.7e31), ("brawsc", M*48*4, np.float32, 7.7e31),
      ("qrowsc", M*12288*2, np.float16, 7.7), ("krowsc", M*1024*2, np.float16, 7.7),
      ("vrowsc", M*1024*2, np.float16, 7.7), ("qwsc", M*6144*2, np.float16, 7.7),
      ("aosc", M*6144*2, np.float16, 7.7), ("pmS", 4*S*96*4, np.float32, 7.7e31),
      ("psS", 4*S*96*4, np.float32, 7.7e31), ("pAS", 4*S*96*256*4, np.float32, 7.7e31),
      ("scscr", 48*NC*HC64[64], np.uint8, 0xAB), ("sco", M*6144*4, np.float32, 7.7e31)]:
    P.up(nm, np.zeros(nb // (2 if dt == np.float16 else (4 if dt == np.float32 else 1)), dtype=dt))
  P.up("ids_sc", np.zeros(M, dtype=np.int32))
  P.up("pos_arr", np.zeros(NC, dtype=np.int32))
  P.up("pos_w", np.zeros(16, dtype=np.int32))
  dev.synchronize(); P._keep.clear()
  # scan weight plane (all 48 GDN blocks)
  if SCANC:
    from engine0 import parse_gguf, read_raw
    ds_, infos_ = parse_gguf()
    wp = np.zeros(48 * 47200, dtype=np.float32)
    for jj, i in enumerate(E.gdn_idx):
      pre = f"blk.{i}."
      b = jj * 47200
      wp[b:b+40960] = np.frombuffer(read_raw(infos_[pre+"ssm_conv1d.weight"], ds_), dtype="<f4").reshape(-1)
      wp[b+40960:b+41008] = np.frombuffer(read_raw(infos_[pre+"ssm_dt.bias"], ds_), dtype="<f4")
      wp[b+41008:b+41056] = np.frombuffer(read_raw(infos_[pre+"ssm_a"], ds_), dtype="<f4")
      wp[b+41056:b+41184] = np.frombuffer(read_raw(infos_[pre+"ssm_norm.weight"], ds_), dtype="<f4")
    P.up("scwp", wp); dev.synchronize(); P._keep.clear()
  # gemm3 packed weights
  E._pfsc_r7 = {}
  if G3SC:
    P7D = f"{BASE}/packed7"
    _wset = sorted({t for c in G3C for t in _G3W[c]})
    ups = [(t, i) for i in range(G3SCN) for t in _wset
           if os.path.exists(f"{P7D}/{t}{i}.npy")]
    for k, (t, i) in enumerate(ups):
      E._pfsc_r7[(t, i)] = P.up(f"r7s_{t}_{i}", np.load(f"{P7D}/{t}{i}.npy"))
      if k % 16 == 15: dev.synchronize(); P._keep.clear()
    dev.synchronize(); P._keep.clear()
  R7 = E._pfsc_r7
  MP64 = SC // 64        # m64 m-passes
  MP32 = SC // 32        # classic m32 passes
  plan = []
  def A(p, *a, g=1, ls=LS):
    plan.append((p, a, g, ls))
  A(pr["pfk_emb16"], W[("emb", 0)], d["grid512"], d["ids_sc"], d["xAsc"], g=SC)
  cur = 0
  for i in range(64):
    xin = d["xAsc"] if cur == 0 else d["xBsc"]
    xout = d["xBsc"] if cur == 0 else d["xAsc"]
    if i in E.qtypes:
      A(pr["pfk_n16"], xin, W[("nw1", i)], d["xhsc"], g=SC)
      i3 = E.qtypes[i] != 14
      if ("attnqkvi3" if i3 else "attnqkvq6") in G3C and ("k", i) in R7:
        nm = SC_G3["attnqkvi3" if i3 else "attnqkvq6"][0]
        qw_ = R7[("q", i)] if (not i3 and ("q", i) in R7) else W[("q", i)]
        A(pr[nm], qw_, R7[("k", i)], W[("v", i)], d["gridf"], d["xhsc"], d["qrowsc"], d["krowsc"], d["vrowsc"],
          g=SC_G3["attnqkvi3" if i3 else "attnqkvq6"][1]*MP64, ls=SC_G3["attnqkvi3"][2])
      else:
        nm = "pfg2_attnqkvq6_m32_hm_nw8k128" if not i3 else "pfg2_attnqkvi3_m32_hm_nw8k128"
        for p in range(MP32):
          A(pr[nm], W[("q", i)], W[("k", i)], W[("v", i)], d["gridf"],
            d["xhsc"].offset(offset=p*32*5120*2, size=32*5120*2),
            d["qrowsc"].offset(offset=p*32*12288*2, size=32*12288*2),
            d["krowsc"].offset(offset=p*32*1024*2, size=32*1024*2),
            d["vrowsc"].offset(offset=p*32*1024*2, size=32*1024*2), g=224, ls=LS)
      A(pr["pfk_pre64_100k"], d["qrowsc"], d["krowsc"], d["vrowsc"], W[("qnw", i)], W[("knw", i)], d["freqs"],
        d[f"kv{i}"], d[f"sc{i}"], d["pos_arr"], d["qwsc"], g=24*NC)
      for h2 in range(SC // 16):
        qb = d["qwsc"].offset(offset=h2*16*6144*2, size=16*6144*2)
        ab = d["aosc"].offset(offset=h2*16*6144*2, size=16*6144*2)
        rb = d["qrowsc"].offset(offset=h2*16*12288*2, size=16*12288*2)
        pb = d["pos_w"].offset(offset=h2*4, size=4)
        A(pr["pfa16nw32_s32_100k"], d[f"kv{i}"], d[f"sc{i}"], qb, pb, d["pmS"], d["psS"], d["pAS"], g=4*S, ls=(1024,1,1))
        A(pr["pfc16_s32"], d["pmS"], d["psS"], d["pAS"], rb, ab, g=24)
      # attn o-proj: iq3s classic per-32 (not in the packed7 set)
      for p in range(MP32):
        A(pr["pfg_iq3s_m32_hm_nw8k128"], W[("o", i)], d["grid512"],
          d["aosc"].offset(offset=p*32*6144*2, size=32*6144*2),
          d["attn_outsc"].offset(offset=p*32*5120*2, size=32*5120*2), g=80, ls=LS)
    else:
      j = E.gdn_idx.index(i)
      A(pr["pfk_ab16"], xin, W[("nw1", i)], W[("alpha", i)], W[("beta", i)], d["xhsc"], d["arawsc"], d["brawsc"], g=SC*13)
      if "gdnqg" in G3C and ("gate", i) in R7:
        A(pr[SC_G3["gdnqg"][0]], W[("qkv", i)], R7[("gate", i)], d["gridf"], d["xhsc"], d["qkvsc"], d["gatesc"],
          g=SC_G3["gdnqg"][1]*MP64, ls=SC_G3["gdnqg"][2])
      else:
        for p in range(MP32):
          A(pr["pfg2_gdnqg_m32_hm_nw16k128"], W[("qkv", i)], W[("gate", i)], d["gridf"],
            d["xhsc"].offset(offset=p*32*5120*2, size=32*5120*2),
            d["qkvsc"].offset(offset=p*32*10240*2, size=32*10240*2),
            d["gatesc"].offset(offset=p*32*6144*2, size=32*6144*2), g=128, ls=(512,1,1))
      if SCANC:
        scwpb = d["scwp"].offset(offset=j*47200*4, size=47200*4)
        A(pr["pfca_c64_nc4_nw16"], scwpb, d[f"conv{i}_0"], d["qkvsc"], d["arawsc"], d["brawsc"], d["scscr"], g=48*NC, ls=(512,1,1))
        A(pr["pfcb_c64_nc4_nw8"], d["scscr"], d[f"rec{i}"], d["sco"], g=192)
        A(pr["pfcz_c64_nc4_nw8"], d["sco"], d["gatesc"], scwpb.offset(offset=41056*4, size=6144*4),
          d["zsc"], d["qkvsc"], d[f"conv{i}_0"], g=48*NC*8)
      else:
        for p in range(MP32):
          A(pr["pfs16"], d[f"conv{i}_0"], d[f"rec{i}"],
            d["qkvsc"].offset(offset=p*32*10240*2, size=32*10240*2),
            d["gatesc"].offset(offset=p*32*6144*2, size=32*6144*2), W[("convw", i)], W[("dtb", i)], W[("ssma", i)],
            d["arawsc"].offset(offset=p*32*48*4, size=32*48*4), d["brawsc"].offset(offset=p*32*48*4, size=32*48*4),
            d["q"], d["k"], d["v"], d["core"], W[("snw", i)],
            d["zsc"].offset(offset=p*32*6144*2, size=32*6144*2), g=48)
      on3 = ("iq3o" in G3C) and (not E.gdn_oq8[i]) and ("out", i) in R7
      if on3:
        A(pr[SC_G3["iq3o"][0]], R7[("out", i)], d["gridf"], d["zsc"], d["attn_outsc"], g=SC_G3["iq3o"][1]*MP64, ls=SC_G3["iq3o"][2])
      else:
        onm = "pfg_q8o_m32_hm_nw8k64" if E.gdn_oq8[i] else "pfg_iq3o_m32_hm_nw8k128"
        for p in range(MP32):
          A(pr[onm], W[("out", i)], d["gridf"],
            d["zsc"].offset(offset=p*32*6144*2, size=32*6144*2),
            d["attn_outsc"].offset(offset=p*32*5120*2, size=32*5120*2), g=80, ls=LS)
    A(pr["pfk_hh16"], xin, d["attn_outsc"], W[("nw2", i)], d["hhsc"], d["hhxsc"], g=SC)
    if "ffn" in G3C and ("fg", i) in R7:
      A(pr[SC_G3["ffn"][0]], R7[("fg", i)], R7[("fu", i)], d["gridf"], d["hhxsc"], d["gactsc"],
        g=SC_G3["ffn"][1]*MP64, ls=SC_G3["ffn"][2])
    else:
      for p in range(MP32):
        A(pr["pfg_ffn_m32_hm_nw8k128"], W[("fg", i)], W[("fu", i)], d["gridf"],
          d["hhxsc"].offset(offset=p*32*5120*2, size=32*5120*2),
          d["gactsc"].offset(offset=p*32*17408*2, size=32*17408*2), g=272, ls=LS)
    if "iq3d" in G3C and ("fd", i) in R7:
      A(pr[SC_G3["iq3d"][0]], R7[("fd", i)], d["gridf"], d["gactsc"], d["hhsc"], xout,
        g=SC_G3["iq3d"][1]*MP64, ls=SC_G3["iq3d"][2])
    else:
      for p in range(MP32):
        A(pr["pfg_iq3d_m32_res_hm_nw8k128"], W[("fd", i)], d["gridf"],
          d["gactsc"].offset(offset=p*32*17408*2, size=32*17408*2),
          d["hhsc"].offset(offset=p*32*5120*4, size=32*5120*4),
          xout.offset(offset=p*32*5120*4, size=32*5120*4), g=80, ls=LS)
    cur ^= 1
  E._pfsc_plan = plan
  E._pfsc_last = d["xAsc"] if cur == 0 else d["xBsc"]
  dev.synchronize()

def _run_sc_chunk(E, ids, pos0, ci=-1):
  P, d, W, pr = E.P, E.P.d, E.W, E.pr
  P.win_up("ids_sc", 0, np.array([int(t) for t in ids], dtype=np.int32))
  P.win_up("pos_arr", 0, np.array([pos0 + 64*k for k in range(NC)], dtype=np.int32))
  P.win_up("pos_w", 0, np.array([pos0 + 16*k for k in range(16)], dtype=np.int32))
  dev.synchronize()   # P7e LAW: uploads must LAND before the launch train (copy-vs-compute race)
  n = 0
  _step = int(os.getenv("PF_SC_STEP", "0"))
  _csc = int(os.getenv("PF_SC_CLASSSYNC", "-1"))
  _cs = (ci == _csc)
  import time as _time
  _acc = {}; _nc = {}; _cls = None; _tc = _time.perf_counter(); _nsync = 0
  def _clsname(p):
    nm = getattr(p, "name", "?")
    for tag in ("pfca", "pfcb", "pfcz", "pre64", "pre16", "pfa16", "pfaW", "pfcW", "rec16", "emb16", "dnorm16", "ehd", "n16", "dqkv", "argmax", "head8", "n64", "gemvm"):
      if tag in nm: return tag
    base = nm.split("_")[0]
    return (base + ("-m64" if "m64" in nm else "-m32")) if base.startswith("pfg") else nm[:14]
  for p, a, g, ls in E._pfsc_plan:
    p(*a, global_size=(g, 1, 1), local_size=ls)
    n += 1
    if _cs:
      c2 = _clsname(p)
      _nc[c2] = _nc.get(c2, 0) + 1
      if c2 != _cls:
        if _cls is not None:
          _acc[_cls] = _acc.get(_cls, 0.0) + (_time.perf_counter() - _tc) * 1e3
        dev.synchronize(); _nsync += 1
        _cls = c2; _tc = _time.perf_counter()
    if n <= 8 or n % 32 == 0:
      dev.synchronize()
  if _cs:
    _acc[_cls] = _acc.get(_cls, 0.0) + (_time.perf_counter() - _tc) * 1e3
    _rows = sorted(((k, _nc[k], round(_acc.get(k, 0.0), 1)) for k in _nc), key=lambda kv: -kv[2])
    print("[sccs] ci=%d launches=%d syncs=%d | " % (ci, n, _nsync) + " ".join("%s(%dx)=%.1fms" % r for r in _rows), flush=True)
    _steps = [int(x) for x in os.getenv("PF_SC_STEPS", "").split(",") if x]
    _step = int(os.getenv("PF_SC_STEP", "0"))
    if n in _steps or (_step and n <= _step):
      dev.synchronize()
      def _nan(nm, shp, dt):
        try:
          a = P.down(nm, shp, dt).astype(np.float64)
          return f"{nm} {int(np.isnan(a).sum())}/{a.size}"
        except Exception as ex: return f"{nm} ERR"
      print(f"[s] #{n} {getattr(p,'name','?')[:22]} | " + " ".join([_nan("qkvsc", (256*10240,), np.float16), _nan("sco", (256*6144,), np.float32), _nan("rec0", (48*128*128,), np.float32), _nan("zsc", (256*6144,), np.float16)]), flush=True)
      P._keep.clear()

def prefill_batch_sc(E, G, ids, prog=None, log=None, chunk_times=None):
  """Super-chunk prefill (P7e). Same post-conditions as prefill_batch."""
  ensure(E)   # dfill machinery (REC ring, draft bufs) + the M32 tail path
  ensure_sc(E)
  P, d, W, pr = E.P, E.P.d, E.W, E.pr
  pos0 = int(P.down_at("pos_slot", 0, 1)[0])
  N = len(ids)
  nsc = N // SC
  r = N - SC * nsc
  P.win_up("tok_hist", pos0 * 4, np.array([int(t) for t in ids], dtype=np.int32))
  t0 = time.perf_counter()
  for c in range(nsc):
    tc = time.perf_counter()
    _run_sc_chunk(E, ids[SC*c:SC*c+SC], pos0 + SC*c, ci=c)
    E._pfsc_cur = E._pfsc_last
    dev.synchronize()
    if os.getenv("PF_DFILL", "1") == "1":
      for k in range(SC // 16):
        w, wn = d["REC0"] if (k & 1) == 0 else d["REC1"], d["REC1"] if (k & 1) == 0 else d["REC0"]
        curbuf = E._pfsc_cur   # residual buffer of chunk c (ping-pong parity)
        pr["pfk_rec16"](curbuf.offset(offset=16*k*5120*4, size=16*5120*4), w, wn,
                        global_size=(17, 1, 1), local_size=LS)
        pr["pfk_emb16"](W[("emb", 0)], d["grid512"], d["ids_sc"].offset(offset=16*k*4, size=16*4), d["e16f_d"],
                        global_size=(16, 1, 1), local_size=LS)
        pr["pfd_dnorm16"](d["e16f_d"], w, d["d_enw"], d["d_hnw"], d["cat16_d"], global_size=(16, 1, 1), local_size=LS)
        pr["pfg_ehd_res_hm_nw8k128"](d["d_eh"], d["gridf"], d["cat16_d"], d["zed5k16"], d["xin_d16"], global_size=(80, 1, 1), local_size=LS)
        pr["pfk_n16"](d["xin_d16"], d["d_nw1"], d["xh_d16"], global_size=(16, 1, 1), local_size=LS)
        pr["pfg2_dqkv_hm_nw8k128"](d["d_q"], d["d_k"], d["d_v"], d["gridf"], d["xh_d16"],
                                   d["qrow_d16"], d["krow_d16"], d["vrow_d16"], global_size=(224, 1, 1), local_size=LS)
        pr["pfk_pre16_100k"](d["qrow_d16"], d["krow_d16"], d["vrow_d16"], d["d_qnw"], d["d_knw"], d["freqs"],
                             d["kv_d"], d["sc_d"], d["pos_w"].offset(offset=k*4, size=4), d["qw16_d"],
                             global_size=(24, 1, 1), local_size=LS)
      # seed row0 of the next ring from the last trunk hidden of this chunk
      last = E._pfsc_cur
      pr["pfk_rec16"](last.offset(offset=(SC-16)*5120*4, size=16*5120*4), d["REC0"], d["REC1"], global_size=(17, 1, 1), local_size=LS)
    if chunk_times is not None:
      chunk_times.append((pos0 + SC*c, (time.perf_counter() - tc) * 1e3))
    if prog is not None and (c % 2 == 0 or c == nsc - 1):
      prog(SC * (c + 1), N)
    if log is not None and c == nsc - 1:
      log("prefill_sc_chunk", k=c + 1, n=nsc, t=round(time.perf_counter() - t0, 1))
  # head on the last row when the tail is empty
  if r == 0:
    xr = E._pfsc_last.offset(offset=(SC - 1) * 5120 * 4, size=5120 * 4)
    P.win_up("pos_slot", 0, np.array([pos0 + N - 1], dtype=np.int32))
    pr["pfk_n16"](xr, W[("onw", 0)], d["xh"], global_size=(1, 1, 1), local_size=LS)
    pr["head8"](W[("head", 0)], d["xh"], d["logits"], global_size=(VOCAB // 8, 1, 1), local_size=LS)
    pr["h_argmax"](d["logits"], d["tok_slot"], d["pos_slot"], d["tok_hist"], global_size=(1, 1, 1), local_size=LS, wait=True)
    if os.getenv("PF_DFILL", "1") == "1":
      hlast = P.down_at("REC1", 16*5120*4, 5120, np.float32)
      P.win_up("hd_d1", 0, hlast)
      P._keep.clear()
  else:
    if log is not None: log("prefill_sc_tail_m32", n=r)
    prefill_batch(E, G, ids[SC*nsc:], prog=(lambda k, n: prog(SC*nsc + k, N)) if prog is not None else None, log=None)
  dev.synchronize()
  return time.perf_counter() - t0


# ====================== P15: M=64 TRUNK MACHINERY ======================

def _sc_entry(n):
  return n.split("_100k")[0] if n.endswith("_100k") else n

def ensure64(E):
  """P15: build the 64-row trunk plan (bit-identical twins; W7-covered
  classes run the m64 single launch, the rest classic m32 x2 on 32-row
  views). Coexists with the M32 plan (the tail path)."""
  if getattr(E, "_pf_plan64", None) is not None:
    return
  ensure(E)   # cubins + M32 scratch + W7 (the m64 twins read the same r7 tensors)
  P, d, W, pr = E.P, E.P.d, E.W,E.pr
  W7 = getattr(E, "_pf_W7", {})
  W5 = getattr(E, "_pf_W5", {})
  _m64l = getattr(E, "_pf_m64loaded", None)
  if _m64l is None: _m64l = E._pf_m64loaded = set()   # Bufs has no __contains__
  for n in M64_CUBINS:
    if n in _m64l: continue
    _m64l.add(n)
    lib = open(f"{BASE}/{n}.cubin", "rb").read()
    pr[n] = NVProgram(dev, TinyELF(lib=lib, name=_sc_entry(n), target=dev.renderer.target, signature=tuple()))
  if ABW:
    if "pfk_ab16w" not in _m64l:
      _m64l.add("pfk_ab16w")
      lib = open(f"{BASE}/pfk_ab16w.cubin", "rb").read()
      pr["pfk_ab16w"] = NVProgram(dev, TinyELF(lib=lib, name="pfk_ab16w", target=dev.renderer.target, signature=tuple()))
  if M64QKV:
    for n in ["pfg3m_attnqkvi3_r7_m64_nw8k128", "pfg3m_attnqkvq6_r7_m64_nw8k128"]:
      if n in _m64l: continue
      _m64l.add(n)
      lib = open(f"{BASE}/{n}.cubin", "rb").read()
      pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
  if RING4:   # R2d: the ring-4 gdnqg twin (loaded only when the plan uses it)
    for n in [GQG] + (["pfg3_iq3s_m64_nw8k128"] if OP64 else []):
      if n in _m64l: continue
      _m64l.add(n)
      lib = open(f"{BASE}/{n}.cubin", "rb").read()
      pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
  if FFNSPLIT:
    for n in ["pfg3_fgp_r7_m64_nw8k128", "pfg3_fup_r7_m64_nw8k128", "pfk_smul64"]:
      if n in _m64l: continue
      _m64l.add(n)
      lib = open(f"{BASE}/{n}.cubin", "rb").read()
      pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
  if SCANC:
    _wy = ["pfca_c32_nc2_nw16", "pfcb_c32_nc2_nw8", "pfcz_c32_nc2_nw8"] if SCANC_N2 else           ["pfca_c32_nc1_nw16", "pfcb_c32_nc1_nw8", "pfcz_c32_nc1_nw8"]
    for n in _wy:
      if n in _m64l: continue
      _m64l.add(n)
      lib = open(f"{BASE}/{n}.cubin", "rb").read()
      pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
  if ATTNW:
    for n, ent in [("pfaw_w64h_s13_100k", "pfaw64h"), ("pfaw_w64_s13_100k", "pfaw64"),
                   ("pfcw64h_s13", "pfcw64h"), ("pfcw64_s13", "pfcw64")]:
      lib = open(f"{BASE}/{n}.cubin", "rb").read()
      pr[n] = NVProgram(dev, TinyELF(lib=lib, name=ent, target=dev.renderer.target, signature=tuple()))
    for nm, nb in [("pmW", 19968*4), ("psW", 19968*4), ("pAW", 19968*256*4)]:
      P.poison(nm, nb, np.float32, 7.7e31)
    print("[pf17] wide attn cubins live (w64h<8k / w64>=8k)", flush=True)
  M = 64
  for nm, nb, dt, v in [
      ("xA64", M*5120*4, np.float32, 7.7e31), ("xB64", M*5120*4, np.float32, 7.7e31),
      ("xh64", M*5120*2, np.float16, 7.7), ("hh64", M*5120*4, np.float32, 7.7e31),
      ("hhx64", M*5120*2, np.float16, 7.7), ("attn_out64", M*5120*2, np.float16, 7.7),
      ("qkv64", M*10240*2, np.float16, 7.7), ("gate64", M*6144*2, np.float16, 7.7),
      ("z64", M*6144*2, np.float16, 7.7), ("gact64", M*17408*2, np.float16, 7.7),
      ("araw64", M*48*4, np.float32, 7.7e31), ("braw64", M*48*4, np.float32, 7.7e31),
      ("qrow64", M*12288*2, np.float16, 7.7), ("krow64", M*1024*2, np.float16, 7.7),
      ("vrow64", M*1024*2, np.float16, 7.7), ("qw64", M*24*256*2, np.float16, 7.7),
      ("ao64", M*6144*2, np.float16, 7.7)]:
    P.poison(nm, nb, dt, v)
  if FFNSPLIT:
    P.poison("ag64", M*17408*2, np.float16, 7.7)
    P.poison("au64", M*17408*2, np.float16, 7.7)
  if W4FFN:
    for nm, nb, dt, v in [("xq64", 64*5120, np.int8, -19), ("sx64", 64*40*4, np.float32, 7.7e31),
                          ("rs64", 64*40*4, np.int32, 0x5a5a5a5a)]:
      P.poison(nm, nb, dt, v)
    from pack_w4 import grid_f32 as _g4f
    _lut8 = np.clip(np.rint(_g4f().reshape(-1) / 4.0507), 0, 15).astype(np.int8)
    P.up("w4lut8", _lut8)
    for n, ent in [("p8q8x_nw8k128", "p8q8x"), ("p8w4ffn7_nw8k128", "p8w4ffn7")]:
      if n not in _m64l:
        _m64l.add(n)
        lib = open(f"{BASE}/{n}.cubin", "rb").read()
        pr[n] = NVProgram(dev, TinyELF(lib=lib, name=ent, target=dev.renderer.target, signature=tuple()))
    print("[w4] v2 cubins live (packed7-reading IMMA, zero new W planes)", flush=True)
  P.up("ids64", np.zeros(M, dtype=np.int32))
  P.up("pos_arr64", np.zeros(1, dtype=np.int32))     # pre64: [pos0]
  P.up("pos_w64", np.zeros(4, dtype=np.int32))       # 4x16-row attention windows
  dev.synchronize(); P._keep.clear()
  if SCANC:
    # R2: the WY-C32 chunked scan (P7C machinery) in the M64 trunk: 2 C=32
    # sub-chunks per 64-row chunk; in-graph-legal smem (pfca 19KB / pfcb 32KB).
    HC32 = 125712   # scratch bytes per (head, chunk) at C=32 (HC_BYTES anchor)
    P.poison("scscr64", 48 * 2 * HC32, np.uint8, 0xAB)
    P.poison("sco64", 64 * 6144 * 4, np.float32, 7.7e31)
    from engine0 import parse_gguf, read_raw
    ds_, infos_ = parse_gguf()
    wp = np.zeros(48 * 47200, dtype=np.float32)
    for jj, i in enumerate(E.gdn_idx):
      pre = f"blk.{i}."
      b = jj * 47200
      wp[b:b+40960] = np.frombuffer(read_raw(infos_[pre+"ssm_conv1d.weight"], ds_), dtype="<f4").reshape(-1)
      wp[b+40960:b+41008] = np.frombuffer(read_raw(infos_[pre+"ssm_dt.bias"], ds_), dtype="<f4")
      wp[b+41008:b+41056] = np.frombuffer(read_raw(infos_[pre+"ssm_a"], ds_), dtype="<f4")
      wp[b+41056:b+41184] = np.frombuffer(read_raw(infos_[pre+"ssm_norm.weight"], ds_), dtype="<f4")
    P.up("scwp", wp); dev.synchronize(); P._keep.clear()
    print("[r2] WY-C32 scan plane ready (48 blocks, 9.0MB)", flush=True)
  plan = []
  def A(p, *a, g=1, ls=LS):
    plan.append((p, a, g, ls))
  def V(nm, off, sz):   # 32-row view of a 64-row buffer
    return d[nm].offset(offset=off, size=sz)
  A(pr["pfk_emb16"], W[("emb", 0)], d["grid512"], d["ids64"], d["xA64"], g=64)
  cur = 0
  for i in range(64):
    xin = d["xA64"] if cur == 0 else d["xB64"]
    xout = d["xB64"] if cur == 0 else d["xA64"]
    if i in E.qtypes:
      A(pr["pfk_n16"], xin, W[("nw1", i)], d["xh64"], g=64)
      # attn qkv: m32 x2 (the m64 twin stays QUARANTINED -- P7E4 in-plan corrupt)
      _r7qkv = ("k", i) in W7 and (E.qtypes[i] == 14 or ("q", i) in W7)
      if M64QKV and _r7qkv:
        # R2b rung-6 retry: the m64 twin in ONE launch (grid 224 x 256thr, the
        # P7E4 wiring class; standalone P7B bit-identical, P7E4 in-plan corrupt
        # was SC-path launch-history-rooted)
        mq64 = "pfg3m_attnqkvq6_r7_m64_nw8k128" if E.qtypes[i] == 14 else "pfg3m_attnqkvi3_r7_m64_nw8k128"
        qw_ = W[("q", i)] if E.qtypes[i] == 14 else W7[("q", i)]
        A(pr[mq64], qw_, W7[("k", i)], W[("v", i)], d["gridf"], d["xh64"],
          d["qrow64"], d["krow64"], d["vrow64"], g=224, ls=(256, 1, 1))
      else:
        if _r7qkv:
          mqn = "pfg3m_attnqkvq6_r7_m32_nw8k128" if E.qtypes[i] == 14 else "pfg3m_attnqkvi3_r7_m32_nw8k128"
          qw_ = W[("q", i)] if E.qtypes[i] == 14 else W7[("q", i)]
          kk_ = W7[("k", i)]
        else:
          mqn = "pfg2_attnqkvq6_m32_hm_nw8k128" if E.qtypes[i] == 14 else "pfg2_attnqkvi3_m32_hm_nw8k128"
          qw_, kk_ = W[("q", i)], W[("k", i)]
        for p in range(2):
          A(pr[mqn], qw_, kk_, W[("v", i)], d["gridf"], V("xh64", p*32*5120*2, 32*5120*2),
            V("qrow64", p*32*12288*2, 32*12288*2), V("krow64", p*32*1024*2, 32*1024*2),
            V("vrow64", p*32*1024*2, 32*1024*2), g=224, ls=LS)
      if M64_PRE32:
        for p in range(2):
          A(pr["pfk_pre32_100k"], V("qrow64", p*32*12288*2, 32*12288*2), V("krow64", p*32*1024*2, 32*1024*2),
            V("vrow64", p*32*1024*2, 32*1024*2), W[("qnw", i)], W[("knw", i)], d["freqs"],
            d[f"kv{i}"], d[f"sc{i}"], d["pos_w64"].offset(offset=p*2*4, size=4),
            V("qw64", p*32*6144*2, 32*6144*2), g=24)
      else:
        A(pr["pfk_pre64_100k"], d["qrow64"], d["krow64"], d["vrow64"], W[("qnw", i)], W[("knw", i)],
          d["freqs"], d[f"kv{i}"], d[f"sc{i}"], d["pos_arr64"], d["qw64"], g=24)
      if ATTNW:
        # P17 wide-M attention: ONE 64-row window per layer (BIT-IDENTICAL per
        # row to the 4x t32 windows — pf17_probe corr nz=0/393216 det x2; the
        # causal mask makes extra KV rows contribute exactly 0 to earlier rows).
        # Low arm w64h (HRP=1, 312 CTAs — the low-pos underfill fix); the HYB
        # plan26 arm swaps to w64 (2 head-rows/CTA, 156 CTAs) at pos >= ATTN_THR.
        A(pr["pfaw_w64h_s13_100k"], d[f"kv{i}"], d[f"sc{i}"], d["qw64"], d["pos_w64"],
          d["pmW"], d["psW"], d["pAW"], g=4*S13*6, ls=(512, 1, 1))
        A(pr["pfcw64h_s13"], d["pmW"], d["psW"], d["pAW"], d["qrow64"], d["ao64"], g=24)
      else:
        # 4x t32 attention windows (per-window interleave: the shared pm/ps/pA
        # scratch is consumed by its combine before the next window writes it)
        for h4 in range(4):
          qb = d["qw64"].offset(offset=h4*16*6144*2, size=16*6144*2)
          ab = d["ao64"].offset(offset=h4*16*6144*2, size=16*6144*2)
          rb = d["qrow64"].offset(offset=h4*16*12288*2, size=16*12288*2)
          pb = d["pos_w64"].offset(offset=h4*4, size=4)
          A(pr["pfa32c_t32_s13_100k"], d[f"kv{i}"], d[f"sc{i}"], qb, pb, d["pmA"], d["psA"], d["pAA"],
            g=4*S13*3, ls=(512, 1, 1))
          A(pr["pfc16t_s13"], d["pmA"], d["psA"], d["pAA"], rb, ab, g=24)
      if OP64:   # R8: ONE m64 g=80 launch (1 M-block of 64) — the M-grid fold
        A(pr["pfg3_iq3s_m64_nw8k128"], W[("o", i)], d["grid512"], d["ao64"], d["attn_out64"], g=80, ls=LS)
      else:      # attn o-proj: classic m32 (iq3s not in packed7)
        for p in range(2):
          A(pr["pfg_iq3s_m32_hm_nw8k128"], W[("o", i)], d["grid512"],
            V("ao64", p*32*6144*2, 32*6144*2), V("attn_out64", p*32*5120*2, 32*5120*2), g=80, ls=LS)
    else:
      j = E.gdn_idx.index(i)   # R2: WY-plane block rank (0..47)
      if ABW:
        A(pr["pfk_ab16w"], xin, W[("nw1", i)], W[("alpha", i)], W[("beta", i)],
          d["xh64"], d["araw64"], d["braw64"], g=192, ls=(1024, 1, 1))
      else:
        A(pr["pfk_ab16"], xin, W[("nw1", i)], W[("alpha", i)], W[("beta", i)],
          d["xh64"], d["araw64"], d["braw64"], g=64*13)
      if ("gate", i) in W7:
        A(pr[GQG], (W5[i] if P5 else W[("qkv", i)]), W7[("gate", i)], d["gridf"], d["xh64"],
          d["qkv64"], d["gate64"], g=256, ls=(256, 1, 1))
      else:
        for p in range(2):
          A(pr["pfg2_gdnqg_m32_hm_nw16k128"], W[("qkv", i)], W[("gate", i)], d["gridf"],
            V("xh64", p*32*5120*2, 32*5120*2), V("qkv64", p*32*10240*2, 32*10240*2),
            V("gate64", p*32*6144*2, 32*6144*2), g=128, ls=(512, 1, 1))
      if SCANC:
        # R2b: WY chunked scan (kernel fixed: see R2B_WYFIX.md). Two tiers:
        # N2 (default-env 0): ONE launch triple per 64-row chunk -- pfca grid
        #   48*2 computes both C=32 sub-chunks (each (h,c) CTA reads its rows
        #   directly from qkv64; the c=1 boundary conv reads kvbuf rows like
        #   the chained tier's pfcz-roundtrip -- identical values by
        #   construction), pfcb chains c=0->c=1 in-CTA, pfcz writes conv at
        #   c==NC-1. 3 launches/chunk.
        # chained (PF_SCANC_N2=0 fallback): 2x the NC=1 triple (6 launches).
        scwpb = d["scwp"].offset(offset=j*47200*4, size=47200*4)
        if SCANC_N2:
          A(pr["pfca_c32_nc2_nw16"], scwpb, d[f"conv{i}_0"],
            d["qkv64"], d["araw64"], d["braw64"], d["scscr64"], g=48*2, ls=(512, 1, 1))
          A(pr["pfcb_c32_nc2_nw8"], d["scscr64"], d[f"rec{i}"], d["sco64"], g=192)
          A(pr["pfcz_c32_nc2_nw8"], d["sco64"], d["gate64"],
            scwpb.offset(offset=41056*4, size=6144*4), d["z64"], d["qkv64"],
            d[f"conv{i}_0"], g=48*2*4)
        else:
          for p in range(2):
            A(pr["pfca_c32_nc1_nw16"], scwpb, d[f"conv{i}_0"],
              V("qkv64", p*32*10240*2, 32*10240*2), V("araw64", p*32*48*4, 32*48*4),
              V("braw64", p*32*48*4, 32*48*4), d["scscr64"], g=48, ls=(512, 1, 1))
            A(pr["pfcb_c32_nc1_nw8"], d["scscr64"], d[f"rec{i}"],
              V("sco64", p*32*6144*4, 32*6144*4), g=192)
            A(pr["pfcz_c32_nc1_nw8"], V("sco64", p*32*6144*4, 32*6144*4),
              V("gate64", p*32*6144*2, 32*6144*2), scwpb.offset(offset=41056*4, size=6144*4),
              V("z64", p*32*6144*2, 32*6144*2), V("qkv64", p*32*10240*2, 32*10240*2),
              d[f"conv{i}_0"], g=192)
      elif M64_SCAN32:
        for p in range(2):
          A(pr["pfs32"], d[f"conv{i}_0"], d[f"rec{i}"], V("qkv64", p*32*10240*2, 32*10240*2),
            V("gate64", p*32*6144*2, 32*6144*2), W[("convw", i)], W[("dtb", i)], W[("ssma", i)],
            V("araw64", p*32*48*4, 32*48*4), V("braw64", p*32*48*4, 32*48*4),
            d["q"], d["k"], d["v"], d["core"], W[("snw", i)], V("z64", p*32*6144*2, 32*6144*2), g=48)
      else:
        A(pr["pfs64"], d[f"conv{i}_0"], d[f"rec{i}"], d["qkv64"], d["gate64"], W[("convw", i)],
          W[("dtb", i)], W[("ssma", i)], d["araw64"], d["braw64"], d["q"], d["k"], d["v"], d["core"],
          W[("snw", i)], d["z64"], g=48)
      if (not E.gdn_oq8[i]) and ("out", i) in W7:
        A(pr["pfg3_iq3o_r7_m64_nw8k128"], W7[("out", i)], d["gridf"], d["z64"], d["attn_out64"], g=80, ls=LS)
      else:
        on = "pfg_q8o_m32_hm_nw8k64" if E.gdn_oq8[i] else "pfg_iq3o_m32_hm_nw8k128"
        for p in range(2):
          A(pr[on], W[("out", i)], d["gridf"], V("z64", p*32*6144*2, 32*6144*2),
            V("attn_out64", p*32*5120*2, 32*5120*2), g=80, ls=LS)
    A(pr["pfk_hh16"], xin, d["attn_out64"], W[("nw2", i)], d["hh64"], d["hhx64"], g=64)
    if W4FFN and ("fg", i) in W7:
      A(pr["p8q8x_nw8k128"], d["hhx64"], d["xq64"], d["sx64"], d["rs64"], d["gridf"], d["gridf"], g=64, ls=LS)
      A(pr["p8w4ffn7_nw8k128"], W7[("fg", i)], W7[("fu", i)], d["w4lut8"], d["xq64"], d["sx64"],
        d["gact64"], g=272, ls=LS)
    elif ("fg", i) in W7 and FFNSPLIT:
      A(pr["pfg3_fgp_r7_m64_nw8k128"], W7[("fg", i)], d["gridf"], d["hhx64"], d["ag64"], g=272, ls=LS)
      A(pr["pfg3_fup_r7_m64_nw8k128"], W7[("fu", i)], d["gridf"], d["hhx64"], d["au64"], g=272, ls=LS)
      A(pr["pfk_smul64"], d["ag64"], d["au64"], d["gact64"], g=544, ls=LS)
    elif ("fg", i) in W7:
      A(pr["pfg3_ffn_r7_m64_nw4k128"], W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx64"],
        d["gact64"], g=544, ls=(128, 1, 1))
    else:
      for p in range(2):
        if NT32:
          A(pr["pfg_ffn_m32_nt32_hm_nw4k128"], W[("fg", i)], W[("fu", i)], d["gridf"],
            V("hhx64", p*32*5120*2, 32*5120*2), V("gact64", p*32*17408*2, 32*17408*2), g=544, ls=(128, 1, 1))
        else:
          A(pr["pfg_ffn_m32_hm_nw8k128"], W[("fg", i)], W[("fu", i)], d["gridf"],
            V("hhx64", p*32*5120*2, 32*5120*2), V("gact64", p*32*17408*2, 32*17408*2), g=272, ls=LS)
    if ("fd", i) in W7:
      A(pr["pfg3_iq3d_r7_m64_nw8k128"], W7[("fd", i)], d["gridf"], d["gact64"], d["hh64"], xout, g=80, ls=LS)
    else:
      for p in range(2):
        A(pr["pfg_iq3d_m32_res_hm_nw8k128"], W[("fd", i)], d["gridf"],
          V("gact64", p*32*17408*2, 32*17408*2), V("hh64", p*32*5120*4, 32*5120*4),
          xout.offset(offset=p*32*5120*4, size=32*5120*4), g=80, ls=LS)
    cur ^= 1
  E._pf_plan64 = plan
  E._pf_last64 = d["xA64"] if cur == 0 else d["xB64"]   # 64 blocks even -> xA64
  if HYB:
    _p13a, _p13c = pr["pfa32c_t32_s13_100k"], pr["pfc16t_s13"]
    _p26a, _p26c = pr["pfa32c_t32_s26_100k"], pr["pfc16t_s26"]
    if ATTNW:
      # P17: the high-pos arm is w64 (NOT S26 — pf17 bench: w64-S13 beats every
      # S26 shape at pos>=8k: 4.29 vs 4.63 @48k, 8.41 vs 9.07 @100k-end).
      _wha, _whc = pr["pfaw_w64h_s13_100k"], pr["pfcw64h_s13"]
      _w4a, _w4c = pr["pfaw_w64_s13_100k"], pr["pfcw64_s13"]
      E._pf_plan26_64 = [(_w4a, a, 4 * S13 * 3, ls) if p is _wha else
                         (_w4c, a, g, ls) if p is _whc else (p, a, g, ls)
                         for (p, a, g, ls) in plan]
      print("[pf17] plan26_64 = w64 arm (pos >= ATTN_THR)", flush=True)
    else:
      E._pf_plan26_64 = [(_p26a, a, 4 * 26 * 3, ls) if p is _p13a else
                         (_p26c, a, g, ls) if p is _p13c else (p, a, g, ls)
                         for (p, a, g, ls) in plan]
  dev.synchronize()
  print(f"[pf15] M64 plan ready: {len(plan)} launches/chunk (scan={'WY-C32x2' if SCANC else ('pfs32x2' if M64_SCAN32 else 'pfs64')}, "
        f"pre={'pre32x2' if M64_PRE32 else 'pre64'})", flush=True)

def _pf_dfill_seq64(E):
  """The four per-window draft-fill windows as plan tuples (M64: FIXED args
  every chunk; ring alternates per window -- 4 windows (even) -> the last
  window's rows land in REC1, SAME law as the M32 2-window chunk)."""
  d, W, pr = E.P.d, E.W,E.pr
  seq = []
  for k in range(4):
    xk = d["xA64"].offset(offset=k*16*5120*4, size=16*5120*4)
    idk = d["ids64"].offset(offset=k*16*4, size=16*4)
    pk = d["pos_w64"].offset(offset=k*4, size=4)
    w, wn = (d["REC0"], d["REC1"]) if (k & 1) == 0 else (d["REC1"], d["REC0"])
    seq.append((pr["pfk_rec16"], (xk, w, wn), 17, LS))
    seq.append((pr["pfk_emb16"], (W[("emb", 0)], d["grid512"], idk, d["e16f_d"]), 16, LS))
    seq.append((pr["pfd_dnorm16"], (d["e16f_d"], w, d["d_enw"], d["d_hnw"], d["cat16_d"]), 16, LS))
    seq.append((pr["pfg_ehd_res_hm_nw8k128"], (d["d_eh"], d["gridf"], d["cat16_d"], d["zed5k16"], d["xin_d16"]), 80, LS))
    seq.append((pr["pfk_n16"], (d["xin_d16"], d["d_nw1"], d["xh_d16"]), 16, LS))
    seq.append((pr["pfg2_dqkv_hm_nw8k128"], (d["d_q"], d["d_k"], d["d_v"], d["gridf"], d["xh_d16"],
                                             d["qrow_d16"], d["krow_d16"], d["vrow_d16"]), 224, LS))
    seq.append((pr["pfk_pre16_100k"], (d["qrow_d16"], d["krow_d16"], d["vrow_d16"], d["d_qnw"], d["d_knw"],
                                        d["freqs"], d["kv_d"], d["sc_d"], pk, d["qw16_d"]), 24, LS))
  return seq

def prefill_batch_m64(E, G, ids, prog=None, log=None, chunk_times=None, on_chunk=None):
  """P15: 64-row chunks (bit-identical twins). Same post-conditions as
  prefill_batch; the r = N % 64 tail delegates to the M32 path.
  R1: on_chunk(pos_after) fires at every 64-chunk quiescent boundary (after
  the wait) — the prompt-cache ingest hook (capture legal there: state at
  pos_after is complete and the device is quiescent)."""
  ensure64(E)
  P, d, W, pr = E.P, E.P.d, E.W,E.pr
  pos0 = int(P.down_at("pos_slot", 0, 1)[0])
  N = len(ids)
  nc = N // 64
  r = N - 64 * nc
  P.win_up("tok_hist", pos0 * 4, np.array([int(t) for t in ids], dtype=np.int32))
  t0 = time.perf_counter()
  # R7a r==0 FIX (BEFORE the chunk loop — the loop bound is nc): n%64==0
  # never has a tail and the m64-own final head (_pf_last64 row 63) is
  # UNGATED/BUGGY (argmax=0 on the api-gate n=64 prompt; the M32 head is the
  # proven one). Keep the LAST 64 tokens OUT of the M64 chunk loop and
  # delegate them to M32 exactly like the r>0 tail.
  if r == 0 and nc >= 1:
    r = 64
    nc -= 1
    if log is not None: log("prefill_batch64_r0_tail_m32", n=64)
  for c in range(nc):
    tc = time.perf_counter()
    p0 = pos0 + 64 * c
    P.win_up("ids64", 0, np.array([int(t) for t in ids[64*c:64*c+64]], dtype=np.int32))
    P.win_up("pos_arr64", 0, np.array([p0], dtype=np.int32))
    P.win_up("pos_w64", 0, np.array([p0, p0 + 16, p0 + 32, p0 + 48], dtype=np.int32))
    if os.getenv("PF_PG", "1") == "1":
      vend = _pf_submit_chunk(E, p0, which="m64")
      if os.getenv("PG_WAIT", "1") == "1":
        dev.timeline_signal.wait(vend)
    else:
      _run_plan(E._pf_plan64)
    if c == 0 and os.getenv("PF_NPROBE") == "1":
      dev.synchronize()
      def _nan(nm, shp, dt):
        try:
          a = P.down(nm, shp, dt).astype(np.float64)
          nn = int(np.isnan(a).sum())
          return f"{nm} nan={nn}/{a.size} absmax={np.nanmax(np.abs(a)) if nn < a.size else float('nan'):.3e}"
        except Exception as ex:
          return f"{nm} ERR {ex}"
      print("[r2probe] c0 " + " | ".join([
        _nan("qkv64", (64 * 10240,), np.float16), _nan("gate64", (64 * 6144,), np.float16),
        _nan("sco64", (64 * 6144,), np.float32), _nan("z64", (64 * 6144,), np.float16),
        _nan("rec0", (48 * 128 * 128,), np.float32), _nan("conv0", (4 * 10240,), np.float32),
        _nan("xA64", (64 * 5120,), np.float32), _nan("xB64", (64 * 5120,), np.float32)]), flush=True)
      P._keep.clear()
    if chunk_times is not None:
      chunk_times.append((p0, (time.perf_counter() - tc) * 1e3))
    if on_chunk is not None:
      on_chunk(pos0 + 64 * (c + 1))   # R1: quiescent 64-boundary (PG_WAIT waits the chunk)
    if prog is not None and (c % 4 == 0 or c == nc - 1):
      prog(64 * (c + 1), N)
    if log is not None and (c % 64 == 63 or c == nc - 1):
      log("prefill_batch64_chunk", k=c + 1, n=nc, t=round(time.perf_counter() - t0, 1))
  if r > 0:
    if log is not None: log("prefill_batch64_tail_m32", n=r)
    # P15 FIX: the M64 chunks track position in pos_arr64/pos_w64 and never
    # touch pos_slot -> upload the running pos BEFORE the M32 tail delegation
    # (the tail reads pos_slot as ITS pos0; a stale value re-runs the tail at
    # pos 0 and clobbers kv rows 0..31 -- the 8k-gate failure root cause).
    P.win_up("pos_slot", 0, np.array([pos0 + 64*nc], dtype=np.int32))
    _sav = _M64ON
    m64_set(False)   # P15 FIX 2: the M32 tail MUST run the M32 plan/graphs (_pf_graphs keys on the ambient flag -- with it on, the 32-token tail replayed the STALE M64 graph on ids64/pos_arr64 = deterministic garbage; the 8k-gate root cause)
    try:
      prefill_batch(E, G, ids[64*nc:], prog=(lambda k, n: prog(64*nc + k, N)) if prog is not None else None,
                    log=None, chunk_times=chunk_times)
    finally:
      m64_set(_sav)
    dev.synchronize()
    return time.perf_counter() - t0   # (the M32 tail did its own head/hlast/seed)
  xr = E._pf_last64.offset(offset=63 * 5120 * 4, size=5120 * 4)
  P.win_up("pos_slot", 0, np.array([pos0 + N - 1], dtype=np.int32))
  pr["pfk_n16"](xr, W[("onw", 0)], d["xh"], global_size=(1, 1, 1), local_size=LS)
  pr["head8"](W[("head", 0)], d["xh"], d["logits"], global_size=(VOCAB // 8, 1, 1), local_size=LS)
  pr["h_argmax"](d["logits"], d["tok_slot"], d["pos_slot"], d["tok_hist"], global_size=(1, 1, 1), local_size=LS, wait=True)
  if DFILL:
    # 4 windows/chunk (even count) -> last window rows in REC1 (same law as M32)
    hlast = P.down_at("REC1", 16*5120*4, 5120, np.float32)
    P.win_up("hd_d1", 0, hlast)
    P._keep.clear()
  dev.synchronize()
  return time.perf_counter() - t0



# ====================== R2c: M=128 TRUNK MACHINERY ======================

M128 = os.getenv("PF_M128", "0") == "1"
assert not M128 or (M64 and SCANC and SCANC_N2 and ATTNW), "PF_M128 requires the M64/WY-N2/wide-attn config"
_M128ON = M128

# ---- R2d priced scraps (each bit-identical-class, individually gated) ----
# RING4: DBUF ring-depth-4 gdnqg (r2d_ring4/r2d_ring4b: qg@512 x1.098 BIT-IDENTICAL;
#   fd x0.93/fd128 x0.79, out@160 x0.765, qkv neutral/regressed -> ADOPTED FOR gdnqg
#   ONLY; fd/out/qkv ring4 BANKED NEGATIVE).
RING4 = os.getenv("PF_RING4", "0") == "1"
assert not RING4 or M64, "PF_RING4 requires PF_M64=1 (the gdnqg r7 m64 twin)"
P5 = os.getenv("PF_P5", "0") == "1"      # R8: gdnqg qkv-seg packed5 (true-16B Q5_K units; +1.9GB VRAM)
OP64 = os.getenv("PF_OP64", "0") == "1"  # R8: attn o-proj 4xM32 -> ONE m64 g=160 M-grid fold
W4FFN = os.getenv("PF_W4A8", "0") == "1"   # P8w4: TIER-2 W4A8 fused ffn (int4 W planes + int8 acts, IMMA m16n8k32)
W4MB = float(os.getenv("PF_W4A8_MB", "6000"))  # int4-plane upload budget (MB); uncovered blocks keep packed7 fp16
assert not P5 or RING4, "PF_P5 requires PF_RING4=1 (the r7q4p5 twin)"
GQG = "pfg3m_gdnqg_r7q4p5_m64_nw8k128" if (RING4 and P5) else ("pfg3m_gdnqg_r7q4_m64_nw8k128" if RING4 else "pfg3m_gdnqg_r7_m64_nw8k128")
# QKV1: the M128 attnqkv 2-M-block consolidation — ONE g=448 launch replaces the
#   2x g=224 pair (r2d_ring4b: i3 x1.109 / q6 x1.098, BIT-IDENTICAL both).
QKV1 = os.getenv("PF_QKV1", "0") == "1"
assert not QKV1 or M128, "PF_QKV1 is an M128-trunk knob"
# PRE32X4: M128 pre = pre32 x4 (the P15 underfill law; pos_arr128 = 4 per-launch bases).
PRE32X4 = os.getenv("PF_PRE32X4", "0") == "1"
assert not PRE32X4 or M128, "PF_PRE32X4 is an M128-trunk knob"

def m128_set(on):
  global _M128ON
  _M128ON = bool(on)

# NOTE: pfaw_w128h/pfcw128h built + corr-probed (r2c_w128h_corr.py): BIT-IDENTICAL
# at pos 100224 but NONDET-WRONG at low pos (pos 64 nz=747 det-x2 False; pos 2032
# nz=8 nondet) -- the P17 w64q ROWS-extension register class. BANKED NEGATIVE; the
# M128 attention = 2x the SHIPPED w64h windows (bit-identical, zero risk).
M128_CUBINS = ["pfca_c32_nc4_nw16", "pfcb_c32_nc4_nw8", "pfcz_c32_nc4_nw8"] + \
              (["pfk_smul128"] if FFNSPLIT else [])

def ensure128(E):
  """R2c: the 128-row trunk plan. GEMMs = m64 cubins x2 M-blocks; scan =
  WY-C32 NC=4; attention = one w128h window; norms/emb g=128 (row-per-CTA);
  pre = pre64 x2. Coexists with the M64/M32 plans (the tail chain)."""
  if getattr(E, "_pf_plan128", None) is not None:
    return
  ensure64(E)   # cubins + W7 + the M64 world (the tail path)
  P, d, W, pr = E.P, E.P.d, E.W, E.pr
  W7 = getattr(E, "_pf_W7", {})
  W5 = getattr(E, "_pf_W5", {})
  _l = E._pf_m128loaded = getattr(E, "_pf_m128loaded", set())
  for n in M128_CUBINS:
    if n in _l: continue
    _l.add(n)
    lib = open(f"{BASE}/{n}.cubin", "rb").read()
    pr[n] = NVProgram(dev, TinyELF(lib=lib, name=_sc_entry(n) if n.endswith("_100k") else n,
                                   target=dev.renderer.target, signature=tuple()))
  M = 128
  for nm, nb, dt, v in [
      ("xA128", M*5120*4, np.float32, 7.7e31), ("xB128", M*5120*4, np.float32, 7.7e31),
      ("xh128", M*5120*2, np.float16, 7.7), ("hh128", M*5120*4, np.float32, 7.7e31),
      ("hhx128", M*5120*2, np.float16, 7.7), ("attn_out128", M*5120*2, np.float16, 7.7),
      ("qkv128", M*10240*2, np.float16, 7.7), ("gate128", M*6144*2, np.float16, 7.7),
      ("z128", M*6144*2, np.float16, 7.7), ("gact128", M*17408*2, np.float16, 7.7),
      ("araw128", M*48*4, np.float32, 7.7e31), ("braw128", M*48*4, np.float32, 7.7e31),
      ("qrow128", M*12288*2, np.float16, 7.7), ("krow128", M*1024*2, np.float16, 7.7),
      ("vrow128", M*1024*2, np.float16, 7.7), ("qw128", M*24*256*2, np.float16, 7.7),
      ("ao128", M*6144*2, np.float16, 7.7)]:
    P.poison(nm, nb, dt, v)
  if FFNSPLIT:
    P.poison("ag128", M*17408*2, np.float16, 7.7)
    P.poison("au128", M*17408*2, np.float16, 7.7)
  if W4FFN:
    for nm, nb, dt, v in [("xq128", 128*5120, np.int8, -19), ("sx128", 128*40*4, np.float32, 7.7e31),
                          ("rs128", 128*40*4, np.int32, 0x5a5a5a5a)]:
      P.poison(nm, nb, dt, v)
  P.up("ids128", np.zeros(M, dtype=np.int32))
  P.up("pos_arr128", np.zeros(4 if PRE32X4 else 2, dtype=np.int32))   # pre bases: [p0, p0+64] pre64x2 / [p0, p0+32, p0+64, p0+96] pre32x4
  P.up("pos_w128", np.zeros(8, dtype=np.int32))
  P.poison("scscr128", 48 * 4 * 125712, np.uint8, 0xAB)   # WY C=32 NC=4 scratch
  P.poison("sco128", M*6144*4, np.float32, 7.7e31)
  dev.synchronize(); P._keep.clear()
  plan = []
  def A(p, *a, g=1, ls=LS):
    plan.append((p, a, g, ls))
  def V(nm, off, sz):
    return d[nm].offset(offset=off, size=sz)
  A(pr["pfk_emb16"], W[("emb", 0)], d["grid512"], d["ids128"], d["xA128"], g=128)
  cur = 0
  for i in range(64):
    xin = d["xA128"] if cur == 0 else d["xB128"]
    xout = d["xB128"] if cur == 0 else d["xA128"]
    if i in E.qtypes:
      A(pr["pfk_n16"], xin, W[("nw1", i)], d["xh128"], g=128)
      _r7qkv = ("k", i) in W7 and (E.qtypes[i] == 14 or ("q", i) in W7)
      if M64QKV and _r7qkv:
        mq64 = "pfg3m_attnqkvq6_r7_m64_nw8k128" if E.qtypes[i] == 14 else "pfg3m_attnqkvi3_r7_m64_nw8k128"
        qw_ = W[("q", i)] if E.qtypes[i] == 14 else W7[("q", i)]
        if QKV1:
          # R2d scrap-2: ONE g=448 2-M-block launch (r2d_ring4b: x1.109/x1.098,
          # BIT-IDENTICAL vs 2x g=224 — the P7E4 corruptor class cleanly retried)
          A(pr[mq64], qw_, W7[("k", i)], W[("v", i)], d["gridf"], d["xh128"],
            d["qrow128"], d["krow128"], d["vrow128"], g=448, ls=(256, 1, 1))
        else:
          for p in range(2):   # 2x the R2b-shipped g=224 pattern (the 2-M-block M-grid twin = the P7E4 in-plan corruptor class)
            A(pr[mq64], qw_, W7[("k", i)], W[("v", i)], d["gridf"],
              V("xh128", p*64*5120*2, 64*5120*2), V("qrow128", p*64*12288*2, 64*12288*2),
              V("krow128", p*64*1024*2, 64*1024*2), V("vrow128", p*64*1024*2, 64*1024*2), g=224, ls=(256, 1, 1))
      else:
        for p in range(4):
          A(pr["pfg2_attnqkvq6_m32_hm_nw8k128" if E.qtypes[i] == 14 else "pfg2_attnqkvi3_m32_hm_nw8k128"],
            W[("q", i)], W[("k", i)], W[("v", i)], d["gridf"],
            V("xh128", p*32*5120*2, 32*5120*2), V("qrow128", p*32*12288*2, 32*12288*2),
            V("krow128", p*32*1024*2, 32*1024*2), V("vrow128", p*32*1024*2, 32*1024*2), g=224, ls=LS)
      if PRE32X4:
        # R2d scrap-3: pre32 x4 (the P15 underfill law) — each launch appends 32
        # rows at ITS OWN base (pos_arr128[4] = [p0, p0+32, p0+64, p0+96]; the
        # multi-launch pre base-pos law)
        for p in range(4):
          A(pr["pfk_pre32_100k"], V("qrow128", p*32*12288*2, 32*12288*2), V("krow128", p*32*1024*2, 32*1024*2),
            V("vrow128", p*32*1024*2, 32*1024*2), W[("qnw", i)], W[("knw", i)], d["freqs"],
            d[f"kv{i}"], d[f"sc{i}"], d["pos_arr128"].offset(offset=p*4, size=4),
            V("qw128", p*32*6144*2, 32*6144*2), g=24)
      else:
        for p in range(2):   # pre64 x2: half p appends at base p0 + 64p (pos_arr128[2] = [p0, p0+64])
          A(pr["pfk_pre64_100k"], V("qrow128", p*64*12288*2, 64*12288*2), V("krow128", p*64*1024*2, 64*1024*2),
            V("vrow128", p*64*1024*2, 64*1024*2), W[("qnw", i)], W[("knw", i)], d["freqs"],
            d[f"kv{i}"], d[f"sc{i}"], d["pos_arr128"].offset(offset=p*4, size=4),
            V("qw128", p*64*6144*2, 64*6144*2), g=24)
      for hp2 in range(2):   # 2x the shipped w64h windows (w128h banked negative)
        qb = d["qw128"].offset(offset=hp2*64*6144*2, size=64*6144*2)
        ab = d["ao128"].offset(offset=hp2*64*6144*2, size=64*6144*2)
        rb = d["qrow128"].offset(offset=hp2*64*12288*2, size=64*12288*2)
        pb = d["pos_w128"].offset(offset=hp2*4*4, size=4)
        A(pr["pfaw_w64h_s13_100k"], d[f"kv{i}"], d[f"sc{i}"], qb, pb,
          d["pmW"], d["psW"], d["pAW"], g=4*S13*6, ls=(512, 1, 1))
        A(pr["pfcw64h_s13"], d["pmW"], d["psW"], d["pAW"], rb, ab, g=24)
      if OP64:   # R8: ONE m64 g=160 launch (2 M-blocks) replaces 4x m32 g=80 (the qkv g=448 fold pattern)
        A(pr["pfg3_iq3s_m64_nw8k128"], W[("o", i)], d["grid512"], d["ao128"], d["attn_out128"], g=160, ls=LS)
      else:      # attn o-proj: classic m32 (iq3s not in packed7)
        for p in range(4):
          A(pr["pfg_iq3s_m32_hm_nw8k128"], W[("o", i)], d["grid512"],
            V("ao128", p*32*6144*2, 32*6144*2), V("attn_out128", p*32*5120*2, 32*5120*2), g=80, ls=LS)
    else:
      j = E.gdn_idx.index(i)
      if ABW:
        A(pr["pfk_ab16w"], xin, W[("nw1", i)], W[("alpha", i)], W[("beta", i)],
          d["xh128"], d["araw128"], d["braw128"], g=384, ls=(1024, 1, 1))
      else:
        A(pr["pfk_ab16"], xin, W[("nw1", i)], W[("alpha", i)], W[("beta", i)],
          d["xh128"], d["araw128"], d["braw128"], g=128*13)
      if ("gate", i) in W7:
        A(pr[GQG], (W5[i] if P5 else W[("qkv", i)]), W7[("gate", i)], d["gridf"], d["xh128"],
          d["qkv128"], d["gate128"], g=512, ls=(256, 1, 1))
      else:
        for p in range(4):
          A(pr["pfg2_gdnqg_m32_hm_nw16k128"], W[("qkv", i)], W[("gate", i)], d["gridf"],
            V("xh128", p*32*5120*2, 32*5120*2), V("qkv128", p*32*10240*2, 32*10240*2),
            V("gate128", p*32*6144*2, 32*6144*2), g=128, ls=(512, 1, 1))
      # WY-C32 NC=4: ONE launch triple per 128-row chunk
      scwpb = d["scwp"].offset(offset=j*47200*4, size=47200*4)
      A(pr["pfca_c32_nc4_nw16"], scwpb, d[f"conv{i}_0"],
        d["qkv128"], d["araw128"], d["braw128"], d["scscr128"], g=48*4, ls=(512, 1, 1))
      A(pr["pfcb_c32_nc4_nw8"], d["scscr128"], d[f"rec{i}"], d["sco128"], g=192)
      A(pr["pfcz_c32_nc4_nw8"], d["sco128"], d["gate128"],
        scwpb.offset(offset=41056*4, size=6144*4), d["z128"], d["qkv128"],
        d[f"conv{i}_0"], g=48*4*4)
      if (not E.gdn_oq8[i]) and ("out", i) in W7:
        A(pr["pfg3_iq3o_r7_m64_nw8k128"], W7[("out", i)], d["gridf"], d["z128"], d["attn_out128"], g=160, ls=LS)
      else:
        on = "pfg_q8o_m32_hm_nw8k64" if E.gdn_oq8[i] else "pfg_iq3o_m32_hm_nw8k128"
        for p in range(4):
          A(pr[on], W[("out", i)], d["gridf"], V("z128", p*32*6144*2, 32*6144*2),
            V("attn_out128", p*32*5120*2, 32*5120*2), g=80, ls=LS)
    A(pr["pfk_hh16"], xin, d["attn_out128"], W[("nw2", i)], d["hh128"], d["hhx128"], g=128)
    if W4FFN and ("fg", i) in W7:
      A(pr["p8q8x_nw8k128"], d["hhx128"], d["xq128"], d["sx128"], d["rs128"], d["gridf"], d["gridf"], g=128, ls=LS)
      A(pr["p8w4ffn7_nw8k128"], W7[("fg", i)], W7[("fu", i)], d["w4lut8"], d["xq128"], d["sx128"],
        d["gact128"], g=544, ls=LS)
    elif ("fg", i) in W7 and FFNSPLIT:
      A(pr["pfg3_fgp_r7_m64_nw8k128"], W7[("fg", i)], d["gridf"], d["hhx128"], d["ag128"], g=544, ls=LS)
      A(pr["pfg3_fup_r7_m64_nw8k128"], W7[("fu", i)], d["gridf"], d["hhx128"], d["au128"], g=544, ls=LS)
      A(pr["pfk_smul128"], d["ag128"], d["au128"], d["gact128"], g=1088, ls=LS)
    elif ("fg", i) in W7:
      A(pr["pfg3_ffn_r7_m64_nw4k128"], W7[("fg", i)], W7[("fu", i)], d["gridf"], d["hhx128"],
        d["gact128"], g=1088, ls=(128, 1, 1))
    else:
      for p in range(4):
        A(pr["pfg_ffn_m32_nt32_hm_nw4k128" if NT32 else "pfg_ffn_m32_hm_nw8k128"], W[("fg", i)], W[("fu", i)], d["gridf"],
          V("hhx128", p*32*5120*2, 32*5120*2), V("gact128", p*32*17408*2, 32*17408*2),
          g=544 if NT32 else 272, ls=(128, 1, 1) if NT32 else LS)
    if ("fd", i) in W7:
      A(pr["pfg3_iq3d_r7_m64_nw8k128"], W7[("fd", i)], d["gridf"], d["gact128"], d["hh128"], xout, g=160, ls=LS)
    else:
      for p in range(4):
        A(pr["pfg_iq3d_m32_res_hm_nw8k128"], W[("fd", i)], d["gridf"],
          V("gact128", p*32*17408*2, 32*17408*2), V("hh128", p*32*5120*4, 32*5120*4),
          xout.offset(offset=p*32*5120*4, size=32*5120*4), g=80, ls=LS)
    cur ^= 1
  E._pf_plan128 = plan
  E._pf_last128 = d["xA128"] if cur == 0 else d["xB128"]
  dev.synchronize()
  print(f"[r2c] M128 plan ready: {len(plan)} launches/chunk "
        f"(scan=WY-C32-NC4, attn=w128h, ffn={'split' if FFNSPLIT else 'fused'}, "
        f"pre={'pre32x4' if PRE32X4 else 'pre64x2'}, qkv={'g448' if QKV1 else '2xg224'}, gqg={'r7q4' if RING4 else 'r7'})", flush=True)

def _pf_dfill_seq128(E):
  """8 x 16-row draft-fill windows (even count -> last window rows in REC1)."""
  d, W, pr = E.P.d, E.W, E.pr
  seq = []
  for k in range(8):
    xk = d["xA128"].offset(offset=k*16*5120*4, size=16*5120*4)
    idk = d["ids128"].offset(offset=k*16*4, size=16*4)
    pk = d["pos_w128"].offset(offset=k*4, size=4)
    w, wn = (d["REC0"], d["REC1"]) if (k & 1) == 0 else (d["REC1"], d["REC0"])
    seq.append((pr["pfk_rec16"], (xk, w, wn), 17, LS))
    seq.append((pr["pfk_emb16"], (W[("emb", 0)], d["grid512"], idk, d["e16f_d"]), 16, LS))
    seq.append((pr["pfd_dnorm16"], (d["e16f_d"], w, d["d_enw"], d["d_hnw"], d["cat16_d"]), 16, LS))
    seq.append((pr["pfg_ehd_res_hm_nw8k128"], (d["d_eh"], d["gridf"], d["cat16_d"], d["zed5k16"], d["xin_d16"]), 80, LS))
    seq.append((pr["pfk_n16"], (d["xin_d16"], d["d_nw1"], d["xh_d16"]), 16, LS))
    seq.append((pr["pfg2_dqkv_hm_nw8k128"], (d["d_q"], d["d_k"], d["d_v"], d["gridf"], d["xh_d16"],
                                             d["qrow_d16"], d["krow_d16"], d["vrow_d16"]), 224, LS))
    seq.append((pr["pfk_pre16_100k"], (d["qrow_d16"], d["krow_d16"], d["vrow_d16"], d["d_qnw"], d["d_knw"],
                                        d["freqs"], d["kv_d"], d["sc_d"], pk, d["qw16_d"]), 24, LS))
  return seq

def prefill_batch_m128(E, G, ids, prog=None, log=None, chunk_times=None, on_chunk=None):
  """R2c: 128-row chunks. r = N % 128 tail delegates to the M64 path (which
  tails r%64 -> M32). Same post-conditions + the P15 tail laws (running pos
  BEFORE delegation; ambient-flag graph keying)."""
  ensure128(E)
  P, d, W, pr = E.P, E.P.d, E.W, E.pr
  pos0 = int(P.down_at("pos_slot", 0, 1)[0])
  N = len(ids)
  nc = N // 128
  r = N - 128 * nc
  P.win_up("tok_hist", pos0 * 4, np.array([int(t) for t in ids], dtype=np.int32))
  t0 = time.perf_counter()
  for c in range(nc):
    tc = time.perf_counter()
    p0 = pos0 + 128 * c
    P.win_up("ids128", 0, np.array([int(t) for t in ids[128*c:128*c+128]], dtype=np.int32))
    P.win_up("pos_arr128", 0, np.array([p0, p0 + 32, p0 + 64, p0 + 96] if PRE32X4
                                       else [p0, p0 + 64], dtype=np.int32))
    P.win_up("pos_w128", 0, np.array([p0 + 16*k for k in range(8)], dtype=np.int32))
    if os.getenv("PF_PG", "1") == "1":
      vend = _pf_submit_chunk(E, p0, which="m128")
      if os.getenv("PG_WAIT", "1") == "1":
        dev.timeline_signal.wait(vend)
    else:
      _run_plan(E._pf_plan128)
    if chunk_times is not None:
      chunk_times.append((p0, (time.perf_counter() - tc) * 1e3))
    if on_chunk is not None:
      on_chunk(pos0 + 128 * (c + 1))
    if prog is not None and (c % 2 == 0 or c == nc - 1):
      prog(128 * (c + 1), N)
    if log is not None and (c % 32 == 31 or c == nc - 1):
      log("prefill_batch128_chunk", k=c + 1, n=nc, t=round(time.perf_counter() - t0, 1))
  if r > 0:
    if log is not None: log("prefill_batch128_tail_m64", n=r)
    P.win_up("pos_slot", 0, np.array([pos0 + 128*nc], dtype=np.int32))
    _s128, _s64 = _M128ON, _M64ON
    m128_set(False); m64_set(False)   # P15 law-2: clear BOTH flags — the tail runs the
    try:                              # M32 plan AND the M32 graphs (m64 left on = the M64 graph)
      prefill_batch(E, G, ids[128*nc:], prog=(lambda k, n: prog(128*nc + k, N)) if prog is not None else None,
                    log=None, chunk_times=chunk_times)
    finally:
      m128_set(_s128); m64_set(_s64)
    dev.synchronize()
    return time.perf_counter() - t0
  xr = E._pf_last128.offset(offset=127 * 5120 * 4, size=5120 * 4)
  P.win_up("pos_slot", 0, np.array([pos0 + N - 1], dtype=np.int32))
  pr["pfk_n16"](xr, W[("onw", 0)], d["xh"], global_size=(1, 1, 1), local_size=LS)
  pr["head8"](W[("head", 0)], d["xh"], d["logits"], global_size=(VOCAB // 8, 1, 1), local_size=LS)
  pr["h_argmax"](d["logits"], d["tok_slot"], d["pos_slot"], d["tok_hist"], global_size=(1, 1, 1), local_size=LS, wait=True)
  if DFILL:
    # 8 windows/chunk (even) -> last window rows in REC1 (same law)
    hlast = P.down_at("REC1", 16*5120*4, 5120, np.float32)
    P.win_up("hd_d1", 0, hlast)
    P._keep.clear()
  dev.synchronize()
  return time.perf_counter() - t0

def prefill_batch(E, G, ids, prog=None, log=None, chunk_times=None, on_chunk=None):
  """Batched M=16 prefill of ids from the CURRENT pos_slot. Leaves the same
  post-conditions as prefill_t1 (see module docstring). Returns wall seconds.
  prog(done, total) fires per chunk; chunk_times (list) collects per-chunk ms.
  on_chunk(pos_after) (R1) only fires from the M64 chunk path (>=64 tokens)."""
  if SC >= 64 and os.getenv("PF_SUPER", "0") == "1" and len(ids) >= SC:  # P7e: OFF until the post-reference late-OOB NaN is solved (see P7DE doc)
    return prefill_batch_sc(E, G, ids, prog=prog, log=log, chunk_times=chunk_times)
  if _M128ON and M128 and len(ids) >= 128:  # R2c: the M=128 trunk (tails r%128 -> M64)
    return prefill_batch_m128(E, G, ids, prog=prog, log=log, chunk_times=chunk_times, on_chunk=on_chunk)
  if _M64ON and M32 and len(ids) >= 64:   # P15: the M=64 trunk (tails r%64 -> M32 below)
    return prefill_batch_m64(E, G, ids, prog=prog, log=log, chunk_times=chunk_times, on_chunk=on_chunk)
  ensure(E)
  P, d, W, pr = E.P, E.P.d, E.W, E.pr
  pos0 = int(P.down_at("pos_slot", 0, 1)[0])
  N = len(ids)
  if N < CH:
    if log is not None: log("prefill_batch_delegated_t1", n=N)
    return E.prefill_t1(G, ids)
  nc = N // CH
  r = N - CH * nc
  # benign tok_hist placeholder rows (T=1 writes argmax-after-p; only rows >=
  # final pos are ever read — see docstring)
  P.win_up("tok_hist", pos0 * 4, np.array([int(t) for t in ids], dtype=np.int32))
  t0 = time.perf_counter()
  plan = E._pf_plan

  def dfill_window(w, wn, xbuf, idsbuf, posslot):
    """One 16-pos draft-fill window (P4 Stage 2, verbatim; M32 calls it per
    half — the REC ring alternates per HALF: seed row = prev half row 15)."""
    pr["pfk_rec16"](xbuf, w, wn, global_size=(17, 1, 1), local_size=LS)
    pr["pfk_emb16"](W[("emb", 0)], d["grid512"], idsbuf, d["e16f_d"], global_size=(16, 1, 1), local_size=LS)
    pr["pfd_dnorm16"](d["e16f_d"], w, d["d_enw"], d["d_hnw"], d["cat16_d"], global_size=(16, 1, 1), local_size=LS)
    pr["pfg_ehd_res_hm_nw8k128"](d["d_eh"], d["gridf"], d["cat16_d"], d["zed5k16"], d["xin_d16"], global_size=(80, 1, 1), local_size=LS)
    pr["pfk_n16"](d["xin_d16"], d["d_nw1"], d["xh_d16"], global_size=(16, 1, 1), local_size=LS)
    pr["pfg2_dqkv_hm_nw8k128"](d["d_q"], d["d_k"], d["d_v"], d["gridf"], d["xh_d16"],
                               d["qrow_d16"], d["krow_d16"], d["vrow_d16"], global_size=(224, 1, 1), local_size=LS)
    pr["pfk_pre16_100k"](d["qrow_d16"], d["krow_d16"], d["vrow_d16"], d["d_qnw"], d["d_knw"], d["freqs"],
                         d["kv_d"], d["sc_d"], posslot, d["qw16_d"], global_size=(24, 1, 1), local_size=LS)

  for c in range(nc):
    tc = time.perf_counter()
    if M32:
      if N32:
        P.win_up("ids32", 0, np.array([int(t) for t in ids[CH*c:CH*c+CH]], dtype=np.int32))
      else:
        P.win_up("ids16a", 0, np.array([int(t) for t in ids[CH*c:CH*c+16]], dtype=np.int32))
        P.win_up("ids16b", 0, np.array([int(t) for t in ids[CH*c+16:CH*c+CH]], dtype=np.int32))
      P.win_up("pos_slot", 0, np.array([pos0 + CH*c], dtype=np.int32))
      P.win_up("pos_slot_b", 0, np.array([pos0 + CH*c + 16], dtype=np.int32))
    else:
      P.win_up("ids16", 0, np.array([int(t) for t in ids[16*c:16*c+16]], dtype=np.int32))
      P.win_up("pos_slot", 0, np.array([pos0 + 16*c], dtype=np.int32))
    if os.getenv("PF_PG", "1") == "1":   # P7F-1: default-ON (gates green 09-17; PF_PG=0 restores eager)
      # P7F: captured chunk (plan [+ dfill windows when M32+DFILL]) as graph
      # submits; per-chunk host work = the win_up DMAs above (timeline-chained).
      vend = _pf_submit_chunk(E, pos0 + CH * c, which="m32")
      if os.getenv("PG_WAIT", "1") == "1":
        dev.timeline_signal.wait(vend)   # chunk-end completion (matches eager sync)
      # PG_WAIT=0: pipelined -- the win_up copy-queue DMAs order themselves after
      # this chunk (they wait timeline value-1 = vend), so no ids race; the host
      # runs ~1-2 chunks ahead (b[] staging ring paces); final state guaranteed
      # by the tail/head eager launches (timeline-ordered) + end-of-call sync.
      if DFILL and not M32:
        w, wn = d["REC0"] if (c & 1) == 0 else d["REC1"], d["REC1"] if (c & 1) == 0 else d["REC0"]
        dfill_window(w, wn, d["xA16"], d["ids16"], d["pos_slot"])
    else:
      _run_plan(plan)
      if DFILL:
        if M32:
          # half A (ring k=2c: src REC0 -> dst REC1), then half B (k=2c+1)
          _ida = d["ids32"] if N32 else d["ids16a"]
          _idb = d["ids32"].offset(offset=16 * 4, size=16 * 4) if N32 else d["ids16b"]
          dfill_window(d["REC0"], d["REC1"], d["xA32"], _ida, d["pos_slot"])
          dfill_window(d["REC1"], d["REC0"], E._pf_hv["xA32b"], _idb, d["pos_slot_b"])
        else:
          w, wn = d["REC0"] if (c & 1) == 0 else d["REC1"], d["REC1"] if (c & 1) == 0 else d["REC0"]
          dfill_window(w, wn, d["xA16"], d["ids16"], d["pos_slot"])
    if chunk_times is not None:
      chunk_times.append((pos0 + CH*c, (time.perf_counter() - tc) * 1e3))
    if prog is not None and (c % 8 == 0 or c == nc - 1):
      prog(CH * (c + 1), N)
    if log is not None and (c % 128 == 127 or c == nc - 1):
      log("prefill_batch_chunk", k=c + 1, n=nc, t=round(time.perf_counter() - t0, 1))
  if r > 0:
    tail = [int(t) for t in ids[CH*nc:]]
    if log is not None: log("prefill_batch_tail_t1", n=r)
    # P0 fix: the M32 chunk graphs never advance pos_slot (it holds the LAST
    # chunk win_up = pos0 + CH*(nc-1)) — the T1 tail would re-feed at that
    # stale pos (pos 0 on a fresh world, clobbering kv rows; the serve
    # pos=n-32 signature). Upload the running pos first (the P15 m64-tail law).
    P.win_up("pos_slot", 0, np.array([pos0 + CH*nc], dtype=np.int32))
    E.prefill_t1(G, tail, log=None, prog=(lambda k, n: prog(CH*nc + k, N)) if prog is not None else None)
  if DFILL:
    # draft tail (r>0): sequential T=1 fill of the remainder, seeded from the
    # last recorded trunk hidden (documented: chains its own hd for <16 rows).
    # M32: the ring alternates per HALF (2/chunk) -> last half (odd k) leaves
    # its rows in REC1 (row 16 = the final trunk hidden).
    lastnm = "REC1" if M32 else ("REC0" if ((nc - 1) & 1) == 0 else "REC1")
    hlast = P.down_at(lastnm, 16*5120*4, 5120, np.float32)
    P.win_up("hd_d1", 0, hlast)
    P._keep.clear()
    if r > 0:
      E.fill_draft(ids[CH*nc:], start_pos=CH*nc, seed_hd=hlast, prog=None)
  if r > 0:
    pass  # (tail already delegated below)
  else:
    # head on the LAST row only: norm the last row (15 or 31) -> head8 ->
    # h_argmax at pos N-1 (advances pos_slot to pos0+N, sets tok_slot)
    xr15 = E._pf_last.offset(offset=(CH - 1) * 5120 * 4, size=5120 * 4)
    P.win_up("pos_slot", 0, np.array([pos0 + N - 1], dtype=np.int32))
    pr["pfk_n16"](xr15, W[("onw", 0)], d["xh"], global_size=(1, 1, 1), local_size=LS)
    pr["head8"](W[("head", 0)], d["xh"], d["logits"], global_size=(VOCAB // 8, 1, 1), local_size=LS)
    pr["h_argmax"](d["logits"], d["tok_slot"], d["pos_slot"], d["tok_hist"], global_size=(1, 1, 1), local_size=LS, wait=True)
  dev.synchronize()
  return time.perf_counter() - t0
