# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R6 PHASE 3 — the SERVING batch engine wrapper (engine level, no sockets).

Layers:
  r6_boot(E)             — called from test_w100k BEFORE any graph build:
                           extra cubins (M5/M10 families), the BT>RM probe-scratch
                           resize (BTMAX=10), stream-1 state banks (fixed-handle:
                           banks + resize must precede EVERY ParityGraph build).
  R6Scheduler(E)         — the graph MATRIX (solo1 / b35 / b53 / b55) + one
                           step() per cycle with per-stream deep flags (the
                           DecodeSession data-dependent mode selection, batched)
                           + the rung-4 draft-skip on repeated all-hit (5,5)
                           cycles + rebuild() for the global fence-all.

Compositions (BT sums; M-families that exist as cubins):
  solo1 = [{s:1,T:3}]           BT=3   (slot-1 solo k2; slot-0 solo is served by
                                       the canonical DecodeSession — full deep-K)
  b35   = [{s:0,T:3},{s:1,T:5}] BT=8
  b53   = [{s:0,T:5},{s:1,T:3}] BT=8
  b55   = [{s:0,T:5},{s:1,T:5}] BT=10
  (3,3) has NO M6 family -> falls back to b35 (forcing s1 k4 is output-lossless:
  greedy verification accepts the model's own argmax regardless of proposal
  source — Tier-1 per stream at every R6 rung).

DIFFERENCE vs the r6_batch.py harness builder: the T=3 (k2) dseq gains the
lookup8_nw32 overwrite so batch k2 cycles IDENTIFY with the canonical daemon's
base set at LOOKUP_K=7 (chains + lookup8; M3 probe reads cur+dr0+dr1) AND the
per-stream hit flag (emit e[9], from l_hist) stays live on k2 streams so the
deep escalation logic keeps its signal.

Laws: fixed-handle (banks+resize pre-graph), one kernel per cubin, 16B-aligned
slices, lone-graph flusher, depth-1 submit/wait per step (the re-patch race
class), ~904 in-flight (ONE graph set of ~452k submits per step at any B),
READOUT-ORDER (mode selection reads the PREVIOUS cycle's completed emits).
"""
import os, sys, time
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram, nv_wait_timeline
from gcycle import ParityGraph
from rung_manifest import assert_batch_wiring
from engine0 import dev

SKV_S = int(os.getenv("SKV_S", "256"))
QH = os.getenv("QH", "1") == "1"
KV8 = os.getenv("KV8", "1") == "1"
DR7 = bool(int(os.getenv("PF_DR7", "0")))
BTMAX = 10

# ---- M-family tables (verbatim from r6_batch.py; launch counts proven) ----
MSET = {
  3:  dict(n="k0n3", n_g=3, ab="k0ab3", ab_g=39, q5="q5g8_3", q5_g=2048,
           aq3="aq3k8v_3", aq6="aq6k8_3", aq_g=1792, ao="ao8nw32_3", ao_g=160,
           k3ao="k3aonw32_3", op3="op38nw32_3", og=160,
           hh="hh3", hh_g=3, ffn=("ffn8v3r7" if DR7 else "ffn8v_3"),
           down=("down8nw32v3r7" if DR7 else "down8nw32_3"), ffw_g=2176, dwn_g=160,
           head="head8v_3"),
  8:  dict(n="k0n8", n_g=8, ab="k0ab8", ab_g=104, q5="q5g8v8", q5_g=2048,
           aq3="aq3k8v8", aq6="aq6k8v8", aq_g=1792, ao="ao8nw32_8", ao_g=160,
           k3ao="k3aonw32_8", op3="op38nw32_8", og=160,
           hh="hh8", hh_g=8, ffn="ffn8v8r7", down="down8nw32v8r7", ffw_g=2176, dwn_g=160,
           head="head8v8"),
  10: dict(n="k0n10", n_g=10, ab="k0ab10", ab_g=130, q5="q5g8v10", q5_g=2048,
           aq3="aq3k8v10", aq6="aq6k8v10", aq_g=1792, ao="ao8nw32_10", ao_g=160,
           k3ao="k3aonw32_10", op3="op38nw32_10", og=160,
           hh="hh10", hh_g=10, ffn="ffn8v10r7", down="down8nw32v10r7", ffw_g=2176, dwn_g=160,
           head="head8v10"),
}
# T rows -> per-stream stateful cubins. lookup: T=3 uses lookup8 (canonical-base
# parity at LK=7: chains + lookup8 + M3 probe); T=5 uses lookup5 (the
# rung-3/4-proven k4 deep form).
TSET = {
  3: dict(embed="h_embed3", embed_g=3, k2s="k2s3", acc="acceptk", accsel="acceptsel", lookup="lookup8_nw32"),
  5: dict(embed="h_embed5", embed_g=1, k2s="k2s5", acc="accept5k", accsel="acceptsel5k", lookup="lookup5_nw32"),
}

def _log(*a): print("[r6s]", *a, flush=True)

def r6_boot(E):
  """All pre-graph-build batch state. Returns the boot handle (Swap factory)."""
  from mtp import RM, RBLK, CBLK, CTXK
  P = E.P
  # ---- extra cubins: the M5 (T=5 set) + M10 (BT=10 family) worlds ----
  need = ["h_embed5", "k2s5", "accept5k", "acceptsel5k", "lookup5_nw32",
          "k0n10", "k0ab10", "q5g8v10", "aq3k8v10", "aq6k8v10", "ao8nw32_10",
          "k3aonw32_10", "op38nw32_10", "hh10", "ffn8v10r7", "down8nw32v10r7", "head8v10",
          # pcache restore's cur-derivation fallback (nodes captured without
          # meta cur, e.g. legacy PF-world nodes restoring into this daemon)
          "pfk_n16"]
  for n in need:
    if n in E.pr: continue
    lib = open(f"~/tinygrad-metal/engine0/{n}.cubin", "rb").read()
    E.pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
    _log(f"extra cubin {n}")
  # TLX W4.4: batch K-suffix manifest assert (the MSET/TSET world must be
  # suffix-consistent with the shipped manifests — catches accept/lookup
  # mispairing in the composition matrix).
  assert_batch_wiring(set(E.pr.keys()),
                      {bt: [v for k, v in fam.items() if k not in ("n_g","ab_g","q5_g","aq_g","ao_g","og","hh_g","ffw_g","dwn_g")]
                       for bt, fam in MSET.items()},
                      TSET)
  # ---- BT>RM probe-scratch resize (harness boot-10/11-proven; MUST precede
  # every graph build incl. the canonical ones — the fixed-handle law) ----
  if BTMAX > RM:
    from trunk import VOCAB
    _SC = [("xA", 5120*4, np.float32), ("xB", 5120*4, np.float32),
           ("xh3", 5120*2, np.float16), ("hh3b", 5120*4, np.float32), ("hhx3", 5120*2, np.float16),
           ("qkv3", 10240*2, np.float16), ("gate3", 6144*2, np.float16),
           ("araw3", 48*4, np.float32), ("braw3", 48*4, np.float32), ("z3", 6144*2, np.float16),
           ("attn_out3", 5120*2, np.float16), ("gact3", 17408*2, np.float16),
           ("qrow3", 12288*2, np.float16), ("krow3", 1024*2, np.float16), ("vrow3", 1024*2, np.float16),
           ("ao_row3", 6144*2, np.float16), ("logits3", VOCAB*2, np.float16), ("amds", 4, np.int32),
           ("qw3", 24*256*4, np.float32), ("qw16_3", 24*256*2, np.float16),
           ("pm3", 4*SKV_S*6*4, np.float32), ("ps3", 4*SKV_S*6*4, np.float32),
           ("pA3", 4*SKV_S*6*256*4, np.float32)]
    for nm, rb, dt in _SC:
      P.d[nm] = P.alloc(f"{nm}_r6bt{BTMAX}", rb*BTMAX)
    dev.synchronize()
    _log(f"probe scratch resized to {BTMAX} rows (boot RM={RM})")
  # ---- stream-1 banks (device-zeroed via mfill — DART law: no host DMA) ----
  def alloc_bank(s):
    sz = {}
    def A(name, nbytes): sz[name] = nbytes; P.d[f"{name}_s{s}"] = P.alloc(f"{name}_s{s}", nbytes)
    A("rec4", 48*5*RBLK*4); A("conv4", 48*5*CBLK*4); A("conv5x", 48*CBLK*4)
    A("cur_slot", 4); A("pos_slot", 4); A("tok_slot", 4)
    A("tok_hist", (CTXK+256)*4); A("m_slot", 4); A("cyc_slot", 4)
    A("m_hist", (1 << 20)*4); A("l_hist", (1 << 20)*4)
    for k in range(10): A(f"dring{k}", 4)
    A("emit", 26*4); A("h_seed", 5120*4); A("dhd_seed", 5120*4)
    A("hd_d0", 5120*4); A("hd_d1", 5120*4); A("dpos1", 4); A("dpos2", 4)
    A("kv_d", 2*4*CTXK*256); A("sc_d", 2*4*CTXK*8*2)
    for i in E.attn_idx:
      A(f"kv{i}", 2*4*CTXK*256); A(f"sc{i}", 2*4*CTXK*8*2)
    dev.synchronize()
    zero = ["rec4", "conv4", "conv5x", "cur_slot", "pos_slot", "tok_slot", "m_slot", "cyc_slot",
            "m_hist", "l_hist", "emit", "h_seed", "dhd_seed", "hd_d0", "hd_d1", "dpos1", "dpos2",
            "kv_d", "sc_d", *[f"dring{k}" for k in range(2, 10)],
            *[f"kv{i}" for i in E.attn_idx], *[f"sc{i}" for i in E.attn_idx]]
    for nm in zero: E._mfill(f"{nm}_s{s}", 0, sz[nm] // 4)
    E._mfill(f"tok_hist_s{s}", -1, CTXK + 256)
    # W4.5 (kimi F4): in-vocab 0, not -1 (stale dring reads must not wild-index
    # the embed row; h_embed3/5 clamp as belt-and-braces).
    P.win_up(f"dring0_s{s}", 0, np.zeros(1, dtype=np.int32))
    P.win_up(f"dring1_s{s}", 0, np.zeros(1, dtype=np.int32))
    dev.synchronize()
    _log(f"stream-{s} banks allocated + zeroed")
  alloc_bank(1)
  SWAP_KEYS = (["rec4", "conv4", "conv5x", "cur_slot", "pos_slot", "tok_slot", "tok_hist",
                "m_slot", "cyc_slot", "m_hist", "l_hist", "emit", "h_seed", "dhd_seed",
                "hd_d0", "hd_d1", "dpos1", "dpos2", "kv_d", "sc_d"]
               + [f"dring{k}" for k in range(10)]
               + [f"kv{i}" for i in E.attn_idx] + [f"sc{i}" for i in E.attn_idx])
  class _Swap:
    """Temporarily remap P.d[canonical] -> P.d[name_s{s}] (slot 0 = no-op).
    All eager launch sites read P.d at call time (the R6 swap trick)."""
    def __init__(self, s): self.s = s
    def __enter__(self):
      self.saved = {}
      if self.s:
        for k in SWAP_KEYS:
          self.saved[k] = P.d[k]; P.d[k] = P.d[f"{k}_s{self.s}"]
      return self
    def __exit__(self, *a): P.d.update(self.saved)
  class R6Boot: pass
  b = R6Boot(); b.Swap = _Swap; b.SWAP_KEYS = SWAP_KEYS; b.BTMAX = BTMAX
  return b


# ============================ graph builder ============================
def stream_bufs(E, s):
  pre = "" if s == 0 else f"_s{s}"
  d = E.P.d
  names = ("cur_slot", "pos_slot", "tok_hist", "m_slot", "cyc_slot", "m_hist", "l_hist",
           "h_seed", "dhd_seed", "hd_d0", "hd_d1", "dpos1", "dpos2", "emit",
           "rec4", "conv4", "conv5x", "kv_d", "sc_d", *[f"dring{k}" for k in range(10)])
  b = {n: d[f"{n}{pre}"] for n in names}
  for i in E.attn_idx:
    b[f"kv{i}"] = d[f"kv{i}{pre}"]; b[f"sc{i}"] = d[f"sc{i}{pre}"]
  return b

def build_batch_graphs(E, specs):
  """specs: [{"s": slot, "T": 3|5}] — ONE fixed graph set for the composition."""
  from mtp import RBLK, CBLK
  from trunk import VOCAB
  d, W, pr = E.P.d, E.W, E.pr
  bt = sum(x["T"] for x in specs)
  assert bt in MSET, f"BT={bt} has no M-family (have {sorted(MSET)})"
  M = MSET[bt]
  st = []
  r0 = 0
  for x in specs:
    b = stream_bufs(E, x["s"])
    b.update(T=x["T"], r0=r0, s=x["s"])
    r0 += x["T"]
    st.append(b)
  # ---- draft (per stream: chains + dposadd + chains + THE LOOKUP for its T) ----
  dseq = []
  for b in st:
    dd = dict(d); dd["kv_d"] = b["kv_d"]; dd["sc_d"] = b["sc_d"]
    dseq += E._draft_entries(b["cur_slot"], b["pos_slot"], b["h_seed"], b["hd_d0"], b["dring0"], dd=dd)
    dseq.append((pr["dposadd"], (b["pos_slot"], b["dpos1"]), 1))
    dseq += E._draft_entries(b["dring0"], b["dpos1"], b["hd_d0"], b["hd_d1"], b["dring1"], dd=dd)
    lu = TSET[b["T"]]["lookup"]
    if lu == "lookup8_nw32":
      dseq.append((pr[lu], (b["tok_hist"], b["pos_slot"], b["cur_slot"], b["dring0"], b["dring1"],
                            b["dring2"], b["dring3"], b["dring4"], b["dring5"], b["dring6"],
                            b["l_hist"], b["cyc_slot"]), 1))
    else:   # lookup5: 4-proposal form
      dseq.append((pr[lu], (b["tok_hist"], b["pos_slot"], b["cur_slot"], b["dring0"], b["dring1"],
                            b["dring2"], b["dring3"], b["l_hist"], b["cyc_slot"]), 1))
  draft_g = ParityGraph(dseq, tag=f"r6D{bt}")
  # ---- probe (shared M-trunk over r0-row slices + per-stream stateful) ----
  seq = []
  for b in st:
    xav = d["xA"].offset(offset=b["r0"]*5120*4, size=b["T"]*5120*4)
    if b["T"] == 3:
      seq.append((pr["h_embed3"], (W[("emb",0)], d["grid512"], b["cur_slot"], b["dring0"], b["dring1"], xav), 3))
    elif b["T"] == 5:
      seq.append((pr["h_embed5"], (W[("emb",0)], d["grid512"], b["cur_slot"], b["dring0"], b["dring1"],
                                   b["dring2"], b["dring3"], xav), 5))
    else: raise AssertionError(f"T={b['T']} unsupported")
  cur = 0
  for i in range(64):
    xin, xout = (d["xA"] if cur == 0 else d["xB"]), (d["xB"] if cur == 0 else d["xA"])
    if i in E.qtypes:
      qkname = M["aq6"] if E.qtypes[i] == 14 else M["aq3"]
      a = [(pr[M["n"]], (xin, W[("nw1",i)], d["xh3"]), M["n_g"]),
           (pr[qkname], (W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh3"], d["qrow3"], d["krow3"], d["vrow3"]), M["aq_g"])]
      for b in st:
        sca = (b[f"sc{i}"],) if KV8 else ()
        qsl = d["qrow3"].offset(offset=b["r0"]*12288*2, size=b["T"]*12288*2)
        ksl = d["krow3"].offset(offset=b["r0"]*1024*2, size=b["T"]*1024*2)
        vsl = d["vrow3"].offset(offset=b["r0"]*1024*2, size=b["T"]*1024*2)
        pmb = d["pm3"].offset(offset=b["r0"]*4*SKV_S*6*4, size=b["T"]*4*SKV_S*6*4)
        psb = d["ps3"].offset(offset=b["r0"]*4*SKV_S*6*4, size=b["T"]*4*SKV_S*6*4)
        pab = d["pA3"].offset(offset=b["r0"]*4*SKV_S*6*256*4, size=b["T"]*4*SKV_S*6*256*4)
        aosl = d["ao_row3"].offset(offset=b["r0"]*6144*2, size=b["T"]*6144*2)
        a += [(pr[f"spk_pre{b['T']}"], (qsl, ksl, vsl, W[("qnw",i)], W[("knw",i)], d["freqs"], b[f"kv{i}"], *sca, b["pos_slot"], d["qw3"], *((d["qw16_3"],) if QH else ())), 24),
              (pr[f"spk_a{b['T']}"], (b[f"kv{i}"], *sca, *((d["qw16_3"],) if QH else d["qw3"]), b["pos_slot"], pmb, psb, pab), 4*SKV_S),
              (pr[f"spk_c{b['T']}"], (pmb, psb, pab, qsl, aosl), 24)]
      a += [(pr[M["ao"]], (W[("o",i)], d["grid512"], d["ao_row3"], d["attn_out3"]), M["ao_g"]),
            (pr[M["hh"]], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), M["hh_g"]),
            (pr[M["ffn"]], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), M["ffw_g"]),
            (pr[M["down"]], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), M["dwn_g"])]
    else:
      gi = E.gdn_idx.index(i)
      a = [(pr[M["ab"]], (xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh3"], d["araw3"], d["braw3"]), M["ab_g"]),
           (pr[M["q5"]], (W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh3"], d["qkv3"], d["gate3"]), M["q5_g"])]
      for b in st:
        conv_b = b["conv4"].offset(offset=gi*5*CBLK*4, size=5*CBLK*4)
        rec_b = b["rec4"].offset(offset=gi*5*RBLK*4, size=5*RBLK*4)
        qksl = d["qkv3"].offset(offset=b["r0"]*10240*2, size=b["T"]*10240*2)
        gtsl = d["gate3"].offset(offset=b["r0"]*6144*2, size=b["T"]*6144*2)
        arsl = d["araw3"].offset(offset=b["r0"]*48*4, size=b["T"]*48*4)
        brsl = d["braw3"].offset(offset=b["r0"]*48*4, size=b["T"]*48*4)
        zsl = d["z3"].offset(offset=b["r0"]*6144*2, size=b["T"]*6144*2)
        base_args = (qksl, gtsl, W[("convw",i)], W[("dtb",i)], W[("ssma",i)], arsl, brsl,
                     d["q"], d["k"], d["v"], d["core"], W[("snw",i)], zsl)
        if b["T"] == 3:
          a.append((pr["k2s3"], (conv_b, rec_b, *base_args), 48))
        else:
          c5x = b["conv5x"].offset(offset=gi*CBLK*4, size=CBLK*4)
          a.append((pr["k2s5"], (conv_b, rec_b, c5x, *base_args), 48))
      if E.gdn_oq8[i]:
        a.append((pr[M["k3ao"]], (W[("out",i)], d["z3"], d["attn_out3"]), M["og"]))
      else:
        a.append((pr[M["op3"]], (W[("out",i)], d["gridf"], d["z3"], d["attn_out3"]), M["og"]))
      a += [(pr[M["hh"]], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), M["hh_g"]),
            (pr[M["ffn"]], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), M["ffw_g"]),
            (pr[M["down"]], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), M["dwn_g"])]
    seq += a
    cur ^= 1
  seq.append((pr[M["n"]], (d["xA"], W[("onw",0)], d["xh3"]), M["n_g"]))
  seq.append((pr[M["head"]], (W[("head",0)], d["xh3"], d["logits3"]), VOCAB//8))
  seq.append((pr["amx3"], (d["logits3"], d["amds"]), bt))
  probe_g = ParityGraph(seq, tag=f"r6P{bt}")
  # ---- accept (per stream) ----
  aseq = []
  for b in st:
    asl = d["amds"].offset(offset=b["r0"]*4, size=b["T"]*4)
    xsl = d["xA"].offset(offset=b["r0"]*5120*4, size=b["T"]*5120*4)
    if b["T"] == 3:
      aseq.append((pr["acceptk"], (asl, b["dring0"], b["dring1"], xsl, b["m_slot"], b["m_hist"],
                                   b["cyc_slot"], b["pos_slot"], b["cur_slot"], b["tok_hist"], b["h_seed"],
                                   b["hd_d0"], b["hd_d1"], b["dhd_seed"], b["emit"], b["l_hist"]), 1))
      aseq.append((pr["acceptsel"], (b["rec4"], b["conv4"], b["m_slot"]), 48))
    else:
      aseq.append((pr["accept5k"], (asl, b["dring0"], b["dring1"], b["dring2"], b["dring3"], xsl,
                                    b["m_slot"], b["m_hist"], b["cyc_slot"], b["pos_slot"], b["cur_slot"],
                                    b["tok_hist"], b["h_seed"], b["hd_d0"], b["hd_d1"], b["dhd_seed"],
                                    b["emit"], b["l_hist"]), 1))
      aseq.append((pr["acceptsel5k"], (b["rec4"], b["conv4"], b["m_slot"], b["conv5x"]), 48))
  accept_g = ParityGraph(aseq, tag=f"r6A{bt}")
  flush_g = ParityGraph([(pr["dposadd"], (d["fillpos"], d["dpos1"]), 1)], tag=f"r6F{bt}")
  # rung-4 draft-skip: the lookup-only draft graph. LEGAL ONLY when every
  # stream in the composition is k4 (T=5) AND each one's PREVIOUS lookup hit —
  # the scheduler enforces both (prev composition == this one, all hits >= 9).
  lu_only = [e for e in dseq if str(e[0].name).startswith("lookup")]
  all_k4 = all(b["T"] == 5 for b in st)
  draft_lu_g = ParityGraph(lu_only, tag=f"r6DL{bt}") if (lu_only and all_k4) else None
  _log(f"r6 graphs (BT={bt}): draft {len(dseq)}k, probe {len(seq)}k, accept {len(aseq)}k"
       f"{f', draft_lu {len(lu_only)}k' if draft_lu_g else ''}")
  return (draft_g, probe_g, accept_g, flush_g), st, draft_lu_g


class R6Scheduler:
  """The serving batch step. step(slots, deeps) submits ONE composition's
  4-graph chain (depth-1: waited every step — the in-flight/re-patch laws),
  returns {slot: emit-dict}. The `prev` timeline chain persists across
  compositions (continuous chain, no begin needed on composition switch);
  begin() re-anchors after eager work or a DecodeSession interleave.
  Slot-0 solo NEVER comes here (serve routes it to the canonical DecodeSession,
  which has full deep-K); solo slot-1 runs the (1,3) k2 graph."""
  COMPS = {
    (1, 3):         [{"s": 1, "T": 3}],
    (0, 3, 1, 5):   [{"s": 0, "T": 3}, {"s": 1, "T": 5}],
    (0, 5, 1, 3):   [{"s": 0, "T": 5}, {"s": 1, "T": 3}],
    (0, 5, 1, 5):   [{"s": 0, "T": 5}, {"s": 1, "T": 5}],
  }
  def __init__(self, E):
    self.E = E
    self.prev = None
    self.g = {}          # comp key -> (graphs, st, draft_lu_g)
    self.lu_key = None   # composition eligible for next-cycle draft-skip
    self.rebuild()
  def rebuild(self):
    self.g = {key: build_batch_graphs(self.E, spec) for key, spec in self.COMPS.items()}
    self.lu_key = None
    self.prev = None

  _KEYS = None   # filled lazily (insertion order of COMPS)
  def rebuild_one(self, i):
    """ROTATED fence support: rebuild a single composition (leak-bounded
    cadence — see serve._fence_all_rebuild)."""
    if self._KEYS is None: self._KEYS = list(self.COMPS.keys())
    key = self._KEYS[i % len(self._KEYS)]
    self.g[key] = build_batch_graphs(self.E, self.COMPS[key])
    self.lu_key = None
    self.prev = None
  def begin(self):
    self.prev = dev.timeline_value - 1
    self.lu_key = None
  def _comp_key(self, slots, deeps):
    if len(slots) == 1:
      return (slots[0], 3)                       # solo slot-1: k2 (no solo M5)
    t0 = 5 if deeps.get(0) else 3
    t1 = 5 if deeps.get(1) else 3
    if (t0, t1) == (3, 3): t1 = 5                # no M6 family: force s1 k4 (lossless)
    return (0, t0, 1, t1)
  def step(self, slots, deeps):
    E = self.E
    if self.prev is None: self.begin()
    key = self._comp_key(slots, deeps)
    graphs, st, draft_lu_g = self.g[key]
    draft_g, probe_g, accept_g, flush_g = graphs
    # rung-4 draft-skip: ONLY on an all-k4 composition whose PREVIOUS cycle was
    # the SAME composition with every lookup hit (the dring-stale safety law).
    if draft_lu_g is not None and self.lu_key == key:
      draft_g = draft_lu_g
    prev = self.prev
    vd = dev.next_timeline(); draft_g.submit(prev, vd)
    vp = dev.next_timeline(); probe_g.submit(vd, vp)
    va = dev.next_timeline(); accept_g.submit(vp, va)
    vf = dev.next_timeline(); flush_g.submit(va, vf)
    self.prev = vf
    nv_wait_timeline(dev, vf, what="R6Scheduler.step")   # W4.2/V-50 deadline wait
    res = {}; hits_out = {}
    for b in st:
      nm = "emit" if b["s"] == 0 else f"emit_s{b['s']}"
      e = E.P.down_at(nm, 0, 26, np.int32)
      m = int(e[1])
      hits_out[b["s"]] = int(e[9])
      res[b["s"]] = {"pos_new": int(e[0]), "m": m, "stop": int(e[7]), "cycle": int(e[8]), "hit": int(e[9]),
                     "tokens": [int(t) for t in e[2:3+m]][:m+1]}
    self.lu_key = key if (draft_lu_g is not None and all(h >= 9 for h in hits_out.values())) else None
    return res
