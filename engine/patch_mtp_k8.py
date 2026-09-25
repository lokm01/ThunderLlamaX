# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7a K=8 wiring: mtp.py LOOKUP_K=8 support (RM=9, M9 set, dring7, spk9,
_probe9_seq, accept9k graphs, 20-word emit parse)."""
import sys
BASE = "~/tinygrad-metal/engine0"
s = open(f"{BASE}/mtp.py").read()

def rep(a, b, label, n=1):
  global s
  c = s.count(a)
  assert c == n, f"{label}: found {c} != {n}"
  s = s.replace(a, b)
  print(f"[patch] {label:52s} ok")

# 1) LOOKUP_K domain + RM
rep('assert LOOKUP_K in (0, 4, 5, 6, 7), "LOOKUP_K: 0 (off), 4/5/6/7 (deep K, R5/R5d)"',
    'assert LOOKUP_K in (0, 4, 5, 6, 7, 8), "LOOKUP_K: 0 (off), 4/5/6/7/8 (deep K, R5/R5d/R7a)"',
    "LOOKUP_K domain")
rep('  RM = {4: 5, 5: 6, 6: 7, 7: 8}[LOOKUP_K]',
    '  RM = {4: 5, 5: 6, 6: 7, 7: 8, 8: 9}[LOOKUP_K]',
    "RM map K8")

# 2) M9_CUBINS after M8_CUBINS
rep('''M8_CUBINS = ["h_embed8","k0n8","k0ab8","q5g8v8","k2s8","op38nw32_8","k3aonw32_8","ao8nw32_8",
             "hh8","ffn8v8","down8nw32_8","aq3k8v8","aq6k8v8","head8v8","lookup8_nw32","acceptk","accept8k","acceptsel8k"]''',
    '''M8_CUBINS = ["h_embed8","k0n8","k0ab8","q5g8v8","k2s8","op38nw32_8","k3aonw32_8","ao8nw32_8",
             "hh8","ffn8v8","down8nw32_8","aq3k8v8","aq6k8v8","head8v8","lookup8_nw32","acceptk","accept8k","acceptsel8k"]
M9_CUBINS = ["h_embed9","k0n9","k0ab9","q5g8v9","k2s9","op38nw32_9","k3aonw32_9","ao8nw32_9",
             "hh9","ffn8v9r7","down8nw32v9r7","aq3k8v9","aq6k8v9","head8v9","lookup9_nw32","acceptk","accept9k","acceptsel9k"]''',
    "M9_CUBINS")

# 3) cubin load loop
rep(' + (M8_CUBINS if LOOKUP_K == 7 else [])',
    ' + (M8_CUBINS if LOOKUP_K == 7 else []) + (M9_CUBINS if LOOKUP_K == 8 else [])',
    "cubin load loop")

# 4) scratch alloc K=8
rep('''    if LOOKUP_K == 7:
      # R5d K=7: k2s8 scratch — conv5x/6x/7x/8x (t=4..7 windows) + rec6x/7x/8x (t=5..7).''',
    '''    if LOOKUP_K == 8:
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
    if LOOKUP_K == 7:
      # R5d K=7: k2s8 scratch — conv5x/6x/7x/8x (t=4..7 windows) + rec6x/7x/8x (t=5..7).''',
    "scratch alloc K8")

# 5) dring7 in the up-list + reset block
rep('"dring4", "dring5", "dring6", "m_slot", "cyc_slot", "dpos1", "dpos2", "fillpos", "dtokf"):',
    '"dring4", "dring5", "dring6", "dring7", "m_slot", "cyc_slot", "dpos1", "dpos2", "fillpos", "dtokf"):',
    "dring7 up-list")
rep('    P.up("dring6", np.full(1, 0, dtype=np.int32))',
    '    P.up("dring6", np.full(1, 0, dtype=np.int32))\n    P.up("dring7", np.full(1, 0, dtype=np.int32))',
    "dring7 reset up")

# 6) spk9 loads
rep('''      if LOOKUP_K == 7:
        # R5d: the DEEP (ROWS=8) attention set (RMAX=48, RP=48 — NO padding rows, MAXOWN=2)
        for key, n in (("spk_pre8", "spk_pre8qh_100k"), ("spk_a8", "spk_g4nw32hm8_100k"), ("spk_c8", "spk_c8g_100k")):''',
    '''      if LOOKUP_K == 8:
        # R7a: the DEEP (ROWS=9) attention set (RMAX=54, RP=64 — 10 pad rows, MAXOWN=2)
        for key, n in (("spk_pre9", "spk_pre9qh_100k"), ("spk_a9", "spk_g4nw32hm9_100k"), ("spk_c9", "spk_c9g_100k")):
          lib = open(f"{BASE}/{n}.cubin", "rb").read()
          self.pr[key] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
      if LOOKUP_K == 7:
        # R5d: the DEEP (ROWS=8) attention set (RMAX=48, RP=48 — NO padding rows, MAXOWN=2)
        for key, n in (("spk_pre8", "spk_pre8qh_100k"), ("spk_a8", "spk_g4nw32hm8_100k"), ("spk_c8", "spk_c8g_100k")):''',
    "spk9 loads")

# 7) lookup9 in dseq
rep('''    elif LOOKUP_K == 7:
      # R5d: 7-proposal overwrite (scan range i in [0, pos-15] — the deep-K law at K=7).
      dseq += [(pr["lookup8_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"],
                                     d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["l_hist"], d["cyc_slot"]), 1)]''',
    '''    elif LOOKUP_K == 7:
      # R5d: 7-proposal overwrite (scan range i in [0, pos-15] — the deep-K law at K=7).
      dseq += [(pr["lookup8_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"],
                                     d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["l_hist"], d["cyc_slot"]), 1)]
    elif LOOKUP_K == 8:
      # R7a: 8-proposal overwrite (scan range i in [0, pos-17] — the deep-K law at K=8).
      dseq += [(pr["lookup9_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"],
                                     d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["dring7"], d["l_hist"], d["cyc_slot"]), 1)]''',
    "lookup9 dseq")

# 8) graphs6 seq selector + accept9 branch
rep('      p6seq = self._probe6_seq() if LOOKUP_K == 5 else (self._probe7_seq() if LOOKUP_K == 6 else self._probe8_seq())',
    '      p6seq = self._probe6_seq() if LOOKUP_K == 5 else (self._probe7_seq() if LOOKUP_K == 6 else (self._probe9_seq() if LOOKUP_K == 8 else self._probe8_seq()))',
    "p6seq selector")
rep('''      else:
        aseqN = [(pr["accept8k"], (d["amds"], d["dring0"], d["dring1"], d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["xA"],
                                   d["m_slot"], d["m_hist"], d["cyc_slot"], d["pos_slot"], d["cur_slot"],
                                   d["tok_hist"], d["h_seed"], d["hd_d0"], d["hd_d1"], d["dhd_seed"],
                                   d["emit"], d["l_hist"]), 1),
                 (pr["acceptsel8k"], (d["rec4"], d["conv4"], d["m_slot"], d["conv5x"], d["conv6x"], d["conv7x"], d["conv8x"], d["rec6x"], d["rec7x"], d["rec8x"]), 48)]''',
    '''      elif LOOKUP_K == 8:
        aseqN = [(pr["accept9k"], (d["amds"], d["dring0"], d["dring1"], d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["dring7"], d["xA"],
                                   d["m_slot"], d["m_hist"], d["cyc_slot"], d["pos_slot"], d["cur_slot"],
                                   d["tok_hist"], d["h_seed"], d["hd_d0"], d["hd_d1"], d["dhd_seed"],
                                   d["emit"], d["l_hist"]), 1),
                 (pr["acceptsel9k"], (d["rec4"], d["conv4"], d["m_slot"], d["conv5x"], d["conv6x"], d["conv7x"], d["conv8x"], d["conv9x"], d["rec6x"], d["rec7x"], d["rec8x"], d["rec9x"]), 48)]
      else:
        aseqN = [(pr["accept8k"], (d["amds"], d["dring0"], d["dring1"], d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["xA"],
                                   d["m_slot"], d["m_hist"], d["cyc_slot"], d["pos_slot"], d["cur_slot"],
                                   d["tok_hist"], d["h_seed"], d["hd_d0"], d["hd_d1"], d["dhd_seed"],
                                   d["emit"], d["l_hist"]), 1),
                 (pr["acceptsel8k"], (d["rec4"], d["conv4"], d["m_slot"], d["conv5x"], d["conv6x"], d["conv7x"], d["conv8x"], d["rec6x"], d["rec7x"], d["rec8x"]), 48)]''',
    "accept9 graphs")

# 9) emit parse
rep('''    if ran_deep5 and LOOKUP_K == 7:  # R5d: accept8k 19-word {pos,m,tok0..7,stop,cyc,hit}
      stop, cyc, hit = int(e[10]), int(e[11]), int(e[12])''',
    '''    if ran_deep5 and LOOKUP_K == 8:  # R7a: accept9k 20-word {pos,m,tok0..8,stop,cyc,hit}
      stop, cyc, hit = int(e[11]), int(e[12]), int(e[13])
    elif ran_deep5 and LOOKUP_K == 7:  # R5d: accept8k 19-word {pos,m,tok0..7,stop,cyc,hit}
      stop, cyc, hit = int(e[10]), int(e[11]), int(e[12])''',
    "emit parse K8")

# 10) DR7 assert
rep('        "PF_DR7 requires GEMVV=1, K3 off, LOOKUP_K in (0,7)"',
    '        "PF_DR7 requires GEMVV=1, K3 off, LOOKUP_K in (0,7,8)"',
    "DR7 assert")

open(f"{BASE}/mtp.py", "w").write(s)
print("[patch] mtp.py K8 wiring complete")
