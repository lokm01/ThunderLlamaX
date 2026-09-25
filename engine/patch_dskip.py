# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7a draft-skip: lookup-only draft graph for deep cycles. On a deep cycle
(prev emit hit) the 2-step draft chain's dring0/1 are overwritten by the
lookup kernel — the chain is pure waste (5.3ms x 78% of cycles). Skip it:
draft_lu_g = ParityGraph([lookup entry]). Output-lossless BY CONSTRUCTION
(every committed token is amds-verified; the draft is only a proposal source
for K2/miss cycles, which keep the full draft graph)."""
BASE = "~/tinygrad-metal/engine0"
s = open(f"{BASE}/mtp.py").read()

def rep(a, b, label):
  global s
  c = s.count(a)
  assert c == 1, f"{label}: found {c}"
  s = s.replace(a, b)
  print(f"[patch] {label:44s} ok")

# 1) capture the K=8 lookup entry as lu8 (before appending to dseq)
rep('''    elif LOOKUP_K == 8:
      # R7a: 8-proposal overwrite (scan range i in [0, pos-17] — the deep-K law at K=8).
      dseq += [(pr["lookup9_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"],
                                     d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["dring7"], d["l_hist"], d["cyc_slot"]), 1)]''',
    '''    elif LOOKUP_K == 8:
      # R7a: 8-proposal overwrite (scan range i in [0, pos-17] — the deep-K law at K=8).
      lu8 = (pr["lookup9_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"],
                                  d["dring2"], d["dring3"], d["dring4"], d["dring5"], d["dring6"], d["dring7"], d["l_hist"], d["cyc_slot"]), 1)
      dseq += [lu8]''',
    "lu8 entry captured")

# 2) build draft_lu_g after draft_g
rep('''      dseq += [(pr["lookup_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"], d["l_hist"], d["cyc_slot"]), 1)]
    draft_g = ParityGraph(dseq, tag="mtpD")''',
    '''      dseq += [(pr["lookup_nw32"], (d["tok_hist"], d["pos_slot"], d["cur_slot"], d["dring0"], d["dring1"], d["l_hist"], d["cyc_slot"]), 1)]
    draft_g = ParityGraph(dseq, tag="mtpD")
    if LOOKUP_K == 8:
      # R7a draft-skip: deep cycles (prev emit hit) run LOOKUP ONLY — the draft
      # chain's dring0/1 would be overwritten by the lookup; skipping it is
      # output-lossless (probe-verifies amds; draft is only the K2/miss
      # proposal source, and those cycles keep the full draft graph). On a deep
      # cycle whose lookup MISSES, dring0/1 hold stale-but-valid ids (the
      # deep-K Tier-1-safety law). Saves ~5.3ms x deep-fraction per cycle.
      self.draft_lu_g = ParityGraph([lu8], tag="mtpDL")''',
    "draft_lu_g built")

# 3) step() dispatch
rep('''    if LOOKUP_K >= 5: g = E.graphs6 if self.deep else E.graphs
    elif LOOKUP_K: g = E.graphs5 if self.deep else E.graphs
    else: g = E.graphs
    ran_deep5 = (LOOKUP_K >= 5 and self.deep)   # the set that ran THIS cycle (emit layout)
    draft_g, probe_g, accept_g, flush_g = g''',
    '''    if LOOKUP_K >= 5: g = E.graphs6 if self.deep else E.graphs
    elif LOOKUP_K: g = E.graphs5 if self.deep else E.graphs
    else: g = E.graphs
    ran_deep5 = (LOOKUP_K >= 5 and self.deep)   # the set that ran THIS cycle (emit layout)
    draft_g, probe_g, accept_g, flush_g = g
    if LOOKUP_K == 8 and self.deep and getattr(E, "draft_lu_g", None) is not None:
      draft_g = E.draft_lu_g   # R7a draft-skip on deep cycles''',
    "step dispatch")

open(f"{BASE}/mtp.py", "w").write(s)
print("[patch] draft-skip wired")
