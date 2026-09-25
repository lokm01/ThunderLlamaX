# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7a K=8: append _probe9_seq (M=9/T=9 deep probe) after _probe8_seq."""
BASE = "~/tinygrad-metal/engine0"
s = open(f"{BASE}/mtp.py").read()
anchor = "  # ================= M1-A: fixed-handle state machine ================="
assert s.count(anchor) == 1
probe9 = '''  def _probe9_seq(self):
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

'''
s = s.replace(anchor, probe9 + anchor, 1)
open(f"{BASE}/mtp.py", "w").write(s)
print("[patch] _probe9_seq appended")
