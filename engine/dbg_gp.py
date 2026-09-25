# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2 probe-graph prefix bisect (flusher pattern)."""
import os, sys
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from mtp import MTPEngine, SLICE, RBLK, CBLK
from gcycle import ParityGraph
from engine0 import dev
from trunk import VOCAB

snap = np.load("~/w1b_state_2k.npz")
E = MTPEngine(float(snap["theta"].reshape(-1)[0]))
ids = snap["ids"].reshape(-1).tolist()
E.init_draft((ids * 4000)[:SLICE])
E.restore_mtp(snap)
dev.synchronize()
d, W, pr = E.P.d, E.W, E.pr

seq = [(pr["h_embed3"], (W[("emb",0)], d["grid512"], d["cur_slot"], d["dring0"], d["dring1"], d["xA"]), 1)]
cur = 0
for i in range(64):
  xin, xout = (d["xA"] if cur == 0 else d["xB"]), (d["xB"] if cur == 0 else d["xA"])
  if i in E.qtypes:
    qkname = "aq6k8_3" if E.qtypes[i] == 14 else "aq3k8_3"
    a = [(pr["k0n3"], (xin, W[("nw1",i)], d["xh3"]), 1),
         (pr[qkname], (W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh3"], d["qrow3"], d["krow3"], d["vrow3"]), 1792),
         (pr["aattn3"], (d["qrow3"], d["krow3"], d["vrow3"], W[("qnw",i)], W[("knw",i)], d["freqs"], d[f"kv{i}"], d["pos_slot"], d["ao_row3"]), 24),
         (pr["ao8_3"], (W[("o",i)], d["grid512"], d["ao_row3"], d["attn_out3"]), 640),
         (pr["hh3"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 1),
         (pr["ffn8_3"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176),
         (pr["down8_3"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 640)]
  else:
    gi = E.gdn_idx.index(i)
    conv_b = d["conv4"].offset(offset=gi*5*CBLK*4, size=5*CBLK*4)
    rec_b = d["rec4"].offset(offset=gi*5*RBLK*4, size=5*RBLK*4)
    a = [(pr["k0ab3"], (xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh3"], d["araw3"], d["braw3"]), 13),
         (pr["q5g8_3"], (W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh3"], d["qkv3"], d["gate3"]), 2048),
         (pr["k2s3"], (conv_b, rec_b, d["qkv3"], d["gate3"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
                       d["araw3"], d["braw3"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], d["z3"]), 48)]
    if E.gdn_oq8[i]:
      a.append((pr["k3ao3"], (W[("out",i)], d["z3"], d["attn_out3"]), 640))
    else:
      a.append((pr["op38_3"], (W[("out",i)], d["gridf"], d["z3"], d["attn_out3"]), 640))
    a.append((pr["hh3"], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), 1))
    a.append((pr["ffn8_3"], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), 2176))
    a.append((pr["down8_3"], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), 640))
  seq += a
  cur ^= 1
seq.append((pr["k0n3"], (d["xA"], W[("onw",0)], d["xh3"]), 1))
seq.append((pr["head8_3"], (W[("head",0)], d["xh3"], d["logits3"]), VOCAB//8))
seq.append((pr["amx3"], (d["logits3"], d["amds"]), 3))
print(f"[dbgp] probe seq {len(seq)} kernels", flush=True)

fl = ParityGraph([(pr["dposadd"], (d["fillpos"], d["dpos1"]), 1)], tag="pfl")
import time
def name_of(k):
  return {"k0n3":"k0n3","k0ab3":"k0ab3","q5g8_3":"q5g8_3","k2s3":"k2s3","k3ao3":"k3ao3","op38_3":"op38_3",
          "hh3":"hh3","ffn8_3":"ffn8_3","down8_3":"down8_3","aq6k8_3":"aq6","aq3k8_3":"aq3","aattn3":"aattn3",
          "ao8_3":"ao8_3","h_embed3":"emb3","head8_3":"head8_3","amx3":"amx3"}[k]
for n in [2] + list(range(8, len(seq)+1, 8)) + [len(seq)]:
  if n > len(seq): continue
  try:
    g = ParityGraph(seq[:n], tag=f"P{n}")
    prev = dev.timeline_value - 1
    v = dev.next_timeline(); g.submit(prev, v)
    vf = dev.next_timeline(); fl.submit(v, vf)
    dev.timeline_signal.wait(vf, timeout=10000)
    print(f"[dbgp] prefix {n} ({name_of(seq[n-1][0].name) if hasattr(seq[n-1][0],'name') else '?'}) OK", flush=True)
    del g
    time.sleep(0.2)
  except Exception as ex:
    print(f"[dbgp] prefix {n} FAULT: {type(ex).__name__} {str(ex)[:100]}", flush=True)
    break
print("[dbgp] done", flush=True)
