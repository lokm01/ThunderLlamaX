# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
import numpy as np
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
from mtp import MTPEngine, SLICE
from engine0 import dev
snap = np.load("~/w1b_state_2k.npz")
E = MTPEngine(float(snap["theta"].reshape(-1)[0]))
ids = snap["ids"].reshape(-1).tolist()
seen, sl = set(), []
for t in ([3204,40224]*30) + ids:
  if t not in seen: seen.add(t); sl.append(t)
base = sl[:]
while len(sl) < SLICE: sl += base
E.init_draft(sl[:SLICE])
E.restore_mtp(snap)
E.fill_draft(ids)
E.build_graphs()
dev.synchronize()
d, pr, P = E.P.d, E.pr, E.P
LS=(256,1,1)
draft_g, probe_g, accept_g, flush_g = E.graphs
prev = dev.timeline_value - 1
for c in range(3):
  vd = dev.next_timeline(); draft_g.submit(prev, vd)
  vp = dev.next_timeline(); probe_g.submit(vd, vp)
  va = dev.next_timeline(); accept_g.submit(vp, va)
  vf = dev.next_timeline(); flush_g.submit(va, vf)
  dev.timeline_signal.wait(vf); prev = vf
  amds = P.down("amds", (3,), np.int32)
  print(f"[h] cyc{c}: amds={amds.tolist()} dring=({int(P.down('dring0',(1,),np.int32)[0])},{int(P.down('dring1',(1,),np.int32)[0])}) m={int(P.down('m_slot',(1,),np.int32)[0])} cur={int(P.down('cur_slot',(1,),np.int32)[0])} h_seed absmax={np.abs(P.down('h_seed',(5120,))).max():.3g}", flush=True)
# eager draft step with the REAL h_seed + cur
pr["h_embed"](E.W[("emb",0)], d["grid512"], d["cur_slot"], d["e_buf"], global_size=(1,1,1), local_size=LS)
pr["dnorm2"](d["e_buf"], d["h_seed"], d["d_enw"], d["d_hnw"], d["cat"], global_size=(1,1,1), local_size=LS)
pr["ehproj"](d["d_eh"], d["cat"], d["zed5k"], d["xin_d"], global_size=(640,1,1), local_size=LS)
pr["k0_norm"](d["xin_d"], d["d_nw1"], d["xh_d"], global_size=(1,1,1), local_size=LS)
pr["dq"](d["d_q"], d["xh_d"], d["zed5k"], d["qrow_d"], global_size=(1536,1,1), local_size=LS)
pr["dkv"](d["d_k"], d["d_v"], d["xh_d"], d["krow_d"], d["vrow_d"], global_size=(256,1,1), local_size=LS)
pr["aattn_d"](d["qrow_d"], d["krow_d"], d["vrow_d"], d["d_qnw"], d["d_knw"], d["freqs"], d["kv_d"], d["pos_slot"], d["ao_row_d"], global_size=(24,1,1), local_size=LS)
pr["doproj"](d["d_o"], d["ao_row_d"], d["hh_d"], d["attn_out_d"], global_size=(640,1,1), local_size=LS)
pr["k3m_hh"](d["xin_d"], d["attn_out_d"], d["d_nw2"], d["hh_d"], d["hhx_d"], global_size=(1,1,1), local_size=LS)
pr["dfgu"](d["d_fg"], d["d_fu"], d["hhx_d"], d["gact_d"], global_size=(2176,1,1), local_size=LS)
pr["ddown"](d["d_fd"], d["gact_d"], d["hh_d"], d["hd_d0"], global_size=(640,1,1), local_size=LS)
pr["k0_norm"](d["hd_d0"], d["d_shnw"], d["xh_d"], global_size=(1,1,1), local_size=LS)
pr["shead"](d["slice_w"], d["xh_d"], d["slogits"], global_size=(SLICE//8,1,1), local_size=LS)
P.up("dr1w", np.array([-9], np.int32)); dev.synchronize()
pr["samx"](d["slogits"], d["stab"], d["dr1w"], global_size=(1,1,1), local_size=LS, wait=True)
prop = int(P.down("dr1w",(1,),np.int32)[0])
slog = P.down("slogits",(SLICE,),np.float16).astype(np.float32)
stab = P.down("stab",(SLICE,),np.int32)
i3204 = int(np.argmax(stab == 3204))
print(f"[h] EAGER proposal with real h_seed: {prop}; logit(3204)={slog[i3204]:.3f} top={slog.max():.3f}", flush=True)
