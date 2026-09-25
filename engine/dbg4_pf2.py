# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Attn block-0 stage diff: my M=16 chain vs the T=1 SKV canonical pieces."""
import os, sys
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
M = 16; S = 32; CTXK = 100352
E = TrunkEngineW1C(theta=1e7)
P, W, d, pr = E.P, E.W, E.P.d, E.pr
KSYM = {"pfk_pre16_100k": "pfk_pre16", "pfa16nw32_s32_100k": "pfa16", "pfc16_s32": "pfc16"}
def prog(n):
  return NVProgram(dev, TinyELF(lib=open(f"{BASE}/{n}.cubin","rb").read(), name=KSYM.get(n, n), target=dev.renderer.target, signature=tuple()))
PF = {n: prog(n) for n in ["pfk_emb16","pfk_n16","pfk_pre16_100k","pfa16nw32_s32_100k","pfc16_s32",
                           "pfg_q6q_hm_nw8k64","pfg_iq3q_hm_nw8k128","pfg_iq3k_hm_nw8k128","pfg_q4v_hm_nw8k128","pfg_iq3s_hm_nw8k128"]}
for nm, nb, dt, v in [("xA16", M*5120*4, np.float32, 7.7e31), ("xh16", M*5120*2, np.float16, 7.7),
                      ("qrow16", M*12288*2, np.float16, 7.7), ("krow16", M*1024*2, np.float16, 7.7),
                      ("vrow16", M*1024*2, np.float16, 7.7), ("qw16", M*24*256*2, np.float16, 7.7),
                      ("ao16", M*6144*2, np.float16, 7.7), ("attn_out16", M*5120*2, np.float16, 7.7),
                      ("pm16", 4*S*96*4, np.float32, 7.7e31), ("ps16", 4*S*96*4, np.float32, 7.7e31),
                      ("pA16", 4*S*96*256*4, np.float32, 7.7e31)]:
  P.poison(nm, nb, dt, v)
P.up("ids16", np.arange(500, 500+M, dtype=np.int32))
dev.synchronize(); P._keep.clear()

def rel(tag, mine, ref):
  mine = np.asarray(mine, np.float32); ref = np.asarray(ref, np.float32)
  if not np.isfinite(mine).all(): print(f"[{tag}] MINE NONFINITE"); return
  act = np.abs(ref) > 1e-6
  e = np.abs(mine[act]-ref[act])/np.abs(ref[act]) if act.sum() else np.zeros(1)
  print(f"[{tag}] med {np.median(e):.3e} F {np.linalg.norm(mine-ref)/max(np.linalg.norm(ref),1e-9):.3e}", flush=True)

i = E.attn_idx[0]
print(f"[blk {i}] qtype={E.qtypes[i]} (14=Q6_K, 18=IQ3)", flush=True)
PF["pfk_emb16"](W[("emb",0)], d["grid512"], d["ids16"], d["xA16"], global_size=(16,1,1), local_size=LS)
P.win_up("tok_slot", 0, np.array([500], dtype=np.int32))
pr["h_embed"](W[("emb",0)], d["grid512"], d["tok_slot"], d["x0"], global_size=(1,1,1), local_size=LS, wait=True)
PF["pfk_n16"](d["xA16"], W[("nw1",i)], d["xh16"], global_size=(16,1,1), local_size=LS)
pr["k0_norm"](d["x0"], W[("nw1",i)], d["xh"], global_size=(1,1,1), local_size=LS, wait=True)

qn = "pfg_q6q_hm_nw8k64" if E.qtypes[i] == 14 else "pfg_iq3q_hm_nw8k128"
PF[qn](W[("q",i)], d["gridf"], d["xh16"], d["qrow16"], global_size=(192,1,1), local_size=LS)
PF["pfg_iq3k_hm_nw8k128"](W[("k",i)], d["gridf"], d["xh16"], d["krow16"], global_size=(16,1,1), local_size=LS)
PF["pfg_q4v_hm_nw8k128"](W[("v",i)], d["gridf"], d["xh16"], d["vrow16"], global_size=(16,1,1), local_size=LS)
rq = "aq6k8" if E.qtypes[i] == 14 else "aq3k8"
pr[rq](W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh"], d["qrow"], d["k_row"], d["v_row"], global_size=(1792,1,1), local_size=LS, wait=True)
rel("qrow", P.down("qrow16", (M,12288), np.float16)[0], P.down("qrow", (12288,), np.float16))
rel("krow", P.down("krow16", (M,1024), np.float16)[0], P.down("k_row", (1024,), np.float16))
rel("vrow", P.down("vrow16", (M,1024), np.float16)[0], P.down("v_row", (1024,), np.float16))

# appends: mine (16 rows at pos 0) vs T=1 (row 0 at pos 0)
P.up("pos_slot", np.array([0], dtype=np.int32))
P.up("kvm", np.zeros(2*4*CTXK*256, dtype=np.uint8))
P.up("scm", np.zeros(2*4*CTXK*8, dtype=np.float16))
dev.synchronize()
PF["pfk_pre16_100k"](d["qrow16"], d["krow16"], d["vrow16"], W[("qnw",i)], W[("knw",i)], d["freqs"],
                     d["kvm"], d["scm"], d["pos_slot"], d["qw16"], global_size=(24,1,1), local_size=LS, wait=True)
kvm = P.down("kvm", (2*4*CTXK*256,), np.uint8).copy()
scm = P.down("scm", (2*4*CTXK*8,), np.float16).astype(np.float32).copy()
qw16m = P.down("qw16", (M, 24*256), np.float16).copy()
P.up("pos_slot", np.array([0], dtype=np.int32))
pr["spk_pre1"](d["qrow"], d["k_row"], d["v_row"], W[("qnw",i)], W[("knw",i)], d["freqs"],
                      d["kvm"], d["scm"], d["pos_slot"], d["qw1"], d["qw16_1"], global_size=(24,1,1), local_size=LS, wait=True)
kvr = P.down("kvm", (2*4*CTXK*256,), np.uint8)
scr = P.down("scm", (2*4*CTXK*8,), np.float16).astype(np.float32)
print(f"[append row0] kv byte mismatches (rows 0..15 vs row0 appended twice): {int((kvm != kvr).sum())}", flush=True)
print(f"[append row0] first-512B equal: {np.array_equal(kvm[:512], kvr[:512])}", flush=True)

# attention: mine vs T=1 chain (using row-0 q on the kv state after BOTH appended row 0 — close enough for sanity)
P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
PF["pfa16nw32_s32_100k"](d["kvm"], d["scm"], d["qw16"], d["pos_slot"], d["pm16"], d["ps16"], d["pA16"], global_size=(4*S,1,1), local_size=(1024,1,1))
PF["pfc16_s32"](d["pm16"], d["ps16"], d["pA16"], d["qrow16"], d["ao16"], global_size=(24,1,1), local_size=LS, wait=True)
pr["spk_a1"](d["kvm"], d["scm"], d["qw16_1"], d["pos_slot"], d["pm1"], d["ps1"], d["pA1"], global_size=(4*256,1,1), local_size=(1024,1,1))
pr["spk_c1"](d["pm1"], d["ps1"], d["pA1"], d["qrow"], d["ao_row"], global_size=(24,1,1), local_size=LS, wait=True)
rel("ao row0", P.down("ao16", (M,6144), np.float16)[0], P.down("ao_row", (6144,), np.float16))
PF["pfg_iq3s_hm_nw8k128"](W[("o",i)], d["grid512"], d["ao16"], d["attn_out16"], global_size=(80,1,1), local_size=LS)
pr["ao8"](W[("o",i)], d["grid512"], d["ao_row"], d["attn_out"], global_size=(640,1,1), local_size=LS, wait=True)
rel("attn_out", P.down("attn_out16", (M,5120), np.float16)[0], P.down("attn_out", (5120,), np.float16))
print("[dbg4 done]", flush=True)
