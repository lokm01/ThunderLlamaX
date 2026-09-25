# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Stage-by-stage GDN block-0 diff: T=1 piece kernels vs my M=16 chain."""
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
M = 16
E = TrunkEngineW1C(theta=1e7)
P, W, d, pr = E.P, E.W, E.P.d, E.pr
KSYM = {"pfk_pre16_100k": "pfk_pre16", "pfa16nw32_s32_100k": "pfa16", "pfc16_s32": "pfc16", "pfs16": "pfs16"}
def prog(n):
  return NVProgram(dev, TinyELF(lib=open(f"{BASE}/{n}.cubin","rb").read(), name=KSYM.get(n, n), target=dev.renderer.target, signature=tuple()))
PF = {n: prog(n) for n in ["pfk_emb16","pfk_n16","pfk_ab16","pfk_hh16","pfs16",
                           "pfg_q5kv_hm_nw16k128","pfg_iq3g_hm_nw16k128","pfg_ffn_hm_nw8k128",
                           "pfg_iq3d_res_hm_nw8k128","pfg_iq3o_hm_nw8k128","pfg_q8o_hm_nw8k64"]}
for nm, nb, dt, v in [("xA16", M*5120*4, np.float32, 7.7e31), ("xh16", M*5120*2, np.float16, 7.7),
                      ("hh16", M*5120*4, np.float32, 7.7e31), ("hhx16", M*5120*2, np.float16, 7.7),
                      ("attn_out16", M*5120*2, np.float16, 7.7), ("qkv16", M*10240*2, np.float16, 7.7),
                      ("gate16", M*6144*2, np.float16, 7.7), ("z16", M*6144*2, np.float16, 7.7),
                      ("gact16", M*17408*2, np.float16, 7.7), ("araw16", M*48*4, np.float32, 7.7e31),
                      ("braw16", M*48*4, np.float32, 7.7e31), ("xB16", M*5120*4, np.float32, 7.7e31)]:
  P.poison(nm, nb, dt, v)
P.up("ids16", np.arange(100, 100+M, dtype=np.int32))
P.up(f"convp0", np.zeros(3*10240, dtype=np.float32))
P.up(f"recp0", np.zeros(48*128*128, dtype=np.float32))
P.up("conv0_0", np.zeros(3*10240, dtype=np.float32)); P.up("conv0_1", np.zeros(3*10240, dtype=np.float32))
P.up("rec0", np.zeros(48*128*128, dtype=np.float32))
dev.synchronize(); P._keep.clear()
i = E.gdn_idx[0]
rng = np.random.default_rng(1)

def rel(tag, mine, ref):
  mine = np.asarray(mine, np.float32); ref = np.asarray(ref, np.float32)
  if not np.isfinite(mine).all(): print(f"[{tag}] MINE HAS NONFINITE"); return
  act = np.abs(ref) > 1e-6
  e = np.abs(mine[act]-ref[act])/np.abs(ref[act]) if act.sum() else np.zeros(1)
  print(f"[{tag}] med {np.median(e):.3e} F {np.linalg.norm(mine-ref)/max(np.linalg.norm(ref),1e-9):.3e}", flush=True)

# --- embed: mine (16 rows) vs ref (row 0 token 100) ---
PF["pfk_emb16"](W[("emb",0)], d["grid512"], d["ids16"], d["xA16"], global_size=(16,1,1), local_size=LS)
P.win_up("tok_slot", 0, np.array([100], dtype=np.int32))
pr["h_embed"](W[("emb",0)], d["grid512"], d["tok_slot"], d["x0"], global_size=(1,1,1), local_size=LS, wait=True)
x16 = P.down("xA16", (M, 5120), np.float32); x0 = P.down("x0", (5120,), np.float32)
rel("embed", x16[0], x0)

# --- norm+ab: mine row0 vs ref ---
PF["pfk_ab16"](d["xA16"], W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh16"], d["araw16"], d["braw16"], global_size=(16*13,1,1), local_size=LS)
pr["k0ab"](d["x0"], W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh"], d["alpharaw"], d["betaraw"], global_size=(13,1,1), local_size=LS, wait=True)
xh16 = P.down("xh16", (M, 5120), np.float16)
rel("xh", xh16[0], P.down("xh", (5120,), np.float16))
rel("araw", P.down("araw16", (M,48), np.float32)[0], P.down("alpharaw", (48,), np.float32))

# --- qkv+gate GEMMs ---
PF["pfg_q5kv_hm_nw16k128"](W[("qkv",i)], d["gridf"], d["xh16"], d["qkv16"], global_size=(80,1,1), local_size=(512,1,1))
PF["pfg_iq3g_hm_nw16k128"](W[("gate",i)], d["gridf"], d["xh16"], d["gate16"], global_size=(48,1,1), local_size=(512,1,1))
pr["q5g8"](W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh"], d["qkv_row"], d["gate_row"], global_size=(2048,1,1), local_size=LS, wait=True)
rel("qkv", P.down("qkv16", (M,10240), np.float16)[0], P.down("qkv_row", (10240,), np.float16))
rel("gate", P.down("gate16", (M,6144), np.float16)[0], P.down("gate_row", (6144,), np.float16))

# --- scan: mine (16 steps from zeros) vs ref (1 step) — compare only row 0 ---
# ref: k2s reads alpharaw/betaraw from k0ab above (row-0 values)
pr["k2s"](d["conv0_0"], d["conv0_1"], d["qkv_row"], d["gate_row"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
          d["alpharaw"], d["betaraw"], d["q"], d["k"], d["v"], d["rec0"], d["core"], W[("snw",i)], d["z"],
          global_size=(48,1,1), local_size=LS, wait=True)
zref = P.down("z", (6144,), np.float16).copy()
recref = P.down("rec0", (48*128*128,), np.float32).copy()
convref = P.down("conv0_1", (3*10240,), np.float32).copy()
# mine: araw16/braw16 rows for 16 steps; rows 1..15 inputs are junk-but-finite (row 0 is what we compare)
P.win_up("recp0", 0, np.zeros(48*128*128, dtype=np.float32))
PF["pfs16"](d["convp0"], d["recp0"], d["qkv16"], d["gate16"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
            d["araw16"], d["braw16"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], d["z16"],
            global_size=(48,1,1), local_size=LS, wait=True)
z16 = P.down("z16", (M,6144), np.float16)
rel("z row0", z16[0], zref)
recp = P.down("recp0", (48*128*128,), np.float32)
# NOTE: mine ran 16 steps (junk rows 1..15 feed back into rec) — only z row0 + the STEP-0 state are comparable;
# instead compare the FULL 16-step mine vs 16 sequential ref steps below for state.

# --- o-proj ---
on = "pfg_q8o_hm_nw8k64" if E.gdn_oq8[i] else "pfg_iq3o_hm_nw8k128"
print(f"[oproj] class {on} oq8={E.gdn_oq8[i]}")
PF[on](W[("out",i)], d["gridf"], d["z16"], d["attn_out16"], global_size=(80,1,1), local_size=LS)
if E.gdn_oq8[i]: pr["k3a_oproj"](W[("out",i)], d["z"], d["attn_out"], global_size=(640,1,1), local_size=LS)
else: pr["op38"](W[("out",i)], d["gridf"], d["z"], d["attn_out"], global_size=(640,1,1), local_size=LS, wait=True)
rel("attn_out", P.down("attn_out16", (M,5120), np.float16)[0], P.down("attn_out", (5120,), np.float16))

# --- hh + ffn + down ---
PF["pfk_hh16"](d["xA16"], d["attn_out16"], W[("nw2",i)], d["hh16"], d["hhx16"], global_size=(16,1,1), local_size=LS)
pr["k3m_hh"](d["x0"], d["attn_out"], W[("nw2",i)], d["hh"], d["hhx"], global_size=(1,1,1), local_size=LS)
rel("hh", P.down("hh16", (M,5120), np.float32)[0], P.down("hh", (5120,), np.float32))
rel("hhx", P.down("hhx16", (M,5120), np.float16)[0], P.down("hhx", (5120,), np.float16))
PF["pfg_ffn_hm_nw8k128"](W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx16"], d["gact16"], global_size=(272,1,1), local_size=LS)
pr["ffn8"](W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx"], d["gact"], global_size=(2176,1,1), local_size=LS, wait=True)
rel("gact", P.down("gact16", (M,17408), np.float16)[0], P.down("gact", (17408,), np.float16))
PF["pfg_iq3d_res_hm_nw8k128"](W[("fd",i)], d["gridf"], d["gact16"], d["hh16"], d["xB16"], global_size=(80,1,1), local_size=LS, wait=True)
pr["down8"](W[("fd",i)], d["gridf"], d["gact"], d["hh"], d["x1"], global_size=(640,1,1), local_size=LS, wait=True)
rel("xout", P.down("xB16", (M,5120), np.float32)[0], P.down("x1", (5120,), np.float32))

# --- full 16-step state compare: run ref k2s 16x with MY qkv16/gate16 rows + MY araw/braw rows ---
P.up("conv0_0", np.zeros(3*10240, dtype=np.float32)); P.up("rec0", np.zeros(48*128*128, dtype=np.float32))
P.up("convp0", np.zeros(3*10240, dtype=np.float32)); P.up("recp0", np.zeros(48*128*128, dtype=np.float32))
dev.synchronize()
ar16 = P.down("araw16", (M,48), np.float32); br16 = P.down("braw16", (M,48), np.float32)
par = 0
for t in range(M):
  P.win_up("alpharaw", 0, ar16[t]); P.win_up("betaraw", 0, br16[t])
  qr = d["qkv16"].offset(offset=t*10240*2, size=10240*2)
  gr = d["gate16"].offset(offset=t*6144*2, size=6144*2)
  pr["k2s"](d[f"conv0_{par}"], d[f"conv0_{par^1}"], qr, gr, W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
            d["alpharaw"], d["betaraw"], d["q"], d["k"], d["v"], d["rec0"], d["core"], W[("snw",i)], d["z"],
            global_size=(48,1,1), local_size=LS, wait=(t==M-1))
  par ^= 1
PF["pfs16"](d["convp0"], d["recp0"], d["qkv16"], d["gate16"], W[("convw",i)], W[("dtb",i)], W[("ssma",i)],
            d["araw16"], d["braw16"], d["q"], d["k"], d["v"], d["core"], W[("snw",i)], d["z16"],
            global_size=(48,1,1), local_size=LS, wait=True)
convr = P.down(f"conv0_{par^1}", (3*10240,), np.float32)
recr = P.down("rec0", (48*128*128,), np.float32)
recp = P.down("recp0", (48*128*128,), np.float32); convp = P.down("convp0", (3*10240,), np.float32)
rel("rec16", recp, recr)
rel("conv16", convp, convr)
zr = P.down("z", (6144,), np.float16)
rel("z16-final-row", P.down("z16", (M,6144), np.float16)[M-1], zr)
print("[dbg3 done]", flush=True)
