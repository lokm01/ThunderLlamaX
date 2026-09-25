# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Truncated token-0 blk0: ref _seq entries vs my chain, inside harness env."""
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
_lib = open(f"{BASE}/spk_c1g_100k.cubin", "rb").read()
pr["spk_c1"] = NVProgram(dev, TinyELF(lib=_lib, name="spk_c1g_100k", target=dev.renderer.target, signature=tuple()))
KSYM = {"pfs16": "pfs16"}
def prog(n):
  return NVProgram(dev, TinyELF(lib=open(f"{BASE}/{n}.cubin","rb").read(), name=KSYM.get(n, n), target=dev.renderer.target, signature=tuple()))
PF = {n: prog(n) for n in ["pfk_emb16","pfk_ab16","pfs16","pfg_q5kv_hm_nw16k128","pfg_iq3g_hm_nw16k128"]}
for nm, nb, dt, v in [("xA16", M*5120*4, np.float32, 7.7e31), ("xh16", M*5120*2, np.float16, 7.7),
                      ("qkv16", M*10240*2, np.float16, 7.7), ("gate16", M*6144*2, np.float16, 7.7),
                      ("z16", M*6144*2, np.float16, 7.7), ("araw16", M*48*4, np.float32, 7.7e31),
                      ("braw16", M*48*4, np.float32, 7.7e31)]:
  P.poison(nm, nb, dt, v)
P.up("ids16", np.arange(100, 100+M, dtype=np.int32))
for i in E.gdn_idx:
  P.up(f"conv{i}_0", np.zeros(3*10240, dtype=np.float32)); P.up(f"conv{i}_1", np.zeros(3*10240, dtype=np.float32))
  P.up(f"rec{i}", np.zeros(48*128*128, dtype=np.float32))
P.up("convp0", np.zeros(3*10240, dtype=np.float32)); P.up("recp0", np.zeros(48*128*128, dtype=np.float32))
dev.synchronize(); P._keep.clear()

def rel(tag, mine, ref):
  mine = np.asarray(mine, np.float32); ref = np.asarray(ref, np.float32)
  act = np.abs(ref) > 1e-6
  e = np.abs(mine[act]-ref[act])/np.abs(ref[act]) if act.sum() else np.zeros(1)
  print(f"[{tag}] med {np.median(e):.3e} F {np.linalg.norm(mine-ref)/max(np.linalg.norm(ref),1e-9):.3e}", flush=True)

# --- my chain: embed -> ab16 -> q5kv+iq3g (block 0 rows all 16) ---
PF["pfk_emb16"](W[("emb",0)], d["grid512"], d["ids16"], d["xA16"], global_size=(16,1,1), local_size=LS)
i0 = E.gdn_idx[0]
PF["pfk_ab16"](d["xA16"], W[("nw1",i0)], W[("alpha",i0)], W[("beta",i0)], d["xh16"], d["araw16"], d["braw16"], global_size=(16*13,1,1), local_size=LS)
PF["pfg_q5kv_hm_nw16k128"](W[("qkv",i0)], d["gridf"], d["xh16"], d["qkv16"], global_size=(80,1,1), local_size=(512,1,1))
PF["pfg_iq3g_hm_nw16k128"](W[("gate",i0)], d["gridf"], d["xh16"], d["gate16"], global_size=(48,1,1), local_size=(512,1,1), wait=True)
qkv16 = P.down("qkv16", (M,10240), np.float16); gate16 = P.down("gate16", (M,6144), np.float16)
araw16 = P.down("araw16", (M,48), np.float32); braw16 = P.down("braw16", (M,48), np.float32)

# --- ref: _seq[0] entries 0..3 (embed, k0ab, q5g8) for token 0 ---
P.win_up("tok_slot", 0, np.array([100], dtype=np.int32))
P.win_up("pos_slot", 0, np.array([0], dtype=np.int32))
dev.synchronize()
if not hasattr(E, "_seq"): E._build_seqs()
seq = E._seq[0]
for n, (p, a, g) in enumerate(seq[:4]):
  p(*a, global_size=(g[0],1,1) if isinstance(g, tuple) else (g,1,1), local_size=LS)
  if getattr(p, "name", "") == "q5g8": break
dev.synchronize()
rel("qkv row0", qkv16[0], P.down("qkv_row", (10240,), np.float16))
rel("gate row0", gate16[0], P.down("gate_row", (6144,), np.float16))
rel("araw row0", araw16[0], P.down("alpharaw", (48,), np.float32))
rel("braw row0", braw16[0], P.down("betaraw", (48,), np.float32))

# --- now the k2s entry (index 3) + compare rec/conv/z ---
p, a, g = seq[3]
print(f"[seq3] {getattr(p,'name','?')}")
p(*a, global_size=(g[0],1,1) if isinstance(g, tuple) else (g,1,1), local_size=LS, wait=True)
zref1 = P.down("z", (6144,), np.float16).copy()
recref1 = P.down(f"rec{i0}", (48*128*128,), np.float32).copy()
convref1 = P.down(f"conv{i0}_1", (3*10240,), np.float32).copy()
# mine: pfs16 16 steps (only row 0 comparable for z)
PF["pfs16"](d["convp0"], d["recp0"], d["qkv16"], d["gate16"], W[("convw",i0)], W[("dtb",i0)], W[("ssma",i0)],
            d["araw16"], d["braw16"], d["q"], d["k"], d["v"], d["core"], W[("snw",i0)], d["z16"],
            global_size=(48,1,1), local_size=LS, wait=True)
rel("z row0", P.down("z16", (M,6144), np.float16)[0], zref1)
rel("conv after16 vs ref step1?? (only shape check)", P.down("convp0", (3*10240,), np.float32), convref1)
print("[dbg8 done]", flush=True)
