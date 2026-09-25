# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P13 ext: test_p7b's EXACT ffn flow (setup + P6 ref + r7 gate), then my
minimal A-style launches appended IN THE SAME PROCESS. If the p7b part passes
and mine passes too -> cross-setup state (parse_gguf etc). If mine faults ->
per-launch issue in my buffers/args."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev, parse_gguf, read_raw, iq3_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
PACKED = f"{BASE}/packed"
P7 = f"{BASE}/packed7"
LS = (256, 1, 1)
KD, ND, NGRID = 5120, 17408, 272
P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

P.up("gridf", iq3_grid_f32())
dev.synchronize()
ds, infos = parse_gguf()
attn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" in infos]
gdn_idx = [i for i in range(64) if i not in set(attn_idx)]
G0 = gdn_idx[0]
print(f"[blocks] gdn0={G0}", flush=True)
P.up("w_fg", np.load(f"{PACKED}/fg{G0}.npy"))
P.up("w_fu", np.load(f"{PACKED}/fu{G0}.npy"))
P.up("r_fg", np.load(f"{P7}/fg{G0}.npy"))
P.up("r_fu", np.load(f"{P7}/fu{G0}.npy"))
P.up("w_fd", np.load(f"{PACKED}/fd{G0}.npy"))
OQ = next(i for i in gdn_idx if os.path.exists(f"{PACKED}/out{i}.npy"))
P.up("w_o18", np.load(f"{PACKED}/out{OQ}.npy"))
P.up("w_qkv5", np.frombuffer(read_raw(infos[f"blk.{G0}.attn_qkv.weight"], ds), dtype=np.uint8))
A18 = next(i for i in attn_idx if infos[f"blk.{i}.attn_q.weight"][0] != 14)
P.up("w_v4", np.frombuffer(read_raw(infos[f"blk.{A18}.attn_v.weight"], ds), dtype=np.uint8))
P.up("r_fd", np.load(f"{P7}/fd{G0}.npy"))
P.up("r_o", np.load(f"{P7}/out{OQ}.npy"))
P.up("r_gate", np.load(f"{P7}/gate{G0}.npy"))
P.up("r_q", np.load(f"{P7}/q{A18}.npy"))
P.up("r_k", np.load(f"{P7}/k{A18}.npy"))
rng = np.random.default_rng(7)
P.up("res64", (rng.standard_normal((64, 5120)) * 4.0).astype(np.float32).reshape(-1))
dev.synchronize()
print("[weights] up", flush=True)

# ---- the p7b ffn gate flow verbatim ----
P.up("xffn", (rng.standard_normal((64, KD)) * 0.8).astype(np.float16).reshape(-1))
P.poison("refffn", 64 * ND, np.float16, 7.7)
P.poison("mineffn", 64 * ND, np.float16, 7.7)
dev.synchronize()
# P6 ref (classic)
pr6 = prog("pfg_ffn_m32_hm_nw8k128")
for i in range(2):
  x = P.d["xffn"] if i == 0 else P.d["xffn"].offset(offset=32 * KD * 2, size=32 * KD * 2)
  o = P.d["refffn"] if i == 0 else P.d["refffn"].offset(offset=32 * ND * 2, size=32 * ND * 2)
  pr6(P.d["w_fg"], P.d["w_fu"], P.d["gridf"], x, o, global_size=(NGRID, 1, 1), local_size=LS)
dev.synchronize()
print("[p7b] P6 ref CLEAN", flush=True)
# r7 gate (the kernel under test)
pr7 = prog("pfg3_ffn_r7_m32_nw8k128")
for i in range(2):
  x = P.d["xffn"] if i == 0 else P.d["xffn"].offset(offset=32 * KD * 2, size=32 * KD * 2)
  o = P.d["mineffn"] if i == 0 else P.d["mineffn"].offset(offset=32 * ND * 2, size=32 * ND * 2)
  pr7(P.d["r_fg"], P.d["r_fu"], P.d["gridf"], x, o, global_size=(NGRID, 1, 1), local_size=LS)
dev.synchronize()
ref = P.down("refffn", (64, ND), np.float16)
mine = P.down("mineffn", (64, ND), np.float16)
print(f"[p7b] r7 gate nz={int((mine != ref).sum())}", flush=True)

# ---- now MY minimal A-style launches, same process ----
P.up("x32", (rng.standard_normal((32, KD)) * 0.8).astype(np.float16).reshape(-1))
P.poison("o32", 32 * ND, np.float16, 7.7)
dev.synchronize()
print("[mine] A-style launch (fresh bufs)...", flush=True)
pr7(P.d["r_fg"], P.d["r_fu"], P.d["gridf"], P.d["x32"], P.d["o32"], global_size=(NGRID, 1, 1), local_size=LS)
dev.synchronize()
print("[mine] A CLEAN", flush=True)
# and with block-0 weights (different files, same sizes)
P.up("fg0", np.load(f"{P7}/fg0.npy")); P.up("fu0", np.load(f"{P7}/fu0.npy"))
P.poison("o32b", 32 * ND, np.float16, 7.7)
dev.synchronize()
print("[mine] block0-weights launch...", flush=True)
pr7(P.d["fg0"], P.d["fu0"], P.d["gridf"], P.d["x32"], P.d["o32b"], global_size=(NGRID, 1, 1), local_size=LS)
dev.synchronize()
print("[mine] block0 CLEAN", flush=True)
