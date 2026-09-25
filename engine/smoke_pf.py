# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P1 integration smoke (PF_GEMM=1): the FFN stage of a GDN block end-to-end —
k3m_hh (norm) -> [ffn8 per-row | pfg_ffn 16-row] -> down8 (residual+proj) — on
REAL packed weights from the engine's own buffers. Gates: relerr <= 3e-3 both on
gact and on the block-output y (fp32)."""
import os, sys
os.environ.setdefault("DEV", "NV")
os.environ["PF_GEMM"] = "1"
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import Bufs, dev, parse_gguf, read_raw, iq3_grid_f32
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

BASE = "~/tinygrad-metal/engine0"
PACKED = f"{BASE}/packed"
LS = (256, 1, 1)
P = Bufs()
def prog(n):
  lib = open(f"{BASE}/{n}.cubin", "rb").read()
  return NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))

ds, infos = parse_gguf()
gdn_idx = [i for i in range(64) if f"blk.{i}.attn_q.weight" not in infos]
G0 = gdn_idx[0]
P.up("gridf", iq3_grid_f32())
P.up("nw2", np.frombuffer(read_raw(infos[f"blk.{G0}.post_attention_norm.weight"], ds), dtype="<f4"))
P.up("w_fg", np.load(f"{PACKED}/fg{G0}.npy"))
P.up("w_fu", np.load(f"{PACKED}/fu{G0}.npy"))
P.up("w_fd", np.load(f"{PACKED}/fd{G0}.npy"))
for nm, nb, dt, pv in [("xin", 5120*4, np.float32, 7.7e31), ("ao", 5120*2, np.float16, 7.7),
                       ("hh", 5120*4, np.float32, 7.7e31), ("hhx", 5120*2, np.float16, 7.7),
                       ("gact", 17408*2, np.float16, 7.7), ("y", 5120*4, np.float32, 7.7e31)]:
  P.poison(nm, nb, dt, pv)
P.poison("x16", 16*5120*2, np.float16, 7.7)
P.poison("gact16", 16*17408*2, np.float16, 7.7)
dev.synchronize()

rng = np.random.default_rng(21)
XS = (rng.standard_normal((16, 5120)) * 0.7).astype(np.float32)
AOS = (rng.standard_normal((16, 5120)) * 0.5).astype(np.float16)

# --- reference chain, row by row (the exact T=1 engine sequence) ---
gacts_ref = np.zeros((16, 17408), np.float32)
ys_ref = np.zeros((16, 5120), np.float32)
hhx16 = np.zeros((16, 5120), np.float16)
hhs = np.zeros((16, 5120), np.float32)
k3m = prog("k3m_hh"); ffn8 = prog("ffn8"); down8 = prog("down8")
for m in range(16):
  P.win_up("xin", 0, XS[m]); P.win_up("ao", 0, AOS[m]); dev.synchronize()
  k3m(P.d["xin"], P.d["ao"], P.d["nw2"], P.d["hh"], P.d["hhx"], global_size=(1,1,1), local_size=LS, wait=True)
  hhx16[m] = P.down("hhx", (5120,), np.float16)
  hhs[m] = P.down("hh", (5120,), np.float32)
  ffn8(P.d["w_fg"], P.d["w_fu"], P.d["gridf"], P.d["hhx"], P.d["gact"], global_size=(2176,1,1), local_size=LS, wait=True)
  gacts_ref[m] = P.down("gact", (17408,), np.float16).astype(np.float32)
  down8(P.d["w_fd"], P.d["gridf"], P.d["gact"], P.d["hh"], P.d["y"], global_size=(640,1,1), local_size=LS, wait=True)
  ys_ref[m] = P.down("y", (5120,), np.float32)

# --- batched path: one pfg_ffn launch on the 16 hhx rows (PF_GEMM=1 wiring) ---
from trunk_w1c import pf_ffn16_program   # env-gated loader (PF_GEMM)
pf = pf_ffn16_program()
P.up("x16", hhx16.reshape(-1)); dev.synchronize()
pf(P.d["w_fg"], P.d["w_fu"], P.d["gridf"], P.d["x16"], P.d["gact16"], global_size=(17408//64,1,1), local_size=LS, wait=True)
gacts16 = P.down("gact16", (16, 17408), np.float16).astype(np.float32)
ys16 = np.zeros((16, 5120), np.float32)
for m in range(16):
  P.win_up("gact", 0, gacts16[m].astype(np.float16))
  P.win_up("hh", 0, hhs[m]); dev.synchronize()
  down8(P.d["w_fd"], P.d["gridf"], P.d["gact"], P.d["hh"], P.d["y"], global_size=(640,1,1), local_size=LS, wait=True)
  ys16[m] = P.down("y", (5120,), np.float32)

def rep(tag, a, b):
  e = np.abs(a - b) / np.maximum(np.abs(b), 1e-6)
  fn = np.linalg.norm(a - b) / np.linalg.norm(b)
  print(f"[smoke] {tag}: relerr med {np.median(e):.3e} p999 {np.percentile(e,99.9):.3e} F {fn:.3e} -> {'PASS' if fn <= 3e-3 else 'FAIL'}", flush=True)
  return fn <= 3e-3
ok = rep("gact16 (ffn stage)", gacts16, gacts_ref)
ok &= rep("y16 (block output)", ys16, ys_ref)
print(f"[smoke] {'PASS' if ok else 'FAIL'}", flush=True)
