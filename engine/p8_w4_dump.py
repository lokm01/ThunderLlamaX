# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P8w4 dump probe: p8w4ffn7d raw-facc dump vs the linearized W columns —
pins the k-permutation bug. k=0..7 single-nonzero probes; gout[0][n] must
equal 127 * Wlin[n,k] exactly."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from mtp import MTPEngine, CBLK
import pf_prefill

E = MTPEngine(theta=1e7)
E.reset_fresh(101)
E.stload_trunk()
for i in E.gdn_idx:
  E._mfill(f"conv{i}_1", 0, CBLK)
dev.synchronize()
pf_prefill.ensure64(E)
pf_prefill.ensure128(E)
dev.synchronize(); E.P._keep.clear()
P, d, W, pr = E.P, E.P.d, E.W, E.pr
W7 = getattr(E, "_pf_W7", {})
LS = (256, 1, 1)
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram
pr["p8w4ffn7d_nw8k128"] = NVProgram(dev, TinyELF(
    lib=open("~/tinygrad-metal/engine0/p8w4ffn7d_nw8k128.cubin", "rb").read(),
    name="p8w4ffn7d", target=dev.renderer.target, signature=tuple()))
from pack_w4 import grid_f32
DELTA = 4.0507
G = grid_f32().reshape(-1)
lut8 = np.clip(np.rint(G / DELTA), 0, 15).astype(np.int8)
P.up("w7t_lut", lut8)
KD, ND, M = 5120, 17408, 128
NCH = KD // 128
P.poison("w7t_gout", M*ND*4, np.float32, 7.7e31)
P.poison("w7t_uout", M*ND*4, np.float32, 7.7e31)
P.up("dbg_xq", np.zeros((M, KD), dtype=np.int8))
P.up("dbg_sx", np.ones((M, NCH), dtype=np.float32))
dev.synchronize(); P._keep.clear()

i = 3
# WARM the plan world first (the P8 standalone-launch law)
import json as _json
_wids = [int(t) for t in _json.load(open("~/ids8k.json"))[:2048]]
from mtp import SLICE
_seen, sl = set(), []
for t in _wids:
  if t not in _seen:
    _seen.add(t); sl.append(t)
_base = sl[:]
while len(sl) < SLICE:
  sl += _base
E.init_draft(sl[:SLICE])
E.reset_fresh(_wids[0]); E.stload_trunk()
for ii in E.gdn_idx:
  E._mfill(f"conv{ii}_1", 0, CBLK)
dev.synchronize()
pf_prefill.prefill_batch(E, None, _wids)
dev.synchronize(); E.P._keep.clear()
print("[warm] 2k prefill done", flush=True)

# linearized W columns (numpy, dq with Glin grid + fp32 sdb)
import pack_w4
def linW(tag, blk):
    arr = np.load(f"~/tinygrad-metal/engine0/packed/{tag}{blk}.npy")
    if arr.ndim == 1: arr = arr.reshape(arr.shape[0], -1)
    n = arr.shape[0]; nb = KD >> 8
    row16 = arr.view(np.uint16)
    q = row16[:, :nb*32]
    scp = np.ascontiguousarray(arr[:, 64*nb:96*nb]).view(np.uint32)
    dpp = np.ascontiguousarray(row16[:, nb*48:nb*49]).view(np.float16)
    d = dpp.astype(np.float32)
    Wl = np.empty((n, nb, 32, 8), dtype=np.float32)
    for lc in range(32):
        cc = lc & 3
        qv = q[:, lc::32].astype(np.uint32)
        swv = scp[:, (lc >> 2)::8]
        sdb = d * ((swv >> 28).astype(np.float32) + 0.5) * 0.5 * DELTA
        sidx = (swv >> (7 * cc)) & 0x7F
        spar = (sidx ^ (sidx>>1) ^ (sidx>>2) ^ (sidx>>3) ^ (sidx>>4) ^ (sidx>>5) ^ (sidx>>6)) & 1
        i0 = ((qv & 0xFF) << 2)[:, :, None] + np.arange(4)[None, None, :]
        i1 = ((qv >> 8) << 2)[:, :, None] + np.arange(4)[None, None, :]
        l0 = lut8.astype(np.float32)[i0.reshape(-1)].reshape(n, -1, 4)
        l1 = lut8.astype(np.float32)[i1.reshape(-1)].reshape(n, -1, 4)
        sg = np.empty((n, sidx.shape[1], 8), dtype=np.float32)
        for b in range(7):
            sg[:, :, b] = np.where((sidx >> b) & 1, -1.0, 1.0)
        sg[:, :, 7] = np.where(spar != 0, -1.0, 1.0)
        Wl[:, :, lc, :] = sdb[..., None] * (np.concatenate([l0, l1], axis=2) * sg)
    return Wl.reshape(n, KD)

WgL = linW("fg", i)
# warm-ish: one launch first
pr["p8w4ffn7d_nw8k128"](W7[("fg", i)], W7[("fu", i)], d["w7t_lut"], d["dbg_xq"], d["dbg_sx"],
                        d["w7t_gout"], d["w7t_uout"], global_size=(544, 1, 1), local_size=LS, wait=True)
dev.synchronize()
print("[dump] warm launch OK", flush=True)

for k0 in range(8):
    xqd = np.zeros((M, KD), dtype=np.int8)
    xqd[0, k0] = 127
    P.up("dbg_xq", xqd)
    dev.synchronize(); P._keep.clear()
    pr["p8w4ffn7d_nw8k128"](W7[("fg", i)], W7[("fu", i)], d["w7t_lut"], d["dbg_xq"], d["dbg_sx"],
                            d["w7t_gout"], d["w7t_uout"], global_size=(544, 1, 1), local_size=LS, wait=True)
    g = P.down("w7t_gout", (M, ND), np.float32)[0]
    P._keep.clear()
    # search which W column matches best
    best = []
    for kk in range(16):
        wl = WgL[:, kk] * 127.0
        rel = np.linalg.norm(g - wl) / max(np.linalg.norm(wl), 1e-9)
        best.append((rel, kk))
    best.sort()
    print(f"[dump] xq@k={k0}: best match col {best[0][1]} (rel {best[0][0]:.2e}), 2nd {best[1][1]} ({best[1][0]:.2e})", flush=True)
print("[dump] DONE", flush=True)
