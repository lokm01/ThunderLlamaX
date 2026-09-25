# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""p8_imma numerics debug (bare world — no engine boot): run the cubin once,
compare vs numpy ref of the same quantized data, print error STRUCTURE
(per row-group / col-block / specific cells) to pin the fragment-map bug."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev, Bufs
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

P = Bufs()
IMMA = "p8_imma_ffn_m64_nw8k128"
lib = open(f"~/tinygrad-metal/engine0/{IMMA}.cubin", "rb").read()
pr = NVProgram(dev, TinyELF(lib=lib, name=IMMA, target=dev.renderer.target, signature=tuple()))
rng = np.random.default_rng(7)
KD, ND, M, NCH = 5120, 17408, 64, 40
NB = KD >> 8
nib = rng.integers(0, 16, (ND, KD), dtype=np.uint8)
out = np.zeros((ND // 8, NCH, 32, 4), dtype=np.uint32)
for c in range(NCH):
    for qc in range(4):
        blk = nib[:, c*128 + qc*32 : c*128 + qc*32 + 32]
        v = np.zeros((ND, 4), dtype=np.uint32)
        for w in range(32):
            v[:, w >> 3] |= (blk[:, w].astype(np.uint32) << (4 * (w & 7)))
        out[:, c, qc::4, :] = v.reshape(ND // 8, 8, 4)
swd = (np.abs(rng.standard_normal((ND, NB))) * 0.01 + 0.002).astype(np.float16)
xf = (rng.standard_normal((M, KD)) * 0.35).astype(np.float16)
xq = np.zeros((M, KD), dtype=np.int8); sx = np.zeros((M, NCH), dtype=np.float32)
rowsum = np.zeros((M, NCH), dtype=np.int32)
for c in range(NCH):
    seg = xf[:, c*128:(c+1)*128].astype(np.float32)
    s = np.maximum(np.abs(seg).max(axis=1), 1e-6) / 127.0
    sx[:, c] = s
    qq = np.clip(np.rint(seg / s[:, None]), -127, 127).astype(np.int8)
    xq[:, c*128:(c+1)*128] = qq
    rowsum[:, c] = qq.sum(axis=1).astype(np.int32)
wdeq = (nib.astype(np.float64) - 8.0) * swd.astype(np.float64)[:, (np.arange(KD) >> 8)]
xdeq = xq.astype(np.float64) * sx.astype(np.float64)[:, (np.arange(KD) // 128)]
ref = xdeq @ wdeq.T
P.up("imma_w4", np.frombuffer(out.tobytes(), dtype=np.uint8)); P.up("imma_swd", swd)
P.up("imma_xq", xq); P.up("imma_sx", sx); P.up("imma_rs", rowsum)
P.poison("imma_out", M*ND*4, np.float32, 7.7e31)
dev.synchronize(); P._keep.clear()
d = P.d
pr(d["imma_w4"], d["imma_swd"], d["imma_xq"], d["imma_sx"], d["imma_rs"], d["imma_out"],
   global_size=(ND//64,1,1), local_size=(256,1,1), wait=True)
got = P.down("imma_out", (M, ND), np.float32)
P._keep.clear()
err = np.abs(got - ref); rel = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9)
print(f"[dbg] relerr {rel:.3e} absmax {err.max():.4f}")
# structure: per 16-row group and per 1024-col block
for rg in range(4):
    e = err[rg*16:(rg+1)*16]; r = np.linalg.norm(ref[rg*16:(rg+1)*16])
    print(f"[dbg] rows {rg*16}-{rg*16+15}: rel {np.linalg.norm(e)/max(r,1e-9):.3e} absmax {e.max():.4f}")
for cb in range(0, ND, 2048):
    e = err[:, cb:cb+2048]; r = np.linalg.norm(ref[:, cb:cb+2048])
    print(f"[dbg] cols {cb}-{cb+2047}: rel {np.linalg.norm(e)/max(r,1e-9):.3e}")
# cell-level: compare per-chunk contributions for (m=0, n=0)
m_, n_ = 0, 0
perchunk_ref = []
for c in range(NCH):
    xs_ = xdeq[m_, c*128:(c+1)*128]
    ws_ = wdeq[n_, c*128:(c+1)*128]
    perchunk_ref.append(float(xs_ @ ws_))
print(f"[dbg] cell (0,0): kernel {got[m_,n_]:.4f} ref {ref[m_,n_]:.4f}")
print(f"[dbg] per-chunk ref partials (first 6): {[f'{v:.3f}' for v in perchunk_ref[:6]]}")
cum = np.cumsum(perchunk_ref)
# reconstruct what the kernel would give if it dropped/scaled one chunk class
for hypo, name in [("half", "sx*sw applied twice?"), ("no8", "offset-8 missing?")]:
    if hypo == "half":
        alt = sum(perchunk_ref) * 0.5
    else:
        alt = sum(perchunk_ref) + 8.0*float((xq[m_].astype(np.float64) * swd[n_].astype(np.float64)[np.arange(KD)>>8]).sum())
    print(f"[dbg] hypothesis {name}: {alt:.4f}")
print("[dbg] DONE")
