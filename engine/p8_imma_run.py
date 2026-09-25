# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P8 IMMA discriminator runner: W4A8 int8-tensor-core GEMM at the prefill FFN
shape (M=64, K=5120, N=17408) vs the SHIPPED pfg3_ffn_r7_m64_nw4k128 fp16
stream (real weights, the r7b_classes recipe). Random int4 weights; numerics
vs a numpy reference of the same quantized data; synced min-of-8 timing.
Info-only (Tier-2): NOT wired into the engine."""
import os, sys, time
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
rng = np.random.default_rng(7)

from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram
IMMA = "p8_imma_ffn_m64_nw8k128"
lib = open(f"~/tinygrad-metal/engine0/{IMMA}.cubin", "rb").read()
pr[IMMA] = NVProgram(dev, TinyELF(lib=lib, name=IMMA, target=dev.renderer.target, signature=tuple()))

KD, ND, M, NCH = 5120, 17408, 64, 40
NB = KD >> 8  # 20
# ---- W4 random (unit layout): [group][chunk][32][4] u32 ----
nib = rng.integers(0, 16, (ND, KD), dtype=np.uint8)         # logical nibbles
out = np.zeros((ND // 8, NCH, 32, 4), dtype=np.uint32)
for c in range(NCH):
    for qc in range(4):
        blk = nib[:, c*128 + qc*32 : c*128 + qc*32 + 32]    # [ND,32]
        v = np.zeros((ND, 4), dtype=np.uint32)
        for w in range(32):
            v[:, w >> 3] |= (blk[:, w].astype(np.uint32) << (4 * (w & 7)))
        out[:, c, qc::4, :] = v.reshape(ND // 8, 8, 4)
w4u = out.tobytes()
swd = (np.abs(rng.standard_normal((ND, NB))) * 0.01 + 0.002).astype(np.float16)
# ---- X random fp16 -> per-(row,chunk) int8 quant ----
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
# ---- numpy reference (fp64 of the QUANTIZED values) ----
wdeq = (nib.astype(np.float64) - 8.0) * swd.astype(np.float64)[:, (np.arange(KD) >> 8)]
xdeq = xq.astype(np.float64) * sx.astype(np.float64)[:, (np.arange(KD) // 128)]
ref = xdeq @ wdeq.T
P.up("imma_w4", np.frombuffer(w4u, dtype=np.uint8)); P.up("imma_swd", swd)
P.up("imma_xq", xq); P.up("imma_sx", sx); P.up("imma_rs", rowsum)
P.poison("imma_out", M*ND*4, np.float32, 7.7e31)
dev.synchronize(); P._keep.clear()

def run_imma(obuf):
  pr[IMMA](d["imma_w4"], d["imma_swd"], d["imma_xq"], d["imma_sx"], d["imma_rs"], obuf,
           global_size=(ND//64, 1, 1), local_size=LS, wait=True)
P._keep.clear()

# ---- shipped fp16 kernel on real weights (census recipe) ----
# warm the plan + channel exactly like r7b_classes (a real 2k prefill first —
# the never-executed plan world faulted the first standalone ffn launch)
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
for i in E.gdn_idx:
  E._mfill(f"conv{i}_1", 0, CBLK)
dev.synchronize()
pf_prefill.prefill_batch(E, None, _wids)
dev.synchronize(); E.P._keep.clear()
print("[warm] 2k prefill done", flush=True)
ffn_blocks = [i for i in E.gdn_idx if ("fg", i) in W7][:6]
P.up("xh128t", (rng.standard_normal(M*KD) * 0.7).astype(np.float16))
P.poison("gact128b", M*17408*2, np.float16, 7.7)
dev.synchronize(); P._keep.clear()
def run_ffn(obuf, i):
  pr["pfg3_ffn_r7_m64_nw4k128"](W7[("fg", i)], W7[("fu", i)], d["gridf"], d["xh128t"], obuf,
      global_size=(1088,1,1), local_size=(128,1,1), wait=True)

def bench(name, fn, blocks):
  # per-launch synced min-of-10 (the fault-isolating form: every launch is
  # followed by its own dev.synchronize + progress print)
  ts = []
  for k, i in enumerate(blocks):
    fn(i); dev.synchronize()
    t0 = time.perf_counter(); fn(i); dev.synchronize(); ts.append(time.perf_counter() - t0)
    print(f"[bench:{name}] launch {k} ok", flush=True)
  return min(ts)

p_ffn  = bench("ffn",  lambda i: run_ffn(d["gact128b"], i), ffn_blocks)
print("[phase] ffn bench done", flush=True)
# ---- IMMA numerics + bench LAST (fault-isolated: if the repeat-launch fault
# class fires, we keep the ffn number + partial imma timings) ----
run_imma(d["imma_out"])
got = P.down("imma_out", (M, ND), np.float32)
rel = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9)
print(f"[imma] numerics relerr vs fp64-of-quantized: {rel:.3e} (absmax {np.abs(got-ref).max():.4f})", flush=True)
P._keep.clear()
try:
  ts = []
  for k in range(8):
    t0 = time.perf_counter(); run_imma(d["imma_out"]); dev.synchronize(); ts.append(time.perf_counter() - t0)
    print(f"[bench:imma] launch {k} ok {ts[-1]*1e6:.1f}us", flush=True)
  p_imma = min(ts)
except Exception as e:
  print(f"[bench:imma] FAULT at launch {k}: {e!r} (partial min {min(ts)*1e6:.1f}us)" if ts else f"[bench:imma] FAULT: {e!r}", flush=True)
  p_imma = min(ts) if ts else None
print("[phase] imma bench done", flush=True)
w_imma = (ND*KD*0.5 + ND*NB*2)/1e6; w_ffn = 2*8704*KD*0.625/1e6
print(f"[perf] HMMA packed7 nw4   : {p_ffn*1e6:8.1f} us/launch | W {w_ffn:.1f} MB -> {w_ffn/1e3/p_ffn:6.1f} GB/s | pool48 {p_ffn*48*1e3:6.1f} ms/chunk", flush=True)
if p_imma is not None:
  print(f"[perf] IMMA W4A8 m16n8k32 : {p_imma*1e6:8.1f} us/launch | W {w_imma:.1f} MB -> {w_imma/1e3/p_imma:6.1f} GB/s | pool48 {p_imma*48*1e3:6.1f} ms/chunk", flush=True)
  print(f"[verdict] speedup x{p_ffn/max(p_imma,1e-12):.3f} ({(p_ffn-p_imma)*48*1e3:+.1f} ms/chunk pool delta)", flush=True)
else:
  print("[verdict] imma timing unavailable (fault class) — see bare-world fallback", flush=True)
print("[imma] DONE", flush=True)
