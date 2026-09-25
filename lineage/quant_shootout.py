# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P1e: quant-format GEMV shootout for the r_544 FFN shape [17408,5120].
For each GGML quant type supported by the fork's _GGML_QUANT table, synthesize
raw block bytes, dequant lazily via ggml_data_to_tensor (fused into GEMV by
scheduler, exactly as in-model), time y = x @ W.T in a warm TinyJit (BEAM=1),
and project full-model GEMV time + total model bytes. NO model download here.
"""
import os, sys, time, struct
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("BEAM", "1")
sys.path.insert(0, "~/tinygrad-src")
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import GlobalCounters
from tinygrad.llm.gguf import ggml_data_to_tensor

DEV = Device.DEFAULT
R, K = 17408, 5120            # r_544 GEMV: ffn_gate/up [5120->17408]
N = R * K                     # 89,081,600 elements
ITERS = int(os.getenv("ITERS", "50"))
POOL_ELEMS = 23.189e9         # main-model per-token GEMV pool (this GGUF, measured)
TOTAL_ELEMS = 27.321e9        # all params (this GGUF)
EMBD_BYTES = 0.763e9          # token_embd IQ3_S stays as-is in any whole-file download est
NORM_F32_BYTES = 0.104e9      # F32 norms etc.
BW_PROVEN = 447e9             # proven streaming GB/s on this card

# ggml_type -> (block_elems, block_bytes)  [fork _GGML_QUANT]
TYPES = [(2,"Q4_0"), (12,"Q4_K"), (13,"Q5_K"), (14,"Q6_K"),
         (23,"IQ4_XS"), (8,"Q8_0"), (18,"IQ3_XXS")]
BPQ = {2:4.5, 12:4.5, 13:5.5, 14:6.5625, 23:4.25, 8:8.5, 18:3.0625}  # bits/elem = bytes*8/nelems

def vram(): return GlobalCounters.mem_used_per_device.get(DEV, 0)/1e9

x = Tensor.kaiming_uniform(1, K).half().to(DEV).contiguous().realize()
results = []
print(f"== P1e quant shootout shape [{R},{K}] iters={ITERS} BEAM=1 ==", flush=True)
print(f"start VRAM {vram():.2f} GB", flush=True)

# ---- fp16 reference ----
W16 = (Tensor.kaiming_uniform(R, K) * 0.05).half().to(DEV).contiguous().realize()
j16 = TinyJit(lambda xx: (xx @ W16.T).realize())
j16(x.clone()); j16(x.clone())
Device[DEV].synchronize(); t0 = time.perf_counter()
for _ in range(ITERS): j16(x.clone())
Device[DEV].synchronize(); ms16 = (time.perf_counter()-t0)/ITERS*1e3
del j16, W16
results.append(("fp16_ref", None, ms16, N*2, 16.0))
print(f"fp16_ref: {ms16:7.3f} ms  raw {N*2/1e6:8.1f} MB  eff {N*2/1e6/ms16/1e3:6.1f} GB/s"
      f"  {N/1e6/ms16:7.1f} Melem/ms", flush=True)

iq3_proj = None
for gid, name in TYPES:
    ne, nb = {2:(32,18),12:(256,144),13:(256,176),14:(256,210),
              23:(256,136),8:(32,34),18:(256,98)}[gid]
    nblocks = N // ne
    nbytes = nblocks * nb
    assert nblocks * ne == N
    rng = np.random.default_rng(gid)
    raw = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
    v_before = vram()
    t8 = Tensor(raw).to(DEV).contiguous().realize()   # single <=95MB upload, then raw freed
    del raw
    W = ggml_data_to_tensor(t8, N, gid).reshape(R, K).half()   # lazy: cast fuses into GEMV
    j = TinyJit(lambda xx: (xx @ W.T).realize())
    try:
        t_cap0 = time.perf_counter()
        j(x.clone()); j(x.clone())
        Device[DEV].synchronize()
        print(f"  [{name}] capture+tune {(time.perf_counter()-t_cap0)/1:.0f}s", flush=True)
        t0 = time.perf_counter()
        for _ in range(ITERS): j(x.clone())
        Device[DEV].synchronize()
        ms = (time.perf_counter()-t0)/ITERS*1e3
    finally:
        pass
    dv = vram() - v_before
    gbps = nbytes/1e6/ms
    mel = N/1e6/ms
    proj_pool_ms = POOL_ELEMS/N*ms                      # element-rate projection
    byte_floor_ms = POOL_ELEMS*(BPQ[gid]/8)/BW_PROVEN*1e3
    total_gb = (TOTAL_ELEMS-1.271e9)*(BPQ[gid]/8)/1e9 + EMBD_BYTES/1e9 + NORM_F32_BYTES/1e9
    results.append((name, gid, ms, nbytes, BPQ[gid]))
    print(f"{name}: {ms:7.3f} ms  raw {nbytes/1e6:8.1f} MB ({dv:+.2f}GB vram)  eff {gbps:6.1f} GB/s"
          f"  {mel:7.1f} Melem/ms  proj_pool {proj_pool_ms:6.2f} ms  byte_floor {byte_floor_ms:6.2f} ms"
          f"  model~{total_gb:5.2f} GB", flush=True)
    del j, W, t8
    if name == "IQ3_XXS": iq3_proj = proj_pool_ms

print("\n== DECISION (win = proj_pool < IQ3_XXS*0.70 AND model <=20GB) ==", flush=True)
for name, gid, ms, nbytes, bpq in results:
    if gid is None: continue
    proj = POOL_ELEMS/N*ms
    total_gb = (TOTAL_ELEMS-1.271e9)*(bpq/8)/1e9 + EMBD_BYTES/1e9 + NORM_F32_BYTES/1e9
    win = proj < iq3_proj*0.70 and total_gb <= 20.0
    print(f"  {name}: proj {proj:6.2f} ms ({(1-proj/iq3_proj)*100:+.0f}% vs IQ3_XXS)"
          f"  model~{total_gb:5.2f} GB  -> {'WIN' if win else 'no'}", flush=True)
print("DONE", flush=True)
