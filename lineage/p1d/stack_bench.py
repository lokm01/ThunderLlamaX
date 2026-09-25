# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P1d step 1: stack RAW IQ3_XXS quant blocks (pre-dequant) for ffn_gate|ffn_up.
Cat at byte level -> single lazy-dequant expr -> ONE big GEMV kernel reading 68MB.
Compare vs two separate GEMVs in one jit. Also dump GGUF dtypes of GDN inputs."""
import os, sys, time, struct
sys.path.insert(0, "~/tinygrad-src")
import numpy as _np
from tinygrad import Tensor, Device
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import getenv, prod
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_NATIVE, _GGML_QUANT

ITERS = int(getenv("ITERS", 50))
MODEL = os.getenv("MODEL", "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf")
print(f"[cfg] BEAM={getenv('BEAM',0)} DEV={Device.DEFAULT} ITERS={ITERS}", flush=True)

# --- targeted GGUF header parse (u32 type enum, u64 dims/offsets) ---
def rstr(r):
    n = struct.unpack("<Q", r.read(8))[0]; return r.read(n).decode()
def rdval(r, t):
    if t == 8: return rstr(r)
    if t == 9:
        et = struct.unpack("<I", r.read(4))[0]; n = struct.unpack("<Q", r.read(8))[0]
        return [rdval(r, et) for _ in range(n)]
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t]
    v = struct.unpack("<"+{0:"c",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}[t], r.read(sz))[0]
    return v

r = open(MODEL, "rb"); assert r.read(4) == b"GGUF"
struct.unpack("<I", r.read(4)); nt = struct.unpack("<Q", r.read(8))[0]; nk = struct.unpack("<Q", r.read(8))[0]
align = 32
for _ in range(nk):
    k = rstr(r); t = struct.unpack("<I", r.read(4))[0]; v = rdval(r, t)
    if k == "general.alignment": align = int(v)
infos = []
for _ in range(nt):
    nm = rstr(r); nd = struct.unpack("<I", r.read(4))[0]
    dims = tuple(struct.unpack("<Q", r.read(8))[0] for _ in range(nd))
    typ = struct.unpack("<I", r.read(4))[0]; off = struct.unpack("<Q", r.read(8))[0]
    infos.append((nm, dims, typ, off))
data_start = (r.tell()+align-1)//align*align

want = ["blk.0.ffn_gate.weight","blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight","blk.1.attn_qkv.weight","blk.1.attn_gate.weight",
        "blk.1.ssm_alpha.weight","blk.1.ssm_beta.weight"]
by_name = {nm:(dims,typ,off) for nm,dims,typ,off in infos}
for w in want:
    if w in by_name:
        d,t,o = by_name[w]
        print(f"[hdr] {w:28s} dims={d} typ={t}", flush=True)
    else:
        print(f"[hdr] {w:28s} MISSING", flush=True)

def raw_bytes(name):
    dims, typ, off = by_name[name]
    n = prod(dims)
    ne, nb = _GGML_QUANT[typ]
    nbytes = (n//ne)*nb
    r.seek(data_start+off); b = r.read(nbytes)
    return _np.frombuffer(b, _np.uint8).copy(), n, typ, dims

rawg, ng, tg, dg = raw_bytes("blk.0.ffn_gate.weight")
rawu, nu, tu, du = raw_bytes("blk.0.ffn_up.weight")
assert tg == tu == 18, (tg, tu)
assert dg == du, (dg, du)   # dims equal => same K => row-aligned block concat
N, K = dg[1], dg[0]         # gguf dims [K, N]; tinygrad shape (N, K)
print(f"[shape] N={N} K={K} raw each={len(rawg)} ({N*K//256}*98={N*K//256*98})", flush=True)

DEV = Device.DEFAULT
t8g = Tensor(rawg).to(DEV).realize()
t8u = Tensor(rawu).to(DEV).realize()
Wg = ggml_data_to_tensor(t8g, ng, tg).reshape(N, K).half()
Wu = ggml_data_to_tensor(t8u, nu, tu).reshape(N, K).half()

# stacked: cat RAW BLOCKS (byte level), then ONE dequant expr over the union
t8gu = t8g.cat(t8u)  # (2*nblocks, 98) uint8 -- pure byte concat
Wgu = ggml_data_to_tensor(t8gu, ng+nu, tg).reshape(2*N, K).half()

x = Tensor.kaiming_uniform(1, K).half().to(DEV).contiguous().realize()

# correctness first (eager)
y_sep = (x @ Wg.T).cat(x @ Wu.T, dim=-1).realize()
y_stk = (x @ Wgu.T).realize()
d = (y_sep - y_stk).abs().max().float().item()
a1, a2 = y_sep.argmax().int().item(), y_stk.argmax().int().item()
print(f"[check] maxdiff={d:.6f} argmax {a1} vs {a2} {'OK' if d < 0.05 and a1==a2 else 'FAIL'}", flush=True)

def run(tag, jit, rawbytes):
    for _ in range(5): jit(x.clone())
    Device[DEV].synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS): jit(x.clone())
    Device[DEV].synchronize()
    dt = (time.perf_counter()-t0)/ITERS
    print(f"[time] {tag:26s} {dt*1e3:7.3f} ms  {rawbytes/dt/1e9:7.1f} GB/s", flush=True)

j_sep = TinyJit(lambda xx: ((xx @ Wg.T).realize(), (xx @ Wu.T).realize()))
run("2 separate GEMVs (34MB ea)", j_sep, 2*N*K*98)

j_stk = TinyJit(lambda xx: (xx @ Wgu.T).realize())
run("1 stacked GEMV (68MB raw)", j_stk, 2*N*K*98)

# repeat stacked second time (order effect check)
run("1 stacked GEMV again", j_stk, 2*N*K*98)
print("DONE_STACKBENCH", flush=True)
