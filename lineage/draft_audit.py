# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Draft-only diagnostic: loads JUST blk.64 weights, tests MTP head numerics. No main model."""
import sys
sys.path.insert(0,"~/tinygrad-src") if sys.platform=="darwin" else None
import struct
import numpy as np
from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.model import TransformerBlock, Linear
from tinygrad.nn import RMSNorm
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_QUANT, _GGML_NATIVE
from tinygrad.nn.state import load_state_dict
from tinygrad.uop.ops import UOp

GGUF="~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"

def _rstr(r):
    n=struct.unpack("<Q",r.read(8))[0]; return r.read(n).decode()
def _rd_val(r, typ):
    if typ==8: return _rstr(r)
    if typ==9:
        it=struct.unpack("<I",r.read(4))[0]; n=struct.unpack("<Q",r.read(8))[0]
        return [_rd_val(r,it) for _ in range(n)]
    fmt={0:"c",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}[typ]
    nb=struct.calcsize("<"+fmt)
    return struct.unpack("<"+fmt, r.read(nb))[0]

def extract(path, prefix):
    r=open(path,"rb"); assert r.read(4)==b"GGUF"
    struct.unpack("<I",r.read(4)); nt=struct.unpack("<Q",r.read(8))[0]; nk=struct.unpack("<Q",r.read(8))[0]
    align=32
    for _ in range(nk):
        k=_rstr(r); t=struct.unpack("<I",r.read(4))[0]; v=_rd_val(r,t)
        if k=="general.alignment": align=int(v)
    infos=[]
    for _ in range(nt):
        nm=_rstr(r); nd=struct.unpack("<I",r.read(4))[0]
        dims=tuple(struct.unpack("<Q",r.read(8))[0] for _ in range(nd))
        typ=struct.unpack("<I",r.read(4))[0]; off=struct.unpack("<Q",r.read(8))[0]
        infos.append((nm,dims,typ,off))
    data_start=(r.tell()+align-1)//align*align
    out={}
    from tinygrad.helpers import prod
    for nm,dims,typ,off in infos:
        if not nm.startswith(prefix): continue
        n=prod(dims)
        if typ in _GGML_NATIVE: nbytes=_GGML_NATIVE[typ].itemsize*n
        else:
            ne,nb=_GGML_QUANT[typ]; nbytes=(n//ne)*nb
        r.seek(data_start+off); raw=r.read(nbytes)
        t8=Tensor(np.frombuffer(raw,np.uint8).copy())
        t=ggml_data_to_tensor(t8, n, typ).reshape(*reversed(dims))
        # bring to numpy for CPU-side inspection (weights are quantized -> dequant via tinygrad on CPU)
        out[nm]=t
    r.close()
    return out

print("extracting blk.64 tensors (CPU)...", flush=True)
sd = extract(GGUF, "blk.64")
for k in sorted(sd.keys()):
    t = sd[k]
    print(f"  {k}: shape={t.shape} dtype={t.dtype}", flush=True)

# key checks BEFORE any device math:
# 1. eh_proj present & shape sane?
w = sd.get("blk.64.nextn.eh_proj.weight")
print("\neh_proj.weight present:", w is not None, flush=True)
if w is not None:
    print("  raw shape:", w.shape, "dtype:", w.dtype, flush=True)

# 2. norm weights: are they real vectors or something odd?
for nm in ["blk.64.nextn.enorm.weight","blk.64.nextn.hnorm.weight","blk.64.nextn.shared_head_norm.weight"]:
    t=sd.get(nm)
    if t is None: print(f"{nm}: MISSING", flush=True); continue
    # dequant on CPU: cast chain realized lazily; force via numpy through tinygrad CPU
    tn = t.to(None).realize()
    arr = tn.numpy()
    print(f"{nm}: shape={arr.shape} absmax={abs(arr).max():.4f} mean={arr.mean():.4f} first5={arr.flatten()[:5].tolist()}", flush=True)

# 3. eh_proj numeric content (dequant to fp32 numpy)
if w is not None:
    wn = w.to(None).realize().numpy()
    print(f"\neh_proj dequant stats: shape={wn.shape} absmax={abs(wn).max():.4f} mean={wn.mean():.4f}", flush=True)
    # column-half analysis: which half feeds e vs h depends on layout!
    # Linear computes x @ W.T with W (out,in). cat order [e(5120), h(5120)] -> in-dim split:
    #   cols 0..5119 of W.T correspond to e-half, 5120..10239 to h-half.
    # In W (out x in) row-major that's W[:, :5120] vs W[:, 5120:] AFTER correct orientation.
    # But if reshape/reversed-dims put it as (in,out)=(10240,5120) we must transpose!
    if wn.shape == (10240, 5120):
        Wt = wn.T  # (5120 out, 10240 in)
    else:
        Wt = wn
    e_half = np.abs(Wt[:, :5120]).mean()
    h_half = np.abs(Wt[:, 5120:]).mean()
    print(f"  |W| mean e-cols={e_half:.5f} h-cols={h_half:.5f}  (both should be similar & nonzero)", flush=True)
    # check for all-zero halves
    print(f"  e-half zeros: {(Wt[:, :5120]==0).all()}, h-half zeros: {(Wt[:, 5120:]==0).all()}", flush=True)

# 4. shared_head_norm + a sanity look at blk64 attn_q (is the DRAFT BLOCK itself real weights?)
for nm in ["blk.64.attn_q.weight","blk.64.ffn_gate.weight"]:
    t=sd.get(nm)
    if t is None: continue
    tn=t.to(None).realize().numpy()
    print(f"{nm}: shape={tn.shape} absmax={abs(tn).max():.4f} mean={tn.mean():.5f}", flush=True)

print("\nDONE (CPU-side audit complete)", flush=True)
