# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import sys, struct
fn = "models/Qwen3.8-27B-IQ3_XXS.gguf"
f = open(fn, "rb")
def rd(fmt): 
    s = struct.calcsize(fmt); return struct.unpack(fmt, f.read(s))
magic, ver = rd("<4sI")
n_tensors, n_kv = rd("<Qq") if ver >= 3 else rd("<ii")
def rstr():
    n, = rd("<Q"); return f.read(n).decode()
def rval(t):
    fmts = {0:"<b",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<B",8:"<B",10:"<H"}
    if t == 9:  # array
        et, n = rd("<II"); return [rval(et) for _ in range(n)]
    return rd(fmts[t])[0]
for _ in range(n_kv):
    k = rstr(); vt, = rd("<I"); v = rval(vt)
    if any(w in k for w in ["head_count","embedding_length","block_length","key_length","value_length","expert_count","expert_used","norm","feed_forward","rope.dimension","key_length_ml","value_length_ml","conv"]):
        print(k, "=", v)
