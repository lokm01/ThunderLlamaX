# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import sys, time
sys.path.insert(0,"~/tinygrad-src")
from tinygrad import Tensor, UOp, TinyJit
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer

f='~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf'
pc=time.perf_counter
model, kv = Transformer.from_gguf(f, 1024)
tok = SimpleTokenizer.from_ggup_kv if hasattr(SimpleTokenizer,'from_ggup_kv') else SimpleTokenizer.from_gguf_kv
tok = SimpleTokenizer.from_gguf_kv(kv)

ids=[0]+tok.encode("The theory of relativity transformed our understanding of space and time.")

# ---- build a forward returning next-token id, single token, jitted like rollout ----
temp=Tensor([0.0])
v_sp=UOp.variable("sp",0,model.max_context-2)
roll_jit=TinyJit(model.forward)

def step(tok_tensor, pos):
    return roll_jit(tok_tensor, v_sp.bind(pos), temp).realize()

# reference run: 8 sequential tokens from prompt end
t=Tensor(ids+[0]*(model.max_context-len(ids)), dtype="int32").reshape(1,model.max_context)
# prefill whole prompt through prefill path (chunk 32 -> single chunk here)
out=model(t[:, 0:len(ids)], UOp.variable("startpos",0,model.max_context-2).bind(0), temp).realize()
seq=[]
cur=int(out.item())
seq.append(cur)
for i in range(7):
    cur=int(step(Tensor([[cur]],dtype="int32"), len(ids)+i).item())
    seq.append(cur)
print("REFERENCE:", seq, flush=True)

# ---- now: 3 tokens, snapshot, 2 tokens, RESTORE, 2 tokens -> must equal reference[3:8] ----
cur=int(out.item())
for i in range(2):
    cur=int(step(Tensor([[cur]],dtype="int32"), len(ids)+i).item())
print("after 2 extra:", cur, flush=True)

blks=[b for b in model.blk if hasattr(b,"conv_state")]
t0=pc()
snap=[(b.conv_state.clone(), b.recurrent_state.clone()) for b in blks]
print(f"snapshot {pc()-t0:.2f}s", flush=True)

# advance 2 (dirty) tokens
dirty=[cur]
for i in range(2):
    cur=int(step(Tensor([[cur]],dtype="int32"), len(ids)+2+i).item())
    dirty.append(cur)
print("dirty tokens:", dirty, flush=True)

t0=pc()
for b,(cs,rs) in zip(blks,snap):
    b.conv_state.assign(cs).realize()
    b.recurrent_state.assign(rs).realize()
print(f"restore {pc()-t0:.2f}s", flush=True)

# resume from seq[2] position: last good token emitted was seq[2] at index len(ids)+1
cur2=seq[2]
resumed=[]
for i in range(5):
    nxt=int(step(Tensor([[cur2]],dtype="int32"), len(ids)+2+i).item())
    resumed.append(nxt); cur2=nxt
expect=seq[3:8]
print("RESUMED :", resumed, flush=True)
print("EXPECTED:", expect, flush=True)
print("RESTORE CORRECT" if resumed==expect else "RESTORE MISMATCH", flush=True)
print("DONE", flush=True)
