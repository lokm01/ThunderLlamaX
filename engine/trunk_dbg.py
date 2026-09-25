# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""trunk_dbg: staged first token — isolate faulting kernel or queue-depth."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from trunk import TrunkEngine, CTX, VOCAB
from engine0 import dev

snap = np.load(os.getenv("SNAP", "~/w1b_state_2k.npz"))
theta = float(snap["theta"].reshape(-1)[0])
E = TrunkEngine(theta)
E.restore(snap)
d, W, pr, LS = E.P.d, E.W, E.pr, (256,1,1)
P0 = int(snap["P"].reshape(-1)[0])
print(f"[dbg] P={P0} tok={int(snap['ids'].reshape(-1)[-1])}", flush=True)

def Wt(name): print(f"[ok] {name}", flush=True)

# staged: embed
pr["h_embed"](W[("emb",0)], d["grid512"], d["tok_slot"], d["x0"], global_size=(1,1,1), local_size=LS, wait=True); Wt("h_embed")
# staged: GDN block E.gdn_idx[0]
i = E.gdn_idx[0]
pr["k0_norm"](d["x0"], W[("nw1",i)], d["xh"], global_size=(1,1,1), local_size=LS, wait=True); Wt("k0 gdn")
E.gdn(i, d["x0"], d["x1"], f"conv{i}_0", f"conv{i}_1")
dev.synchronize(); Wt(f"gdn blk {i} (10 pipelined)")
# staged: attn block E.attn_idx[0]
i = E.attn_idx[0]
E.attn(i, d["x1"], d["x0"])
dev.synchronize(); Wt(f"attn blk {i} (8 pipelined)")
# staged: head + argmax
E.head(d["x0"])
dev.synchronize(); Wt("head+argmax")
tok = E.P.down("tok_slot", (1,), np.int32)
pos = E.P.down("pos_slot", (1,), np.int32)
print(f"[dbg] after 1 token-ish: tok_slot={tok} pos_slot={pos} (expect pos={P0})", flush=True)

# now FULL tokens with periodic waits every 128 launches
def token_chunked(it, CHUNK=128):
    n = [0]
    def maybe_wait():
        n[0] += 1
        if n[0] % CHUNK == 0: dev.synchronize()
    orig = E.token
    # manual: embed
    pr["h_embed"](W[("emb",0)], d["grid512"], d["tok_slot"], d["x0"], global_size=(1,1,1), local_size=LS); maybe_wait()
    par = it & 1; cur = 0
    for i in range(64):
        if i in E.qtypes: E.attn(i, d[f"x{cur}"], d[f"x{cur^1}"])
        else: E.gdn(i, d[f"x{cur}"], d[f"x{cur^1}"], f"conv{i}_{par}", f"conv{i}_{par^1}")
        maybe_wait()
        cur ^= 1
    E.head(d[f"x{cur}"], wait=True)

# per-block-synced full token to find the faulting block
pr["h_embed"](W[("emb",0)], d["grid512"], d["tok_slot"], d["x0"], global_size=(1,1,1), local_size=LS, wait=True)
par, cur = 0, 0
for i in range(64):
    if i in E.qtypes: E.attn(i, d[f"x{cur}"], d[f"x{cur^1}"])
    else: E.gdn(i, d[f"x{cur}"], d[f"x{cur^1}"], f"conv{i}_{par}", f"conv{i}_{par^1}")
    cur ^= 1
    dev.synchronize()
    print(f"[blk] {i} {'attn' if i in E.qtypes else 'gdn'} ok", flush=True)
E.head(d[f"x{cur}"], wait=True)
tok = E.P.down("tok_slot", (1,), np.int32)
print(f"[dbg] full staged token done, tok={tok}", flush=True)
print("[dbg] staged path works", flush=True)
