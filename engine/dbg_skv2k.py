# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
os.environ.setdefault("SKV", "1")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from mtp import MTPEngine
from trunk import CTX
from engine0 import dev
LS = (256,1,1)
snap = np.load("~/w1b_state_2k.npz")
theta = float(snap["theta"].reshape(-1)[0])
E = MTPEngine(theta)
P, W, d, pr = E.P, E.W, E.P.d, E.pr
# restore only what block-3 attention needs + states for norm input realism
P.up("pos_slot", np.array([int(snap["P"].reshape(-1)[0]) - 1], dtype=np.int32))
i0 = E.attn_idx[0]
kvs = snap[f"kv{E.attn_idx.index(i0)}"].astype(np.float16).reshape(2, 4, 2048, 256)
kvbig = np.full((2, 4, 2304, 256), 7.7, dtype=np.float16); kvbig[:, :, :2048] = kvs
P.up(f"kv{i0}", kvbig.reshape(-1))
dev.synchronize()
# fake x input: use random (numerics irrelevant; fault isolation)
rng = np.random.default_rng(1)
P.up("x0", (0.1*rng.standard_normal(5120)).astype(np.float32))
P.poison("ao_row", 6144*2, np.float16, 7.7)
P.poison("attn_out", 5120*2, np.float16, 7.7)
dev.synchronize()
print("[dbg] eager old-path a_attn...", flush=True)
pr["k0_norm"](d["x0"], W[("nw1",i0)], d["xh"], global_size=(1,1,1), local_size=LS)
qkname = "aq6k8" if E.qtypes[i0] == 14 else "aq3k8"
pr[qkname](W[("q",i0)], W[("k",i0)], W[("v",i0)], d["gridf"], d["xh"], d["qrow"], d["k_row"], d["v_row"], global_size=(1792,1,1), local_size=LS)
pr["a_attn"](d["qrow"], d["k_row"], d["v_row"], W[("qnw",i0)], W[("knw",i0)], d["freqs"], d[f"kv{i0}"], d["pos_slot"], d["ao_row"], global_size=(24,1,1), local_size=LS, wait=True)
old = P.down("ao_row", (6144,), np.float16).copy()
print(f"[dbg] old OK absmax={np.abs(old.astype(np.float32)).max():.3f}", flush=True)
# now trio on a fresh kv copy (rows<=pos identical)
P.up(f"kv{i0}", kvbig.reshape(-1))
P.poison("ao_row", 6144*2, np.float16, 7.7)
dev.synchronize()
pr["k0_norm"](d["x0"], W[("nw1",i0)], d["xh"], global_size=(1,1,1), local_size=LS)
pr[qkname](W[("q",i0)], W[("k",i0)], W[("v",i0)], d["gridf"], d["xh"], d["qrow"], d["k_row"], d["v_row"], global_size=(1792,1,1), local_size=LS)
print("[dbg] launching spk_pre1/spk_a1/spk_c1 eagerly...", flush=True)
pr["spk_pre1"](d["qrow"], d["k_row"], d["v_row"], W[("qnw",i0)], W[("knw",i0)], d["freqs"], d[f"kv{i0}"], d["pos_slot"], d["qw1"], global_size=(24,1,1), local_size=LS)
pr["spk_a1"](d[f"kv{i0}"], d["qw1"], d["pos_slot"], d["pm1"], d["ps1"], d["pA1"], global_size=(128,1,1), local_size=LS)
pr["spk_c1"](d["pm1"], d["ps1"], d["pA1"], d["qrow"], d["ao_row"], global_size=(24,1,1), local_size=LS, wait=True)
new = P.down("ao_row", (6144,), np.float16).copy()
rel = np.abs(new.astype(np.float64)-old.astype(np.float64)).max()/max(np.abs(old.astype(np.float64)).max(),1e-30)
print(f"[dbg] trio OK relerr={rel:.3e}", flush=True)
print("[dbg done]", flush=True)
