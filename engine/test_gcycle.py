# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W1-c G_CYCLE gate: graph-replayed 60-token decode vs stock baseline + timing."""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from trunk_w1c import TrunkEngineW1C
from trunk import CTX
from engine0 import dev
from gcycle import GCycleEngine

snap = np.load(os.getenv("SNAP", "~/w1b_state_2k.npz"))
theta = float(snap["theta"].reshape(-1)[0])
print("[gcycle] loading engine weights (~13GB)...", flush=True)
t0 = time.perf_counter()
E = TrunkEngineW1C(theta)
print(f"[gcycle] engine loaded in {time.perf_counter()-t0:.1f}s", flush=True)
G = GCycleEngine(E)

def fresh():
  E.restore(snap)
  G.build()

P0 = int(snap["P"].reshape(-1)[0]); base = snap["base_out"].reshape(-1).tolist()
NTOK = 60

# ---- correctness: per-token waits first (safest first run; skipped with SKIPVAL=1) ----
if not os.getenv("SKIPVAL"):
  print("== correctness pass (wait per token) ==", flush=True)
  fresh()
  dev.synchronize()
  G.run_tokens(NTOK, wait_each=True)
  hist = E.P.down("tok_hist", (CTX+128,), np.int32)
  eng = hist[P0-1:P0-1+NTOK]
  agree = int((eng == np.array(base)).sum())
  print(f"[agree] graph replay vs stock baseline: {agree}/{NTOK} exact, first divergence at {next((k for k in range(NTOK) if eng[k] != base[k]), None)}", flush=True)
  assert agree == NTOK, "graph replay broke exactness"
hist = E.P.down("tok_hist", (CTX+128,), np.int32)
eng = hist[P0-1:P0-1+NTOK]
agree = int((eng == np.array(base)).sum())
first_div = next((k for k in range(NTOK) if eng[k] != base[k]), None)
print(f"[agree] graph replay vs stock baseline: {agree}/{NTOK} exact, first divergence at token {first_div}", flush=True)
if agree < NTOK:
  print(f"[engine toks] {eng.tolist()}", flush=True)
  print(f"[base   toks] {base}", flush=True)

# ---- speed: pipelined (one wait at the end) ----
SYNC_EVERY = int(os.getenv("SYNC_EVERY", "1"))
print(f"== timing (60-token graph replays, sync_every={SYNC_EVERY}) ==", flush=True)
times = []
for rep in range(3):
  fresh()
  dev.synchronize()
  t0 = time.perf_counter()
  G.run_tokens(NTOK, sync_every=SYNC_EVERY)
  dt = time.perf_counter() - t0
  times.append(dt)
  print(f"[time] rep{rep}: {dt*1e3:.1f} ms -> {NTOK/dt:.2f} tok/s", flush=True)
best = min(times)
print(f"[time] BEST: {best*1e3:.1f} ms/60tok = {best/NTOK*1e3:.2f} ms/tok = {NTOK/best:.2f} tok/s", flush=True)
print("[done]", flush=True)
