# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""W2H v2: graph-exec dump FIRST (qw16_3 + pm3/ps3/pA3), then direct-launch
of the captured pre3qh + spk_a3 tuples (wait=True) on the same buffers."""
import os, sys, json
os.environ["SKV"] = "1"; os.environ["SKV_CTXK"] = "100352"
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from mtp import MTPEngine, RBLK, CBLK, SLICE, CTXK
from engine0 import dev

SNAP = os.getenv("SNAPDIR", "~/snap100k")
LS = (256, 1, 1)
meta = json.load(open(f"{SNAP}/meta.json"))
P0 = int(meta["P"]); CUR0 = int(meta["cur0"])
ids = np.load(f"{SNAP}/ids.npy").tolist()
print(f"[dbg] P={P0} CTXK={CTXK} cur0={CUR0}", flush=True)

def win_up(P, name, off, arr):
  a = np.ascontiguousarray(arr)
  dev.allocator._copyin(P.d[name].offset(offset=off, size=a.nbytes), memoryview(a.data).cast("B"))
  P._keep.append(a)

def load100k(E):
  P = E.P
  for j, i in enumerate(E.attn_idx):
    a = np.load(f"{SNAP}/kv_{i}.npy", mmap_mode="r")
    gg = np.asarray(a, dtype=np.float32).reshape(2, 4, CTXK, 8, 32)
    am = np.abs(gg).max(axis=-1)
    scq = (np.maximum(am, 1e-8) * (1.0 / 127.0)).astype(np.float16)
    qq = (np.clip(np.rint(gg / scq.astype(np.float32)[..., None]), -127, 127) + 128).astype(np.uint8)
    P.up(f"kv{i}", qq.reshape(-1)); P.up(f"sc{i}", scq.reshape(-1))
    del gg, am, scq, qq
    if j % 4 == 0: E._flush()
  for j, i in enumerate(E.gdn_idx):
    P.up(f"conv{i}_0", np.load(f"{SNAP}/conv_{i}.npy", mmap_mode="r"))
    P.up(f"rec{i}", np.load(f"{SNAP}/rc_{i}.npy", mmap_mode="r"))
    if j % 8 == 0: E._flush()
  P.up("tok_slot", np.array([CUR0], dtype=np.int32))
  P.up("pos_slot", np.array([P0], dtype=np.int32))
  E._flush()

def reset_spec(E):
  P = E.P
  rec4 = np.full((48, 5, RBLK), 7.7e31, dtype=np.float32)
  conv4 = np.full((48, 5, CBLK), 7.7e31, dtype=np.float32)
  for j, i in enumerate(E.gdn_idx):
    rec4[j, 4] = np.load(f"{SNAP}/rc_{i}.npy", mmap_mode="r")
    conv4[j, 4] = np.load(f"{SNAP}/conv_{i}.npy", mmap_mode="r")
  P.up("rec4", rec4.reshape(-1)); P.up("conv4", conv4.reshape(-1))
  P.up("cur_slot", np.array([CUR0], dtype=np.int32))
  P.up("pos_slot", np.array([P0], dtype=np.int32))
  P.up("h_seed", np.zeros(5120, dtype=np.float32))
  P.up("m_hist", np.zeros(1024, dtype=np.int32))
  P.up("cyc_slot", np.zeros(1, dtype=np.int32))
  P.up("m_slot", np.zeros(1, dtype=np.int32))
  P.up("dring0", np.array([int(ids[0])], dtype=np.int32))
  P.up("dring1", np.array([int(ids[1])], dtype=np.int32))
  P.up("tok_hist", np.full(CTXK + 256, -1, dtype=np.int32))
  E._flush(); E.build_graphs()
  for nm, arr in (("cur_slot", np.array([CUR0], dtype=np.int32)), ("pos_slot", np.array([P0], dtype=np.int32)),
                  ("m_slot", np.zeros(1, dtype=np.int32)), ("cyc_slot", np.zeros(1, dtype=np.int32)),
                  ("dring0", np.array([int(ids[0])], dtype=np.int32)), ("dring1", np.array([int(ids[1])], dtype=np.int32)),
                  ("m_hist", np.zeros(1024, dtype=np.int32)), ("h_seed", np.zeros(5120, dtype=np.float32)),
                  ("tok_hist", np.full(CTXK + 256, -1, dtype=np.int32))):
    win_up(P, nm, 0, arr)
  dev.synchronize()

def poison_partials(E):
  E.P.poison("pm3", 4 * 256 * 18 * 4, np.float32, 7.7e31)
  E.P.poison("ps3", 4 * 256 * 18 * 4, np.float32, 7.7e31)
  E.P.poison("pA3", 4 * 256 * 18 * 256 * 4, np.float32, 7.7e31)
  E._flush(); dev.synchronize()

def dump(tag):
  qw = E.P.down("qw16_3", (3 * 24 * 256,), np.float16).astype(np.float32)
  out = {}
  for nm, shape in (("qw16_3", None), ("pm3", (4 * 256 * 18,)), ("ps3", (4 * 256 * 18,)), ("pA3", (4 * 256 * 18 * 256,))):
    v = qw if nm == "qw16_3" else E.P.down(nm, shape, np.float32)
    npois = int((v == 7.7e31).sum()); nzero = int((v == 0).sum())
    fin = v[np.isfinite(v) & (v != 7.7e31)]
    rng = f"min {fin.min():.5g} max {fin.max():.5g} med {np.median(fin):.5g}" if fin.size else "EMPTY"
    print(f"[{tag}] {nm}: poison {npois}/{v.size} zero {nzero}/{v.size} finite {fin.size} | {rng}", flush=True)
    out[nm] = v.copy()
  return out

E = MTPEngine(theta=1e7)
load100k(E)
ref = np.load(f"{SNAP}/engine_t1_ref_kv8_qh.npy").tolist()
seen, sl = set(), []
for t in ref + ids:
  if t not in seen: seen.add(t); sl.append(t)
base = sl[:]
while len(sl) < SLICE: sl += base
E.init_draft(sl[:SLICE])

# ---------- (1) GRAPH-EXEC (graphs acquired AFTER reset_spec re-capture) ----------
reset_spec(E)
probe_g, flush_g = E.graphs[1], E.graphs[3]
seq = probe_g.seq
print(f"[dbg] probe seq {len(seq)}k", flush=True)
poison_partials(E)
E.run_cycles(1)   # faithful gate-path exec: draft->probe->accept->flush
print("[graph] run_cycles(1) done", flush=True)
g1 = dump("graph")

# ---------- (2) direct launch of captured pre3qh + spk_a3 tuples ----------
tup_pre = tup_a3 = None
for n, (p, a, g) in enumerate(seq):
  nm = str(getattr(p, "name", "?"))
  if nm == "spk_g4nwhm3_100k": tup_a3 = (n, p, a, g); break
if tup_a3 is not None:
  n0 = tup_a3[0]
  for n in range(n0 - 1, max(0, n0 - 40), -1):
    nm = str(getattr(seq[n][0], "name", "?"))
    if "pre3qh" in nm: tup_pre = (n,) + seq[n]; break
  poison_partials(E)
  if tup_pre is not None:
    n, p, a, g = tup_pre
    print(f"[direct] #{n} {p.name} grid={g}", flush=True)
    p(*a, global_size=(g, 1, 1), local_size=LS, wait=True)
  n, p, a, g = tup_a3
  print(f"[direct] #{n} {p.name} grid={g}", flush=True)
  p(*a, global_size=(g, 1, 1), local_size=LS, wait=True)
  d1 = dump("direct")
  for nm in ("pm3", "ps3", "pA3"):
    x, y = g1[nm], d1[nm]
    print(f"[cmp graph-vs-direct] {nm}: equal {int((x == y).sum())}/{x.size} maxdiff {np.abs(x - y).max():.5g}", flush=True)
print("[dbg done]", flush=True)
