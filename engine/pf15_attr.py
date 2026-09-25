# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P15 Stage 0: the M=64 trunk adjudication (in-plan, the pf12_attr law).

Arms (FULL env + PF_M64=1; the P12 diet envs; readout-order law):
  1. BIT-IDENTITY: M64 prefill(2048) vs M32 prefill(2048) — end states
     (logits, tok/pos slots, rec/conv x5 GDN blocks, kv/sc tails x3 attn)
     + M64 determinism x2 (fresh() is a full world reset).
  2. ATTRIBUTION: isolated per-class PfGraph on the M64 plan at the LAST
     chunk (rewind pos 1984), full-graph refs at PG_SPLIT 2/3/4, M32 full
     graph ref (split 2) — the same-boot control.
  3. WALL: chunk-time medians M64 vs M32 (2048-token prefills).
Answers: (a) GEMM per 64-pass (P12-rate vs P7E7-SC), (b) attention per-64
(4x t32 — grid-doubling illegal by construction), (c) norms sublinearity.
"""
import os, sys, time
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from mtp import MTPEngine, CBLK, SLICE
import pf_prefill
from pf_prefill import PfGraph

E = MTPEngine(theta=1e7)
P, W, d = E.P, E.W, E.P.d
rng = np.random.default_rng(0)
IDS = [int(t) for t in rng.integers(1000, 60000, 2048)]

def fresh_world():
  E.reset_fresh(IDS[0])
  E.stload_trunk()
  for i in E.gdn_idx:
    E._mfill(f"conv{i}_1", 0, CBLK)
  dev.synchronize()

# create the draft buffers once (mirrors the test_w100k boot; DFILL reuses)
fresh_world()
_seen, sl = set(), []
for t in IDS:
  if t not in _seen:
    _seen.add(t); sl.append(t)
_base = sl[:]
while len(sl) < SLICE:
  sl += _base
E.init_draft(sl[:SLICE])
E.fill_draft(IDS, start_pos=0, seed_hd=None)
dev.synchronize()

GDN5 = [E.gdn_idx[0], E.gdn_idx[len(E.gdn_idx)//3], E.gdn_idx[2*len(E.gdn_idx)//3], E.gdn_idx[-2], E.gdn_idx[-1]]
ATTN3 = list(E.qtypes)[:3]

def snap():
  S = {"logits": P.down("logits", (248320,), np.float16).copy(),
       "tok_slot": int(P.down_at("tok_slot", 0, 1)[0]),
       "pos_slot": int(P.down_at("pos_slot", 0, 1)[0])}
  for i in GDN5:
    S[f"rec{i}"] = P.down(f"rec{i}", (48*128*128,), np.float32).copy()
    S[f"conv{i}"] = P.down(f"conv{i}_0", (3*10240,), np.float32).copy()
  for i in ATTN3:
    S[f"kv{i}"] = P.down_at(f"kv{i}", (2048-96)*2048, 96*2048, np.int8).copy()
    S[f"sc{i}"] = P.down_at(f"sc{i}", (2048-96)*64, 96*64, np.int8).copy()
  P._keep.clear()
  return S

CT = []
def run_prefill(mode):
  fresh_world()
  pf_prefill.m64_set(mode == "m64")
  CT.clear()
  t0 = time.perf_counter()
  dt = pf_prefill.prefill_batch(E, None, IDS, chunk_times=CT)
  dev.synchronize()
  return dt, snap(), list(CT)

print("== ARM 1: M64 bit-identity (x2) vs M32 ==", flush=True)
dt64a, S64a, CT64a = run_prefill("m64")
print(f"[p15] M64 prefill A: {dt64a:.2f}s chunks={len(CT64a)} med={sorted(t for _,t in CT64a)[len(CT64a)//2]:.1f}ms", flush=True)
dt64b, S64b, CT64b = run_prefill("m64")
bad_det = [k for k in S64a if not np.array_equal(np.asarray(S64a[k]), np.asarray(S64b[k]))]
print(f"[p15] M64 determinism x2: {'BIT-IDENTICAL' if not bad_det else 'MISMATCH ' + str(bad_det)}", flush=True)
dt32, S32, CT32 = run_prefill("m32")
print(f"[p15] M32 prefill: {dt32:.2f}s chunks={len(CT32)} med={sorted(t for _,t in CT32)[len(CT32)//2]:.1f}ms", flush=True)
bad = [k for k in S32 if not np.array_equal(np.asarray(S32[k]), np.asarray(S64a[k]))]
print(f"[GATE P15-BIT] M64 vs M32 end states: {'BIT-IDENTICAL (all keys)' if not bad else 'MISMATCH ' + str(bad)}", flush=True)
lg64 = S64a["logits"].astype(np.float64); lg32 = S32["logits"].astype(np.float64)
F = np.linalg.norm(lg64 - lg32) / max(np.linalg.norm(lg32), 1e-9)
print(f"[p15] M64-vs-M32 logits F-relerr: {F:.3e} (0.0 == bit-identical)", flush=True)
c64 = sorted(t for _, t in CT64a); c32 = sorted(t for _, t in CT32)
print(f"[p15] WALL: M64 med {c64[len(c64)//2]:.1f} (min {c64[0]:.1f}) vs M32 med {c32[len(c32)//2]:.1f} (min {c32[0]:.1f}) "
      f"-> tok/s M64 {2048/dt64a:.1f} vs M32 {2048/dt32:.1f}", flush=True)

print("== ARM 2: M64 per-class attribution (last chunk rewind) ==", flush=True)
pf_prefill.m64_set(True)
P.win_up("ids64", 0, np.array(IDS[1984:2048], dtype=np.int32))
P.win_up("pos_arr64", 0, np.array([1984], dtype=np.int32))
P.win_up("pos_w64", 0, np.array([1984, 2000, 2016, 2032], dtype=np.int32))
dev.synchronize()

plan = list(E._pf_plan64)
dseq = pf_prefill._pf_dfill_seq64(E) if pf_prefill.DFILL else []
full = plan + dseq
PLAN_CLS = [("pfk_emb16", "emb"), ("pfk_n16", "nrm_n"), ("pfk_ab16", "nrm_ab"), ("pfk_hh16", "nrm_hh"),
            ("pfk_pre", "pre"), ("pfa32c", "attn"), ("pfc16", "attn_comb"),
            ("pfs64", "scan"), ("pfs32", "scan"), ("pfs16", "scan"),
            ("attnqkv", "gemm_qkv"), ("gdnqg", "gemm_qg"), ("pfg_ffn", "gemm_ffn"),
            ("iq3d", "gemm_fd"), ("iq3s", "gemm_oa"), ("iq3o", "gemm_og"), ("q8o", "gemm_og"),
            ("m64", "gemm64"), ("_m32", "gemm32")]
DF_CLS = [("pfk_rec16", "df_rec"), ("pfk_emb16", "df_emb"), ("pfd_dnorm16", "df_dn"),
          ("pfg_ehd", "df_ehd"), ("pfk_n16", "df_n"), ("dqkv", "df_dqkv"), ("pfk_pre16", "df_pre")]

def classify(p, table):
  nm = getattr(p, "name", "?")
  for tag, c in table:
    if tag in nm:
      return c
  return "misc_" + nm[:12]

groups = {}
for t in plan:
  groups.setdefault(classify(t[0], PLAN_CLS), []).append(t)
for t in dseq:
  groups.setdefault(classify(t[0], DF_CLS), []).append(t)

def time_graph(gs, n=5):
  best = 1e9
  for _ in range(n):
    prev = dev.timeline_value - 1
    t1 = time.perf_counter()
    for g in gs[:-1]:
      v = dev.next_timeline(); g.submit(prev, v); prev = v
    v = dev.next_timeline(); gs[-1].submit(prev, v)
    dev.timeline_signal.wait(v)
    best = min(best, time.perf_counter() - t1)
  return best * 1e3

for split in (2, 3, 4):
  n = split
  k = (len(full) + n - 1) // n
  subs = [full[i:i + k] for i in range(0, len(full), k)]
  gs = [PfGraph(sub, f"af{j}") for j, sub in enumerate(subs)]
  print(f"[p15] M64 FULL graph split={split}: {time_graph(gs):.2f}ms ({len(subs)} queues x ~{k})", flush=True)

rows = []
for cname in sorted(groups):
  seq = groups[cname]
  g = PfGraph(seq, f"a_{cname}")
  ms = time_graph([g])
  rows.append((cname, len(seq), ms))
  print(f"[p15] {cname:12s} n={len(seq):4d} {ms:7.2f}ms  ({ms/len(seq)*1000:6.0f}us/launch)", flush=True)
tot = sum(r[2] for r in rows)
print(f"[p15] SUM(isolated)={tot:.2f}ms per 64-token chunk ({len(full)} launches)", flush=True)
print("[p15] DONE", flush=True)
