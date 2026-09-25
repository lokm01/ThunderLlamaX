# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R6 BATCH-MODE: multi-stream batched decode — B concurrent streams share ONE
M=BT probe trunk (BT = sum of per-stream T rows) so the weight read is shared.
Tier-1-per-stream by construction: row-independent M families (norms/GEMVs/FFN/
head — per-row fp order identical across M, the R4 rowchk law) + per-stream
STATEFUL kernels (spk attention, k2s GDN scan, accept, draft, lookup) run as
per-stream launches over sliced row-major scratch + per-stream state banks.

RUNG 1 (this file): B=2 @100k-class, ZERO new CUDA kernels:
  s0 = canonical 100k snapshot stream, T=3 (K2 rows)     -> Tier-1 vs BANKED refs
  s1 = fresh 8k prompt stream,       T=5 (K=4-deep rows) -> Tier-1 vs in-session T=1
  BT=8 -> the M8 family (ffn8v8r7/down8nw32v8r7 under PF_DR7; boot LOOKUP_K=7 for
  RM=8 scratch + the M8/spk3/spk5 cubin loads).
Env: the canonical daemon env with LOOKUP_K=7 (not 10) + R6_B=2 + R6_SPEC.
Laws respected: fixed-handle (all bank allocs BEFORE graph builds; win_up-only
after), READOUT-ORDER, ROWS>64 (BT<=8), ONE KERNEL PER CUBIN, 16B-aligned
slices (all row strides are 16B multiples), lone-graph flusher, big-copyin
(banks device-zeroed via mfill — no host DMA), ~950-cycle budget (60-cyc gates).
"""
import os, sys, time, json, collections
os.environ["SKV"] = "1"
os.environ["SKV_CTXK"] = "100352"
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
import mtp
from mtp import MTPEngine, RBLK, CBLK, SLICE, CTXK, RM
from trunk import CTX as TRUNK_CTX, VOCAB
from engine0 import dev
from gcycle import GCycleEngine, ParityGraph
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_nv import NVProgram

SNAP = os.getenv("SNAPDIR", "~/snap100k")
NTOK = int(os.getenv("NTOK", "60"))
R6_B = int(os.getenv("R6_B", "2"))
R6_SPEC = json.loads(os.getenv("R6_SPEC", '[{"T":3,"mode":"k2","s":0},{"T":5,"mode":"k4","s":1}]'))
P8K_IDS = os.getenv("R6_P8K", "~/r6_p8k_ids.npy")
QH = os.getenv("QH", "1") == "1"
KV8 = os.getenv("KV8", "1") == "1"
SKV_S = int(os.getenv("SKV_S", "256"))
DR7 = bool(int(os.getenv("PF_DR7", "0")))
GEMVV = os.getenv("GEMVV", "1") == "1"
assert GEMVV, "R6 runs the GEMVV (W2D) decode families"

# ---- M-family tables (DR7-legal BT values; launch counts verbatim) ----
MSET = {
  3:  dict(n="k0n3", n_g=3, ab="k0ab3", ab_g=39, q5="q5g8_3", q5_g=2048,
           aq3="aq3k8v_3", aq6="aq6k8_3", aq_g=1792, ao="ao8nw32_3", ao_g=160,
           k3ao="k3aonw32_3", op3="op38nw32_3", og=160,
           hh="hh3", hh_g=3, ffn=("ffn8v3r7" if DR7 else "ffn8v_3"),
           down=("down8nw32v3r7" if DR7 else "down8nw32_3"), ffw_g=2176, dwn_g=160,
           head="head8v_3"),
  8:  dict(n="k0n8", n_g=8, ab="k0ab8", ab_g=104, q5="q5g8v8", q5_g=2048,
           aq3="aq3k8v8", aq6="aq6k8v8", aq_g=1792, ao="ao8nw32_8", ao_g=160,
           k3ao="k3aonw32_8", op3="op38nw32_8", og=160,
           hh="hh8", hh_g=8, ffn=("ffn8v8r7" if DR7 else "ffn8v8"),
           down=("down8nw32v8r7" if DR7 else "down8nw32_8"), ffw_g=2176, dwn_g=160,
           head="head8v8"),
  # M9 loads at LOOKUP_K=8 boot (RM=9); M10 at LOOKUP_K=9 (RM=10) — counts from
  # _probe9_seq/_probe10_seq (13*T for k0ab, T for k0n/hh).
  9:  dict(n="k0n9", n_g=9, ab="k0ab9", ab_g=117, q5="q5g8v9", q5_g=2048,
           aq3="aq3k8v9", aq6="aq6k8v9", aq_g=1792, ao="ao8nw32_9", ao_g=160,
           k3ao="k3aonw32_9", op3="op38nw32_9", og=160,
           hh="hh9", hh_g=9, ffn="ffn8v9r7", down="down8nw32v9r7", ffw_g=2176, dwn_g=160,
           head="head8v9"),
  10: dict(n="k0n10", n_g=10, ab="k0ab10", ab_g=130, q5="q5g8v10", q5_g=2048,
           aq3="aq3k8v10", aq6="aq6k8v10", aq_g=1792, ao="ao8nw32_10", ao_g=160,
           k3ao="k3aonw32_10", op3="op38nw32_10", og=160,
           hh="hh10", hh_g=10, ffn="ffn8v10r7", down="down8nw32v10r7", ffw_g=2176, dwn_g=160,
           head="head8v10"),
}
R6_SPECS = json.loads(os.getenv("R6_SPECS", "null")) or [R6_SPEC]
BTMAX = max(sum(x["T"] for x in _sp) for _sp in R6_SPECS)
for _sp in R6_SPECS:
  _bt = sum(x["T"] for x in _sp)
  assert _bt in MSET, f"BT={_bt} has no M-family in table (have {sorted(MSET)})"
  assert _bt <= BTMAX, "internal"
BT = sum(s["T"] for s in R6_SPEC)

TSET = {
  3: dict(embed="h_embed3", embed_g=3, k2s="k2s3", acc="acceptk", accsel="acceptsel", lookup=None),
  # embed_g verbatim from the proven seqs: h_embed3 launches 3 (K2 graph),
  # h_embed5 launches 1 (_probe5_seq; per-M row-tiling differs per cubin).
  5: dict(embed="h_embed5", embed_g=1, k2s="k2s5", acc="accept5k", accsel="acceptsel5k", lookup="lookup5_nw32"),
}
EXTRA_CUBINS = sorted({v for s in R6_SPEC for k, v in TSET[s["T"]].items() if k != "embed_g" and v is not None}
                      - set(mtp.M3_CUBINS) - set(mtp.M8_CUBINS))

def log(*a): print("[r6]", *a, flush=True)

# ============================ boot =====================================
meta = json.load(open(f"{SNAP}/meta.json"))
P0 = int(meta["P"]); CUR0 = int(meta["cur0"])
ids = np.load(f"{SNAP}/ids.npy").tolist()
ids8k = [int(t) for t in np.load(P8K_IDS)]
log(f"P={P0} cur0={CUR0} prompt {len(ids)}; SPEC={R6_SPEC} BT={BT} RM={RM}")

t0 = time.perf_counter()
E = MTPEngine(theta=1e7)
log(f"engine loaded {time.perf_counter()-t0:.1f}s")
for n in EXTRA_CUBINS:
  if n in E.pr: continue
  lib = open(f"{mtp.BASE}/{n}.cubin", "rb").read()
  E.pr[n] = NVProgram(dev, TinyELF(lib=lib, name=n, target=dev.renderer.target, signature=tuple()))
  log(f"extra cubin {n}")

# ---- BT>RM support WITHOUT the LOOKUP_K=9 boot (whose +755MB deep-set scratch
# tips the VRAM floor — the boot-8 fault-as-OOM): resize the RM-sized probe
# scratch to R6_BTMAX rows + load the needed M-family cubins directly. The
# scratch replacement is safe pre-graph-build (fixed-handle law: nothing has
# baked the old handles yet; the trunk T=1 path uses x0/x1 not xA).
BTMAX = max(sum(x["T"] for x in _sp) for _sp in R6_SPECS)
if BTMAX > RM:
  assert BTMAX in MSET, f"BTMAX={BTMAX} not in MSET"
  P = E.P
  _SC = [("xA", 5120*4, np.float32), ("xB", 5120*4, np.float32),
         ("xh3", 5120*2, np.float16), ("hh3b", 5120*4, np.float32), ("hhx3", 5120*2, np.float16),
         ("qkv3", 10240*2, np.float16), ("gate3", 6144*2, np.float16),
         ("araw3", 48*4, np.float32), ("braw3", 48*4, np.float32), ("z3", 6144*2, np.float16),
         ("attn_out3", 5120*2, np.float16), ("gact3", 17408*2, np.float16),
         ("qrow3", 12288*2, np.float16), ("krow3", 1024*2, np.float16), ("vrow3", 1024*2, np.float16),
         ("ao_row3", 6144*2, np.float16), ("logits3", VOCAB*2, np.float16), ("amds", 4, np.int32),
         ("qw3", 24*256*4, np.float32), ("qw16_3", 24*256*2, np.float16),
         ("pm3", 4*SKV_S*6*4, np.float32), ("ps3", 4*SKV_S*6*4, np.float32),
         ("pA3", 4*SKV_S*6*256*4, np.float32)]
  for nm, rb, dt in _SC:
    P.d[nm] = P.alloc(f"{nm}_r6bt{BTMAX}", rb*BTMAX)
  dev.synchronize()
  M = MSET[BTMAX]
  for _n in [M[k] for k in ("n","ab","q5","aq3","aq6","ao","k3ao","op3","hh","ffn","down","head")]:
    if _n not in E.pr:
      lib = open(f"{mtp.BASE}/{_n}.cubin", "rb").read()
      E.pr[_n] = NVProgram(dev, TinyELF(lib=lib, name=_n, target=dev.renderer.target, signature=tuple()))
  dev.synchronize()
  log(f"BT{BTMAX} world: probe scratch resized to {BTMAX} rows + M{BTMAX} cubins loaded (boot RM={RM})")

# ---- stream banks s=1..B-1 (device-zeroed; no host DMA — DART law) ----
def alloc_bank(s):
  P = E.P
  sz = {}
  def A(name, nbytes): sz[name] = nbytes; P.d[f"{name}_s{s}"] = P.alloc(f"{name}_s{s}", nbytes)
  A("rec4", 48*5*RBLK*4); A("conv4", 48*5*CBLK*4); A("conv5x", 48*CBLK*4)
  A("cur_slot", 4); A("pos_slot", 4); A("tok_slot", 4)
  A("tok_hist", (CTXK+256)*4); A("m_slot", 4); A("cyc_slot", 4)
  A("m_hist", (1 << 20)*4); A("l_hist", (1 << 20)*4)
  for k in range(10): A(f"dring{k}", 4)
  A("emit", 26*4); A("h_seed", 5120*4); A("dhd_seed", 5120*4)
  A("hd_d0", 5120*4); A("hd_d1", 5120*4); A("dpos1", 4); A("dpos2", 4)
  A("kv_d", 2*4*CTXK*256); A("sc_d", 2*4*CTXK*8*2)
  for i in E.attn_idx:
    A(f"kv{i}", 2*4*CTXK*256); A(f"sc{i}", 2*4*CTXK*8*2)
  dev.synchronize()
  zero = ["rec4", "conv4", "conv5x", "cur_slot", "pos_slot", "tok_slot", "m_slot", "cyc_slot",
          "m_hist", "l_hist", "emit", "h_seed", "dhd_seed", "hd_d0", "hd_d1", "dpos1", "dpos2",
          "kv_d", "sc_d", *[f"dring{k}" for k in range(2, 10)],
          *[f"kv{i}" for i in E.attn_idx], *[f"sc{i}" for i in E.attn_idx]]
  for nm in zero: E._mfill(f"{nm}_s{s}", 0, sz[nm] // 4)
  E._mfill(f"tok_hist_s{s}", -1, CTXK + 256)
  P.win_up(f"dring0_s{s}", 0, np.full(1, -1, dtype=np.int32))
  P.win_up(f"dring1_s{s}", 0, np.full(1, -1, dtype=np.int32))
  dev.synchronize()
  log(f"stream-{s} banks allocated + zeroed")

for s in range(1, R6_B): alloc_bank(s)

SWAP_KEYS = (["rec4", "conv4", "conv5x", "cur_slot", "pos_slot", "tok_slot", "tok_hist",
              "m_slot", "cyc_slot", "m_hist", "l_hist", "emit", "h_seed", "dhd_seed",
              "hd_d0", "hd_d1", "dpos1", "dpos2", "kv_d", "sc_d"]
             + [f"dring{k}" for k in range(10)]
             + [f"kv{i}" for i in E.attn_idx] + [f"sc{i}" for i in E.attn_idx])
class Swap:
  def __init__(self, s): self.s = s
  def __enter__(self):
    self.saved = {}
    if self.s:
      for k in SWAP_KEYS:
        self.saved[k] = E.P.d[k]; E.P.d[k] = E.P.d[f"{k}_s{self.s}"]
    return self
  def __exit__(self, *a): E.P.d.update(self.saved)

# ---- stream 0: canonical snapshot load, OR a second FRESH 8k stream ----
S0_FRESH = os.getenv("R6_S0_FRESH", "0") == "1"
if not S0_FRESH:
  assert TRUNK_CTX == CTXK
  E.load_snapshot_kv(SNAP)
  for j, i in enumerate(E.gdn_idx):
    E.P.win_up(f"conv{i}_0", 0, np.load(f"{SNAP}/conv_{i}.npy", mmap_mode="r"))
    E.P.win_up(f"rec{i}", 0, np.load(f"{SNAP}/rc_{i}.npy", mmap_mode="r"))
    if j % 8 == 0: E._flush()
  E.P.win_up("tok_slot", 0, np.array([CUR0], dtype=np.int32))
  E.P.win_up("pos_slot", 0, np.array([P0], dtype=np.int32))
  E._flush()
  ref0 = np.load(f"{SNAP}/engine_t1_ref.npy").tolist()
  assert len(ref0) >= NTOK
  ids0 = ids
  log(f"s0 banked ref[:8] {ref0[:8]}")
else:
  _off = int(os.getenv("R6_S0_OFF", "40000"))
  ids0 = [int(t) for t in ids[_off:_off + len(ids8k)]]
  assert len(ids0) == len(ids8k), (len(ids0), len(ids8k))
  ref0 = None
  log(f"s0 FRESH ids0 (100k-slice @{_off}) len {len(ids0)} first {ids0[:6]}")

# ---- draft slice covering BOTH streams' token distributions ----
seen, sl = set(), []
for t in ((ref0 or []) + ids0 + ids8k + json.load(open("~/tinygrad-metal/spec_base_8k.json"))):
  if t not in seen: seen.add(t); sl.append(t)
for t, _ in collections.Counter(ids0 + ids8k).most_common():
  if t not in seen: seen.add(t); sl.append(t)
base = sl[:]
while len(sl) < SLICE: sl += base
sl = sl[:SLICE]
log(f"slice {len(set(sl))} distinct; ref0 covered "
    f"{sum(1 for t in (ref0 or []) if t in set(sl))}/{len(ref0) if ref0 else 'fresh'};"
    f" ids8k[:2000] covered {sum(1 for t in ids8k[:2000] if t in set(sl))}/2000")
if ref0 is not None:
  assert all(t in set(sl) for t in ref0), "draft slice lost ref0 coverage"
E.init_draft(sl)

if not S0_FRESH:
  t0 = time.perf_counter()
  E.fill_draft(ids)
  log(f"s0 fill_draft(100k) {time.perf_counter()-t0:.0f}s")
else:
  log("s0 FRESH: skipping the 100k fill (fresh prefill below)")

# ---- per-stream anchors ----
def anchor_save(s):
  P = E.P; pre = "" if s == 0 else f"_s{s}"
  A = {"rec": [], "conv": []}
  for j in range(48):
    A["rec"].append(P.down_at(f"rec4{pre}", (j*5+4)*RBLK*4, RBLK, np.float32))
    A["conv"].append(P.down_at(f"conv4{pre}", (j*5+4)*CBLK*4, CBLK, np.float32))
  A["cur"] = int(P.down_at(f"cur_slot{pre}", 0, 1)[0])
  A["pos"] = int(P.down_at(f"pos_slot{pre}", 0, 1)[0])
  A["hist"] = np.asarray(ids0 if s == 0 else ids8k, dtype=np.int32)
  return A
def anchor_load(s, A):
  P = E.P; pre = "" if s == 0 else f"_s{s}"
  pv = np.float32(7.7e31).view(np.int32).item()
  E._mfill(f"rec4{pre}", pv, 48*5*RBLK)
  E._mfill(f"conv4{pre}", pv, 48*5*CBLK)
  E._mfill(f"conv5x{pre}", pv, 48*CBLK)
  dev.synchronize()
  for j in range(48):
    P.win_up(f"rec4{pre}", (j*5+4)*RBLK*4, A["rec"][j])
    P.win_up(f"conv4{pre}", (j*5+4)*CBLK*4, A["conv"][j])
    if j % 16 == 0: dev.synchronize()
  P.win_up(f"cur_slot{pre}", 0, np.array([A["cur"]], dtype=np.int32))
  P.win_up(f"pos_slot{pre}", 0, np.array([A["pos"]], dtype=np.int32))
  P.win_up(f"tok_hist{pre}", 0, A["hist"])
  for nm in ("m_hist", "l_hist"): E._mfill(f"{nm}{pre}", 0, 1 << 20)
  P.win_up(f"cyc_slot{pre}", 0, np.zeros(1, dtype=np.int32))
  P.win_up(f"m_slot{pre}", 0, np.zeros(1, dtype=np.int32))
  P.win_up(f"h_seed{pre}", 0, np.zeros(5120, dtype=np.float32))
  P.win_up(f"dhd_seed{pre}", 0, np.zeros(5120, dtype=np.float32))
  for k in range(10):
    P.win_up(f"dring{k}{pre}", 0, np.array([-1 if k < 2 else 0], dtype=np.int32))
  dev.synchronize()

# ---- stream 1: FRESH 8k prefill (the serve FRESH sequence, swap-scoped) ----
with Swap(1):
  E.reset_fresh(ids8k[0])
  E.stload_trunk()
  for _i in E.gdn_idx: E._mfill(f"conv{_i}_1", 0, CBLK)
  dev.synchronize()
  E.fill_draft(ids8k, start_pos=0)
  if hasattr(E, "_seq"): del E._seq
  E._build_seqs()
  G1 = GCycleEngine(E); G1.build(); dev.synchronize()
  t0 = time.perf_counter()
  if os.getenv("R6_PF_T1", "1") == "1":
    # boot3 law: the PF chunk path (ensure64/128 scratch + G3M/W4A8 planes) faults
    # with the batch banks resident (moving-floor VRAM exhaustion, fault-as-OOM
    # class). The T=1 trunk prefill needs ZERO extra machinery (G1 already built).
    E.prefill_t1(G1, ids8k, log_every=2000)
    log(f"s1 prefill_t1 {len(ids8k)} toks {time.perf_counter()-t0:.1f}s")
  else:
    import pf_prefill
    pf_prefill.prefill_batch(E, G1, ids8k)
    log(f"s1 prefill_batch {len(ids8k)} toks {time.perf_counter()-t0:.1f}s")
  E.stseed_spec(len(ids8k) & 1)
  newcur1 = int(E.P.down_at("tok_slot", 0, 1)[0])
  P1 = int(E.P.down_at("pos_slot", 0, 1)[0])
  E.P.win_up("cur_slot", 0, np.array([newcur1], dtype=np.int32))
  E.P.win_up("h_seed", 0, np.zeros(5120, dtype=np.float32))
  E.P.win_up("dring0", 0, np.full(1, -1, dtype=np.int32))
  E.P.win_up("dring1", 0, np.full(1, -1, dtype=np.int32))
  E.P.win_up("tok_hist", 0, np.array(ids8k, dtype=np.int32))
  for nm in ("m_hist", "l_hist"): E._mfill(nm, 0, 1 << 20)
  E.P.win_up("cyc_slot", 0, np.zeros(1, dtype=np.int32))
  E.P.win_up("m_slot", 0, np.zeros(1, dtype=np.int32))
  dev.synchronize()
if hasattr(E, "_seq"): del E._seq   # trunk seq is s1-baked; force s0 rebuild if ever used
log(f"s1 prefill done: pos={P1} cur={newcur1}")
ANC1 = anchor_save(1)   # BEFORE the T=1 ref (the ref advances s1 state)

# ---- s1 T=1 greedy reference ----
with Swap(1):
  E.P.win_up("tok_slot", 0, np.array([newcur1], dtype=np.int32))
  dev.synchronize()
  t0 = time.perf_counter()
  G1.run_tokens(NTOK, wait_each=True)
  log(f"s1 T=1 ref {NTOK} toks {time.perf_counter()-t0:.1f}s")
  ref1 = E.P.down_at("tok_hist", P1*4, NTOK, np.int32).tolist()
assert all(t >= 0 for t in ref1), "s1 T=1 ref incomplete"
log(f"s1 ref[:10] {ref1[:10]}")

if not S0_FRESH:
  E.reset_snapshot(SNAP, CUR0, P0)
  E.P.win_up("tok_hist", 0, np.array(ids, dtype=np.int32))
  dev.synchronize()
else:
  # s0 FRESH prefill on the canonical buffers (no swap) — the same M1 sequence
  E.reset_fresh(ids0[0])
  E.stload_trunk()
  for _i in E.gdn_idx: E._mfill(f"conv{_i}_1", 0, CBLK)
  dev.synchronize()
  E.fill_draft(ids0, start_pos=0)
  E._build_seqs()
  G0 = GCycleEngine(E); G0.build(); dev.synchronize()
  t0 = time.perf_counter()
  E.prefill_t1(G0, ids0, log_every=2000)
  log(f"s0 prefill_t1 {len(ids0)} toks {time.perf_counter()-t0:.1f}s")
  E.stseed_spec(len(ids0) & 1)
  newcur0 = int(E.P.down_at("tok_slot", 0, 1)[0])
  P0 = int(E.P.down_at("pos_slot", 0, 1)[0])
  E.P.win_up("cur_slot", 0, np.array([newcur0], dtype=np.int32))
  E.P.win_up("h_seed", 0, np.zeros(5120, dtype=np.float32))
  E.P.win_up("dring0", 0, np.full(1, -1, dtype=np.int32))
  E.P.win_up("dring1", 0, np.full(1, -1, dtype=np.int32))
  E.P.win_up("tok_hist", 0, np.array(ids0, dtype=np.int32))
  for nm in ("m_hist", "l_hist"): E._mfill(nm, 0, 1 << 20)
  E.P.win_up("cyc_slot", 0, np.zeros(1, dtype=np.int32))
  E.P.win_up("m_slot", 0, np.zeros(1, dtype=np.int32))
  dev.synchronize()
  log(f"s0 prefill done: pos={P0} cur={newcur0}")
ANC0 = anchor_save(0)   # BEFORE any s0 T=1 ref decode
if S0_FRESH:
  E.P.win_up("tok_slot", 0, np.array([newcur0], dtype=np.int32))
  dev.synchronize()
  t0 = time.perf_counter()
  G0.run_tokens(NTOK, wait_each=True)
  log(f"s0 T=1 ref {NTOK} toks {time.perf_counter()-t0:.1f}s")
  ref0 = E.P.down_at("tok_hist", P0*4, NTOK, np.int32).tolist()
  assert all(t >= 0 for t in ref0), "s0 T=1 ref incomplete"
  log(f"s0 ref[:10] {ref0[:10]}")
if hasattr(E, "_seq"): del E._seq   # any trunk seq built above is stale for later use
log("anchors saved")

# ====================== the r6 graph builder ======================
def stream_bufs(s):
  pre = "" if s == 0 else f"_s{s}"
  d = E.P.d
  names = ("cur_slot", "pos_slot", "tok_hist", "m_slot", "cyc_slot", "m_hist", "l_hist",
           "h_seed", "dhd_seed", "hd_d0", "hd_d1", "dpos1", "dpos2", "emit",
           "rec4", "conv4", "conv5x", "kv_d", "sc_d", *[f"dring{k}" for k in range(10)])
  b = {n: d[f"{n}{pre}"] for n in names}
  for i in E.attn_idx:
    b[f"kv{i}"] = d[f"kv{i}{pre}"]; b[f"sc{i}"] = d[f"sc{i}{pre}"]
  return b

def build_r6_graphs(specs):
  """specs: [{"s": stream, "T": rows, "mode": "k2"|"k4"}]; one fixed graph set."""
  d, W, pr = E.P.d, E.W, E.pr
  bt = sum(x["T"] for x in specs)
  M = MSET[bt]
  st = []
  r0 = 0
  for x in specs:
    b = stream_bufs(x["s"])
    b.update(T=x["T"], mode=x["mode"], r0=r0, s=x["s"])
    r0 += x["T"]
    st.append(b)
  # ---- draft ----
  dseq = []
  for b in st:
    dd = dict(E.P.d); dd["kv_d"] = b["kv_d"]; dd["sc_d"] = b["sc_d"]
    dseq += E._draft_entries(b["cur_slot"], b["pos_slot"], b["h_seed"], b["hd_d0"], b["dring0"], dd=dd)
    dseq.append((pr["dposadd"], (b["pos_slot"], b["dpos1"]), 1))
    dseq += E._draft_entries(b["dring0"], b["dpos1"], b["hd_d0"], b["hd_d1"], b["dring1"], dd=dd)
    lu = TSET[b["T"]]["lookup"]
    if lu is not None:
      dseq.append((pr[lu], (b["tok_hist"], b["pos_slot"], b["cur_slot"], b["dring0"], b["dring1"],
                            b["dring2"], b["dring3"], b["l_hist"], b["cyc_slot"]), 1))
  draft_g = ParityGraph(dseq, tag="r6D")
  # ---- probe ----
  seq = []
  for b in st:
    xav = d["xA"].offset(offset=b["r0"]*5120*4, size=b["T"]*5120*4)
    if b["T"] == 3:
      seq.append((pr["h_embed3"], (W[("emb",0)], d["grid512"], b["cur_slot"], b["dring0"], b["dring1"], xav), 3))
    elif b["T"] == 5:
      seq.append((pr["h_embed5"], (W[("emb",0)], d["grid512"], b["cur_slot"], b["dring0"], b["dring1"],
                                   b["dring2"], b["dring3"], xav), 5))
    else: raise AssertionError(f"T={b['T']} unsupported")
  cur = 0
  for i in range(64):
    xin, xout = (d["xA"] if cur == 0 else d["xB"]), (d["xB"] if cur == 0 else d["xA"])
    if i in E.qtypes:
      qkname = M["aq6"] if E.qtypes[i] == 14 else M["aq3"]
      a = [(pr[M["n"]], (xin, W[("nw1",i)], d["xh3"]), M["n_g"]),
           (pr[qkname], (W[("q",i)], W[("k",i)], W[("v",i)], d["gridf"], d["xh3"], d["qrow3"], d["krow3"], d["vrow3"]), M["aq_g"])]
      for b in st:
        sca = (b[f"sc{i}"],) if KV8 else ()
        qsl = d["qrow3"].offset(offset=b["r0"]*12288*2, size=b["T"]*12288*2)
        ksl = d["krow3"].offset(offset=b["r0"]*1024*2, size=b["T"]*1024*2)
        vsl = d["vrow3"].offset(offset=b["r0"]*1024*2, size=b["T"]*1024*2)
        pmb = d["pm3"].offset(offset=b["r0"]*4*SKV_S*6*4, size=b["T"]*4*SKV_S*6*4)
        psb = d["ps3"].offset(offset=b["r0"]*4*SKV_S*6*4, size=b["T"]*4*SKV_S*6*4)
        pab = d["pA3"].offset(offset=b["r0"]*4*SKV_S*6*256*4, size=b["T"]*4*SKV_S*6*256*4)
        aosl = d["ao_row3"].offset(offset=b["r0"]*6144*2, size=b["T"]*6144*2)
        a += [(pr[f"spk_pre{b['T']}"], (qsl, ksl, vsl, W[("qnw",i)], W[("knw",i)], d["freqs"], b[f"kv{i}"], *sca, b["pos_slot"], d["qw3"], *((d["qw16_3"],) if QH else ())), 24),
              (pr[f"spk_a{b['T']}"], (b[f"kv{i}"], *sca, *((d["qw16_3"],) if QH else d["qw3"]), b["pos_slot"], pmb, psb, pab), 4*SKV_S),
              (pr[f"spk_c{b['T']}"], (pmb, psb, pab, qsl, aosl), 24)]
      a += [(pr[M["ao"]], (W[("o",i)], d["grid512"], d["ao_row3"], d["attn_out3"]), M["ao_g"]),
            (pr[M["hh"]], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), M["hh_g"]),
            (pr[M["ffn"]], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), M["ffw_g"]),
            (pr[M["down"]], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), M["dwn_g"])]
    else:
      gi = E.gdn_idx.index(i)
      a = [(pr[M["ab"]], (xin, W[("nw1",i)], W[("alpha",i)], W[("beta",i)], d["xh3"], d["araw3"], d["braw3"]), M["ab_g"]),
           (pr[M["q5"]], (W[("qkv",i)], W[("gate",i)], d["gridf"], d["xh3"], d["qkv3"], d["gate3"]), M["q5_g"])]
      for b in st:
        conv_b = b["conv4"].offset(offset=gi*5*CBLK*4, size=5*CBLK*4)
        rec_b = b["rec4"].offset(offset=gi*5*RBLK*4, size=5*RBLK*4)
        qksl = d["qkv3"].offset(offset=b["r0"]*10240*2, size=b["T"]*10240*2)
        gtsl = d["gate3"].offset(offset=b["r0"]*6144*2, size=b["T"]*6144*2)
        arsl = d["araw3"].offset(offset=b["r0"]*48*4, size=b["T"]*48*4)
        brsl = d["braw3"].offset(offset=b["r0"]*48*4, size=b["T"]*48*4)
        zsl = d["z3"].offset(offset=b["r0"]*6144*2, size=b["T"]*6144*2)
        base_args = (qksl, gtsl, W[("convw",i)], W[("dtb",i)], W[("ssma",i)], arsl, brsl,
                     d["q"], d["k"], d["v"], d["core"], W[("snw",i)], zsl)
        if b["T"] == 3:
          a.append((pr["k2s3"], (conv_b, rec_b, *base_args), 48))
        else:
          c5x = b["conv5x"].offset(offset=gi*CBLK*4, size=CBLK*4)
          a.append((pr["k2s5"], (conv_b, rec_b, c5x, *base_args), 48))
      if E.gdn_oq8[i]:
        a.append((pr[M["k3ao"]], (W[("out",i)], d["z3"], d["attn_out3"]), M["og"]))
      else:
        a.append((pr[M["op3"]], (W[("out",i)], d["gridf"], d["z3"], d["attn_out3"]), M["og"]))
      a += [(pr[M["hh"]], (xin, d["attn_out3"], W[("nw2",i)], d["hh3b"], d["hhx3"]), M["hh_g"]),
            (pr[M["ffn"]], (W[("fg",i)], W[("fu",i)], d["gridf"], d["hhx3"], d["gact3"]), M["ffw_g"]),
            (pr[M["down"]], (W[("fd",i)], d["gridf"], d["gact3"], d["hh3b"], xout), M["dwn_g"])]
    seq += a
    cur ^= 1
  seq.append((pr[M["n"]], (d["xA"], W[("onw",0)], d["xh3"]), M["n_g"]))
  seq.append((pr[M["head"]], (W[("head",0)], d["xh3"], d["logits3"]), VOCAB//8))
  seq.append((pr["amx3"], (d["logits3"], d["amds"]), bt))
  probe_g = ParityGraph(seq, tag="r6P")
  # ---- accept ----
  aseq = []
  for b in st:
    asl = d["amds"].offset(offset=b["r0"]*4, size=b["T"]*4)
    xsl = d["xA"].offset(offset=b["r0"]*5120*4, size=b["T"]*5120*4)
    if b["T"] == 3:
      aseq.append((pr["acceptk"], (asl, b["dring0"], b["dring1"], xsl, b["m_slot"], b["m_hist"],
                                   b["cyc_slot"], b["pos_slot"], b["cur_slot"], b["tok_hist"], b["h_seed"],
                                   b["hd_d0"], b["hd_d1"], b["dhd_seed"], b["emit"], b["l_hist"]), 1))
      aseq.append((pr["acceptsel"], (b["rec4"], b["conv4"], b["m_slot"]), 48))
    else:
      aseq.append((pr["accept5k"], (asl, b["dring0"], b["dring1"], b["dring2"], b["dring3"], xsl,
                                    b["m_slot"], b["m_hist"], b["cyc_slot"], b["pos_slot"], b["cur_slot"],
                                    b["tok_hist"], b["h_seed"], b["hd_d0"], b["hd_d1"], b["dhd_seed"],
                                    b["emit"], b["l_hist"]), 1))
      aseq.append((pr["acceptsel5k"], (b["rec4"], b["conv4"], b["m_slot"], b["conv5x"]), 48))
  accept_g = ParityGraph(aseq, tag="r6A")
  flush_g = ParityGraph([(pr["dposadd"], (d["fillpos"], d["dpos1"]), 1)], tag="r6F")
  # R6 rung 4: the LOOKUP-ONLY draft graph (the R7a draft-skip law, batched):
  # valid ONLY on cycles where EVERY k4 stream's PREVIOUS lookup hit (>=8) — on
  # a miss the chains' drings would be stale-but-valid (probe-verifiable, safe
  # but low-acceptance); host picks per cycle via prev emits. Skips both draft
  # chains (~10.6ms) on both-hit cycles.
  lu_only = [e for e in dseq if str(e[0].name).startswith("lookup")]
  draft_lu_g = ParityGraph(lu_only, tag="r6DL") if lu_only else None
  log(f"r6 graphs (BT={bt}): draft {len(dseq)}k, probe {len(seq)}k, accept {len(aseq)}k"
      f"{f', draft_lu {len(lu_only)}k' if draft_lu_g else ''}")
  return (draft_g, probe_g, accept_g, flush_g), st, draft_lu_g

class BatchSession:
  def __init__(self, graphs, st, draft_lu_g=None, lu_mode=0):
    self.g, self.st, self.draft_lu_g, self.lu_mode = graphs, st, draft_lu_g, lu_mode
    self.prev = None
    self.prev_hits = None
  def begin(self): self.prev = dev.timeline_value - 1; self.prev_hits = None
  def step(self):
    if self.prev is None: self.begin()
    draft_g, probe_g, accept_g, flush_g = self.g
    if self.draft_lu_g is not None and self.prev_hits is not None and        all(h >= 9 for h in self.prev_hits):
      draft_g = self.draft_lu_g   # rung 4: both-hit cycle -> chains skipped
    prev = self.prev
    vd = dev.next_timeline(); draft_g.submit(prev, vd)
    vp = dev.next_timeline(); probe_g.submit(vd, vp)
    va = dev.next_timeline(); accept_g.submit(vp, va)
    vf = dev.next_timeline(); flush_g.submit(va, vf)
    self.prev = vf
    dev.timeline_signal.wait(vf)
    res = []
    hits = []
    for b in self.st:
      nm = "emit" if b["s"] == 0 else f"emit_s{b['s']}"
      e = E.P.down_at(nm, 0, 26, np.int32)
      m = int(e[1])
      hits.append(int(e[9]))
      res.append({"pos_new": int(e[0]), "m": m, "stop": int(e[7]), "cyc": int(e[8]), "hit": int(e[9]),
                  "tokens": [int(t) for t in e[2:3+m]][:m+1]})
    self.prev_hits = hits
    return res

# ============================ gates =====================================
def _nb(b):
  n = getattr(b, "nbytes", None)
  if n is not None: return int(n)
  try: return int(getattr(b, "size", 0)) * 4
  except Exception: return 0
vram = sum(_nb(b) for b in E.P.d.values())
log(f"[vram] arithmetic sum after banks: {vram/1e9:.2f} GB ({len(E.P.d)} allocs)")

def hist_s(s):
  pre = "" if s == 0 else f"_s{s}"
  return E.P.down_at(f"tok_hist{pre}", 0, CTXK+256, np.int32)

# ---- (G1) SOLO s1 gate: B=1 T=3 (M3) vs its T=1 ref, x2 det ----
solo_g, solo_st, _ = build_r6_graphs([{"s": 1, "T": 3, "mode": "k2"}])
solo_outs = []
for rep in range(2):
  anchor_load(1, ANC1)
  S1 = BatchSession(solo_g, solo_st)
  S1.begin()
  t0 = time.perf_counter()
  emt = []
  for _ in range(NTOK):
    r = S1.step()
    if r[0]["stop"]: log(f"[solo-s1] NOTE stop flag at cyc {r[0]['cyc']} (continuing — T=1 ref has no stop)")
    emt += r[0]["tokens"]
  dt = time.perf_counter() - t0
  h = hist_s(1)
  outw = h[P1:P1+NTOK]
  agree = int((outw == np.array(ref1[:NTOK])).sum())
  fd = next((k for k in range(NTOK) if outw[k] != ref1[k]), None)
  m1 = E.P.down_at("m_hist_s1", 0, NTOK, np.int32)
  tpc = float((m1[:NTOK] + 1).sum()) / NTOK
  log(f"[solo-s1] rep{rep}: {agree}/{NTOK} exact (first div {fd}), {dt/NTOK*1e3:.2f} ms/cyc, "
      f"tok/cyc={tpc:.2f}, tok/s={tpc/(dt/NTOK):.2f}")
  solo_outs.append(outw.copy())
log(f"[solo-s1] deterministic x2: {bool((solo_outs[0] == solo_outs[1]).all())}")

# ---- (G2) SOLO s0 timing baseline: B=1 T=3 (M3) on the 100k snapshot, 1 rep ----
solo0_g, solo0_st, _ = build_r6_graphs([{"s": 0, "T": 3, "mode": "k2"}])
anchor_load(0, ANC0)
S0 = BatchSession(solo0_g, solo0_st)
S0.begin()
t0 = time.perf_counter()
for _ in range(NTOK): S0.step()
dt0 = time.perf_counter() - t0
h0 = hist_s(0)
out0s = h0[P0:P0+NTOK]
m0 = E.P.down("m_hist", (NTOK,), np.int32)
tpc0 = float((m0[:NTOK] + 1).sum()) / NTOK
ag0 = int((out0s == np.array(ref0[:NTOK])).sum())
log(f"[solo-s0] {ag0}/{NTOK} exact vs banked ref, {dt0/NTOK*1e3:.2f} ms/cyc, tok/cyc={tpc0:.2f}, "
    f"tok/s={tpc0/(dt0/NTOK):.2f}")

# ---- (G3) BATCHED gates: per-spec Tier-1 x2 det + aggregate timing ----
for _si, _spec in enumerate(R6_SPECS):
  batch_g, batch_st, batch_lu = build_r6_graphs(_spec)
  _tag = "+".join(f"s{x['s']}:{x['T']}{'k4' if x['mode']=='k4' else 'k2'}" for x in _spec)
  agg_outs = []
  reps_t = []
  for rep in range(2):
    anchor_load(0, ANC0)
    anchor_load(1, ANC1)
    SB = BatchSession(batch_g, batch_st, draft_lu_g=batch_lu, lu_mode=1)
    SB.begin()
    t0 = time.perf_counter()
    for _ in range(NTOK):
      rs = SB.step()
      for r in rs:
        if r["stop"]: log(f"[batch] NOTE stop flag cyc {r['cyc']} (continuing — T=1 refs have no stop)")
    dtb = time.perf_counter() - t0
    reps_t.append(dtb)
    h0 = hist_s(0); h1 = hist_s(1)
    b0 = h0[P0:P0+NTOK]; b1 = h1[P1:P1+NTOK]
    a0 = int((b0 == np.array(ref0[:NTOK])).sum())
    a1 = int((b1 == np.array(ref1[:NTOK])).sum())
    mB0 = E.P.down("m_hist", (NTOK,), np.int32)
    mB1 = E.P.down_at("m_hist_s1", 0, NTOK, np.int32)
    tc0 = float((mB0[:NTOK] + 1).sum()); tc1 = float((mB1[:NTOK] + 1).sum())
    agg = (tc0 + tc1) / dtb
    log(f"[batch/{_tag}] rep{rep}: s0 {a0}/{NTOK} exact, s1 {a1}/{NTOK} exact; {dtb/NTOK*1e3:.2f} ms/cyc; "
        f"tok/cyc s0={tc0/NTOK:.2f} s1={tc1/NTOK:.2f} sum={(tc0+tc1)/NTOK:.2f}; "
        f"AGGREGATE {agg:.2f} tok/s (s0 {tc0/dtb:.2f}, s1 {tc1/dtb:.2f})")
    if a0 < NTOK:
      fd = next((k for k in range(NTOK) if b0[k] != ref0[k]), None)
      log(f"[batch/{_tag}] s0 first div {fd}: out {b0[max(0,(fd or 0)-3):(fd or 0)+5].tolist()} ref {ref0[max(0,(fd or 0)-3):(fd or 0)+5]}")
    if a1 < NTOK:
      fd = next((k for k in range(NTOK) if b1[k] != ref1[k]), None)
      log(f"[batch/{_tag}] s1 first div {fd}: out {b1[max(0,(fd or 0)-3):(fd or 0)+5].tolist()} ref {ref1[max(0,(fd or 0)-3):(fd or 0)+5]}")
    agg_outs.append((b0.copy(), b1.copy()))
  best = min(reps_t)
  mB0 = E.P.down("m_hist", (NTOK,), np.int32); mB1 = E.P.down_at("m_hist_s1", 0, NTOK, np.int32)
  _tc = float((mB0[:NTOK]+1).sum()) + float((mB1[:NTOK]+1).sum())
  log(f"[batch/{_tag}] det x2: s0 {bool((agg_outs[0][0] == agg_outs[1][0]).all())} "
      f"s1 {bool((agg_outs[0][1] == agg_outs[1][1]).all())}; BEST {best/NTOK*1e3:.2f} ms/cyc "
      f"-> {_tc/best:.2f} tok/s aggregate")

# stock cross-check (s0; snapshot mode only — the fresh-slice prompt differs)
try:
  if S0_FRESH: raise FileNotFoundError("fresh s0")
  sb = np.array(json.load(open("~/tinygrad-metal/spec_base_100k.json")), dtype=np.int64)
  ov = int((np.array(ref0[:NTOK-1]) == sb[1:NTOK]).sum())
  log(f"[stock] s0 ref0[:59] vs spec_base_100k[1:60]: {ov}/{NTOK-1}")
except Exception as e:
  log(f"[stock] unavailable: {e}")

# lookup stats for s1 (k4 stream)
lh1 = E.P.down_at("l_hist_s1", 0, 4096, np.int32)
ncy1 = int((lh1 > 0).sum())
log(f"[lookup-s1] cycles {ncy1}, hits(l>=8) {int((lh1 >= 9).sum())}")
log("[r6 done]")
