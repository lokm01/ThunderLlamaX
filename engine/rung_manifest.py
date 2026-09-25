# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""TLX W4.4 — K-suffix rung manifests (the R4/R7 mispairing tripwire).

The bug class (ledger V-52, kimi C4/F1/C6): the deep-K graph set pairs
h_embed{M}/k2s{M} trunk cubins with accept{M}k / acceptsel{M}k / lookup{M}_nw32
run-mates and scratch buffers conv5x..conv{M}x / rec6x..rec{M}x. The rung
siblings were HAND-MAINTAINED (gen_mN regenerates only mN.cu) — a missing
acceptsel{M}k silently OOBs the rec4 buffer (the K=10 accept11k+acceptsel10k
pairing gap read ~15MB past rec4's end).

This module is PURE (no engine0.dev import — safe in the GPU-free battery):
  load_manifest(K)          -> the rung's manifest dict (manifests/manifest_k{K}.json)
  assert_rung_wiring(...)   -> mtp.build_graphs calls this: every wired cubin
                               name / scratch buffer / emit layout must match
                               the manifest EXACTLY (env TLX_MANIFEST=0 skips).
  assert_batch_wiring(...)  -> r6_boot calls this for the R6 Phase-3 world
                               (TSET 3/5 + MSET BT 3/8/10 families).
  audit_k2s(src, M)         -> the V-59 textual audits over a k2s{M} kernel
                               body (exactly 2 barriers, no continue/return in
                               the t-loop, warp-uniform store zone).
Manifests are EMITTED by gen_manifest.py from the gen scripts' ground truth
(the m{M}.cu kernel tables + the accept/lookup cubin set on disk).
"""
import json, os, re

BASE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.join(BASE, "manifests")

RBLK, CBLK = 48 * 128 * 128, 3 * 10240   # mtp layout constants (rec4/conv4 rows)

def name_law_ls(nm: str) -> int:
  """The NAME-ENCODED LAUNCH CONFIG LAW (gcycle/ParityGraph)."""
  return 1024 if "nw32" in nm else 768 if "nw24" in nm else 512 if "nw16" in nm else 256

def manifest_path(K): return os.path.join(MDIR, f"manifest_k{'k2' if K == 0 else K}.json")

def load_manifest(K):
  p = manifest_path(K)
  if not os.path.exists(p): raise FileNotFoundError(f"rung manifest missing: {p} (run gen_manifest.py)")
  m = json.load(open(p))
  assert m["K"] == K, f"manifest K mismatch: {m['K']} != {K}"
  return m

# ---- the k2 (legacy, LOOKUP_K=0) manifest fields are emitted too; wiring law:
# emit record {pos_new, m, tok0..tokm, stop, cyc[, hit]} — deep rungs: stop = K+3.
def emit_offsets(K):
  if K == 0: return {"stop": 5, "cyc": 6, "hit": None, "words": 8}   # legacy accept 8-word
  return {"stop": K + 3, "cyc": K + 4, "hit": K + 5, "words": K + 6}

def _need(cond, msg):
  if not cond: raise RuntimeError(f"[manifest] {msg}")

def assert_rung_wiring(K, lookup_on, pr_keys, rm, d_names, env_on=True, dr7=False):
  """mtp.build_graphs choke point. K = LOOKUP_K (0 = the k2 world).
  pr_keys = loaded NVProgram keys; rm = the active RM (M rows); d_names =
  allocated buffer names (P.d). Raises on ANY mispairing."""
  if not env_on: return
  man = load_manifest(K)
  M = man["M"]
  _need(rm == M, f"rung K={K}: RM={rm} != manifest M={M} (probe trunk/scatch shape mismatch)")
  _need(man["emit_stop_off"] == emit_offsets(K)["stop"] and man["emit_words"] == emit_offsets(K)["words"],
        f"rung K={K}: emit layout law")
  # -- trunk + stateful cubins present and correctly suffixed
  # TLX W5 BOOT FIX (found on the live rig, 09-24): under PF_DR7 the loader
  # swaps the ffn8*/down8* trunk class to the r7 twins — the M9/10/11 load
  # lists (mtp.py M{N}_CUBINS) carry ONLY the twins, so the base twins are
  # NOT loadable in that world and must not be required. Rungs without built
  # twins (M=4..7) are untouched.
  dr7_list = list(man.get("dr7_cu", []) or [])
  req_trunk = list(man["trunk_cu"])
  if dr7 and dr7_list:
    for _cls in ("ffn8", "down8"):
      if any(n.startswith(_cls) for n in dr7_list):
        req_trunk = [n for n in req_trunk if not n.startswith(_cls)]
  alt = man.get("trunk_gemvv_alt", {})
  for n in req_trunk + (dr7_list if dr7 else []):
    ok = (n in pr_keys) or (n in alt and alt[n] in pr_keys)
    _need(ok, f"rung K={K}: trunk cubin {n} (or GEMVV alt {alt.get(n)}) not loaded (truncated boot?)")
  for role in ("accept", "acceptsel", "lookup"):
    want = man[role]
    if want is None:
      _need(role != "lookup" or not lookup_on or K == 0, f"rung K={K}: lookup wired but manifest has none")
      continue
    if role == "lookup" and K == 0 and not lookup_on:
      continue   # the R3 lookup_nw32 loads only under LOOKUP=1
    _need(want in pr_keys, f"rung K={K}: {role} cubin {want} not loaded")
    # the K-suffix pairing law: every stateful sibling carries the SAME M suffix
    if K >= 4 and role != "lookup":
      _need(want.endswith(f"{M}k"), f"rung K={K}: {role}={want} lacks the M={M} suffix (mispairing)")
  if K >= 4:
    _need(man["lookup"] == f"lookup{M}_nw32", f"rung K={K}: lookup name {man['lookup']} != lookup{M}_nw32")
  # -- scratch rows the k2s/acceptsel kernels address
  for n in man["scratch_conv"] + man["scratch_rec"]:
    _need(n in d_names, f"rung K={K}: scratch buffer {n} not allocated (acceptsel OOB class)")
  # -- deep attention set (ROWS=M) present
  for n in man.get("attn_deep", []):
    _need(n in pr_keys, f"rung K={K}: deep attention cubin {n} not loaded")
  # -- drings: h_embed{M} reads cur_slot + K drings
  for i in range(K):
    _need(f"dring{i}" in d_names, f"rung K={K}: dring{i} missing")
  return man

def assert_batch_wiring(pr_keys, mset_names, tset):
  """R6 Phase-3: the composition families (MSET per BT sum) + the T-run stateful
  set (TSET). mset_names = the cubin names the scheduler will wire (from
  r6_serve.MSET/TSET); pr_keys = loaded programs."""
  for bt, fam in mset_names.items():
    for n in fam:
      _need(n in pr_keys, f"batch BT={bt}: M-family cubin {n} not loaded")
  for T, d in tset.items():
    for role in ("embed", "k2s", "acc", "accsel", "lookup"):
      _need(d[role] in pr_keys, f"batch T={T}: {role} cubin {d[role]} not loaded")
    # suffix pairing inside the batch world: T=5 pairs accept5k/acceptsel5k/lookup5
    if T == 5:
      _need(d["acc"] == "accept5k" and d["accsel"] == "acceptsel5k" and d["lookup"] == "lookup5_nw32",
            f"batch T=5 mispairing: {d}")
    else:
      _need(d["embed"] == "h_embed3" and d["k2s"] == "k2s3" and d["acc"] == "acceptk",
            f"batch T=3 mispairing: {d}")

# ---- V-59 k2s textual audits (used by gen_mN and the offline battery) ----
def audit_k2s(cu_src: str, M: int, kname: str = None):
  """Audits over the k2s{M} kernel body: exactly TWO __syncthreads (the
  core->z4 cross-warp visibility carriers, corrected kimi C7), no
  continue/return inside the t-loop, the t-loop bound is t < M, and the
  rec-chain tail reads rec{M-1}x at t == M-1 (the REC-CHAIN SLOT LAW)."""
  kname = kname or f"k2s{M}"
  m = re.search(rf'extern "C" __global__ void __launch_bounds__\(\d+\) {kname}\(', cu_src)
  _need(m is not None, f"{kname}: kernel not found")
  # the opening brace of the kernel BODY (first '{' after the signature; args are plain)
  i0 = cu_src.index("{", m.end())
  i1 = cu_src.find('extern "C" __global__', i0)
  body = cu_src[i0: i1 if i1 > 0 else len(cu_src)]
  nbar = len(re.findall(r"__syncthreads\(\)", body))
  _need(nbar == 2, f"{kname}: {nbar} __syncthreads (expected exactly 2 — the core->z4 carriers)")
  tloop = re.search(rf"for \(int t = 0; t < {M}; \+\+t\)", body)
  _need(tloop is not None, f"{kname}: t-loop bound 't < {M}' not found")
  # no control-flow escapes inside the t-loop (warp divergence across the barriers)
  t0, t1 = tloop.span()
  loop_body = body[t1: body.find("\n  }", t1)]
  _need("continue" not in loop_body and not re.search(r"\breturn\b", loop_body),
        f"{kname}: continue/return inside the t-loop breaks barrier uniformity")
  # REC-CHAIN SLOT LAW: t reads the t-1 state. t=0 reads slot 4 (live); t<=5
  # reads rec_b slots (slot 4 = live); rec{n}x holds the t=n-1 state, so the
  # tail read at t=M-1 is rec{M-1}x (first appears at M=7 / t=6; k2s6's t=5
  # still reads slot 4 — its only scratch is the rec6x WRITE).
  if M >= 7: _need(f"(t == {M-1}) ? (rec{M-1}x" in body, f"{kname}: t={M-1} rec_in does not read rec{M-1}x")
  elif M == 6: _need("(t == 5) ? (rec6x" in body, f"{kname}: t=5 rec_out does not write rec6x")
  elif M == 5: _need("conv5x" in body, f"{kname}: conv5x window missing")
  return True
