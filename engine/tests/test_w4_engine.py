# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""TLX W4 battery — engine tripwires/asserts (GPU-free; run on the rig with
the tg311 python:  ~/tg311/bin/python engine0/tests/test_w4_engine.py
Also pytest-compatible (all test_* sync).

Covers the W4 fix list (TLX_REVIEW_LEDGER section D / wave plan W4.1-W4.4):
  W4.1 exec tripwires: maxntid/name-law mismatch (over+under launch), the
      regfile max_threads cap, the P18 spill policy (hard >100B, warn 1..100B,
      strict ==0), symbolic-dim skip. Validated against FAKE program metadata
      shaped exactly like the census table (engine0/w4_census.json).
  W4.2 wait deadline: nv_wait_timeline stamps the watchdog marker + re-raises;
      the sync_every<=2 race-law guard trips at depth 3.
  W4.3 emit monotonicity: the pure _emit_seq_violation law (pos = prev+m+1,
      tokens = m+1) — the h_generate/_process_emit choke point.
  W4.4 manifests: all 8 rungs load + self-validate; wiring happy path at
      K=10; the R4/R7 mispairing class (accept10k under K=10) raises; missing
      scratch raises; RM mismatch raises; k2 GEMVV alternates accepted;
      k2s textual audits pass on the shipped sources and catch mutations.
  W4.5 offline DEC7 golden: packed7 <-> packed inverse roundtrip on real
      weight bytes (the decode-r7 law, pure permutation, nz=0 sample).
"""
import io, os, sys, time, types, contextlib, traceback

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # engine0/
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, BASE)

import numpy as np

# ---------------- W4.1: fake-metadata exec-assert paths ----------------
class FakePrg:
  """Shaped like NVProgram's tripwire metadata (census-verified classes)."""
  def __init__(self, name="kfake", maxntid=(256, 1, 1), stack=0, regs=40):
    from tinygrad.helpers import round_up as _ru
    self.name, self.maxntid, self.stack_usage, self.regs_usage = name, list(maxntid), stack, regs
    # replicate ops_nv's regfile cap math (Registers allocation granularity)
    self.max_threads = ((65536 // _ru(max(1, regs) * 32, 256)) // 4) * 4 * 32

def _asserts(prg, ls, mode=None, spill_hard=None):
  import tinygrad.runtime.ops_nv as ops_nv
  old_m, old_h = ops_nv.NV_GRAPH_ASSERTS, ops_nv.NV_SPILL_HARD_B
  if mode is not None: ops_nv.NV_GRAPH_ASSERTS = mode
  if spill_hard is not None: ops_nv.NV_SPILL_HARD_B = spill_hard
  try:
    return ops_nv.NVComputeQueue._graph_exec_asserts(prg, ls)
  finally:
    ops_nv.NV_GRAPH_ASSERTS, ops_nv.NV_SPILL_HARD_B = old_m, old_h

def test_w4_graph_asserts_canon_class_passes():
  # h_embed11 census: regs 40, stack 0, maxntid (256,1,1), launched (256,1,1)
  _asserts(FakePrg("h_embed11", (256, 1, 1), 0, 40), (256, 1, 1))

def test_w4_graph_asserts_nw32_class_passes():
  # lookup11_nw32 census: maxntid (1024,1,1), launched 1024 via name law
  _asserts(FakePrg("lookup11_nw32", (1024, 1, 1), 0, 44), (1024, 1, 1))

def test_w4_graph_asserts_overlaunch_raises():
  # the k0ab11 misname class: 1024-thread launch on a 256-bounds cubin
  try:
    _asserts(FakePrg("k0ab11-misnamed-nw32", (256, 1, 1), 0, 40), (1024, 1, 1))
    raise AssertionError("over-launch not caught")
  except RuntimeError as e:
    assert "maxntid" in str(e), e

def test_w4_graph_asserts_underlaunch_raises():
  # stale-columns class: 128-thread launch on a 256-bounds cubin
  try:
    _asserts(FakePrg("ffn8v10-shrunk", (256, 1, 1), 0, 70), (128, 1, 1))
    raise AssertionError("under-launch not caught")
  except RuntimeError as e:
    assert "maxntid" in str(e), e

def test_w4_graph_asserts_max_threads_cap():
  # 128 regs -> max_threads 512; a 1024 launch exceeds the regfile
  try:
    _asserts(FakePrg("pfs64-class", (1024, 1, 1), 0, 128), (1024, 1, 1))
    raise AssertionError("RF overflow not caught")
  except RuntimeError as e:
    assert "max_threads" in str(e), e

def test_w4_graph_asserts_spill_hard_over_100b():
  # the P18 nondet class: 592B frame (pfg3m_gdnqg_r7_m128 census) hard-fails
  try:
    _asserts(FakePrg("pfg3m-m128", (256, 1, 1), 592, 40), (256, 1, 1))
    raise AssertionError("592B spill not caught")
  except RuntimeError as e:
    assert "592B" in str(e) and "policy" in str(e), e

def test_w4_graph_asserts_spill_warn_band():
  # hm11's documented 8B cold spill: warn-strict default keeps it bootable
  prg = FakePrg("spk_g4nw32hm11_100k", (1024, 1, 1), 8, 64)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    _asserts(prg, (1024, 1, 1))          # default mode 1
    _asserts(prg, (1024, 1, 1))          # second exec: warn ONCE
  assert buf.getvalue().count("[NV-GA] warn") == 1, buf.getvalue()
  # strict mode (2): any stack hard-fails
  try:
    _asserts(prg, (1024, 1, 1), mode=2)
    raise AssertionError("strict mode did not hard-fail 8B")
  except RuntimeError as e:
    assert "8B" in str(e), e

def test_w4_graph_asserts_symbolic_dims_skip():
  # capture-time var dims (UOp sints) must not explode the int-only checks
  class Sym: pass
  _asserts(FakePrg("var-launch", (256, 1, 1), 0, 40), (Sym(), 1, 1))

# ---------------- W4.2: wait deadline + sync_every guard ----------------
class _WedgeSig:
  def __init__(self): self.wait_called_with = None
  def wait(self, value, timeout=None):
    self.wait_called_with = (value, timeout)
    raise RuntimeError(f"Wait timeout: {timeout} ms! (the signal is not set to {value})")

class _WedgeDev:
  timeline_signal = _WedgeSig()

def test_w4_wait_timeline_marker_and_reraise():
  import tinygrad.runtime.ops_nv as ops_nv
  dev = _WedgeDev()
  buf = io.StringIO()
  try:
    with contextlib.redirect_stdout(buf):
      ops_nv.nv_wait_timeline(dev, 1234, what="unit-test", timeout_s=300)
    raise AssertionError("wedge wait did not raise")
  except RuntimeError as e:
    assert "Wait timeout" in str(e), e
  out = buf.getvalue()
  assert "[NV-WAIT-TIMEOUT]" in out and "unit-test" in out and "300" in out, out
  assert dev.timeline_signal.wait_called_with[1] == 300000, dev.timeline_signal.wait_called_with

def test_w4_sync_every_race_law_guard():
  # gcycle imports engine0.dev (GPU) at module load — stub it (import-time only;
  # run_tokens' guard fires before any dev use when given bad args).
  stub = types.ModuleType("engine0"); stub.dev = None
  held = sys.modules.get("engine0")
  sys.modules["engine0"] = stub
  try:
    import gcycle
    G = gcycle.GCycleEngine.__new__(gcycle.GCycleEngine)   # no __init__ (no dev)
    G.graphs = {0: object(), 1: object()}
    G._kick_q = None
    try:
      G.run_tokens(4, sync_every=3)
      raise AssertionError("sync_every=3 not caught")
    except AssertionError as e:
      assert "re-patch race" in str(e), e
    try:
      G.run_tokens(4, sync_every=0)
      raise AssertionError("sync_every=0 not caught")
    except AssertionError as e:
      assert "sync_every=0" in str(e), e
  finally:
    if held is not None: sys.modules["engine0"] = held
    else: sys.modules.pop("engine0", None)

# ---------------- W4.3: emit-sequence monotonicity ----------------
def test_w4_emit_seq_law():
  import serve
  ok = {"pos_new": 105, "m": 2, "tokens": [1, 2, 3]}
  assert serve._emit_seq_violation(102, ok) is None
  assert "pos_new" in serve._emit_seq_violation(101, ok)          # pos went backwards
  assert "tokens" in serve._emit_seq_violation(102, {"pos_new": 105, "m": 2, "tokens": [1, 2]})
  assert "m=" in serve._emit_seq_violation(102, {"pos_new": 105, "m": 99, "tokens": [1] * 100})
  assert "shape" in serve._emit_seq_violation(102, {"pos_new": 105})   # malformed emit

# ---------------- W4.4: manifests + audits ----------------
def test_w4_manifests_load_and_law():
  from rung_manifest import load_manifest, emit_offsets
  for K in (0, 4, 5, 6, 7, 8, 9, 10):
    man = load_manifest(K)
    assert man["M"] == (3 if K == 0 else K + 1)
    eo = emit_offsets(K)
    assert man["emit_stop_off"] == eo["stop"] and man["emit_words"] == eo["words"]
    if K >= 4:
      assert man["accept"] == f"accept{K+1}k" and man["acceptsel"] == f"acceptsel{K+1}k"
      assert man["lookup"] == f"lookup{K+1}_nw32"
      assert man["scratch_conv"] == [f"conv{i}x" for i in range(5, K + 2)]
      assert man["scratch_rec"] == [f"rec{i}x" for i in range(6, K + 2)]

def _k10_world():
  # TLX W5: the REAL canonical loaded world (mirrors mtp.py M11_CUBINS +
  # R7D/trunk extras): under DR7 the ffn8*/down8* class loads ONLY the r7
  # twins — the base twins (ffn8v11/down8nw32_11) are never loaded. The
  # pre-W5 battery modeled the manifest itself (both sets) and so never
  # caught the boot failure this class produced on the rig.
  from rung_manifest import load_manifest
  man = load_manifest(10)
  pr = {n for n in man["trunk_cu"] if not (n.startswith("ffn8") or n.startswith("down8"))} \
       | set(man["dr7_cu"]) | {man["accept"], man["acceptsel"], man["lookup"]} \
       | set(man["attn_deep"]) | {"accept", "acceptsel", "amx3", "h_embed", "k0_norm"}
  dn = set(man["scratch_conv"] + man["scratch_rec"]) | {f"dring{i}" for i in range(10)} | {"emit", "rec4", "conv4"}
  return man, pr, dn

def test_w4_manifest_wiring_happy_k10():
  from rung_manifest import assert_rung_wiring
  man, pr, dn = _k10_world()
  assert_rung_wiring(10, False, pr, 11, dn, dr7=True)   # the canonical DR7 boot

def test_w4_manifest_dr7_replacement_law():
  # TLX W5 (the live-boot finding): dr7 requires the twins and NOT the base
  # ffn/down; non-dr7 requires the base and not the twins; neither set
  # present must raise.
  from rung_manifest import assert_rung_wiring
  man, pr, dn = _k10_world()
  base = {n for n in man["trunk_cu"] if n.startswith("ffn8") or n.startswith("down8")}   # ffn8v11/down8nw32_11
  twins = set(man["dr7_cu"])                                                            # ffn8v11r7/down8nw32v11r7
  # (a) non-dr7 world: base twins loaded, r7 not — must PASS with dr7=False
  pr_base = (pr - twins) | base
  assert_rung_wiring(10, False, pr_base, 11, dn, dr7=False)
  # (b) dr7 world with NEITHER set -> raises naming the r7 twin
  pr_none = pr - twins
  try:
    assert_rung_wiring(10, False, pr_none, 11, dn, dr7=True)
    raise AssertionError("dr7 boot with no ffn/down pair did not raise")
  except RuntimeError as e:
    assert "ffn8v11r7" in str(e), e
  # (c) dr7=False with ONLY the r7 twins -> raises naming a BASE twin
  try:
    assert_rung_wiring(10, False, pr, 11, dn, dr7=False)
    raise AssertionError("non-dr7 boot with only r7 twins did not raise")
  except RuntimeError as e:
    assert ("down8nw32_11" in str(e)) or ("ffn8v11" in str(e)), e

def test_w4_manifest_wiring_k10_gemvv_k2():
  from rung_manifest import assert_rung_wiring
  man, pr, dn = _k10_world()
  # k2 world with GEMVV alternates swapped in
  alt = {"q5g8_3": "q5g8v_3", "ffn8_3": "ffn8v_3", "down8_3": "down8nw32_3",
         "op38_3": "op38nw32_3", "k3ao3": "k3aonw32_3", "ao8_3": "ao8nw32_3",
         "aq3k8_3": "aq3k8v_3", "head8_3": "head8v_3"}
  man0 = {"k2-pr": True}
  from rung_manifest import load_manifest
  m0 = load_manifest(0)
  # swap every GEMVV alternate IN (base names replaced by their v2 variants)
  pr0 = {alt.get(n, n) for n in m0["trunk_cu"]} | {"accept", "acceptsel", "k0_norm", "h_embed"}
  dn0 = {"dring0", "dring1", "emit", "rec4", "conv4"}
  assert_rung_wiring(0, False, pr0, 3, dn0)   # LOOKUP=0: no lookup_nw32 needed

def test_w4_manifest_mispair_accept10k_under_k10():
  from rung_manifest import assert_rung_wiring
  man, pr, dn = _k10_world()
  pr.remove("accept11k"); pr.add("accept10k")     # the V-52 class
  try:
    assert_rung_wiring(10, False, pr, 11, dn, dr7=True)
    raise AssertionError("accept10k-under-K10 mispairing not caught")
  except RuntimeError as e:
    assert "accept11k" in str(e), e

def test_w4_manifest_missing_scratch_raises():
  from rung_manifest import assert_rung_wiring
  man, pr, dn = _k10_world()
  dn.remove("conv11x")
  try:
    assert_rung_wiring(10, False, pr, 11, dn, dr7=True)
    raise AssertionError("missing conv11x not caught")
  except RuntimeError as e:
    assert "conv11x" in str(e), e

def test_w4_manifest_rm_mismatch_raises():
  from rung_manifest import assert_rung_wiring
  man, pr, dn = _k10_world()
  try:
    assert_rung_wiring(10, False, pr, 10, dn, dr7=True)     # RM=10 (the M10 set) under K=10
    raise AssertionError("RM mismatch not caught")
  except RuntimeError as e:
    assert "RM=10" in str(e), e

def test_w4_k2s_textual_audits_catch_mutations():
  from rung_manifest import audit_k2s
  src = open(f"{BASE}/m11.cu").read()
  audit_k2s(src, 11)                                       # shipped source passes
  for mutate, why in (
    (lambda s: s.replace("    __syncthreads();", "    __syncthreads(); __syncthreads();", 1), "3 barriers"),
    (lambda s: s.replace("for (int t = 0; t < 11; ++t) {", "for (int t = 0; t < 11; ++t) { if (t == 3) continue;"), "continue in t-loop"),
    (lambda s: s.replace("(t == 10) ? (rec10x", "(t == 10) ? (rec9x", 1), "rec-chain tail"),
  ):
    try:
      audit_k2s(mutate(src), 11)
      raise AssertionError(f"mutation not caught: {why}")
    except RuntimeError:
      pass

def test_w4_gen_manifest_emitter_idempotent():
  import subprocess
  r = subprocess.run([sys.executable, f"{BASE}/gen_manifest.py"], capture_output=True, text=True)
  assert r.returncode == 0, r.stderr[-800:]
  assert "ALL RUNGS EMITTED + AUDITED" in r.stdout

# ---------------- W4.5: offline DEC7 golden (packed7 roundtrip) ----------------
def test_w4_dec7_golden_roundtrip():
  KCH = 128
  orig = np.load(f"{BASE}/packed/fd0.npy", mmap_mode="r")       # [N, rowb] u8
  p7 = np.load(f"{BASE}/packed7/fd0.npy")                       # 5.5MB, real load
  N, rowb = orig.shape
  kdim = 17408
  nb, nch = kdim >> 8, kdim // KCH
  assert rowb == 98 * nb and N % 8 == 0
  u = p7.view(np.uint16).reshape(N // 8, nch, 32, 8)
  o16 = np.asarray(orig[:64]).view(np.uint16)                   # first 8 groups
  NCL = 4
  nz = 0
  for chunk in range(0, min(nch, 16)):                          # sample 16 chunks
    b, h = chunk >> 1, chunk & 1
    lc0 = h * 16
    for c in range(4):
      for r in range(8):
        for g in range(8):
          unit = u[g, chunk, r * 4 + c]
          row = g * 8 + r
          got = o16[row, 32*b + lc0 + c*NCL : 32*b + lc0 + c*NCL + NCL]
          nz += int((got != unit[:NCL]).sum())
          w = 8*b + (lc0 >> 2) + c
          nz += int(o16[row, nb*32 + 2*w] != unit[NCL]) + int(o16[row, nb*32 + 2*w + 1] != unit[NCL + 1])
          nz += int(o16[row, nb*48 + b] != unit[NCL + 2])
  assert nz == 0, f"packed7 inverse mismatch on fd0 sample: {nz} words differ"

def test_w4_hembed_clamp_landed():
  import glob as _g, re
  for path in sorted(_g.glob(f"{BASE}/h_embed*.cu")):
    src = open(path).read()
    assert "min(max(" in src, f"{path}: W4.5 token clamp missing"
  # rebuilt cubins stay policy-clean (census classes: 256-bounds, zero stack)
  sys.path.insert(0, BASE)
  from w4_census import parse_cubin
  for n in ("h_embed", "h_embed3", "h_embed5", "h_embed11"):
    r = parse_cubin(f"{BASE}/{n}.cubin")
    assert tuple(r["maxntid"] or ()) == (256, 1, 1), (n, r)
    assert r["min_stack"] == 0, (n, r)

def test_w4_census_policy_calibration():
  # the shipped canon+batch sets must be BOOTABLE under the default policy:
  # zero name-law-vs-maxntid mismatches, max stack 32B (warn band).
  import json
  d = json.load(open(f"{BASE}/w4_census.json"))["cubins"]
  shipped = {k: v for k, v in d.items() if v["canon"] or v["batch"]}
  mism = [k for k, v in shipped.items() if v["maxntid"] and v["name_law_ls"] != v["maxntid"][0]]
  assert not mism, f"shipped name-law mismatches: {mism}"
  stacks = sorted(((v["min_stack"] or 0, k) for k, v in shipped.items()), reverse=True)
  assert stacks[0][0] <= 100, f"shipped spill over policy: {stacks[:3]}"
  assert stacks[0][1] == "spk_pre11qh_100k" and stacks[0][0] == 32, stacks[:2]  # calibrated warn band

def _all_tests():
  return [(n[4:], getattr(sys.modules[__name__], n)) for n in dir(sys.modules[__name__])
          if n.startswith("test_") and callable(getattr(sys.modules[__name__], n))]

def main():
  tests = _all_tests()
  print(f"TLX W4 engine battery: {len(tests)} tests\n" + "=" * 60)
  failed = []
  for name, fn in tests:
    t0 = time.time()
    try:
      fn()
      print(f"PASS  {name}  ({time.time()-t0:.1f}s)")
    except Exception as e:
      failed.append(name)
      print(f"FAIL  {name}: {e}")
      traceback.print_exc()
  print("=" * 60)
  print(f"{len(tests)-len(failed)}/{len(tests)} passed")
  if failed:
    print("FAILED:", ", ".join(failed))
    return 1
  return 0

if __name__ == "__main__":
  sys.exit(main())
