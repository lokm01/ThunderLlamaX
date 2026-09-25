# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""TLX W3 battery — pcache hardening (GPU-free; needs numpy -> run on the rig
with the tg311 python:  ~/tg311/bin/python engine0/tests/test_pcache_w3.py
Also pytest-compatible (all test_* sync).

Covers the W3 fix list (TLX_REVIEW_LEDGER section C):
  V-39 durability: fsync'd npy writes + per-node sha256/shapes; restore
      validates size+shape+hash BEFORE any win_up (corrupt/planted -> refuse,
      never reach the GPU); crash-between-save-and-rename healed at boot.
  V-41 concurrency: lookup protects the chain inside the lock (TOCTOU
      repro); graveyard eviction; transactional restore (zero uploads on
      refusal); broken-chain fallback + quarantine.
  V-42 write safety: orphan reap; _save_man-failure reaps the renamed node;
      queued-node bytes counted toward the quota.
  V-40 backpressure: put_nowait + drop-on-full (never blocks the caller);
      bounded flush().
  V-44 TTL plumbing: pin(chain, key, ttl) honored + clamped.
  V-45 pin budgets: byte cap, per-key cap, oldest-pin expiry.
  V-47 LRU monotonic counter (equal wall-clock -> insertion order).
  V-43 manifest trust: strict hkey regex; {"dir":"../../evil"} rejected
      without deleting outside the root.
  W3.5 probe gap: lookup probes EVERY 64-boundary (turn-end nodes at
      non-STRIDE positions reachable from longer requests).
  W3.6 G1 law: midprefill hlast per trunk generation (M128 reads xA128 row
      127, M64 reads xA64 row 63; non-finite -> loud refusal) + the
      multi-node mock-engine restore gate (cur derived from hlast, all
      windows uploaded, byte-exact roundtrip).
"""
import os, sys, json, time, shutil, tempfile, threading, types, queue, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ENG0 = os.path.dirname(HERE)
sys.path.insert(0, ENG0)

import numpy as np

# ---- GPU-free engine stubs (pcache imports these lazily inside the
# capture/restore functions; inject BEFORE any call) ------------------------
def _install_stubs(m128=False, m128on=False):
    mtp = types.ModuleType("mtp"); mtp.RBLK = 1024; mtp.CBLK = 64
    eng0 = types.ModuleType("engine0")
    eng0.dev = types.SimpleNamespace(synchronize=lambda: None, timeline_value=0)
    trunk = types.ModuleType("trunk"); trunk.VOCAB = 4096
    pfp = types.ModuleType("pf_prefill"); pfp.M128 = m128; pfp._M128ON = m128on
    for name, mod in (("mtp", mtp), ("engine0", eng0), ("trunk", trunk), ("pf_prefill", pfp)):
        sys.modules[name] = mod
    return pfp

_install_stubs()
import pcache
from pcache import PromptCache, chain_keys, config_fp, NodeCorrupt

REAL_QUOTA = pcache.QUOTA_BYTES
REAL_PIN_BUDGET = pcache.PIN_BYTE_BUDGET

def _restore_env():
    pcache.QUOTA_BYTES = REAL_QUOTA
    pcache.PIN_BYTE_BUDGET = REAL_PIN_BUDGET
    pcache.PIN_MAX_NODES = 96

CTXK = 8192
# REAL engine index classes: attn layers 0..15, GDN blocks 16..63 — the
# trunk buffers rec{i}/conv{i}_0 (i>=16) never collide with the spec
# slot-4 banks rec4/conv4 (the (j*5+4)-strided [48][5] layout)
NL, NG = 16, 48          # attn / gdn block counts (mock)

# ---- the mock engine --------------------------------------------------------
class MockP:
    """P.d[name] = 1-D backing array; down_at slices by dtype items; win_up
    WRITES THROUGH into the backing buffer (so mock kernels see uploads) and
    records (name, byte-off, copy)."""
    def __init__(self):
        self.d = {}
        self.win = []
        self._keep = []
    def _buf(self, name, n, dtype, fill):
        self.d[name] = np.full(n, fill, dtype=dtype)
    def down_at(self, name, off, n, dtype=np.int32):
        a = self.d[name]
        item = np.dtype(dtype).itemsize
        s = off // item
        assert s + n <= a.size, (name, s, n, a.size)
        return np.ascontiguousarray(a[s:s + n]).astype(dtype)
    def win_up(self, name, off, arr):
        arr = np.ascontiguousarray(arr)
        self.win.append((name, off, arr.copy()))
        a = self.d[name]
        item = arr.dtype.itemsize
        s = off // item
        a[s:s + arr.size] = arr.reshape(-1)

class MockE:
    def __init__(self, ctxk=CTXK):
        self.attn_idx = list(range(NL))
        self.gdn_idx = list(range(16, 16 + NG))
        self.P = MockP()
        P = self.P
        for i in self.attn_idx:
            P._buf(f"kv{i}", ctxk * 2048, np.uint8, (i * 7 + 1) % 251)
            P._buf(f"sc{i}", ctxk * 64, np.float16, 0.5 + i * 0.01)
        P._buf("kv_d", ctxk * 2048, np.uint8, 200)
        P._buf("sc_d", ctxk * 64, np.float16, 0.75)
        for i in self.gdn_idx:                       # trunk GDN (midprefill source)
            P._buf(f"rec{i}", 1024, np.float32, 0.1 * (i + 1))
            P._buf(f"conv{i}_0", 64, np.float32, 0.01 * (i + 1))
        P._buf("rec4", NG * 5 * 1024, np.float32, 0.0)   # restore target (slot 4)
        P._buf("conv4", NG * 5 * 64, np.float32, 0.0)
        P._buf("REC1", 17 * 5120, np.float32, 0.0)       # draft ring; row16 = dhd law
        P.d["REC1"][16 * 5120:17 * 5120] = np.linspace(0.5, 1.5, 5120).astype(np.float32)
        P._buf("xA64", 64 * 5120, np.float32, 0.0)
        P._buf("xA128", 128 * 5120, np.float32, 0.0)
        P._buf("hd_d0", 5120, np.float32, 0.0)
        P._buf("xh", 5120, np.float32, 0.0)
        P._buf("logits", 4096, np.float32, -1e9)
        P._buf("tok_slot", 1, np.int32, 0)
        P._buf("pos_slot", 1, np.int32, 0)
        P._buf("tok_hist", 1024, np.int32, -1)
        P._buf("dhd_seed", 5120, np.float32, 0.0)
        P._buf("h_seed", 5120, np.float32, 0.0)
        self.W = {("onw", 0): 1, ("head", 0): 1}
        self.resets = []
        self.flushes = 0
        self.pr = {"pfk_n16": self._k_norm, "head8": self._head8, "h_argmax": self._h_argmax}
    # mock kernels: norm = identity, head = one-hot at int(h[0]), argmax -> tok_slot
    def _k_norm(self, src, w, dst, **kw):
        self.P.d["xh"][:] = src
    def _head8(self, w, xh, logits, **kw):
        cur = int(xh[0]) % 4096
        self.P.d["logits"][:] = -1e9
        self.P.d["logits"][cur] = 1e9
    def _h_argmax(self, logits, tok_slot, pos_slot, tok_hist, **kw):
        self.P.d["tok_slot"][0] = int(np.argmax(logits))
    def _flush(self):
        self.flushes += 1
    def _reset_slots(self, cur, B):
        self.resets.append((cur, B))

def _seed_boundary_hlast(E, B, m128, token):
    """Plant the trunk hidden at boundary B-1 into the generation's buffer."""
    P = E.P
    row = np.full(5120, 0.5, dtype=np.float32)
    row[0] = float(token)          # the mock head derives cur = int(hlast[0])
    if m128:
        P.d["xA128"][127 * 5120:128 * 5120] = row
        P.d["xA64"][63 * 5120:64 * 5120] = 7.7e31      # ensure64 poison class
    else:
        P.d["xA64"][63 * 5120:64 * 5120] = row

def _capture_chain(root_dir, boundaries, ids, m128=False):
    """Fresh-cache capture of a multi-node chain at the given boundaries."""
    pc = PromptCache(root=root_dir)
    E = MockE()
    parent = None
    prev = 0
    for B in boundaries:
        _seed_boundary_hlast(E, B, m128, token=1100 + B % 900)
        node = pcache.capture_node(E, prev, B, "midprefill", fed_prefix=ids, parent=parent)
        r = pc.write_node(node)
        assert not r.get("dropped"), r
        parent = node["hkey"]
        prev = B
    pc.flush(timeout=20)
    return pc, E

IDS = list(range(5000, 5000 + 8192))     # in-vocab distinct ids

# ==============================================================================
def test_w3_multinode_roundtrip_and_cur():
    """The G1-class mock gate: 3-node chain capture -> lookup -> restore.
    All windows upload in chain order, GDN slot-4 state from the LAST node,
    cur derived from the recorded hlast (identity-norm mock head)."""
    d = tempfile.mkdtemp(prefix="pcw3_rt_")
    try:
        pc, E = _capture_chain(d, [1024, 2048, 3072], IDS)
        B, chain = pc.lookup(IDS[:3072], min_hit=64)
        assert B == 3072 and len(chain) == 3, (B, len(chain))
        E2 = MockE()
        rb, cur = pcache.restore_chain(E2, chain, root=d)
        assert rb == 3072
        assert cur == 1100 + 3072 % 900, cur
        # KV windows uploaded per layer at the right byte offsets
        for i in E2.attn_idx:
            for A in (0, 1024, 2048):
                got = [w[2] for w in E2.P.win if w[0] == f"kv{i}" and w[1] == A * 2048]
                assert got and got[0][0] == (i * 7 + 1) % 251, (i, A)
        # GDN state from the LAST capture (trunk rec{i} -> slot-4 rec4 upload)
        rec_ups = [w for w in E2.P.win if w[0] == "rec4"]
        assert len(rec_ups) == NG
        for j, i in enumerate(E2.gdn_idx):
            assert np.allclose(rec_ups[j][2], 0.1 * (i + 1), atol=1e-6), i
        # dhd from the REC1 row-16 law roundtrips into dhd_seed
        dh = [w for w in E2.P.win if w[0] == "dhd_seed"]
        assert dh and np.allclose(dh[-1][2], np.linspace(0.5, 1.5, 5120), atol=1e-6)
        assert E2.resets == [(cur, 3072)]
        assert E2.flushes >= 3
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_w3_g1_hlast_per_trunk_generation():
    """W3.6: M128 world reads xA128 row 127 (xA64 holds poison and is NOT
    read); M64 world reads xA64 row 63. Non-finite source refuses loudly."""
    d = tempfile.mkdtemp(prefix="pcw3_g1_")
    try:
        # M128: xA64 poisoned, xA128 row 127 valid -> capture succeeds
        _install_stubs(m128=True, m128on=True)
        E = MockE()
        _seed_boundary_hlast(E, 1024, m128=True, token=1234)
        node = pcache.capture_node(E, 0, 1024, "midprefill", fed_prefix=IDS)
        assert node["hlast"][0] == 1234.0, node["hlast"][:3]
        assert np.isfinite(node["hlast"]).all()
        # M128 with a POISONED xA128 too -> loud refusal (the G1 guard; the
        # 7.7e31 ensure64 poison is FINITE fp32 — only the sane-range check
        # catches it before the norm overflow turns it into argmax-0 garbage)
        E.P.d["xA128"][127 * 5120:128 * 5120] = 7.7e31
        try:
            pcache.capture_node(E, 0, 1024, "midprefill", fed_prefix=IDS)
            assert False, "poisoned boundary hlast must raise"
        except RuntimeError as e:
            assert "out of range" in str(e)
        # M64 world: reads xA64 row 63
        _install_stubs(m128=False, m128on=False)
        E = MockE()
        _seed_boundary_hlast(E, 1024, m128=False, token=777)
        node = pcache.capture_node(E, 0, 1024, "midprefill", fed_prefix=IDS)
        assert node["hlast"][0] == 777.0
    finally:
        _install_stubs(m128=False, m128on=False)
        shutil.rmtree(d, ignore_errors=True)

def test_w3_restore_refuses_corrupt_zeroed_npy():
    """THE pre-win_up invariant: a correctly-sized zero-filled (torn write)
    kvb fails the sha256 check and the engine sees ZERO uploads."""
    d = tempfile.mkdtemp(prefix="pcw3_c0_")
    try:
        pc, E = _capture_chain(d, [1024, 2048], IDS)
        B, chain = pc.lookup(IDS[:2048], min_hit=64)
        ent = pc.man["entries"][chain[1][0]]
        p = os.path.join(d, ent["dir"], "kvb.npy")
        with open(p, "r+b") as f:           # same size, header intact, data zeroed
            f.seek(256)
            f.write(b"\x00" * (os.path.getsize(p) - 256))
        E2 = MockE()
        try:
            pcache.restore_chain(E2, chain, root=d)
            assert False, "corrupt node must refuse"
        except NodeCorrupt as nc:
            assert nc.hkey == chain[1][0]
            assert "sha256" in nc.why
        assert E2.P.win == [], "no win_up may run before full-chain validation"
        assert E2.resets == []
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_w3_restore_truncated_and_planted_shapes():
    d = tempfile.mkdtemp(prefix="pcw3_tr_")
    try:
        pc, E = _capture_chain(d, [1024], IDS)
        B, chain = pc.lookup(IDS[:1024], min_hit=64)
        ent = pc.man["entries"][chain[0][0]]
        # truncated file -> size mismatch
        p = os.path.join(d, ent["dir"], "rec.npy")
        with open(p, "r+b") as f:
            f.truncate(os.path.getsize(p) - 4096)
        E2 = MockE()
        try:
            pcache.restore_chain(E2, chain, root=d)
            assert False
        except NodeCorrupt as nc:
            assert "size" in nc.why
        assert E2.P.win == []
        # planted shape (same dtype+nbytes, transposed) -> shape mismatch
        pc2, _ = _capture_chain(d + "b", [1024], IDS)
        _, chain2 = pc2.lookup(IDS[:1024], min_hit=64)
        ent2 = pc2.man["entries"][chain2[0][0]]
        p2 = os.path.join(d + "b", ent2["dir"], "rec.npy")
        old = np.load(p2)
        np.save(p2, old.reshape(old.shape[1], old.shape[0]))   # same nbytes
        E3 = MockE()
        try:
            pcache.restore_chain(E3, chain2, root=d + "b")
            assert False
        except NodeCorrupt as nc:
            assert "shape" in nc.why
        assert E3.P.win == []
    finally:
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(d + "b", ignore_errors=True)

def test_w3_adjacency_break_detected():
    d = tempfile.mkdtemp(prefix="pcw3_aj_")
    try:
        pc, _ = _capture_chain(d, [1024, 2048], IDS)
        B, chain = pc.lookup(IDS[:2048], min_hit=64)
        hk2 = chain[1][0]
        pc.man["entries"][hk2]["pos_start"] = 1536     # hand-edit the window
        E2 = MockE()
        try:
            pcache.restore_chain(E2, chain, root=d)
            assert False
        except NodeCorrupt as nc:
            assert "adjacency" in nc.why
        assert E2.P.win == []
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_w3_quarantine_and_broken_chain_fallback():
    """Corrupt node quarantined -> the NEXT lookup falls back to the shallower
    complete chain (never serves partial state)."""
    d = tempfile.mkdtemp(prefix="pcw3_q_")
    try:
        pc, _ = _capture_chain(d, [1024, 2048], IDS)
        B, chain = pc.lookup(IDS[:2048], min_hit=64)
        pc.quarantine(chain[1][0])
        assert chain[1][0] not in pc.man["entries"]
        assert not os.path.isdir(os.path.join(d, chain[1][0]))
        B2, chain2 = pc.lookup(IDS[:2048], min_hit=64)
        assert B2 == 1024 and len(chain2) == 1, (B2, len(chain2))
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_w3_crash_windows_heal_at_boot():
    """Crash between np.save and rename (staging junk) AND between rename and
    manifest (orphan final dir): both reaped at the next boot; foreign dirs
    and valid nodes untouched."""
    d = tempfile.mkdtemp(prefix="pcw3_bo_")
    try:
        pc, _ = _capture_chain(d, [1024], IDS)
        hk = next(iter(pc.man["entries"]))
        os.makedirs(os.path.join(d, "staging", "n_xx_torn"))
        np.save(os.path.join(d, "staging", "n_xx_torn", "kvb.npy"), np.zeros(8, np.uint8))
        orphan = "ab" * 32                                   # hkey-shaped
        os.makedirs(os.path.join(d, orphan))
        np.save(os.path.join(d, orphan, "rec.npy"), np.zeros(8, np.float32))
        foreign = os.path.join(d, "not_a_node"); os.makedirs(foreign)
        pc2 = PromptCache(root=d)
        assert set(pc2.man["entries"]) == {hk}
        assert not os.path.exists(os.path.join(d, "staging", "n_xx_torn"))
        assert not os.path.exists(os.path.join(d, orphan))
        assert os.path.isdir(foreign), "foreign dirs must never be touched"
        B, chain = pc2.lookup(IDS[:1024], min_hit=64)
        assert B == 1024
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_w3_manifest_evil_dir_rejected():
    """V-43: a torn/hand-edited manifest with dir traversal is rejected and
    NOTHING outside the cache root is deleted."""
    d = tempfile.mkdtemp(prefix="pcw3_ev_")
    outside = tempfile.mkdtemp(prefix="pcw3_victim_")
    try:
        pc, _ = _capture_chain(d, [1024], IDS)
        m = json.load(open(os.path.join(d, "manifest.json")))
        evil = "cd" * 32
        m["entries"][evil] = dict(next(iter(m["entries"].values())))
        m["entries"][evil]["dir"] = "../../" + os.path.basename(outside)
        json.dump(m, open(os.path.join(d, "manifest.json"), "w"))
        pc2 = PromptCache(root=d)
        assert evil not in pc2.man["entries"]
        assert os.path.isdir(outside), "path outside the cache root was deleted!"
        assert len(pc2.man["entries"]) == 1
    finally:
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)

def test_w3_lookup_probes_all_64_boundaries():
    """W3.5: a node at a non-STRIDE 64-boundary (turn-end class) is reachable
    from a LONGER request (the old probe set missed it)."""
    d = tempfile.mkdtemp(prefix="pcw3_pb_")
    try:
        pc, _ = _capture_chain(d, [2112], IDS)      # 2112 = 33*64, not a 1024 multiple
        B, chain = pc.lookup(IDS[:2500], min_hit=64)
        assert B == 2112 and len(chain) == 1, B
        B0, c0 = pc.lookup(IDS[:2113], min_hit=4096)    # min_hit still gates
        assert B0 == 0
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_w3_toctou_lookup_protects_chain():
    """V-41 repro: an eviction storm concurrent with lookup->restore never
    reaps the chain being served (protect happens inside the lookup lock)."""
    d = tempfile.mkdtemp(prefix="pcw3_to_")
    try:
        pc, E = _capture_chain(d, [1024, 2048], IDS)
        B0, chain0 = pc.lookup(IDS[:2048], min_hit=64)   # protect BEFORE the storm
        assert B0 == 2048 and len(chain0) == 2
        pcache.QUOTA_BYTES = 1          # evict wants EVERYTHING gone
        stop = threading.Event()
        def storm():
            while not stop.is_set():
                pc.evict()
        t = threading.Thread(target=storm, daemon=True); t.start()
        try:
            for _ in range(20):
                B, chain = pc.lookup(IDS[:2048], min_hit=64)
                assert B == 2048 and len(chain) == 2, (B, len(chain))
                assert all(os.path.isdir(os.path.join(d, hk)) for hk, _ in chain)
                pc.touch(chain)
        finally:
            stop.set(); t.join(2)
        # release protection -> the storm-class evict can now reclaim (still
        # under the tiny quota; restore it only afterwards)
        pc.clear_protect()
        pc.evict()
        assert len(pc.man["entries"]) <= 1, list(pc.man["entries"])
        pcache.QUOTA_BYTES = REAL_QUOTA
    finally:
        pcache.QUOTA_BYTES = REAL_QUOTA
        shutil.rmtree(d, ignore_errors=True)

def test_w3_lru_monotonic_counter():
    """V-47: equal wall-clock last_hit values evict in insertion (lrc) order,
    leaves first."""
    d = tempfile.mkdtemp(prefix="pcw3_lru_")
    try:
        pc = PromptCache(root=d)
        hks = []
        for i, pos in enumerate((1024, 2048, 3072)):
            hk = chain_keys(IDS)[pos]
            hks.append(hk)
            os.makedirs(os.path.join(d, hk))
            json.dump({"pos_start": pos - 1024, "pos_end": pos, "has_hlast": False},
                      open(os.path.join(d, hk, "meta.json"), "w"))
            with open(os.path.join(d, hk, "rec.npy"), "wb") as f: f.write(b"\x00" * 10)
            with pc.lock:
                pc.man["entries"][hk] = {
                    "dir": hk, "pos_start": pos - 1024, "pos_end": pos,
                    "parent": hks[i - 1] if i else None, "config_fp": config_fp(),
                    "bytes": 100, "sizes": {"rec": 10}, "last_hit": 1000.0,
                    "lrc": pc._next_lrc(), "pin": None, "state": "ready"}
        pcache.QUOTA_BYTES = 200           # 3 x 100 -> one leaf must go
        pc.evict()
        assert hks[2] not in pc.man["entries"] and hks[0] in pc.man["entries"]
        pcache.QUOTA_BYTES = 150           # node2 is now the leaf
        pc.evict()
        assert hks[1] not in pc.man["entries"] and hks[0] in pc.man["entries"]
    finally:
        pcache.QUOTA_BYTES = REAL_QUOTA
        shutil.rmtree(d, ignore_errors=True)

def test_w3_pin_ttl_budgets_and_expiry():
    """V-44/V-45: per-request TTL honored + clamped; byte budget + per-key cap
    enforced; oldest pins expire; expired pins evictable."""
    d = tempfile.mkdtemp(prefix="pcw3_pin_")
    try:
        pc, _ = _capture_chain(d, [1024, 2048], IDS)
        _, chain = pc.lookup(IDS[:2048], min_hit=64)
        # byte budget below one node -> nothing pins
        pcache.PIN_BYTE_BUDGET = 10
        r = pc.pin(chain, "k1", ttl=3600)
        pcache.PIN_BYTE_BUDGET = REAL_PIN_BUDGET
        assert r["pinned"] == 0, r
        assert pc.man["entries"][chain[0][0]]["pin"] is None
        # ttl clamp + honoring
        r = pc.pin(chain, "k1", ttl=1)                     # clamped to >= 60
        assert r["pinned"] == 2
        until = pc.man["entries"][chain[0][0]]["pin"]["until"]
        assert 59 <= until - time.time() <= 61
        pc.pin(chain, "k1", ttl=10**9)                     # clamped to <= 7d
        assert pc.man["entries"][chain[0][0]]["pin"]["until"] <= time.time() + 7 * 86400 + 5
        # oldest-pin expiry: a second chain under a tight budget expires k1
        hkA, eA = chain[0]; hkB, eB = chain[1]
        for e in pc.man["entries"].values(): e["pin"] = None
        pcache.PIN_BYTE_BUDGET = int(eA["bytes"])
        pc.pin([(hkA, eA)], "k1", ttl=3600)                # fits exactly
        assert pc.man["entries"][hkA]["pin"]["key"] == "k1"
        pc.pin([(hkB, eB)], "k2", ttl=3600)                # must expire k1's pin
        pcache.PIN_BYTE_BUDGET = REAL_PIN_BUDGET
        assert pc.man["entries"][hkA]["pin"] is None
        assert pc.man["entries"][hkB]["pin"]["key"] == "k2"
        # per-key node cap
        for e in pc.man["entries"].values(): e["pin"] = None
        pcache.PIN_MAX_NODES = 1
        pc.pin(chain, "k3", ttl=3600)
        pcache.PIN_MAX_NODES = 96
        n_k3 = sum(1 for e in pc.man["entries"].values() if e.get("pin") and e["pin"]["key"] == "k3")
        assert n_k3 <= 1, n_k3
        # expired pins evictable: force-expire, tiny quota, evict reclaims
        # (leaf-first: one pass takes the leaf, a second the new leaf)
        for e in pc.man["entries"].values():
            if e.get("pin"): e["pin"]["until"] = time.time() - 1
        pcache.QUOTA_BYTES = 1
        pc.clear_protect()
        pc.evict()
        pc.evict()
        assert set(pc.man["entries"]) <= {hkA}, list(pc.man["entries"])
    finally:
        _restore_env()
        shutil.rmtree(d, ignore_errors=True)

def test_w3_write_node_nonblocking_drop():
    """V-40: a saturated writer queue drops (slog pc_drop) instead of blocking
    the GPU/serving thread; flush() is bounded."""
    d = tempfile.mkdtemp(prefix="pcw3_nb_")
    try:
        pc = PromptCache(root=d)
        gate = threading.Event()
        orig = pc._write_node_sync
        def slow(node):
            gate.wait(10)
            orig(node)
        pc._write_node_sync = slow
        E = MockE()
        sent = []
        for i, pos in enumerate((1024, 2048, 3072, 4096, 5120, 6144, 7168)):
            _seed_boundary_hlast(E, pos, m128=False, token=100 + i)
            node = pcache.capture_node(E, pos - 1024, pos, "midprefill", fed_prefix=IDS)
            t0 = time.time()
            r = pc.write_node(node)
            dt = time.time() - t0
            assert dt < 0.5, f"write_node blocked {dt:.2f}s"
            sent.append(r)
        assert any(r.get("dropped") for r in sent), "queue-full must drop, not block"
        with pc.lock:
            assert pc._pend > 0, "queued bytes must count toward the quota"
        t0 = time.time()
        pc.flush(timeout=0.3)             # wedged writer -> bounded return
        assert time.time() - t0 < 2.0
        gate.set()
        pc.flush(timeout=20)
        with pc.lock:
            assert pc._outstanding == 0
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_w3_save_man_failure_reaps_renamed_node():
    """V-42: a manifest-save failure after the final rename graveyards the
    node (no invisible orphans); a retry lands it."""
    d = tempfile.mkdtemp(prefix="pcw3_sm_")
    node = None; pc = None; orig = None
    try:
        pc = PromptCache(root=d)
        E = MockE()
        _seed_boundary_hlast(E, 1024, m128=False, token=42)
        node = pcache.capture_node(E, 0, 1024, "midprefill", fed_prefix=IDS)
        orig = pc._save_man
        fails = [True]
        def flaky():
            if fails[0]:
                fails[0] = False
                raise OSError("disk full (simulated)")
            return orig()
        pc._save_man = flaky
        pc._write_node_sync(node)         # must raise through after reaping
        assert False, "should have raised"
    except OSError:
        hk = node["hkey"]
        assert not os.path.isdir(os.path.join(d, hk)), "orphan final dir left behind"
        assert hk not in pc.man["entries"]
        pc._save_man = orig               # disk "recovers": retry lands the node
        pc._write_node_sync(node)
        assert hk in pc.man["entries"]
        assert os.path.isdir(os.path.join(d, hk))
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_w3_graveyard_and_inflight_mmap_survives():
    """V-41: eviction renames + reclaims outside the locks; an in-flight
    mmapped reader of the victim keeps reading valid data (POSIX law)."""
    d = tempfile.mkdtemp(prefix="pcw3_gv_")
    try:
        pc, _ = _capture_chain(d, [1024, 2048], IDS)
        _, chain = pc.lookup(IDS[:2048], min_hit=64)
        leaf = chain[1][0]                     # leaf-first: the deepest node
        arr = np.load(os.path.join(d, leaf, "kvb.npy"), mmap_mode="r")
        sentinel = int(arr[0, 0])
        pc.clear_protect()
        pcache.QUOTA_BYTES = 1
        pc.evict()
        pcache.QUOTA_BYTES = REAL_QUOTA
        assert leaf not in pc.man["entries"]
        assert not os.path.exists(os.path.join(d, leaf))
        assert int(arr[0, 0]) == sentinel      # mapping still valid (POSIX)
        pc.evict()                             # second pass: the parent is now a leaf
        # (the last node is never evicted — the keep-one guard)
        assert set(pc.man["entries"]) <= {chain[0][0]}
    finally:
        pcache.QUOTA_BYTES = REAL_QUOTA
        shutil.rmtree(d, ignore_errors=True)

def test_w3_legacy_r1_nodes_still_restore():
    """Live-cache compatibility: R1-era nodes (sizes only — no sha256/shapes,
    hlast.npy unrecorded in sizes) restore via the size+shape+finite path;
    a POISONED legacy hlast (the G1 live-cache case) refuses."""
    d = tempfile.mkdtemp(prefix="pcw3_lg_")
    try:
        pc, E = _capture_chain(d, [1024], IDS)
        hk = next(iter(pc.man["entries"]))
        e = pc.man["entries"][hk]
        e.pop("sha256", None); e.pop("shapes", None)      # the live daemon's format
        e["sizes"].pop("hlast", None)
        json.dump({"entries": {hk: e}, "fmt": pcache.FMT},
                  open(os.path.join(d, "manifest.json"), "w"))
        pc2 = PromptCache(root=d)
        B, chain = pc2.lookup(IDS[:1024], min_hit=64)
        assert B == 1024
        E2 = MockE()
        rb, cur = pcache.restore_chain(E2, chain, root=d)
        assert rb == 1024 and cur == 1100 + 1024 % 900
        # poisoned legacy hlast -> NodeCorrupt, zero uploads
        p = os.path.join(d, hk, "hlast.npy")
        hl = np.load(p); hl[:] = 7.7e31; np.save(p, hl)
        E3 = MockE()
        try:
            pcache.restore_chain(E3, chain, root=d)
            assert False, "poisoned legacy hlast must refuse"
        except NodeCorrupt as nc:
            assert "out of range" in nc.why
        assert E3.P.win == []
    finally:
        shutil.rmtree(d, ignore_errors=True)

def _all_tests():
    return [(n, f) for n, f in sorted(globals().items())
            if n.startswith("test_") and callable(f)]

def main():
    tests = _all_tests()
    print(f"TLX W3 pcache battery: {len(tests)} tests\n" + "=" * 60)
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
