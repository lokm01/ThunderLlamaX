# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R1 prompt-cache gates (PC_GATE=1 inside the test_w100k host; FULL env; the
daemon must be DOWN — GPU exclusive).

G0 unit (no GPU): hash chain determinism + config-fingerprint sensitivity,
crash safety (staging leftovers, dangling manifest refs), LRU eviction with
pin + protect survival, broken-chain fallback.
G1 8k-class: FRESH+ingest (nodes at 1024 boundaries + final 64-boundary) ->
dirty -> CACHE_HIT restore -> decode bit-exact x2; tail variant (restore +
200-token M64 tail vs full fresh); cached_tokens accounting.
G2 GAP-2: follow_up batch(M64) vs T1 — cur + decode bit-identical + alpha.
G3 turn-end node: decode to a 64-aligned pos -> capture -> dirty -> restore ->
decode continuation == uninterrupted reference (validates slot-4 GDN capture,
h_seed-as-hlast law, kv_d roundtrip through a real decode).
G4 100k restart: boot node@P0 from snap + device -> dirty -> lookup(ids) ->
restore -> decode 60/60 vs the T1 reference (+restore timing = the UX number).
"""
import os, sys, time, json, shutil, tempfile
import numpy as np
import pcache
from pcache import PromptCache, chain_keys, hkey_prefix, config_fp
from engine0 import dev

MAIN = pcache.ROOT
SNAP = os.getenv("SNAPDIR", "~/snap100k")
NDEC = int(os.getenv("PC_GATE_NDEC", "30"))
ok_all = True

def G(name, cond, extra=""):
    global ok_all
    print(f"[pcgate] {'PASS' if cond else 'FAIL'} {name} {extra}", flush=True)
    if not cond: ok_all = False

def decode(sess, E, n):
    sess.begin()
    emt = []
    for _ in range(n):
        r = sess.step()
        emt += r["tokens"]
    m = E.P.down("m_hist", (1024,), np.int32)
    alpha = float(m[:n].sum()) / (2.0 * n)
    return emt, alpha

class ChainState:
    """the serve ingest-callback bookkeeping, verbatim"""
    def __init__(self, pc, E, toks):
        self.pc, self.E, self.toks = pc, E, toks
        self.last_end, self.last_hkey = 0, None
    def cb(self, pos_after, final_b):
        want = (pos_after % pcache.STRIDE == 0 and pos_after >= pcache.STRIDE) or \
               (pos_after == final_b and pos_after >= pcache.HASH_BLK)
        if not want or pos_after - self.last_end < pcache.HASH_BLK: return
        t0 = time.perf_counter()
        node = pcache.capture_node(self.E, self.last_end, pos_after, "midprefill",
                                   fed_prefix=self.toks, parent=self.last_hkey)
        self.pc.write_node(node)
        self.last_end, self.last_hkey = pos_after, node["hkey"]
        print(f"[pcgate]   ingest node -> {pos_after} ({time.perf_counter()-t0:.2f}s down)", flush=True)
    def hkey_now(self):
        return self.last_hkey

def fresh(E, G, toks, pc=None):
    t0 = time.perf_counter()
    cs = ChainState(pc, E, toks) if pc is not None else None
    cur, posn, f64 = pcache.fresh_prefill(E, G, toks, ingest=cs.cb if cs else None,
                                          log_ingest=(lambda s, **kw: print(f"[pcgate] ingest-log {s} {kw}", flush=True)) if cs else None)
    return cur, posn, time.perf_counter() - t0, cs

# ============================ G0: unit (no GPU) ============================
def g0():
    root = tempfile.mkdtemp(prefix="pcg0_")
    os.environ.pop("PC_ROOT", None)
    pc = PromptCache(root=root)
    ids = list(range(0, 3000))
    ck = chain_keys(ids)
    G("G0.chain-determinism", ck == chain_keys(ids) and len(ck) == 3000 // 64 + 1)  # +1 partial-tail key
    ck2 = chain_keys(ids[:2048])
    G("G0.chain-prefix-consistency", ck2[2048] == ck[2048] and ck2[1024] == ck[1024])
    os.environ["SKV_K"] = "g4nw32_X"   # config flip -> different chain ROOT
    try:
        G("G0.config-fp-sensitivity", chain_keys(ids).get(2048) != ck[2048])
    finally:
        os.environ["SKV_K"] = "g4nw32"
    G("G0.config-fp-restore", chain_keys(ids).get(2048) == ck[2048])

    tiny = lambda W: {"kvb": np.zeros((16, W * 2048), np.uint8), "sc": np.zeros((16, W * 64), np.float16),
                      "kvd": np.zeros(W * 2048, np.uint8), "scd": np.zeros(W * 64, np.float16),
                      "rec": np.zeros(48 * 100, np.float32), "conv": np.zeros(48 * 10, np.float32),
                      "dhd": np.zeros(5120, np.float32)}
    fp = config_fp()
    n1 = dict(tiny(1024), pos_start=0, pos_end=1024, parent=None, config_fp=fp, hkey=ck[1024])
    n2 = dict(tiny(1024), pos_start=1024, pos_end=2048, parent=ck[1024], config_fp=fp, hkey=ck[2048])
    n3 = dict(tiny(64), pos_start=2048, pos_end=2112, parent=ck[2048], config_fp=fp, hkey=ck[2112])
    pc.write_node(n1); pc.write_node(n2); pc.write_node(n3)
    time.sleep(1.5)  # writer
    B, chain = pc.lookup(ids[:2112])
    G("G0.lookup-chain", B == 2112 and len(chain) == 3, f"B={B}")
    B2, _ = pc.lookup(ids[:2048])
    G("G0.lookup-prefix", B2 == 2048)
    # crash safety: dangling manifest ref dropped at reload; staging cleaned
    ent = pc.man["entries"][ck[2048]]
    shutil.rmtree(os.path.join(root, ent["dir"]))
    pc2 = PromptCache(root=root)
    G("G0.dangling-ref-dropped", ck[2048] not in pc2.man["entries"])
    B3, chain3 = pc2.lookup(ids[:2112])
    G("G0.broken-chain-fallback", B3 == 1024 and len(chain3) == 1, f"B={B3}")
    junk = os.path.join(root, "staging", "junkdir"); os.makedirs(junk)
    PromptCache(root=root)
    G("G0.staging-cleaned", not os.path.exists(junk))
    # eviction: tiny quota -> leaf-first LRU; pin + protect survive
    pc3 = PromptCache(root=root)
    pc3.protect = {ck[1024]}
    pc3.pin(pc3.lookup(ids[:2112])[1], "user-key")
    big = pc3.man["entries"][ck[1024]]["bytes"]
    _sav = pcache.QUOTA_BYTES
    try:
        pcache.QUOTA_BYTES = int(big * 1.05)
        pc3.evict()
        ents = set(pc3.man["entries"])
        G("G0.evict-leaf-lru-pin-protect", ck[1024] in ents and ents <= {ck[1024], ck[2048], ck[2112]},
          f"left={len(ents)}")
    finally:
        pcache.QUOTA_BYTES = _sav
    shutil.rmtree(root, ignore_errors=True)

# ============================ G1: 8k-class ============================
SLICE0 = int(os.getenv("PC_GATE_SLICE", "40000"))

def g1(E, G_, sess, ids):
    pc = PromptCache(root=MAIN)
    doc = ids[SLICE0:SLICE0 + 8192]
    cur, posn, secs, cs = fresh(E, G_, doc, pc=pc)
    cur_doc = int(cur)   # TLX W5: the DOC's fresh cur (the only valid reference; the
                         # old gate read tok_slot AFTER dirtying with another doc)
    print(f"[pcgate] G1 fresh 8192: {secs:.1f}s nodes->{cs.last_end} cur_doc={cur_doc}", flush=True)
    G("G1.ingest-depth", cs.last_end == 8192, f"last_end={cs.last_end}")
    # TLX W5: writes are ASYNC since W3 (put_nowait + writer thread + fsync) —
    # drain BEFORE any manifest read / lookup, or the gate races the writer
    # (KeyError on last_hkey / empty chains; the mock battery never saw this).
    _f = getattr(pc, "flush", None)
    _f() if _f else time.sleep(2.0)   # TLX W5 bisect: flush is W3+; old pcache sleeps
    # W3.6: the deepest ingested node's hlast must be SANE (finite, O(1..1e3))
    # — the G1 cur=0 law was a trunk-generation mismatch reading the ensure64
    # poison (7.7e31, FINITE fp32) through the stale xA64-row-63 law under the
    # M128 trunk. pcapture now refuses out-of-range hlast; this gate catches a
    # regression at CAPTURE time on the real engine.
    nd = os.path.join(MAIN, pc.man["entries"][cs.last_hkey]["dir"], "hlast.npy")
    hl = np.load(nd)
    G("G1.node-hlast-sane", bool(np.isfinite(hl).all() and np.abs(hl).max() <= 1e8),
      f"absmax={float(np.abs(hl).max()):.3e}")
    fingerprint(E, 8192, "fresh", all_layers=True)
    ref1, a1 = decode(sess, E, NDEC)
    print(f"[pcgate] G1 fresh emit[:10] = {ref1[:10]}", flush=True)

    # dirty with an unrelated doc, then CACHE_HIT restore
    fresh(E, G_, ids[30000:34000])
    B, chain = pc.lookup(doc)
    G("G1.lookup-hit", B == 8192 and len(chain) == 8, f"B={B} nodes={len(chain)}")
    t0 = time.perf_counter()
    cur_dirty = int(E.P.down_at("tok_slot", 0, 1)[0])   # the DIRTY doc's argmax (info only)
    B2, cur = pcache.restore_chain(E, chain, root=MAIN)
    t_rest = time.perf_counter() - t0
    G("G1.restore-pos", B2 == 8192)
    G("G1.cur-exact", cur == cur_doc, f"restored={cur} cur_doc={cur_doc} (dirty-argmax {cur_dirty})")
    fp_compare("fresh", "postrestore_dummy", "G1.fp-placeholder") if False else None
    fingerprint(E, 8192, "restored", all_layers=True)
    fp_compare("fresh", "restored", "G1.fingerprint-fresh-vs-restored")
    print(f"[pcgate] G1 cur: fresh-end vs restored: (see cur-exact below)", flush=True)
    out1, ra1 = decode(sess, E, NDEC)
    print(f"[pcgate] G1 restored emit[:10] = {out1[:10]}", flush=True)
    G("G1.restore-decode-exact", out1 == ref1, f"{sum(a==b for a,b in zip(out1,ref1))}/{min(len(out1),len(ref1))}")
    G("G1.restore-alpha-sane", ra1 > 0.35, f"alpha={ra1:.3f} (fresh {a1:.3f})")

    # determinism x2: dirty + restore again
    fresh(E, G_, ids[30000:34000])
    _, chain = pc.lookup(doc)
    pcache.restore_chain(E, chain, root=MAIN)
    out2, _ = decode(sess, E, NDEC)
    print(f"[pcgate] G1 determinism: out2==out1 {out2 == out1} (both restores identical?)", flush=True)
    G("G1.restore-deterministic", out2 == ref1)

    # TLX W5 discriminator: restore + MANUAL cur override (the doc's own fresh
    # cur). If THIS decodes exact, the state roundtrip is bit-exact and only
    # the derived cur was wrong; if it still diverges, the restored STATE
    # itself is not bit-exact (the midprefill capture class under this trunk).
    fresh(E, G_, ids[30000:34000])
    _, chain = pc.lookup(doc)
    B3m, _curm = pcache.restore_chain(E, chain, root=MAIN)
    # slot-only override (a full _reset_slots would re-zero the restored seeds)
    E.P.win_up("cur_slot", 0, np.array([int(cur_doc)], dtype=np.int32))
    E.P.win_up("tok_slot", 0, np.array([int(cur_doc)], dtype=np.int32))
    E.P.win_up("pos_slot", 0, np.array([int(B3m)], dtype=np.int32))
    outm, _ = decode(sess, E, NDEC)
    G("G1.restore-manual-cur-decode-exact", outm == ref1,
      f"{sum(a==b for a,b in zip(outm,ref1))}/{min(len(outm),len(ref1))} (cur={cur_doc} vs derived={_curm})")
    print(f"[pcgate] G1 restore@8192: {t_rest:.1f}s (fresh was {secs:.1f}s)", flush=True)

    # tail variant: +200 tokens the cache hasn't seen
    doc2 = doc + ids[20000:20200]
    cur2, posn2, secs2, cs2 = fresh(E, G_, doc2, pc=pc)
    ref2, _ = decode(sess, E, 20)
    G("G1.tail-ingest", cs2.last_end == 8384, f"last_end={cs2.last_end}")
    fresh(E, G_, ids[30000:34000])
    Bt, chaint = pc.lookup(doc2)
    G("G1.tail-lookup", Bt == 8384, f"B={Bt}")
    t0 = time.perf_counter()
    pcache.restore_chain(E, chaint, root=MAIN)
    E.P.win_up("cur_slot", 0, np.array([int(doc2[Bt])], dtype=np.int32))
    E.follow_up(G_, doc2[Bt + 1:])
    t_tail = time.perf_counter() - t0
    out3, _ = decode(sess, E, 20)
    G("G1.tail-decode-exact", out3 == ref2, f"{sum(a==b for a,b in zip(out3,ref2))}/{min(len(out3),len(ref2))}")
    print(f"[pcgate] G1 tail(8): {t_tail:.2f}s; full fresh doc2: {secs2:.1f}s", flush=True)
    return pc, t_rest

# ============================ G2: GAP-2 ============================
def g2(E, G_, sess, ids, pc):
    doc = ids[SLICE0:SLICE0 + 8192]
    delta = ids[SLICE0 + 8192:SLICE0 + 8392]
    def prep():
        B, chain = pc.lookup(doc)
        pcache.restore_chain(E, chain)
    prep()
    t0 = time.perf_counter()
    curA, posA, _ = E.follow_up(G_, delta, batch=False)
    tA = time.perf_counter() - t0
    outA, aA = decode(sess, E, 20)
    prep()
    t0 = time.perf_counter()
    curB, posB, _ = E.follow_up(G_, delta, batch=True)
    tB = time.perf_counter() - t0
    outB, aB = decode(sess, E, 20)
    G("G2.cur-bit-identical", curA == curB, f"curA={curA} curB={curB}")
    G("G2.decode-bit-identical", outA == outB)
    G("G2.alpha-parity", abs(aA - aB) < 0.15, f"alpha T1={aA:.3f} M64={aB:.3f}")
    print(f"[pcgate] G2 201-tok delta: T1 {tA:.1f}s vs M64 {tB:.1f}s ({tA/max(tB,1e-9):.1f}x)", flush=True)
    return tA, tB

# ============================ G3: turn-end node ============================
def g3(E, G_, sess, ids, pc):
    doc = ids[SLICE0:SLICE0 + 8192]
    B, chain = pc.lookup(doc)
    pcache.restore_chain(E, chain)
    emt = []
    pos = B
    for _ in range(NDEC):
        r = sess.step(); emt += r["tokens"]; pos = r["pos_new"]
    guard = 0
    while pos % pcache.HASH_BLK != 0 and guard < 8:
        r = sess.step(); emt += r["tokens"]; pos = r["pos_new"]; guard += 1
    print(f"[pcgate] G3 turn-end pos={pos} (aligned={pos % 64 == 0}; partial-tail hash used when not)", flush=True)
    fed = doc + emt
    node = pcache.capture_node(E, B, pos, "turnend", fed_prefix=fed, parent=chain[-1][0])
    pc.write_node(node)
    _f = getattr(pc, "flush", None)
    _f() if _f else time.sleep(2.0)   # TLX W5 bisect tolerance
    # continuation reference (uninterrupted)
    ref_cont, _ = decode(sess, E, 25)
    # dirty + restore through the turn-end node
    fresh(E, G_, ids[30000:34000])
    chain_full = chain + [(node["hkey"], pc.man["entries"][node["hkey"]])]
    B3, cur3 = pcache.restore_chain(E, chain_full, root=MAIN)
    G("G3.restore-pos", B3 == pos, f"B={B3} want {pos}")
    out, a = decode(sess, E, 25)
    G("G3.turnend-decode-exact", out == ref_cont, f"{sum(x==y for x,y in zip(out,ref_cont))}/{min(len(out),len(ref_cont))}")
    G("G3.turnend-alpha-sane", a > 0.35, f"alpha={a:.3f}")

# ============================ G4: 100k restart ============================
FP = {}

def fingerprint(E, P0, tag, all_layers=False):
    # spot rows near P0 for every attn layer + GDN slot4 blocks {0, 17, 31, 47}
    d = {}
    idxs = E.attn_idx if all_layers else E.attn_idx[:1]
    for i in idxs:
        d[f"kv{i}"] = E.P.down_at(f"kv{i}", (P0 - 64) * 2048, 64 * 2048, np.uint8).copy()
        d[f"sc{i}"] = E.P.down_at(f"sc{i}", (P0 - 64) * 64, 64 * 64, np.float16).copy()
    d["kvd"] = E.P.down_at("kv_d", (P0 - 64) * 2048, 64 * 2048, np.uint8).copy()
    from mtp import RBLK, CBLK
    for j in (0, 17, 31, 47):
        d[f"rec{j}"] = E.P.down_at("rec4", (j * 5 + 4) * RBLK * 4, 4096, np.float32).copy()
        d[f"conv{j}"] = E.P.down_at("conv4", (j * 5 + 4) * CBLK * 4, 4096, np.float32).copy()
    d["tokslot"] = E.P.down_at("tok_slot", 0, 1, np.int32).copy()
    d["hseed"] = E.P.down_at("h_seed", 0, 5120, np.float32).copy()
    FP[tag] = d

def fp_compare(a, b, name):
    keys = sorted(set(FP[a]) & set(FP[b]))
    bad = []
    for k in keys:
        if not np.array_equal(FP[a][k], FP[b][k]):
            x, y = FP[a][k].astype(np.int64), FP[b][k].astype(np.int64)
            bad.append(f"{k}:md{int(np.abs(x - y).max())}")
    G(f"{name}", not bad, "; ".join(bad[:6]))

def boot_build(E, ids, P0, CUR0):
    pc = PromptCache(root=MAIN)
    hk = hkey_prefix(ids[:P0])
    with pc.lock:
        have = hk in pc.man["entries"]
    E.reset_snapshot(SNAP, CUR0, P0)   # PARK (PC_GATE host never parked; slot4 would be init-poison)
    fingerprint(E, P0, "boot")
    if not have:
        t0 = time.perf_counter()
        node = pcache.boot_node(E, SNAP, P0, CUR0, ids)
        pc.write_node(node)
        print(f"[pcgate] boot node built {time.perf_counter()-t0:.1f}s", flush=True)
        # wait for the async writer (3.6GB) before anything reads the node
        pc.wq.join()
        print("[pcgate] boot node WRITTEN", flush=True)
    return pc

def g4(E, G_, sess, ref, ids, P0, CUR0, pc):
    # dirty, then restart-resume purely from the cache
    fresh(E, G_, ids[:4096])
    B, chain = pc.lookup(ids[:P0])
    G("G4.lookup-boot-node", B == P0 and len(chain) == 1, f"B={B}")
    t0 = time.perf_counter()
    B4, cur4 = pcache.restore_chain(E, chain, root=MAIN)
    t_rest = time.perf_counter() - t0
    G("G4.cur-exact", cur4 == CUR0, f"cur={cur4} want {CUR0}")
    fingerprint(E, P0, "after", all_layers=True)
    fingerprint(E, P0, "boot2", all_layers=True) if False else None
    fp_compare("boot", "after", "G4.fingerprint-bootstate")
    out, a = decode(sess, E, 60)
    G("G4.decode-60-of-60", out[:60] == list(ref[:60]), f"{sum(x==y for x,y in zip(out,ref))}/60")
    print(f"[pcgate] G4 emit[:12]  = {out[:12]}", flush=True)
    print(f"[pcgate] G4 ref[:12]   = {list(ref[:12])}", flush=True)
    print(f"[pcgate] G4 100k restart-resume: restore {t_rest:.1f}s, decode alpha={a:.3f}", flush=True)

def run_gates(E, G_, sess, ref, ids, CTXK):
    t00 = time.perf_counter()
    meta = json.load(open(f"{SNAP}/meta.json"))
    P0, CUR0 = int(meta["P"]), int(meta["cur0"])
    import traceback as _tb
    def guarded(name, fn, *a):
        print(f"[pcgate] ==== {name} ====", flush=True)
        try:
            return fn(*a)
        except Exception as e:
            global ok_all
            ok_all = False
            print(f"[pcgate] FAIL {name} EXCEPTION {e!r}", flush=True)
            _tb.print_exc()
            return None
    guarded("G0 unit", g0)
    pc = guarded("boot node (before any dirtying)", boot_build, E, ids, P0, CUR0)
    if pc is None: pc = PromptCache(root=MAIN)
    r = guarded("G1 8k-class", g1, E, G_, sess, ids)
    if r is not None: pc, t_rest = r
    guarded("G2 GAP-2", g2, E, G_, sess, ids, pc)
    guarded("G3 turn-end", g3, E, G_, sess, ids, pc)
    guarded("G4 100k restart", g4, E, G_, sess, ref, ids, P0, CUR0, pc)
    print(f"[pcgate] ==== SUMMARY: {'ALL GREEN' if ok_all else 'FAILURES PRESENT'} "
          f"({time.perf_counter()-t00:.0f}s) ====", flush=True)
    if not ok_all: sys.exit(1)
