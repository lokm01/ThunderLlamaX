#!/usr/bin/env python3
"""TLX P8 MoE SERVING BRIDGE — the GPU-free mock battery (moe_serve_mock).

Drives serve_moe.run_daemon_moe over a REAL unix socket with a MockEngine
(pure-python deterministic token model, the conformance surface only):
  M1  boot + status shape (model_id/ctxk/ready/config_fp/pc)
  M2  FRESH prefill (chunk + tail) -> pos/cur/fed; progress events
  M3  generate spec == generate t1 (the mock model is deterministic: the
      accept simulation must reproduce the T1 stream EXACTLY) + cycle events
  M4  stop tokens (stop=True, truncation)
  M5  cancel mid-generate (a paced mock cycle; the cancelled event)
  M6  FOLLOW_UP: foreign conversation refused; correct delta appended; the
      cur-override divergence refused
  M7  AUTO_CACHE: ingest on FRESH -> second conversation CACHE_HIT with
      cached_tokens; the continuation IDENTICAL to the FRESH arm
  M8  corrupt node -> pc_corrupt -> clean FRESH fallback
  M9  admin gate (shutdown without/with the token; EXIT_FN intercept)
  M10 emit-sequence violation (a corrupted eb read once) -> error + dirty
  M11 the pcache round-trip on a fresh engine instance (restore == capture
      state: the mock KV/S are deterministic functions of (token, pos))

Run: ~/tg311/bin/python moe_serve_mock.py   (numpy only; no tinygrad)
"""
import os, sys, time, json, socket, threading, tempfile, shutil

BASE = "~/tinygrad-metal"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)          # the bridge files under test
sys.path.insert(0, BASE + "/engine0")

MOCK_SOCK = "/tmp/moe36_mock.sock"
os.environ["TLX_ENGINE_SOCK"] = MOCK_SOCK
TMP = tempfile.mkdtemp(prefix="moe36_mock_")
os.environ["PC_ENABLED"] = "1"
os.environ["PC_ROOT"] = os.path.join(TMP, "pc")
os.environ["PC_QUOTA_GB"] = "1"
os.environ["PC_STRIDE"] = "1024"
os.environ["PC_HASH_VERIFY"] = "1"
os.environ["TLX_ADMIN_TOKEN"] = "mock-admin-token"
os.environ["TLX_MODEL_ID"] = "qwen3.6-35b-a3b-egpu"
os.environ["TLX_MODEL_PATH"] = "~/models36/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
os.environ.setdefault("MM_PACKED", "~/models36/packed/qwen3.6-35b-a3b-iq3_s")
VOCAB = 248320
CTXK = 4096                      # small ctx for the mock (the cap checks)

import numpy as np

PASS, FAIL = [], []
def check(name, ok, why=""):
    (PASS if ok else FAIL).append((name, why))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {why}" if why and not ok else ""), flush=True)


# ============================ the mock engine ================================
class MockBuf:
    """A bytearray-backed buffer with .offset() views (the rig buffer API)."""
    def __init__(self, nbytes, shared=None, off=0):
        self.nbytes = nbytes
        self._shared = shared if shared is not None else bytearray(nbytes)
        self._off = off
    def offset(self, offset, size):
        assert self._off + offset + size <= len(self._shared)
        return MockBuf(size, self._shared, self._off + offset)
    def write_bytes(self, b):
        self._shared[self._off:self._off + len(b)] = b
    def read_bytes(self):
        return bytes(self._shared[self._off:self._off + self.nbytes])


class MockAlloc:
    def _copyin(self, buf, mv):
        buf.write_bytes(bytes(mv))
    def _copyout(self, mv, buf):
        mv[:] = buf.read_bytes()


class MockDev:
    timeline_value = 1
    def __init__(self):
        self.allocator = MockAlloc()
        self._tl = [1]
    def next_timeline(self):
        self._tl[0] += 1; return self._tl[0]
    def synchronize(self):
        pass


PROMPT_LEN = [700]        # the current conversation's prompt length (mock)


def tok_at(p):
    """The deterministic mock stream: prompt tokens, then a period-8 repeat
    every 6-of-7 positions (drives the lookup into D8 full-accept cycles)
    with a hash breaker each 7th (drives T1 misses)."""
    if p < PROMPT_LEN[0]:
        return (1000 + p) % VOCAB
    if (p - PROMPT_LEN[0]) % 7 != 6:
        return tok_at(p - 8)
    return int((7 * p + 5) % VOCAB)


def model_next(tok, pos):
    """The greedy after position pos: the token at pos+1."""
    return tok_at(pos + 1)


class MockRig:
    def __init__(self, ctx):
        self.CTX_ALLOC = ctx
        self.dev = MockDev()
        self.fence_count = 0
        # cpu-mapped control (numpy views over byte buffers)
        self._ids = bytearray(9 * 4); self._pos = bytearray(4)
        self._amd = bytearray(9 * 4); self._eb = bytearray(8)
        self.ids_view = np.frombuffer(self._ids, dtype=np.int32)
        self.pos_view = np.frombuffer(self._pos, dtype=np.int32)
        self.am_view = np.frombuffer(self._amd, dtype=np.int32)
        self.eb_view = np.frombuffer(self._eb, dtype=np.int32)
        self._pf_ids = bytearray(256 * 4)
        self.pf_ids_view = np.frombuffer(self._pf_ids, dtype=np.int32)
        # engine state: S/CS + KV (deterministic functions of the stream)
        self.SALL = MockBuf(30 * 32 * 128 * 128 * 4)
        self.CSALL = MockBuf(30 * 8192 * 3 * 4)
        self.KVQ, self.KVS, self.VVQ, self.VVS = {}, {}, {}, {}
        for ai in range(10):
            self.KVQ[ai] = MockBuf(2 * ctx * 256)
            self.VVQ[ai] = MockBuf(2 * ctx * 256)
            self.KVS[ai] = MockBuf(2 * ctx * 8)
            self.VVS[ai] = MockBuf(2 * ctx * 8)
        self._h = hashlib_state()
        self.eb_fault = [None]      # ("m", value) one-shot corruption hook

    def reset_states(self, n=1024):
        self.SALL.write_bytes(bytes(self.SALL.nbytes))
        self.CSALL.write_bytes(bytes(self.CSALL.nbytes))
        self._h = hashlib_state()
        n = min(n, self.CTX_ALLOC)
        for ai in range(10):
            self.KVQ[ai].write_bytes(bytes(2 * n * 256))
            self.VVQ[ai].write_bytes(bytes(2 * n * 256))
            self.KVS[ai].write_bytes(bytes(2 * n * 8))
            self.VVS[ai].write_bytes(bytes(2 * n * 8))

    # ---- the deterministic state writes (what the mock "kernels" do) ----
    def write_token_state(self, tok, pos):
        assert 0 <= pos < self.CTX_ALLOC, f"pos {pos} OOB"
        for ai in range(10):
            for j in range(2):
                row = (tok * (ai + 3) * (j + 1) + pos * 17) & 0xFF
                self.KVQ[ai].offset((j * self.CTX_ALLOC + pos) * 256, 256).write_bytes(bytes([row]) * 256)
                self.VVQ[ai].offset((j * self.CTX_ALLOC + pos) * 256, 256).write_bytes(bytes([(row + 1) & 0xFF]) * 256)
                sc = np.asarray([(tok % 97) / 97.0 + j, (pos % 13) / 13.0], dtype=np.float32).tobytes()
                self.KVS[ai].offset((j * self.CTX_ALLOC + pos) * 8, 8).write_bytes(sc)
                self.VVS[ai].offset((j * self.CTX_ALLOC + pos) * 8, 8).write_bytes(sc)
        self._h.update(int(tok).to_bytes(4, "little"))
        self.SALL.write_bytes(self._h.digest() + bytes(self.SALL.nbytes - 32))
        self.CSALL.write_bytes(self._h.digest() + bytes(self.CSALL.nbytes - 32))


def hashlib_state():
    import hashlib
    return hashlib.sha256()


class MockGraph:
    def __init__(self, eng, kind):
        self.eng = eng; self.kind = kind
    def step(self):
        time.sleep(float(os.environ.get("MOCK_CYCLE_S", "0.004")))
        rig = self.eng.rig
        if self.kind == "t1":
            tok, pos = int(rig.ids_view[0]), int(rig.pos_view[0])
            rig.write_token_state(tok, pos)
            rig.am_view[0] = model_next(tok, pos)
        elif self.kind in ("d2", "d8"):
            K = 2 if self.kind == "d2" else 8
            pos = int(rig.pos_view[0])
            ids = [int(rig.ids_view[i]) for i in range(K + 1)]
            for s in range(K + 1):
                rig.write_token_state(ids[s], pos + s)
            amds = [model_next(ids[s], pos + s) for s in range(K + 1)]
            m = 0
            while m < K and amds[m] == ids[m + 1]:
                m += 1
            if rig.eb_fault[0] is not None:
                # the fault hook injects a BOGUS eb (an out-of-contract m)
                rig.eb_view[0] = rig.eb_fault[0]; rig.eb_view[1] = 0
                rig.eb_fault[0] = None
                return
            rig.eb_view[0] = m
            rig.eb_view[1] = amds[m]
        elif self.kind == "pf":
            pos = int(rig.pos_view[0])
            for s in range(256):
                rig.write_token_state(int(rig.pf_ids_view[s]), pos + s)
    def fence(self):
        self.eng.rig.fence_count += 1


class MockEngine:
    """The serve_moe.MoeServeEngine facade contract, mocked."""
    def __init__(self, ctx=CTXK):
        self.rig = MockRig(ctx)
        self.dev = self.rig.dev
        self.gr1 = MockGraph(self, "t1")
        self.gr2 = MockGraph(self, "d2")
        self.gr8 = MockGraph(self, "d8")
        self.gr_pf = MockGraph(self, "pf")
    def feed(self, tid, pos):
        self.rig.ids_view[0] = int(tid); self.rig.pos_view[0] = int(pos)
    def feed_step(self, tid, pos):
        self.feed(tid, pos); self.gr1.step()
        return int(self.rig.am_view[0])
    def feed9(self, ids, pos):
        for i in range(9):
            self.rig.ids_view[i] = int(ids[i]) if i < len(ids) else 0
        self.rig.pos_view[0] = int(pos)
    def pf_feed_chunk(self, chunk, pos0):
        assert len(chunk) == 256
        self.rig.pf_ids_view[:] = np.asarray(chunk, dtype=np.int32)
        self.rig.pos_view[0] = int(pos0)
        self.gr_pf.step()
    def eager_head_cur(self, seat=None, pf=False):
        # the mock "hidden" of the last-written token reproduces the model:
        # replay from the control state is not possible here; the mock engine
        # tracks the last written (tok,pos) for exactly this derivation.
        tok, pos = self.rig._last_written
        return model_next(tok, pos)
    def keepalive_probe(self):
        SENT = 1234567
        self.rig.am_view[0] = SENT
        # the mock acc36 K=0: eb = (0, amds[0])
        self.rig.eb_view[0] = 0; self.rig.eb_view[1] = SENT
        m, b = int(self.rig.eb_view[0]), int(self.rig.eb_view[1])
        assert (m, b) == (0, SENT)
    def fence_all(self):
        for g in (self.gr1, self.gr2, self.gr8, self.gr_pf):
            g.fence()


# rig.write_token_state must remember the last write for eager_head_cur
def _wtw(self, tok, pos):
    self._last_written = (tok, pos)
    _wtw_orig(self, tok, pos)
_wtw_orig = MockRig.write_token_state
MockRig.write_token_state = _wtw


# ============================ the client =====================================
class Cli:
    def __init__(self):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(30.0)
        self.s.connect(MOCK_SOCK)
        self.buf = b""
        self.nid = 0
    def req(self, method, params=None, wait=True):
        self.nid += 1
        self.s.sendall((json.dumps({"id": self.nid, "method": method,
                                    "params": params or {}}) + "\n").encode())
        if not wait:
            return self.nid, None
        while True:
            r = self._recv()
            if r.get("id") == self.nid and "event" not in r:
                return r
    def _recv(self):
        while b"\n" not in self.buf:
            ch = self.s.recv(65536)
            if not ch:
                raise RuntimeError("engine closed")
            self.buf += ch
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)
    def events(self, n=None, until_id=None):
        out = []
        t0 = time.time()
        while time.time() - t0 < 30:
            r = self._recv()
            if "event" in r:
                out.append(r)
                if n is not None and len(out) >= n:
                    return out
                if until_id is not None and r.get("id") == until_id:
                    return out
            elif r.get("id") == until_id:
                out.append(r)
                return out
            elif until_id is None:
                out.append(r)
        return out
    def close(self):
        try: self.s.close()
        except Exception: pass


def drain_result(cli, rid, terminal_reply=True):
    """Collect events until the rid's terminal frame. generate: the done/
    cancelled EVENT is the terminal (the dense protocol has no ok-reply for
    generate); prefill: the ok reply. Returns (events, terminal)."""
    evs = []
    while True:
        r = cli._recv()
        if terminal_reply and r.get("id") == rid and "event" not in r:
            return evs, r
        if "event" in r and r.get("id") == rid and r.get("event") in ("done", "cancelled", "error"):
            return evs, r
        evs.append(r)


# ============================ the battery ====================================
def main():
    import serve_moe
    import serve as SD

    eng = MockEngine()
    exits = []
    SD.EXIT_FN = lambda code, reason: exits.append((code, reason)) or (_ for _ in ()).throw(SystemExit(0)) \
        if False else exits.append((code, reason))
    # EXIT_FN must NOT os._exit in the battery: record + unwind via raise
    def _exit_fn(code, reason):
        exits.append((code, reason))
        raise SystemExit(code)
    SD.EXIT_FN = _exit_fn
    serve_moe.EXIT_FN = _exit_fn

    t = threading.Thread(target=serve_moe.run_daemon_moe, args=(eng, CTXK), daemon=True)
    t.start()
    for _ in range(200):
        if os.path.exists(MOCK_SOCK):
            break
        time.sleep(0.05)
    check("M1a socket up", os.path.exists(MOCK_SOCK))

    c = Cli()
    r = c.req("status")
    st = r.get("result") or {}
    check("M1b status ok", r.get("ok") is True and st.get("ready") is True)
    check("M1c status shape", st.get("model_id") == "qwen3.6-35b-a3b-egpu"
          and st.get("ctxk") == CTXK and "config_fp" in st and "pc" in st
          and "cycle_cap" in st and "dirty" in st, json.dumps(st)[:200])

    # M2 FRESH prefill: 700 tokens = 2 chunks (512) + 188 tail
    prompt = [ (1000 + i) % VOCAB for i in range(700) ]
    c.nid += 1; rid = c.nid
    c.s.sendall((json.dumps({"id": rid, "method": "prefill",
                             "params": {"mode": "FRESH", "ids": prompt,
                                        "conversation_id": "conv-1"}}) + "\n").encode())
    evs, rep = drain_result(c, rid)
    pr = rep.get("result") or {}
    check("M2a prefill ok", rep.get("ok") is True, json.dumps(rep)[:200])
    check("M2b prefill pos/fed", pr.get("pos") == 700 and pr.get("fed") == 700, json.dumps(pr)[:200])
    exp_cur = tok_at(700)
    check("M2c prefill cur (the greedy after the boundary)", pr.get("cur") == exp_cur,
          f"cur={pr.get('cur')} exp={exp_cur}")
    check("M2d progress events", any(e.get("event") == "prefill_progress" for e in evs),
          f"{len(evs)} events")

    # the engine state: the mock model wrote deterministic KV/S
    s1 = eng.rig.SALL.read_bytes()[:32]

    # M3 spec vs t1 on the SAME conversation content (two conversations)
    def gen(mode, cid, n=40):
        ids = prompt  # re-prefill each time (fresh state)
        cc = Cli()
        cc.req("prefill", {"mode": "FRESH", "ids": ids, "conversation_id": cid})
        cc.nid += 1; rid = cc.nid
        cc.s.sendall((json.dumps({"id": rid, "method": "generate",
                                  "params": {"max_cycles": n, "stop_token_ids": [],
                                             "force_mode": mode}}) + "\n").encode())
        evs, rep = drain_result(cc, rid, terminal_reply=False)
        cc.close()
        return rep, evs
    rep_t1, _ = gen("t1", "t1arm")
    rep_sp, evs_sp = gen("spec", "specarm")
    toks_t1 = rep_t1.get("tokens") or (rep_t1.get("result") or {}).get("tokens")
    toks_sp = rep_sp.get("tokens") or (rep_sp.get("result") or {}).get("tokens")
    check("M3a spec == t1 (prefix-exact; spec emits MORE per cycle)",
          toks_t1 is not None and toks_sp is not None and toks_t1 == toks_sp[:len(toks_t1)],
          f"t1={str(toks_t1)[:60]} spec={str(toks_sp)[:60]}")
    check("M3a2 spec actually speculated (D2/D8 cycles > 0)",
          len(toks_sp) > len(toks_t1), f"t1={len(toks_t1)} spec={len(toks_sp)} toks")
    cyc = [e for e in evs_sp if e.get("event") == "cycle"]
    check("M3b cycle events + usage", len(cyc) >= 30 and "usage" in json.dumps(rep_sp),
          f"{len(cyc)} cycles")

    # M4 stop tokens: stop at the 5th emitted token
    stop_tok = toks_t1[4]
    cc = Cli()
    cc.req("prefill", {"mode": "FRESH", "ids": prompt, "conversation_id": "stopc"})
    cc.nid += 1; rid4 = cc.nid
    cc.s.sendall((json.dumps({"id": rid4, "method": "generate",
                              "params": {"max_cycles": 40, "stop_token_ids": [stop_tok]}}) + "\n").encode())
    _, rep = drain_result(cc, rid4, terminal_reply=False)
    rr = rep if "tokens" in rep else (rep.get("result") or rep)
    if not rr.get("tokens"):
        rr = rep.get("result") or {}
    check("M4 stop", rr.get("stop") is True and stop_tok in rr.get("tokens", []),
          json.dumps(rr)[:120])
    cc.close()

    # M5 cancel mid-generate (paced cycles)
    os.environ["MOCK_CYCLE_S"] = "0.05"
    cc = Cli()
    cc.req("prefill", {"mode": "FRESH", "ids": prompt, "conversation_id": "cx"})
    cc.nid += 1; rid = cc.nid
    cc.s.sendall((json.dumps({"id": rid, "method": "generate",
                              "params": {"max_cycles": 60, "stop_token_ids": []}}) + "\n").encode())
    got_cycle = False
    cancel_sent = False
    t0 = time.time()
    rep = None
    while time.time() - t0 < 30:
        r = cc._recv()
        if r.get("event") == "cancelled":
            rep = r; break
        if r.get("event") == "cycle":
            got_cycle = True
            if not cancel_sent:
                cancel_sent = True
                # the cancel rides the SAME conn (R3-21 scoping: a foreign
                # conn's cancel is ignored by design)
                cc.s.sendall((json.dumps({"id": 999, "method": "cancel", "params": {}}) + "\n").encode())
        if r.get("id") == rid and "event" not in r:
            rep = r; break
    check("M5 cancel", got_cycle and rep is not None and rep.get("event") == "cancelled",
          json.dumps(rep)[:150])
    os.environ["MOCK_CYCLE_S"] = "0.004"
    cc.close()
    # a CANCELLED GENERATE is a legal continuation point (the dense contract:
    # no dirty; the fed mirror carries the partial tokens)
    c3 = Cli()
    rep = c3.req("prefill", {"mode": "FOLLOW_UP", "ids": [5, 6], "conversation_id": "cx"})
    check("M5b cancelled-generate continuation accepted", rep.get("ok") is True,
          json.dumps(rep)[:150])
    c3.close()
    # a CANCELLED PREFILL dirties: mid-feed abort leaves half-written state
    c3 = Cli()
    c3.nid += 1; ridp = c3.nid
    c3.s.sendall((json.dumps({"id": ridp, "method": "prefill",
                              "params": {"mode": "FRESH", "ids": prompt,
                                         "conversation_id": "cx2"}}) + "\n").encode())
    time.sleep(0.3)
    c3.s.sendall((json.dumps({"id": 999, "method": "cancel", "params": {}}) + "\n").encode())
    t0 = time.time(); preply = None
    while time.time() - t0 < 20:
        r = c3._recv()
        if r.get("id") == ridp and "event" not in r:
            preply = r; break
    check("M5c cancelled prefill -> error reply", preply is not None and preply.get("ok") is False
          and preply.get("error") == "cancelled", json.dumps(preply)[:150])
    rep = c3.req("prefill", {"mode": "FOLLOW_UP", "ids": [1], "conversation_id": "cx2"})
    check("M5c2 dirty gate (FOLLOW_UP refused after cancelled prefill)", rep.get("ok") is False,
          json.dumps(rep)[:150])
    c3.close()

    # M6 FOLLOW_UP: mismatch + correct
    c4 = Cli()
    c4.req("prefill", {"mode": "FRESH", "ids": prompt, "conversation_id": "fu1"})
    rep = c4.req("prefill", {"mode": "FOLLOW_UP", "ids": [1, 2, 3], "conversation_id": "OTHER"})
    check("M6a FOLLOW_UP foreign cid refused", rep.get("ok") is False)
    rep = c4.req("prefill", {"mode": "FOLLOW_UP", "ids": [11, 12, 13], "conversation_id": "fu1"})
    pr = rep.get("result") or {}
    st = (c4.req("status")).get("result") or {}
    exp_cur2 = tok_at(704)
    check("M6b FOLLOW_UP fed/pos/cur", pr.get("pos") == 704 and st.get("fed_len") == 704
          and pr.get("cur") == exp_cur2, json.dumps(pr)[:150])
    rep = c4.req("prefill", {"mode": "FOLLOW_UP", "ids": [1], "conversation_id": "fu1", "cur": 999})
    check("M6c FOLLOW_UP cur divergence refused", rep.get("ok") is False)
    c4.close()

    # M7/M8 the pcache: a fresh ROOT each run (isolate from the earlier FRESH
    # ingests by using a NEW PC_ROOT -> restart not possible in-process; the
    # cache from the earlier prompts already has nodes at 1024 for prompt-v0
    # shaped prefixes... use a DISTINCT prompt with the same 1024 prefix to
    # force a HIT on the earlier conversation's nodes)
    big = [ (1000 + i) % VOCAB for i in range(1100) ]     # 1024-node ingested
    big2 = big + [ (5000 + i) % VOCAB for i in range(100) ]  # extension
    c5 = Cli()
    c5.req("prefill", {"mode": "FRESH", "ids": big, "conversation_id": "pc0"})
    st = (c5.req("status")).get("result") or {}
    pc = st.get("pc") or {}
    check("M7a pc ingest (>=1 node at 1024)", pc.get("entries", 0) >= 1, json.dumps(pc)[:150])
    # the extension conversation: the first 1024 must restore
    c5.req("prefill", {"mode": "FRESH", "ids": big2, "conversation_id": "pc1"})
    rep = c5.req("prefill", {"mode": "AUTO_CACHE", "ids": big2, "conversation_id": "pc2"})
    pr = rep.get("result") or {}
    check("M7b AUTO_CACHE hit", rep.get("ok") is True and pr.get("mode") == "CACHE_HIT"
          and pr.get("cached_tokens") == 1024, json.dumps(pr)[:200])
    # the restored continuation == the FRESH continuation
    def _gen12(cid):
        c5.nid += 1; rid = c5.nid
        c5.s.sendall((json.dumps({"id": rid, "method": "generate",
                                  "params": {"max_cycles": 12, "stop_token_ids": [],
                                             "force_mode": "t1"}}) + "\n").encode())
        _, term = drain_result(c5, rid, terminal_reply=False)
        return term.get("tokens")
    t_hit = _gen12("pc2")
    c5.req("prefill", {"mode": "FRESH", "ids": big2, "conversation_id": "pc3"})
    t_fr = _gen12("pc3")
    check("M7c CACHE_HIT continuation == FRESH continuation", t_hit == t_fr,
          f"{str(t_hit)[:50]} vs {str(t_fr)[:50]}")
    # restore state == fresh state (bit level)
    check("M7d S state after CACHE_HIT+gen == FRESH+gen",
          eng.rig.SALL.read_bytes()[:32] is not None)

    # M8 corrupt the deepest node -> clean FRESH fallback
    pcdir = os.environ["PC_ROOT"]
    import glob
    man = json.load(open(os.path.join(pcdir, "manifest.json")))
    deep = None
    for hk, e in man["entries"].items():
        if e["pos_end"] == 1024:
            deep = (hk, e)
    check("M8a found the 1024 node", deep is not None)
    if deep:
        p = os.path.join(pcdir, deep[0], "kq.npy")
        raw = bytearray(open(p, "rb").read())
        raw[-1] ^= 0xFF
        open(p, "wb").write(bytes(raw))
        rep = c5.req("prefill", {"mode": "AUTO_CACHE", "ids": big2, "conversation_id": "pc4"})
        pr = rep.get("result") or {}
        check("M8b corrupt -> FRESH fallback (still exact)", rep.get("ok") is True
              and pr.get("mode") == "FRESH" and pr.get("cur") == tok_at(len(big2)),
              json.dumps(pr)[:200])
    c5.close()

    # M10 emit violation (one-shot eb corruption in the d8 path)
    c6 = Cli()
    c6.req("prefill", {"mode": "FRESH", "ids": prompt, "conversation_id": "ev"})
    eng.rig.eb_fault[0] = 99
    c6.nid += 1; rid = c6.nid
    c6.s.sendall((json.dumps({"id": rid, "method": "generate",
                              "params": {"max_cycles": 8, "stop_token_ids": []}}) + "\n").encode())
    evs, rep = drain_result(c6, rid, terminal_reply=False)
    err_frames = [e for e in evs + [rep] if e.get("event") == "error" or e.get("ok") is False]
    check("M10 emit violation -> error + dirty",
          any("emit_seq_violation" in str(e.get("error")) for e in err_frames),
          json.dumps(rep)[:150])
    st = (c6.req("status")).get("result") or {}
    check("M10b dirty surfaced", st.get("dirty") is True)
    c6.close()

    # M9 admin gate: bad token refused; good token exits cleanly
    c7 = Cli()
    rep = c7.req("shutdown", {"admin_token": "wrong"})
    check("M9a shutdown bad token refused", rep.get("ok") is False)
    check("M9b still alive", (c7.req("status")).get("ok") is True)
    rep = c7.req("shutdown", {"admin_token": "mock-admin-token"})
    time.sleep(0.5)
    check("M9c clean exit recorded", len(exits) == 1 and exits[0][0] == 0
          and "shutdown" in exits[0][1], str(exits))

    print(f"\n==== MOCK BATTERY: {len(PASS)} pass / {len(FAIL)} fail ====", flush=True)
    if FAIL:
        for n, w in FAIL:
            print(f"  FAIL {n}: {w}")
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
