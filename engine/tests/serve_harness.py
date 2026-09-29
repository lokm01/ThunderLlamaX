"""W-E.1 REAL-LISTENER HARNESS (TLX_REVIEW_LEDGER_R3 §4 W-E): run the REAL
serve.py daemon code — listener threads, framing, per-conn locks, watchdog,
RPC handlers, BOTH the legacy single-stream loop and the R6 batch scheduler —
against a FAKE engine object.  No GPU, no tinygrad import, no fastapi.

Run (tg311 python — numpy only):
    ~/tg311/bin/python engine0/tests/test_serve_listener.py

Mechanics:
  - serve.run_daemon's heavy imports (mtp / engine0.dev / trunk / r6_serve /
    gcycle) are pre-seeded in sys.modules with fakes BEFORE run_daemon boots,
    so the module-level daemon code (the code under test) is 100% real.
  - serve.EXIT_FN is overridden: _clean_exit records (code, reason) and
    raises HarnessExit to unwind the daemon thread instead of os._exit.
  - PC_ENABLED=0 for the harness process: the real pcache dir is never
    touched (pcache-specific behavior is covered by test_pcache_w3.py).
  - Sockets/logs/staydown paths are re-pointed into a per-boot tmpdir.
"""
import os, sys, json, time, socket, threading, types, tempfile, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENG0 = os.path.dirname(HERE)
if ENG0 not in sys.path:
    sys.path.insert(0, ENG0)

VOCAB = 260                    # the mock tokenizer vocab (engine0/tests/mock_engine)
PARK_IDS = list(range(10, 34)) # the 24-token parked base prompt


# ============================ fake modules ====================================
class _FakeDev:
    hang_synchronize = False          # R3-03 test hook: the wedged-GPU class
    @staticmethod
    def synchronize():
        if _FakeDev.hang_synchronize:
            time.sleep(600)
    timeline_value = -1


def _install_fake_modules():
    """Pre-seed sys.modules so serve.run_daemon's imports resolve to fakes.
    Idempotent; returns the mtp fake (tests may tweak LOOKUP flags on it)."""
    if "engine0" not in sys.modules or not hasattr(sys.modules["engine0"], "dev"):
        m0 = types.ModuleType("engine0"); m0.dev = _FakeDev; sys.modules["engine0"] = m0
    if "trunk" not in sys.modules:
        m1 = types.ModuleType("trunk"); m1.VOCAB = VOCAB; sys.modules["trunk"] = m1
    if "mtp" not in sys.modules:
        m2 = types.ModuleType("mtp")
        m2.RBLK = 1024; m2.CBLK = 512; m2.RM = 3
        m2.LOOKUP = 0; m2.LOOKUP_K = 0; m2.DEEP_TRIG = 1
        sys.modules["mtp"] = m2
    return sys.modules["mtp"]


# ============================ fake engine =====================================
class _DDict(dict):
    """P.d fake: unallocated names read as zeros(1) (the real plane has every
    buffer pre-allocated; a plain dict would KeyError inside keepalive)."""
    def __missing__(self, k):
        return np.zeros(1, dtype=np.int32)


class FakeP:
    """Engine buffer-plane fake: win_up records + serves down_at reads."""
    def __init__(self):
        self.d = _DDict()
        self._keep = []
        self.slots = {"pos_slot": 0, "cur_slot": 0, "tok_slot": 0,
                      "cyc_slot": 0, "m_slot": 0, "dpos1": 0}
        self.win_ups = []           # (name, off, numel)
    def win_up(self, name, off, arr):
        a = np.asarray(arr)
        self.win_ups.append((name, off, int(a.size)))
        self._keep.append(a)
        self.d[name] = a.reshape(-1)
        if name in self.slots and a.size >= 1:
            self.slots[name] = int(a.reshape(-1)[0])
    def down_at(self, name, off, n, dtype=None):
        if n == 1 and name in self.slots:
            return np.array([self.slots[name]], dtype=np.int32)
        dt = dtype or np.float32
        v = self.d.get(name)
        if v is not None and v.size >= int(n):
            return v[:int(n)].astype(dt)
        return np.zeros(int(n), dtype=dt)


class FakeEngine:
    """The engine object run_daemon drives: records every call; prefill /
    follow_up advance a scalar pos/cur the way the real one does."""
    def __init__(self):
        self.P = FakeP()
        self.attn_idx = list(range(4))     # small: snapshot loops stay fast
        self.gdn_idx = list(range(4))
        self.calls = []                    # (t, name, kw) timeline
        self.build_graphs_fails = 0        # scripted consecutive failures
        self.reset_fresh_seeds = []
        self.keepalive_fails = False       # R3-02: the keepalive wedge class
        self.reset_snapshot_hangs = False  # R3-04b test hook: the wedged boot
        self.prefill_delay = 0.0           # R3-13: slow-prefill hook
        self.prefill_t1_calls = 0
        self._install_dposadd()
    # ---- plumbing ----
    def _install_dposadd(self):
        def _dposadd(a, b, global_size=None, local_size=None, **kw):
            if self.keepalive_fails:
                raise RuntimeError("fake keepalive wedge (scripted)")
            src = self.P.d.get("fillpos")
            self.P.slots["dpos1"] = int(src.reshape(-1)[0]) + 1 if src is not None else 124
        self.pr = {"dposadd": _dposadd}
    def _rec(self, op, **kw):
        self.calls.append((time.time(), op, kw))
    # ---- boot / park ----
    def reset_snapshot(self, snap, cur0, p0):
        self._rec("reset_snapshot", snap=snap, cur0=cur0, p0=p0)
        if self.reset_snapshot_hangs:
            time.sleep(600)
        self.P.slots.update({"pos_slot": p0, "cur_slot": cur0, "tok_slot": cur0})
    def reset_fresh(self, tok):
        self.reset_fresh_seeds.append(tok)
        self._rec("reset_fresh", tok=tok)
        self.P.slots["pos_slot"] = 0
    def stload_trunk(self): self._rec("stload_trunk")
    def _mfill(self, name, val, n): self._rec("_mfill", name=name)
    def _flush(self): pass
    def stseed_spec(self, parity): self._rec("stseed_spec", parity=parity)
    def _reset_slots(self, cur, B):
        self.P.slots.update({"cur_slot": cur, "pos_slot": B, "tok_slot": cur,
                             "m_slot": 0, "cyc_slot": 0})
    def build_graphs(self):
        self._rec("build_graphs")
        if self.build_graphs_fails > 0:
            self.build_graphs_fails -= 1
            raise RuntimeError("fake build_graphs failure (scripted ka-slab class)")
    # ---- prefill paths (legacy + batch share these) ----
    def prefill_t1(self, G, toks, log=None, prog=None):
        self._rec("prefill_t1", n=len(toks))
        if prog:
            step = max(1, len(toks) // 8)
            for i in range(0, len(toks), step):
                if self.prefill_delay:
                    time.sleep(self.prefill_delay)
                prog(i, len(toks))
            prog(len(toks), len(toks))
        pos = int(self.P.slots["pos_slot"]) + len(toks)
        cur = int(toks[-1]) if toks else int(self.P.slots["cur_slot"])
        self.P.slots.update({"pos_slot": pos, "tok_slot": cur})
        self.prefill_t1_calls += 1
        return pos, cur
    def fill_draft(self, toks, start_pos=0, seed_hd=None, prog=None):
        self._rec("fill_draft", n=len(toks), start_pos=start_pos)
        if prog: prog(0, len(toks))
    def follow_up(self, G, toks, log=None, prog=None, batch=False):
        self._rec("follow_up", n=len(toks), batch=batch)
        if prog:
            prog(0, len(toks)); prog(len(toks), len(toks))
        cur = int(self.P.slots["cur_slot"])
        pos = int(self.P.slots["pos_slot"]) + 1 + len(toks)
        newcur = int(toks[-1]) if toks else cur
        self.P.slots.update({"pos_slot": pos, "tok_slot": newcur})
        return newcur, pos, len(toks)


class FakeSess:
    """DecodeSession fake: begin() re-anchors; step() emits scripted batches."""
    def __init__(self, engine):
        self.E = engine
        self.deep = 0
        self.script = [[65, 66], [67]]       # default 2-batch then 1-batch
        self.fail_at = None                  # cycle index that raises
        self.wedge_at = None                 # cycle index that hangs
        self.cycle_delay = 0.0
        self._k = 0
        self.begins = 0
    def begin(self):
        self.begins += 1; self._k = 0
    def step(self):
        self._k += 1
        if self.cycle_delay:
            time.sleep(self.cycle_delay)
        if self.wedge_at == self._k:
            time.sleep(600)
        if self.fail_at == self._k:
            raise RuntimeError("fake sess.step fault (scripted)")
        batch = list(self.script[(self._k - 1) % len(self.script)])
        pos = int(self.E.P.slots["pos_slot"])
        self.E.P.slots["pos_slot"] = pos + len(batch)
        return {"m": len(batch) - 1, "pos_new": pos + len(batch),
                "tokens": batch, "cycle": self._k, "hit": 0}


# ---- batch-path fakes (R6 scheduler + trunk engine) ----
class FakeR6:
    """r6_serve.R6Scheduler fake: composition steps driven by FakeSess logic."""
    def __init__(self, engine):
        self.E = engine
        self.g = {"solo1": object(), "b35": object(), "b53": object(), "b55": object()}
        self.begins = 0; self.steps = 0
        self.cycle_delay = 0.0
        self.script = {0: [[65, 66], [67]], 1: [[75, 76], [77]]}
        self._k = {0: 0, 1: 0}
    def rebuild_one(self, i):
        self.E._rec("r6_rebuild_one", which=i)
    def begin(self):
        self.begins += 1; self._k = {0: 0, 1: 0}
    def step(self, slots, deep):
        self.steps += 1
        if self.cycle_delay:
            time.sleep(self.cycle_delay)
        out = {}
        for s in slots:
            self._k[s] += 1
            batch = list(self.script[s][(self._k[s] - 1) % len(self.script[s])])
            pos = int(self.E.P.slots["pos_slot"]) if s == 0 else 4096 + self._k[s] * 2
            out[s] = {"m": len(batch) - 1, "pos_new": pos + len(batch),
                      "tokens": batch, "cycle": self._k[s], "hit": 0}
        return out


class FakeGCycle:
    """gcycle.GCycleEngine fake (the per-slot T=1 trunk engine)."""
    built = 0
    def __init__(self, engine):
        self.E = engine
    def build(self):
        FakeGCycle.built += 1
    def begin(self): pass
    def step(self):
        batch = [1]
        pos = int(self.E.P.slots["pos_slot"])
        self.E.P.slots["pos_slot"] = pos + len(batch)
        return {"m": 0, "pos_new": pos + len(batch), "tokens": batch,
                "cycle": 1, "hit": 0}


@contextlib.contextmanager
def FakeSwap(s):
    yield


class FakeR6Host:
    Swap = staticmethod(FakeSwap)


# ============================ the daemon wrapper ==============================
_SERVE_DEFAULTS = None

class HarnessExit(BaseException):
    pass


class ServeDaemon:
    """Boot ONE real serve.run_daemon on a private socket + tmp logs."""

    def __init__(self, batch=False, ctxk=1000, keepalive_s=None, engine=None,
                 knobs=None, boot_timeout=20):
        self.tmp = tempfile.mkdtemp(prefix="tlx_serve_h_")
        self.sock_path = os.path.join(self.tmp, "engine.sock")
        self.ctxk = ctxk
        os.environ["PC_ENABLED"] = "0"           # NEVER touch the real pcache
        _install_fake_modules()
        if batch:
            # batch-path fakes must exist before serve imports them
            fe = FakeEngine()
            sys.modules["r6_serve"] = _r6_module(fe)
            sys.modules["gcycle"] = _gcycle_module()
        import serve
        self.serve = serve
        # per-boot knob isolation: restore defaults, then apply this boot's
        global _SERVE_DEFAULTS
        try:
            if _SERVE_DEFAULTS is None:
                raise TypeError
        except (NameError, TypeError):
            _SERVE_DEFAULTS = {k: getattr(serve, k) for k in
                               ("STEP_TIMEOUT_S", "KEEPALIVE_S", "BATCH_B",
                                "GEN_REBUILD_EVERY", "MAX_LINE_BYTES",
                                "MAX_Q_DEPTH", "BATCH_REBUILD_EVERY",
                                "BATCH_PF_CHUNK", "MAX_PENDING", "MAX_CONNS",
                                "GLOBAL_REBUILD_EVERY")}
        for k, v in _SERVE_DEFAULTS.items():
            setattr(serve, k, v)
        for k, v in (knobs or {}).items():
            setattr(serve, k, v)
        # per-boot path re-pointing (module globals read at call time)
        serve.SOCK = self.sock_path
        serve.LOGF = os.path.join(self.tmp, "m1a_serve.log")
        serve.LOGF_PERSIST = serve.LOGF + ".persist"   # distinct: slog writes BOTH
        serve.LOGS_DIR = self.tmp
        serve.STAYDOWN = os.path.join(self.tmp, "llm_engine_staydown")
        serve.ADMIN_TOKEN = "harness-admin-token"
        if keepalive_s is not None:
            serve.KEEPALIVE_S = keepalive_s
        # fresh module state
        serve.ST.ready = False; serve.ST.busy = False; serve.ST.cancel = False
        serve.ST.dirty = False; serve.ST.fed = []; serve.ST.convo_id = None
        serve.ST.cycles_since_rebuild = 0; serve.ST.rebuild_fails = 0
        serve.ST.rpc = None; serve.ST.pos_cache = 0; serve.ST.cur_cache = 0
        serve.ST.last_keepalive_ok = time.time(); serve.ST.step_beat = time.time()
        serve.ST.active_conn = None     # R3-21 scoping state (cross-boot)
        serve.ST.cancel_armed_by = None  # L7 abort-safety state (cross-boot)
        serve.ST._pf_stream = None
        serve.ST.inline_status = lambda req_id: {"id": req_id, "ok": True, "result": {}}
        serve.ST.cancel_hook = lambda conn: setattr(serve.ST, "cancel", True)
        serve.ST.unbind_hook = None
        # the engine triple
        self.E = engine if engine is not None else FakeEngine()
        self.G = object()
        self.sess = FakeSess(self.E)
        self.exits = []                          # (code, reason, t)
        serve.EXIT_FN = self._exit_fn
        self._thread = threading.Thread(
            target=self._boot, args=(batch,), daemon=True, name="serve-daemon")
        self.boot_failed = False
        self._thread.start()
        try:
            self._wait_boot(timeout=boot_timeout)
        except RuntimeError as e:
            # a watched-boot wedge exits through the watchdog instead of
            # finishing boot (R3-04b tests expect exactly that)
            if not self.exits:
                raise
            self.boot_failed = True

    def _exit_fn(self, code, reason):
        self.exits.append((code, reason, time.time()))
        raise HarnessExit(reason)

    def _boot(self, batch):
        r6h = FakeR6Host() if batch else None
        if batch:
            self.serve.BATCH_B = 2
        try:
            self.serve.run_daemon(self.E, self.G, self.sess, None, None,
                                  PARK_IDS, len(PARK_IDS), 20, "snap", self.ctxk,
                                  r6h=r6h)
        except HarnessExit:
            pass

    def _wait_boot(self, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(self.sock_path):
                try:
                    st = self.status()
                    if st.get("ready"):
                        return
                except Exception:
                    pass
            time.sleep(0.02)
        raise RuntimeError("serve harness failed to boot; logs: " + self.logs()[-800:])

    # ---- client ----
    def client(self, timeout=5.0):
        return Client(self.sock_path, timeout)

    def status(self, timeout=5.0):
        c = self.client(timeout)
        try:
            return c.rpc("status")["result"]
        finally:
            c.close()

    # ---- slog capture ----
    def logs(self):
        try:
            with open(self.serve.LOGF) as f:
                return f.read()
        except Exception:
            return ""

    def slog_records(self, **match):
        out = []
        for line in self.logs().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if all(rec.get(k) == v for k, v in match.items()):
                out.append(rec)
        return out

    def wait_slog(self, op, timeout=10, **match):
        match = dict(match, op=op)
        deadline = time.time() + timeout
        while time.time() < deadline:
            recs = self.slog_records(**match)
            if recs:
                return recs
            time.sleep(0.02)
        raise AssertionError(f"slog op={op} {match} not seen; tail: "
                             + self.logs().splitlines()[-12:].__str__())


def _r6_module(engine):
    m = types.ModuleType("r6_serve")
    m.R6Scheduler = FakeR6
    return m


def _gcycle_module():
    m = types.ModuleType("gcycle")
    m.GCycleEngine = FakeGCycle
    return m


class Client:
    """Line-JSON client for the real listener socket."""
    def __init__(self, path, timeout=5.0):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(timeout)
        self.s.connect(path)
        self.buf = b""
    def send(self, obj):
        self.s.sendall((json.dumps(obj) + "\n").encode())
    def send_raw(self, b):
        self.s.sendall(b)
    def recv(self):
        while b"\n" not in self.buf:
            ch = self.s.recv(65536)
            if not ch:
                raise EOFError("engine closed")
            self.buf += ch
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)
    def rpc(self, method, params=None, id_=1, timeout=None):
        if timeout is not None:
            self.s.settimeout(timeout)
        self.send({"id": id_, "method": method, "params": params or {}})
        while True:
            r = self.recv()
            if r.get("id") == id_ and "event" not in r:
                return r
    def events(self, id_, timeout=5.0):
        """Collect event frames for id_ until a terminal; returns list."""
        out = []
        self.s.settimeout(timeout)
        while True:
            r = self.recv()
            if r.get("id") == id_ and "event" in r:
                out.append(r)
                if r["event"] in ("done", "cancelled"):
                    return out
            elif r.get("id") == id_ and not r.get("ok", True):
                out.append(r)
                return out
    def close(self):
        try: self.s.close()
        except Exception: pass
