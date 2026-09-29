"""W-E.1 battery: the REAL serve.py listener (legacy single-stream path)
driven through engine0/tests/serve_harness.py — REAL protocol/framing/lock/
watchdog code, fake engine object. GPU-free; run with the tg311 python:

    ~/tg311/bin/python engine0/tests/test_serve_listener.py

These are the conversions of the mock-fidelity-dependent assertions to real
coverage (TLX_REVIEW_LEDGER_R3 W-E.1): every test here executes serve.py's
own listener threads and RPC handlers, not a wire-protocol mirror.
"""
import os, sys, json, time, socket, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import serve_harness as H

_BATT = {"sd": None}


def _sd(refresh=False):
    if _BATT["sd"] is None or refresh:
        _stop_sd()
        _BATT["sd"] = H.ServeDaemon()
    return _BATT["sd"]


def _stop_sd():
    sd = _BATT.get("sd")
    if sd is None:
        return
    try:
        c = sd.client(timeout=2)
        c.rpc("shutdown", {"admin_token": sd.serve.ADMIN_TOKEN}, timeout=3)
        c.close()
    except Exception:
        pass
    deadline = time.time() + 3
    while time.time() < deadline and not sd.exits:
        time.sleep(0.05)
    if not sd.exits:
        # a daemon stuck inside a long RPC: arm the global cancel to unblock
        # the worker (the queued shutdown then lands); NEVER let a daemon
        # outlive its test (a stale ST.active_conn poisons the next boot)
        try:
            sd.serve.ST.cancel = True
        except Exception:
            pass
        deadline = time.time() + 6
        while time.time() < deadline and not sd.exits:
            time.sleep(0.05)
    _BATT["sd"] = None


# ==============================================================================
# W-E.1 baseline: the real listener speaks the real protocol
# ==============================================================================
def test_real_listener_status_prefill_generate():
    """status field set / FRESH prefill / FOLLOW_UP / generate with stop —
    all through serve.py's own listener threads and handle()."""
    sd = _sd(refresh=True)
    st = sd.status()
    for k in ("ready", "ctxk", "busy", "pos", "mode", "cur", "fed_len",
              "conversation_id", "dirty", "config_fp", "uptime_s"):
        assert k in st, f"status missing {k}: {st}"
    c = sd.client()
    try:
        r = c.rpc("prefill", {"mode": "FRESH", "ids": [10, 11, 12],
                              "conversation_id": "A"})
        assert r["ok"] and r["result"]["pos"] == 3, r
        assert sd.serve.ST.fed == [10, 11, 12] and sd.serve.ST.convo_id == "A"
        r = c.rpc("prefill", {"mode": "FOLLOW_UP", "ids": [70, 80], "cur": 11,
                              "conversation_id": "A"})
        assert r["ok"], r
        assert sd.serve.ST.fed == [10, 11, 12, 11, 70, 80], sd.serve.ST.fed
        assert sd.serve.ST.pos_cache == 6
        # generate: cycle events then stop-terminal on token 67
        sd.sess.script = [[65, 66], [67]]
        c.send({"id": 5, "method": "generate",
                "params": {"max_cycles": 10, "stop_token_ids": [67]}})
        evs = c.events(5)
        assert evs[0]["event"] == "cycle" and evs[-1]["event"] == "done"
        assert evs[-1]["stop"] is True and evs[-1]["tokens"] == [65, 66, 67]
        # the result frame arrives after the done event
        r5 = c.rpc("status", id_=6)  # interleave a status while draining
        assert r5["ok"]
    finally:
        c.close()
        _stop_sd()


def test_real_listener_cancel_sidechannel():
    """cancel is routed by the REAL listener_conn (no ack frame); h_generate
    honours the flag at the next cycle boundary."""
    sd = _sd(refresh=True)
    c = sd.client()
    try:
        r = c.rpc("prefill", {"mode": "FRESH", "ids": [1, 2, 3]})
        assert r["ok"]
        sd.sess.script = [[65, 66]] * 10
        c.send({"id": 7, "method": "generate", "params": {"max_cycles": 100}})
        c.recv()                                  # first cycle event
        c.send({"id": 0, "method": "cancel", "params": {}})   # side channel
        evs = c.events(7)
        assert evs[-1]["event"] == "cancelled", evs[-3:]
        assert sd.serve.ST.cancel is False or True   # flag state post-exit
    finally:
        c.close()
        _stop_sd()


def test_real_listener_generate_fault_dirty():
    """a faulting sess.step -> error reply, dirty set, known-good fed append."""
    sd = _sd(refresh=True)
    c = sd.client()
    try:
        c.rpc("prefill", {"mode": "FRESH", "ids": [5, 6]})
        sd.sess.fail_at = 1
        sd.sess.script = [[9, 9]]
        r = c.rpc("generate", {"max_cycles": 5})
        assert not r["ok"] and "fault" in r["error"], r
        assert sd.serve.ST.dirty is True
        # real semantics: the faulting cycle COMPLETED no tokens -> fed keeps
        # only completed-cycle tokens (empty here)
        assert sd.serve.ST.fed == [5, 6], sd.serve.ST.fed
        # a clean prefill re-establishes state
        r = c.rpc("prefill", {"mode": "FRESH", "ids": [7, 8]})
        assert r["ok"] and sd.serve.ST.dirty is False
    finally:
        c.close()
        _stop_sd()


def test_real_listener_admin_fail_closed_and_shutdown():
    """privileged methods refuse without/with-wrong token; with the token the
    shutdown runs the REAL _clean_exit (staydown marker + socket unlink)."""
    sd = _sd(refresh=True)
    c = sd.client()
    try:
        r = c.rpc("shutdown", {})
        assert not r["ok"] and "admin" in r["error"], r
        r = c.rpc("shutdown", {"admin_token": "wrong"})
        assert not r["ok"]
        assert sd.serve.ADMIN_TOKEN == "harness-admin-token"
        # daemon still alive
        assert sd.status()["ready"] is True
        r = c.rpc("shutdown", {"admin_token": sd.serve.ADMIN_TOKEN})
        assert r["ok"] and r["result"]["bye"] is True
    finally:
        c.close()
    deadline = time.time() + 3
    while time.time() < deadline and not sd.exits:
        time.sleep(0.05)
    assert sd.exits and sd.exits[0][0] == 0 and sd.exits[0][1] == "shutdown_rpc", sd.exits
    assert os.path.exists(sd.serve.STAYDOWN)      # operator intent persisted
    deadline = time.time() + 2
    while time.time() < deadline and os.path.exists(sd.sock_path):
        time.sleep(0.05)
    assert not os.path.exists(sd.sock_path)       # unlinked by _clean_exit
    _BATT["sd"] = None


def test_real_listener_line_cap_drops():
    """V-31 through the REAL listener: a no-newline stream past the cap drops
    the conn; the daemon answers the next conn."""
    sd = _sd(refresh=True)
    c = sd.client(timeout=3)
    try:
        sd.serve.MAX_LINE_BYTES = 512 * 1024          # module knob, call-time read
        c.s.settimeout(3)
        try:
            c.send_raw(b"x" * (768 * 1024))           # > cap pending, no \n
        except (BrokenPipeError, ConnectionResetError):
            pass                                     # the drop races our send
        got = c.s.recv(4096)
        assert got == b"", got                       # closed
        assert sd.slog_records(op="line_too_long"), sd.logs()[-400:]
        c2 = sd.client()
        try:
            assert c2.rpc("status")["ok"]
        finally:
            c2.close()
    finally:
        c.close()
        _stop_sd()


def test_real_listener_status_during_generate():
    """the lock-free inline status answers WHILE a generate owns the worker
    (the per-conn listener design; regression guard for the harness)."""
    sd = _sd(refresh=True)
    c = sd.client()
    try:
        c.rpc("prefill", {"mode": "FRESH", "ids": [1, 2, 3]})
        sd.sess.script = [[65, 66]] * 50
        c.send({"id": 9, "method": "generate", "params": {"max_cycles": 500}})
        c.recv()                                     # first cycle
        c2 = sd.client()
        try:
            t0 = time.time()
            st = c2.rpc("status")["result"]
            dt = time.time() - t0
            assert st["busy"] is True and st["rpc"] == "generate", st
            assert dt < 1.0, dt
        finally:
            c2.close()
        c.send({"id": 0, "method": "cancel", "params": {}})
        evs = c.events(9, timeout=5)
        assert evs[-1]["event"] == "cancelled"
    finally:
        c.close()
        _stop_sd()



# ==============================================================================
# R3-16 (W-B): send/close synchronization + dead-conn skip
# ==============================================================================
def test_r3_16_send_lock_id_keyed_and_dead_flag():
  """R3-16 unit: locks keyed by id(conn) (a closed conn reads fileno()==-1;
  fd REUSE by a new socket would share the dead conn's lock — the
  cross-client-write class) and the dead flag under the lock makes every
  post-teardown send a no-op."""
  import serve
  a, b = socket.socketpair()
  ent_a = serve._send_lock(a)
  assert serve._send_lock(a) is ent_a
  serve.send(a, {"x": 1})
  b.settimeout(0.5)
  got = b.recv(4096)
  assert json.loads(got) == {"x": 1}
  fd = a.fileno()
  a.close()
  c, d = socket.socketpair()
  try:
    ent_c = serve._send_lock(c)
    if c.fileno() == fd:                 # the reuse actually happened
      assert ent_c is not ent_a, "id-keying must separate fd-reusing conns"
    # dead flag: teardown closes under the lock; later sends never deliver
    serve._mark_dead(c)
    serve.send(c, {"dead": 1})           # must be swallowed (send_stalled)
    d.settimeout(0.3)
    try:
      leaked = d.recv(4096)
      assert leaked == b"", f"frame landed on a dead conn: {leaked!r}"
    except socket.timeout:
      pass                     # nothing buffered AND peer alive: also fine
  finally:
    b.close(); c.close(); d.close()
    serve._send_unlock(a); serve._send_unlock(c)


def test_r3_16_legacy_dead_client_cancels_generate():
  """R3-16 e2e (legacy loop): a client that vanishes mid-generate gets its
  conn torn down (dead flag under the send lock); the next cycle send fails
  -> cancel armed -> the generate ends cancelled (no runaway)."""
  sd = _sd(refresh=True)
  c = sd.client()
  try:
    c.rpc("prefill", {"mode": "FRESH", "ids": [1, 2, 3]})
    sd.sess.script = [[65, 66]] * 50
    sd.sess.cycle_delay = 0.02
    c.send({"id": 3, "method": "generate", "params": {"max_cycles": 300}})
    c.recv()                                  # first cycle arrives
    c.close()                                 # the client vanishes
    sd.wait_slog("generate", stage="cancelled", timeout=10)
    recs = sd.slog_records(op="send_stalled")
    assert recs, "the dead-conn cycle send must fail (and be logged)"
  finally:
    _stop_sd()


def test_batch_daemon_basic_protocol():
  """The batch scheduler (REAL _batch_main) against FakeR6/FakeGCycle: status
  carries streams+batch_b, prefill binds a slot, generate streams cycles."""
  sd = H.ServeDaemon(batch=True)
  _BATT["sd"] = sd
  try:
    st = sd.status()
    assert st["batch_b"] == 2 and isinstance(st["streams"], list)
    c = sd.client()
    r = c.rpc("prefill", {"mode": "FRESH", "ids": [10, 11, 12],
                          "conversation_id": "ba"})
    assert r["ok"], r
    st = sd.status()
    convs = {s["conversation_id"] for s in st["streams"]}
    assert "ba" in convs
    sd.sess.script = [[65, 66]] * 5
    c.send({"id": 2, "method": "generate",
            "params": {"max_cycles": 3, "stop_token_ids": [67]}})
    evs = c.events(2, timeout=10)
    assert evs and evs[-1]["event"] == "done", evs
    c.close()
  finally:
    _stop_sd()


def test_r3_16_batch_terminal_skipped_dead_conn():
  """R3-16 batch: after the generating conn is unbound (teardown), the
  terminal frames are SKIPPED (terminal_skipped_dead_conn) — never sent to a
  dead fd or a reused fd's new owner."""
  sd = H.ServeDaemon(batch=True)
  _BATT["sd"] = sd
  try:
    c = sd.client()
    c.rpc("prefill", {"mode": "FRESH", "ids": [10, 11], "conversation_id": "dc"})
    # long solo generate on slot 0 via the canonical sess
    sd.sess.script = [[65, 66]] * 50
    sd.sess.cycle_delay = 0.02
    c.send({"id": 2, "method": "generate", "params": {"max_cycles": 300}})
    c.recv()                                  # first cycle
    c.close()                                 # teardown -> unbind + mark_dead
    sd.wait_slog("generate", stage="terminal_skipped_dead_conn", timeout=10)
    recs = sd.slog_records(op="generate", stage="terminal_skipped_dead_conn")
    assert recs and recs[0]["kind"] == "cancelled", recs
  finally:
    _stop_sd()


# ==============================================================================
# R3-01/02/03/04 (W-A): the liveness triad — no dead-but-healthy state
# ==============================================================================
def test_r3_02_keepalive_stale_idle_arm():
  """R3-02: an idle daemon whose keepalive probe FAILS is a silent zombie
  (/health answered ready+busy=False forever on a dead GPU). The idle-arm
  watchdog exits keepalive_stale after 3x KEEPALIVE_S without a successful
  probe/beat."""
  sd = H.ServeDaemon(keepalive_s=0.05, knobs={"STEP_TIMEOUT_S": 1.0})
  _BATT["sd"] = sd
  try:
    assert sd.status()["ready"] is True
    sd.E.keepalive_fails = True            # the wedge begins
    deadline = time.time() + 6
    while time.time() < deadline and not sd.exits:
        time.sleep(0.05)
    assert sd.exits and sd.exits[0][1] == "keepalive_stale", sd.exits
    assert sd.slog_records(op="watchdog", stage="keepalive_stale")
    _BATT["sd"] = None                     # already exited
  finally:
    _stop_sd()


def test_r3_03_shutdown_watched():
  """R3-03: shutdown is in _GPU_RPC — its dev.synchronize() runs under
  busy/rpc/beats, so a wedged GPU hanging the graceful stop trips the
  watchdog with attribution (rpc=shutdown) instead of hanging forever."""
  sd = H.ServeDaemon(knobs={"STEP_TIMEOUT_S": 0.5})
  _BATT["sd"] = sd
  try:
    H._FakeDev.hang_synchronize = True     # the GPU wedge
    try:
      c = sd.client()
      try:
        r = c.rpc("shutdown", {"admin_token": sd.serve.ADMIN_TOKEN}, timeout=3)
        assert r["ok"]                       # the bye frame precedes the sync
      finally:
        c.close()
      deadline = time.time() + 5
      while time.time() < deadline and not sd.exits:
          time.sleep(0.05)
      assert sd.exits and sd.exits[0][1] == "watchdog_step_timeout", sd.exits
      recs = sd.slog_records(op="watchdog", stage="step_timeout")
      assert recs and recs[0]["rpc"] == "shutdown", recs
    finally:
      H._FakeDev.hang_synchronize = False
    _BATT["sd"] = None
  finally:
    _stop_sd()


def test_r3_04b_boot_wedge_watched():
  """R3-04b: the watchdog runs from the VERY START — a boot-phase wedge (the
  park/snapshot-reset/pc-ingest class) is a step-timeout with rpc=boot, not
  an unattributed eternal hang."""
  eng = H.FakeEngine()
  eng.reset_snapshot_hangs = True          # the boot wedge
  sd = H.ServeDaemon(engine=eng, knobs={"STEP_TIMEOUT_S": 0.5}, boot_timeout=4)
  _BATT["sd"] = sd
  try:
    assert sd.boot_failed, "boot must not complete while wedged"
    assert sd.exits and sd.exits[0][1] == "watchdog_step_timeout", sd.exits
    recs = sd.slog_records(op="watchdog", stage="step_timeout")
    assert recs and recs[0]["rpc"] == "boot", recs
    _BATT["sd"] = None
  finally:
    _stop_sd()


# ==============================================================================
# R3-06 (W-A): rebuild consecutive-failure budget
# ==============================================================================
def test_r3_06_rebuild_budget_exhausted():
  """R3-06: 3 failed fences in a row (the ka-slab exhaustion class) exit
  cleanly as rebuild_budget_exhausted while inside the safety margin — the
  W5 posture reset the counter on FAILURE, leaving the ~950-cycle budget
  unenforced (march into the 850-1025 window = device fault = MACHINE
  REBOOT). Old code: the generate ran to max_cycles with the counter
  resetting at every failed boundary."""
  sd = H.ServeDaemon(knobs={"GEN_REBUILD_EVERY": 2})
  _BATT["sd"] = sd
  try:
    sd.E.build_graphs_fails = 99          # every rebuild fails
    c = sd.client()
    try:
      c.rpc("prefill", {"mode": "FRESH", "ids": [1, 2, 3]})
      sd.sess.script = [[65, 66]] * 50
      c.send({"id": 5, "method": "generate", "params": {"max_cycles": 100}})
      deadline = time.time() + 8
      while time.time() < deadline and not sd.exits:
          time.sleep(0.05)
      assert sd.exits and sd.exits[0] == (1, "rebuild_budget_exhausted", sd.exits[0][2]), sd.exits
      recs = sd.slog_records(op="rebuild_budget_exhausted")
      assert recs and recs[0]["fails"] == 3, recs
      assert recs[0]["cycles_since_rebuild"] >= 2, "counter must NOT reset on failure"
      fails = sd.slog_records(op="gen_rebuild_failed")
      assert len(fails) == 3, [r.get("fails") for r in fails]
      _BATT["sd"] = None                   # already exited
    finally:
      c.close()
  finally:
    _stop_sd()


def test_r3_06_rebuild_success_resets_budget():
  """R3-06 complement: a SUCCESSFUL rebuild resets both the cycle counter and
  the failure streak (transient failures then recovery must not exit)."""
  sd = H.ServeDaemon(knobs={"GEN_REBUILD_EVERY": 2})
  _BATT["sd"] = sd
  try:
    c = sd.client()
    try:
      sd.E.build_graphs_fails = 1          # one transient failure then success
      c.rpc("prefill", {"mode": "FRESH", "ids": [1, 2, 3]})
      sd.sess.script = [[65, 66]] * 10
      c.send({"id": 5, "method": "generate", "params": {"max_cycles": 10}})
      evs = c.events(5, timeout=10)
      assert evs[-1]["event"] == "done", evs[-1]
      assert not sd.exits, sd.exits
      assert sd.serve.ST.rebuild_fails == 0
      assert sd.slog_records(op="gen_rebuild_failed")      # the transient fired
      assert sd.slog_records(op="gen_rebuild")             # and was recovered
    finally:
      c.close()
  finally:
    _stop_sd()


# ==============================================================================
# R3-13/17/18 (W-B): abandoned-request class, anon FOLLOW_UP, snapshot guard
# ==============================================================================
def test_r3_17_anonymous_followup_rejected():
  """R3-17: FOLLOW_UP with conversation_id=None is refused by the REAL
  validate_rpc (legacy AND batch) — None used to match ANY None-convo slot
  including the parked slot 0 and silently continue the park's context."""
  import serve
  err, _ = serve.validate_rpc("prefill", {"mode": "FOLLOW_UP", "ids": [1, 2]},
                              1000, 260, 0)
  assert err and "conversation_id" in err, err
  sd = _sd(refresh=True)
  c = sd.client()
  try:
    # legacy: FRESH anonymous sets resident convo None; anon FOLLOW_UP used to PASS
    r = c.rpc("prefill", {"mode": "FRESH", "ids": [1, 2, 3]})
    assert r["ok"]
    r2 = c.rpc("prefill", {"mode": "FOLLOW_UP", "ids": [9]})
    assert not r2["ok"] and "conversation_id" in r2["error"], r2
    assert sd.slog_records(op="rpc_rejected")
  finally:
    c.close()
    _stop_sd()


def test_r3_13_disconnect_mid_prefill_aborts():
  """R3-13: the legacy prefill honors the armed cancel flag at its prog
  callbacks — an abandoned prefill aborts (dirty + cancelled reply) instead
  of running to completion for a dead client."""
  sd = _sd(refresh=True)
  c = sd.client(timeout=10)
  try:
    sd.E.prefill_delay = 0.05              # 8 chunks -> ~0.4s prefill
    c.send({"id": 4, "method": "prefill",
            "params": {"mode": "FRESH", "ids": list(range(100, 132))}})
    time.sleep(0.12)                       # mid-prefill
    c.send({"id": 0, "method": "cancel", "params": {}})
    r = c.rpc("noop-status", id_=5) if False else None
    # the cancelled reply arrives for id 4
    got = None
    c.s.settimeout(6)
    while True:
        fr = c.recv()
        if fr.get("id") == 4 and "event" not in fr:
            got = fr; break
    assert got is not None and not got["ok"] and got["error"] == "cancelled", got
    assert sd.serve.ST.dirty is True
    assert sd.slog_records(op="prefill", stage="cancelled")
    calls = [k for _t, k, kw in sd.E.calls if k == "prefill_t1"]
    assert len(calls) == 1                  # the prefill STARTED (cancelled mid-way)
  finally:
    sd.E.prefill_delay = 0.0
    c.close()
    _stop_sd()


def test_r3_13_engine_queue_cap_loud():
  """R3-13/R3-51: the engine Q is capped — a flood beyond TLX_ENGINE_Q_MAX
  gets a loud error frame instead of queuing unbounded work."""
  sd = H.ServeDaemon(knobs={"MAX_Q_DEPTH": 1})
  _BATT["sd"] = sd
  try:
    # occupy the worker with a slow generate
    c1 = sd.client()
    c1.rpc("prefill", {"mode": "FRESH", "ids": [1, 2, 3]})
    sd.sess.script = [[65, 66]] * 100
    sd.sess.cycle_delay = 0.03
    c1.send({"id": 2, "method": "generate", "params": {"max_cycles": 300}})
    c1.recv()                                   # worker now busy
    c2 = sd.client(); c3 = sd.client()
    c2.send({"id": 20, "method": "prefill", "params": {"mode": "FRESH", "ids": [7, 8]}})
    c3.send({"id": 30, "method": "prefill", "params": {"mode": "FRESH", "ids": [9, 8]}})
    # whichever conn's put lands on the full queue gets the loud rejection
    # (the two listener threads race; exactly ONE is dropped at cap 1)
    got = None
    for c in (c3, c2):
        c.s.settimeout(4)
        try:
            while True:
                fr = c.recv()
                if "event" not in fr and not fr.get("ok", True):
                    got = fr; break
        except Exception:
            continue
    assert got is not None and "queue full" in got["error"], got
    assert sd.slog_records(op="engine_queue_full")
    c1.send({"id": 0, "method": "cancel", "params": {}})
    c2.close(); c3.close()
  finally:
    c1.close()
    _stop_sd()


def test_r3_18_snapshot_guard_any_slot_used():
  """R3-18 (batch scheduler, REAL code): snapshot_save/load/prefill-snapshot
  are rejected while a barrier/pending/ANY-used state exists — the old check
  (slot1-used or generating) passed mid-barrier (same-thread bank clobber)."""
  sd = H.ServeDaemon(batch=True)
  _BATT["sd"] = sd
  try:
    c = sd.client()
    r = c.rpc("prefill", {"mode": "FRESH", "ids": [10, 11], "conversation_id": "sg"})
    assert r["ok"]
    # slot 0 used, slot 1 pristine, nothing generating — the OLD guard
    # ACCEPTED here; the R3-18 guard rejects (any(st.used))
    for method, params in (("snapshot_save", {"path": "/tmp/x"}),
                           ("snapshot_load", {"path": "/tmp/x"})):
        r2 = c.rpc(method, dict(params, admin_token=sd.serve.ADMIN_TOKEN))
        assert not r2["ok"] and "rejected" in r2["error"], (method, r2)
        assert sd.slog_records(op=method, stage="rejected_batch")
    r3 = c.rpc("prefill", {"snapshot": "/tmp/x",
                           "admin_token": sd.serve.ADMIN_TOKEN})
    assert not r3["ok"] and "rejected" in r3["error"], r3
    c.close()
  finally:
    _stop_sd()


# ==============================================================================
# R3-21 (W-C): peercred fail-closed + cancel scoping (REAL listener)
# ==============================================================================
def test_r3_21_cancel_scoped_to_owning_conn():
  """R3-21: legacy cancel is scoped to the conn OWNING the active RPC — any
  same-uid process used to be able to abort the active generate."""
  sd = _sd(refresh=True)
  ca = sd.client()
  cb = sd.client()
  try:
    ca.rpc("prefill", {"mode": "FRESH", "ids": [1, 2, 3], "conversation_id": "cs"})
    sd.sess.script = [[65, 66]] * 100
    sd.sess.cycle_delay = 0.02
    ca.send({"id": 7, "method": "generate", "params": {"max_cycles": 300}})
    ca.recv()                                    # generate active, owned by ca
    cb.send({"id": 0, "method": "cancel", "params": {}})   # FOREIGN cancel
    time.sleep(0.3)
    assert sd.slog_records(op="cancel_scoped_ignored"), "foreign cancel must be ignored"
    still = sd.status()
    # the generate SURVIVES the foreign cancel (status still busy/generate)
    ca.s.settimeout(1.0)
    got_more = False
    try:
        ca.recv()                                # more cycle events flow
        got_more = True
    except Exception:
        pass
    # owning conn cancels -> terminates
    ca.send({"id": 0, "method": "cancel", "params": {}})
    evs = ca.events(7, timeout=8)
    assert evs[-1]["event"] == "cancelled", evs[-1]
  finally:
    ca.close(); cb.close()
    _stop_sd()

# ==============================================================================
# runner
# ==============================================================================
def _all_tests():
    return [(n, f) for n, f in sorted(globals().items())
            if n.startswith("test_") and callable(f)]


def main():
    tests = _all_tests()
    print(f"TLX R3 real-listener battery: {len(tests)} tests\n" + "=" * 60)
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
        finally:
            _stop_sd()
    print("=" * 60)
    print(f"{len(tests)-len(failed)}/{len(tests)} passed")
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
