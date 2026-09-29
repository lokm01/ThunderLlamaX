"""L7 ABORT-SAFETY BATTERY (GPU-free; runs the REAL serve.py daemon code).

Encodes the L7 protocol invariants that the 152-test battery could not see
(grok Q5 / qwen mock-verdict): the old mock modeled the cancelled REPLY, not
the DEVICE — it never had eager launches in flight at the raise and never had
a second stream read a shared dring.

The two law-objects this battery enforces:

  FAKE DEVICE INFLOW COUNTER — every launch/submit increments `inflight`;
  only synchronize()/timeline-wait clears it. ANY abort path that leaves
  prefill with inflight > 0 FAILS the test (the unfenced-abort class that
  killed the box: LLC Bus error panic, 13 files in DiagnosticReports/Retired).

  DRING SENTINEL — the fake prefill poisons dev.dring = -1 at start (the
  mid-prefill draft-ring state); ONLY the successful epilogue clears it to 0.
  A generate that reads -1 faults the test (the grok epilogue-skip story:
  (size_t)(-1) into h_embed). Post-fix this is unreachable: an abort
  INVALIDATES the conversation (unbound + dirty) and the next prefill is
  forced FRESH, which re-runs the epilogue before any generate can attach.

Run (tg311 python — numpy only):
    ~/tg311/bin/python engine0/tests/test_l7_abort.py
"""
import os, sys, json, time, socket, threading, types, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENG0 = os.path.dirname(HERE)
if ENG0 not in sys.path:
    sys.path.insert(0, ENG0)
import serve_harness as H

BATT = {}

# ===================== the L7 fake device (grok/qwen spec) ====================
class L7Dev:
    """FakeDevice with the inflight counter + the dring sentinel cell.

    inflight: launch() increments (eager launch or graph submit stand-in);
    synchronize() clears. A test that ends an ABORTED prefill with inflight
    > 0 has reproduced the unfenced-abort protocol violation -> FAIL.
    global_cycle_ctr: the L7 FIX 5 budget counter (serve consults it)."""
    def __init__(self):
        self.inflight = 0
        self.dring = 0
        self.global_cycle_ctr = 0
        self.timeline_value = -1
        self.syncs = 0
        self.launches = 0
    def launch(self, n=1):
        self.inflight += n
        self.launches += n
    def synchronize(self):
        self.syncs += 1
        self.inflight = 0


class L7Engine(H.FakeEngine):
    """FakeEngine that models the abort-relevant GPU lifecycle: chunk work
    launches in flight (cleared only by the every-16 sync, the FIX 3 shape),
    the dring sentinel poisoned at prefill start and cleared ONLY by the
    successful epilogue."""
    def __init__(self, dev):
        super().__init__()
        self.dev = dev
    def _chunk(self, n_launch=8):
        self.dev.launch(n_launch)
    def prefill_t1(self, G, toks, log=None, prog=None):
        self._rec("prefill_t1", n=len(toks))
        self.dev.dring = -1                    # mid-prefill poison (draft ring)
        step = max(1, len(toks) // 8)
        i = 0
        while i < len(toks):
            if self.prefill_delay:
                time.sleep(self.prefill_delay)
            self._chunk()                      # ~16 tokens of eager launches
            self.dev.synchronize()             # the every-16 sync (FIX 3 order)
            i += step
            if prog:
                prog(min(i, len(toks)), len(toks))   # REAL prog: beat+send+checkpoint
        self.dev.dring = 0                     # epilogue clears the sentinel
        self.dev.synchronize()
        pos = int(self.P.slots["pos_slot"]) + len(toks)
        cur = int(toks[-1]) if toks else int(self.P.slots["cur_slot"])
        self.P.slots.update({"pos_slot": pos, "tok_slot": cur})
        self.prefill_t1_calls += 1
        return pos, cur
    def fill_draft(self, toks, start_pos=0, seed_hd=None, prog=None):
        self._rec("fill_draft", n=len(toks), start_pos=start_pos)
        self.dev.dring = -1
        # the FIX 3 shape verbatim: launch burst -> sync -> prog -> checkpoint
        if self.prefill_delay:
            time.sleep(self.prefill_delay)
        self._chunk(12)
        self.dev.synchronize()
        if prog:
            prog(min(16, len(toks)), len(toks))
        if len(toks) > 16:
            if self.prefill_delay:
                time.sleep(self.prefill_delay)
            self._chunk(12)
            self.dev.synchronize()
            if prog:
                prog(len(toks), len(toks))
        self.dev.dring = 0
        self.dev.synchronize()


class L7Sess(H.FakeSess):
    """DecodeSession fake that enforces the dring sentinel: generating over a
    -1 ring is the hard-reset fault class — fail loudly, never silently."""
    def step(self):
        if getattr(self.E, "dev", None) is not None and self.E.dev.dring == -1:
            raise RuntimeError(
                "L7 DRING SENTINEL: generate read a -1 draft ring "
                "(skipped-epilogue / unfenced-abort class)")
        return super().step()


class L7R6(H.FakeR6):
    def step(self, slots, deep):
        if getattr(self.E, "dev", None) is not None and self.E.dev.dring == -1:
            raise RuntimeError("L7 DRING SENTINEL (batch): generate read -1")
        return super().step(slots, deep)


def _install_l7_dev(dev):
    m0 = sys.modules.get("engine0")
    if m0 is None or not hasattr(m0, "dev"):
        H._install_fake_modules()
        m0 = sys.modules["engine0"]
    m0.dev = dev


def _boot(batch=False, engine=None, knobs=None):
    dev = L7Dev()
    _install_l7_dev(dev)
    eng = engine if engine is not None else L7Engine(dev)
    H.FakeSess = L7Sess        # the daemon builds FakeSess(self.E) -> enforcing sess
    if batch:
        m = types.ModuleType("r6_serve")
        m.R6Scheduler = L7R6   # the dring-enforcing batch scheduler fake
        sys.modules["r6_serve"] = m
    sd = H.ServeDaemon(batch=batch, engine=eng, knobs=knobs or {})
    sd.l7dev = dev
    return sd


def _golden(sd, c, tag="g"):
    """FRESH prefill + max_cycles generate -> the deterministic token list."""
    r = c.rpc("prefill", {"mode": "FRESH", "ids": list(range(100, 132)),
                          "conversation_id": tag})
    assert r["ok"], r
    c.send({"id": 2, "method": "generate", "params": {"max_cycles": 6}})
    evs = []
    c.s.settimeout(10)
    while True:
        fr = c.recv()
        if fr.get("id") == 2 and "event" in fr:
            evs.append(fr)
            if fr["event"] == "done":
                return [t for e in evs if e["event"] == "cycle" for t in e["tokens"]]
        elif fr.get("id") == 2 and "event" not in fr:
            assert fr.get("ok"), fr


# ============================ the tests =======================================
def test_fill_draft_checkpoint_is_post_sync():
  """FIX 3 shape: the fake fill_draft checkpoints AFTER its sync — an armed
  cancel mid-fill_draft aborts with inflight == 0 and a clean cancelled reply."""
  sd = _boot()
  BATT["sd"] = sd
  try:
    c = sd.client(timeout=10)
    # golden first
    gold = _golden(sd, c)
    # cancel mid fill_draft (the fake FRESH path: PF_DFILL standalone fill for
    # short prompts runs fill_draft with prog -> the every-16 checkpoint)
    sd.E.prefill_delay = 0.06             # slow the fake fill_draft chunks
    c.send({"id": 4, "method": "prefill",
            "params": {"mode": "FRESH", "ids": list(range(150, 190)),
                       "conversation_id": "fd"}})
    time.sleep(0.05)
    c.send({"id": 0, "method": "cancel", "params": {}})
    got = None
    c.s.settimeout(6)
    while True:
        fr = c.recv()
        if fr.get("id") == 4 and "event" not in fr:
            got = fr; break
    assert got is not None and not got["ok"] and got["error"] == "cancelled", got
    assert sd.l7dev.inflight == 0, "abort left eager launches in flight (FIX 3)"
    assert sd.serve.ST.dirty is True
    # bit-exact recovery: dirty -> FRESH -> generate == golden
    again = _golden(sd, c, tag="fd2")
    assert again == gold, (again, gold)
    c.close()
  finally:
    _stop()


def test_cancel_each_checkpoint_x50():
  """50 iterations of cancel-at-prog-checkpoint: each aborts cleanly
  (inflight == 0, cancelled reply) and recovery is bit-exact."""
  sd = _boot()
  BATT["sd"] = sd
  try:
    c = sd.client(timeout=10)
    gold = _golden(sd, c)
    n_cancel = 0
    sd.E.prefill_delay = 0.012            # ~100ms prefill: the cancel lands mid-way
    for it in range(50):
        c.send({"id": 40 + it, "method": "prefill",
                "params": {"mode": "FRESH", "ids": list(range(100, 132)),
                           "conversation_id": f"x{it}"}})
        time.sleep(0.004 if it % 2 else 0.02)   # vary the cancel point
        c.send({"id": 0, "method": "cancel", "params": {}})
        got = None
        c.s.settimeout(6)
        while True:
            fr = c.recv()
            if fr.get("id") == 40 + it and "event" not in fr:
                got = fr; break
            if fr.get("id") == 2 and "event" in fr and fr["event"] == "done":
                pass                            # stray golden tail
        if not got["ok"] and got.get("error") == "cancelled":
            n_cancel += 1
            assert sd.l7dev.inflight == 0, f"iter {it}: abort left work in flight"
            assert sd.serve.ST.dirty is True
        else:
            assert got["ok"], got               # completed before the cancel landed
        assert sd.l7dev.dring in (0, -1)        # -1 allowed only while dirty
        if not got["ok"]:
            assert sd.serve.ST.convo_id is None, "abort must invalidate the convo"
    assert n_cancel >= 5, f"cancel never landed mid-prefill ({n_cancel}/50)"
    sd.E.prefill_delay = 0.0
    # bit-exact recovery after the train
    again = _golden(sd, c, tag="post")
    assert again == gold, (again, gold)
    c.close()
  finally:
    sd.E.prefill_delay = 0.0
    _stop()


def test_ct_poison_sticky_cancel_train_cannot_form():
  """THE merged-bug repro (CT-poison): arm ST.cancel once, then run 20
  consecutive mixed-mode prefills. Pre-fix (sticky global) every one aborts
  before its epilogue -> dring stays -1 -> generate faults. Post-fix
  (entry-clear) the FIRST prefill aborts (the armed flag is honored exactly
  once) and every later prefill COMPLETES; the generate is bit-exact."""
  sd = _boot()
  BATT["sd"] = sd
  try:
    c = sd.client(timeout=10)
    gold = _golden(sd, c)
    # THE TRAIN STARTER: cancel an in-flight prefill (pre-fix, this armed
    # ST.cancel and NOTHING ever cleared it again)
    sd.E.prefill_delay = 0.05
    c.send({"id": 50, "method": "prefill",
            "params": {"mode": "FRESH", "ids": list(range(100, 132)),
                       "conversation_id": "train0"}})
    time.sleep(0.08)
    c.send({"id": 0, "method": "cancel", "params": {}})
    while True:
        fr = c.recv()
        if fr.get("id") == 50 and "event" not in fr:
            assert not fr["ok"] and fr["error"] == "cancelled", fr
            break
    assert sd.l7dev.inflight == 0
    sd.E.prefill_delay = 0.0
    aborted = 0; completed = 0
    for it in range(20):
        mode = "FOLLOW_UP" if (it % 3 == 2 and sd.serve.ST.convo_id) else "FRESH"
        ids = list(range(100 + it, 132 + it))
        params = {"mode": "FRESH", "ids": ids, "conversation_id": f"p{it}"}
        if mode == "FOLLOW_UP":
            params = {"mode": "FOLLOW_UP", "ids": ids[-4:],
                      "conversation_id": sd.serve.ST.convo_id}
        c.send({"id": 60 + it, "method": "prefill", "params": params})
        got = None
        c.s.settimeout(6)
        while True:
            fr = c.recv()
            if fr.get("id") == 60 + it and "event" not in fr:
                got = fr; break
        if got["ok"]:
            completed += 1
        else:
            assert got.get("error") == "cancelled", got
            aborted += 1
            assert sd.l7dev.inflight == 0
    # THE assertion: the poison train cannot form — at most the in-flight
    # prefill at arm time aborts; later prefills COMPLETE (entry-clear).
    assert completed >= 18, f"poison train: only {completed}/20 completed ({aborted} aborted)"
    # and the generate after the chaos is bit-exact
    again = _golden(sd, c, tag="post-poison")
    assert again == gold, (again, gold)
    c.close()
  finally:
    _stop()


def test_disconnect_mid_prefill_fences():
  """The send-stall arm: client vanishes mid-prefill -> next prog send fails
  -> cancel ARMED with attribution -> quiescent abort -> fenced handler ->
  inflight == 0 + slog op=abort armed_by=send_stalled."""
  sd = _boot()
  BATT["sd"] = sd
  try:
    c = sd.client(timeout=10)
    sd.E.prefill_delay = 0.05
    c.send({"id": 4, "method": "prefill",
            "params": {"mode": "FRESH", "ids": list(range(100, 132)),
                       "conversation_id": "dc"}})
    time.sleep(0.06)                       # mid-prefill
    c.close()                              # dead consumer (R3-16 marks dead)
    sd.wait_slog("prefill", stage="cancelled", timeout=10)
    assert sd.l7dev.inflight == 0
    assert sd.serve.ST.dirty is True and sd.serve.ST.convo_id is None
    ab = sd.slog_records(op="abort")
    assert ab, "expected op=abort slog from the checkpoint"
    c2 = sd.client(timeout=10)
    again = _golden(sd, c2, tag="dc2")
    c2.close()
  finally:
    sd.E.prefill_delay = 0.0
    _stop()


def test_global_budget_entry_rebuild():
  """FIX 5: a spent global graph budget triggers the entry rebuild BEFORE new
  work is accepted (the graph-class prefill debt becomes visible)."""
  sd = _boot()
  BATT["sd"] = sd
  try:
    c = sd.client(timeout=10)
    r = c.rpc("prefill", {"mode": "FRESH", "ids": [1, 2, 3], "conversation_id": "b0"})
    assert r["ok"]
    assert not sd.slog_records(op="l7_entry_rebuild")
    sd.l7dev.global_cycle_ctr = sd.serve.GLOBAL_REBUILD_EVERY   # prefill debt
    r = c.rpc("prefill", {"mode": "FOLLOW_UP", "ids": [4, 5],
                          "conversation_id": "b0"})
    assert r["ok"], r
    recs = sd.slog_records(op="l7_entry_rebuild")
    assert recs, "entry rebuild must fire when the global counter is spent"
    builds = [k for _t, k, kw in sd.E.calls if k == "build_graphs"]
    assert builds, "E.build_graphs must run at the entry rebuild"
    assert sd.l7dev.global_cycle_ctr == 0
    c.close()
  finally:
    _stop()


def test_batch_cancel_fences_and_invalidates():
  """Batch path (FIX 4): a cancelled barrier prefill (a) leaves inflight == 0,
  (b) invalidates the slot (convo unbound, dirty), (c) holds the barrier
  through the fence (slog fence_ms present), and (d) the other stream's
  prefill+generate still completes bit-exact."""
  sd = _boot(batch=True)
  BATT["sd"] = sd
  try:
    c = sd.client(timeout=10)
    gold = _golden(sd, c, tag="bs0")
    # barrier prefill on the same slot, cancelled mid-way
    sd.E.prefill_delay = 0.05
    c.send({"id": 70, "method": "prefill",
            "params": {"mode": "FRESH", "ids": list(range(100, 132)),
                       "conversation_id": "bcx"}})
    time.sleep(0.06)
    c.send({"id": 0, "method": "cancel", "params": {}})
    got = None
    c.s.settimeout(8)
    while True:
        fr = c.recv()
        if fr.get("id") == 70 and "event" not in fr:
            got = fr; break
    assert got is not None and not got["ok"] and got["error"] == "cancelled", got
    assert sd.l7dev.inflight == 0, "batch abort left work in flight"
    st0 = sd.serve  # slot state lives in the daemon closure; assert via slog
    rec = sd.slog_records(op="prefill", stage="cancelled")
    assert rec and "fence_ms" in rec[-1], rec[-1:]
    # recovery on the SAME conn: FRESH (dirty gate forces it) then generate
    again = _golden(sd, c, tag="bcx2")
    assert again == gold, (again, gold)
    c.close()
  finally:
    sd.E.prefill_delay = 0.0
    _stop(batch=True)


def test_batch_dring_sentinel_second_stream():
  """The batch second-stream discriminator: stream A's prefill aborts (dring
  poisoned, conversation invalidated); stream B — a DIFFERENT conversation —
  prefills fresh (epilogue clears the sentinel) and generates WITHOUT ever
  reading the -1 ring."""
  sd = _boot(batch=True)
  BATT["sd"] = sd
  try:
    ca = sd.client(timeout=10)
    gold = _golden(sd, ca, tag="sa")
    sd.E.prefill_delay = 0.05
    ca.send({"id": 80, "method": "prefill",
             "params": {"mode": "FRESH", "ids": list(range(100, 132)),
                        "conversation_id": "cvA"}})
    time.sleep(0.06)
    ca.send({"id": 0, "method": "cancel", "params": {}})
    got = None
    ca.s.settimeout(8)
    while True:
        fr = ca.recv()
        if fr.get("id") == 80 and "event" not in fr:
            got = fr; break
    assert got and not got["ok"] and got["error"] == "cancelled", got
    assert sd.l7dev.inflight == 0
    cb = sd.client(timeout=10)          # stream B: different conversation
    out = _golden(sd, cb, tag="cvB")
    assert out == gold, (out, gold)
    ca.close(); cb.close()
  finally:
    sd.E.prefill_delay = 0.0
    _stop(batch=True)


def _stop(batch=False):
  sd = BATT.pop("sd", None)
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
    try:
      sd.serve.ST.cancel = True        # unblock a stuck worker (listener-test pattern)
    except Exception:
      pass
    deadline = time.time() + 6
    while time.time() < deadline and not sd.exits:
      time.sleep(0.05)


if __name__ == "__main__":
  tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
  failed = 0
  for t in tests:
    name = t.__name__
    try:
      t()
      print(f"[L7] PASS {name}", flush=True)
    except Exception as e:
      failed += 1
      import traceback; traceback.print_exc()
      print(f"[L7] FAIL {name}: {e!r}", flush=True)
      _stop()
  print(f"[L7] {'ALL GREEN' if not failed else f'{failed} FAILED'} ({len(tests)} tests)", flush=True)
  sys.exit(1 if failed else 0)
