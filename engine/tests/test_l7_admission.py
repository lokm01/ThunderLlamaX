"""L7.1 admission-hardening battery (GPU-free): the leaked-permit heal.

Post-L7 live finding: with the engine surviving the dead-consumer mix, an API
admission leak (the R3-10 class) surfaced — QSTATE active=1 + waiters parked
while the ENGINE sat idle; permanent 429s until API restart. The hardening:
(a) the guard belt (a cancelled _slot_guard releases), (b) bounded conv-lock
acquire, (c) the admission watchdog that heals ONLY when the engine is
demonstrably idle. This file pins (c) with a simulated leak against the real
mock-engine stack.

Run: /usr/bin/python3 engine0/tests/test_l7_admission.py  (fastapi python)
"""
import os, sys, time, asyncio, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ENG0 = os.path.dirname(HERE)
if ENG0 not in sys.path:
    sys.path.insert(0, ENG0)
    sys.path.insert(0, HERE)

import test_api_mocks as TAM
import mock_engine as ME

def test_watchdog_heals_leaked_permit():
  """Simulate the observed wedge (active=1 + a parked waiter, mock engine
  idle): the watchdog must release within ~HEAL_AFTER and admission must
  serve a request afterwards."""
  with TAM.MockCtx() as ctx:
    a = ctx.a
    a.ADMIT_WATCHDOG_S = 0.2
    a.ADMIT_HEAL_AFTER_S = 0.6
    async def drive():
      a.start_admission_watchdog()
      # the observed wedge, verbatim: a holder nobody owns + a parked waiter
      a.QSTATE.update({"active": 1, "permits": 1, "holder_rid": "leak-sim",
                       "holder_since": time.time()})
      waiter = asyncio.get_running_loop().create_future()
      a.QSTATE["wait"].append(waiter)
      t0 = time.time()
      # success oracle: the heal releases the leaked permit and the PROMOTER
      # hands it to the parked waiter (waiter.done() = the wedge cleared;
      # active stays 1 because the promoted waiter is now the legitimate
      # holder — exactly the production recovery semantics)
      while time.time() - t0 < 6 and not waiter.done():
        await asyncio.sleep(0.1)
      return waiter.done(), time.time() - t0
    healed, dt = asyncio.run(drive())
    assert healed, f"watchdog did not heal within {dt:.1f}s"
    assert dt < 4, f"heal too slow ({dt:.1f}s)"
    # clear the simulated-wedge residue (the promoted phantom waiter owned a
    # now-dead loop) before the recovery probe
    a.QSTATE.update({"active": 0, "wait": collections.deque(), "holder": None,
                     "permits": 1, "holder_rid": None, "holder_since": None})
    # admission recovered: a normal request completes against the mock engine
    async def probe():
      return await ME.asgi_request(a.app, "POST", "/v1/chat/completions",
                                   json_body=TAM.no_think(TAM.body("hi")))
    r = asyncio.run(probe())
    assert r.status == 200, (r.status, r.body[:200])

def test_watchdog_never_heals_when_engine_busy():
  """The discriminator: an active holder whose engine is BUSY (a real
  prefill/generate) must NEVER be healed — the watchdog keeps hands off."""
  with TAM.MockCtx(reply_tokens=mock_reply()) as ctx:
    a = ctx.a
    a.ADMIT_WATCHDOG_S = 0.2
    a.ADMIT_HEAL_AFTER_S = 0.6
    async def drive():
      a.start_admission_watchdog()
      # holder + waiter + the MOCK ENGINE REPORTING BUSY (the discriminator)
      a.QSTATE.update({"active": 1, "permits": 1, "holder_rid": "real-work",
                       "holder_since": time.time()})
      a.QSTATE["wait"].append(asyncio.get_running_loop().create_future())
      a._batt_busy_override = True     # see _mock_status_busy below
      await asyncio.sleep(1.6)         # > heal window
      return a.QSTATE["active"]
    active = asyncio.run(drive())
    assert active == 1, "watchdog healed a BUSY holder (false positive!)"
    a._batt_busy_override = False

def mock_reply():
  return ME.mock_encode("ok") + [ME.ID_IMEND]

# The mock engine reports busy via its cfg; simplest override: patch eng_status
_orig_eng_status = None
def _mock_status_busy():
  """Patch api.eng_status to report busy=True (the discriminator input)."""
  a = TAM.api()
  global _orig_eng_status
  if _orig_eng_status is None:
    _orig_eng_status = a.eng_status
  def fake(timeout=2.0):
    st = dict(_orig_eng_status(timeout))
    if getattr(a, "_batt_busy_override", False):
      st["busy"] = True; st["rpc"] = "generate"
    return st
  a.eng_status = fake

# install the busy patch before the tests run
_mock_status_busy()

if __name__ == "__main__":
  failed = 0
  for name, fn in sorted(globals().items()):
    if name.startswith("test_") and callable(fn):
      try:
        fn(); print(f"[L7.1] PASS {name}", flush=True)
      except Exception as e:
        failed += 1
        import traceback; traceback.print_exc()
        print(f"[L7.1] FAIL {name}: {e!r}", flush=True)
  print(f"[L7.1] {'ALL GREEN' if not failed else 'FAILED'}", flush=True)
  sys.exit(1 if failed else 0)
