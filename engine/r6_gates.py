# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R6 PHASE 3 GPU gates — per-stream bit-exactness THROUGH THE SERVING PATH
(the engine socket = the daemon's batch scheduler, not the r6_batch harness).

Conversations:
  A = prompt8k (7775 toks)  conv "ga"
  B = 100k-corpus slice @40000 (7775 toks) conv "gb"   (no cache coverage ->
      a TRUE FRESH T=1 prefill; A likewise in the fresh batch-daemon namespace)

Phases:
  1. refA = FRESH(A)+gen   ; refB = FRESH(B)+gen           [solo, slots 0/1]
  2. A2/B2 = AUTO_CACHE replay + gen                        [cache exactness +
                                                              slot reuse]
  3. x2: A3+B3 prefilled, generates started back-to-back    [the batch window]
       -> per-stream token streams identical to refA/refB; mid-run status shows
          BOTH slots generating; deterministic across reps.
  4. perf: solo 300-token timing per stream vs the concurrent pair window ->
          aggregate tok/s (report; bar = 1.5x mean solo).

Run on the rig against a live batch daemon (BATCH_B=2). Stdlib only.
"""
import os, sys, json, time, socket, threading

SOCK = os.getenv("ENGINE_SOCK", "/tmp/llm-engine.sock")
NCYC_EXACT = int(os.getenv("R6G_NCYC", "40"))
NCYC_PERF = int(os.getenv("R6G_NCYC_PERF", "130"))
CMP_N = int(os.getenv("R6G_CMP_N", "50"))
FAILED = [False]
RAW = []   # every line received (failure forensics)

def step(name, ok, extra=""):
  print(f"[r6gate] {'PASS' if ok else 'FAIL'}: {name} {extra}", flush=True)
  if not ok: FAILED[0] = True

class Eng:
  def __init__(self):
    self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); self.s.settimeout(900)
    self.s.connect(SOCK); self.buf = b""
  def send(self, o): self.s.sendall((json.dumps(o) + "\n").encode())
  def recv(self):
    while b"\n" not in self.buf:
      ch = self.s.recv(1 << 20)
      if not ch: raise EOFError("engine closed")
      self.buf += ch
    line, self.buf = self.buf.split(b"\n", 1)
    r = json.loads(line)
    RAW.append(r)
    return r
  def reply(self, wid):
    while True:
      r = self.recv()
      if r.get("id") == wid and "event" not in r:
        if not r.get("ok"): raise RuntimeError(r.get("error"))
        return r["result"]
  def status(self):
    self.send({"id": 9999, "method": "status"})
    return self.reply(9999)
  def prefill(self, ids, conv, mode="AUTO_CACHE"):
    self.send({"id": 10, "method": "prefill", "params":
               {"mode": mode, "ids": ids, "conversation_id": conv}})
    return self.reply(10)
  def generate(self, max_cycles, on_first_cycle=None, stop_ids=None):
    """Returns (tokens, cycles, t0, t_first, t_end). Runs in THIS thread."""
    self.send({"id": 11, "method": "generate", "params":
               {"max_cycles": max_cycles, "stop_token_ids": stop_ids or []}})
    toks, k, t0, tfirst = [], 0, None, None
    while True:
      r = self.recv()
      if r.get("id") != 11: continue
      ev = r.get("event")
      if ev == "cycle":
        if t0 is None: t0 = time.time()
        if tfirst is None:
          tfirst = time.time()
          if on_first_cycle: on_first_cycle()
        toks += r["tokens"]; k = r["cycle"]
      elif ev in ("done", "cancelled"):
        toks += []  # terminal tokens already in cycle batches
        return toks, k, t0, tfirst, time.time()
      elif not r.get("ok", True):
        raise RuntimeError(r.get("error"))
  def close(self):
    try: self.s.close()
    except Exception: pass

def wait_ready(timeout=900):
  dl = time.time() + timeout
  while time.time() < dl:
    try:
      e = Eng(); st = e.status(); e.close()
      if st.get("ready"):
        if not st.get("batch_b"):
          step("daemon batch capability", False, f"batch_b={st.get('batch_b')} — daemon not BATCH_B=2")
          sys.exit(1)
        return st
    except Exception:
      pass
    time.sleep(2)
  step("daemon ready", False, "timeout")
  sys.exit(1)

def load_ids():
  import numpy as np
  a = [int(t) for t in np.load("~/r6_p8k_ids.npy")]
  snap = np.load("~/snap100k/ids.npy")
  b = [int(t) for t in snap[40000:40000 + len(a)]]
  return a, b

def solo_run(ids, conv, ncyc, tag):
  e = Eng()
  pr = e.prefill(ids, conv)
  toks, k, t0, tf, t1 = e.generate(ncyc)
  e.close()
  return pr, toks, (t1 - tf), tag

def main():
  st0 = wait_ready()
  print(f"[r6gate] daemon ready: streams={len(st0.get('streams', []))} "
        f"config_fp={str(st0.get('config_fp'))[:16]}", flush=True)
  idsA, idsB = load_ids()
  print(f"[r6gate] convA {len(idsA)} toks; convB {len(idsB)} toks", flush=True)

  # ---- phase 1: solo FRESH runs (slots 0 and 1 exercised solo) ----
  t0 = time.time()
  prA, refA, _, _ = solo_run(idsA, "ga", NCYC_EXACT, "refA")
  print(f"[r6gate] refA prefill {time.time()-t0:.0f}s mode={prA.get('mode')} "
        f"cached={prA.get('cached_tokens')}; gen {len(refA)} toks: {refA[:12]}...", flush=True)
  t0 = time.time()
  prB, refB, _, _ = solo_run(idsB, "gb", NCYC_EXACT, "refB")
  print(f"[r6gate] refB prefill {time.time()-t0:.0f}s mode={prB.get('mode')} "
        f"cached={prB.get('cached_tokens')}; gen {len(refB)} toks: {refB[:12]}...", flush=True)
  step("solo FRESH both streams produced output", len(refA) >= CMP_N and len(refB) >= CMP_N,
       f"(A {len(refA)}, B {len(refB)})")

  # ---- phase 2: AUTO_CACHE replay (cache exactness + slot reuse) ----
  prA2, A2, _, _ = solo_run(idsA, "ga", NCYC_EXACT, "A2")
  prB2, B2, _, _ = solo_run(idsB, "gb", NCYC_EXACT, "B2")
  step("phase2 A cache-replay == refA", A2[:CMP_N] == refA[:CMP_N],
       f"mode={prA2.get('mode')} cached={prA2.get('cached_tokens')}")
  step("phase2 B cache-replay == refB", B2[:CMP_N] == refB[:CMP_N],
       f"mode={prB2.get('mode')} cached={prB2.get('cached_tokens')}")

  # ---- phase 3 x2: the concurrent batch window ----
  for rep in range(2):
    eA, eB = Eng(), Eng()
    pa = eA.prefill(idsA, "ga"); pb = eB.prefill(idsB, "gb")
    res = {}
    mid = {"seen": False}
    def genA():
      res["A"] = eA.generate(NCYC_EXACT)
    def genB():
      res["B"] = eB.generate(NCYC_EXACT)
    thA = threading.Thread(target=genA); thB = threading.Thread(target=genB)
    thA.start(); thB.start()
    # mid-run: BOTH slots generating simultaneously
    def probe():
      time.sleep(1.0)
      try:
        pe = Eng(); st = pe.status(); pe.close()
        gens = [s["generating"] for s in st.get("streams", [])]
        mid["seen"] = (gens == [True, True])
        mid["st"] = gens
      except Exception as exc:
        mid["err"] = repr(exc)
    pr = threading.Thread(target=probe); pr.start()
    thA.join(); thB.join(); pr.join()
    eA.close(); eB.close()
    toksA, toksB = res["A"][0], res["B"][0]
    okA = toksA[:CMP_N] == refA[:CMP_N]
    okB = toksB[:CMP_N] == refB[:CMP_N]
    step(f"phase3 rep{rep} A batched == refA", okA,
         f"({len(toksA)} toks, first div {next((i for i in range(min(len(toksA), CMP_N)) if toksA[i] != refA[i]), None)})")
    step(f"phase3 rep{rep} B batched == refB", okB,
         f"({len(toksB)} toks, first div {next((i for i in range(min(len(toksB), CMP_N)) if toksB[i] != refB[i]), None)})")
    step(f"phase3 rep{rep} both slots generating mid-run", mid.get("seen", False), str(mid))
    if rep == 0:
      batch_out = (list(toksA), list(toksB))
    else:
      step("phase3 deterministic x2 (A)", batch_out[0][:CMP_N] == toksA[:CMP_N])
      step("phase3 deterministic x2 (B)", batch_out[1][:CMP_N] == toksB[:CMP_N])

  # ---- phase 4: aggregate perf (steady-state batch window) ----
  eA, eB = Eng(), Eng()
  eA.prefill(idsA, "ga"); eB.prefill(idsB, "gb")
  r = {}
  def gA(): r["A"] = eA.generate(NCYC_PERF)
  def gB(): r["B"] = eB.generate(NCYC_PERF)
  tA = threading.Thread(target=gA); tB = threading.Thread(target=gB)
  tA.start(); tB.start(); tA.join(); tB.join()
  eA.close(); eB.close()
  toksA, kA, _, tfA, t1A = r["A"]
  toksB, kB, _, tfB, t1B = r["B"]
  w = max(t1A, t1B) - max(tfA, tfB)      # the overlap window
  agg = (len(toksA) + len(toksB)) / w
  print(f"[r6gate] concurrent: A {len(toksA)} in {t1A-tfA:.1f}s ({len(toksA)/(t1A-tfA):.1f} tok/s), "
        f"B {len(toksB)} in {t1B-tfB:.1f}s ({len(toksB)/(t1B-tfB):.1f} tok/s); "
        f"window {w:.1f}s AGGREGATE {agg:.2f} tok/s", flush=True)
  eS = Eng()
  eS.prefill(idsA, "ga")
  toksAs, kAs, _, tfAs, t1As = eS.generate(NCYC_PERF)
  eS.close()
  soloA = len(toksAs) / (t1As - tfAs)
  print(f"[r6gate] solo A: {len(toksAs)} in {t1As-tfAs:.1f}s = {soloA:.2f} tok/s", flush=True)
  ratio = agg / soloA
  step("aggregate >= 1.5x solo", ratio >= 1.5, f"agg {agg:.2f} vs solo {soloA:.2f} = {ratio:.2f}x")

  print("=" * 60, flush=True)
  print(f"[r6gate] {'ALL GREEN' if not FAILED[0] else 'FAILURES PRESENT'}", flush=True)
  sys.exit(1 if FAILED[0] else 0)

if __name__ == "__main__":
  main()
