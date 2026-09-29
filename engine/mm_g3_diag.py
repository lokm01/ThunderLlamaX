#!/usr/bin/env python3
"""G3 divergence diagnostic: WHERE does spec != t1 diverge at 4k ctx, and is
it the adjudicated drift class (both arms deterministic; the t1 arm's own
continuation passes through the same tokens) or a wiring bug?
  D1: t1 gen 40 (x2, det) + spec gen 40 (x2, det) -> first-divergence index
  D2: at the divergence position k: re-prefill, run t1 for k tokens, then
      ONE more t1 step vs ONE spec cycle (both at the same state) -> compare
      the greedy at that position directly (isolates the probe-vs-t1 numerics)
"""
import os, sys, time, json, socket
import numpy as np

BASE = "~/tinygrad-metal"
sys.path.insert(0, BASE + "/engine0")
SOCK = "/tmp/llm-engine.sock"
DOC = os.path.expanduser("~/mm_p5_doc100k_ids.npy")


class Eng:
    def __init__(self, timeout=1800.0):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(timeout)
        self.s.connect(SOCK)
        self.buf = b""; self.n = 0
    def send(self, method, params):
        self.n += 1
        self.s.sendall((json.dumps({"id": self.n, "method": method, "params": params}) + "\n").encode())
        return self.n
    def recv(self):
        while b"\n" not in self.buf:
            ch = self.s.recv(1 << 20)
            if not ch: raise RuntimeError("engine closed")
            self.buf += ch
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)
    def rpc(self, method, params):
        rid = self.send(method, params)
        while True:
            r = self.recv()
            if r.get("id") == rid and "event" not in r:
                if not r.get("ok"):
                    raise RuntimeError(f"{method}: {r.get('error')}")
                return r["result"]
    def prefill(self, ids, cid):
        return self.rpc("prefill", {"mode": "FRESH", "ids": [int(x) for x in ids], "conversation_id": cid})
    def generate(self, mc, force=None):
        p = {"max_cycles": mc, "stop_token_ids": []}
        if force: p["force_mode"] = force
        rid = self.send("generate", p)
        toks = []
        while True:
            r = self.recv()
            if r.get("id") != rid: continue
            if r.get("event") == "cycle":
                toks += r.get("tokens", [])
            elif r.get("event") in ("done", "cancelled"):
                return toks, r
            elif r.get("event") == "error":
                raise RuntimeError(r.get("error"))


def main():
    doc = np.load(DOC).astype(np.int32)
    N = 4096
    e = Eng()
    e.prefill(doc[:N], "d-t1a")
    t1a, _ = e.generate(40, "t1")
    e.prefill(doc[:N], "d-t1b")
    t1b, _ = e.generate(40, "t1")
    e.prefill(doc[:N], "d-spa")
    spa, _ = e.generate(40, "spec")
    e.prefill(doc[:N], "d-spb")
    spb, _ = e.generate(40, "spec")
    print(f"D1 det: t1 {t1a == t1b} | spec {spa == spb}")
    k = next((i for i, (a, b) in enumerate(zip(t1a, spa)) if a != b), None)
    print(f"D1 first divergence: index {k} of {len(t1a)}")
    print(f"   t1  [{k-2}:{k+6}] = {t1a[max(0,k-2):k+6]}")
    print(f"   spec[{k-2}:{k+6}] = {spa[max(0,k-2):k+6]}")
    # D2: the position-isolated probe — feed the SAME prefix through t1 steps
    # to the divergence position, then one t1 cycle vs one spec cycle
    if k is not None:
        # the full stream = prompt + generated[:k]; run t1 cycles to reach it
        stream = list(doc[:N]) + t1a[:k]
        e.prefill(stream, "d-iso")
        # one t1 cycle: emits 1 token
        t_one, _ = e.generate(1, "t1")
        # redo; the spec cycle at this position (no lookup hit likely at the
        # divergence — force spec; if it drafts, the accept compares)
        e.prefill(stream, "d-iso2")
        s_one, _ = e.generate(1, "spec")
        print(f"D2 isolated: t1 {t_one} vs spec {s_one} -> {'AGREE' if t_one == s_one else 'DISAGREE'}")
        # a second isolated probe 2 tokens earlier (context sensitivity)
        stream2 = list(doc[:N]) + t1a[:max(0, k - 1)]
        e.prefill(stream2, "d-iso3")
        t_two, _ = e.generate(2, "t1")
        e.prefill(stream2, "d-iso4")
        s_two, _ = e.generate(2, "spec")
        print(f"D2b (k-1 start): t1 {t_two} vs spec {s_two} -> {'AGREE' if t_two == s_two[:len(t_two)] else 'DISAGREE'}")


if __name__ == "__main__":
    main()
