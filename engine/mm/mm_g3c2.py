#!/usr/bin/env python3
"""MM SESSION C -- G3 round 3: position-vs-ctx discrimination with a
line-draining RPC reader (the mm_g3c round-2 assert was an undrained
prefill_progress event, NOT a daemon fault).
  D1: 2k feed, LONG window (600 cycles t1 vs spec) -- divergence at all?
  D3: 1k feed, long window (small-ctx control)
Output: /tmp/mm_g3c2.json"""
import socket, json, sys, os
import numpy as np

SOCK = "/tmp/llm-engine.sock"
DOC = "~/mm_p5_doc100k_ids.npy"


class Eng:
    def __init__(self):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.connect(SOCK)
        self.s.settimeout(900)
        self.n = 0
        self.buf = b""

    def _line(self):
        while b"\n" not in self.buf:
            d = self.s.recv(1 << 20)
            if not d:
                return None
            self.buf += d
        line, self.buf = self.buf.split(b"\n", 1)
        return line

    def rpc(self, method, **kw):
        self.n += 1
        req = json.dumps({"id": self.n, "method": method, "params": kw}).encode() + b"\n"
        self.s.sendall(req)
        while True:
            line = self._line()
            if line is None:
                raise RuntimeError("socket closed")
            r = json.loads(line)
            if r.get("id") == self.n and "event" not in r:
                return r
            # else: progress/event line -- keep draining

    def prefill(self, ids, cid, mode="FRESH"):
        r = self.rpc("prefill", mode=mode, ids=[int(x) for x in ids], conversation_id=cid)
        assert r.get("ok"), r
        return r["result"]

    def gen(self, force, cid, max_cycles=600):
        self.n += 1
        params = {"max_cycles": max_cycles, "stop_token_ids": []}
        if force:
            params["force_mode"] = force
        req = json.dumps({"id": self.n, "method": "generate", "params": params}).encode() + b"\n"
        self.s.sendall(req)
        toks = []
        while True:
            line = self._line()
            if line is None:
                return toks
            r = json.loads(line)
            if r.get("event") == "cycle":
                toks += r.get("tokens", [])
            elif r.get("event") in ("done", "cancelled"):
                toks += r.get("tokens", [])
                return toks
            elif r.get("event") == "error":
                raise RuntimeError(r.get("error"))


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def arm(e, doc, N, tag, out, cyc=600):
    e.prefill(doc[:N], cid=tag + "a")
    t1 = e.gen("t1", tag + "a", cyc)
    e.prefill(doc[:N], cid=tag + "b")
    sp = e.gen("spec", tag + "b", cyc)
    fd = first_diff(t1, sp)
    out[tag] = {"match": fd is None, "first_diff": fd, "n_t1": len(t1), "n_spec": len(sp)}
    print(f"[g3c2] {tag}: {out[tag]}", flush=True)
    json.dump(out, open("/tmp/mm_g3c2.json", "w"), indent=1)


def main():
    doc = np.load(DOC).astype(np.int32)
    e = Eng()
    out = {}
    arm(e, doc, 3072, "d2_3k", out)
    doc = None  # placeholder
    print("[g3c2] done", flush=True)


if __name__ == "__main__":
    main()
