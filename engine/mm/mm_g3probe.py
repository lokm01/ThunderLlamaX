#!/usr/bin/env python3
"""MM SESSION B -- G3 discrimination: FRESH-feed t1 vs FRESH-feed spec at 4k
(no pcache in the loop), then AUTO_CACHE restore vs FRESH continuation, then
t1-after-restore. Through the live daemon RPC (no GPU process)."""
import socket, json, sys
import numpy as np

SOCK = "/tmp/llm-engine.sock"
DOC = "~/mm_p5_doc100k_ids.npy"

class Eng:
    def __init__(self):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.connect(SOCK); self.s.settimeout(600)
        self.n = 0
    def rpc(self, method, **kw):
        self.n += 1
        self.s.sendall(json.dumps({"id": self.n, "method": method, "params": kw}).encode() + b"\n")
        buf = b""
        while b"\n" not in buf:
            d = self.s.recv(1 << 20)
            if not d: break
            buf += d
        return json.loads(buf.split(b"\n")[0])
    def prefill(self, ids, cid, mode="FRESH"):
        r = self.rpc("prefill", mode=mode, ids=[int(x) for x in ids],
                     conversation_id=cid)
        assert r.get("ok"), r
        return r["result"]
    def gen(self, force, cid, ntok=32):
        self.s.sendall(json.dumps({"id": self.n, "method": "generate", "params":
            {"max_cycles": 200, "stop_token_ids": [], **({"force_mode": force} if force else {})}}).encode() + b"\n")
        toks = []
        buf = b""
        while True:
            while b"\n" not in buf:
                d = self.s.recv(1 << 20)
                if not d: return toks
                buf += d
            line, buf = buf.split(b"\n", 1)
            r = json.loads(line)
            if r.get("event") == "cycle":
                toks += r.get("tokens", [])
            elif r.get("event") in ("done", "cancelled"):
                toks += r.get("tokens", [])
                return toks
            elif r.get("event") == "error":
                raise RuntimeError(r.get("error"))
            elif "event" not in r and not r.get("ok"):
                raise RuntimeError(r.get("error"))
    def cancel(self):
        try: self.rpc("cancel")
        except Exception: pass

def main():
    doc = np.load(DOC).astype(np.int32)
    N = 4096
    e = Eng()
    out = {}
    # A: FRESH + t1  vs  B: FRESH + spec  (pcache out of the loop)
    e.prefill(doc[:N], cid="pA"); tA = e.gen("t1", "pA")
    e.prefill(doc[:N], cid="pB"); tB = e.gen("spec", "pB")
    out["fresh_t1_vs_fresh_spec"] = (tA == tB[:len(tA)], tA[:8], tB[:8])
    print("[g3p] FRESH t1 vs FRESH spec:", out["fresh_t1_vs_fresh_spec"], flush=True)
    # C: MTP-mode default (no force) vs t1
    e.prefill(doc[:N], cid="pC"); tC = e.gen(None, "pC")
    out["fresh_t1_vs_mtp"] = (tA == tC[:len(tA)], tC[:8])
    print("[g3p] FRESH t1 vs MTP default:", out["fresh_t1_vs_mtp"], flush=True)
    # D: restore path: re-prefill AUTO_CACHE + t1  vs  tA
    r = e.prefill(doc[:N], cid="pD", mode="AUTO_CACHE")
    tD = e.gen("t1", "pD")
    out["restore_t1_vs_fresh_t1"] = (tA == tD[:len(tA)], r.get("mode"), r.get("cached_tokens"), tD[:8])
    print("[g3p] RESTORE t1 vs FRESH t1:", out["restore_t1_vs_fresh_t1"], flush=True)
    # E: 2048-class control (the G4 shape)
    e.prefill(doc[:2048], cid="pE"); tE = e.gen("t1", "pE")
    r = e.prefill(doc[:2048], cid="pF", mode="AUTO_CACHE")
    tF = e.gen("t1", "pF")
    out["restore2048_t1_vs_fresh"] = (tE == tF[:len(tE)], r.get("mode"), r.get("cached_tokens"))
    print("[g3p] RESTORE-2048 control:", out["restore2048_t1_vs_fresh"], flush=True)
    json.dump({k: (v[0] if isinstance(v, tuple) else v) for k, v in out.items()},
              open("/tmp/mm_g3probe.json", "w"), indent=1)
    e.rpc("shutdown") if False else None
    print("[g3p] done", flush=True)

if __name__ == "__main__":
    main()
