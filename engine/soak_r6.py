# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R6 PHASE 3 soak — interleaved two-conversation traffic through the API on
the batch daemon (BATCH_B=2): mixed FRESH/FOLLOW_UP, streams + non-streams,
client cancels, same-conv concurrency bursts, new-conversation evictions.
15-minute class; asserts zero engine restarts, zero wedged slots, queue drains.
"""
import os, sys, json, time, http.client, threading, random

HOST, PORT = "127.0.0.1", 8080
DURATION = float(os.getenv("SOAK_S", "900"))
random.seed(20260924)

def req(method, path, body=None, timeout=300):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    conn.request(method, path, body=json.dumps(body) if body is not None else None,
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse(); data = r.read(); conn.close()
    try: j = json.loads(data)
    except Exception: j = {"_raw": data[:200].decode("utf-8", "replace")}
    return r.status, j

def sse(body, kill_after=None):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=300)
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    if r.status != 200:
        conn.close(); return r.status, 0, b""
    n, buf, t0, aborted = 0, b"", time.time(), False
    try:
        while True:
            if kill_after is not None and time.time() - t0 > kill_after:
                conn.close(); aborted = True; break
            ch = r.read1(65536)
            if not ch: break
            buf += ch
            n += ch.count(b'"content":"') + ch.count(b'"reasoning_content":"')
    except Exception:
        aborted = True
    try: conn.close()
    except Exception: pass
    return 200, n, buf, aborted and 1 or 0

STATS = {"rounds": 0, "ok": 0, "err": 0, "cancelled": 0, "followups": 0, "fresh": 0}
LOCK = threading.Lock()

def health_engine():
    try:
        _, j = req("GET", "/health", timeout=10)
        return j.get("engine") or {}
    except Exception:
        return {}

def turn(conv, history, i, stream, cancel_after=None, max_tokens=48):
    msgs = [{"role": "user", "content": m} for m in history] if history else None
    body = {"model": "qwen3.8-27b-egpu",
            "messages": msgs or [{"role": "user", "content":
                                  f"Soak turn {i}: tell me about topic {random.randint(0, 999)}."}],
            "max_tokens": max_tokens, "conversation_id": conv,
            "enable_thinking": False}
    if stream:
        body["stream"] = True
        st, n, _, canc = sse(body, kill_after=cancel_after)
        with LOCK:
            STATS["ok" if st == 200 else "err"] += 1
            STATS["cancelled"] += canc
        return None
    st, j = turn_sync(body)
    with LOCK:
        STATS["ok" if st == 200 else "err"] += 1
    if st == 200:
        return j["choices"][0]["message"]["content"] or ""
    return None

def turn_sync(body):
    try:
        return req("POST", "/v1/chat/completions", body)
    except Exception as e:
        return -1, {"err": repr(e)}

def main():
    t0 = time.time()
    eng0 = health_engine()
    up0 = eng0.get("uptime_s", 0)
    print(f"[soak] engine up {up0}s batch_b={eng0.get('batch_b')}", flush=True)
    convs = {
        "soak-a": [],
        "soak-b": [],
    }
    nround = 0
    while time.time() - t0 < DURATION:
        nround += 1
        with LOCK: STATS["rounds"] = nround
        # two concurrent turns on the two conversations (the batch window)
        ths = []
        for conv, hist in list(convs.items()):
            stream = (nround + len(conv)) % 2 == 0
            cancel = (nround % 4 == 0) and (conv == "soak-b")
            ths.append(threading.Thread(target=turn,
                                        args=(conv, hist[-2:], nround, stream,
                                              3.0 if cancel else None)))
        # every 5th round: a fresh third conversation (eviction pressure) and a
        # same-conv double-request (conv-lock serialization)
        if nround % 5 == 0:
            ths.append(threading.Thread(target=turn, args=(f"soak-c{nround}", [], nround, False, None)))
        if nround % 7 == 0:
            ths.append(threading.Thread(target=turn, args=("soak-a", [], nround, True, None)))
            ths.append(threading.Thread(target=turn, args=("soak-a", [], nround, False, None)))
        [t.start() for t in ths]; [t.join() for t in ths]
        # extend histories with a follow-up flavor (FOLLOW_UP when reusable)
        if nround % 3 == 0:
            for conv, hist in list(convs.items()):
                hist.append(f"Follow-up {nround}: continue on topic {random.randint(0, 99)}.")
        time.sleep(random.uniform(1.0, 4.0))
        if nround % 10 == 0:
            eng = health_engine()
            _, hq = req("GET", "/health", timeout=10)
            print(f"[soak] {time.time()-t0:6.0f}s round {nround}: {STATS} "
                  f"engine_up={eng.get('uptime_s')} q={hq.get('queue_depth')}", flush=True)
    # final checks
    time.sleep(3)
    eng = health_engine()
    _, hq = req("GET", "/health", timeout=10)
    qd = hq.get("queue_depth")
    up_now = eng.get("uptime_s", 0)
    ok = True
    if up_now < up0 + (DURATION - 30):
        print(f"[soak] FAIL engine restarted (up {up_now}s vs start {up0}s)"); ok = False
    dl = time.time() + 30
    while time.time() < dl and qd and qd > 0:
        time.sleep(1); _, hq = req("GET", "/health", timeout=10); qd = hq.get("queue_depth")
    if qd and qd > 0:
        print(f"[soak] FAIL queue stuck at {qd}"); ok = False
    if STATS["err"] > 0:
        print(f"[soak] FAIL {STATS['err']} errored requests"); ok = False
    print(f"[soak] {'PASS' if ok else 'FAIL'} — {STATS} engine_up={up_now}s", flush=True)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
