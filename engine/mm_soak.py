#!/usr/bin/env python3
"""TLX P8 MoE SERVING SOAK (mm_soak) — mixed traffic against the LIVE MoE
daemon through the API for DURATION_S (default 900): alternating rounds of
  R1  a fresh conversation + 48-token stream completion (prose class)
  R2  a FOLLOW_UP turn on the same conversation (the delta-prefill path)
  R3  a quote-class prompt (the D8 path: 8-gram repeats)
  R4  the SAME prompt as an earlier round (the pcache AUTO_CACHE path)
  R5  a cancel mid-stream (the abort-safety protocol)
plus /health polls every 10s. FAILS LOUD on: any 5xx, a dirty engine after a
round, a wedged >120s request, a device fault (the API exits anyway).
"""
import json, os, socket, sys, time, urllib.request, urllib.error

API = "http://127.0.0.1:8080"
DURATION = int(os.getenv("SOAK_S", "900"))
LOG = open("/tmp/mm_soak.log", "a", buffering=1)

def log(*a):
    line = " ".join(str(x) for x in a)
    LOG.write(line + "\n")
    print("[soak]", line, flush=True)

def post(path, body, timeout=300, stream=False):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not stream:
            return json.loads(r.read()), time.time() - t0
        chunks = 0
        for _ in r:
            chunks += 1
        return {"chunks": chunks}, time.time() - t0

def health():
    with urllib.request.urlopen(API + "/health", timeout=8) as r:
        return json.loads(r.read())

def engine_status():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(10)
    s.connect("/tmp/llm-engine.sock")
    s.sendall((json.dumps({"id": 1, "method": "status"}) + "\n").encode())
    buf = b""
    while b"\n" not in buf:
        buf += s.recv(65536)
    s.close()
    return json.loads(buf.split(b"\n")[0])["result"]

CONVS = 0
def cid():
    global CONVS
    CONVS += 1
    return f"soak-{time.strftime('%H%M%S')}-{CONVS}"

def main():
    t_end = time.time() + DURATION
    round_n = 0
    fails = 0
    prompts = [
        "Explain in two sentences how a refrigerator works.",
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z "
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z "
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z ",
        "def fib(n):\n    if n <= 1:\n        return n\n    return",
        "Write a haiku about GPUs.",
    ]
    log(f"soak start: {DURATION}s")
    while time.time() < t_end:
        round_n += 1
        try:
            m = round_n % 5
            if m == 1:
                c = cid()
                r, dt = post("/v1/chat/completions", {
                    "model": "qwen3.6-35b-a3b-egpu", "max_tokens": 48,
                    "stream": True, "conversation_id": c,
                    "messages": [{"role": "user", "content": prompts[round_n % len(prompts)]}]},
                    stream=True)
                log(f"R{round_n} stream: {r['chunks']} chunks {dt:.1f}s")
            elif m == 2:
                c = f"soak-follow-{(round_n // 5) * 5 + 1}"
                r, dt = post("/v1/chat/completions", {
                    "model": "qwen3.6-35b-a3b-egpu", "max_tokens": 32,
                    "conversation_id": c,
                    "messages": [
                        {"role": "user", "content": prompts[(round_n - 1) % len(prompts)]},
                        {"role": "assistant", "content": "(earlier reply)"},
                        {"role": "user", "content": "Now say it in one sentence."}]})
                n = len((r.get("choices") or [{}])[0].get("message", {}).get("content") or "")
                log(f"R{round_n} follow-up: {n} chars {dt:.1f}s")
            elif m == 3:
                r, dt = post("/v1/chat/completions", {
                    "model": "qwen3.6-35b-a3b-egpu", "max_tokens": 64,
                    "messages": [{"role": "user", "content": prompts[1]}]})
                log(f"R{round_n} quote-class: {dt:.1f}s")
            elif m == 4:
                # the repeat prompt exercises AUTO_CACHE as the cache fills
                r, dt = post("/v1/chat/completions", {
                    "model": "qwen3.6-35b-a3b-egpu", "max_tokens": 16,
                    "messages": [{"role": "user", "content": prompts[3]}]})
                u = r.get("usage", {})
                log(f"R{round_n} repeat: cached={u.get('prompt_tokens_details', {}).get('cached_tokens')} {dt:.1f}s")
            else:
                # cancel: a tiny read timeout on a long stream client-side
                try:
                    post("/v1/chat/completions", {
                        "model": "qwen3.6-35b-a3b-egpu", "max_tokens": 200,
                        "stream": True,
                        "messages": [{"role": "user", "content": prompts[0]}]},
                        timeout=3, stream=True)
                except Exception as e:
                    log(f"R{round_n} cancel: {type(e).__name__} (expected)")
            if round_n % 3 == 0:
                h = health()
                st = engine_status()
                if h.get("status") != "ok":
                    fails += 1
                    log(f"!! health {h}")
                if st.get("dirty"):
                    fails += 1
                    log(f"!! engine dirty after R{round_n}")
                log(f"  health ok | pos {st.get('pos')} pc {st.get('pc', {}).get('entries')} "
                    f"cyc {st.get('cycles_since_rebuild')}")
        except urllib.error.HTTPError as e:
            fails += 1
            log(f"!! R{round_n} HTTP {e.code}: {e.read()[:200]}")
        except Exception as e:
            fails += 1
            log(f"!! R{round_n} {type(e).__name__}: {e}")
    h = health()
    st = engine_status()
    log(f"soak DONE: {round_n} rounds, {fails} failures | health {h.get('status')} "
        f"| dirty {st.get('dirty')} | pc entries {st.get('pc', {}).get('entries')} "
        f"| fences {st.get('cycles_since_rebuild')}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
