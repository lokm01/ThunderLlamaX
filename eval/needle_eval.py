#!/usr/bin/env python3
"""P9 EVAL — 60k-token needle retrieval spot check (dense model's signature).

Builds ~60k-token documents from Gutenberg filler, embeds a 5-digit secret
code + fake fact at 10 different depths (5%..95%), asks for the code via the
chat API (thinking off, 48 max tokens). Reports exact-retrieval accuracy,
TTFT (prefill-dominated), prompt sizes.

Usage: python3 needle_eval.py --tag dense [--trials 10]
"""
import argparse, json, random, re, time, urllib.request, os

API = "http://127.0.0.1:8080"
DATA = "~/tinygrad-metal/eval/data"
OUTD = "~/tinygrad-metal/eval/results"

# ~4.1 chars/token (P9: 60k-class default; NEEDLE_CHARS shrinks for the
# sysmem-exhaustion fallback — see P9 findings: >=54k prompts cross ATTN_THR
# and build the gs26 graph class until the dext MAP_SYSMEM_FD pool runs dry)
CHARS_PER_TRIAL = int(os.getenv("NEEDLE_CHARS", str(int(62000 * 4.1))))

def resident_model():
    with urllib.request.urlopen(API + "/v1/models", timeout=10) as r:
        d = json.load(r)
    for m in d["data"]:
        if m.get("status") == "resident":
            return m["id"]
    raise SystemExit("no resident model")

def load_filler():
    txt = open(f"{DATA}/gutenberg_pap.txt", encoding="utf-8", errors="replace").read()
    i = txt.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    if i >= 0:
        txt = txt[txt.find("\n", i) + 1:]
    j = txt.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if j >= 0:
        txt = txt[:j]
    return txt

def wait_healthy(timeout=1500):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(API + "/health", timeout=5) as r:
                if json.load(r).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(15)
    return False


def run_one(model, user_content, max_tokens=48):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    req = urllib.request.Request(API + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    content, ttft, usage = [], None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        buf = b""
        for raw in r:
            buf += raw
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if ev.get("error"):
                    raise RuntimeError("API error event: " + json.dumps(ev)[:300])
                if ev.get("usage"):
                    usage = ev["usage"]
                delta = ((ev.get("choices") or [{}])[0].get("delta") or {})
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    content.append(delta["content"])
    return "".join(content), ttft, time.perf_counter() - t0, usage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    outp = f"{OUTD}/needle_{args.tag}.jsonl"

    model = resident_model()
    filler = load_filler()
    while len(filler) < CHARS_PER_TRIAL * 2:
        filler = filler + "\n" + filler
    print(f"[needle] model={model} filler={len(filler)} chars", flush=True)

    rng = random.Random(20260925)
    n_ok = n_hit = 0
    for t in range(args.trials):
        depth = 0.05 + 0.90 * (t / max(1, args.trials - 1))     # 5%..95%
        code = f"{rng.randint(10000, 99999)}"
        pin = f"{rng.randint(100, 999)}"
        needle = (f"\nInternal memo (do not distribute): the Aurora lab access code for this "
                  f"week is {code}, and the backup door PIN is {pin}. Keep these confidential.\n")
        cut = int(CHARS_PER_TRIAL * depth)          # constant-size docs
        doc = filler[:cut] + needle + filler[cut:CHARS_PER_TRIAL]
        ask = ("\n\nThe document above contains an internal memo with a five-digit access "
               "code for the Aurora lab. What is that five-digit access code? Reply with "
               "only the code.")
        content, ttft, total, usage = None, None, None, None
        for attempt in range(4):
            try:
                content, ttft, total, usage = run_one(model, doc + ask)
                break
            except Exception as e:
                msg = str(e)
                print(f"[needle] t{t} attempt {attempt} failed ({msg[:120]}) — waiting",
                     flush=True)
                wait_healthy()
        hit = bool(content) and code in (content or "")
        pin_hit = bool(content) and pin in (content or "")
        n_ok += 1; n_hit += int(hit)
        rec = {"trial": t, "depth": round(depth, 3), "code": code, "pin": pin,
               "retrieved": hit, "pin_retrieved": pin_hit,
               "response": (content or "")[:120],
               "ttft": round(ttft, 1) if ttft else None,
               "total_s": round(total, 1) if total else None,
               "prompt_tokens": (usage or {}).get("prompt_tokens"),
               "completion_tokens": (usage or {}).get("completion_tokens")}
        with open(outp, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[needle] t{t} depth={depth:.2f} code={code} hit={hit} pin={pin_hit} "
              f"ttft={rec['ttft']}s ptok={rec['prompt_tokens']} resp={rec['response'][:60]!r}", flush=True)
    print(f"[needle] SUMMARY accuracy={n_hit}/{n_ok} = {n_hit/max(1,n_ok):.2f}", flush=True)

if __name__ == "__main__":
    main()
