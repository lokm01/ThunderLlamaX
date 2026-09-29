#!/usr/bin/env python3
"""P9 EVAL — GSM8K 4-shot accuracy + speed through the rig's OpenAI API.

Protocol: first 4 TRAIN problems as the few-shot primer (the paper's format,
calculator annotations kept), FIRST 100 test problems (deterministic),
greedy decode (the engine is greedy-always; temperature is ignored),
enable_thinking=false (the standard non-thinking GSM8K protocol; comparable
to literature), max_tokens=320, stop on a new "Question:".

Streaming is used so we can split TTFT (prefill-dominated) from decode rate.
Results: eval/results/gsm8k_<tag>.jsonl (per-problem, incremental) +
a summary dict printed at the end.

Usage: python3 gsm8k_eval.py --tag moe_mtp --n 100 [--start 0]
"""
import argparse, json, re, sys, time, urllib.request

API = "http://127.0.0.1:8080"
DATA = "~/tinygrad-metal/eval/data"
OUTD = "~/tinygrad-metal/eval/results"

def resident_model():
    with urllib.request.urlopen(API + "/v1/models", timeout=10) as r:
        d = json.load(r)
    for m in d["data"]:
        if m.get("status") == "resident":
            return m["id"]
    raise SystemExit("no resident model in /v1/models: " + json.dumps(d))

def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def norm_num(s):
    if s is None:
        return None
    s = str(s).replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        v = float(s)
        return int(v) if v == int(v) else v
    except ValueError:
        return None

def gold_from_answer(a):
    m = re.findall(r"####\s*(.+)", a)
    return norm_num(m[-1]) if m else None

def extract_pred(text):
    """(value, method) — last '#### N' wins; fallback last number; None if neither."""
    if text is None:
        return None, "empty"
    m = re.findall(r"####\s*(-?[$,]?[\d,]*\.?\d+)", text)
    if m:
        return norm_num(m[-1]), "hash"
    m2 = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", text)
    if m2:
        return norm_num(m2[-1]), "lastnum"
    return None, "none"

def wait_healthy(timeout=1500):
    """Ride out engine crash-heal cycles (the dense engine faulted ~every
    10-15 min under GSM8K load during P9; launchd heals in ~9-10 min)."""
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


def run_one(model, prompt, max_tokens=320):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "stop": ["\nQuestion:", "Question:"],
        "enable_thinking": False,
    }
    req = urllib.request.Request(API + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    content, reasoning, ttft, usage, finish = [], [], None, None, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        buf = b""
        for raw in r:
            buf += raw
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    ev = json.loads(payload)
                except ValueError:
                    continue
                if ev.get("error"):
                    raise RuntimeError("API error event: " + json.dumps(ev)[:300])
                if "usage" in ev and ev.get("usage"):
                    usage = ev["usage"]
                ch = (ev.get("choices") or [{}])[0]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
    t1 = time.perf_counter()
    return {
        "text": "".join(content), "reasoning": "".join(reasoning),
        "ttft": ttft, "total_s": t1 - t0, "usage": usage, "finish": finish,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=320)
    args = ap.parse_args()

    import os
    os.makedirs(OUTD, exist_ok=True)
    outp = f"{OUTD}/gsm8k_{args.tag}.jsonl"
    model = resident_model()
    print(f"[gsm8k] model={model} tag={args.tag} n={args.n} (from {args.start}) -> {outp}", flush=True)

    shots = load_jsonl(f"{DATA}/gsm8k_train.jsonl")[:args.shots]
    tests = load_jsonl(f"{DATA}/gsm8k_test.jsonl")[args.start:args.start + args.n]
    primer = "\n\n".join(f"Question: {s['question']}\nAnswer: {s['answer']}" for s in shots) + "\n\n"

    done = set()
    if os.path.exists(outp):
        for r in load_jsonl(outp):
            if r.get("ok"):
                done.add(r["idx"])
        print(f"[gsm8k] resuming: {len(done)} problems already done", flush=True)

    n_ok = n_ext_fail = n_correct = 0
    lat, ttfts, gen_ts, cptoks, prtoks, tps = [], [], [], [], [], []
    t_all = time.perf_counter()
    for i, ex in enumerate(tests):
        idx = args.start + i
        if idx in done:
            continue
        gold = gold_from_answer(ex["answer"])
        prompt = primer + f"Question: {ex['question']}\nAnswer:"
        r = None
        for attempt in range(4):
            try:
                r = run_one(model, prompt, args.max_tokens)
                break
            except Exception as e:
                msg = str(e)
                transient = ("503" in msg or "HTTP" in msg or "socket" in msg.lower()
                             or "Connection" in msg or "URLError" in type(e).__name__
                             or "timed out" in msg.lower() or "reset" in msg.lower())
                print(f"[gsm8k] #{idx} attempt {attempt} failed ({msg[:120]}) "
                      f"{'- waiting for engine health' if transient else '- NON-TRANSIENT'}",
                      flush=True)
                if not transient:
                    break
                wait_healthy()
        if r is None:
            with open(outp, "a") as f:
                f.write(json.dumps({"idx": idx, "ok": False, "error": msg[:300]}) + "\n")
            continue
        pred, method = extract_pred(r["text"])
        correct = (pred is not None and gold is not None and
                   abs(float(pred) - float(gold)) < 1e-6)
        u = r.get("usage") or {}
        rec = {
            "idx": idx, "ok": True, "gold": gold, "pred": pred,
            "correct": bool(correct), "extract": method,
            "finish": r["finish"], "ttft": round(r["ttft"], 3) if r["ttft"] else None,
            "total_s": round(r["total_s"], 3),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": u.get("completion_tokens_details", {}).get("reasoning_tokens")
                                 if isinstance(u.get("completion_tokens_details"), dict) else None,
            "text": r["text"][-400:],
        }
        with open(outp, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n_ok += 1
        n_correct += int(correct)
        n_ext_fail += int(method in ("none", "empty"))
        if r["ttft"]:
            ttfts.append(r["ttft"])
        lat.append(r["total_s"])
        ct = u.get("completion_tokens")
        if ct:
            cptoks.append(ct)
            gen_t = r["total_s"] - (r["ttft"] or 0)
            gen_ts.append(gen_t)
            tps.append(ct / gen_t if gen_t > 0 else 0)
        if u.get("prompt_tokens"):
            prtoks.append(u["prompt_tokens"])
        if n_ok % 10 == 0:
            acc = n_correct / max(1, n_ok + len(done))
            print(f"[gsm8k] {n_ok + len(done)}/{args.n} acc={acc:.3f} "
                  f"mean_lat={sum(lat)/len(lat):.1f}s med_tps={sorted(tps)[len(tps)//2] if tps else 0:.1f}", flush=True)

    n_tot = n_ok + len(done)
    def med(x): return sorted(x)[len(x) // 2] if x else 0.0
    summary = {
        "tag": args.tag, "model": model, "n": n_tot, "shots": args.shots,
        "max_tokens": args.max_tokens, "thinking": False,
        "accuracy": round(n_correct / max(1, n_tot), 4),
        "n_correct": n_correct, "n_completed": n_ok, "n_prev_done": len(done),
        "extract_failures": n_ext_fail,
        "latency_mean_s": round(sum(lat) / len(lat), 2) if lat else None,
        "latency_median_s": round(med(lat), 2),
        "ttft_median_s": round(med(ttfts), 2),
        "gen_time_median_s": round(med(gen_ts), 2),
        "completion_tokens_median": med(cptoks),
        "prompt_tokens_median": med(prtoks),
        "decode_tps_median": round(med(tps), 2),
        "decode_tps_mean": round(sum(tps) / len(tps), 2) if tps else None,
        "wall_s": round(time.perf_counter() - t_all, 1),
    }
    with open(f"{OUTD}/gsm8k_{args.tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("[gsm8k] SUMMARY " + json.dumps(summary), flush=True)

if __name__ == "__main__":
    main()
