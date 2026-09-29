#!/usr/bin/env python3
"""MM P3+P4 — the fp32 full-model ANCHOR run (CPU-only; no GPU touch).

Builds the 20-prompt battery from the GGUF tokenizer, runs every prompt
through the Anchor reference token-by-token (fresh GDN/conv/KV state per
prompt), and records per-position: top-1, top-5, the top1-top2 logit gap
(near-tie flag), + per-layer h traces. Saves ~/mm_p34_anchor.npz.

Run: ~/tg311/bin/python MM_P34_anchor.py  (~10-20 min CPU)
"""
import os, sys, time, hashlib
import numpy as np

sys.path.insert(0, "~/tinygrad-metal")
sys.path.insert(0, "~/tinygrad-metal/engine0")
os.environ.setdefault("DEV", "CPU")

from MM_P34_ports import Anchor, fresh_state, ATTN_LAYERS
from MM_P34_tok import parse_gguf_kv, SimpleTokenizer
GGUF = os.path.expanduser("~/models36/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")
OUT = os.path.expanduser("~/mm_p34_anchor.npz")

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote valley. The unicorns",
    "Water boils at 100 degrees Celsius at sea level because",
    "SELECT name, COUNT(*) FROM orders GROUP BY",
    "The three laws of robotics were formulated by",
    "Translate to French: The weather is beautiful today.",
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
    "The complexity of quicksort in the average case is O(n log n), but in the worst case it degrades to",
    "Dear hiring manager,\n\nI am writing to apply for the position of",
    "The mitochondria is the powerhouse of",
    "2 + 2 = 4. 4 + 4 = 8. 8 + 8 = 16. 16 + 16 =",
    "One of the most important discoveries in physics was general relativity, published by Albert",
    "The Great Wall of China was built over many centuries, starting in approximately",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
    "La capitale dell'Italia e la città più grande è",
    "The process by which plants convert sunlight into chemical energy is called photosynthesis, which occurs in the",
    "HTTP status code 404 means the server",
    "print(\"Hello, world!\") is traditionally the first program written in a new language, originating from",
    "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals",
]

def main():
    t0 = time.time()
    kv = parse_gguf_kv(GGUF)
    tok = SimpleTokenizer.from_gguf_kv(kv)
    print("[anchor] tokenizer loaded", flush=True)
    battery = []
    for pr in PROMPTS:
        ids = tok.encode(pr)
        battery.append((pr, ids))
        print(f"  [{len(battery)-1:2d}] {len(ids):3d} tok: {ids[:10]}...", flush=True)

    an = Anchor(ctx=1024)
    top1s, gaps, top5s, traces = [], [], [], []
    PART = os.path.expanduser("~/mm_p34_anchor_part.pkl")
    import pickle
    if os.path.exists(PART):
        top1s, gaps, top5s, traces = pickle.load(open(PART, "rb"))
        print(f"[anchor] resuming from {len(top1s)} prompts", flush=True)
    for pi, (pr, ids) in enumerate(battery):
        if pi < len(top1s): continue
        S, convst, KV = fresh_state(1024)
        pt1, pgap, ptop5 = [], [], []
        ptr = []
        for pos, tid in enumerate(ids):
            tr = [] if pos == len(ids) - 1 else None
            t1, logits, _ = an.forward_token(tid, pos, S, convst, KV, want_logits=True, trace=tr)
            if tr is not None: ptr.append(np.stack(tr))          # [42][2048]
            idx5 = np.argsort(-logits)[:5]
            pt1.append(int(t1))
            ptop5.append(idx5.astype(np.int32))
            srt = np.sort(logits)[::-1]
            pgap.append(float(srt[0] - srt[1]))
        top1s.append(np.array(pt1, dtype=np.int32))
        top5s.append(np.stack(ptop5))
        gaps.append(np.array(pgap, dtype=np.float32))
        traces.append(ptr[0])                                     # last position only
        print(f"[anchor] prompt {pi}: {len(ids)} pos done ({time.time()-t0:.0f}s) last top1={pt1[-1]} gap={pgap[-1]:.4f}", flush=True)
        pickle.dump((top1s, gaps, top5s, traces), open(PART, "wb"))

    np.savez_compressed(OUT,
        prompts=np.array([p for p, _ in battery], dtype=object),
        npos=np.array([len(i) for _, i in battery], dtype=np.int32),
        ids=np.array([np.array(i, dtype=np.int32) for _, i in battery], dtype=object),
        top1=np.array(top1s, dtype=object),
        top5=np.array(top5s, dtype=object),
        gap=np.array(gaps, dtype=object),
        htrace=np.array(traces, dtype=object),                    # [20][42][2048] fp32
        allow_pickle=True)
    near = sum(int((g < 1e-3).sum()) for g in gaps)
    tot = sum(len(g) for g in gaps)
    print(f"[anchor] DONE {time.time()-t0:.0f}s -> {OUT}; near-tie positions (<1e-3): {near}/{tot}", flush=True)

if __name__ == "__main__":
    main()
