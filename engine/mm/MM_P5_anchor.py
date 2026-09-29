#!/usr/bin/env python3
"""MM P5 — the anchor runner: A16 (P5.1: 20-prompt engine-order anchor with
fp16 MoE partials) and A60 (P5.2: the 60-prompt Tier-1 battery anchor).

Usage: MM_P5_anchor.py A16|A60 [fp16|f32]
  A16 -> ~/mm_p5_anchor16.npz  (20 prompts, PROMPTS from MM_P34_anchor)
  A60 -> ~/mm_p5_anchor60.npz  (60 prompts = the 20 + 40 new diverse)
Resumable via ~/mm_p5_anchor_part.pkl (per-mode parts: a16_/a60_ prefixes).
"""
import os, sys, time, pickle
import numpy as np

sys.path.insert(0, "~/tinygrad-metal")
sys.path.insert(0, "~/tinygrad-metal/engine0")
os.environ.setdefault("DEV", "CPU")

from MM_P34_ports import Anchor, fresh_state
from MM_P34_anchor import PROMPTS as PROMPTS20
from MM_P34_tok import parse_gguf_kv, SimpleTokenizer
GGUF = os.path.expanduser("~/models36/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")

PROMPTS40 = [
    "The largest planet in our solar system is",
    "function add(a, b) {\n    return",
    "Romeo and Juliet was written by William",
    "The chemical formula for table salt is",
    "To convert Celsius to Fahrenheit, multiply by 9/5 and then add",
    "The Amazon River flows through the rainforest of",
    "import numpy as np\narr = np.array([1, 2, 3, 4])\nmean =",
    "In machine learning, overfitting occurs when a model",
    "The Magna Carta was signed in England in the year",
    "Bonjour, je m'appelle Marie et je habite à",
    "The speed of light in a vacuum is approximately 299",
    "A red blood cell's primary function is to carry",
    "SELECT * FROM users WHERE age > 25 ORDER BY",
    "The Pythagoreans believed that numbers were the fundamental substance of",
    "Gravity on the Moon is about one sixth of gravity on",
    "Ein guter Freund ist jemand, der",
    "The Great Depression began with the stock market crash of October",
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid =",
    "Photosynthesis converts carbon dioxide and water into glucose using energy from",
    "The capital of Japan is Tokyo, and the capital of South Korea is",
    "World War II ended in Europe on May 8, 1945, when",
    "The freezing point of water in Fahrenheit is 32 degrees, and in Kelvin it is",
    "Los ríos más largos del mundo incluyen el Amazonas y",
    "A hash table provides average-case constant time for lookup by using",
    "The human heart has four chambers: the left and right atria and the",
    "UPDATE employees SET salary = salary * 1.05 WHERE department =",
    "The Mona Lisa was painted by Leonardo da Vinci and is displayed in the Louvre museum in",
    "Newton's second law of motion states that force equals mass times",
    "La Torre Eiffel fue construida en 1889 para la Exposición Universal de",
    "The stock market index that tracks 30 large American companies is called the Dow Jones",
    "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.next =",
    "Mitochondrial DNA is inherited exclusively from the",
    "The Antarctic continent is surrounded by the Southern Ocean and covered by ice that averages about",
    "TheRecursion in computer science is a method of solving problems where a function",
    "Shakespeare wrote 37 plays and 154 sonnets, including the famous tragedy about the Prince of Denmark called",
    "The distance from the Earth to the Sun is about 150 million kilometers, also known as one",
    "DNA replication is semiconservative, meaning each new molecule contains one original strand and one",
    "The tallest mountain in the world above sea level is Mount Everest, located in the",
    "INSERT INTO orders (customer_id, product, quantity) VALUES",
    "The internet protocol suite consists of the application layer, the transport layer, and the",
]

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "A16"
    fp16 = (sys.argv[2] if len(sys.argv) > 2 else "fp16") == "fp16"
    prompts = PROMPTS20 if mode == "A16" else PROMPTS20 + PROMPTS40
    out = os.path.expanduser("~/mm_p5_anchor16.npz" if mode == "A16" else "~/mm_p5_anchor60.npz")
    part = os.path.expanduser(f"~/mm_p5_anchor_{mode.lower()}_part.pkl")
    t0 = time.time()
    kv = parse_gguf_kv(GGUF)
    tok = SimpleTokenizer.from_gguf_kv(kv)
    print(f"[anchor {mode}] tokenizer loaded eos={tok.eos_id} eot={tok.eot_id}", flush=True)
    battery = []
    for pr in prompts:
        ids = tok.encode(pr)
        # roundtrip verification on the battery itself
        rt = tok.decode(ids)
        assert rt == pr, f"roundtrip FAIL: {pr[:40]!r} -> {rt[:40]!r}"
        battery.append((pr, ids))
    print(f"[anchor {mode}] {len(battery)} prompts encoded+roundtrip OK, fp16_partials={fp16}", flush=True)
    an = Anchor(ctx=1024, fp16_partials=fp16)
    top1s, gaps, top5s = [], [], []
    if os.path.exists(part):
        top1s, gaps, top5s = pickle.load(open(part, "rb"))
        print(f"[anchor {mode}] resuming from {len(top1s)} prompts", flush=True)
    elif mode == "A60" and os.path.exists(os.path.expanduser("~/mm_p5_anchor16.npz")):
        a16 = np.load(os.path.expanduser("~/mm_p5_anchor16.npz"), allow_pickle=True)
        top1s = [np.asarray(x, dtype=np.int32) for x in a16["top1"]]
        gaps = [np.asarray(x, dtype=np.float32) for x in a16["gap"]]
        top5s = [np.asarray(x, dtype=np.int32) for x in a16["top5"]]
        assert len(top1s) == 20 and all(
            [int(x) for x in i] == [int(x) for x in j]
            for i, j in zip(a16["ids"][:20], [b[1] for b in battery[:20]])), "A16/A60 battery mismatch"
        print(f"[anchor {mode}] SEEDED from A16 (20 prompts)", flush=True)
    for pi, (pr, ids) in enumerate(battery):
        if pi < len(top1s): continue
        S, convst, KV = fresh_state(1024)
        pt1, pgap, ptop5 = [], [], []
        for pos, tid in enumerate(ids):
            t1, logits, _ = an.forward_token(tid, pos, S, convst, KV, want_logits=True)
            idx5 = np.argsort(-logits)[:5]
            pt1.append(int(t1)); ptop5.append(idx5.astype(np.int32))
            srt = np.sort(logits)[::-1]
            pgap.append(float(srt[0] - srt[1]))
        top1s.append(np.array(pt1, dtype=np.int32))
        top5s.append(np.stack(ptop5))
        gaps.append(np.array(pgap, dtype=np.float32))
        print(f"[anchor {mode}] prompt {pi}: {len(ids)} pos ({time.time()-t0:.0f}s) last={pt1[-1]} gap={pgap[-1]:.4f}", flush=True)
        pickle.dump((top1s, gaps, top5s), open(part, "wb"))
    np.savez_compressed(out,
        prompts=np.array([p for p, _ in battery], dtype=object),
        npos=np.array([len(i) for _, i in battery], dtype=np.int32),
        ids=np.array([np.array(i, dtype=np.int32) for _, i in battery], dtype=object),
        top1=np.array(top1s, dtype=object), top5=np.array(top5s, dtype=object),
        gap=np.array(gaps, dtype=object), fp16=np.array(fp16), allow_pickle=True)
    near = sum(int((g < 1e-3).sum()) for g in gaps); near2 = sum(int((g < 1e-2).sum()) for g in gaps)
    tot = sum(len(g) for g in gaps)
    print(f"[anchor {mode}] DONE {time.time()-t0:.0f}s -> {out}; near-tie <1e-3: {near}/{tot} <1e-2: {near2}/{tot}", flush=True)

if __name__ == "__main__":
    main()
