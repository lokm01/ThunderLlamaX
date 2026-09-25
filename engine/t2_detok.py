# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Tier-2 battery helpers: detok of continuation tokens (the P7E7 coherence
convention). Reuses api_server's vendored tokenizer by exec'ing only its
head section (lines before the web imports) — zero GPU, zero web deps.

Usage:  from t2_detok import T2Tok ; T2Tok().decode([4471, 6545, ...])
  or:    python t2_detok.py file1.json ...   (json int arrays -> text)
"""
import sys, json

class T2Tok:
    def __init__(self):
        src = open("~/tinygrad-metal/engine0/api_server.py").read()
        head = src[:src.index("\nfrom fastapi")]
        g = {}
        exec(compile(head, "api_server_head", "exec"), g)
        r = g["load_tokenizer"]()          # (SimpleTokenizer, meta...) or the tok itself
        self.tk = r[0] if isinstance(r, tuple) else r
    def decode(self, toks):
        return self.tk.decode([int(t) for t in toks])

if __name__ == "__main__":
    tk = T2Tok()
    for fn in sys.argv[1:]:
        try:
            arr = json.load(open(fn))
            print(f"== {fn} ==")
            print(tk.decode(arr)[:2000])
        except Exception as e:
            print(f"{fn}: {e!r}")
