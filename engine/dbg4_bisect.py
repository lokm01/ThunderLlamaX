# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV"); os.environ.setdefault("SKV", "1")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from mtp import MTPEngine
from engine0 import dev
LS = (256,1,1)
snap = np.load("~/w1b_state_2k.npz")
E = MTPEngine(float(snap["theta"].reshape(-1)[0]))
E.restore(snap)
E._build_seqs()
seq = E._seq[0]
log = open("~/dbg4_trace.txt", "w")
try:
    for n, (p, a, g) in enumerate(seq):
        log.write(f"{n} {getattr(p, chr(110)+chr(97)+chr(109)+chr(101), chr(63))} grid={g}\n"); log.flush()
        p(*a, global_size=(g[0],1,1), local_size=LS, wait=True)
    print("[dbg4] full token OK", flush=True)
except Exception as ex:
    print(f"[dbg4] FAULT at kernel index {n}: {getattr(p, chr(110)+chr(97)+chr(109)+chr(101), chr(63))} grid={g} nargs={len(a)}", flush=True)
    print(f"[dbg4] {ex}", flush=True)
finally:
    log.close()
