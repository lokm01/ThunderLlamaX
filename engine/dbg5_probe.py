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
E.restore_mtp(snap)          # spec-path seeding (kv padded 2304, rec4/conv4, slots)
ids5 = snap["ids"].reshape(-1).tolist()
sl5 = (ids5 * 40)[:40960]
E.init_draft(sl5)
print('[dbg5] SKIPPING fill_draft', flush=True)
E.build_graphs()
probe_seq = E.graphs[1].seq
P5 = E.P
_a0 = np.array([int(ids5[5])], dtype=np.int32); _a1 = np.array([int(ids5[6])], dtype=np.int32)
dev.allocator._copyin(P5.d["dring0"], memoryview(_a0.data).cast("B"))
dev.allocator._copyin(P5.d["dring1"], memoryview(_a1.data).cast("B"))
P5._keep += [_a0, _a1]
dev.synchronize()
print(f"[dbg5] probe seq {len(probe_seq)}k; dring pre-seeded; replaying eagerly", flush=True)
log = open("~/dbg5_trace.txt", "w")
n = -1
try:
    skip_spk = os.getenv("SKIP_SPK", "1") == "1"
    probe_seq = [e for e in probe_seq if not (skip_spk and getattr(e[0], chr(110)+chr(97)+chr(109)+chr(101), "").startswith("spk_"))]
    print(f"[dbg5] SKIP_SPK={skip_spk} -> {len(probe_seq)}k", flush=True)
    for n, (p, a, g) in enumerate(probe_seq):
        log.write(f"{n} {getattr(p, chr(110)+chr(97)+chr(109)+chr(101), chr(63))} grid={g}\n"); log.flush()
        p(*a, global_size=(g,1,1), local_size=LS, wait=True)
    print("[dbg5] probe replay OK", flush=True)
except Exception as ex:
    print(f"[dbg5] FAULT at {n}: {getattr(p, chr(110)+chr(97)+chr(109)+chr(101), chr(63))} grid={g}", flush=True)
finally:
    log.close()
