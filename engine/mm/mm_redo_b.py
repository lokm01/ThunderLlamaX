#!/usr/bin/env python3
"""MM SESSION A -- FINAL PROBE WINDOW: l2bw (fixed: no in-kernel gridDim) ->
dsmem. eh + g0 + l1 already banked on disk."""
import os, sys, json, time

BASE = "~/tinygrad-metal"
MM = os.path.join(BASE, "engine0", "mm")
sys.path.insert(0, BASE); sys.path.insert(0, BASE + "/engine0"); sys.path.insert(0, MM)
os.environ.setdefault("DEV", "NV")

def log(s, lf):
    print(s, flush=True); lf.write(s + "\n"); lf.flush(); os.fsync(lf.fileno())

def load_json(p):
    try: return json.load(open(p))
    except Exception: return {}

def main():
    lf = open(os.path.expanduser("~/mm_a_session.log"), "a")
    t0 = time.time()
    log(f"[A-redoB] start {time.strftime('%H:%M:%S')}", lf)
    from MM_P7_lib import Rig7
    import mm_probes
    rig = Rig7(ctx_alloc=int(os.getenv("MM_CTXS", "98304")), load_p6=False)
    log(f"[A-redoB] rig up {(time.time()-t0)/60:.1f} min", lf)
    for name, fn in (("bw", lambda: mm_probes.run_l2bw(rig, load_json(mm_probes.OUT))),
                     ("ds", lambda: mm_probes.run_dsmem(rig, load_json(mm_probes.OUT)))):
        ts = time.time()
        try:
            fn(); log(f"[A-redoB] {name}: ok ({(time.time()-ts)/60:.1f} min)", lf)
        except Exception as e:
            import traceback
            log(f"[A-redoB] {name}: FAILED: {e}", lf)
            log(traceback.format_exc()[-1200:], lf)
    log(f"[A-redoB] done {(time.time()-t0)/60:.1f} min", lf)

if __name__ == "__main__":
    main()
