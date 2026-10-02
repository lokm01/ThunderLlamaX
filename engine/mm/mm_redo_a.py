#!/usr/bin/env python3
"""MM SESSION A -- REDO WINDOW (minimal, value-ordered): eh -> l2bw -> dsmem.

The wedge forensics: session 2's cp4k 1-tuple grid (FIXED: scalar) + session
3's boot-onto-a-wedged-dext. This runner does the three remaining probes in
VALUE order with the safest first, one Rig7 boot, per-stage fsync.
"""
import os, sys, json, time

BASE = "~/tinygrad-metal"
MM = os.path.join(BASE, "engine0", "mm")
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, MM)
os.environ.setdefault("DEV", "NV")

def log(s, lf):
    print(s, flush=True); lf.write(s + "\n"); lf.flush(); os.fsync(lf.fileno())

def load_json(p):
    try: return json.load(open(p))
    except Exception: return {}

def main():
    lf = open(os.path.expanduser("~/mm_a_session.log"), "a")
    t0 = time.time()
    log("=" * 72, lf)
    log(f"[A-redo] start {time.strftime('%H:%M:%S')}", lf)
    from MM_P7_lib import Rig7
    import mm_probes
    log("[A-redo] booting Rig7...", lf)
    rig = Rig7(ctx_alloc=int(os.getenv("MM_CTXS", "98304")), load_p6=True)
    log(f"[A-redo] rig up {(time.time()-t0)/60:.1f} min", lf)
    for name, fn in (("eh", lambda: mm_probes.run_ehist(rig, load_json(mm_probes.OUT))),
                     ("bw", lambda: mm_probes.run_l2bw(rig, load_json(mm_probes.OUT))),
                     ("ds", lambda: mm_probes.run_dsmem(rig, load_json(mm_probes.OUT)))):
        ts = time.time()
        try:
            fn(); log(f"[A-redo] {name}: ok ({(time.time()-ts)/60:.1f} min)", lf)
        except Exception as e:
            import traceback
            log(f"[A-redo] {name}: FAILED: {e}", lf)
            log(traceback.format_exc()[-1500:], lf)
    log(f"[A-redo] done {(time.time()-t0)/60:.1f} min", lf)

if __name__ == "__main__":
    main()
