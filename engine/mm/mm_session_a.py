#!/usr/bin/env python3
"""MM MoE PREFILL CAMPAIGN -- SESSION A orchestrator (measurement only, NO
production changes).

THE GPU-EXIT/REBOOT LAW: this rig gets ONE GPU process per boot (a GPU
python exit while the dext is alive reboots the box). So the whole session
runs in THIS process, one Rig7 boot, stages in value order with per-stage
fsynced JSONs (resumable across the reboots a fault may provoke):

  g0 (attribution bisect) -> l1 (raster + swap + seat-loop POCs) ->
  ehist (real expert-M histogram) -> l2bw (L2/DRAM read BW) ->
  dsmem LAST (a dyn-smem fault can wedge the process; everything else is
  already on disk, and the pre-attempt fsync IS the fault marker).

Usage: python3 engine0/mm/mm_session_a.py [--no-dsmem]
"""
import os, sys, json, time

BASE = "~/tinygrad-metal"
MM = os.path.join(BASE, "engine0", "mm")
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/engine0")
sys.path.insert(0, MM)
os.environ.setdefault("DEV", "NV")

def log(s, lf=None):
    print(s, flush=True)
    if lf:
        lf.write(s + "\n"); lf.flush(); os.fsync(lf.fileno())

def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}

def main():
    no_dsmem = "--no-dsmem" in sys.argv
    lf = open(os.path.expanduser("~/mm_a_session.log"), "a")
    t0 = time.time()
    log("=" * 72, lf)
    log(f"[A] SESSION A start {time.strftime('%H:%M:%S')} (one process, one boot)", lf)

    from MM_P7_lib import Rig7
    import mm_pf_bisect, mm_l1_poc, mm_probes

    log("[A] booting Rig7 (the single session rig)...", lf)
    rig = Rig7(ctx_alloc=int(os.getenv("MM_CTXS", "98304")), load_p6=True)
    log(f"[A] rig up in {(time.time()-t0)/60:.1f} min", lf)

    for name, fn in (("g0", lambda: mm_pf_bisect.run_g0(load_json(mm_pf_bisect.OUT) or None, rig=rig)),
                     ("l1a", lambda: mm_l1_poc.run_l1(rig, load_json(mm_l1_poc.OUT))),
                     ("l1b", lambda: mm_l1_poc.run_seatloop(rig, load_json(mm_l1_poc.OUT))),
                     ("bw",  lambda: mm_probes.run_l2bw(rig, load_json(mm_probes.OUT))),
                     ("eh",  lambda: mm_probes.run_ehist(rig, load_json(mm_probes.OUT)))):
        ts = time.time()
        try:
            fn()
            log(f"[A] {name}: ok ({(time.time()-ts)/60:.1f} min)", lf)
        except Exception as e:
            import traceback
            log(f"[A] {name}: FAILED: {e}", lf)
            log(traceback.format_exc()[-2000:], lf)

    if not no_dsmem:
        ts = time.time()
        try:
            mm_probes.run_dsmem(rig, load_json(mm_probes.OUT))
            log(f"[A] dsmem: ok ({(time.time()-ts)/60:.1f} min)", lf)
        except Exception as e:
            import traceback
            log(f"[A] dsmem: FAILED (a hard fault here is itself the probe verdict): {e}", lf)
            log(traceback.format_exc()[-2000:], lf)

    log(f"[A] session done in {(time.time()-t0)/60:.1f} min; results in {MM}/*.json", lf)
    log("[A] NOTE: this process exit will trigger the GPU-EXIT reboot (expected).", lf)

if __name__ == "__main__":
    main()
