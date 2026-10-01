#!/usr/bin/env python3
"""TLX P11-oblit pack_runner: run the offline packers for a model WITHOUT
touching the dext (engine0.py does `dev = Device["NV"]` at import — normally a
GPU grab; Device is replaced by a lazy stub BEFORE the import, so
parse_gguf/read_raw work from pure file I/O while any real dev use fails loud).

Env (required): TLX_MODEL_PATH (+ per-dir envs TLX_PACKED/TLX_PACKED7/
TLX_PACKED5/TLX_DRAFT_PACK). Usage:
  ~/tg311/bin/python -u ops/pack_runner.py <packer> [...]  (w1c|w7|w5|q4)
"""
import os, sys

assert os.environ.get("TLX_MODEL_PATH"), "TLX_MODEL_PATH must be set"

BASE = "~/tinygrad-metal/engine0"
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, BASE)

import tinygrad.device as _td

class _LazyDev:
    def __getattr__(self, n):
        raise RuntimeError("pack_runner no-GPU mode: dev.%s touched" % n)

class _DevMap:
    _opened_devices = ()   # tinygrad atexit reads this
    def __getitem__(self, k):
        print(f"[pack_runner] Device[{k!r}] stubbed (no dext init)", flush=True)
        return _LazyDev()

_td.Device = _DevMap()

import runpy

STEPS = {
    "w1c": "pack_w1c.py",
    "w7":  "pack_w7.py",
    "w5":  "pack_w5.py",
    "q4":  "q4pack.py",
}
failed = []
for step in sys.argv[1:]:
    script = STEPS[step]
    print(f"\n[pack_runner] ===== {step}: {script} =====", flush=True)
    sys.argv = [script]   # packers parse argv (tags/--check-only); isolate them
    try:
        runpy.run_path(os.path.join(BASE, script), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            failed.append((step, e.code))
            print(f"[pack_runner] {step} EXITED {e.code}", flush=True)
print("\n[pack_runner] ALL DONE" + (f" (failures: {failed})" if failed else ""), flush=True)
sys.exit(1 if failed else 0)
