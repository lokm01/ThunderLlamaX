lineage/ — the pre-engine tinygrad-stack speculative-decode work (HISTORICAL)
==============================================================================

This directory preserves the campaign that preceded the hand-kernelled engine
(engine/): mtp_v3.py and friends built exact MTP speculative decoding INSIDE the
tinygrad scheduler (graph-captured TinyJit families, a3-family kernel
substitutions), reaching 7.28 tok/s @100k Tier-1 exact (run39) and providing every
correctness contract the engine inherited (the emission/accept semantics, PROBE_RO,
the 60-token gates, the baselines/ files).

Contents
--------
- mtp_v3.py, mtp_config.py, mtp_spec.py, spec_test.py, gen_base*.py, head_i8.py,
  oneblock*.py, fwd3_split.py, chain_test.py, repro_*.py, probe_fault_repro.py,
  iso_*.py, t3_*.py, mtpv3_*.py ...  — the tinygrad-stack MTP pipeline and its
  bisection/isolation harnesses.
- a3a/ a3b/ a3d/ a4/ p1c/ p1d/ pv3/ splitkv/ — kernel research: hand dequant-GEMV
  microbenches, the 1:1 kernel-substitution machinery (a3b override*.json — the
  "template" paths are relative to THIS directory), the GDN scan megakernels (a4),
  attention/PV decode work (pv3, splitkv).
- bench_*.py, kern_hist*.py, diag_*.py, gemv_*.py ... — the measurement harness
  that drove attribution (see docs/history/PERFLOG.md).
- run_*.sh, chain100k.sh — the original rig launch recipes (paths generalized to
  placeholders; see docs/SETUP.md).

Requirements
------------
This code is NOT standalone: it requires the tinygrad FORK (patches/
tinygrad-fork.patch — env-gated MTP_* hooks, graph submit path, kernargs fix) and
the TinyGPU DriverKit dext. It is preserved for reproducibility and for the
negative results banked in its logs; the live engine is engine/.

NOTE: scripts reference `~/tinygrad-metal`, `~/tinygrad-src`, `~/tg311/bin/python`
and `unix://<colima-socket>` — placeholders for the original rig layout; replace
per docs/SETUP.md. The a3b override*.json template paths are relative to this
directory (run from here).
