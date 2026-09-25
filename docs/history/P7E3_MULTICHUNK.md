# P7-E3 — Multi-chunk SC: EXONERATED in the clean world; gate world blocked by a NEW machine fault class

Status: **the P7E2 "multi-chunk SC divergence (chunk>=2)" does NOT reproduce in the
clean world at ANY scale tested: 2 chunks and 30 chunks are EXACT vs the M32
ground truth (logits medrel 0/5.9e-4, argmax identical, rec/conv/kv fp16-class,
all stages fp16-class), including the SC-AFTER-M32 order. The gate-world failure
(GATE A 0/60, F 5.07e-1) could NOT be re-verified because the machine now has a
DETERMINISTIC deep-world device-fault class (T=1 decode @97810 AND the boot
fill_draft @97810) that survives host reboots. Service relaunched on the P6
canonical (PF_SUPER=0). PF_G3SC split env landed.**

## The clean-world evidence (pf_scdbg5.py / pf_scdbg7.py, rng(99) ids)

Phases: M32 truth vs SC, hard-zero reset (kv+conv_0+conv_1+rec, pos_slot=0),
env SKV=1 KV8=1 QH=1 SKV_CTXK=100352 PF_GEMM3=0 PF_SCANC=1 PF_DFILL=0.

- N=512 (2 chunks): S1 (chunk 1 carry-out @256) EXACT vs M32@256; S2 (2 chunks)
  EXACT vs M32@512 — rec medrel 4.8e-4..2.3e-3, conv <=1.6e-2 maxabs (fp16),
  logits medrel 5.9e-4, kv int8 +-1 quant-noise only. No fault.
- N=7680 (30 chunks, AFTER a full M32 pass in the same process — the P7E2
  "fault inside chunk 2" config): 34.2s wall, logits medrel 0.0/maxabs 1.1e-2,
  top1/top2 identical (62@2.83 / 220@2.19), rec/conv fp16-class. NO fault.
- Bench in the same runs: chunk-1 843-862ms (~300 tok/s pos-0 class), 30-chunk
  plateau ~1200ms; 30 chunks/7680 tok = 34.2s = 224.6 tok/s vs M32 40.3s =
  190.6 tok/s in-world (1.18x) — and STATE-EXACT (P7E2's 222.7 was state-wrong).

## What was verified correct along the way (dead suspects, do not re-chase)

1. pfk_pre64 chunk-2 addressing: pos = pos_arr[ywin] absolute; RoPE freqs x
   (pos+t); kv/sc appends at (pos+t); ALL row reads/writes use g=ywin*TROWS+t.
2. The c64_nc4 state carry: pfca reads conv{i}_0 for the window-0 halo
   (c0+i<3 -> convlive[i+0..2]); pfcz writes conv{i}_0 from qkvsc rows
   [NC*C-3..NC*C-1] at c==NC-1; pfcb chains rec in place with per-window LOCAL
   g (S' = 2^g_end * S + KhatT(sg.d)) — no cross-chunk g leak; oout rows are
   chunk-global (c*C+i0). Conv halo/decay/pos all correct.
3. The SC dfill ring across chunk boundaries (pfk_rec16 semantics:
   rec[1..16]=xbuf rows, recseed[0]=xbuf row15): after chunk c's k=15,
   REC0[0]=h_255 naturally; chunk c+1's k=0 re-records REC0[1..16] from the new
   trunk rows before dnorm reads -> token j gets h_{j-1} correctly. The extra
   "seed" call (REC0->REC1) is redundant but harmless.
4. gemm3-m64@M256 ARITHMETIC (static): XPR loads use mb*MTILE+m AND the
   epilogue m0 = mb*MTILE + rg*16 + g — no missing global base (NOT the
   pfk_pre64 bug-1 class). With P7E2's re-upload test (mapping intact) the
   wrong-VA write stays a stale-record/adjacency class -> PARKED.

## The NEW blocker: deterministic deep-world fault (machine class)

Symptom: `test_w100k.py` faults ("Device fault detected") at
`G.run_tokens(NTOK, wait_each=True)` — the T=1 reference decode @97810 —
deterministically across 3 attempts (2 pre-reboot, 1 post-REBOOT), with the
exact P7E2 gate env, and with both kernargs geometries (MTP_KERNARGS_MB=64
emulating the pre-knob ring, and =256). A cold HOST reboot did NOT clear it.
With DO_T1=0 (cached ref + G built without the decode — workaround landed in
test_w100k.py) the flow advances to the boot `E.fill_draft(ids)` @97810 and
faults there too (before the first 2000-position progress line).

Interpretation: the full w100k world's deep-position ops (split-KV T1 attention
reading ~3.3GB int8 KV + the 97k-position draft fill) hit a fault region that
appeared AFTER the P7E2 gate completed (Sep-17 10:22) — i.e. after the late-
P7E2 device faults (N=512 fault runs) — and survives host reboots (the TB dock
stays powered across them). The serving daemon does NOT run these ops at boot
(park_pos attach; dozens of clean boots in logs/m1a_serve.log incl. one
device_fault at uptime 167.5s that self-exited per the GPU-EXIT law).
=> NEXT SESSION: PHYSICAL cold cycle (dock power-off) first; then rerun the
staged gate5 flow (the cached-ref workaround + the SCDBG_GATE_B phase are
already in the tree).

## The staged gate experiment (ready to run when the machine is healthy)

pf_gate2k.py now has (env-gated, SCDBG_GATE_B=1): after A2 (SC prefill with
PF_DFILL=1), a B phase reruns the SC prefill with PF_DFILL=0 IN THE SAME WORLD
(+ SCDBG_GATE_DUMP=1 dumps A1/A2/B end-state rec/conv/logits to
~/gate_{a1,a2,b}_state.npz). Since the clean world is exact, B-vs-A2 isolates
the dfill/adjacency interaction inside the gate world. NOTE: A1 (the T=1
prefill over 7714) uses E.prefill_t1 — that path is 7.7k-deep, likely safe
even under the current fault class; the blocker is only the 97k boot ops.

## GEMM3-m64@M256 (bug 2) — parked with the split env

- PF_G3SC landed (pf_prefill.py): SC-side gemm3 control, defaults to PF_GEMM3.
  Serving recipe keeps PF_GEMM3=1 (M32 path, P7B-gated) + PF_G3SC=0 for SC.
- Cost of the classic-m32 fallback on the SC side: from the P7E2 attribution,
  ~150ms per 256-tok chunk (858ms pos-0 -> ~710 if fixed = ~17% prefill time
  at pos-0; ~12.5% at the 1.2s plateau). The m64-vs-4xm32 validation (P7E TODO)
  plus a PF_SC_STEPS launch-sequence log around the chunk-2 launch-226 window
  remain the next forensic steps.

## Benchmark table (P7E3)

| class | P6 record (shipped) | SC P7E3 (clean world, state-EXACT) | note |
|---|---|---|---|
| 2k-class pos-0 chunk | 245.0 tok/s | 297-303 tok/s (843-862ms/256) | exactness now at 30 chunks too |
| 8k-class (30 chunks, r=0) | 190.6 tok/s in-world (40.3s/7680) | **224.6 tok/s (34.2s/7680)** | 1.18x, state EXACT vs M32 |
| 100k rebuild | 165.3 tok/s | not run (deep-world fault class) | projection ~200-230 unchanged |

## Service state

Daemon relaunched on the P6 canonical (PF_SUPER=0 — SC stays default-OFF until
the serving gates can actually run): M1C recipe + PF_PREFILL=1 PF_GEMM3=1
PF_SCANC=1 PF_SUPER=0 PF_G3SC=0 MTP_KERNARGS_MB=256 + api_server on 8080.

## P7f attention implications (unchanged from P7E2, now on firmer ground)

Non-attention ~800ms/chunk at 100k (G3SC=0). 300 tok/s needs attention
<=50-200ms/chunk => IMMA/dp4a int8-QK required; m64-at-M256 (if fixed) buys
~150ms/chunk free. With multi-chunk correctness no longer in doubt (clean
world), the SC integration risk is concentrated in (a) the machine fault
class (physical cold cycle) and (b) the gate-world dfill/adjacency bisection
(the staged B-phase).

## Files

- engine0/pf_scdbg5.py (2-chunk carry + stage diff), pf_scdbg7.py (30-chunk
  scale ladder + post-M32 order), pf_prefill.py (PF_G3SC), pf_gate2k.py
  (SCDBG_GATE_B/DUMP phases), test_w100k.py (DO_T1=0 cached-ref G-build
  workaround), this doc. Logs: ~/p7e3_sd5.log, ~/p7e3_sd7*.log,
  ~/p7e3_gate{,2,3,4,5}.log. Data: ~/scdbg5.npz, ~/scdbg7.npz.
