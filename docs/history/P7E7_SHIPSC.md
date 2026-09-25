# P7-E7 — The SC tolerance-gate trial: VERDICT = DO NOT SHIP (8k real-text decode degenerates; 2k exact; SC deterministic + 1.14-1.16x faster); dp4a no-op ROOT-CAUSED (gridDim law) + working; G3SC per-class knob + 4 clean m64 twins VALIDATED in-plan (−54 ms/chunk)

Status: **the tolerance gate was tried per the orchestrator's product decision and
FAILED on its own criterion. The full trial protocol ran on the canonical 8k-class
real-text world (ids8k): SC-prefill decode = " gravity gravity gravity ..." ×60
(single-token degenerate attractor, gap 7.0) while the T1 reference and the M32
path both produce the coherent physics-term cycle the document calls for
(" charge phase momentum gauge energy ..."). F(SC vs T1) = 5.11e-1 vs F(M32 vs
T1) = 4.26e-2 (12x); 0/60 vs 12/60 agreement; at 2k BOTH are exact-class
(F≈1.06e-3, 60/60) — the divergence is document-length/maturity-driven exactly as
the P7E6 amplifier law predicts. SC is bit-deterministic across processes and
decode α/speed are unchanged (2.67→2.97 tok/cyc after SC prefill) — but the state
is decorrelated: 100k rebuild cur=1428 vs 4471, rec relerr 0.54, conv relerr 1.0.
PF_SUPER stays 0; P6 canonical (245.0 @2k / 190.8 @8k / 165.3 @100k) STANDS and
the service was relaunched on it (verified end-to-end incl. an 8k FRESH API chat
+ FOLLOW_UP). Consolation prizes banked: (1) the dp4a no-op was the gridDim.x
law all along — now working, 1.0-1.1 TOPS probe-class; (2) PF_G3SC is now a
per-class list knob and the four P7E4-clean m64 twins are IN-PLAN BIT-EXACT at
−54 ms/chunk (16-block coverage); (3) classsync attribution: the SC chunk is
~80% LAUNCH-BOUND (258 ms of kernel vs ~1240 ms wall at pos 48k) — launch-count
reduction, not kernel math, is the true P7f lever.**

## 1. The quality trial (mission step 1 — the decisive evidence)

Harness: pf_gate2k inside test_w100k (HOST-PROCESS BOOT LAW), FULL env (see §6
env law), NTOK=60 greedy, PF_DFILL=0, ids8k.json (7714-tok real text
"Foundations of modern physics..." with natural document end). PF_TRUNC=2048
for the 2k arm (new env, ids8k head). The gate's early-exit was lifted
(SCDBG_EARLY_EXIT restores the old stop; SCDBG_SKIP_CD shortens).

| run | path | F vs T1 (end logits) | agree vs T1 | decode continuation (detok) |
|---|---|---|---|---|
| 8k SC (run1) | PF_SUPER=1 | **5.109e-01** | **0/60** | " gravity gravity gravity ..." ×60 (1-tok loop, top1 gap 7.0) |
| 8k SC rerun (run3) | PF_SUPER=1 | **5.109e-01** | 0/60 | **bit-identical to run1** (determinism PASS) |
| 8k M32 (run2) | PF_SUPER=0 | 4.261e-02 | 12/60 | " momentum gauge energy gauge phase ..." (coherent term cycle) |
| 2k SC (run4) | PF_SUPER=1 | 1.059e-03 | **60/60** | coherent cycle (== T1) |
| 2k M32 (run5) | PF_SUPER=0 | 1.058e-03 | (tie-assert in gapped decode; F clean) | — |

- T1 reference continuation (tokA): " charge phase momentum gauge energy phase
  charge phase ..." — the document is a physics term list; greedy T1 cycles
  over domain terms. M32's cycle is the SAME cycle phase-shifted (12/60 =
  the known tie-mine class of the gapped-argsort decoder). SC's state collapses
  to a confident single-token attractor ("gravity", gap 7.0) — NOT a coherent
  continuation. **Tolerance-gate criterion ("both outputs coherent") FAILS.**
- Decode path unchanged (criterion c): CTRL α 2.67 tok/cyc (both runs);
  spec-after-prefill α 2.83 (M32) / 2.97 (SC); 60-cycle spec decode 3.3 s in
  every arm. GATE D on the SC state = 60/60 (Tier-1 spec≡greedy on the SAME
  state holds even on the corrupted SC state).
- The 22525-loop is the SAME deterministic signature as P7E2/P7E4 (bit-identical
  F 5.067e-1 → 5.109e-1): the P7E5 hi-lo + P7E6 state-channel fixes did NOT
  move the 8k real-text end-state — the amplifier channel owns it, compounding
  over ~30 chunks x 48 blocks of mature content. 2k (8 chunks) stays clean.
- 100k rebuild (pf_gate100k, PF_SUPER=1): 517.8 s = **188.9 tok/s** (P6 M32:
  165.3 = 1.14x) BUT rebuilt cur=1428 (snapshot 4471 — MISMATCH), state gates
  rec relerr 5.4e-1 / conv relerr 1.0e+0, decode 0/60. At full document length
  the SC mature state is decorrelated, exactly as the 8k trial predicts.

## 2. The ship decision (mission step 2) — NOT TAKEN, per the trial contract

The mission contract: "If both outputs are coherent continuations of the
prompt, the tolerance gate passes." It does not pass at 8k — the serving-
relevant class — while passing trivially at 2k where SC is Tier-1-exact anyway.
**PF_SUPER=0 PF_G3SC=0 remains the serving default.** The daemon was relaunched
on the unchanged P6 canonical recipe (§7) and verified: health ok, 8k-class
FRESH API chat (prefill progress events watched, M32 path) + one FOLLOW_UP
delta-prefill turn (§7 numbers). THE GATE CONTRACT, as evaluated: Tier-1
(spec≡greedy on the same state) UNCHANGED and re-verified (60/60 on the SC
state; 60/60-class on M32 per P6); prefill cross-path agreement is NOT a
bit-gate and NOT even a tolerance gate at >=8k on real text — the reassociation
floor (P7E6: 1-2 fp16 ULP z flips at block 0, ~1.25x/block trunk amplification
on mature streams) compounds to state decorrelation by ~8k tokens. Any future
SC ship requires either sub-ULP z agreement (the P7E6 §4 leads) or a product
acceptance of degenerate continuations — the latter is now MEASURED, not
hypothetical: the 8k-class customer gets a 1-token loop.

## 3. The benchmark ladder + per-stage attribution (mission step 3)

Fresh-prefill wall (harness, PF_DFILL=0, fill_draft billed separately):

| length | P6 M32 (shipped) | P7E7 SC (PF_G3SC=0) | SC + 4 clean m64 twins (16 blk) |
|---|---|---|---|
| 2k fresh | 245.0 tok/s (P6 banked) | 252.9 tok/s (8.1 s) | 272.5 tok/s (7.5 s, exact) |
| 8k fresh | 190.9 tok/s (40.4 s; P6 daemon 190.8) | 221.2 tok/s (34.9 s) | — |
| 100k rebuild | 165.3 tok/s (P6 banked) | **188.9 tok/s** (517.8 s) | — |

100k chunk-time ladder (SC): 898 ms @pos0 → ~1237 ms @pos 10-40k → 1597 ms
@pos 97k (attention growth). fill_draft 100k: 257 s = 380.5 tok/s.

**Per-stage attribution (new instrument: PF_SC_CLASSSYNC=k — class-boundary
synced timing of chunk k; syncs inflate the wall, kernel times are clean).**
Chunk @pos 48640, wall 1342 ms (1237 ms clean-classsync-free neighbor):

| class | launches | ms |
|---|---|---|
| pfg-m32 classic GEMM twins (G3SC off) | 1536 | 182.1 |
| pfg2-m32 merged twins | 512 | 42.6 |
| pfa16 + pfc16 attention pair | 256+256 | 21.4 |
| chunked scan pfca/pfcb/pfcz (P7C) | 48x3 | 6.0 |
| norms + small (hh16/ab16/n16/pre64/emb16) | 145 | 6.3 |
| **launch/sync overhead (wall − kernels)** | — | **~980 (79%)** |

=> The 256-tok SC chunk is ~80% LAUNCH-BOUND (~2849 launches x ~0.34 ms
enqueue+spacing vs 258 ms of actual kernel work). Kernel-side heroes are already
tiny (scan 6 ms!). The true P7f prefill lever is LAUNCH-COUNT REDUCTION: the
m64 twins (8x fewer launches per family) and/or persistent CTAs. This reframes
the P7E4 "+50-55 ms/chunk" m64 estimate — the measured −54 ms/chunk at 16-block
coverage is mostly launch-count, so full-coverage twins (VRAM permitting) and
attn twins (still quarantined) are worth far more than their kernel-time deltas.

## 4. P7f groundwork (mission step 4)

(a) **dp4a no-op ROOT-CAUSED AND FIXED** (pf_dp4a.cu): the kernel computed
`nseg = gridDim.x * 8 / npairs` — gridDim.x READS 0 on the dext (the documented
hardcode law) => nseg = 0 => empty per-warp L-ranges => every launch silently
wrote nothing. nseg is architecturally 8 (NSEG) — hardcoded, no bisect needed
once seen. Rebuilt (54 regs, 0 spill) and run on real KV (kv_3.npy, engine kv8
layout): **relerr med 1.03e-03 / F95 6.6e-03 (the expected q-int8+fp16-out
class), T=16: 3.457 ms full-K pass = 40.9 GB/s K-read, 1.0 TOPS dp4a; T=64:
11.9 ms, 18.4 GB/s, 1.1 TOPS.** Per-32-row equivalent 5.9-6.9 ms vs the shipped
pfa16 pair 5.85 ms: the probe shape (lane-per-position, no staging) is PARITY,
not a win — the P7f IMMA attention kernel needs smem staging / mma structure to
beat the pair; dp4a itself is DE-RISKED (instruction works and validates on the
dext).
(b) **PF_G3SC is now a per-class comma-list knob** (pf_prefill.py): "1" = all
(legacy), "0" = none, e.g. "ffn,iq3d,iq3o,gdnqg". Weights + cubins load only
for enabled classes; ffn and iq3d became independently selectable (producer
buffers compose: m64 whole-M writes vs classic per-32 slices cover identical
bytes). **In-plan validation of the four P7E4-clean twins PASSED**: 2k SC world
with PF_G3SC=ffn,iq3d,iq3o,gdnqg (G3SCN=16) — end logits IDENTICAL to the
G3SC=0 baseline (top1 29877 / 15.438 / gap 0.0703; F 1.059e-03; GATE A 60/60;
tokB identical), launches 2849→2513, chunk ms 887/925/1055/982/1011/1039/1068/
1097 → 828/859/991/916/944/974/1002/1030 (**mean 1009→955 = −54 ms/chunk**,
matching the P7E4 estimate at 16-block coverage). The attnqkvi3/attnqkvq6 m64
twins stay quarantined (P7E4 in-plan corrupt).

## 5. What would actually ship SC (for the next session)

- The blocker is ONLY the amplifier channel (P7E6 §4): candidates (i) >=24-bit
  o/z path for the cancellation heads (fp64 or deeper splits at the z
  quantization boundary), (ii) bit-exact adoption of M32's z-rounding
  semantics, (iii) hybrid: SC for the first <=2-4k tokens (EXACT-class today),
  M32 beyond (chunk-count-dependent switch is trivial in prefill_batch_sc's
  caller — PF_SC_MAXPOS knob candidate; ~half the 8k speedup, zero quality
  risk at the lengths where SC is exact). NOT STARTED this session per mission.
- The G3SC twins + full-block coverage (VRAM: packed7 9.3 GB total; 16 blocks
  ≈ 2.3 GB — raising G3SCN needs free VRAM headroom at 100k) and the launch-
  bound discovery (§3) are the perf path for whichever prefill ships.

## 6. Rig/ops notes this session

- **P7E7 ENV LAW (extends the P7E6 one)**: harness gate runs inside test_w100k
  ALSO need the full DECODE env (SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352
  GEMVV=1 KV8=1 QH=1 PVH=1 HM=1) — a gate run with only the P7E6 subset faulted
  deterministically at the FIRST decode-graph execution (w1c0/w1c1 build OK,
  fault inside G.run_tokens; log ~/p7e7_t8k_sc.log). Same signature as the
  P7E6 end-of-session "decode-graph fault class" — that class is reachable by
  env omission, not only by dirty dext. The daemon-style env passed 5/5 runs.
- Physical dock power-cycle + relaunch verified: daemon boot ok, health ok,
  one chat clean (41 tokens) — the P7E6 fault class WAS cleared by the cycle.
- Daemon shutdown RPC = JSON line {"id":0,"method":"shutdown"} on
  /tmp/llm-engine.sock; GPU-EXIT/REBOOT law fired as documented (machine
  rebooted seconds later; launchd healed nothing this time — manual relaunch).
- The nv_usb4.lock (at /var/folders/.../T/, NOT /tmp) correctly serializes GPU
  processes: an accidental triple-launch left two runs dead on lock acquisition
  (clean RuntimeError, no GPU damage).
- pf_gate2k t1_decode_gapped can ASSERT on exact ties (argsort vs kernel
  argmax order — tok 4649 vs 43614 at 2k M32): known tie-mine artifact, not a
  state bug; the F-relerr is the robust signal.

## 7. Service state at session end

Relaunched on the P6 canonical recipe (unchanged):
`cd ~/tinygrad-metal/engine0 && nohup env PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
DOCKER_HOST=unix://~/.colima/default/docker.sock DEV=NV M1A_SERVE=1
SKV=1 SKV_K=g4nw32 SKV_S=256 SKV_CTXK=100352 GEMVV=1 KV8=1 QH=1 PVH=1 HM=1
M1A_KEEPALIVE_S=10 M1A_GEN_REBUILD_EVERY=256 PF_PREFILL=1 PF_GEMM3=1 PF_SCANC=1
PF_SUPER=0 PF_G3SC=0 MTP_KERNARGS_MB=256 ~/tg311/bin/python -u test_w100k.py
> ~/w100k_serve_p7e7.log 2>&1 &` + `python3 api_server.py` (:8080).
Logs: ~/p7e7_t8k_sc{,2,3}.log (SC trial + determinism), ~/p7e7_t8k_m32.log,
~/p7e7_t2k_{sc,m32,g3sc}.log, ~/p7e7_r100k_sc.log, ~/w100k_serve_p7e7.log.
API verification (M1C contract, P6 config): health ok; 8k-class FRESH chat
(prompt8k.txt regenerated from ids8k.json) = 41.7-42.5 s end-to-end (~185
tok/s incl. decode+dfill; coherent on-topic output); pinned FOLLOW_UP turn
(client conversation_id) = **1.1 s delta prefill** (vs 42.5 s FRESH). The
prefix-decision log line confirms mode=FOLLOW_UP reuse.

## 8. Files changed (engine0)

- pf_gate2k.py — SCDBG_EARLY_EXIT (restores old stop), PF_TRUNC (ids8k head),
  SCDBG_SKIP_CD (skip CTRL/D arms).
- pf_prefill.py — PF_SC_CLASSSYNC=k per-class chunk attribution; PF_G3SC
  per-class list; ffn/iq3d independent twins; _run_sc_chunk(ci=).
- pf_dp4a.cu — nseg hardcoded 8 (gridDim law fix); pfk_dp4a.cubin rebuilt.
- test_p7f.py unchanged (now passes); /tmp/detok.py = standalone detokenizer.
