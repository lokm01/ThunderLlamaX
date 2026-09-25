# P7-E4 — Cold-cycle verdict + the staged gates: SC does NOT ship; the gate bug isolated to the NONZERO INITIAL GDN STATE; gemm3-m64 attn twin indicted in-plan

Status: **the P7E3 dock-power fault class is CONFIRMED CLEARED by the physical
cold cycle (full canonical pass on the first try). The staged SC gates
(A2 dfill-on, B dfill-off) FAIL DETERMINISTICALLY ON THE CLEAN MACHINE
(F 5.067e-1, 0/60, the 22525 loop — bit-identical to the P7E2 gate numbers)
=> PF_SUPER stays 0; the daemon ships the P6 canonical unchanged. The staged
isolation WORKED: dfill exonerated (A2 == B bit-identical logits/conv),
REAL-PROMPT CONTENT exonerated (scdbg5 on ids8k = EXACT), and the failure is
reproduced MINIMALLY by seeding the scdbg world with the SNAPSHOT (nonzero)
GDN initial state instead of zeros: SC-vs-M32 diverges (conv61 maxabs 0.508,
logits medrel 3.0e-2 vs 1.7e-3 zero-seed; late-block onset, early blocks
clean — the gate signature). => ROOT-CAUSE LEAD: the c64_nc4 chunked scan
mishandles the NONZERO INITIAL recurrent state (the gate world's fresh()
restores the 97810-position snapshot via stload_trunk; every clean-world
harness started from zeros, which is why they were exact). GEMM3-M64 verdict:
standalone at M=256 ALL 5 CLASSES BIT-IDENTICAL (+guard canaries clean); in
the SC plan the attnqkvi3-m64 twin ALONE corrupts (NaN cascade from the first
attn block, chunk 1, ring-size-independent) while ffn/iq3d/gdnqg m64 are
in-plan EXACT — a ~50ms/chunk partial PF_G3SC win is proven behind a
per-class knob. dp4a probe: kernel built + harness ready; the launch no-ops
(empty AND typed signature) — parked, numbers INVALID.**

## 1. The fault class (mission step 1) — CLEARED

`test_w100k.py` canonical (M1C env + PF_PREFILL/GEMM3/SCANC=1 PF_SUPER=0
PF_G3SC=0 MTP_KERNARGS_MB=256), first run after the cold cycle
(~/p7e4_canon.log):

- T=1 reference @97810: **COMPLETED** 45.83 ms/tok = 21.82 tok/s (toks 6545,
  9956, ... correct).
- boot fill_draft @97810: COMPLETED (246s).
- Tier-1: 60/60 exact x2; Tier-2 cross-checks 60/60 + 60/60.
- Timing: **69.06 ms/cyc best = 40.30 tok/s** (rep2), alpha 0.892, 2.78 tok/cyc.
- Stock: 59/59. [done] clean.

=> the deterministic deep-world fault (T1 decode + boot fill @97810) was the
DOCK-POWER/dirty-dext class; a PHYSICAL cold cycle clears it. BANKED: after
this fault class appears, do not chase software — cold-cycle.

## 2. The staged gates (step 2) — A2/B FAIL on the clean machine; NO SHIP

pf_gate2k staged flow (SCDBG_GATE_B=1 SCDBG_GATE_DUMP=1, PF_SUPER=1
PF_GEMM3=0 PF_G3SC=0 PF_SCANC=1, 7714-pos ids8k class; ~/p7e4_gateAB.log):

| phase | result |
|---|---|
| A1 T1-prefill + T1-decode | 306.4s + 3.2s; tokA sane (6545/9956/22546 tie family) |
| A2 SC prefill (DFILL=1) | end-state logits F 5.067e-1 vs T1 — WRONG (top1 22525) |
| B SC prefill (DFILL=0, same world) | F 5.067e-1; GATE B 0/60; decode = 22525 loop |

State-dump forensics (~/gate_{a1,a2,b}_state.npz):
- A2 vs B: logits + conv ALL BIT-IDENTICAL (a2-vs-b = 0.0) => **the dfill
  ring is fully exonerated** — identical failure with dfill on AND off.
- conv0/1/2 + rec0/1/2 (early GDN): fp16-class vs A1 (1e-4..6e-4) — CLEAN.
- conv61/62 + rec61/62 (late GDN): 5.8e-1 / 2.6e-1 — WRONG. Late-block onset.
- B's early recs differ from BOTH (4.0e-1): B's fresh() restored slot-4 spec
  state after A2's trunk writes — initial-state path, consistent with the
  root-cause lead below (not an independent corruption).

### The root-cause lead (minimal repro banked)

pf_scdbg5 variants (W1C world, 512 tok, 2 chunks, SC vs M32, G3SC=0):
- ids8k REAL prompt content instead of rng(99) (pf_scdbg5r.py): **EXACT**
  (S1/S2 logits medrel 1.7e-3/8.1e-4) => content exonerated.
- ids8k + rec/conv seeded from the SNAPSHOT npys instead of zeros
  (pf_scdbg5n.py): **WRONG** — S1 conv61 maxabs 5.08e-1 medrel 5.4e-2,
  logits medrel 3.0e-2 (18x the zero-seed class); early blocks clean. Same
  late-block signature as the gate failure. Logs: ~/p7e4_sd5real.log,
  ~/p7e4_sd5nonzero.log.

=> NEXT SESSION (sharply scoped): the gate world's fresh() = reset_fresh +
stload_trunk loads the NONZERO snapshot GDN state into rec{i}/conv{i}_0; the
T1 path (A1) handles it correctly, the c64_nc4 scan (pfcb initial-state
chaining / pfca conv-halo from a nonzero conv_0) does not. First look:
pfcb_c64_nc4's FIRST window/chunk treatment of the incoming S_in (the
per-window LOCAL g decay S' = 2^g_end * S + KhatT(...) must apply the
window-0 decay to S_in; if window 0 assumed S_in == 0 — which the zero-seed
harnesses satisfied — that is exactly this bug). Fix + rerun the staged gate
(A2/B) + THEN the ship decision. (Also note: with S_in mishandled, the
30-chunk P7E3 "state-EXACT" result was exact only because S_in was zeros.)

## 3. Ship decision — P6 canonical UNCHANGED

- PF_SUPER stays 0 (default-off). PF_G3SC stays 0. Serving recipe: M1C +
  MTP_KERNARGS_MB=256 PF_PREFILL=1 PF_GEMM3=1 PF_SCANC=1 PF_SUPER=0
  PF_G3SC=0 (+M1A_SERVE=1 + api_server :8080).
- 100k rebuild benchmark: NOT run — it is conditioned on shipping SC; on the
  unchanged P6 config the P6 record **165.3 tok/s stands** (per P6 protocol).
- 2k/8k-class P6 numbers stand (245.0 / ~190 in-world).

## 4. GEMM3-M64 @ M256 verdict (step 3)

test_p7b256.py (NEW — the exact SC call pattern: ONE launch, grid=NGRID*4,
flat M-grid over full 256-row buffers, no offsets; classic m32x8 with 32-row
offsets as truth; 64KB guard canaries after every mine output):

| class | m64@M256 vs m32x8 | guard | bench m64 / m32x8 (ms/256r) |
|---|---|---|---|
| ffn (nw4) | BIT-IDENTICAL | clean | 3.191 / 4.652 |
| iq3d | BIT-IDENTICAL | clean | 1.424 / 2.367 |
| iq3o | BIT-IDENTICAL | clean | 0.564 / 0.935 |
| gdnqg | BIT-IDENTICAL | clean | 1.576 / 3.001 |
| attnqkvi3 | BIT-IDENTICAL | clean | 1.170 / 2.275 |

=> the m64 ARITHMETIC at M=256 is exact — the P7E2 "m64 corrupts memory"
was NOT arithmetic. But IN-PLAN (pf_scdbg5, PF_G3SC=1): NaN cascade from the
FIRST attn block (3) at chunk 1, late blocks NaN, early GDN clean
(~/p7e4_g3sc.log). Isolation rounds:
- MTP_KERNARGS_MB=512: SAME corruption (~/p7e4_g3sc512.log) => kernargs ring
  size exonerated.
- packed7 k*/q* hidden (attn twins fall back to classic; ALL other m64 live):
  **EXACT** (logits 5.2e-4/5.9e-4, rec/conv clean, both chunks;
  ~/p7e4_g3scnoattn.log) => **the attnqkvi3-m64 twin launch is THE in-plan
  corruptor** (its qrowsc writes), while ffn/iq3d/gdnqg(+iq3o) m64 are
  in-plan EXACT. Standalone the identical attnqkvi3 launch is bit-identical
  => launch-history/plan-adjacency class specific to that kernel's launch.
- Chunk timing: 843-862ms (G3SC=0, P7E3) -> ~791ms (G3SC minus attn twins)
  = **~50-55ms/chunk REAL saving at exact state** (the repacked-16-block
  subset); full G3SC (~795ms) adds ~4ms more but corrupts.
=> NEXT: (a) per-class G3SC knob (PF_G3SC_ATTN=0 default-off) banks the
~50ms now; (b) kernargs-VA logging on the in-plan attnqkvi3-m64 launch vs
its standalone run (the stale-record lead) for the full fix. SERVING STAYS
PF_G3SC=0 until the full serving gate passes with it.

## 5. P7F dp4a probe (step 4) — built, launch-blocked, PARKED

Delivered: engine0/pf_dp4a.cu (pfk_dp4a: warp-per-(t,h,L-seg), LANE-PER-POS
dp4a QK over the engine kv8 layout, K'=K-128 precompute trick, Q
per-(row,head)-128ch quant, no cross-lane reduce, fixed ranges, no
grid-stride; 56 regs 0 spill), build_p7f.py (docker nvcc + symbol check OK),
test_p7f.py (real kv_3.npy KV @100352, engine-quantization verbatim,
host-fp64 validation vs the fp16-pipeline inputs, synced min-of-10 bench at
T=16/64).
BLOCKED: the launch is a NO-OP — out stays poison, under BOTH
signature=tuple() AND the typed ("v",0,dtypes.int32,())x6 form
(e1_sync-style); a wrong-name load ("pfk") FAULTS (split-symbol law
reconfirmed). test_p7b kernels launch fine with empty signatures from the
same harness => the distinguishing factor is UNIDENTIFIED (first __dp4a
kernel?). The 0.09-0.15ms/21-157TOPS prints are INVALID (kernel didn't
execute the work). NEXT: bisect the body against e1_sync's k_triv
(known-good fresh-cubin + typed-sig pattern): k_triv clone -> add args ->
add the loop -> add dp4a, one step per launch.

## Benchmark table (P7E4)

| class | P6 record (shipped) | SC this session | note |
|---|---|---|---|
| 2k-class chunk | 245.0 tok/s | ~791ms/256 = 323 tok/s-class (harness, G3SC-noattn, EXACT) | still not shippable (gate) |
| 8k-class (2-chunk carry) | — | EXACT (zero-seed AND real-content) | state bug only w/ nonzero S_in |
| 100k rebuild | **165.3 tok/s** | not run | P6 stands; SC blocked by the S_in bug |
| canonical decode | 40.16-40.30 tok/s @100k | re-verified this session | 69.06 ms/cyc best |

## Service state

Daemon relaunched FRESH on the ship config (P6 canonical, PF_SUPER=0
PF_G3SC=0 PF_GEMM=1-class env above) + api_server on 8080; one API chat
verified after boot. Logs: /tmp/serve_m1c.log, /tmp/api_m1c.log (and the
reboot-survivor mirror ~/tinygrad-metal/logs/m1a_serve.log).

## Files

- engine0/test_p7b256.py (NEW — M256 gemm3 validation + guards + bench)
- engine0/pf_dp4a.cu + build_p7f.py + test_p7f.py (NEW — probe, parked)
- engine0/pf_scdbg5r.py (real-content variant), pf_scdbg5n.py (nonzero-S_in
  variant — THE minimal repro of the gate bug)
- pf_gate2k.py staged phases ran as committed (no changes needed)
- Logs: ~/p7e4_canon.log, ~/p7e4_gateAB.log, ~/p7e4_g3sc.log,
  ~/p7e4_g3sc512.log, ~/p7e4_g3scnoattn.log, ~/p7e4_sd5real.log,
  ~/p7e4_sd5nonzero.log. Data: ~/gate_{a1,a2,b}_state.npz.
