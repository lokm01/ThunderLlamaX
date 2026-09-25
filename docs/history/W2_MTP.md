# W2-MTP: speculative decoding (K=2) on the engine — Tier-1 EXACT, 40.14 tok/s @2k

Gate targets: (1) Tier-1 exactness engine-spec(K=2) == engine-greedy(T=1) 60/60
bit-exact, deterministic; (2) >=40 tok/s @2k; (3) alpha + breakdown report.

## Results (final run: ~/w2_test.log, 2026-09-14)
| metric | value |
|---|---|
| Tier-1 exactness | **60/60 EXACT vs engine T=1 greedy, rep0+rep1, deterministic across reps** |
| tok/s (best of 3 x 60-cycle timed runs) | **40.14** (39.81-40.14; 61.86 ms/cycle) |
| T=1 engine baseline | 25.59 tok/s -> **1.57x** |
| per-position acceptance alpha | **0.742** (44.5 accepted / 60x2 drafted) |
| tokens/cycle | 2.48 (m_hist: mostly m=2 full accepts; m=1 on cycle 0, occasional m=0) |
| cycle kernel count | 484 (draft 29 + probe 452 + accept 2 + flusher 1), 4 graph submits |

## Per-phase breakdown (phase-waited run)
| phase | ms/cycle | note |
|---|---|---|
| draft (2 steps) | 4.60 | Q4_0 blk.64 + eh_proj + 40960-row Q5_K slice head + slice argmax |
| probe T=3 | 56.64 | 452 kernels, M=3 batched (weights read once) |
| accept+select | 1.15 | m calc + emit + pos advance + rec/conv slot-m copy + h_seed select |
| total | 62.3 | incl ~0.5 ms submits/waits |

vs T=1 39.08 ms/token: probe M=3 = 56.6 (+17.5: 3-step k2s3 scan ~+6ms, 3-row
GEMV x-loads ~+8ms, aattn3 3-row online-softmax ~+2ms).

## What was built (engine0/)
- `m3.cu`/`m3b.cu` -> per-kernel cubins (build_w2.py): **h_embed3, k0n3, k0ab3,
  q5g8_3, k2s3, op38_3, k3ao3, hh3, ffn8_3, down8_3, aq6k8_3, aq3k8_3, aattn3,
  ao8_3, head8_3, amx3** — the full T=3 probe trunk. Every M=3 kernel keeps the
  T=1 kernel's per-row fp op order EXACTLY (same loads, same mul/add sequence,
  same shfl tree) -> rows are BIT-IDENTICAL to the T=1 path. Tier-1 passed on
  the first complete run after the accept-arg fix (no near-tie flips at 2k).
- `q4v.cu` (-DKNAME/-DNOUT/-DNGRP/-DADDHH per-use cubins: ehproj, dq, doproj,
  ddown) + `mtpd.cu` (dnorm2, dfgu, dkv, aattn_d=CTX-2304 a_attn clone, shead,
  samx, dposadd, accept, acceptsel) — the draft chain + accept/commit, all Q4_0
  draft weights in an aligned two-region pack ([qs NGRP*128B][d NGRP*16B]).
- `q4pack.py` -> draft_pack/*.npy (offline repack, 168MB); draft = blk.64
  (ALL Q4_0: q/k/v/o/ffn/eh_proj; norms f32) + shared lm_head restricted to a
  40960-row Q5_K slice (144MB raw rows + id table; duplicates tie-break to
  first index so cycling the 26-id slice is safe).
- `mtp.py` — MTPEngine(TrunkEngineW1C): M=3 scratch, block-major per-step GDN
  states rec4/conv4 [48][5][...] (slot 4 live, 0..2 per-step; no scalar kernel
  args — offsets via buffer.offset()), draft weights, graphs, cycle loop.
  Cycle = draft_g -> probe_g -> accept_g -> flush_g, all device-resident
  (tok_slot/pos_slot/dring/m_hist; ZERO host work inside the loop).
- `test_w2.py` — the Tier-1 gate: engine T=1 60-tok reference -> deterministic
  slice -> spec x2 (exactness+det) -> 3 timed reps -> phase breakdown.
- Probe-kernel smokes: test_w2d.py (draft kernels), test_w2p.py (probe kernels
  on real single-block weights); debug harnesses dbg_*.py kept for the record.

## The MTP contract implemented (mtp_v3 semantics, device-side)
- Cycle order: DRAFT (2 steps at pos, pos+1: xin = eh_proj([enorm(emb(prev)),
  hnorm(hm)]), draft block w/ own KV, head_norm + slice head + slice argmax ->
  dring) -> PROBE (feeds [cur, dring0, dring1] at [pos, pos+1, pos+2] through
  the M=3 trunk; per-row argmax -> amds) -> ACCEPT (m = longest prefix of
  dring matching amds; tok_hist[pos+t] = amds[t] for t<=m — exactly the T=1
  h_argmax emission contract; pos += m+1; cur = amds[m] = bonus; h_seed =
  probe trunk hidden row m; acceptsel copies rec4/conv4 slot m -> slot 4).
- Draft KV: prompt-filled over all ids positions (chained draft hiddens,
  zeros at pos 0); each chain step overwrites its position's KV; rejected
  positions are always overwritten by the next chain (m+K >= K always);
  accepted positions keep draft-hidden KV (no trunk-hidden resync — the
  mtp_v3 D8 resync is a known ~+0.05 alpha lever, costs 2 draft-block runs).

## NEW DEXT/FORK GOTCHAS (each cost a debug cycle — LAW now)
20. **A LONE-submitted graph never completes** (signal sticks at value-1;
    ANY length, incl. a full 452-kernel graph; reboot does not clear it —
    it is a submit-path property, not corruption). Every graph needs a
    follow-up submit in flight before its signal fires. W1C's GCycleEngine
    always had >=2 graphs chained. FIX: a 1-kernel flusher graph chained
    after the last real graph each cycle; phase timing must wait a phase
    only while a later graph is already submitted.
21. **Fork Q4_0 layout is a NIBBLE-PLANE split, NOT ggml consecutive pairs**:
    within each 32-elem block, elements 0..15 = LOW nibbles of bytes 0..15,
    elements 16..31 = HIGH nibbles (gguf.py q_to_uint8 transpose-flatten).
    Kernel mapping: element e -> byte (e&15), nibble (e>>4); per-lane run of
    8: u64 chunk = lane&1, byte-in-chunk = j, nibble = (lane>>1)&1. Using
    ggml pairs gives relerr ~1.0 — draft predicts garbage (tie-argmax to
    slice idx 0). Diagnose with indicator-vector probes (weff vs fork
    dequant ground truth).
22. **TinyELF finds the kernel by `name` = the cubin's symbol**: a template
    .cu must rename its kernel per build (-DKNAME=...); mismatched name =
    Illegal Instruction Encoding faults / silent no-ops. Also `do` is a C
    keyword (doproj).
23. **Arg-count mismatch (kernel params vs launch bufs) is a SILENT arg
    shift** (empty signature = no validation): accept took 10 params, got 11
    -> every arg shifted one slot -> device fault. Count args on both sides.
24. **q4v half-vs-float output dtype**: kernels consumed by k0_norm
    (float input) must emit FLOAT (build with -DADDHH + a zeros residual
    buffer); half-writes into a float buffer look "unwritten" in poison checks.
25. **Poisoned dring(-1) read by h_embed3** = emb[-2200] negative-offset
    fault — isolation tests that submit the probe without the draft graph
    running first must pre-seed dring with valid ids.

## K=3 notes (not built)
K=3 needs: amds[4]/dring[2] (accept m-ladder), probe T=4 (M=4 GEMV variants —
same template; k2s3 -> k2s4), draft step 2 (+dposadd chain), 40960-slice alpha
at K=3 only pays if alpha(pos) ~0.75 holds at depth 2 (m_hist shows occasional
m=0 at depth 0 — investigate those before K=3).

## What 100k needs
- Graphs are ctx-STATIC (pos from pos_slot device-side) — architecture
  carries over; only CTXK-sized buffers grow (kv 16x, kv_d, tok_hist) and
  aattn3's online-softmax l-loop becomes BW-bound (~the 19.7ms@8k-class per
  layer): needs the split-KV attention kernel family before 100k is fast.
- Draft KV prompt fill is O(prompt): 1988 steps = 4.6s @2k -> ~4 min @100k
  (batch the fill T>1 or fill only the tail window + accept the KV warm-up).
- Slice: prompt-frequency table on the real 100k prompt (48k+ distinct ids
  -> real 40960 coverage, no cycling duplicates).
- Probe M=3 at 56.6ms is the speed lever: k2s3 3-step scan (+6ms), 3-row
  GEMV x-loads (+8ms); sync_every=2 pipelining (966 kernels in flight —
  above the ~904 proven-stable point, needs a ceiling test).

## Run
`cd ~/tinygrad-metal/engine0 && DEV=NV PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
DOCKER_HOST=unix://<colima-socket> ~/tg311/bin/python test_w2.py`
(needs ~/w1b_state_2k.npz + packed/ + draft_pack/; SYNC_EVERY=1 default)
