# P7-F1 — CHUNK-GRAPH CAPTURE: the M32 prefill chunk as QMD-chained graphs (SHIPPED, default-on)

Status: **the per-chunk prefill program (the fixed 706-launch M32 plan [+ 14
dfill launches when PF_DFILL=1]) is captured as 2 QMD-chained NVComputeQueues
per chunk — 2 gpfifo submits replace ~706 eager launches. BIT-IDENTICAL to the
eager path (proven at 2k and 8k on logits/tok_slot/pos_slot/rec/conv/kv/sc,
including a mid-run graph rebuild). Ladder: 2k 247 (P6 245.0) / 8k 212.0
(P6 190.9) / 100k rebuild 183.8 (P6 165.3) = +11% @100k, +11% @8k, parity @2k.
100k cur=4471 EXACT; 11 clean rebuilds over 3056 chunks; daemon relaunched on
PF_PG=1 and verified end-to-end (health, 8k FRESH chat 38.4-39.1s vs P7E7's
41.7-42.5s, pinned FOLLOW_UP 2.2s = M1C class, client_smoke PASS incl.
streaming turn-3). THE HONEST HEADLINE: the M32 path was NOT 80% launch-bound
(that was SC's 2849-launch chunk); it is ~12% launch-bound at 2k/8k shrinking
with position — the capture banks it all, and the path is now KERNEL-BOUND:
the GEMM family quant-load stream (~200 GB/s P5/P6 wall) + attention growth
(144.5 ms/chunk @pos0 → 204.9 @pos97k). The next lever is the DBUF-M32 hybrid
+ the @100k attention retune, NOT more launch work.**

## 1. Design (engine0/pf_prefill.py, PF_PG)

- `PfGraph(seq, tag)`: gcycle's ParityGraph pattern applied to the cached
  M32 chunk plan — one host-mapped kernargs slab (`cpu_access, nolru`,
  per-kernel `fill_kernargs` at build = pointers baked, fixed-handle law);
  `NVComputeQueue` with `wait(timeline, prev_var) → memory_barrier → exec×N
  (QMD dependent-pointer chaining) → signal(timeline, cur_var)`; submit
  var-patches the two timeline values. Local sizes come from the PLAN TUPLES
  (explicit ls — immune to the name-encoded warp-token trap that ParityGraph's
  name-substring heuristic has).
- `PG_SPLIT=2` (default): 706 launches → 2 queues × 353 (each under the
  decode-proven 452-484/queue envelope). Ring cost per submit: 48 bytes.
- `PG_REBUILD=256` (default): rebuild the graphs every 256 chunk-replays at a
  quiescent point (the M1A_GEN_REBUILD_EVERY discipline for the ~950-replay
  dext budget). 100k = 3056 chunks = 11 rebuilds, all clean.
- Per-chunk host work stays the win_up DMAs (ids16a/ids16b 64 B, pos_slot/
  pos_slot_b 4 B, tok_hist window 128 B). NO ids race and NO extra syncs:
  the allocator copy-queue `_copyin` waits `timeline value-1` (= the chunk
  graph's own signal) and signals a new value; the next graph waits that —
  the global timeline chains uploads → chunk → uploads → chunk.
- `PG_WAIT=1` (default): host waits each chunk's final signal (matches the
  eager chunk-end sync semantics; comparable chunk_times, readout-order safe).
- smem audit (cuobjdump, all plan cubins): max is pfa16 35200 B and the two
  43520 B m32 GEMMs (ffn/gdnqg). The "36.8KB in-graph law" is NOT a hard cap —
  the W2H control proved >32 KB (cfg-17 carveout) works in-graph, decode's
  spk_g4hm runs the same 64 KB carveout class, and 1 CTA/SM means no
  co-residency conflict. All three ran captured clean at 2k/8k/100k.
- DFILL: when PF_DFILL=1 (daemon default), the two 7-launch draft-fill
  windows are appended to the captured seq (M32 halves have FIXED args every
  chunk — ring alternates per half). Exercised by both daemon 8k FRESH chats.

## 2. THE NEW LAW (fault): graph-graph back-to-back = SKEDCHECK22

Pipelined chunk submits (PG_WAIT=0, host running ahead) fault deterministically
with GSP `SKEDCHECK22_INVALIDATE_ACTIVE_QMD`: the NO_WFI shader-cache
invalidate collides with still-active QMDs of the prior chunk. Putting the
timeline wait BEFORE the memory_barrier in the queue does NOT save it (the
NO_WFI invalidate is not ordered behind pushbuffer semaphore stalls on this
dext). Eager launches survive full pipelining because EACH kernel's queue
individually waits the prior kernel's signal before its barrier. **Law: on
this dext, captured graphs on the compute channel must be waited before the
next graph's leading invalidate executes; PG_WAIT=0 exists env-gated but is
a fault class — do not ship.** (PG_PIPE_BIT=1 in pf_gate2k reproduces it.)

## 3. Gates (readout-order law: first clean runs)

| gate | result |
|---|---|
| PG-BIT 2k (eager vs captured, same world) | **BIT-IDENTICAL (all keys)**: logits(248320), tok_slot, pos_slot, rec×4, conv×4, kv-tail×3, sc-tail×3 |
| PG-BIT 8k (241 chunks incl. rebuild@256) | **BIT-IDENTICAL (all keys)** |
| 2k A2 logits | top1 29877 / 15.438 / gap 0.0703, F 1.058e-03 — exact P6/P7E7 class; deterministic ×3 processes |
| 8k full gate set | F 4.261e-02, GATE A 12/60, CTRL 13/60 α=2.67, GATE D 5/60, D2 0/160 — **line-for-line == the P7E7 eager M32 run** (~/p7e7_t8k_m32.log; D/D2 are the pre-existing 8k-world signature, NOT a regression) |
| 100k rebuild | 532.2 s = **183.8 tok/s**; **cur=4471 EXACT**; state vs snapshot max rec relerr 3.45e-2 / conv 6.0e-2 (the M32 reassociation-floor class; SC was 5.4e-1/1.0); decode-on-rebuilt 27/60 (tie-mine 6545/9956/4649 cycle class) |
| serving | health ok; 8k FRESH chat 38.4-39.1 s (P7E7: 41.7-42.5); pinned FOLLOW_UP 2.2 s mode=FOLLOW_UP; client_smoke PASS (streaming turn-3); dfill-captured graph (dfill=True) served both 8k chats; zero faults |

## 4. The ladder + attribution

| length | P6 M32 (eager) | P7F-1 (PF_PG=1) | delta |
|---|---|---|---|
| 2k fresh | 245.0 tok/s (banked) | 247 tok/s (8.3 s; this run's eager arm 218) | parity (in-run A/B: 146.6→129.0 ms/chunk = 1.14x) |
| 8k fresh | 190.9 (40.4 s) | **212.0** (36.4 s) | +11% (167.7→150.8 ms/chunk = 1.11x) |
| 100k rebuild | 165.3 (591 s) | **183.8** (532.2 s) | +11% (−19.4 ms/chunk × 3056) |

Chunk-time profile (graph mode): 144.5 ms @pos0 → 158.8 @pos10k → 160.8 @pos48k
→ 204.9 @pos97k. Launch overhead is now ≈ zero (2 submits + 5 DMAs + 1 wait);
the Δ60 ms pos0→pos97k is attention/KV growth; the 144.5 ms base is dominated
by the GEMM family (whole-model weight read per chunk = 12.6 GB; at the
measured 163-214 GB/s family rates ≈ 60-70 ms floor; the m32 twins run
sub-linear per P6) + scan/norms/head scraps.

**What binds next (P7f-2 candidates):** (1) DBUF-M32 hybrid on the GEMM family
(priced 1.3-1.6x in P6 — the biggest kernel-side lever); (2) attention retune
@100k (the +60 ms/chunk growth); (3) norm-GEMM fusion (the 2×16 halves doubled
the small-kernel count — now pure kernel-time, no launch penalty to hide them).
NOT worth chasing: more launch work (done), SC-256 correctness (rejected),
G3SC/m64 twins (SC-path-only machinery — PF_SUPER is rejected for long-ctx
state; on the correct M32 path there are no m64 kernels to enable, and the
SC-side −54 ms/chunk measurement does not transfer).

## 5. Ops

Daemon (live, verified): the P7E7 §7 line + `PF_PG=1 PG_WAIT=1` (PF_PG now
defaults to 1 in code — PF_PG=0 restores the eager path byte-for-byte).
Harnesses: unchanged full-env law; PF_PG_BIT=1 adds the bit-identity arm to
PF_GATE=1 runs. Logs: ~/p7f1_g2k{,b,c}.log, ~/p7f1_g8k{,b,c}.log,
~/p7f1_r100k.log, ~/w100k_serve_p7f1.log, ~/api_p7f1.log.
Machine: one scheduled cold-cycle was NOT needed (both SKEDCHECK faults
cleared on process exit; reboot law fired once at session start for the
daemon shutdown as documented).

## 6. Files

- engine0/pf_prefill.py — PfGraph, _pf_dfill_seq, _pf_graphs,
  _pf_submit_chunk, PG_SPLIT/PG_REBUILD/PG_WAIT knobs, the chunk-loop capture
  branch, PF_PG default-on.
- engine0/pf_gate2k.py — PG-BIT arm (eager vs captured bit-identity + chunk
  timing; PG_PIPE_BIT reproduces the SKEDCHECK fault, off by default).
