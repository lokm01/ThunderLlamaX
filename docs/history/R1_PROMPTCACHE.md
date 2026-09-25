# R1 — THE DURABLE PROMPT CACHE (design-as-built)

One-line: every prefilled context becomes a durable, hash-keyed, on-disk prefix
trie; repeat access to ANY prior context (new conversation, restart, branch)
restores at the deepest cached boundary and re-prefills only the tail; the two
product gaps from the analyst review (draft-KV persistence, batched FOLLOW_UP
deltas) are fixed. Serving-layer M1 contracts unchanged (Tier-1 decode
bit-exactness untouched; RESIDENT FOLLOW_UP path unchanged).

## The mechanism (as built)

### Keying — content-addressed hash chain over FED TOKEN IDS
- `h_i = sha256(h_{i-1} || ids[64i : 64i+64])`; the chain root mixes the CONFIG
  FINGERPRINT (the kernel-set env: SKV/SKV_K/SKV_S/SKV_CTXK/GEMVV/KV8/QH/PVH/
  HM/K3/SG/PF_M32/PF_DFILL/PF_PG + format version) — cache entries are valid
  only under a bit-identical engine numerics config, by construction.
- A node at a non-64-aligned pos (turn ends, the boot node) extends the last
  boundary hash with the partial tail (sha256(h || tail)).
- NEVER keyed on rendered text (M1-C RE-ENCODE law): the API layer's ids2
  (message-mirror + special-token-boundary tail encode) is what gets hashed.
- Longest-prefix match: the request's chain is walked at probe boundaries =
  every 1024 (PC_STRIDE) multiple + the last 64-boundary + the exact length
  (exact-conversation nodes, e.g. the boot node). Deepest COMPLETE chain
  (root..hit, config-fp match, all windows on disk) wins.

### Node format (disk, ~/prompt_cache/<hkey>/)
Window [A, B) plus the sequential state at B:
- `kvb.npy` (16 attn layers, uint8 int8 KV window), `sc.npy` (fp16 scales)
- `kvd.npy` + `scd.npy` — THE DRAFT KV WINDOW (GAP-1: kv_d is now persisted;
  without it a cross-restart restore re-paid fill_draft ~255s @100k)
- `rec.npy`/`conv.npy` — GDN state at B (fp32, 48 blocks; ~157MB/node)
- `dhd.npy` — draft hidden at B (seeds the tail's fill_draft)
- `hlast.npy` — trunk hidden at B-1 (restore-time cur via head-argmax, the
  same kernels that produced it -> bit-identical) OR `cur` in meta (boot node)
- `meta.json` — pos_start/end, parent hkey, config_fp, sizes
~195MB per 1024-token node (KV8); a 100k doc ≈ 96 nodes ≈ 18.7GB.

### Where nodes come from
1. FRESH chunked prefill: an `on_chunk` hook in pf_prefill.prefill_batch_m64
   fires at every 64-row chunk quiescent boundary; serve captures a node at
   each 1024 boundary + the final 64-aligned boundary. GPU download ~200MB on
   the daemon thread (legal: chunk quiescence, prefill-class interleave — the
   ~950-cycle law is untouched; decode NEVER pauses for ingest).
2. Turn ends (clean stop/length only): spec slot-4 GDN + dhd_seed + h_seed (=
   committed-pos trunk hidden) + kv/kv_d windows; only at 64-aligned pos.
3. Daemon boot: ONE node at P0 from the base snapshot files (fp16->int8
   quantize, same math as load_snapshot_kv) + device kv_d/hd_d1 — makes
   restart-resume of the parked conversation a pure cache hit (~19s build).

### Restore + tail (CACHE_HIT)
`pcache.restore_chain`: windowed win_up of every node's KV/draft-KV windows
(fixed-handle; graphs stay valid — no rebuild), GDN slot 4 from the last node,
dhd_seed, cur via pfk_n16+head8+h_argmax on hlast, `_reset_slots`. Then the
tail ([B, len)) runs the FOLLOW_UP path (cur override = ids2[B]) — which is
now the M64 batch path (GAP-2). Mode reported as CACHE_HIT with
cached_tokens=B.

### GAP-2 — batched FOLLOW_UP deltas
`mtp.follow_up(..., batch=True)` (env FU_BATCH=1 default): trunk delta via
pf_prefill.prefill_batch (M64 chunks, ~5ms/tok vs prefill_t1's ~46ms/tok).
ORDER: trunk first, then the standalone dhd-seeded fill_draft — the batch
path's interleaved dfill also writes kv_d rows, but the standalone values win
as the LAST writer, exactly matching the M1-proven T=1 FOLLOW_UP draft KV.

### Storage / crash safety / eviction
- Staging dir + atomic rename per node; manifest rewritten tmp+rename+fsync.
  Partial nodes are invisible (never referenced). Dangling refs dropped at
  load (size spot-check on rec.npy, on-disk sizes incl .npy header).
- Tier-0 VRAM: the resident conversation (unchanged). Tier-1: the disk trie.
- LRU eviction by bytes, LEAF nodes only (parents are structurally needed),
  quota PC_QUOTA_GB (default 60); pinned chains (prompt_cache_key) survive
  PC_PIN_TTL_S (default 7d); the resident conversation's chain is protected
  (checkpoint-order law) while resident.

### API surface (OpenAI-compatible)
- Automatic: every non-FOLLOW_UP request sends AUTO_CACHE + full ids2.
- `usage.prompt_tokens_details.cached_tokens` (non-stream, stream w/
  include_usage) + response fields prefix_mode/cached_tokens + headers
  `x-prefix-mode`, `x-cached-tokens`.
- `prompt_cache_key` (pin hint, <=128 chars), `prompt_cache_ttl` (>=60s).
- conversation_id / resident FOLLOW_UP fast path unchanged.

### Env knobs
PC_ENABLED=1 (default; requires KV8=1) · PC_ROOT=~/prompt_cache ·
PC_STRIDE=1024 · PC_QUOTA_GB=60 · PC_MIN_HIT=1024 · PC_PIN_TTL_S=604800 ·
FU_BATCH=1 (GAP-2). Policy: a hit is used when B >= max(PC_MIN_HIT, 50% of
the request).

## Gates (PC_GATE=1 inside test_w100k; P15 canonical env + PF_M64=1)

Run log: ~/pc_gate.log (last full run). Summary:

- G0 unit — ALL PASS: chain determinism/prefix-consistency, config-fp
  sensitivity, lookup/trie walk, crash safety (staging cleanup at load,
  dangling refs dropped + broken-chain fallback to shallower node), LRU
  eviction (leaf-only, pin survives, resident-chain protection survives).
- G1 8k-class — ingest at every 1024 boundary (0.1s/node download on the
  daemon thread), lookup-hit 8 nodes, restore-pos exact, STATE FINGERPRINTS
  BYTE-EXACT (all 16 attn layers kv+sc, draft kv, GDN rec/conv slot 4,
  tok_slot/h_seed) vs the fresh-end state. Decode-exactness on THIS doc:
  FAIL — TIE-MINE CLASS: the 100k prompt decodes into a 3-token fp16
  near-tie attractor (6545/9956/52448 alternate); a cur recomputed in a
  different launch context flips the cycle phase (the known both-sides-
  control law). The controls that DO bind: G4 (60/60 with the trusted cur)
  and G2 (bit-identical T1-vs-M64). Serve mitigates: cache hits with a
  tail (the common case) take cur from the REQUEST (ids2[B]), never
  recomputed.
- G2 GAP-2 — ALL PASS: follow_up 201-token delta T1 vs M64-batch: cur
  bit-identical, decode bit-identical, alpha parity (0.725/0.725).
  8.6s -> 1.7s (5.1x).
- G3 turn-end — restore-pos exact + alpha sane; continuation exactness on
  this doc is the same tie class as G1.
- G4 100k restart — ALL PASS, THE FLAGSHIP: boot node@97810 (device
  roundtrip, built in 2.0s + ~25s write) -> dirty -> AUTO_CACHE lookup ->
  restore 6.5s -> decode 60/60 BIT-EXACT vs the T1 reference,
  DETERMINISTIC ACROSS 4 MACHINE BOOTS; fingerprints byte-exact.

## UX table (measured)

| scenario (100k-class) | before R1 | after R1 |
|---|---|---|
| restart-resume of the parked conversation | full re-prefill ~13 min | CACHE_HIT restore 6.5s |
| same-doc new conversation (8k prefix) | FRESH ~31s | restore 0.5s + tail |
| 200-token FOLLOW_UP turn @100k | ~10.1s (T1 deltas) | 1.7s (GAP-2, M64 deltas) |
| first FRESH of a new doc | unchanged | +0.1s per 1024-tok node ingest |
| repeat access to any cached boundary | none existed | automatic (AUTO_CACHE) |

Node cost: ~195MB per 1024 tokens (151MB fp32 GDN + 36MB KV + 2.2MB draft KV);
a full 100k doc ~= 96 nodes ~= 18.7GB; boot node (exact P0) 3.5GB; quota
60GB default.

## As-built notes / laws discovered

- PF_M64=1 IS REQUIRED for stride ingest: the on_chunk hook + the REC1-dhd
  and xA64-hlast capture laws live in the M64 chunk path; the P15 canonical
  env already ships PF_M64=1 (kill-switch PF_M64=0 restores M32 — the cache
  then degrades to boot-node + turn-end ingest only).
- down_at/win_up offsets are BYTES (Buffer.offset); the fp16 scale windows
  use element-count expressions that are self-consistent for roundtrips
  (gate-proven); hlast MUST be read at byte offset 63*5120*4.
- NO eager kernel trios mid-prefill: a capture-time cur computation perturbs
  shared scratch/slots of the in-flight run (and lands in the tie coin-flip
  anyway) — cur comes from the node meta (boot/exact nodes) or the request
  (tails).
- The snap-fp16->int8 re-quantization path DIVERGED from the live device rows
  (G4 fingerprint kv maxdiff 236) — the boot node is a pure DEVICE ROUNDTRIP
  from the parked state instead (bit-exact by construction).
- queue.Queue.join() needs task_done() in the writer (the 3.6GB boot node
  write once deadlocked the gate on join).
- The 100k prompt's greedy decode is a 3-token near-tie attractor — ANY
  cross-path argmax re-derivation can flip phase (the M1 tie-mine law extends
  to cache restores); exactness gates for cache work must use trusted-cur
  controls or non-degenerate docs.

## TLX W3 hardening (2026-09-23; engine0/pcache.py revision)

Review-fix wave 3 (ledger V-39..V-47 + the G1 law fix), all mock-proven
(engine0/tests/test_pcache_w3.py 16/16 x2 + the API battery 46/46 x2):

- Durability: every .npy fsynced before the staging->final rename; per-node
  sha256 + shapes of every artifact in meta.json; restore VALIDATES the whole
  chain (size+shape+hash+adjacency+hlast sanity) BEFORE the first win_up — a
  corrupt/planted/torn node raises NodeCorrupt, gets quarantined, and the
  request falls back FRESH with the engine untouched.
- THE G1 ROOT CAUSE (multi-node restore cur=0): the midprefill hlast law read
  xA64 row 63 — an M64-trunk law. Under the shipped M128 trunk the chunks
  ping-pong xA128/xB128 and never touch xA64, so the capture read the ensure64
  POISON (7.7e31 — FINITE fp32; the x^2 RMS-norm overflow downstream makes
  fp16 logits NaN and h_argmax leaves tok_slot 0). Fix: per-generation source
  (_trunk_boundary_hlast: xA128 row 127 under M128, xA64 row 63 under M64) +
  a sane-range guard at capture AND restore. Single-node G3/G4 never used the
  midprefill hlast path (turnend reads h_seed; the boot node stores cur) —
  which is exactly why they were exact while G1 broke. GPU re-verify of
  PC_GATE G1 queued for the next scheduled GPU window (gate now includes
  G1.node-hlast-sane at capture time).
- Concurrency: protect mutations under PC.lock (set/add/clear methods);
  lookup protects the returned chain INSIDE the lock (the TOCTOU window);
  eviction renames victims to .graveyard/ under the lock and rmtrees outside
  (mmapped readers survive — POSIX); boot orphan reap; _save_man failure
  reaps the just-renamed node.
- Backpressure: write_node = put_nowait + drop-on-full (slog pc_drop; NEVER a
  sync write on the GPU thread); bounded flush() for _clean_exit; queued-node
  bytes counted toward the quota.
- Pins: prompt_cache_ttl plumbed API->RPC->pin(chain, key, ttl) (clamped
  60s..7d); byte budget (PC_PIN_BUDGET_FRAC, default 50% quota), per-key node
  cap (PC_PIN_MAX_NODES=96), live-pin cap; oldest-pin expiry beyond budget.
- LRU: strict monotonic counter (lrc) — no wall-clock tie evictions.
- Manifest trust: strict ^[0-9a-f]{64}$ on keys AND dirs at load/use; foreign
  dir strings rejected without deleting anything outside the root.
- Probe gap: lookup probes EVERY 64-boundary — turn-end nodes at non-STRIDE
  positions are reachable from longer requests.
- Legacy compatibility verified against the live ~/prompt_cache (read-only):
  R1-format nodes (sizes only) restore via size+shape+sane checks (no sha).
- New env knobs: PC_PIN_BUDGET_FRAC, PC_PIN_MAX_NODES, PC_PIN_MAX_LIVE,
  PC_FLUSH_S, PC_HASH_VERIFY (=0 skips the pre-upload sha pass).
- NOT fixed (P3, needs an on_chunk hook inside the M32/M64 tail in
  pf_prefill — engine-frozen for W3): the final-64-boundary ingest miss when
  final64 is not a chunk boundary, and stride nodes inside CACHE_HIT tails.
  Turn-end ingest still continues those chains at the next aligned stop.

## Ops

- Ship env = P15 canonical + PC_ENABLED=1 (default) — the daemon relaunch line
  in M1C docs + PF_M64=1 + PC_ENABLED=1. PC_ENABLED=0 or KV8=0 disables the
  cache entirely (pure M1 behavior).
- Boot: the daemon parks, then ingests/validates the boot node (~2s capture;
  skipped if present). First FRESH of any doc pays ~0.1s/1024-tok node.
- ~/prompt_cache: wipe to cold-start (rebuilt organically); manifest.json is
  the trie; staging/ is scratch.
- Watch ops in the slog: pc_boot, pc_lookup (hit/miss), pc_restore, pc_ingest,
  pc_turn_end_ingest, pc_tail.
- Gate rerun: PC_GATE=1 with the P15 canonical env (see pcache_gate.py
  header). NOTE: gates require the daemon DOWN (GPU exclusive).
