"""TLX P8 MoE SERVING BRIDGE — the Qwen3.6-35B-A3B prompt cache (pcache_moe).

Node format r1moe1 (model-scoped by config_fp: the MoE env carries
TLX_MODEL_PATH -> a different model identity hash -> a separate chain ROOT
from every dense node; the daemon ALSO mixes an svc_fp extra (the packed-
manifest sha) so a repack invalidates):

  S    (30, 32*128*128)   f32   GDN state SALL at the boundary (62.9MB)
  CS   (30, 8192*3)       f32   conv state CSALL at the boundary (2.8MB)
  kq   (10, 2, W*256)     i8    int8 K rows [A,B) per attn layer per head
  ks   (10, 2, W*2)       f32   K scales  [A,B) per head
  vq   (10, 2, W*256)     i8    int8 V rows per head
  vs   (10, 2, W*2)       f32   V scales per head
  meta cur                the greedy top1 AFTER the boundary token (exact
                          restore; the eager-head derivation runs at capture)

NOTE the KV array layout is HEAD-MAJOR: KVQ[ai] is (2, CTX, 256) — head 0
rows [A,B) live at byte offset A*256, head 1 at (CTX+A)*256. NOT contiguous
across heads; every artifact carries the head axis explicitly.

Sizes: ~66MB fixed + ~10.5KB/token. A 1024-node ~= 76MB; a 100k doc ~=
96 nodes ~= 1GB (vs the dense 18.7GB — the MoE KV is 10 int8 layers and the
GDN state is a fixed 66MB, not per-token).

DISCIPLINE (mirrors pcache.py W3/W3.5 exactly — the same class machinery, a
different artifact set):
  - transactional restore: validate-whole-chain-then-upload (sizes + sha256
    per artifact), NodeCorrupt -> quarantine -> FRESH fallback, the engine
    untouched on any validation failure;
  - staging + atomic rename + fsync + dir-fsync (the L5 law) per node;
  - LRU eviction with byte budget + graveyard + protect/pin (inherited);
  - the same 64-block chain hash (pcache.chain_keys) over the fed stream.

The SPLIT-S NUMERICS CLASS LAW (MM P7 law 4) does NOT touch node bytes: the
stored KV is the QUANTIZED cache (S-independent), and the proven serving
config mixes PF(S=8) feed + S=32 decode (the L64/L96 gates) — a node captured
from that pipeline restores into it exactly.
"""
import os, json
import numpy as np

import pcache
from pcache import (NodeCorrupt, _sha256_file, chain_keys, hkey_prefix,
                    config_fp, HASH_BLK)

FMT_MOE = "r1moe1"
ART_MOE = ("S", "CS", "kq", "ks", "vq", "vs")
N_ATTN = 10       # attn layers (KVQ/KVS/VVQ/VVS index space)
N_GDN = 30        # GDN layers (SALL/CSALL)
CTX_BYTES_I8 = 256        # one row, int8 KV
CTX_BYTES_F32 = 8         # one row of scales (2 heads x 2 f32 = 16B total; 8B/head)


class MoePromptCache(pcache.PromptCache):
    """The MoE node format on the shared manifest/writer/evict machinery."""
    FMT_C = FMT_MOE
    ART = ART_MOE


def _ctx_alloc(rig):
    """The KV arrays' allocation ctx (rows per head). Rig7 sets the MM_P56_lib
    module CTX_ALLOC at boot; lazy so the mock battery never imports the GPU
    stack."""
    c = getattr(rig, "CTX_ALLOC", None)
    if c is not None:
        return c
    import MM_P56_lib as L56
    return L56.CTX_ALLOC


def _dn(rig, buf, shape, dtype=np.float32):
    n = int(np.prod(shape))
    mv = memoryview(bytearray(int(n) * np.dtype(dtype).itemsize)).cast("B")
    rig.dev.allocator._copyout(mv, buf)
    return np.frombuffer(mv, dtype=dtype).reshape(shape).copy()


def capture_node_moe(rig, A, B, fed_prefix=None, parent=None, cur=None):
    """Download a node [A, B) from live MoE engine state at a QUIESCENT
    boundary (a 256-chunk edge of a PF feed, or a turn end). cur: the greedy
    top1 AFTER the token at B-1 (the daemon derives it via the eager head at
    chunk boundaries, or knows it at turn ends)."""
    if B <= A or A < 0:
        raise ValueError(f"invalid cache window [{A},{B})")
    W = B - A
    node = {"pos_start": A, "pos_end": B, "parent": parent, "config_fp": config_fp(),
            "hkey": hkey_prefix(fed_prefix[:B])}
    if cur is not None:
        node["cur"] = int(cur)
    node["S"] = _dn(rig, rig.SALL, (N_GDN, 32 * 128 * 128))
    node["CS"] = _dn(rig, rig.CSALL, (N_GDN, 8192 * 3))
    kq = np.empty((N_ATTN, 2, W * 256), dtype=np.int8)
    vq = np.empty((N_ATTN, 2, W * 256), dtype=np.int8)
    ks = np.empty((N_ATTN, 2, W * 2), dtype=np.float32)
    vs = np.empty((N_ATTN, 2, W * 2), dtype=np.float32)
    for ai in range(N_ATTN):
        for j in range(2):
            off_i8 = (j * _ctx_alloc(rig) + A) * 256
            off_f32 = (j * _ctx_alloc(rig) + A) * 8
            kq[ai, j] = _dn(rig, rig.KVQ[ai].offset(offset=off_i8, size=W * 256), (W * 256,), np.int8)
            vq[ai, j] = _dn(rig, rig.VVQ[ai].offset(offset=off_i8, size=W * 256), (W * 256,), np.int8)
            ks[ai, j] = _dn(rig, rig.KVS[ai].offset(offset=off_f32, size=W * 8), (W * 2,))
            vs[ai, j] = _dn(rig, rig.VVS[ai].offset(offset=off_f32, size=W * 8), (W * 2,))
    node["kq"], node["vq"], node["ks"], node["vs"] = kq, vq, ks, vs
    return node


def _verify_chain_artifacts_moe(root, chain, beat=None):
    """W3.1/W3.2 for the MoE format: EVERY artifact of EVERY node opened +
    size-checked + sha-verified BEFORE any upload. Returns {hk: (meta, arrs)};
    raises NodeCorrupt on the first bad node."""
    out = {}
    prev_end = None
    for hk, e in chain:
        if beat is not None:
            try: beat()
            except Exception: pass
        d = os.path.join(root, e["dir"])
        A, B = int(e["pos_start"]), int(e["pos_end"])
        if A < 0 or B <= A:
            raise NodeCorrupt(hk, f"bad window [{A},{B})")
        if prev_end is not None and A != prev_end:
            raise NodeCorrupt(hk, f"chain adjacency broken: start {A} != prev end {prev_end}")
        prev_end = B
        W = B - A
        want = {"S": (N_GDN, 32 * 128 * 128), "CS": (N_GDN, 8192 * 3),
                "kq": (N_ATTN, 2, W * 256), "vq": (N_ATTN, 2, W * 256),
                "ks": (N_ATTN, 2, W * 2), "vs": (N_ATTN, 2, W * 2)}
        try:
            meta = json.load(open(f"{d}/meta.json"))
        except Exception as ex:
            raise NodeCorrupt(hk, f"meta.json unreadable: {ex!r}")
        if meta.get("fmt") != FMT_MOE:
            raise NodeCorrupt(hk, f"node fmt {meta.get('fmt')!r} != {FMT_MOE!r} (quarantine+FRESH)")
        sizes = e.get("sizes") or {}
        shas = e.get("sha256") or {}
        arrs = {}
        for nm, shp in want.items():
            if nm not in sizes:
                raise NodeCorrupt(hk, f"missing artifact {nm} (hand-crafted/legacy manifest)")
            p = f"{d}/{nm}.npy"
            try:
                if os.path.getsize(p) != int(sizes[nm]):
                    raise NodeCorrupt(hk, f"{nm}.npy size {os.path.getsize(p)} != meta {sizes[nm]}")
                a = np.load(p, mmap_mode="r")
                if tuple(a.shape) != shp:
                    raise NodeCorrupt(hk, f"{nm}.npy shape {tuple(a.shape)} != {shp}")
                if pcache.HASH_VERIFY and nm in shas and _sha256_file(p) != shas[nm]:
                    raise NodeCorrupt(hk, f"{nm}.npy sha256 mismatch (torn/corrupt)")
                arrs[nm] = a
            except NodeCorrupt:
                raise
            except Exception as ex:
                raise NodeCorrupt(hk, f"{nm}: {ex!r}")
        out[hk] = (meta, arrs)
    return out


def restore_chain_moe(rig, chain, root, beat=None):
    """Transactional restore: validate the WHOLE chain on disk, then reset +
    upload. Returns (B, cur|None). The engine is untouched on any validation
    failure (NodeCorrupt propagates; the caller quarantines + falls back)."""
    art = _verify_chain_artifacts_moe(root, chain, beat=beat)
    B = int(chain[-1][1]["pos_end"])
    rig.reset_states(2048)            # zero S/CS + KV head hygiene
    alloc = rig.dev.allocator
    for hk, e in chain:
        meta, arrs = art[hk]
        A = int(e["pos_start"]); W = int(e["pos_end"]) - A
        for ai in range(N_ATTN):
            for j in range(2):
                off_i8 = (j * _ctx_alloc(rig) + A) * 256
                off_f32 = (j * _ctx_alloc(rig) + A) * 8
                alloc._copyin(rig.KVQ[ai].offset(offset=off_i8, size=W * 256),
                              memoryview(np.ascontiguousarray(arrs["kq"][ai, j]).data.cast("B")))
                alloc._copyin(rig.VVQ[ai].offset(offset=off_i8, size=W * 256),
                              memoryview(np.ascontiguousarray(arrs["vq"][ai, j]).data.cast("B")))
                alloc._copyin(rig.KVS[ai].offset(offset=off_f32, size=W * 8),
                              memoryview(np.ascontiguousarray(arrs["ks"][ai, j], dtype=np.float32).data.cast("B")))
                alloc._copyin(rig.VVS[ai].offset(offset=off_f32, size=W * 8),
                              memoryview(np.ascontiguousarray(arrs["vs"][ai, j], dtype=np.float32).data.cast("B")))
        if beat is not None:
            try: beat()
            except Exception: pass
    # GDN + conv state from the DEEPEST node (the running state at B)
    _, deepest = art[chain[-1][0]]
    alloc._copyin(rig.SALL, memoryview(np.ascontiguousarray(deepest["S"]).data.cast("B")))
    alloc._copyin(rig.CSALL, memoryview(np.ascontiguousarray(deepest["CS"]).data.cast("B")))
    rig.dev.synchronize()
    cur = chain[-1][1].get("cur")
    return B, (int(cur) if cur is not None else None)
