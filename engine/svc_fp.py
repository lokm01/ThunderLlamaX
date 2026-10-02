"""TLX W2 (ledger V-28): the ONE config-fingerprint implementation shared by
pcache.py (engine side), serve.py (daemon status) and api_server.py (drift alarm).

ZERO heavy dependencies (stdlib only, numpy-free) so the API process — system
python3 without numpy — imports it freely.

Fingerprint inputs:
  1. The behavior-knob env set (_ENV_KEYS): every env that changes engine
     numerics or decode behavior. pcache mixes this into the chain ROOT, so
     nodes are valid only under a bit-identical config.
  2. The MODEL FILE identity: size + mtime + sha256 of the first 1 MiB.
     Two boots with KV-identical env but a different/touched GGUF must not
     validate each other's cache or report "same config".

W2 additions vs the pcache-only set (V-28): PF_PREFILL, LOOKUP, LOOKUP_K,
M1A_GEN_REBUILD_EVERY, MTP_KERNARGS_MB, PF_ATTNW, PF_SCANC_N2, PF_M64QKV,
PF_ABW, PG_SPLIT, NV_SMEM_CFG_AUTO, NV_SMEM_CFG_AUTO_NAMES.

NOTE (cache-invalidation event, deliberate): extending _ENV_KEYS changes
config_fp for every existing node -> one cold pcache rebuild on the next boot.
Announce + slog it (ledger W2.4 dependency note); do not ship mid-benchmark.
"""
import os, json, hashlib

FMT = "r1pc1"
DEFAULT_MODEL = "~/tinygrad-metal/models/Qwen3.8-27B-IQ3_XXS.gguf"

# Behavior knobs (engine numerics + decode behavior). Keep SORTED-agnostic:
# config_fp() sorts the dict before hashing.
_ENV_KEYS = (
    "SKV", "SKV_K", "SKV_S", "SKV_CTXK", "GEMVV", "KV8", "QH", "PVH", "HM",
    "K3", "SG",
    "PF_M32", "PF_DFILL", "PF_PG", "PF_M64", "PF_GEMM3", "PF_ATTN32", "PF_N32",
    "PF_PRE32", "PF_SCAN32", "PF_A4", "PF_SCANC", "PF_SCANC_N2", "PF_DR7",
    "PF_FFNSPLIT", "PF_M128", "PF_RING4", "PF_QKV1", "PF_PRE32X4", "PF_P5",
    "PF_OP64", "PF_W4A8", "PF_W4A8_MB", "PF_ATTNW", "PF_M64QKV", "PF_ABW",
    "PG_SPLIT", "NV_SMEM_CFG_AUTO", "NV_SMEM_CFG_AUTO_NAMES",
    # W2 (V-28): behavior knobs that were silently missing
    "PF_PREFILL", "LOOKUP", "LOOKUP_K", "M1A_GEN_REBUILD_EVERY", "MTP_KERNARGS_MB",
    # R3-19: the batch knobs (a BATCH_B=2 boot or a BATCH_PF_CHUNK/R6_PF_T1/
    # PF_G3M_MB flip without env edits used to keep the SAME fp and
    # restore_chain uploaded old KV into a different engine — the
    # wrong-answer-with-/health-ok class)
    "BATCH_B", "BATCH_REBUILD_EVERY", "BATCH_PF_CHUNK", "R6_PF_T1", "PF_G3M_MB",
    # P10-dense rung 2: the K=4 EAGLE chain (TLX_EAGLE_K=4 swaps the miss-cycle
    # probe/accept set to the R4 M=5 pair — same class as LOOKUP_K: decode
    # behavior, not prefill numerics). CAche-invalidation event: one cold
    # pcache rebuild per model on the first boot after this (slogged).
    "TLX_EAGLE_K",
    # MM SESSION B (MoE prefill): the grouped-expert / seat-loop / PF64-tail
    # graph knobs -- they change the PF graph KERNEL SET (outputs bit-exact
    # by the G2/F1b gates, but pcache nodes must never cross kernel-set
    # boundaries). Cache-invalidation event: ONE cold MoE pcache rebuild on
    # the first boot after this ships (slogged).
    "MM_PFG", "MM_PFM", "MM_PF64",
)

# R3-19: the cubin set the batch scheduler loads BY PATH (r6_serve.r6_boot).
# Kept HERE (stdlib-only module) so the API process can fingerprint the same
# set the daemon loads — r6_serve itself imports numpy/tinygrad.
CUBINS_NEEDED = (
    "h_embed5", "k2s5", "accept5k", "acceptsel5k", "lookup5_nw32",
    "k0n10", "k0ab10", "q5g8v10", "aq3k8v10", "aq6k8v10", "ao8nw32_10",
    "k3aonw32_10", "op38nw32_10", "hh10", "ffn8v10r7", "down8nw32v10r7", "head8v10",
    # pcache restore's cur-derivation fallback (nodes captured without
    # meta cur, e.g. legacy PF-world nodes restoring into this daemon)
    "pfk_n16",
)

_EXTRA_FP = {}
def set_extra(name, value):
    """R3-19: an additional fp input computed at boot (outside _ENV_KEYS).
    Both the daemon AND api_server set the same extras before hashing."""
    _EXTRA_FP[str(name)] = str(value)

# ---- TLX P8 (MoE bridge): the per-model EXTRA protocol -----------------------
# The dense daemon mixes the cubin-set digest; the MoE daemon mixes the packed-
# weights manifest sha (a repack changes numerics with NO env change). For the
# API's canonical fp to MATCH the resident daemon, BOTH sides must derive the
# SAME extras for the resident model. The helpers below are the shared
# implementation (stdlib-only; svc_fp stays importable by the system python3).
def clear_extras():
    """Reset to the empty extra set (per-model derivation start)."""
    _EXTRA_FP.clear()

def set_mm_pack_extra(env):
    """The MoE engine's packed-weights fp input: sha256 of the packed
    manifest (the pack the rig loads — the weights-of-record for numerics).
    env: the model's env dict (MM_PACKED key) or the daemon environment."""
    mp = (env or {}).get("MM_PACKED") or "~/models36/packed/qwen3.6-35b-a3b-iq4_xs"
    try:
        h = hashlib.sha256()
        with open(os.path.join(mp, "manifest.json"), "rb") as f:
            h.update(f.read(1 << 20))
        _EXTRA_FP["mm_pack"] = h.hexdigest()[:16]
        return _EXTRA_FP["mm_pack"]
    except Exception:
        _EXTRA_FP["mm_pack"] = "missing"
        return "missing"

def set_draft_pack_extra(pack_dir=None):
    """TLX P0-S2 (qwen finding): the draft pack is NOT hashed in config_fp yet
    pcache persists drafter-conditioned state (kvd/dhd in pcache._ART). A pack
    swap with no env change must invalidate the cache. Hash = sha256 over the
    sorted (name, size, content) of the pack dir the ENGINE loads (mtp.DPACK:
    TLX_DRAFT_PACK env or engine0/draft_pack). Stdlib-only (the API process
    derives the SAME extra)."""
    import os as _os
    if pack_dir is None:
        base = _os.path.dirname(_os.path.abspath(__file__))
        pack_dir = _os.environ.get("TLX_DRAFT_PACK") or _os.path.join(base, "draft_pack")
    h = hashlib.sha256()
    try:
        for fn in sorted(_os.listdir(pack_dir)):
            if not fn.endswith(".npy"):
                continue
            p = _os.path.join(pack_dir, fn)
            st = _os.stat(p)
            fh = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    fh.update(chunk)
            h.update(f"{fn}:{st.st_size}:{fh.hexdigest()[:16]}\n".encode())
        _EXTRA_FP["draft_pack"] = h.hexdigest()[:16]
    except Exception:
        _EXTRA_FP["draft_pack"] = "missing"
    return _EXTRA_FP["draft_pack"]

def extra_fp():
    return dict(_EXTRA_FP)

def cubin_set_digest(names=CUBINS_NEEDED, base_dir=None):
    """R3-19: sha256 over (name, size, content-hash) of the cubin set —
    GPU-free (bytes only). A rebuilt kernel changes config_fp even when no
    env knob moved. Missing files hash deterministically ('missing')."""
    base = base_dir or os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for n in sorted(names):
        p = os.path.join(base, f"{n}.cubin")
        try:
            st = os.stat(p)
            fh = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    fh.update(chunk)
            h.update(f"{n}:{st.st_size}:{fh.hexdigest()[:16]}\n".encode())
        except Exception:
            h.update(f"{n}:missing\n".encode())
    return h.hexdigest()[:16]


def model_identity(path):
    """size+mtime+first-1MiB-sha identity of the model file (cheap on 12GB files:
    one 1MiB read). Missing file -> 'none' (fp still differs from any real file)."""
    try:
        st = os.stat(path)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(1024 * 1024))
        return f"sz{st.st_size}:mt{int(st.st_mtime)}:sha{h.hexdigest()[:16]}"
    except Exception:
        return "none"


def config_fp(env=None, model_path=None):
    """16-hex config fingerprint. env: mapping (os.environ default; the API
    passes the parsed ops/env.canonical dict). model_path: the GGUF the ENGINE
    loads (env TLX_MODEL_PATH wins, then the argument, then DEFAULT_MODEL)."""
    e = os.environ if env is None else env
    path = model_path or e.get("TLX_MODEL_PATH") or DEFAULT_MODEL
    kv = {k: e.get(k) for k in _ENV_KEYS}
    kv["fmt"] = FMT
    kv["model"] = model_identity(path)
    kv["extra"] = dict(_EXTRA_FP)      # R3-19: boot-computed inputs (cubins)
    return hashlib.sha256(json.dumps(kv, sort_keys=True).encode()).hexdigest()[:16]


def parse_env_file(path):
    """KEY=VALUE lines (comments with #; optional leading 'export '). Returns
    {} on missing/empty file — callers treat that as 'no canonical env'.
    R3-20: ONE parser that agrees with `zsh source` semantics — the wrapper
    sources this file (zsh expands a leading ~ and strips unquoted # comments)
    while the old python parser did NEITHER: the published
    env.canonical.example carries ~/tinygrad-metal paths, and a user who
    skipped the sed kept a tilde path -> daemon fp = real-file hash vs
    expected fp = model_identity('none') -> PERMANENT 503 config_drift."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" not in line:
                    continue
                k, _, v = line.partition("=")
                v = v.strip()
                if v[:1] in ('"', "'"):
                    q = v[0]
                    end = v.find(q, 1)
                    if end > 0:
                        v = v[1:end]           # quoted: take the inside verbatim
                elif " #" in v:
                    # zsh (non-interactive, default) does NOT strip an
                    # unquoted inline comment — the assignment line becomes a
                    # COMMAND and the var is NEVER SET. Agree exactly: skip
                    # the key (a drifted/typo'd canonical file surfaces as
                    # the missing var on BOTH sides, not a silent mismatch).
                    continue
                else:
                    v = v.strip('"').strip("'")
                if v.startswith("~"):
                    v = os.path.expanduser(v)  # zsh tilde expansion
                out[k.strip()] = v
    except Exception:
        return {}
    return out
