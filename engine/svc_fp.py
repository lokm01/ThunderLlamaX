# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
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
)


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
    return hashlib.sha256(json.dumps(kv, sort_keys=True).encode()).hexdigest()[:16]


def parse_env_file(path):
    """KEY=VALUE lines (comments with #; optional leading 'export '). Returns
    {} on missing/empty file — callers treat that as 'no canonical env'."""
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
                out[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        return {}
    return out
