# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""MTP configuration — single source of truth. Env override + optional JSON."""
from dataclasses import dataclass, fields, asdict
import os, json

@dataclass
class MTPConfig:
    enabled: bool = True             # MTP_ENABLED           master switch
    K: int = 2                       # MTP_K                 drafted tokens per cycle (2..4); K=2 is the only validated config
    commit_mode: str = "auto"        # MTP_COMMIT            PLANNED — NOT YET READ by runner
                                     #   (v223 runner always does exact rewind+commit)
    head_fp16_act: bool = True       # MTP_HEAD_FP16         PLANNED — NOT YET READ (runner always .half())
    state_dtype: str = "fp32"        # MTP_STATE_DTYPE       PLANNED — NOT YET READ (phase C)
    draft_dtype: str = "fp16"        # MTP_DRAFT_DTYPE       fp16 | int8 (future)
    draft_vocab: str = ""            # MTP_DRAFT_VOCAB       path to ids json (phase 5.3)
    eager_cycles: int = 0            # MTP_EAGER_CYCLES      run first N cycles without chain-jit
    max_context: int = 1024
    n_gen: int = 60
    prompt: str = ("The theory of relativity transformed our understanding "
                   "of space and time.")
    dbg: bool = False                # MTPDBG
    accept_log: str = ""             # MTP_ACCEPT_LOG        jsonl path (empty = summary only;
                                     #                        sample size always printed in result)
    out_json: str = "~/tinygrad-metal/spec_out.json"
    base_json: str = "~/tinygrad-metal/spec_base.json"

    @classmethod
    def load(cls, path=None):
        cfg = cls()
        path = path or os.getenv("MTP_CONFIG", "mtp_config.json")
        if path and os.path.exists(path):
            try:
                d = json.load(open(path))
                for k, v in d.items():
                    if hasattr(cfg, k): setattr(cfg, k, v)
            except Exception as e:
                print(f"[cfg] json override failed: {e}", flush=True)
        for f in fields(cfg):
            ev = os.getenv(f"MTP_{f.name.upper()}")
            if ev is None: continue
            cur = getattr(cfg, f.name)
            if isinstance(cur, bool): setattr(cfg, f.name, ev.lower() in ("1","true","yes"))
            elif isinstance(cur, int): setattr(cfg, f.name, int(ev))
            elif isinstance(cur, float): setattr(cfg, f.name, float(ev))
            else: setattr(cfg, f.name, ev)
        return cfg

    def dump(self):
        return json.dumps(asdict(self), indent=2)
