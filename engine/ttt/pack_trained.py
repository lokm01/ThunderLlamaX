# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 1 — pack writer: torch bf16/fp32 -> the engine's Q4_0
# two-region draft_pack layout (+ Q8_0 variant + GPTQ-calibrated option).
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Writes draft_pack_v2/<name>/ with the EXACT files the engine loads
(mtp.py init_draft: d_eh d_q d_k d_v d_o d_fg d_fu d_fd + 7 f32 norms):
  Q4 rows: [qs nin//2 B][d nin//16 B], element e of a 32-block at byte (e&15)
  nibble (e>>4) (q4v.cu truth), scale d = fp16 absmax/8 per 32-block.
  norms: f32 .npy, d_qnw/d_knw exactly (256,).
Modes: rtn (default) | gptq (calibrated error-feedback quantization from the
Hessians dumped by `train.py --mode calib`; per-32-block FIXED absmax/8 scales
so the byte format stays identical — the algorithm optimizes the nibbles).
--q8 additionally writes the Q8_0 analog (engine kernels don't exist yet —
writer-only, per the plan's "if trivial").
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q4pack_lib import Q4_ROWS, NORM_SHAPES, quantize_q4_rtn, quantize_q8_rtn, dequant_q4, pack_row_bytes

TMAP = {
    "mtp.fc.weight": "d_eh",
    "mtp.layers.0.self_attn.q_proj.weight": "d_q",
    "mtp.layers.0.self_attn.k_proj.weight": "d_k",
    "mtp.layers.0.self_attn.v_proj.weight": "d_v",
    "mtp.layers.0.self_attn.o_proj.weight": "d_o",
    "mtp.layers.0.mlp.gate_proj.weight": "d_fg",
    "mtp.layers.0.mlp.up_proj.weight": "d_fu",
    "mtp.layers.0.mlp.down_proj.weight": "d_fd",
    "mtp.layers.0.input_layernorm.weight": "d_nw1",
    "mtp.layers.0.post_attention_layernorm.weight": "d_nw2",
    "mtp.norm.weight": "d_shnw",
    "mtp.pre_fc_norm_embedding.weight": "d_enw",
    "mtp.pre_fc_norm_hidden.weight": "d_hnw",
    "mtp.layers.0.self_attn.q_norm.weight": "d_qnw",
    "mtp.layers.0.self_attn.k_norm.weight": "d_knw",
}
ALT = {  # fetch_weights.py flat names -> HF names
    "mtp_fc_weight": "mtp.fc.weight",
    "mtp_layers_0_self_attn_q_proj_weight": "mtp.layers.0.self_attn.q_proj.weight",
    "mtp_layers_0_self_attn_k_proj_weight": "mtp.layers.0.self_attn.k_proj.weight",
    "mtp_layers_0_self_attn_v_proj_weight": "mtp.layers.0.self_attn.v_proj.weight",
    "mtp_layers_0_self_attn_o_proj_weight": "mtp.layers.0.self_attn.o_proj.weight",
    "mtp_layers_0_mlp_gate_proj_weight": "mtp.layers.0.mlp.gate_proj.weight",
    "mtp_layers_0_mlp_up_proj_weight": "mtp.layers.0.mlp.up_proj.weight",
    "mtp_layers_0_mlp_down_proj_weight": "mtp.layers.0.mlp.down_proj.weight",
    "mtp_layers_0_input_layernorm_weight": "mtp.layers.0.input_layernorm.weight",
    "mtp_layers_0_post_attention_layernorm_weight": "mtp.layers.0.post_attention_layernorm.weight",
    "mtp_norm_weight": "mtp.norm.weight",
    "mtp_pre_fc_norm_embedding_weight": "mtp.pre_fc_norm_embedding.weight",
    "mtp_pre_fc_norm_hidden_weight": "mtp.pre_fc_norm_hidden.weight",
    "mtp_layers_0_self_attn_q_norm_weight": "mtp.layers.0.self_attn.q_norm.weight",
    "mtp_layers_0_self_attn_k_norm_weight": "mtp.layers.0.self_attn.k_norm.weight",
}
# GEMV-input Hessian (train.py calib) consumed by each Q4 tensor
HMAP = {"d_eh": "d_eh", "d_q": "d_qkvxh", "d_k": "d_qkvxh", "d_v": "d_qkvxh",
        "d_o": "d_ao", "d_fg": "d_fgx", "d_fu": "d_fgx", "d_fd": "d_fdg"}


def load_sd(path):
    if path.endswith(".pt"):
        return {k: v.float().numpy() for k, v in torch.load(path, map_location="cpu", weights_only=False)["sd"].items()}
    sd = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".npy") and (f[:-4] in ALT or f[:-4] in TMAP):
            k = f[:-4]
            sd[ALT.get(k, k)] = np.load(os.path.join(path, f))
    return sd


def gptq_quantize(W, H, damp=0.01, block=32, batch=64, device=None):
    """GPTQ with the engine's FIXED per-32-block fp16 absmax/8 scales (format
    byte-identical to RTN; only the nibbles are optimized). fp32 compute
    (Blackwell/Ada fp64 is 1:64); fp64 only for the Cholesky inverse.
    W fp32 [nout, nin]; H [nin, nin] fp64 = X^T X."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    nout, nin = W.shape
    H64 = torch.from_numpy(np.ascontiguousarray(H)).to(torch.float64).to(dev)
    diag = torch.diagonal(H64)
    dead = diag == 0
    if dead.any():
        H64[dead, dead] = 1.0
    H64[range(nin), range(nin)] += damp * torch.diagonal(H64).mean()
    Linv = torch.linalg.solve_triangular(torch.linalg.cholesky(H64), torch.eye(nin, dtype=torch.float64, device=dev), upper=False)
    Hinv = (Linv.T @ Linv).to(torch.float32)
    H = H64.to(torch.float32)
    del H64
    Wcur = torch.from_numpy(W).to(torch.float32).to(dev)
    scale = torch.zeros(nout, nin // block, dtype=torch.float32, device=dev)
    q_int = torch.zeros(nout, nin, dtype=torch.int16, device=dev)
    E = torch.zeros(nout, batch, dtype=torch.float32, device=dev)
    for c0 in range(0, nin, batch):
        c1 = min(c0 + batch, nin)
        Wblk = Wcur[:, c0:c1].clone()
        Hinv_blk = Hinv[c0:c1, c0:c1]
        for k in range(c1 - c0):
            col = c0 + k
            blk, cib = col // block, col % block
            if cib == 0:
                bmax = Wcur[:, blk * block:(blk + 1) * block].abs().amax(dim=1)
                scale[:, blk] = torch.clamp(bmax / 8.0, min=1e-12)
            d_col = scale[:, blk]
            w = Wblk[:, k]
            q = torch.clamp(torch.round(w / d_col + 8.0), 0, 15)
            q_int[:, col] = q.to(torch.int16)
            err = (w - d_col * (q - 8.0)) / Hinv_blk[k, k]
            if k + 1 < c1 - c0:
                Wblk[:, k + 1:] -= torch.outer(err, Hinv_blk[k, k + 1:])
            E[:, col - c0] = err
        if c1 < nin:
            Wcur[:, c1:] -= E @ Hinv[c0:c1, c1:]   # GPTQ lazy update uses Hinv (the inverse)
    q_np = q_int.cpu().numpy().astype(np.uint8)
    d_np = scale.cpu().numpy().astype(np.float16)
    q4 = q_np.reshape(nout, nin // block, block)
    qs = (q4[:, :, :16] | (q4[:, :, 16:] << 4)).reshape(nout, nin // 2)
    out = np.zeros((nout, pack_row_bytes(nin)), np.uint8)
    out[:, : nin // 2] = qs
    out[:, nin // 2:] = d_np.view(np.uint8).reshape(nout, nin // 16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".pt checkpoint or weights dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="rtn", choices=["rtn", "gptq"])
    ap.add_argument("--calib", default=None, help="calib dir with *_hess.npy (gptq)")
    ap.add_argument("--q8", action="store_true")
    a = ap.parse_args()
    sd = load_sd(a.ckpt)
    missing = [k for k in TMAP if k not in sd]
    assert not missing, f"missing tensors: {missing}"
    os.makedirs(a.out, exist_ok=True)
    manifest = {"ckpt": os.path.abspath(a.ckpt), "mode": a.mode, "q8": a.q8, "tensors": {}}
    for hf_name, pack_name in TMAP.items():
        W = sd[hf_name].astype(np.float32)
        if pack_name in Q4_ROWS:
            nout, nin = Q4_ROWS[pack_name]
            assert W.shape == (nout, nin), (hf_name, W.shape, (nout, nin))
            t0 = time.time()
            if a.mode == "rtn":
                packed = quantize_q4_rtn(W)
            else:
                H = np.load(f"{a.calib}/{HMAP[pack_name]}_hess.npy")
                packed = gptq_quantize(W, H)
            if a.q8:
                np.save(f"{a.out}/{pack_name}_q8.npy", quantize_q8_rtn(W))
            back = dequant_q4(packed, nout, nin)
            cos = float((W * back).sum() / (np.linalg.norm(W) * np.linalg.norm(back)))
            np.save(f"{a.out}/{pack_name}.npy", packed)
            manifest["tensors"][pack_name] = {"cos": round(cos, 6), "bytes": int(packed.nbytes)}
            print(f"[pack] {pack_name:6s} {packed.shape} cos={cos:.5f} ({time.time()-t0:.1f}s)", flush=True)
        else:
            assert W.shape == (NORM_SHAPES[pack_name],), (hf_name, W.shape)
            Wf = W + 1.0  # HF stores zero-centered; the engine uses functional form
            np.save(f"{a.out}/{pack_name}.npy", Wf.astype(np.float32))
            print(f"[pack] {pack_name:6s} f32 {W.shape} (+1.0 norm law)")
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=1)
    print(f"[pack done] {a.out}")


if __name__ == "__main__":
    main()
