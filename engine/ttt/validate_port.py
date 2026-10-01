# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 1 — Validation A: the port proves itself against the engine.
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""Run BEFORE any spend. FINDING (2026-09-30): the rig GGUF (trunk + draft) is a
different checkpoint lineage from HF Qwen/Qwen3.8-27B — its f32 norm tensors are
incompatible with HF's (GGUF trunk final norm mean 1.94 all-positive vs HF 0.047
signed; q/k norms likewise), so a pack<->HF cosine check is meaningless across
checkpoints. The pack IS the engine's own self-consistent convention; the port
proof is therefore:

  A1 layout: pack shapes byte-identical to Q4_ROWS/NORM_SHAPES + my
     quantize->dequant roundtrip on random tensors (the writer's layout law).
  A2 structural: torch module (pack-dequant weights, fp32) vs engine_ref pure
     fp32 chain replay on a 6-step serve chain — relerr ~1e-5 required (a
     structural port bug gives ~1.0).
  A3 informational: engine fp16-emulated chain vs fp32 (the engine's own fp16
     noise on the REAL current drafter) — feeds G1's precision decomposition.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_ref as ER
from model import DraftBlock, rms_norm
from q4pack_lib import Q4_ROWS, NORM_SHAPES, load_pack, quantize_q4_rtn, dequant_q4

PACK = os.path.expanduser("~/drafter/ref_pack")


def relerr(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))


def torch_chain(module, emb_fn, h_seed, toks, pos0):
    NKV, HD, CTX = 4, 256, 4096
    K = torch.zeros(1, NKV, CTX, HD)
    V = torch.zeros(1, NKV, CTX, HD)
    hds, his = [], []
    hm = torch.from_numpy(h_seed.astype(np.float32))
    for i, t in enumerate(toks):
        te = torch.from_numpy(emb_fn(t).astype(np.float32)).unsqueeze(0)
        hmb = hm.unsqueeze(0)
        pos = torch.tensor([pos0 + i])
        xin = module.xin_from(te, hmb)
        xh = rms_norm(xin, module.attn_norm_w)
        q, k, v = module.qkv_of(xh)
        qq, g = module.qvec_gate(q, pos)
        kk = module.kvecs(k, pos).to(torch.float16).float()   # engine kv_d is fp16
        vv = v.float().reshape(1, NKV, HD).to(torch.float16).float()
        K[:, :, pos0 + i, :] = kk
        V[:, :, pos0 + i, :] = vv
        qq = qq / 16.0
        sc = module.attn_scores(qq, K[:, :, : pos0 + i + 1, :])
        p = torch.softmax(sc, dim=-1)
        o = torch.matmul(p.unsqueeze(2), V[:, :, : pos0 + i + 1, :].repeat_interleave(6, dim=1))
        o = o.squeeze(2).reshape(1, 24 * HD)
        ao = o * torch.sigmoid(g.reshape(1, 24 * HD))
        hd = module.block_tail(xin, ao)
        hi = rms_norm(hd, module.shared_head_norm_w)
        hds.append(hd[0].numpy())
        his.append(hi[0].numpy())
        hm = hd[0]
    return np.stack(hds), np.stack(his)


HF2 = os.path.expanduser("~/drafter/weights_v2")
V2N = {  # corrected fetch names -> pack names (weight-mapping proof)
    "mtp_fc_weight.npy": "d_eh", "mtp_layers_0_self_attn_q_proj_weight.npy": "d_q",
    "mtp_layers_0_self_attn_k_proj_weight.npy": "d_k", "mtp_layers_0_self_attn_v_proj_weight.npy": "d_v",
    "mtp_layers_0_self_attn_o_proj_weight.npy": "d_o", "mtp_layers_0_mlp_gate_proj_weight.npy": "d_fg",
    "mtp_layers_0_mlp_up_proj_weight.npy": "d_fu", "mtp_layers_0_mlp_down_proj_weight.npy": "d_fd",
    "mtp_norm_weight.npy": "d_shnw", "mtp_pre_fc_norm_embedding_weight.npy": "d_enw",
    "mtp_pre_fc_norm_hidden_weight.npy": "d_hnw", "mtp_layers_0_input_layernorm_weight.npy": "d_nw1",
    "mtp_layers_0_post_attention_layernorm_weight.npy": "d_nw2",
    "mtp_layers_0_self_attn_q_norm_weight.npy": "d_qnw", "mtp_layers_0_self_attn_k_norm_weight.npy": "d_knw",
}


def main():
    print("== loading pack ==")
    w = load_pack(PACK)
    if os.path.isdir(HF2) and len(os.listdir(HF2)) >= 15:
        print("\n== A0 weight mapping: pack-dequant vs CORRECTED HF bf16 ==")
        print("   (the GGUF is the first-party checkpoint; Q4_0-RTN cos ~0.995 expected;")
        print("    norms: pack == HF_stored + 1.0 — the NORM LAW)")
        ok = True
        for f, pn in V2N.items():
            hf = np.load(f"{HF2}/{f}").astype(np.float32)
            pk = w[pn]
            if pn in ("d_qnw", "d_knw"):
                pk = pk[:256]
            if pn in ("d_shnw", "d_enw", "d_hnw", "d_nw1", "d_nw2", "d_qnw", "d_knw"):
                d = float(np.abs(pk - (hf + 1.0)).max())
                line = f"  {pn:6s} |pack-(hf+1)|max = {d:.2e}"
                ok &= d < 1e-3
            else:
                c = float((hf * pk).sum() / (np.linalg.norm(hf) * np.linalg.norm(pk)))
                line = f"  {pn:6s} cos = {c:.5f}"
                ok &= c > 0.99
            print(line)
        assert ok, "A0 FAILED"
        print("  A0 PASS (first-party provenance + norm law)")

    print("\n== A1 layout: pack shapes + quantizer roundtrip ==")
    for nm, (nout, nin) in Q4_ROWS.items():
        arr = np.load(f"{PACK}/{nm}.npy")
        assert arr.shape == (nout, (nin // 256) * 144), (nm, arr.shape)
        print(f"  {nm:6s} {arr.shape} OK")
    for nm, n in NORM_SHAPES.items():
        a = np.load(f"{PACK}/{nm}.npy")
        assert a.shape == (n,) and a.dtype == np.float32, (nm, a.shape, a.dtype)
    rng = np.random.default_rng(1)
    # exact byte-placement law: constructed q patterns land where q4v.cu reads them
    for nin in (5120, 10240, 17408, 6144):
        nout, d0 = 8, 0.03
        qw = (np.arange(32) * 7 + 3) % 16          # distinct nibble per element
        wlaw = np.tile(d0 * (qw.astype(np.float64) - 8), (nout, nin // 32))
        packed = quantize_q4_rtn(wlaw.astype(np.float32))
        back = dequant_q4(packed, nout, nin)
        expect_w = np.tile(((qw.astype(np.float32) - 8) * np.float16(d0).astype(np.float32)), (nout, nin // 32))
        ok = np.allclose(back, expect_w, atol=1e-6)
        # nibble placement: element e at byte (e&15) nibble (e>>4) of its 16B sub-block
        qs = packed[0, :16].astype(np.uint8)
        expect = np.zeros(16, np.uint8)
        for e in range(32):
            expect[e & 15] |= (qw[e] << 4) if (e >> 4) else qw[e]
        assert ok and (qs == expect).all() and packed.shape[1] == (nin // 256) * 144, (nin, ok)
        x = rng.standard_normal((32, nin)) * 0.02
        back2 = dequant_q4(quantize_q4_rtn(x), 32, nin)
        cos = float((x * back2).sum() / (np.linalg.norm(x) * np.linalg.norm(back2)))
        assert cos > 0.99
        print(f"  layout nin={nin}: byte-law exact, RTN cos={cos:.5f} OK")
    print("  A1 PASS")

    print("\n== A2 structural: torch(fp32, pack w) vs engine_ref fp32 chain ==")
    rng = np.random.default_rng(0)
    embtab = (rng.standard_normal((256, 5120)) * 0.02).astype(np.float32)
    embs = lambda t: embtab[int(t) & 255]
    h_seed = (rng.standard_normal(5120) * 8.0).astype(np.float32)
    toks = list(rng.integers(0, 256, 6))
    pos0 = 977
    mod = DraftBlock.from_pack(w, torch.float32)
    # step-1 strict structural check (fp32 noise only; fp16 KV round-trip matched)
    hd_np1, _ = ER.chain(w, embs, h_seed, toks[:1], pos0, emulate_fp16=False)
    with torch.no_grad():
        hd_t1, _ = torch_chain(mod, embs, h_seed, toks[:1], pos0)
    e1 = relerr(hd_t1, hd_np1)
    # 6-step chain: hm feedback amplifies fp32 noise ~2x/step (chaotic chaining);
    # a structural bug would sit at ~1e0 regardless of depth
    hd_np, hi_np = ER.chain(w, embs, h_seed, toks, pos0, emulate_fp16=False)
    with torch.no_grad():
        hd_t, hi_t = torch_chain(mod, embs, h_seed, toks, pos0)
    e_hd = relerr(hd_t, hd_np)
    e_hi = relerr(hi_t, hi_np)
    print(f"  step-1 hd relerr = {e1:.2e} (strict)")
    print(f"  6-step hd relerr = {e_hd:.2e}  head_in = {e_hi:.2e} (amplified fp32 noise)")
    assert e1 < 1e-5 and e_hd < 2e-3, "STRUCTURAL MISMATCH"
    print("  A2 PASS (structural port exact to fp32 noise)")

    print("\n== A3 engine fp16 penalty on the current drafter ==")
    hd16, hi16 = ER.chain(w, embs, h_seed, toks, pos0, emulate_fp16=True)
    print(f"  hd relerr = {relerr(hd16, hd_np):.3e}  (fp16 noise floor, informational)")
    print("\nVALIDATION A: ALL PASS")


if __name__ == "__main__":
    main()
