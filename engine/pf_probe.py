# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""P7E5 forensics: dump pfca outputs (kh/qe/vr/meta) for GDN block j=0, h=0,
c=0 (tokens 0-3) under the C3 seed (row2 only) and diff vs a numpy oracle of
the pfs16 VERBATIM formulas. Exit leaves GPU clean."""
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
from engine0 import dev
from trunk_w1c import TrunkEngineW1C
import pf_prefill

E = TrunkEngineW1C(theta=1e7)
P, d = E.P, E.P.d
import json as _j; ids = np.array(_j.load(open("~/ids8k.json"))[:256], dtype=np.int32)
NCH = 10240
SNAPC = {i: np.load(f"~/snap100k/conv_{i}.npy").reshape(3,10240) for i in E.gdn_idx}
NC, C, LDK, LDTC, HCB = 4, 64, 136, 72, 174608

def reset():
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32)); dev.synchronize()
  for i in E.attn_idx:
    P.win_up(f"kv{i}", 0, np.zeros(2*4*100352*256, dtype=np.uint8)); P._keep.clear()
  for i in E.gdn_idx:
    z = np.zeros_like(SNAPC[i]); z[2] = SNAPC[i][2]   # C3: row2 only
    P.win_up(f"conv{i}_0", 0, z.reshape(-1))
    P.win_up(f"conv{i}_1", 0, np.zeros(3*10240, dtype=np.float32))
    P.win_up(f"rec{i}", 0, np.zeros(48*128*128, dtype=np.float32))
  dev.synchronize(); P._keep.clear()
  P.win_up("pos_slot", 0, np.array([0], dtype=np.int32)); dev.synchronize()

class G: pass
os.environ["PF_SUPER"] = "1"
reset()
# wrap pfcb: on FIRST call (= GDN block j=0) sync + snapshot qkvsc/scscr/arawsc
pf_prefill.ensure(E); pf_prefill.ensure_sc(E)   # loads cubins + builds the plan (guard: second call is a no-op)
_orig_pfcb = E.pr["pfcb_c64_nc4_nw8"]
_snap = {}
_calls = [0]
def _wrap(*a, **k):
  _orig_pfcb(*a, **k)
  _calls[0] += 1
  if _calls[0] == 1:
    dev.synchronize()
    _snap["qkv"] = P.down("qkvsc", (256,10240), np.float16).copy()
    _snap["scr"] = P.down("scscr", (48*NC*HCB//2,), np.uint16).copy()
    _snap["araw"] = P.down("arawsc", (256*48,), np.float32).copy()
    _snap["braw"] = P.down("brawsc", (256*48,), np.float32).copy()
    _snap["sco"] = P.down("sco", (256,6144), np.float32).copy()
    P._keep.clear()
    print("[wrap] block-0 snapshot taken", flush=True)
plan = E._pfsc_plan
E._pfsc_plan = [( (_wrap if ent[0] is _orig_pfcb else ent[0]),) + ent[1:] for ent in plan]
pf_prefill.prefill_batch(E, G, ids); dev.synchronize()
scr = _snap["scr"].view(np.float16).astype(np.float32)
qkv = _snap["qkv"]; araw = _snap["araw"].reshape(256,48)
hc = scr[(0*NC+0)*(HCB//2):]
kh = hc[0:C*LDK].reshape(C, LDK)[:, :128]
qe = hc[(C*LDK + 128*LDTC):(C*LDK + 128*LDTC) + C*LDK].reshape(C, LDK)[:, :128]
vr = hc[(3*C*LDK + 128*LDTC):(3*C*LDK + 128*LDTC) + C*LDK].reshape(C, LDK)[:, :128]
vrl = hc[(5*C*LDK + 128*LDTC + 2*C*LDTC):(5*C*LDK + 128*LDTC + 2*C*LDTC) + C*LDK].reshape(C, LDK)[:, :128]
met = hc[HCB//2 - 70:HCB//2].view(np.uint16).view(np.float32) if False else None
# meta: read via byte offset
meta_off = ((HCB//2 - (2*C+4)) )  # in halfs approx; do exact below
scwp = P.down("scwp", (48*47200,), np.float32).copy(); P._keep.clear()
convw = scwp[0:40960].reshape(10240,4); dtb = scwp[40960:41008]; ssma = scwp[41008:41056]
cv0 = SNAPC[E.gdn_idx[0]]
cv = np.zeros_like(cv0); cv[2] = cv0[2]   # the C3 seed actually uploaded

# ---- numpy oracle (pfs16 formulas verbatim) for tokens 0..3, head 0 ----
h = 0; kh_h = h % 16; qc0, kc0, vc0 = kh_h*128, 2048+kh_h*128, 4096+h*128
def sig(x): return 1.0/(1.0+np.exp2(x*(-1.4426950408889634)))
def softplus(x): return np.maximum(x,0)+np.log1p(np.exp(-np.abs(x)))
print("tok | qe_maxdiff kh_maxdiff vr_maxdiff | lam exp")
for t in range(4):
    q2 = qkv[t, qc0:qc0+128].astype(np.float64); k2 = qkv[t, kc0:kc0+128].astype(np.float64); v2 = qkv[t, vc0:vc0+128].astype(np.float64)
    w0 = qkv[t-3, qc0:qc0+128].astype(np.float64) if t>=3 else cv[t+0, qc0:qc0+128]
    w1 = qkv[t-2, qc0:qc0+128].astype(np.float64) if t>=2 else cv[t+1, qc0:qc0+128]
    w2 = qkv[t-1, qc0:qc0+128].astype(np.float64) if t>=1 else cv[t+2, qc0:qc0+128]
    sq = w0*convw[qc0:qc0+128,0]+w1*convw[qc0:qc0+128,1]+w2*convw[qc0:qc0+128,2]+q2*convw[qc0:qc0+128,3]
    w0k = qkv[t-3, kc0:kc0+128].astype(np.float64) if t>=3 else cv[t+0, kc0:kc0+128]
    w1k = qkv[t-2, kc0:kc0+128].astype(np.float64) if t>=2 else cv[t+1, kc0:kc0+128]
    w2k = qkv[t-1, kc0:kc0+128].astype(np.float64) if t>=1 else cv[t+2, kc0:kc0+128]
    sk = w0k*convw[kc0:kc0+128,0]+w1k*convw[kc0:kc0+128,1]+w2k*convw[kc0:kc0+128,2]+k2*convw[kc0:kc0+128,3]
    w0v = qkv[t-3, vc0:vc0+128].astype(np.float64) if t>=3 else cv[t+0, vc0:vc0+128]
    w1v = qkv[t-2, vc0:vc0+128].astype(np.float64) if t>=2 else cv[t+1, vc0:vc0+128]
    w2v = qkv[t-1, vc0:vc0+128].astype(np.float64) if t>=1 else cv[t+2, vc0:vc0+128]
    sv = w0v*convw[vc0:vc0+128,0]+w1v*convw[vc0:vc0+128,1]+w2v*convw[vc0:vc0+128,2]+v2*convw[vc0:vc0+128,3]
    sq = sq*sig(sq); sk = sk*sig(sk); sv = sv*sig(sv)
    qn = (1.0/max(np.sqrt((sq*sq).sum()), 1e-6)) * 0.08838834764831845
    kn = 1.0/max(np.sqrt((sk*sk).sum()), 1e-6)
    lam = softplus(araw[t,h]+dtb[h])*ssma[h]
    g = lam*1.4426950408889634
    print(f"{t} | qe {np.abs(qe[t]-sq*qn*np.exp2(g)).max():.2e} kh {np.abs(kh[t]-sk*kn).max():.2e} vr {np.abs((vr[t]+vrl[t])-sv).max():.2e} | {lam:.5f}")

# ---- window-0 scan oracle (pfs16 recurrence, fp64) vs SC oout rows 0..15 ----
sco = _snap["sco"]
braw = _snap["araw"]  # wrong name guard
arw = _snap["araw"].reshape(256,48)
# re-derive brw: not dumped; use meta bg from scratch instead: beta_i*2^g_i
meta_off_halfs = HCB//2 - (2*C+4)
met = scr[meta_off_halfs:meta_off_halfs+2*C+4].view(np.uint16).view(np.float32) if False else None
# fallback: recompute beta from braw? not dumped -> recompute oracle state with dumped kh/vr and beta via meta unavailable; use sigmoid of a second dump instead
import numpy as _np
brw = _snap["braw"].reshape(256,48)
bet = 1.0/(1.0+_np.exp2(brw[:16,h].astype(_np.float64)*(-1.4426950408889634)))
KH = kh[:16].astype(_np.float64); VR = (vr[:16]+vrl[:16]).astype(_np.float64); QE = qe[:16].astype(_np.float64)
# qhat = qe / 2^g  (need g cumsum)
lams = _np.array([softplus(arw[t,h]+dtb[h])*ssma[h] for t in range(16)])
g = _np.cumsum(lams*1.4426950408889634)
qh = QE / _np.exp2(g)[:,None]
S = _np.zeros((128,128))
o_err = []
for t in range(16):
    A = 2.0**lams[t]
    S = S*A
    kd = KH[t] @ S            # (128,) v-space
    dl = (VR[t] - kd) * bet[t]
    S = S + _np.outer(KH[t], dl)
    o = qh[t] @ S
    o_sc = sco[t, h*128:(h+1)*128].astype(_np.float64)
    o_err.append((t, _np.abs(o-o_sc).max(), _np.abs(_np.median(_np.abs(o-o_sc)/_np.maximum(_np.abs(o),1e-9)))))
print("tok | o_maxdiff o_medrel   (|v| max %.2f, |d| max %.2f)" % (_np.abs(VR).max(), _np.abs(_np.outer(KH[0], (VR[0]))).max()))
for t,mx,md in o_err: print(f"{t} | {mx:.3e} {md:.3e}")
np.save("~/p7e5_probe.npy", {"kh": kh[:4], "qe": qe[:4], "vr": vr[:4], "vrl": vrl[:4], "qkv": qkv[:4]}, allow_pickle=True)
print("[probe saved]")
