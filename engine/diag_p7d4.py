# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src"); sys.path.insert(0, "~/tinygrad-metal/engine0")
import numpy as np
import test_p7d as T
from engine0 import dev
P = T.P
qrow = P.down("qrow64", (64, 12288), np.float16).astype(np.float32)
qnw = P.down("qnw", (256,), np.float32); freqs = P.down("freqs", (32,), np.float32)
def ref_qw(t, h):
  v = qrow[t, h*512:h*512+256].copy()
  ss = float((v*v).sum()); r = 1.0/np.sqrt(ss/256 + 1e-6)
  v = np.float16(v*r).astype(np.float32) * qnw
  st = v.copy()
  for d in range(32):
    ang = t * freqs[d]; c, s = np.cos(ang), np.sin(ang)
    st[d] = v[d]*c - v[d+32]*s
    st[d+32] = v[d+32]*c + v[d]*s
  return np.float16(st * 0.0625)
T.kpre16("kva", "sca", "qw64a", 0, P.d["pos0"])
T.kpre16("kva", "sca", "qw64a", 16, P.d["pos16"])
T.prog("pfk_pre64_100k")(P.d["qrow64"], P.d["krow64"], P.d["vrow64"], P.d["qnw"], P.d["knw"], P.d["freqs"],
     P.d["kvb"], P.d["scb"], P.d["pos0"], P.d["qw64b"], global_size=(24,1,1), local_size=T.LS)
dev.synchronize()
qa = P.down("qw64a", (64, 12288), np.float16); qb = P.down("qw64b", (64, 12288), np.float16)
for t in (0, 5, 8, 12, 16, 20, 40):
  r = ref_qw(t, 0)
  ea = np.abs(qa[t, 0:256].astype(np.float32)-r.astype(np.float32)).max()
  eb = np.abs(qb[t, 0:256].astype(np.float32)-r.astype(np.float32)).max()
  print(f"[gt] t={t}: kpre16-vs-numpy {ea:.4f} | pre64-vs-numpy {eb:.4f}", flush=True)
