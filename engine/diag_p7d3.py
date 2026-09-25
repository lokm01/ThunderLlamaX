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
# re-run gate1 pieces with pattern dump
T.kpre16("kva", "sca", "qw64a", 0, P.d["pos0"])
T.kpre16("kva", "sca", "qw64a", 16, P.d["pos16"])
T.prog("pfk_pre64_100k")(P.d["qrow64"], P.d["krow64"], P.d["vrow64"], P.d["qnw"], P.d["knw"], P.d["freqs"],
     P.d["kvb"], P.d["scb"], P.d["pos0"], P.d["qw64b"], global_size=(24,1,1), local_size=T.LS)
dev.synchronize()
qa = P.down("qw64a", (64, 12288), np.float16); qb = P.down("qw64b", (64, 12288), np.float16)
mm = (qa != qb)
print("[pat] per-row mismatch counts (rows 0..63):")
print("[pat]", [int(mm[r].sum()) for r in range(64)])
# head-slice structure within a bad row: qw row = 24 slices of 256 (h*2 q?, layout t*6144 per head inside kpre write is qw16[t*6144 + h*256 + d])
mm2 = mm.reshape(64, 48, 256)
print("[pat] row16 per-48-slice:", [int(mm2[16, s].sum()) for s in range(48)])
print("[pat] row0  per-48-slice:", [int(mm2[0, s].sum()) for s in range(48)])
print("[pat] row16 slice0 dims:", np.where(mm2[16,0])[0][:16])
