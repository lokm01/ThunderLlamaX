# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2c fix 2: (a) pre64 x2 must append at p0 and p0+64 — pos_arr128 becomes 2 ints,
each half passes its own 4-byte base view (the pre64 kernel derives [pos0]+t);
(b) the attnqkv m64 twin falls back to 2 separate g=224 launches on 64-row
half-views (the R2b-shipped M64QKV pattern) — the 2-M-block M-grid twin is the
P7E4 in-plan corruptor class."""
def patch(path, subs):
    src = open(path).read()
    for old, new in subs:
        n = src.count(old)
        assert n == 1, (n, old[:80])
        src = src.replace(old, new)
    open(path, "w").write(src)
    print("[patch]", len(subs), "edits")

OLD1 = '''      if M64QKV and _r7qkv:
        mq64 = "pfg3m_attnqkvq6_r7_m64_nw8k128" if E.qtypes[i] == 14 else "pfg3m_attnqkvi3_r7_m64_nw8k128"
        qw_ = W[("q", i)] if E.qtypes[i] == 14 else W7[("q", i)]
        A(pr[mq64], qw_, W7[("k", i)], W[("v", i)], d["gridf"], d["xh128"],
          d["qrow128"], d["krow128"], d["vrow128"], g=448, ls=(256, 1, 1))'''
NEW1 = '''      if M64QKV and _r7qkv:
        mq64 = "pfg3m_attnqkvq6_r7_m64_nw8k128" if E.qtypes[i] == 14 else "pfg3m_attnqkvi3_r7_m64_nw8k128"
        qw_ = W[("q", i)] if E.qtypes[i] == 14 else W7[("q", i)]
        for p in range(2):   # 2x the R2b-shipped g=224 pattern (the 2-M-block M-grid twin = the P7E4 in-plan corruptor class)
          A(pr[mq64], qw_, W7[("k", i)], W[("v", i)], d["gridf"],
            V("xh128", p*64*5120*2, 64*5120*2), V("qrow128", p*64*12288*2, 64*12288*2),
            V("krow128", p*64*1024*2, 64*1024*2), V("vrow128", p*64*1024*2, 64*1024*2), g=224, ls=(256, 1, 1))'''

OLD2 = '''      for p in range(2):   # pre64 x2 on 64-row half-views (pos_w128[p*4:(p+1)*4])
        A(pr["pfk_pre64_100k"], V("qrow128", p*64*12288*2, 64*12288*2), V("krow128", p*64*1024*2, 64*1024*2),
          V("vrow128", p*64*1024*2, 64*1024*2), W[("qnw", i)], W[("knw", i)], d["freqs"],
          d[f"kv{i}"], d[f"sc{i}"], d["pos_arr128"], V("qw128", p*64*6144*2, 64*6144*2), g=24)'''
NEW2 = '''      for p in range(2):   # pre64 x2: half p appends at base p0 + 64p (pos_arr128[2] = [p0, p0+64])
        A(pr["pfk_pre64_100k"], V("qrow128", p*64*12288*2, 64*12288*2), V("krow128", p*64*1024*2, 64*1024*2),
          V("vrow128", p*64*1024*2, 64*1024*2), W[("qnw", i)], W[("knw", i)], d["freqs"],
          d[f"kv{i}"], d[f"sc{i}"], d["pos_arr128"].offset(offset=p*4, size=4),
          V("qw128", p*64*6144*2, 64*6144*2), g=24)'''

OLD3 = '''  P.up("pos_arr128", np.zeros(1, dtype=np.int32))'''
NEW3 = '''  P.up("pos_arr128", np.zeros(2, dtype=np.int32))   # pre64 bases: [p0, p0+64]'''

OLD4 = '''    P.win_up("pos_arr128", 0, np.array([p0], dtype=np.int32))'''
NEW4 = '''    P.win_up("pos_arr128", 0, np.array([p0, p0 + 64], dtype=np.int32))'''

patch("~/tinygrad-metal/engine0/pf_prefill.py", [(OLD1, NEW1), (OLD2, NEW2), (OLD3, NEW3), (OLD4, NEW4)])
