# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R2c fix: drop the banked-negative w128h from the M128 plan; 2x shipped w64h."""
def patch(path, subs):
    src = open(path).read()
    for old, new in subs:
        n = src.count(old)
        assert n == 1, (n, old[:80])
        src = src.replace(old, new)
    open(path, "w").write(src)
    print("[patch]", len(subs), "edits")

OLD_CUBINS = '''M128_CUBINS = ["pfaw_w128h_s13_100k", "pfcw128h_s13",
               "pfca_c32_nc4_nw16", "pfcb_c32_nc4_nw8", "pfcz_c32_nc4_nw8"] + \\
              (["pfk_smul128"] if FFNSPLIT else [])'''
NEW_CUBINS = '''# NOTE: pfaw_w128h/pfcw128h built + corr-probed (r2c_w128h_corr.py): BIT-IDENTICAL
# at pos 100224 but NONDET-WRONG at low pos (pos 64 nz=747 det-x2 False; pos 2032
# nz=8 nondet) -- the P17 w64q ROWS-extension register class. BANKED NEGATIVE; the
# M128 attention = 2x the SHIPPED w64h windows (bit-identical, zero risk).
M128_CUBINS = ["pfca_c32_nc4_nw16", "pfcb_c32_nc4_nw8", "pfcz_c32_nc4_nw8"] + \\
              (["pfk_smul128"] if FFNSPLIT else [])'''

OLD_SCR = '''  P.poison("pmW128", 39968*4, np.float32, 7.7e31)   # w128h: 312 CTAs x 128 rows (+32 pad)
  P.poison("psW128", 39968*4, np.float32, 7.7e31)
  P.poison("pAW128", 39968*256*4, np.float32, 7.7e31)
'''
NEW_SCR = ""

OLD_ATT = '''      A(pr["pfaw_w128h_s13_100k"], d[f"kv{i}"], d[f"sc{i}"], d["qw128"], d["pos_w128"],
        d["pmW128"], d["psW128"], d["pAW128"], g=4*S13*6, ls=(512, 1, 1))
      A(pr["pfcw128h_s13"], d["pmW128"], d["psW128"], d["pAW128"], d["qrow128"], d["ao128"], g=24)'''
NEW_ATT = '''      for hp2 in range(2):   # 2x the shipped w64h windows (w128h banked negative)
        qb = d["qw128"].offset(offset=hp2*64*6144*2, size=64*6144*2)
        ab = d["ao128"].offset(offset=hp2*64*6144*2, size=64*6144*2)
        rb = d["qrow128"].offset(offset=hp2*64*12288*2, size=64*12288*2)
        pb = d["pos_w128"].offset(offset=hp2*4*4, size=4)
        A(pr["pfaw_w64h_s13_100k"], d[f"kv{i}"], d[f"sc{i}"], qb, pb,
          d["pmW"], d["psW"], d["pAW"], g=4*S13*6, ls=(512, 1, 1))
        A(pr["pfcw64h_s13"], d["pmW"], d["psW"], d["pAW"], rb, ab, g=24)'''

patch("~/tinygrad-metal/engine0/pf_prefill.py",
      [(OLD_CUBINS, NEW_CUBINS), (OLD_SCR, NEW_SCR), (OLD_ATT, NEW_ATT)])
