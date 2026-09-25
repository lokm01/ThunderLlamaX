# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R7a K=8: test_w100k.py DIF/TRACE harness K=8 support."""
BASE = "~/tinygrad-metal/engine0"
s = open(f"{BASE}/test_w100k.py").read()

def rep(a, b, label):
  global s
  c = s.count(a)
  assert c == 1, f"{label}: found {c}"
  s = s.replace(a, b)
  print(f"[patch] {label:44s} ok")

# trace: dring seeding + read width
rep('    for nm in ("dring0", "dring1", "dring2", "dring3", "dring4", "dring5", "dring6"):   # valid ids (draft graph not run here)',
    '    for nm in ("dring0", "dring1", "dring2", "dring3", "dring4", "dring5", "dring6", "dring7"):   # valid ids (draft graph not run here)',
    "trace dring7 seed")
rep('      amds = E.P.down_at("amds", 0, 8, np.int32).tolist()',
    '      amds = E.P.down_at("amds", 0, 9 if int(os.getenv("LOOKUP_K", "0") or 0) >= 8 else 8, np.int32).tolist()',
    "trace amds width")
rep('''      d6 = int(E.P.down_at("dring6", 0, 1, np.int32)[0]) if LKX == 7 else 0''',
    '''      d6 = int(E.P.down_at("dring6", 0, 1, np.int32)[0]) if LKX >= 7 else 0
      d7 = int(E.P.down_at("dring7", 0, 1, np.int32)[0]) if LKX >= 8 else 0''',
    "trace d7 read")
rep('''              f"amds={amds} dr=({d0},{d1},{d2},{d3},{d4},{d5},{d6})")''',
    '''              f"amds={amds} dr=({d0},{d1},{d2},{d3},{d4},{d5},{d6},{d7})")''',
    "trace print d7")

# DIF: dring seeding + NR + acceptsel9k re-anchor
rep('''                ("dring4", refa[4] if LK5 else 0), ("dring5", refa[5] if LKX >= 6 else 0),
                ("dring6", refa[6] if LKX == 7 else 0)):''',
    '''                ("dring4", refa[4] if LK5 else 0), ("dring5", refa[5] if LKX >= 6 else 0),
                ("dring6", refa[6] if LKX >= 7 else 0), ("dring7", refa[7] if LKX >= 8 else 0)):''',
    "dif dring7 seed")
rep('  NR = {0: 5, 4: 5, 5: 6, 6: 7, 7: 8}[LKX]',
    '  NR = {0: 5, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}[LKX]',
    "dif NR map")
rep('''  if LKX == 7:
    E.pr["acceptsel8k"](E.P.d["rec4"], E.P.d["conv4"], E.P.d["m_slot"], E.P.d["conv5x"], E.P.d["conv6x"], E.P.d["conv7x"], E.P.d["conv8x"], E.P.d["rec6x"], E.P.d["rec7x"], E.P.d["rec8x"],
                        global_size=(48, 1, 1), local_size=(256, 1, 1))''',
    '''  if LKX == 8:
    E.pr["acceptsel9k"](E.P.d["rec4"], E.P.d["conv4"], E.P.d["m_slot"], E.P.d["conv5x"], E.P.d["conv6x"], E.P.d["conv7x"], E.P.d["conv8x"], E.P.d["conv9x"], E.P.d["rec6x"], E.P.d["rec7x"], E.P.d["rec8x"], E.P.d["rec9x"],
                        global_size=(48, 1, 1), local_size=(256, 1, 1))
  elif LKX == 7:
    E.pr["acceptsel8k"](E.P.d["rec4"], E.P.d["conv4"], E.P.d["m_slot"], E.P.d["conv5x"], E.P.d["conv6x"], E.P.d["conv7x"], E.P.d["conv8x"], E.P.d["rec6x"], E.P.d["rec7x"], E.P.d["rec8x"],
                        global_size=(48, 1, 1), local_size=(256, 1, 1))''',
    "dif acceptsel9k re-anchor")

open(f"{BASE}/test_w100k.py", "w").write(s)
print("[patch] test_w100k.py K8 complete")
