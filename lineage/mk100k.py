# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import random
random.seed(1000)
words = ["quantum","field","vector","phase","energy","symmetry","gauge","entropy","photon","neutrino","boson","fermion","spin","charge","momentum","gravity","lattice","vacuum","flux","current"]
txt = " ".join(random.choice(words) for _ in range(92000))
open("~/prompt100k.txt","w").write("Foundations of modern physics. " + txt)
print("100k prompt written", len(txt))
