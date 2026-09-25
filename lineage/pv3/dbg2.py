# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import os, sys
os.environ.setdefault("DEV", "NV")
sys.path.insert(0, "~/tinygrad-src")
import numpy as np
from tinygrad import dtypes
from tinygrad.tensor import Tensor
from tinygrad.device import Device
dev = Device["NV"]
o = Tensor.full((18432,), -3e38).contiguous().realize()
print("fill check:", o.numpy().min(), "(expect -3e38)", flush=True)
o2 = Tensor.zeros((18432,)).contiguous().realize()
print("zeros check:", o2.numpy().max(), flush=True)
b = Tensor(np.full(18432, -3e38, np.float32)).contiguous().realize()
print("upload check:", b.numpy().min(), flush=True)
