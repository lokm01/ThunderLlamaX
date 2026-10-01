# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
# TLX DRAFTER Phase 1 — standalone GGUF metadata probe (pure python + numpy,
# read-only; runs anywhere including the rig's ~/tg311/bin/python).
# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
import struct
import sys
import numpy as np

T = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
     6: ("<f", 4), 7: ("<B", 1), 8: (None, 0), 9: (None, 0), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


class R:
    def __init__(self, b, src=None):
        self.b = b
        self.i = 0
        self.src = src  # callable -> more bytes when exhausted

    def _more(self):
        if self.src:
            self.b = self.b + self.src()

    def u32(self):
        while self.i + 4 > len(self.b):
            self._more()
        v = struct.unpack_from("<I", self.b, self.i)[0]
        self.i += 4
        return v

    def u64(self):
        while self.i + 8 > len(self.b):
            self._more()
        v = struct.unpack_from("<Q", self.b, self.i)[0]
        self.i += 8
        return v

    def s(self):
        n = self.u64()
        while self.i + n > len(self.b):
            self._more()
        v = self.b[self.i:self.i + n].decode()
        self.i += n
        return v

    def val(self, t):
        if t == 8:
            return self.s()
        if t == 9:
            it = self.u32()
            n = self.u64()
            return [self.val(it) for _ in range(n)]
        f, sz = T[t]
        while self.i + sz > len(self.b):
            self._more()
        v = struct.unpack_from(f, self.b, self.i)[0]
        self.i += sz
        return v


def parse_header(path, want_filter=None):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    r0 = R(f.read(1 << 20), src=lambda: f.read(1 << 20))
    r0.val(4)  # version
    nten = r0.u64()
    nkv = r0.u64()
    meta = {}
    for _ in range(nkv):
        k = r0.s()
        t = r0.u32()
        meta[k] = r0.val(t)
    infos = {}
    for _ in range(nten):
        name = r0.s()
        nd = r0.u32()
        dims = [r0.u64() for _ in range(nd)]
        ty = r0.u32()
        off = r0.u64()
        infos[name] = (ty, dims, off)
    align = meta.get("general.alignment", 32)
    data_start = 12 + r0.i
    data_start = (data_start + align - 1) // align * align
    f.close()
    return meta, infos, data_start


def read_tensor(path, info, data_start, dtype="<f4"):
    ty, dims, off = info
    n = 1
    for d in dims:
        n *= d
    with open(path, "rb") as f:
        f.seek(data_start + off)
        return np.frombuffer(f.read(n * 4), dtype=dtype).copy(), ty, dims


if __name__ == "__main__":
    path = sys.argv[1]
    names = sys.argv[2:] if len(sys.argv) > 2 else None
    meta, infos, ds = parse_header(path)
    print("tensors:", len(infos), "data_start:", ds)
    if names:
        for nm in names:
            if nm in infos:
                ty, dims, off = infos[nm]
                print(nm, "type", ty, "dims", dims)
                if ty == 0:  # f32
                    a, _, _ = read_tensor(path, infos[nm], ds)
                    print("  mean %.5f std %.5f min %.4f max %.4f head %s" % (a.mean(), a.std(), a.min(), a.max(), a[:6]))
            else:
                print(nm, "NOT FOUND")
    else:
        pat = sys.argv[1]
        for k in sorted(infos):
            if "nextn" in k or "blk.64" in k or "norm" in k.lower() and "output" in k.lower():
                print(k, infos[k][0], infos[k][1])
