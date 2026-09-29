"""TLX P10 unit tests for the kernargs-slab pool (GPU-free; fakes only).

Covers the properties the F2/F3 fix depends on:
  - best-fit reuse (steady state never calls MAP_SYSMEM_FD again)
  - zero-on-release = fresh-mapping byte equivalence (determinism contract)
  - idempotent release + marker re-arm across reuse cycles
  - caps: overflow slabs are really freed
  - kill switch TLX_KA_POOL=0 = exact legacy (fresh every build, freed on retire)
  - the defensive dev.synchronize() at quiescent release
"""
import ctypes
import pytest

class FakeView:
  def __init__(self, addr): self.addr = addr

class FakeBuf:
  def __init__(self, size):
    self.size = size
    self._mem = ctypes.create_string_buffer(size)   # keep alive
    self.view = FakeView(ctypes.addressof(self._mem))
  def bytes(self): return (ctypes.c_ubyte * self.size).from_address(self.view.addr)

class FakeAllocator:
  def __init__(self): self.n_allocs = 0; self.freed = []
  def alloc(self, size, spec):
    if spec is not None: assert spec.cpu_access and spec.nolru
    self.n_allocs += 1; return FakeBuf(size)
  def free(self, buf, size, spec): self.freed.append(buf)

class FakeDev:
  def __init__(self): self.allocator = FakeAllocator(); self.n_syncs = 0
  def synchronize(self): self.n_syncs += 1

import ka_pool

@pytest.fixture()
def pool(monkeypatch):
  monkeypatch.setattr(ka_pool, "_free", [])
  monkeypatch.setattr(ka_pool, "_stats", dict(fresh=0, reused=0, released=0, freed_overflow=0,
                                              zero_fail=0, pool_bytes=0, live=0))
  monkeypatch.setattr(ka_pool, "_POOL_ON", True)
  monkeypatch.setattr(ka_pool, "_ZERO", True)
  monkeypatch.setattr(ka_pool, "_POOL_MAX", 32)
  monkeypatch.setattr(ka_pool, "_POOL_MB", 128)
  return ka_pool

def test_reuse_same_slab(pool):
  dev = FakeDev()
  a = pool.ka_alloc(1000, dev)
  assert dev.allocator.n_allocs == 1
  assert pool.ka_release(a, dev) is True
  b = pool.ka_alloc(900, dev)
  assert b is a
  st = pool.ka_stats()
  assert st["fresh"] == 1 and st["reused"] == 1 and st["pooled"] == 0 and st["live"] == 1

def test_repool_after_second_retirement(pool):
  """The marker must re-arm on pop: a slab lives through MANY build/retire
  cycles (the steady state the fix exists for)."""
  dev = FakeDev()
  for _ in range(5):
    a = pool.ka_alloc(512, dev)
    assert pool.ka_release(a, dev) is True
  assert dev.allocator.n_allocs == 1
  assert pool.ka_stats()["reused"] == 4

def test_zero_on_release(pool):
  dev = FakeDev()
  a = pool.ka_alloc(64, dev)
  ctypes.memset(a.view.addr, 0xAB, 64)
  pool.ka_release(a, dev)
  assert all(x == 0 for x in a.bytes()), "pooled slab must be zero-filled (fresh-map equivalence)"

def test_best_fit_smallest(pool):
  dev = FakeDev()
  big = pool.ka_alloc(4096, dev); pool.ka_release(big, dev)
  small = pool.ka_alloc(128, dev); pool.ka_release(small, dev)
  got = pool.ka_alloc(200, dev)
  assert got is small, "smallest fitting slab must win"

def test_idempotent_release(pool):
  dev = FakeDev()
  a = pool.ka_alloc(64, dev)
  assert pool.ka_release(a, dev) is True
  assert pool.ka_release(a, dev) is False
  assert len(pool._free) == 1 and not dev.allocator.freed

def test_kill_switch_off(pool, monkeypatch):
  monkeypatch.setattr(ka_pool, "_POOL_ON", False)
  dev = FakeDev()
  a = pool.ka_alloc(64, dev)
  assert pool.ka_release(a, dev) is False
  assert len(dev.allocator.freed) == 1
  b = pool.ka_alloc(64, dev)
  assert b is not a and dev.allocator.n_allocs == 2

def test_pool_count_cap(pool, monkeypatch):
  monkeypatch.setattr(ka_pool, "_POOL_MAX", 2)
  dev = FakeDev()
  bufs = [pool.ka_alloc(64, dev) for _ in range(4)]
  for b in bufs: pool.ka_release(b, dev)
  assert len(pool._free) == 2 and len(dev.allocator.freed) == 2

def test_sync_before_release(pool):
  dev = FakeDev()
  a = pool.ka_alloc(64, dev)
  pool.ka_release(a, dev)
  assert dev.n_syncs == 1

def test_oversize_request_gets_fresh(pool):
  dev = FakeDev()
  small = pool.ka_alloc(100, dev); pool.ka_release(small, dev)
  big = pool.ka_alloc(10_000_000, dev)
  assert big is not small and dev.allocator.n_allocs == 2
