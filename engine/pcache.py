# ThunderLlamaX — LLM inference on an eGPU, hitched to a Mac.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lokm01
"""R1: THE DURABLE PROMPT CACHE — hash-keyed prefix trie on disk.

Design (as-built; see ~/tinygrad-metal/R1_PROMPTCACHE.md):
- KEYING: content-addressed hash chain over FED TOKEN IDS. h over 64-token
  blocks: h_i = sha256(h_{i-1} || ids[64i:64i+64]); a node at non-aligned pos
  extends the last boundary hash with the partial tail. The config fingerprint
  (the kernel-set env + model identity, svc_fp) is mixed into the chain ROOT —
  entries are valid only under a bit-identical engine config. NEVER keyed on
  rendered text (the M1-C RE-ENCODE law — the API layer builds ids, we hash
  THOSE).
- NODES: a node covers the KV WINDOW [A, B) plus the full sequential state at
  B (GDN rec/conv, draft KV window, dhd draft hidden, trunk hlast or a
  precomputed cur). Nodes land at PC_STRIDE (1024) during FRESH prefill, plus
  the final 64-aligned chunk boundary, plus 64-aligned turn ends. Restore =
  upload the chain root->deepest hit + tail re-prefill (<= 1023+63 tokens).
- GAP 1: the draft KV (kv_d/sc_d) is persisted in every node (and in the
  one-conversation snapshot path in serve.py) — a cross-restart restore no
  longer re-pays fill_draft (~255s @100k).
- GAP 2: FOLLOW_UP delta prefill runs the P-series M64 chunk path
  (mtp.follow_up batch=True; FU_BATCH=1 default).
- STORAGE: disk trie at ~/prompt_cache. TLX W3 hardening (this revision):
  * Durability (V-39): every .npy fsynced before the atomic staging->final
    rename, staging + root dir fsynced; per-node sha256 + shapes of every
    artifact in meta; restore VALIDATES the whole chain (size + shape + hash
    + chain adjacency + hlast finiteness) BEFORE the first win_up — a
    corrupt/planted node can never reach the GPU (the OOB->reboot class).
  * Concurrency (V-41/V-42): protect mutations under self.lock; lookup
    protects the returned chain INSIDE the lock (TOCTOU closed); eviction
    renames victims to .graveyard/ under the lock and rmtrees outside
    (mmapped uploads survive — POSIX rename/unlink keeps mappings); boot
    orphan reap (dirs without a manifest ref); _save_man failure reaps the
    just-renamed node; restore failure -> NodeCorrupt -> caller falls back
    FRESH (validation precedes any upload, so a refused restore leaves the
    engine untouched).
  * Backpressure + budgets (V-40/V-44/V-45/V-47): write_node is
    put_nowait + drop-on-full (NEVER a sync write on the GPU thread; slog
    pc_drop); queued-node bytes counted toward the eviction quota; pins get
    a byte budget (PC_PIN_BUDGET_FRAC, default 50% of quota), a per-key node
    cap and a live-pin cap with oldest-pin expiry; LRU uses a strict
    monotonic counter (lrc) — no wall-clock ties; prompt_cache_ttl plumbed
    through pin(chain, key, ttl).
  * Manifest trust (V-43): strict ^[0-9a-f]{64}$ on manifest keys AND dir
    names at load and at every use; foreign/planted dir strings are rejected
    without deleting anything outside the cache root.
  * Probe gap (W3.5): lookup probes EVERY 64-boundary (chain_keys already
    computes them) — mid-conversation turn-end nodes are reachable from
    longer requests, not just exact-length/STRIDE probes.
  * G1 LAW FIX (W3.6): the midprefill hlast source is per trunk generation.
    The R1-era "xA64 row 63" law is M64-only — under the shipped M128 trunk
    the chunks ping-pong xA128/xB128 and NEVER touch xA64, so the old read
    returned ensure64 POISON (7.7e31) -> fp16-overflow logits -> h_argmax
    against NaN leaves tok_slot 0 = the G1 multi-node "cur=0 post-restore"
    bug (single-node G3 turnend / G4 boot nodes never used that path, hence
    "single-node exact"). Both capture and restore now hard-guard finiteness.
- LAWS kept: delta-windowed down_at only (~200MB/node — never a full-KV
  copyout); ingest only at prefill chunk boundaries / turn ends (the ~950-cycle
  law); windowed win_up (fixed-handle, graphs stay valid); manifest + nodes
  under ~ (reboot-survivor), never /tmp.

Sizes (KV8): 1024-token node ~= 33.6MB kvb + 2.1MB sc + 2.2MB kvd + 0.13MB scd
+ 151MB rec (fp32 GDN @48 blocks) + 5.9MB conv ~= 195MB; a 100k doc ~= 96
nodes ~= 18.7GB.
"""
import os, re, json, time, hashlib, shutil, tempfile, threading, queue
import numpy as np
import svc_fp   # TLX W2: the ONE config-fingerprint implementation (V-28)

HASH_BLK = 64
STRIDE = int(os.getenv("PC_STRIDE", "1024"))
QUOTA_BYTES = int(float(os.getenv("PC_QUOTA_GB", "60")) * 1e9)
MIN_HIT = int(os.getenv("PC_MIN_HIT", "1024"))
PIN_TTL = int(os.getenv("PC_PIN_TTL_S", str(7 * 86400)))
ROOT = os.getenv("PC_ROOT", os.path.expanduser("~/prompt_cache"))
FMT = svc_fp.FMT
# TLX W3 knobs (all env-gated; defaults = the hardened behavior)
PIN_TTL_MIN = 60
PIN_TTL_MAX = 7 * 86400
PIN_BYTE_BUDGET = int(float(os.getenv("PC_PIN_BUDGET_FRAC", "0.5")) * QUOTA_BYTES)
PIN_MAX_NODES = int(os.getenv("PC_PIN_MAX_NODES", "96"))     # per key
PIN_MAX_LIVE = int(os.getenv("PC_PIN_MAX_LIVE", "384"))      # total live pins
FLUSH_S = float(os.getenv("PC_FLUSH_S", "30"))
HASH_VERIFY = os.getenv("PC_HASH_VERIFY", "1") == "1"        # sha256 pre-win_up
# V-43: a node dir is exactly the 64-hex hkey — anything else in the manifest
# is rejected and NOTHING outside root/<hkey> is ever deleted.
_HKEY_RE = re.compile(r"^[0-9a-f]{64}$")
_ART = ("kvb", "sc", "kvd", "scd", "rec", "conv", "dhd", "hlast")

# the kernel-set fingerprint: cache entries valid only under a bit-identical
# engine numerics config (any of these flipping changes KV bytes / GDN math).
# TLX W2 (V-28): the key set + model-file identity live in svc_fp (shared with
# serve.py status + the api drift alarm).
_ENV_KEYS = svc_fp._ENV_KEYS

def config_fp():
    return svc_fp.config_fp()

def _root_h():
    return hashlib.sha256(("r1pc-root|" + config_fp()).encode()).digest()

def chain_keys(ids):
    """{pos: hkey} at every 64-boundary plus the exact end pos."""
    out = {}
    h = _root_h()
    a = np.asarray(ids, dtype=np.int64)
    n = len(a)
    for i in range(n // HASH_BLK):
        h = hashlib.sha256(h + a[HASH_BLK * i: HASH_BLK * i + HASH_BLK].tobytes()).digest()
        out[HASH_BLK * (i + 1)] = h.hex()
    r = n - HASH_BLK * (n // HASH_BLK)
    if r:
        out[n] = hashlib.sha256(h + a[HASH_BLK * (n // HASH_BLK): n].tobytes()).hexdigest()
    return out

def hkey_prefix(ids):
    """Chain hash of the exact prefix ids (== chain_keys(ids)[len(ids)])."""
    return chain_keys(ids)[len(ids)]


# ---------------- W3 pure helpers (unit-testable, no engine) ----------------

def _fsync_dir(path):
    try:
        dfd = os.open(path, os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    except Exception:
        pass

def _save_npy(path, arr):
    """np.save + fsync (V-39: a post-crash correctly-sized zero-filled file
    must never pass as a valid node — the torn-write window closes here)."""
    np.save(path, np.ascontiguousarray(arr))
    with open(path, "rb") as f:
        os.fsync(f.fileno())

def _sha256_file(path, _buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(_buf)
            if not b: break
            h.update(b)
    return h.hexdigest()

def _node_bytes_estimate(node):
    tot = 0
    for nm in _ART:
        v = node.get(nm)
        if v is not None:
            tot += int(np.asarray(v).nbytes)
    return tot + 8192   # meta.json + fs slack


class NodeCorrupt(Exception):
    """W3.1/W3.2: a chain node failed validation. Raised BEFORE any win_up —
    the engine is untouched; the caller quarantines + falls back FRESH."""
    def __init__(self, hkey, why):
        super().__init__(f"node {str(hkey)[:16]}: {why}")
        self.hkey = hkey
        self.why = why


class PromptCache:
    """Manifest + async writer + LRU eviction. GPU work happens in the caller
    (the daemon thread); this class only touches disk."""

    def __init__(self, root=ROOT):
        self.root = root
        self.staging = os.path.join(root, "staging")
        self.grave = os.path.join(root, ".graveyard")
        os.makedirs(self.staging, exist_ok=True)
        os.makedirs(self.grave, exist_ok=True)
        self.lock = threading.Lock()
        self._final = threading.Lock()     # serializes staging->final renames
        self.man = {"entries": {}}
        self.protect = set()               # resident-conversation hkeys (unevictable)
        self._lrc = 1                      # strict monotonic LRU counter (V-47)
        self._pend = 0                     # queued (writer-backed) node bytes
        self._outstanding = 0              # wq items in flight (flush())
        self._quota_warn_ts = 0.0
        self.wq = queue.Queue(maxsize=4)
        self.wth = threading.Thread(target=self._writer, daemon=True)
        self.wth.start()
        self._clean_staging()
        self._load()
        self._reap_graveyard()

    # ---------- manifest ----------
    def _mpath(self): return os.path.join(self.root, "manifest.json")

    def _next_lrc(self):
        self._lrc += 1
        return self._lrc

    def _load(self):
        """Boot heal (V-42/V-43): strict hkey validation, all-artifact size
        spot-checks vs disk, dangling refs graveled, ORPHAN dirs (renamed but
        never manifested — the crash window) reaped. Foreign (non-hkey) dirs
        are left untouched."""
        try:
            m = json.load(open(self._mpath()))
        except Exception:
            m = {"entries": {}}
        ent = {}
        changed = False
        for k, v in (m.get("entries") or {}).items():
            d = str(v.get("dir", ""))
            if not (_HKEY_RE.match(k) and _HKEY_RE.match(d) and k == d):
                print(f"[pcache] manifest reject (bad hkey/dir): {k[:16]}/{d[:24]}", flush=True)
                continue                      # rejected; NOTHING deleted (V-43)
            path = os.path.join(self.root, d)
            ok = os.path.isdir(path) and os.path.isfile(os.path.join(path, "meta.json"))
            sizes = v.get("sizes") or {}
            for nm, sz in sizes.items():      # every artifact, not just rec.npy
                try:
                    ok = ok and os.path.getsize(os.path.join(path, f"{nm}.npy")) == int(sz)
                except Exception:
                    ok = False
            if ok:
                v.setdefault("lrc", self._next_lrc())
                v.setdefault("state", "ready")
                ent[k] = v
            else:
                self._graveyard(path); changed = True
        # orphan reap: hkey-shaped dirs with no manifest ref
        try:
            names = os.listdir(self.root)
        except Exception:
            names = []
        for name in names:
            if name in ent or not _HKEY_RE.match(name): continue
            p = os.path.join(self.root, name)
            if os.path.isdir(p):
                self._graveyard(p); changed = True
        self.man = {"entries": ent}
        if changed:
            try: self._save_man()
            except Exception as e: print(f"[pcache] boot manifest save failed: {e!r}", flush=True)

    def _save_man(self):
        with self.lock:
            ent = dict(self.man["entries"])
        tmp = self._mpath() + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"fmt": FMT, "entries": ent}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._mpath())
        _fsync_dir(self.root)

    def _clean_staging(self):
        try:
            for d in os.listdir(self.staging):
                shutil.rmtree(os.path.join(self.staging, d), ignore_errors=True)
        except Exception:
            pass

    # ---------- graveyard (deferred delete; V-41/V-42) ----------
    def _graveyard(self, path):
        """Atomic rename into .graveyard/ — safe vs in-flight mmapped readers
        (POSIX keeps mappings alive through rename/unlink); the actual rmtree
        happens in _reap_graveyard outside any lock."""
        try:
            dst = os.path.join(self.grave, os.path.basename(path) + f".{time.time_ns()}")
            os.rename(path, dst)
            return dst
        except Exception:
            return None

    def _reap_graveyard(self):
        try:
            for name in os.listdir(self.grave):
                shutil.rmtree(os.path.join(self.grave, name), ignore_errors=True)
        except Exception:
            pass

    # ---------- write path (async; staging + atomic rename) ----------
    def write_node(self, node):
        """node: dict(pos_start, pos_end, parent, config_fp, cur?, hlast?, dhd,
        kvb[(16,W*2048)u8], sc[(16,W*64)f16], kvd, scd, rec, conv). Idempotent:
        if the hkey exists+valid, refreshes LRU and returns. W3 (V-40):
        NON-BLOCKING — put_nowait + drop-on-full (slog pc_drop; the cache is
        best-effort). Sync writes live ONLY on the writer thread / flush()."""
        hk = node["hkey"]
        with self.lock:
            if hk in self.man["entries"] and self.man["entries"][hk].get("state", "ready") == "ready":
                self.man["entries"][hk]["last_hit"] = time.time()
                self.man["entries"][hk]["lrc"] = self._next_lrc()
                return {"hkey": hk, "existed": True}
        est = _node_bytes_estimate(node)
        with self.lock:
            self._pend += est              # account BEFORE enqueue (the writer
            self._outstanding += 1         # may drain it the instant it lands)
        try:
            self.wq.put_nowait(node)
        except queue.Full:
            with self.lock:
                self._pend -= est
                self._outstanding -= 1
            print(f"[pcache] pc_drop writer queue full (hkey={hk[:16]}, ~{est//1048576}MB) — node NOT cached", flush=True)
            return {"hkey": hk, "existed": False, "dropped": True}
        return {"hkey": hk, "existed": False}

    def _write_node_sync(self, node):
        """Writer-thread/clean-shutdown write: staging (+fsync per artifact),
        placeholder-safe final rename under _final, manifest, evict. On a
        manifest-save failure the just-renamed dir is graveled (no orphans)."""
        hk = node["hkey"]
        if not _HKEY_RE.match(hk):
            raise ValueError(f"bad hkey {hk[:24]!r}")
        st = tempfile.mkdtemp(prefix="n_", dir=self.staging)
        sizes, shas, shapes = {}, {}, {}
        for nm in ("kvb", "sc", "kvd", "scd", "rec", "conv", "dhd", "hlast"):
            v = node.get(nm)
            if v is None: continue
            arr = np.ascontiguousarray(v)
            _save_npy(f"{st}/{nm}.npy", arr)
            sizes[nm] = os.path.getsize(f"{st}/{nm}.npy")     # ON-DISK size (npy header included)
            shas[nm] = _sha256_file(f"{st}/{nm}.npy")
            shapes[nm] = list(arr.shape)
        meta = {"pos_start": int(node["pos_start"]), "pos_end": int(node["pos_end"]),
                "parent": node.get("parent"), "config_fp": node.get("config_fp"),
                "cur": node.get("cur"), "has_hlast": node.get("hlast") is not None,
                "fmt": FMT, "created": time.time(),
                "sizes": sizes, "sha256": shas, "shapes": shapes}
        with open(f"{st}/meta.json", "w") as f:
            json.dump(meta, f)
            f.flush()
            os.fsync(f.fileno())
        _fsync_dir(st)
        fin = os.path.join(self.root, hk)
        with self._final:
            with self.lock:
                cur_ent = self.man["entries"].get(hk)
            if os.path.isdir(fin) and cur_ent is not None and cur_ent.get("state", "ready") == "ready":
                # idempotent rewrite of a valid node: refresh + skip the IO
                with self.lock:
                    cur_ent["last_hit"] = time.time()
                    cur_ent["lrc"] = self._next_lrc()
                shutil.rmtree(st, ignore_errors=True)
                self._save_man()
                return
            if os.path.isdir(fin):
                self._graveyard(fin)      # torn leftover (no manifest ref)
            os.rename(st, fin)            # atomic: partial nodes invisible
            _fsync_dir(self.root)
            entry = {"dir": hk, "pos_start": int(node["pos_start"]), "pos_end": int(node["pos_end"]),
                     "parent": node.get("parent"), "config_fp": node.get("config_fp"),
                     "bytes": sum(sizes.values()), "sizes": sizes, "sha256": shas,
                     "shapes": shapes, "last_hit": time.time(),
                     "lrc": 0, "pin": None, "state": "ready"}
            with self.lock:
                entry["lrc"] = self._next_lrc()
                self.man["entries"][hk] = entry
            try:
                self._save_man()
            except Exception:
                # V-42: crash-window hygiene — a renamed-but-unmanifested node
                # would be an invisible orphan; reap it and re-raise.
                self._graveyard(fin)
                with self.lock:
                    self.man["entries"].pop(hk, None)
                raise
        self.evict()

    def _writer(self):
        import traceback
        while True:
            node = self.wq.get()
            if node is None:
                self.wq.task_done(); return
            try:
                self._write_node_sync(node)
            except Exception as e:
                print(f"[pcache] writer error {e!r}", flush=True)
                traceback.print_exc()
            finally:
                with self.lock:
                    self._pend -= _node_bytes_estimate(node)
                    self._outstanding = max(0, self._outstanding - 1)
                self.wq.task_done()

    def flush(self, timeout=FLUSH_S):
        """Bounded clean-shutdown drain (W2 _clean_exit design; py3.9
        Queue.join has no timeout). Best-effort: a wedged writer logs + the
        manifest save still runs."""
        t0 = time.time()
        drained = False
        while time.time() - t0 < timeout:
            with self.lock:
                if self._outstanding == 0:
                    drained = True
                    break
            time.sleep(0.05)
        if not drained:
            print(f"[pcache] flush timeout after {timeout}s ({self._outstanding} writes in flight)", flush=True)
        self._save_man()

    # ---------- lookup / pin / evict ----------
    def lookup(self, ids2, min_hit=MIN_HIT):
        """Deepest complete chain ending at a probed boundary. Returns
        (B, [entry root..hit]) or (0, []). W3.5: probes EVERY 64-boundary
        (turn-end nodes at non-STRIDE positions are reachable from longer
        requests) plus the exact end. W3.2: the returned chain is added to
        self.protect INSIDE the lock — the lookup->protect TOCTOU window
        (evict racing restore) is closed."""
        ck = chain_keys(ids2)
        cands = sorted(ck.keys(), reverse=True)
        with self.lock:
            ent = self.man["entries"]
            for pos in cands:
                hk = ck.get(pos)
                if hk is None or hk not in ent: continue
                chain = []
                cur = hk
                while cur is not None:
                    e = ent.get(cur)
                    if e is None or e.get("config_fp") != config_fp(): chain = []; break
                    chain.append((cur, e))
                    cur = e.get("parent")
                if not chain: continue
                chain.reverse()
                if pos < min_hit: continue
                B = int(chain[-1][1]["pos_end"])
                if B != pos: continue
                self.protect |= set(h for h, _ in chain)   # TOCTOU closed here
                return B, chain
        return 0, []

    def touch(self, chain):
        now = time.time()
        with self.lock:
            for hk, _ in chain:
                if hk in self.man["entries"]:
                    self.man["entries"][hk]["last_hit"] = now
                    self.man["entries"][hk]["lrc"] = self._next_lrc()

    def set_protect(self, hkeys):
        with self.lock:
            self.protect = set(hkeys)

    def add_protect(self, hk):
        with self.lock:
            self.protect.add(hk)

    def clear_protect(self):
        with self.lock:
            self.protect = set()

    def pin(self, chain, key, ttl=None):
        """W3.3 (V-44/V-45): per-request TTL honored (clamped to
        [60s, 7d]); pin budgets — byte cap (PC_PIN_BUDGET_FRAC x quota),
        per-key node cap, live-pin cap; oldest pins expire first when a new
        pin would exceed the budget."""
        ttl = PIN_TTL if ttl is None else int(ttl)
        ttl = max(PIN_TTL_MIN, min(PIN_TTL_MAX, ttl))
        until = time.time() + ttl
        key = str(key)[:128]
        with self.lock:
            ent = self.man["entries"]
            now = time.time()
            # expire stale pins outright
            for hk, e in ent.items():
                p = e.get("pin")
                if p and p.get("until", 0) <= now:
                    e["pin"] = None
            live = {hk: e for hk, e in ent.items() if e.get("pin")}
            pinned_bytes = sum(int(e["bytes"]) for e in live.values())
            per_key = sum(1 for e in live.values() if e["pin"].get("key") == key)
            todo = [hk for hk, _ in chain if hk in ent]
            # make room: oldest pins first (never the chain being pinned now)
            chain_set = set(todo)
            def _room_for(add_bytes, add_nodes):
                return (pinned_bytes + add_bytes <= PIN_BYTE_BUDGET
                        and len(live) + add_nodes <= PIN_MAX_LIVE
                        and per_key + add_nodes <= PIN_MAX_NODES)
            while live:
                add_b = sum(int(ent[hk]["bytes"]) for hk in todo if hk not in live)
                add_n = sum(1 for hk in todo if hk not in live)
                if _room_for(add_b, add_n):
                    break
                # expire the oldest pin not in the current chain
                cands = [(e["pin"].get("at", 0), hk) for hk, e in live.items() if hk not in chain_set]
                if not cands:
                    break
                cands.sort()
                _, victim = cands[0]
                pinned_bytes -= int(live[victim]["bytes"])
                if live[victim]["pin"].get("key") == key: per_key -= 1
                live.pop(victim)
                ent[victim]["pin"] = None
                print(f"[pcache] pc_pin_budget expired pin {victim[:16]} (key={key[:16]})", flush=True)
            # pin as deep a prefix of the chain as the budget allows
            added = 0
            for hk in todo:
                if hk in live:
                    ent[hk]["pin"] = {"key": key, "until": until, "at": time.time()}
                    continue
                b = int(ent[hk]["bytes"])
                if (len(live) + 1 > PIN_MAX_LIVE or per_key + 1 > PIN_MAX_NODES
                        or pinned_bytes + b > PIN_BYTE_BUDGET):
                    print(f"[pcache] pc_pin_budget rejected {hk[:16]} (key={key[:16]}, "
                          f"pinned={pinned_bytes//1048576}MB cap={PIN_BYTE_BUDGET//1048576}MB)", flush=True)
                    break
                ent[hk]["pin"] = {"key": key, "until": until, "at": time.time()}
                live[hk] = ent[hk]
                pinned_bytes += b
                per_key += 1
                added += 1
        self._save_man()
        return {"pinned": added, "until": until}

    def quarantine(self, hk):
        """W3.1: remove a corrupt node (graveyard + manifest pop + save)."""
        with self.lock:
            e = self.man["entries"].pop(hk, None)
            self.protect.discard(hk)
        if e is not None:
            dst = self._graveyard(os.path.join(self.root, hk))
            if dst is not None:
                shutil.rmtree(dst, ignore_errors=True)
            try: self._save_man()
            except Exception: pass

    def total_bytes(self):
        with self.lock:
            return sum(int(e["bytes"]) for e in self.man["entries"].values()) + max(0, self._pend)

    def evict(self):
        """LRU by bytes, LEAF nodes only (a node whose children exist is
        structurally needed by longer prefixes). Pinned + protected survive.
        W3: strict-monotonic lrc ordering (V-47); victims renamed to the
        graveyard UNDER the lock (rmtree outside — mmapped readers survive);
        queued-but-unwritten nodes count toward the total (V-42); an
        over-quota state with no evictable candidates logs pc_quota_hard."""
        while True:
            with self.lock:
                ent = self.man["entries"]
                total = sum(int(e["bytes"]) for e in ent.values()) + max(0, self._pend)
                if total <= QUOTA_BYTES or len(ent) <= 1:
                    if total > QUOTA_BYTES:
                        now = time.time()
                        if now - self._quota_warn_ts > 300:
                            self._quota_warn_ts = now
                            print(f"[pcache] pc_quota_hard over quota ({total//10**9}GB > "
                                  f"{QUOTA_BYTES//10**9}GB) with no evictable leaves", flush=True)
                    return
                parents = {e.get("parent") for e in ent.values()}
                now = time.time()
                cands = [((e.get("lrc", 0), e.get("last_hit", 0.0)), hk) for hk, e in ent.items()
                         if hk not in parents and hk not in self.protect
                         and not (e.get("pin") and e["pin"].get("until", 0) > now)]
                if not cands: return
                cands.sort()
                _, victim = cands[0]
                v = ent.pop(victim)
            try:
                # rename under-lock was fast; rmtree outside all locks (the
                # victim is out of the manifest; POSIX keeps any in-flight
                # mmapped readers of the old inode valid)
                dst = self._graveyard(os.path.join(self.root, v["dir"]))
                if dst is not None:
                    shutil.rmtree(dst, ignore_errors=True)
            except Exception:
                pass
            self._save_man()


# =====================================================================
# Engine-coupled capture / restore (runs on the DAEMON thread: only this
# thread touches the GPU; the writer thread is disk-only).
# =====================================================================

def _trunk_boundary_hlast(P):
    """W3.6 G1 LAW FIX: the trunk hidden at the last on_chunk boundary, per
    trunk generation. M128 chunks ping-pong xA128/xB128 (64 blocks, even ->
    final rows in xA128; row 127 = the boundary token) and NEVER touch xA64 —
    the R1-era xA64-row-63 law is M64-only. Under the M128 trunk the xA64
    read returned ensure64 POISON (7.7e31 — FINITE in fp32, but x^2 in the
    RMS norm overflows to inf -> fp16 head logits NaN -> h_argmax leaves
    tok_slot 0 = the G1 multi-node cur=0 law). Out-of-range values raise
    (the ingest callback skips the node — loudly)."""
    try:
        import pf_prefill as _pfp
        m128 = bool(getattr(_pfp, "M128", False)) and bool(getattr(_pfp, "_M128ON", False))
    except Exception:
        m128 = False
    if m128 and "xA128" in P.d:
        h = P.down_at("xA128", 127 * 5120 * 4, 5120, np.float32)
    else:
        h = P.down_at("xA64", 63 * 5120 * 4, 5120, np.float32)
    _assert_sane_boundary_vec(h, "hlast")
    return h

def _assert_sane_boundary_vec(h, what):
    """Trunk-hidden rows are O(1..1e3); the poison class (7.7e31) and NaN/inf
    both fail. Catches trunk-generation mismatches BEFORE they reach the
    fp16 head path (where they become silent argmax-0 garbage)."""
    if not np.isfinite(h).all() or float(np.abs(h).max()) > 1e8:
        raise RuntimeError(f"pcache: {what} out of range at chunk boundary "
                           f"(absmax={float(np.abs(h).max()):.3e}; trunk-generation mismatch?)")


def capture_node(E, A, B, source, fed_prefix=None, parent=None, dhd=None, hlast=None, cur=None):
    """Download a node [A, B) from live engine state. source:
    - 'midprefill': 64-chunk quiescent boundary (trunk GDN in rec{i}/conv{i}_0,
      dhd via the proven end-of-chunk REC1 law, hlast per trunk generation
      [_trunk_boundary_hlast — the G1 fix])
    - 'turnend':    quiescent turn end (spec GDN slot 4, dhd_seed, h_seed)
    R6 P3: dhd=/hlast= overrides supply the T=1-trunk-world equivalents
    (draft-chain state in hd_d1 + trunk hidden in x0 at the boundary) — used
    by the batch daemon's chunked T=1 prefill where REC1/xA64 do not exist.
    Returns the node dict (caller hands to PC.write_node)."""
    P = E.P
    if "sc_d" not in P.d:
        raise RuntimeError("prompt cache requires KV8=1 (sc_d buffer absent)")
    W = B - A
    node = {"pos_start": A, "pos_end": B, "parent": parent, "config_fp": config_fp(),
            "hkey": hkey_prefix(fed_prefix[:B])}
    if cur is not None:
        node["cur"] = int(cur)   # R6 P3: the T=1 trunk argmax sits in tok_slot at
                                 # the boundary (free, no eager trio) -> restore
                                 # skips the pfk_n16 cur-derivation entirely.
    kvb = np.empty((len(E.attn_idx), W * 2 * 4 * 256), dtype=np.uint8)
    sc = np.empty((len(E.attn_idx), W * 2 * 4 * 8), dtype=np.float16)
    for j, i in enumerate(E.attn_idx):
        kvb[j] = P.down_at(f"kv{i}", A * 2 * 4 * 256, W * 2 * 4 * 256, np.uint8)
        sc[j] = P.down_at(f"sc{i}", A * 2 * 4 * 8, W * 2 * 4 * 8, np.float16)
    node["kvb"], node["sc"] = kvb, sc
    node["kvd"] = P.down_at("kv_d", A * 2 * 4 * 256, W * 2 * 4 * 256, np.uint8)
    node["scd"] = P.down_at("sc_d", A * 2 * 4 * 8, W * 2 * 4 * 8, np.float16)
    from mtp import RBLK, CBLK
    if source == "midprefill":
        rec = np.empty((len(E.gdn_idx), RBLK), dtype=np.float32)
        conv = np.empty((len(E.gdn_idx), CBLK), dtype=np.float32)
        for j, i in enumerate(E.gdn_idx):
            rec[j] = P.down_at(f"rec{i}", 0, RBLK, np.float32)
            conv[j] = P.down_at(f"conv{i}_0", 0, CBLK, np.float32)
        node["dhd"] = dhd if dhd is not None else P.down_at("REC1", 16 * 5120 * 4, 5120, np.float32)  # end-of-chunk law
        # hlast per trunk generation (G1 fix; W3.6). cur is NOT derived here:
        # no eager trio mid-prefill (scratch/slot side effects on the in-flight
        # run); restore_chain derives it quiescently from hlast, or uses meta cur
        # (boot node / exact nodes).
        node["hlast"] = hlast if hlast is not None else _trunk_boundary_hlast(P)
        _assert_sane_boundary_vec(node["dhd"], "dhd")
    else:  # turnend: spec-world slot 4 + committed seeds
        rec = np.empty((len(E.gdn_idx), RBLK), dtype=np.float32)
        conv = np.empty((len(E.gdn_idx), CBLK), dtype=np.float32)
        for j, i in enumerate(E.gdn_idx):
            rec[j] = P.down_at(f"rec4", (j * 5 + 4) * RBLK * 4, RBLK, np.float32)
            conv[j] = P.down_at(f"conv4", (j * 5 + 4) * CBLK * 4, CBLK, np.float32)
        node["dhd"] = P.down_at("dhd_seed", 0, 5120, np.float32)
        node["hlast"] = P.down_at("h_seed", 0, 5120, np.float32)  # committed-pos trunk hidden
    node["rec"], node["conv"] = rec, conv
    return node


def _verify_chain_artifacts(root, chain, attn_n, gdn_n, RBLK, CBLK):
    """W3.1/W3.2 transactional validation: EVERY artifact of EVERY node is
    opened + size-checked + shape-checked (meta AND engine-derived) + sha256-
    verified (when recorded) BEFORE any win_up. Returns {hk: (meta, arrs)};
    raises NodeCorrupt on the first bad node — the caller quarantines it and
    falls back FRESH with the engine untouched."""
    out = {}
    prev_end = None
    legacy = [0]
    for hk, e in chain:
        d = os.path.join(root, e["dir"])
        A, B = int(e["pos_start"]), int(e["pos_end"])
        if A < 0 or B <= A:
            raise NodeCorrupt(hk, f"bad window [{A},{B})")
        if prev_end is not None and A != prev_end:
            raise NodeCorrupt(hk, f"chain adjacency broken: start {A} != prev end {prev_end}")
        prev_end = B
        W = B - A
        want = {"kvb": (attn_n, W * 2048), "sc": (attn_n, W * 64),
                "kvd": (W * 2048,), "scd": (W * 64,),
                "rec": (gdn_n, RBLK), "conv": (gdn_n, CBLK),
                "dhd": (5120,), "hlast": (5120,)}
        try:
            meta = json.load(open(f"{d}/meta.json"))
        except Exception as ex:
            raise NodeCorrupt(hk, f"meta.json unreadable: {ex!r}")
        sizes = e.get("sizes") or {}
        if not sizes:
            raise NodeCorrupt(hk, "no sizes recorded (hand-crafted/legacy manifest)")
        shas = e.get("sha256") or {}
        shapes = e.get("shapes") or {}
        arrs = {}
        for nm in _ART:
            if nm not in sizes:
                continue                      # legacy node (pre-W3): artifact unrecorded
            p = f"{d}/{nm}.npy"
            try:
                if os.path.getsize(p) != int(sizes[nm]):
                    raise NodeCorrupt(hk, f"{nm}.npy size {os.path.getsize(p)} != meta {sizes[nm]}")
                arrs[nm] = np.load(p, mmap_mode="r")
            except NodeCorrupt:
                raise
            except Exception as ex:
                raise NodeCorrupt(hk, f"{nm}.npy unreadable: {ex!r}")
            if tuple(arrs[nm].shape) != want[nm]:
                raise NodeCorrupt(hk, f"{nm}.npy shape {arrs[nm].shape} != engine-derived {want[nm]}")
            if nm in shapes and tuple(arrs[nm].shape) != tuple(shapes[nm]):
                raise NodeCorrupt(hk, f"{nm}.npy shape {arrs[nm].shape} != meta {shapes[nm]}")
            if HASH_VERIFY and nm in shas:
                if _sha256_file(p) != shas[nm]:
                    raise NodeCorrupt(hk, f"{nm}.npy sha256 mismatch (corrupt or planted)")
            elif not shas:
                legacy[0] += 1
        # R1-era nodes wrote hlast.npy WITHOUT a sizes record — pull it in when
        # meta says it exists (shape + finiteness still enforced)
        if meta.get("has_hlast") and "hlast" not in arrs:
            p = f"{d}/hlast.npy"
            try:
                arrs["hlast"] = np.load(p, mmap_mode="r")
            except Exception as ex:
                raise NodeCorrupt(hk, f"hlast.npy unreadable: {ex!r}")
            if tuple(arrs["hlast"].shape) != want["hlast"]:
                raise NodeCorrupt(hk, f"hlast.npy shape {arrs['hlast'].shape} != {want['hlast']}")
            legacy[0] += 1
        # the state-critical hlast must be sane before it feeds the fp16 head
        # path (the ensure64 poison class 7.7e31 is FINITE but overflows the
        # RMS norm -> NaN logits -> the G1 argmax-0 law)
        if arrs.get("hlast") is not None:
            hl = np.asarray(arrs["hlast"])
            if not np.isfinite(hl).all() or float(np.abs(hl).max()) > 1e8:
                raise NodeCorrupt(hk, "hlast out of range (poison/corrupt)")
        out[hk] = (meta, arrs)
    if legacy[0]:
        print(f"[pcache] note: {legacy[0]} legacy artifacts verified size+shape only (no sha256 recorded)", flush=True)
    return out


def restore_chain(E, chain, root=None):
    """Upload a full chain into the engine: KV windows [0,B) per layer, draft
    KV [0,B), GDN slot 4 from the LAST node, dhd_seed, cur via head-argmax on
    hlast (or the stored cur). Leaves the engine at pos=B ready to generate or
    follow_up. Returns (B, cur).

    W3: TRANSACTIONAL — the whole chain is validated on disk (size+shape+sha,
    adjacency, hlast finiteness) BEFORE the first win_up; a corrupt node
    raises NodeCorrupt with the engine untouched (caller: FRESH fallback)."""
    from mtp import RBLK, CBLK
    from engine0 import dev
    from trunk import VOCAB
    P = E.P
    root = root or ROOT
    B = int(chain[-1][1]["pos_end"])
    # ---- phase 1: validate EVERYTHING (no GPU touches yet) ----
    verified = _verify_chain_artifacts(root, chain, len(E.attn_idx), len(E.gdn_idx), RBLK, CBLK)
    # ---- phase 2: upload (all files already open+mmapped: eviction racing
    # us now cannot break the reads — POSIX keeps the mappings) ----
    for hk, e in chain:
        meta, arrs = verified[hk]
        A = int(e["pos_start"])
        kvb, sc = arrs["kvb"], arrs["sc"]
        for j, i in enumerate(E.attn_idx):
            P.win_up(f"kv{i}", A * 2 * 4 * 256, kvb[j])
            P.win_up(f"sc{i}", A * 2 * 4 * 8, sc[j])
        P.win_up("kv_d", A * 2 * 4 * 256, arrs["kvd"])
        P.win_up("sc_d", A * 2 * 4 * 8, arrs["scd"])
        E._flush()
    hk_last, e_last = chain[-1]
    meta, arrs = verified[hk_last]
    rec, conv = arrs["rec"], arrs["conv"]
    for j, i in enumerate(E.gdn_idx):
        P.win_up("rec4", (j * 5 + 4) * RBLK * 4, rec[j])
        P.win_up("conv4", (j * 5 + 4) * CBLK * 4, conv[j])
        if j % 16 == 0: dev.synchronize()
    P.win_up("dhd_seed", 0, arrs["dhd"])
    # cur: stored, or head-argmax on the recorded trunk hidden (same kernels
    # that produced it -> bit-identical). h_argmax advances pos_slot, so seed
    # pos_slot = B-1 first, then reset all slots explicitly.
    if meta.get("cur") is not None:
        cur = int(meta["cur"])
    else:
        if arrs.get("hlast") is None:
            raise NodeCorrupt(hk_last, "cur missing and no hlast to derive it")
        hl = np.array(arrs["hlast"])       # finite-checked in phase 1
        P.win_up("hd_d0", 0, hl)
        P.win_up("pos_slot", 0, np.array([B - 1], dtype=np.int32))
        E.pr["pfk_n16"](P.d["hd_d0"], E.W[("onw", 0)], P.d["xh"],
                        global_size=(1, 1, 1), local_size=(256, 1, 1))
        E.pr["head8"](E.W[("head", 0)], P.d["xh"], P.d["logits"],
                      global_size=(VOCAB // 8, 1, 1), local_size=(256, 1, 1))
        E.pr["h_argmax"](P.d["logits"], P.d["tok_slot"], P.d["pos_slot"], P.d["tok_hist"],
                         global_size=(1, 1, 1), local_size=(256, 1, 1), wait=True)
        cur = int(P.down_at("tok_slot", 0, 1)[0])
    E._reset_slots(cur, B)
    P.win_up("dhd_seed", 0, arrs["dhd"])   # after _reset_slots (it zeros dhd)
    dev.synchronize()
    P._keep.clear()
    return B, cur

def PC_ROOT_DIR():
    return ROOT


# ---------- boot node from the base snapshot (restart-resume of the parked
# conversation; state at P0 from snap files + device draft KV/hd) ----------
def boot_node(E, snapdir, P0, CUR0, ids, progress=None):
    """R1 rev2 — DEVICE ROUNDTTRIP: node at exactly P0 captured from the live
    PARKED buffers (boot: load_snapshot_kv + seed_mtp_slots + fill_draft all
    done). Bit-exact by construction (same down/win expressions as the G1/G3
    roundtrip paths). The rev1 snap-fp16->int8 re-quantization diverged from
    the device rows (G4 fingerprint kv maxdiff 236) — retired.
    Non-aligned end: partial-tail chain hash — only an identical fed stream
    ever probes it. dhd from hd_d1 (the boot fill_draft chain end)."""
    P = E.P
    node = {"pos_start": 0, "pos_end": P0, "parent": None, "config_fp": config_fp(),
            "hkey": hkey_prefix(ids[:P0]), "cur": int(CUR0)}
    kvb = np.empty((len(E.attn_idx), P0 * 2048), np.uint8)
    sc = np.empty((len(E.attn_idx), P0 * 64), np.float16)
    for j, i in enumerate(E.attn_idx):
        kvb[j] = P.down_at(f"kv{i}", 0, P0 * 2048, np.uint8)
        sc[j] = P.down_at(f"sc{i}", 0, P0 * 64, np.float16)
        if progress: progress(j + 1, len(E.attn_idx))
    node["kvb"], node["sc"] = kvb, sc
    node["kvd"] = P.down_at("kv_d", 0, P0 * 2048, np.uint8)
    node["scd"] = P.down_at("sc_d", 0, P0 * 64, np.float16)
    from mtp import RBLK, CBLK
    rec = np.empty((len(E.gdn_idx), RBLK), dtype=np.float32)
    conv = np.empty((len(E.gdn_idx), CBLK), dtype=np.float32)
    for j, i in enumerate(E.gdn_idx):
        rec[j] = P.down_at("rec4", (j * 5 + 4) * RBLK * 4, RBLK, np.float32)
        conv[j] = P.down_at("conv4", (j * 5 + 4) * CBLK * 4, CBLK, np.float32)
    node["rec"], node["conv"] = rec, conv
    node["dhd"] = P.down_at("hd_d1", 0, 5120, np.float32)
    return node


# ---------- FRESH prefill with ingest (single source of truth for
# the daemon path AND the gates) ----------
def fresh_prefill(E, G, toks, prog=None, log=None, ingest=None, log_ingest=None):
    """The serve h_prefill FRESH body verbatim: reset_fresh + stload_trunk +
    conv-parity-0 zero (M1-B FIX #2) + standalone fill_draft for the short/
    no-dfill case + batched trunk prefill + stseed_spec + slot resets.
    ingest: optional callback(pos_after_chunk, final64) fired at every 64-chunk
    quiescent boundary (the caller decides which become nodes). Returns
    (newcur, posn, final64)."""
    from engine0 import dev
    import pf_prefill
    E.reset_fresh(toks[0])
    E.stload_trunk()
    for _i in E.gdn_idx:
        from mtp import CBLK as _CB
        E._mfill(f"conv{_i}_1", 0, _CB)
    dev.synchronize()
    skip_fd = len(toks) >= 16 and os.getenv("PF_DFILL", "1") == "1"
    if not skip_fd:
        E.fill_draft(toks, start_pos=0, seed_hd=None,
                     prog=(lambda d, t: prog(d, t, "fill_draft")) if prog else None)
    final64 = (len(toks) // HASH_BLK) * HASH_BLK
    _noc = [0]
    print(f"[pcache] fresh_prefill n={len(toks)} skip_fd={skip_fd} ingest={'yes' if ingest else 'no'} final64={final64}", flush=True)

    def on_chunk(pos_after):
        _noc[0] += 1
        if _noc[0] == 1 or pos_after % 1024 == 0:
            print(f"[pcache] on_chunk #{_noc[0]} pos={pos_after}", flush=True)
        if ingest is not None:
            try:
                ingest(pos_after, final64)
            except Exception as e:
                import traceback; traceback.print_exc()
                if log_ingest: log_ingest("ingest_error", error=repr(e))

    pf_prefill.prefill_batch(E, G, toks,
                             prog=(lambda d, t: prog(d, t, "prefill_batch")) if prog else None,
                             log=log, on_chunk=on_chunk if skip_fd else None)
    E.stseed_spec(len(toks) & 1)
    newcur = int(E.P.down_at("tok_slot", 0, 1)[0])
    E.P.win_up("cur_slot", 0, np.array([newcur], dtype=np.int32))
    E.P.win_up("h_seed", 0, np.zeros(5120, dtype=np.float32))
    E.P.win_up("dring0", 0, np.full(1, -1, dtype=np.int32))
    E.P.win_up("dring1", 0, np.full(1, -1, dtype=np.int32))
    dev.synchronize()
    posn = int(E.P.down_at("pos_slot", 0, 1)[0])
    return newcur, posn, final64
