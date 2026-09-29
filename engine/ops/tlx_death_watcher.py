#!/usr/bin/env python3
"""TLX L7 death watcher (out-of-process flight recorder for the silent reset).

PURPOSE (engine-side analysis 2026-09-27): the soak deaths leave NOTHING --
no slog tail, no crashlog, no unified-log tail (logd buffers die with the box).
This watcher is a separate, non-GPU process whose ONLY job is to write lines
that SURVIVE the hard reset (fsync per line + dir-fsync), so the next boot can
answer THREE questions the engine log cannot:

  1. Did the GPU python process die/disappear BEFORE the machine reset?
     (jetsam/kill -> GPU-EXIT-law reboot)  vs  the box dying with it alive
     (device-side GSP/SM fault -> reset).
  2. What was the memory-pressure / RSS trajectory of the GPU python in the
     60s before death?  (jetsam hypothesis needs rising pressure + big RSS)
  3. What engine phase was live at death?  (we mirror the engine's own
     persistent slog tail: op/stage/ts of the last line -> correlates the
     reset with generate-first-cycle / rebuild / prefill-cancel windows)

RUN (on the rig):
  nohup python3 /tmp/tlx_death_watcher.py >/tmp/tlx_watcher.out 2>&1 &

STOP: touch /tmp/tlx_watcher.stop  (or pkill -f tlx_death_watcher)

Cost: one fsynced ~200B line per 0.5s (~24KB/min); rotates at 8MB.
Read after a death:  ~/tinygrad-metal/logs/tlx_death_watch.log
"""
import os, re, time, subprocess

LOG = "~/tinygrad-metal/logs/tlx_death_watch.log"
ENGINE_LOG = "~/tinygrad-metal/logs/llm-engine-launchd.log"
PIDFILE = "~/tinygrad-metal/logs/engine.pid"
STOP = "/tmp/tlx_watcher.stop"
INTERVAL = 0.5
ROTATE = 8 * 1024 * 1024

def fsync_dir(path):
    try:
        dfd = os.open(path, os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    except Exception: pass

def wline(line):
    try:
        new = not os.path.exists(LOG)
        with open(LOG, "a") as f:
            f.write(line + "\n"); f.flush(); os.fsync(f.fileno())
        if new: fsync_dir(os.path.dirname(LOG))
        if os.path.getsize(LOG) > ROTATE:
            os.replace(LOG, LOG + ".1"); fsync_dir(os.path.dirname(LOG))
    except Exception: pass

def engine_pid():
    try:
        return int(open(PIDFILE).read().strip())
    except Exception:
        pass
    try:
        out = subprocess.run(["pgrep", "-f", "python.*-u test_w100k.py"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        pids = [int(x) for x in out.split()] if out else []
        return pids[0] if pids else None
    except Exception:
        return None

def proc_info(pid):
    if pid is None: return "gone"
    try:
        out = subprocess.run(["ps", "-o", "rss=,state=,etime=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5).stdout.split()
        return f"rss={int(out[0])//1024}MB st={out[1]} up={out[2]}"
    except Exception:
        return "gone"

_OP_RE = re.compile(r'"op": "([^"]+)"')
_STAGE_RE = re.compile(r'"stage": "([^"]*)"')
_TS_RE = re.compile(r'"ts": ([0-9.]+)')
def engine_tail():
    """Last serve:slog line's (op, stage, ts) -- the engine's own phase marker."""
    try:
        with open(ENGINE_LOG, "rb") as f:
            f.seek(0, 2); n = f.tell()
            f.seek(max(0, n - 65536))
            tail = f.read().decode(errors="replace").splitlines()
        for line in reversed(tail):
            if "serve:slog" in line:
                i = line.find("{"); body = line[i:]
                mo = _OP_RE.search(body); ms = _STAGE_RE.search(body); mt = _TS_RE.search(body)
                return (mo.group(1) if mo else "?", ms.group(1) if ms else "",
                        mt.group(1) if mt else "?")
    except Exception: pass
    return ("?", "", "?")

def mem_pressure():
    try:
        out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        # 1=normal 2=warn 3=critical
        return {"1": "norm", "2": "warn", "3": "crit"}.get(out.strip(), out.strip() or "?")
    except Exception:
        return "?"

def main():
    wline(f"[watcher] start ts={time.time():.1f} pid={os.getpid()}")
    last_pid = None; last_tail = ("", "", ""); n = 0
    while not os.path.exists(STOP):
        t0 = time.time()
        pid = engine_pid()
        info = proc_info(pid)
        # log EXTRA lines on state TRANSITIONS (cheaper discrimination than
        # flooding): pid appears/vanishes, engine phase change, pressure change
        tail = engine_tail()
        pressure = mem_pressure() if n % 6 == 0 else "="   # every 3s
        transition = (pid != last_pid) or (tail[:2] != last_tail[:2])
        if transition or n % 4 == 0:   # ~1 per 2s baseline + all transitions
            wline(f"ts={t0:.1f} eng_pid={pid} {info} pres={pressure} "
                  f"eng_op={tail[0]} eng_stage={tail[1]} eng_ts={tail[2]}")
        if pid is not None and last_pid is not None and pid != last_pid:
            wline(f"[watcher] ENGINE PID CHANGED {last_pid} -> {pid} ts={t0:.1f} (restart?)")
        if pid is None and last_pid is not None:
            wline(f"[watcher] ENGINE PID {last_pid} VANISHED ts={t0:.1f} -- "
                  f"process died while machine still up (kill/jetsam class; "
                  f"GPU-EXIT reboot expected within seconds)")
        last_pid = pid; last_tail = tail; n += 1
        dt = time.time() - t0
        if dt < INTERVAL: time.sleep(INTERVAL - dt)
    wline(f"[watcher] stop ts={time.time():.1f}")

if __name__ == "__main__":
    main()
