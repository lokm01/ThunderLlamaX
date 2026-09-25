# W0 DOSSIER — TinyGPU dext + userspace runtime anatomy (2026-09-13)

Read-only survey. Sources: `~/tinygrad-src/extra/usbgpu/tbgpu/` (dext+installer),
`~/tinygrad-src/tinygrad/runtime/{ops_nv.py, graph/hcq.py, support/hcq.py, support/system.py,
support/nv/nvdev.py, support/nv/ip.py, support/memory.py}`, rig state checks.
All file:line refs are to these files. Abbreviations: ONV=ops_nv.py, HCQ=support/hcq.py,
SYS=support/system.py, GHC=graph/hcq.py, IP=support/nv/ip.py.

==========================================================================================
1. THE DEXT — WHAT THOSE 389 LINES ACTUALLY DO
==========================================================================================
Location: ~/tinygrad-src/extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/
Files: TinyGPUDriver.iig/.cpp (198 ln), TinyGPUDriverUserClient.iig/.cpp (191 ln),
Info.plist, 4 entitlements variants.

* TinyGPUDriver (IOService): matches IOPCIDevice class 0x03000000, IOPCITunnelCompatible
  (Info.plist). Start_Impl: opens the PCI device, enables IO|BusMaster|MemorySpace
  (cpp:57-60), registers service name "tinygpu" (cpp:62-66).
* Local methods (TinyGPUDriver.cpp):
  - MapBar (95-103): maps a BAR via GetBARInfo + _CopyDeviceMemoryWithIndex.
  - SetupDMA (121-136): IODMACommand (maxAddressBits=40!) PrepareForDMA -> up to 32
    IOAddressSegments. This is the ONLY DMA machinery.
  - CreateDMA (138-157): IOBufferMemoryDescriptor (in/out, page aligned) + SetupDMA +
    writes the scatter-gather list [addr0,len0,...,0,0] into the shared buffer at page+1
    offset. I.e. the GPU gets PHYSICAL ADDRESSES of host memory via an SG list; the GPU
    then pushbuffer-references host RAM directly. There is NO copy engine, NO interrupt,
    NO notifier, NO semaphore, NO event, NO workloop in the dext.
  - CfgRead/CfgWrite (159-186), ResetDevice (188-193: FunctionLevelReset else HotReset).
* TinyGPUDriverUserClient — the ENTIRE userspace API surface is 4 ExternalMethod selectors
  (TinyGPURPC enum, UserClient.iig:12-17): ReadCfg(0), WriteCfg(1), Reset(2), PrepareDMA(3).
  Plus CopyClientMemoryForType (UserClient.cpp:167-190): type<6 = map BAR #type;
  else type=size = CreateDMA shared buffer (returned to user as a mapping).
  Every ExternalMethod os_log()s (UserClient.cpp:103) — cheap unless a log stream is live.
* THERE IS NO WAIT/SYNC PATH IN THE DEXT. No sleep, no blocking, no completion interrupt.
  Everything that observes GPU completion is a userspace read of DMA'd host memory.
  => The 17ms sync CANNOT be a dext sleep. (Also no graph/object budget: the only dext-side
  counter is the per-client `dmas[]` array which DOUBLES on demand, UserClient.cpp:15-34.)

THE REAL "DRIVER": a userspace C server, ~/tinygrad-src/extra/usbgpu/tbgpu/installer/Shared/server.c
(285 ln), built INTO TinyGPU.app and run as `TinyGPU server <sock>`:
  * UNIX-socket protocol: PROBE/MAP_BAR/MAP_SYSMEM_FD/CFG_*/RESET/MMIO_READ/MMIO_WRITE/
    MAP_SYSMEM/SYSMEM_READ/SYSMEM_WRITE/RESIZE_BAR (server.c:19-33).
  * MAP_SYSMEM_FD (129-165): shm_open + ftruncate + mmap, then dext RPC 3 (PrepareDMA)
    to get physical segments, writes them into the shm, passes the fd to the client via
    SCM_RIGHTS. *** `#define MAX_SYSMEM 128` (server.c:42) and g_sysmem slots are freed
    ONLY at client disconnect (cleanup(), server.c:171-183) — THIS is the real "~128 host
    mapping cap" (previously attributed to the dext). Fix = rebuild the APP (C file), no
    dext change needed.***
  * MMIO_READ/WRITE (235-247): bulk copies through a 64MB staging buffer over the socket.
  * Single client; second client rejected (274).
* The python client side: APLRemotePCIDevice (SYS:419-459) auto-spawns
  `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU server <sock>` (SYS:439) on
  temp("tinygpu.sock"); device lock file = `temp("nv_usb4.lock")` = /tmp/nv_usb4.lock
  (SYS:140-152 flock_acquire + PCIDevice ctor SYS:162; devpref "nv" + pcibus "usb4" from
  APLRemotePCIDevice super().__init__(devpref, "usb4", ...) SYS:442).
* GPU control stack above the socket: NVDev (nvdev.py:74-119) maps BAR0/BAR1 through the
  RemoteMMIOInterface (= every register read/write is a socket RPC), boots GSP firmware
  (NV_GSP, ip.py:347+), and PCIIface (ONV:560-586) issues RM alloc/control as GSP RPCs
  (rpc_rm_alloc/rpc_rm_control, ONV:578-580). Ampere classes per ip/gsp.

==========================================================================================
2. REBUILDING / RESIGNING THE DEXT (and the app)
==========================================================================================
Three routes, in increasing invasiveness:

A) OFFICIAL (what tinygrad ships) — installer/build_and_sign.sh and build_and_sign_nv.sh:
   xcodebuild (unsigned build) + copy provision profiles from ../profiles/ (NOT present on
   this machine) + codesign with "Developer ID Application: tinygrad, Corp. (9YG3G8543N)"
   (--options runtime --timestamp) on dext then app, then spctl verify; notary_tool.sh for
   notarization (keychain profile "hgwJFhdheiIEy82nDN"). WE CANNOT USE THIS (no tinygrad
   cert, no profiles dir). Current rig runs exactly this build:
   systemextensionsctl list -> [activated enabled] 9YG3G8543N ... (1.0.0/3), SIP ENABLED.

B) NoSIP DEV ROUTE — installer/install_nosip.sh:
   * REQUIRES SIP DISABLED (script hard-fails otherwise; csrutil disable from Recovery).
   * xcodebuild Debug, CODE_SIGNING_REQUIRED=NO, then AD-HOC sign (`codesign --sign -`)
     with TinyGPUDriver.NoSIP.entitlements (matches ANY PCI id 0xFFFFFFFF&0, plus
     com.apple.developer.driverkit.allow-any-userclient-access), app signed with
     macOS/macOS.entitlements (has com.apple.developer.system-extension.install +
     driverkit.userclient-access for org.tinygrad.tinygpu.driver2).
   * Install: rm -rf /Applications/TinyGPU.app; cp -r build/Debug/TinyGPU.app /Applications;
     `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU install` -> OSSystemExtensionRequest
     activation (TinyGPUCLIRunner.swift) -> approve in System Settings > Privacy & Security.
     Reboot typically required to complete activation/replace.
   * NOTE: replacing the dext also replaces server.c (same app bundle) — one rebuild gets
     you both the MAX_SYSMEM raise and any dext change.

C) PATCH-ONLY-WHAT-WE-NEED: the MAX_SYSMEM=128 cap is in server.c (userspace app), so the
   cheapest impactful rebuild is: edit server.c (e.g. MAX_SYSMEM 128 -> 1024), rebuild the
   APP via install_nosip.sh --build (still needs SIP off for the dext load? — the dext
   binary would be the SAME ad-hoc-signed one; if you keep the currently-active tinygrad-
   signed dext and only swap the app wrapper, the dext code signature no longer matches the
   app's... in practice DriverKit requires the dext inside the app to be re-activated, so
   treat route C as route B unless experimenting with `systemextensionsctl developer on`).
   Also: APLRemotePCIDevice.ensure_app (SYS:422-430) re-downloads TinyGPU.zip from
   github.com/tinygrad/tinygpu_releases (commit c0d024f9...) into the downloads dir if
   either the zip or /Applications/TinyGPU.app is missing — after installing a custom
   build, keep/mirror that zip path or ensure_app will clobber /Applications on a fresh
   device init.

Difficulty verdict: dext itself is trivial C++/DriverKit (389 ln) but rebuild+activate
requires either tinygrad's cert (unavailable) or disabling SIP (recovery reboot) and user
approval. One-time cost; not needed for MAX_SYSMEM if... (it IS in the app bundle, so it
does need the same dance). SIP currently ENABLED; disabling is a user decision with
security implications (FileVault/remote-access rig).

==========================================================================================
3. THE SYNC PATH — GPU COMPLETION -> HOST OBSERVING IT (the ~17ms question)
==========================================================================================
Step-by-step, with file:line:

SUBMIT SIDE (before completion):
 1. HCQProgram.__call__ (HCQ:378-404): fills kernargs (host shm), builds an
    NVComputeQueue: wait(timeline-1) + memory_barrier + exec(...) + signal(timeline).
 2. exec (ONV:135-158): QMD is EMBEDDED IN KERNARGS (args_state.buf + round_up(cbuf,256)),
    grid dims written into QMD raster fields (see §5), launch = SEND_PCAS_A (ONV:151-152)
    or chained dependent-QMD (ONV:154).
 3. signal (ONV:159-178): either QMD release{} fields (fused into the running kernel's
    QMD) or NVC56F_SEM_ADDR_LO/SEM_EXECUTE with operation="release", release_wfi="en",
    payload 64bit + release_timestamp + NVC56F_NON_STALL_INTERRUPT (ONV:174-177).
 4. _submit_to_gpfifo (ONV:114-126): unbound queues copy the command list into the shared
    2MB cmdq ring (dev.cmdq, ONV:627-629, BumpAllocator wrap=True, memory.py:5-12),
    write the gpfifo ring entry, then the DOORBELL: `dev.gpu_mmio[0x90//4] = token`.
    For PCIIface gpu_mmio = BAR0 mapping at +0xbb0000 (ONV:575). On this rig BAR access =
    RemoteMMIOInterface (SYS:317-335) => the doorbell is a fire-and-forget UNIX-socket
    MMIO_WRITE (server.c:243-247, no response). ~socket-write latency (tens of us).

COMPLETION -> OBSERVATION:
 5. The GPU's host interface (PBDMA over the GSP-booted channel) executes the pushbuffer,
    which lives in HOST sysmem (the cmdq ring is cpu_access=True -> MAP_SYSMEM_FD shm) —
    fetched over TB4. Kernels run; the final SEM_EXECUTE release (with WFI) DMAs the
    16-byte signal value+timestamp into the signal page = another host shm segment whose
    PHYSICAL addresses were registered with the GPU via PrepareDMA (SYS:444-459 ->
    server.c:129-165 -> dext SetupDMA). Snooped/uncached mapping (PCIIfaceBase.alloc,
    SYS:267-277: map_range(..., snooped=True, uncached=True)).
 6. Host observes it by pure spinning: HCQSignal.wait (HCQ:300-315) loops on
    `self.value` = base_buf.cpu_view().view(0,8,'Q')[0] (HCQ:277) — a direct mmap read of
    the shm page in the python process. *** NVSignal._sleep (ONV:27-30) ONLY sleeps when
    time_spent_since_last_sleep_ms > 200 (and then iface.sleep(200), which for PCIIface
    drains the GSP stat queue, ONV:583-586). CONFIRMED: no sleep quantum under 200ms —
    the "hcq sleep hypothesis" for the 17ms is REFUTED. ***
 7. synchronize() (HCQ:462-479) = timeline_signal.wait(timeline_value-1); on timeout ->
    on_device_hang (ONV:743-761) reads SM error states via GSP RPC.

WHAT REMAINS AS 17ms CANDIDATES (userspace adds ~nothing after completion):
  (a) GPU-side release pipeline: release_wfi forces wait-for-idle before the semaphore
      DMA (ONV:174-177); on this GSP-firmware-driven, TB-attached channel the PBDMA/GSP
      processing latency of the release + the DMA write over TB is the dominant term.
      Note each "sync" also implies the GPU must first DRAIN every queued command
      (in-graph kernel gap 6.7us x queued kernels, eager launch 46us x queued kernels).
  (b) PBDMA pushbuffer fetch from host sysmem over TB (round trips per command chunk).
  (c) DART/IOMMU snooped-write translation on M2 for the signal DMA (unlikely 17ms).
  (d) NOT the doorbell (fire-and-forget socket write), NOT a host sleep, NOT a dext
      roundtrip (dext has no completion path at all).
  Measure-next suggestion: instrument around ONV:125 (time the doorbell write) and
  around HCQ:310 (poll iteration count x poll cost) to split submit vs observe latency;
  also compare a QMD-release signal (fused, ONV:160-172) vs explicit SEM path.

Related userspace waits that DO exist (for completeness):
  - GSP RPC wait_resp (IP:87-91): busy spin, 10s timeout, used at init/rm_control only.
  - copy path _copyout (HCQ:637-650): synchronize() + per-2MB chunk timeline wait.
  - LRUAllocator._free -> dev.synchronize() per freed buffer (HCQ:590-593).

==========================================================================================
4. THE GRAPH BUDGET (~50 blocks-worth of piece-graphs at 100k)
==========================================================================================
Userspace structures that COULD bound captured graphs (all inventoried):
  * graph_cache = weakref.WeakKeyDictionary (engine/realize.py:133) — no fixed cap;
    lifetime bounded by AST references held during a TinyJit capture.
  * Per-graph fixed costs (GHC:19-233): signals from a RECYCLED pool (HCQ:488-494;
    new pages of 0x1000 = one MAP_SYSMEM_FD slot each, class-level, never freed);
    kernargs from the device's DEDICATED graph half-pool: 64MB total, 32MB graph half,
    BumpAllocator wrap=False (HCQ:443-455) -> exhaustion raises RuntimeError("Out of
    memory") LOUDLY (memory.py:9) — a device fault is NOT this.
    Per-exec kernargs+QMD ~2.5KB (ONV:183-184 kernargs_alloc_size = cbuf + 8<<8).
  * Unbound (MTP_GRAPH_NOBIND=1, GHC:10-17,228-230) graph submits share the per-device
    2MB cmdq ring (ONV:627-629) — BumpAllocator wrap=True SILENTLY WRAPS and overwrites
    (memory.py:8-11). If python enqueues >2MB of not-yet-consumed commands (easy when
    100k-shaped kernels take ms each and capture runs ahead), the wrap CLOBBERS pending
    pushbuffer -> gpfifo entries point at garbage -> DEVICE FAULT. Budget would be
    total-bytes-in-flight, per-process, merge-invariant, shape-dependent (slower kernels
    at 100k -> deeper backlog), and 2k-clean. Volume estimate for current workloads
    (~100-300KB in flight) makes this a CANDIDATE, not a proof.
  * gpfifo ring: 0x10000 entries per channel (ONV:623-625); each submit = 1 entry;
    overflow needs 65k in-flight submits — implausible.
  * server.c MAX_SYSMEM 128 slots (server.c:42,129) — refuted for THIS wall (mappings
    measured flat at 43), but it is the historical "128 cap" and still binds TOTAL live
    host mappings per connection (signal pages + kernargs pool + hw_pages + gpfifo area +
    LRU-cached cpu_access buffers).
  * dext: dmas[] grows unbounded (doubling) — not a budget. GSP RM: graphs allocate NO
    RM objects (they are just pushbuffer submissions on the same 2 channels), so a GSP
    RM-object budget is unlikely; however GSP/channel-side pushbuffer tracking state is
    opaque — cannot be ruled out from source.

VERDICT: no fixed userspace cap matches "~50 blocks-worth, merge-invariant, per-process"
except the 2MB cmdq-ring wrap-clobber (userspace, silent, matches the failure MODE but
volume estimate is short) — ranked hypotheses:
  1. GPU/GSP-side consumption of queued work while capture runs ahead (incl. the 2MB
     ring wrap as the corruption vector) — testable by padding the ring (raise
     cmdq_page 0x200000 -> e.g. 8MB at ONV:627) — a ONE-LINE fork experiment.
  2. GSP firmware per-channel queue accounting — needs the fork-maintainer question.
  3. Not mappings (43 flat), not kernargs pool (would raise, not fault), not gpfifo
     entries (65k), not graph_cache (weakref, unbounded).

==========================================================================================
5. gridDim / cp.async (LDGSTS)
==========================================================================================
gridDim:
  * Hardware launch grid = QMD fields, written HOST-side in NVComputeQueue.exec
    (ONV:144-146): global_size -> cta_raster_width/height/depth (QMDV03 bits
    384-463 = bytes 48-56 of the QMD, nv_570.py: NVC6C0_QMDV02_03_CTA_RASTER_WIDTH =
    (415,384) etc); local_size -> cta_thread_dimension0/1/2 (fmt 'H','H','B').
  * THE CATCH (documented in our own harness): nvcc-compiled SASS reads NTID/NCTAID as
    VALUES from CONSTANT BUFFER c[0][0..8], which tinygrad does NOT populate — see
    a4/gdn_t1_test.py:14 and a4/gdn_debug_test.py:14:
      `prog.cbuf_0[0], prog.cbuf_0[1], prog.cbuf_0[2] = 256, 1, 1  # NTID: nvcc SASS
       reads from c[0][0..8]`
    cbuf_0 is zero-initialized (ONV:308-316 area: self.cbuf_0 = [0]*max(cbuf0_size//4,12);
    only entries 6..12 get window addrs on pre-Blackwell). The QMD still launches the
    right CTA count (raster fields), so flat-index kernels with `if (i<n)` guards work —
    but any kernel that READS gridDim/blockDim (grid-stride loops) sees 0 -> infinite
    loop -> 30s watchdog. THE FIX for future hand kernels: either set prog.cbuf_0[0..2]
    (and the NCTAID slots — empirically [0..2] carried NTID in our harness; verify the
    exact c[0] layout per CUDA ABI before relying on gridDim) or never use grid-stride.
  * NOT in the dext (no involvement in launch encoding at all).

cp.async (LDGSTS):
  * Instruction execution is entirely SM-side. The dext cannot drop it. The userspace
    knobs that gate it: (1) the COMPILER — hand kernels go through the nvcc shim
    (~/.local/bin/nvcc -> Docker cuda-nvcc:12.8, verified working: release 12.8,
    V12.8.93) with `-arch=sm_86 -cubin`; in-model compiles use NVRTC via
    device.py compile_cached (stricter than nvcc). Verify the SASS actually contains
    LDGSTS (cuobjdump/nvdisasm) — nvcc can silently drop the async path if the pattern
    does not fit its heuristic. (2) The QMD/L1 config — smem carveout fields
    min/target/max_sm_config_shared_mem_size + shared_memory_size (built in NVProgram
    __init__, ONV:~312-330; smem_cfg from [32,64,100]KB buckets) and cache-invalidation
    flags on launch. The observed ">=4-array static smem staging hangs" smells like
    smem carveout under-configuration (CTAs never get resources -> wait timeout) —
    .nv.shared.{name} section drives shmem_usage (ONV:~330), so a kernel whose real
    static usage exceeds what the section advertises would hang exactly like that.
  * Where cubins get LOADED: NVProgram.__init__ (ONV:249-255) elf_loader(force_section_
    align=128) + relocs + allocator._copyin(lib_gpu) + dev.synchronize(). See §7.

==========================================================================================
6. RIG STATE + THE ENV PREFIX EVERY GPU EXPERIMENT NEEDS
==========================================================================================
Checked 2026-09-13 ~15:35 (read-only):
  * No GPU python running (ps clean). Dext alive: PID 366 (_driverkit,
    org.tinygrad.tinygpu.driver2, [activated enabled], SIP enabled).
  * NO stale /tmp/nv_usb4.lock (does not exist). No tinygpu.sock server running (it is
    spawned on demand by the first DEV=NV process).
  * Colima up (limactl + usernet running), nvcc shim WORKS:
    CUDA 12.8 V12.8.93 (build 35583870). ~/tg311/bin/python -> python3.11 present.
  * Uptime 2 days, load ~1.5, disk 362Gi free. Machine healthy.
ENV PREFIX (non-interactive ssh lacks BOTH env vars):
  PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH \
  DOCKER_HOST=unix://<colima-socket> \
  <command>                      # e.g. nvcc --version
For python GPU work the canonical prefix remains (see AGENTS.md): cd ~/tinygrad-metal &&
DEV=NV BEAM=1 ... ~/tg311/bin/python <script>; python itself does not need the docker env
unless compiling through the nvcc shim/NVRTC server inside the same command.
Fork git state: ~/tinygrad-src on branch master, remote localfork=github.com/lokm01/tinygrad,
latest commits b8fda0b (MTP_ALLOC_STACK), 57dfed3 (MTP_BEAM_MAXGS), 3133838 (SKV-T1)...
DO NOT clobber local patches. Project ~/tinygrad-metal @ 37a30e5 (v3fam split-wall reading).

==========================================================================================
7. STANDALONE-CUBIN HARNESS PATTERN (copy this, don't rediscover)
==========================================================================================
Canonical examples: ~/tinygrad-metal/pv3/val3.py (attention chain: upload/validate/bench)
and ~/tinygrad-metal/a4/gdn_scan_test.py (megakernel test). Elements:

 1. HEADER (order matters — DEV before tinygrad import):
      import os, sys
      os.environ.setdefault("DEV", "NV")
      sys.path.insert(0, "~/tinygrad-src")
      from tinygrad.tensor import Tensor
      from tinygrad.device import Device
      from tinygrad.runtime.ops_nv import NVProgram
      from tinygrad.device import TinyELF
      dev = Device["NV"]

 2. COMPILE (bench_scan.py:22 pattern; nvcc shim cannot see /tmp — keep paths in $HOME):
      subprocess.run(["~/.local/bin/nvcc", "-arch=sm_86", "-cubin",
                      "-o", "~/tinygrad-metal/<dir>/<k>.cubin", src_path],
                     env={**os.environ, "PATH": "~/.local/bin:/opt/homebrew/bin:"+...,
                          "DOCKER_HOST": "unix://<colima-socket>"}, check=True)
    (run from a shell that already has the §6 prefix, it just works)

 3. LOAD (signature MUST be typed; empty tuple = kernel silently does NOTHING):
      def mk(cubin, name):
        return NVProgram(dev, TinyELF(lib=open(cubin,"rb").read(), name=name,
            target=dev.renderer.target, signature=(("v",0,dtypes.int32,()),)))   # 1 int val
    For buffer-only kernels: signature=tuple() is the known TRAP — pass typed entries
    for every val (TinyELF.iter_sig walks them, HCQ:350-352).

 4. UPLOAD with the _KEEP PATTERN (the buffer-freeing trap — ALWAYS keep Tensor refs):
      _KEEP = []
      def up(a):
        t = Tensor(np.ascontiguousarray(a)).contiguous().realize()
        _KEEP.append(t)
        return t.uop.buf_uop.buffer._bufs["NV"]
    Outputs: poison + keep: o = Tensor.full((N,), -3e38).contiguous().realize();
    _KEEP.append(o); obuf = o.uop.buf_uop.buffer._bufs["NV"]

 5. LAUNCH:
      prog(out_buf, a_buf, b_buf, ..., global_size=(gx,gy,gz), local_size=(tx,ty,tz),
           vals=(sp,), wait=True)
    dev.synchronize() after batches. Raw scratch: dev.allocator.alloc(n, BufferSpec())
    + dev.allocator._copyin(buf, memoryview(bytearray(n))) (a4/gdn_scan_test.py:66-67).

 6. READBACK: tensor-backed -> t.numpy(); raw -> mv = memoryview(bytearray(n));
    dev.allocator._copyout(mv, buf); np.frombuffer(...).

 7. NTID/gridDim: set prog.cbuf_0[0..2] = local_size if the SASS reads blockDim/gridDim
    (a4/gdn_t1_test.py:14). Avoid grid-stride loops regardless.

 8. BENCH: warm 2x, then n x (inner launches + 1 wait), divide (val3.py:78-84).

Known harness traps banked in pv3/T1_FAULT_NOTES.md: OOB ws reads fault the device (not
just wrong numbers); distinct buffers for __restrict__ args (same-buffer aliasing = DCE);
dummy int arg to force non-empty signature; NVRTC (in-model) stricter than nvcc.

==========================================================================================
8. TL;DR ANSWERS
==========================================================================================
 (i)  Dext rebuild: possible from source (all 389 ln + Xcode project present) but
      activation requires either tinygrad Corp's cert (unavailable) or SIP-off + ad-hoc
      NoSIP build (install_nosip.sh). One recovery reboot + Settings approval. Note the
      MAX_SYSMEM 128 cap lives in the APP's server.c, not the dext.
 (ii) 17ms sync: NOT a sleep (refuted, ONV:27-30 + HCQ:300-315 spin). Top suspect =
      GPU-side completion->semaphore-DMA pipeline over the GSP/TB channel (WFI release,
      ONV:174-177) plus queue drain; doorbell is a fire-and-forget socket write
      (ONV:125, SYS:329-332, server.c:243-247). Userspace adds ~nothing post-completion.
 (iii)Graph budget: no fixed userspace cap; top suspect = in-flight command overflow of
      the shared 2MB cmdq ring (silent wrap, ONV:627-629 + memory.py:8-11) or GSP-side
      queue state; test = enlarge cmdq_page. Mappings/server-cap refuted for this wall.
 (iv) gridDim=0 = nvcc SASS reads NTID/NCTAID from cbuf c[0][0..8] which tinygrad leaves
      zero (a4/gdn_t1_test.py:14 workaround); QMD raster fields (ONV:144-146) only set
      the CTA count. cp.async: SM-side; check SASS emission + smem carveout/QMD fields.
 (v)  Rig healthy, idle, nvcc shim OK; env prefix in §6; harness = pv3/val3.py + §7.
