// MM P7: acc36 -- THE ON-DEVICE GREEDY ACCEPT (the 140-cross host fold).
// m = longest prefix match amds[s] == drafts[s] for s < K (identical to the
// host loop it replaces: strict equality, s asc, m capped at K). Writes
//   mb[0] = m            (the selc36 slot input -> the IN-GRAPH COMMIT)
//   eb[0] = m            (the EMIT BLOCK, host-mapped: the readback fold)
//   eb[1] = amds[m]      (the boundary token; emitted = drafts[:m]+[amds[m]])
// 1 CTA x 256 thr, thread 0 only -- trivially deterministic. K arrives as a
// runtime scalar (D8 graph K=8, D2 K=2 -- same cubin, no variants).
// NOTE drafts = &idsb[1] (the probe ids are [cur, d0..d(K-1)]) -- passed by
// the caller as an offset view; NO separate drafts buffer.
extern "C" __global__ void __launch_bounds__(256) acc36(
    const int* __restrict__ amds,     // [P] probe argmax per seat (seat s = pred after s-th input)
    const int* __restrict__ drafts,   // [K] = ids[1..K]
    int* __restrict__ mb,             // m slot for selc36
    int* __restrict__ eb,             // [2] emit block (m, amds[m])
    const int K)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int m = 0;
    while (m < K && amds[m] == drafts[m]) ++m;
    mb[0] = m;
    eb[0] = m;
    eb[1] = amds[m];
  }
}
