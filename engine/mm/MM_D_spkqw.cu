// MM SESSION D (L4): spkqw -- THE ROW-GROUPED WIDE PF ATTENTION (the 96k
// lever). Replaces the per-(seat,head,split) CTA structure of spkq256s
// (grid (16P,S): every CTA streams its whole split slice from L2 as byte
// loads -- the 1546ms/76%@96k term) with ROW GROUPS: grid (P/RW, 2*S);
// blockIdx.x = row-group rg; blockIdx.y = j_*S + s.
//   RW=4: CTA = 32 rows (4 seats x 8 q-heads of kv-group j_); warp w owns
//         rows 4w..4w+3 = seat rg*4+(w>>1), heads j_*8+(w&1)*4+hl.
//   RW=8: CTA = 64 rows (8 seats); warp w owns rows 8w..8w+7 = seat
//         rg*8+w, heads j_*8+hl (8 independent online chains per warp --
//         the ILP arm; qq staging lives in dyn smem UNION'd with the K/V
//         tiles, dead once loaded to regs).
// The causal boundary L stays warp-uniform either way. K/V tiles staged to
// DYNAMIC smem once per tile (cooperative uint4) and consumed by ALL warps
// (the spk_g4hm law) -> the KV stream drops from once-per-seat to
// once-per-RW-seats. Per position ONE LDS.64 per tensor + ONE shared
// dequant (identical fp op order to stock: (float)byte*scale first, then
// * qq / * e), reused across the warp's rows.
// PER-ROW PER-SPLIT: ONE online-softmax chain over the split's positions
// ASC (stock = 8 strided warp chains merged by spkc) -> the single real
// partial goes to slot (t*16+h)*S+s; spkc256 runs with NP=S (reads exactly
// the S real slots per row -- no empty slots, 4x less partial traffic).
// Tier-2 numerics (reassociation WITHIN splits; S pinned -> the split
// BOUNDARIES unchanged vs stock). The kernel-order anchor:
// engine0/mm/mm_l4_poc.py spkqw_ref (the spkq_h_split_ref pattern with the
// NP=S combine). Preamble (q norm + rope) VERBATIM stock spkq256s.
// Dynamic smem: max(RW*8KB qq staging, TILE*528 K/V tiles) -- the qq/tile
// UNION (qq dead in smem once in regs). CTXS compile-time rung; S the ONLY
// runtime val. -fmad=false. Det: fixed fp order; tiles byte-identical.
#ifndef CTXS
#define CTXS 16384
#endif
#ifndef RW
#define RW 4
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define KSYM CAT(CAT(CAT(spkqw, RW), _), CTXS)   // spkqw{4,8}_{CTXS}
#define EPS_F 9.999999974752427e-07f
#define SCA_F 0.0625f
#define TILE 64
#define NROW (RW * 8)                        // CTA rows = seats*8 heads
#define QQ_B (NROW * 256 * 4)                // qq staging bytes
#define KV_B (TILE * (256 + 8 + 256 + 8))    // ksm+ksc+vsm+vsc
#define DYN_B (QQ_B > KV_B ? QQ_B : KV_B)    // the union

extern "C" __global__ void __launch_bounds__(256) KSYM(
    const float* __restrict__ qg,        // [P][8192] (q | gate interleaved/head)
    float* __restrict__ pbase,           // [P*16][S][258] partials (NP=S layout)
    const unsigned long long* __restrict__ ptbl,
    const int S)
{
  const float* __restrict__ qw  = (const float*)(size_t)ptbl[1];
  const float* __restrict__ cs  = (const float*)(size_t)ptbl[2];
  const float* __restrict__ sn  = (const float*)(size_t)ptbl[3];
  const signed char* __restrict__ Kq = (const signed char*)(size_t)ptbl[4];
  const float* __restrict__ Ks = (const float*)(size_t)ptbl[5];
  const signed char* __restrict__ Vq = (const signed char*)(size_t)ptbl[6];
  const float* __restrict__ Vs = (const float*)(size_t)ptbl[7];
  const int* __restrict__ posb = (const int*)(size_t)ptbl[8];

  const int rg = blockIdx.x;             // seats [rg*RW, rg*RW+RW)
  const int j_ = blockIdx.y / S;         // kv-group (0/1)
  const int s  = blockIdx.y - j_ * S;    // split
  const int d  = threadIdx.x;
  const int warp = d >> 5, lane = d & 31;
  const int pos0 = posb[0];

  // ---- per-row preamble: q norm + rope, VERBATIM stock spkq256s ----
  __shared__ float wpt[8];
  __shared__ float rstd_s;
  extern __shared__ char dyn[];          // the qq/tile union (host QMD patch)
  float* qq_s = (float*)dyn;             // [NROW][256], dead once in regs
  #pragma unroll 1
  for (int i = 0; i < NROW; ++i) {
    const int t   = (rg * NROW + i) >> 3;
    const int h   = j_ * 8 + ((rg * NROW + i) & 7);
    const int pos = pos0 + t;
    const float qraw = qg[t * 8192 + h * 512 + d];
    float p = qraw * qraw;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
    if (lane == 0) wpt[warp] = p;
    __syncthreads();
    if (d == 0) {
      float sv = wpt[0];
      #pragma unroll
      for (int k = 1; k < 8; ++k) sv += wpt[k];
      rstd_s = 1.0f / sqrtf(sv * (1.0f / 256.0f) + EPS_F);
    }
    __syncthreads();
    const float rstd = rstd_s;
    float q = (qraw * rstd) * qw[d];
    if (d < 64) {
      const int dp = (d < 32) ? d + 32 : d - 32;
      const float qp = (qg[t * 8192 + h * 512 + dp] * rstd) * qw[dp];
      const int ci = (d < 32) ? d : d - 32;
      const float c = cs[pos * 32 + ci];
      const float snv = sn[pos * 32 + ci];
      q = (d < 32) ? (q * c + (-qp) * snv) : (q * c + qp * snv);
    }
    qq_s[i * 256 + d] = q;               // no reuse hazard: disjoint words/row
  }
  __syncthreads();

  // ---- the warp's rows: qq to regs, then the tiles own the union ----
#if RW == 8
  const int t_w = rg * 8 + warp;
  const int hb  = 0;
#else
  const int t_w = rg * 4 + (warp >> 1);
  const int hb  = (warp & 1) << 2;
#endif
  float qq[RW][8];
  #pragma unroll
  for (int hl = 0; hl < RW; ++hl)
    #pragma unroll
    for (int j = 0; j < 8; ++j)
      qq[hl][j] = qq_s[(warp * RW + hl) * 256 + lane * 8 + j];
  __syncthreads();                       // every warp done reading qq_s

  // ---- union split range over the CTA's seats ----
  const int Lmin = pos0 + rg * RW + 1;
  const int Lmax = pos0 + rg * RW + RW;  // P % RW == 0 -> always RW seats
  const int ubeg = (int)((long long)s * Lmin / S);
  const int uend = (int)((long long)(s + 1) * Lmax / S);
  const int wbeg = (int)((long long)s * (pos0 + t_w + 1) / S);
  const int wend = (int)((long long)(s + 1) * (pos0 + t_w + 1) / S);

  float m[RW], z[RW], acc[RW][8];
  #pragma unroll
  for (int hl = 0; hl < RW; ++hl) {
    m[hl] = -3.402823466e38f; z[hl] = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[hl][j] = 0.f;
  }

  // layout: ksm [0,TILE*256) ksc [+TILE*8) vsm [+TILE*256) vsc [+TILE*8)
  signed char* ksm = (signed char*)dyn;
  float* ksc = (float*)(dyn + TILE * 256);
  signed char* vsm = ksm + TILE * 256 + TILE * 8;
  float* vsc = (float*)(dyn + 2 * TILE * 256 + TILE * 8);
  const int kb = lane >> 4;

  for (int tile = ubeg; tile < uend; tile += TILE) {
    const int n = (uend - tile < TILE) ? uend - tile : TILE;
    for (int i = d; i < n * 16; i += 256) {
      const int pl = i >> 4, b16 = (i & 15) << 4;
      *(uint4*)(ksm + pl * 256 + b16) =
        *(const uint4*)(Kq + ((size_t)j_ * CTXS + tile + pl) * 256 + b16);
      *(uint4*)(vsm + pl * 256 + b16) =
        *(const uint4*)(Vq + ((size_t)j_ * CTXS + tile + pl) * 256 + b16);
    }
    for (int i = d; i < n; i += 256) {
      *(float2*)(ksc + i * 2) = *(const float2*)(Ks + ((size_t)j_ * CTXS + tile + i) * 2);
      *(float2*)(vsc + i * 2) = *(const float2*)(Vs + ((size_t)j_ * CTXS + tile + i) * 2);
    }
    __syncthreads();
    int p0 = tile > wbeg ? tile : wbeg;
    int p1 = (tile + n) < wend ? (tile + n) : wend;
    for (int pp = p0; pp < p1; ++pp) {
      const int pl = pp - tile;
      // ONE 8B LDS.64 per tensor per position + ONE shared dequant (the
      // SAME fp op order as stock: (float)byte * scale, THEN * qq / * e)
      const uint2 k8 = *(const uint2*)(ksm + pl * 256 + lane * 8);
      const uint2 v8 = *(const uint2*)(vsm + pl * 256 + lane * 8);
      const float ks = ksc[pl * 2 + kb];
      const float vs = vsc[pl * 2 + kb];
      float kq_[8], vq_[8];
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        kq_[j]     = (float)(signed char)((k8.x >> (8 * j)) & 0xFF) * ks;
        kq_[4 + j] = (float)(signed char)((k8.y >> (8 * j)) & 0xFF) * ks;
        vq_[j]     = (float)(signed char)((v8.x >> (8 * j)) & 0xFF) * vs;
        vq_[4 + j] = (float)(signed char)((v8.y >> (8 * j)) & 0xFF) * vs;
      }
      #pragma unroll
      for (int hl = 0; hl < RW; ++hl) {
        float a = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
          a += qq[hl][j] * kq_[j];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        const float sc = a * SCA_F;
        const float mn = (sc > m[hl]) ? sc : m[hl];
        const float r = expf(m[hl] - mn);
        const float e = expf(sc - mn);
        z[hl] = z[hl] * r + e;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
          acc[hl][j] = acc[hl][j] * r + e * vq_[j];
        m[hl] = mn;
      }
    }
    __syncthreads();                     // tile reuse guard
  }

  // ---- write the S real partials of this warp's rows (NP=S layout) ----
  #pragma unroll
  for (int hl = 0; hl < RW; ++hl) {
    const int h = j_ * 8 + hb + hl;
    float* slot = pbase + ((size_t)(t_w * 16 + h) * S + s) * 258;
    if (lane == 0) { slot[0] = m[hl]; slot[1] = z[hl]; }
    #pragma unroll
    for (int j = 0; j < 8; ++j) slot[2 + lane * 8 + j] = acc[hl][j];
  }
}
