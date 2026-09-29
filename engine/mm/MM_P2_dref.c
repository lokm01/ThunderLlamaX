
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#define QK_K 256
#define GGML_RESTRICT __restrict
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "/tmp/ggml-common.h"
static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign << 31;
        else { int e = -1; uint32_t m = man;
            do { m <<= 1; e++; } while (!(m & 0x400));
            bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FF) << 13); }
    } else if (exp == 0x1F) bits = (sign << 31) | 0x7F800000u | (man << 13);
    else bits = (sign << 31) | ((uint32_t)(exp - 15 + 127) << 23) | (man << 13);
    float out; memcpy(&out, &bits, 4); return out;
}
#define GGML_FP16_TO_FP32(x) fp16_to_f32(x)


void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT ql = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

int main(int argc, char **argv) {
    const char *cls = argv[1]; int kdim = atoi(argv[2]); long nrow = atol(argv[3]);
    int bpb = !strcmp(cls,"Q6_K")?210 : !strcmp(cls,"Q8_0")?34 : -1;
    if (bpb < 0) { fprintf(stderr, "bad class\n"); return 2; }
    long rowb = (long)(kdim/256)*bpb;
    if (!strcmp(cls, "Q8_0")) rowb = (long)(kdim/32)*bpb;
    uint8_t * buf = malloc(rowb*nrow);
    FILE * fi = fopen(argv[4], "rb");
    if (fread(buf, 1, rowb*nrow, fi) != (size_t)(rowb*nrow)) { fprintf(stderr,"short read\n"); return 3; }
    fclose(fi);
    float * out = malloc(sizeof(float)*kdim*nrow);
    if      (!strcmp(cls,"Q6_K")) dequantize_row_q6_K((const block_q6_K *)buf, out, (int64_t)kdim*nrow);
    else if (!strcmp(cls,"Q8_0")) dequantize_row_q8_0((const block_q8_0 *)buf, out, (int64_t)kdim*nrow);
    FILE * fo = fopen(argv[5], "wb");
    fwrite(out, sizeof(float), (size_t)kdim*nrow, fo);
    fclose(fo);
    return 0;
}
