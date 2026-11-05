/*
 * Kernel 5: k4 with the inner loop order chosen by k (see below).
 *
 * Both products (Y = X M and G = Y Y^T) run through one blocked GEMM:
 *
 *   for jc in steps of NC over columns          (B block: KC x NC, lives in L3)
 *     for pc in steps of KC over the K dim
 *       pack B[pc:pc+KC, jc:jc+NC] into 8-column micro-panels (KC x 8, contiguous)
 *       for ic in steps of MC over rows         (A block: MC x KC, lives in L2)
 *         pack A[ic:ic+MC, pc:pc+KC] into 4-row micro-panels (KC x 4, interleaved)
 *         for jr: one B micro-panel (KC*8 doubles = 16 KiB at KC=256: stays in L1)
 *           for ir: one A micro-panel, streamed from L2
 *             4x8 micro-kernel (same register tile as k3, now on unit-stride packed data)
 *
 * Why packing helps: in k3 the B operand for the Gram was Y^T, whose rows are n
 * doubles apart, so every K step touched a new cache line / TLB page; the A
 * operand was 4 rows k apart.  Packed panels are read strictly sequentially.
 * Packing also removes the separate transpose: the Gram's B panel is packed
 * straight from Y.  Panels are zero-padded, so any n and k work; partial tiles
 * go through a small buffer.
 *
 * Gram specifics: only tiles touching j >= i are computed (block and tile level),
 * the partial sums for k > KC are kept in D between pc passes, and on the last
 * pass the epilogue r_i + r_j - 2g, clamp at 0, is applied in registers.
 * Fills only j >= i.
 */
#include "harness.h"
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#ifndef K5_MC
#define K5_MC 96
#endif
#ifndef K5_KC
#define K5_KC 256
#endif
#ifndef K5_NC
#define K5_NC 2048
#endif
#ifndef K5_ROWORDER_KC
#define K5_ROWORDER_KC 64
#endif
#define MR 4
#define NR 8

/* A row-major (lda), rows [i0, i0+mc), cols [p0, p0+kc) -> MR-row panels, ap[t*MR + r]. */
static void pack_A(int mc, int kc, const double *A, int lda, double *ap)
{
    for (int i = 0; i < mc; i += MR) {
        int mr = mc - i < MR ? mc - i : MR;
        for (int t = 0; t < kc; t++) {
            for (int r = 0; r < mr; r++) ap[r] = A[(size_t)(i + r) * lda + t];
            for (int r = mr; r < MR; r++) ap[r] = 0.0;
            ap += MR;
        }
    }
}

/* B given as row-major K x N (ldb) when trans == 0, or as its transpose
 * (B[t][j] = S[j][t], S row-major with leading dim ldb) when trans == 1.
 * Block rows [0, kc), cols [0, nc) -> NR-column panels, bp[t*NR + c]. */
static void pack_B(int kc, int nc, const double *B, int ldb, int trans, double *bp)
{
    for (int j = 0; j < nc; j += NR) {
        int nr = nc - j < NR ? nc - j : NR;
        if (!trans) {
            for (int t = 0; t < kc; t++) {
                const double *src = B + (size_t)t * ldb + j;
                for (int c = 0; c < nr; c++) bp[c] = src[c];
                for (int c = nr; c < NR; c++) bp[c] = 0.0;
                bp += NR;
            }
        } else {
            for (int c = 0; c < NR; c++) {
                if (c < nr) {
                    const double *src = B + (size_t)(j + c) * ldb;
                    for (int t = 0; t < kc; t++) bp[t * NR + c] = src[t];
                } else {
                    for (int t = 0; t < kc; t++) bp[t * NR + c] = 0.0;
                }
            }
            bp += (size_t)kc * NR;
        }
    }
}

enum { MODE_PLAIN = 0, MODE_GRAM = 1 };

/* 4x8 micro-kernel on packed panels, with the write-back folded in.
 *   first: C holds nothing yet (do not read it).   last: final K pass.
 *   MODE_GRAM && last: write max(0, ri + rj - 2 g) instead of g. */
static inline void micro(int kc, const double *ap, const double *bp,
                         double *C, int ldc, int mr, int nr,
                         int first, int last, int mode,
                         const double *ri, const double *rj)
{
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    for (int t = 0; t < kc; t++) {
        __m256d b0 = _mm256_load_pd(bp), b1 = _mm256_load_pd(bp + 4);
        __m256d a;
        a = _mm256_broadcast_sd(ap + 0); c00 = _mm256_fmadd_pd(a, b0, c00); c01 = _mm256_fmadd_pd(a, b1, c01);
        a = _mm256_broadcast_sd(ap + 1); c10 = _mm256_fmadd_pd(a, b0, c10); c11 = _mm256_fmadd_pd(a, b1, c11);
        a = _mm256_broadcast_sd(ap + 2); c20 = _mm256_fmadd_pd(a, b0, c20); c21 = _mm256_fmadd_pd(a, b1, c21);
        a = _mm256_broadcast_sd(ap + 3); c30 = _mm256_fmadd_pd(a, b0, c30); c31 = _mm256_fmadd_pd(a, b1, c31);
        ap += MR; bp += NR;
    }
    const __m256d zero = _mm256_setzero_pd(), m2 = _mm256_set1_pd(-2.0);
    const int epi = (mode == MODE_GRAM && last);

    /* Fast path: full tile, written straight from registers. */
    if (mr == MR && nr == NR) {
#define WB(row, v0, v1, q) do {                                                     \
            double *p_ = C + (size_t)(row) * ldc;                                   \
            __m256d x0 = v0, x1 = v1;                                               \
            if (!first) { x0 = _mm256_add_pd(x0, _mm256_loadu_pd(p_));              \
                          x1 = _mm256_add_pd(x1, _mm256_loadu_pd(p_ + 4)); }        \
            if (epi) { __m256d rr = _mm256_broadcast_sd(ri + (q));                  \
                x0 = _mm256_max_pd(_mm256_fmadd_pd(m2, x0, _mm256_add_pd(rr, rj0)), zero); \
                x1 = _mm256_max_pd(_mm256_fmadd_pd(m2, x1, _mm256_add_pd(rr, rj1)), zero); } \
            _mm256_storeu_pd(p_, x0); _mm256_storeu_pd(p_ + 4, x1); } while (0)
        __m256d rj0 = zero, rj1 = zero;
        if (epi) { rj0 = _mm256_loadu_pd(rj); rj1 = _mm256_loadu_pd(rj + 4); }
        WB(0, c00, c01, 0); WB(1, c10, c11, 1); WB(2, c20, c21, 2); WB(3, c30, c31, 3);
#undef WB
        return;
    }

    __m256d acc[8] = { c00, c01, c10, c11, c20, c21, c30, c31 };

    const int full = (mr == MR && nr == NR);
    double tmp[MR * NR];
    double *dst = full ? C : tmp;
    const int ld = full ? ldc : NR;
    if (!full && !first)
        for (int r = 0; r < mr; r++)
            for (int c = 0; c < nr; c++) tmp[r * NR + c] = C[(size_t)r * ldc + c];

    __m256d rj0 = _mm256_setzero_pd(), rj1 = _mm256_setzero_pd();
    if (epi) {
        double rjb[NR] = {0};
        for (int c = 0; c < nr; c++) rjb[c] = rj[c];
        rj0 = _mm256_loadu_pd(rjb); rj1 = _mm256_loadu_pd(rjb + 4);
    }
    for (int r = 0; r < MR; r++) {
        if (r >= mr) break;
        double *row = dst + (size_t)r * ld;
        __m256d v0 = acc[2 * r], v1 = acc[2 * r + 1];
        if (!first) { v0 = _mm256_add_pd(v0, _mm256_loadu_pd(row)); v1 = _mm256_add_pd(v1, _mm256_loadu_pd(row + 4)); }
        if (epi) {
            __m256d rr = _mm256_broadcast_sd(ri + r);
            v0 = _mm256_max_pd(_mm256_fmadd_pd(m2, v0, _mm256_add_pd(rr, rj0)), zero);
            v1 = _mm256_max_pd(_mm256_fmadd_pd(m2, v1, _mm256_add_pd(rr, rj1)), zero);
        }
        _mm256_storeu_pd(row, v0);
        _mm256_storeu_pd(row + 4, v1);
    }
    if (!full)
        for (int r = 0; r < mr; r++)
            for (int c = 0; c < nr; c++) C[(size_t)r * ldc + c] = tmp[r * NR + c];
}

/* C (m x N, ldc) = A (m x K, lda) * B, B as described in pack_B.
 * MODE_GRAM: m == N, only tiles with some j >= i, epilogue with r on last pass. */
static void gemm_blocked(int m, int N, int K, const double *A, int lda,
                         const double *B, int ldb, int transB,
                         double *C, int ldc, int mode, const double *r,
                         double *ap, double *bp)
{
    for (int jc = 0; jc < N; jc += K5_NC) {
        int nc = N - jc < K5_NC ? N - jc : K5_NC;
        for (int pc = 0; pc < K; pc += K5_KC) {
            int kc = K - pc < K5_KC ? K - pc : K5_KC;
            int first = (pc == 0), last = (pc + kc == K);
            if (!transB) pack_B(kc, nc, B + (size_t)pc * ldb + jc, ldb, 0, bp);
            else         pack_B(kc, nc, B + (size_t)jc * ldb + pc, ldb, 1, bp);
            int iend = (mode == MODE_GRAM) ? (jc + nc < m ? jc + nc : m) : m;
            for (int ic = 0; ic < iend; ic += K5_MC) {
                int mc = iend - ic < K5_MC ? iend - ic : K5_MC;
                pack_A(mc, kc, A + (size_t)ic * lda + pc, lda, ap);
                if (kc > K5_ROWORDER_KC) {
                    /* Large k: BLIS order.  One B micro-panel (kc*8) stays in L1
                     * while the A panels stream from L2; C is visited in column strips. */
                    for (int jr = 0; jr < nc; jr += NR) {
                        int nr = nc - jr < NR ? nc - jr : NR;
                        int j0 = jc + jr;
                        for (int ir = 0; ir < mc; ir += MR) {
                            int i0 = ic + ir;
                            if (mode == MODE_GRAM && j0 + nr <= i0) break; /* rest is below diagonal */
                            int mr = mc - ir < MR ? mc - ir : MR;
                            micro(kc, ap + (size_t)ir * kc, bp + (size_t)jr * kc,
                                  C + (size_t)i0 * ldc + j0, ldc, mr, nr, first, last, mode,
                                  r ? r + i0 : NULL, r ? r + j0 : NULL);
                        }
                    }
                } else {
                    /* Small k: each tile is only 2*kc*32 flops but writes 256 bytes of C,
                     * so the C write pattern dominates.  Sweep tiles along rows of C
                     * (4 rows, left to right) so stores are sequential within each row;
                     * the A micro-panel (kc*4) stays in L1 and B is reread from L2. */
                    for (int ir = 0; ir < mc; ir += MR) {
                        int i0 = ic + ir;
                        int mr = mc - ir < MR ? mc - ir : MR;
                        int jr0 = 0;
                        if (mode == MODE_GRAM && i0 > jc) jr0 = ((i0 - jc) / NR) * NR;
                        for (int jr = jr0; jr < nc; jr += NR) {
                            int nr = nc - jr < NR ? nc - jr : NR;
                            int j0 = jc + jr;
                            micro(kc, ap + (size_t)ir * kc, bp + (size_t)jr * kc,
                                  C + (size_t)i0 * ldc + j0, ldc, mr, nr, first, last, mode,
                                  r ? r + i0 : NULL, r ? r + j0 : NULL);
                        }
                    }
                }
            }
        }
    }
}

void kernel_k5(int n, int d, int k, const double *X, const double *M, double *D)
{
    double *Y  = dalloc((size_t)n * k);
    double *r  = dalloc((size_t)n);
    double *ap = dalloc((size_t)(K5_MC + MR) * K5_KC);
    double *bp = dalloc((size_t)(K5_NC + NR) * K5_KC);

    /* Y = X M */
    gemm_blocked(n, k, d, X, d, M, k, 0, Y, k, MODE_PLAIN, NULL, ap, bp);

    for (int i = 0; i < n; i++) {
        const double *yi = Y + (size_t)i * k;
        double s = 0.0;
        for (int c = 0; c < k; c++) s += yi[c] * yi[c];
        r[i] = s;
    }

    /* G = Y Y^T (B packed straight from Y), epilogue fused. */
    gemm_blocked(n, n, k, Y, k, Y, k, 1, D, n, MODE_GRAM, r, ap, bp);
    for (int i = 0; i < n; i++) D[(size_t)i * n + i] = 0.0;

    free(Y); free(r); free(ap); free(bp);
}
