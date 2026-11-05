/*
 * Kernel 3: k2's algorithm, with an AVX2 register-blocked micro-kernel.
 *
 * Same math as k2 (Y = XM, r_i = ||y_i||^2, D_ij = r_i + r_j - 2 y_i.y_j,
 * upper triangle, clamp at 0).  What changes is how the two matrix products
 * are executed: both go through one 4x8 micro-kernel that computes
 *
 *     C[4][8] += A[4][K] * B[K][8]        (B row-major, leading dim ldb)
 *
 * holding the 4x8 tile of C in 8 YMM registers (4 doubles each).  Per step of
 * the K loop it does 2 vector loads of B, 4 broadcasts of A and 8 FMAs, so the
 * tile is reused from registers instead of memory.  Register budget: 8 acc +
 * 2 B + 1 broadcast = 11 of 16 YMM.
 *
 *   Y = X M : A = X (lda = d), B = M (ldb = k), K = d.
 *   G = Y Y^T: A = Y (lda = k), B = Y^T (k x n, ldb = n), K = k.  Y^T is an
 *              explicit transposed copy (n k doubles, cheap) so that 8
 *              consecutive columns j are contiguous in memory.
 *
 * The epilogue (r_i + r_j - 2g, clamp) is applied to the tile while it is
 * still in registers, so D is written once and G is never stored.
 *
 * No cache blocking or packing yet (that is k4): for large K, the B rows the
 * kernel walks are n doubles apart, and A/B are re-read from L2/L3/DRAM.
 * Edges (n not a multiple of 4/8, k not a multiple of 8) fall back to scalar.
 * Fills only j >= i.
 */
#include "harness.h"
#include <immintrin.h>
#include <stdlib.h>

static inline void mk4x8(int K, const double *A, int lda,
                         const double *B, int ldb, __m256d c[8])
{
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    const double *a0 = A, *a1 = A + lda, *a2 = A + 2 * (size_t)lda, *a3 = A + 3 * (size_t)lda;
    for (int t = 0; t < K; t++) {
        const double *bt = B + (size_t)t * ldb;
        __m256d b0 = _mm256_loadu_pd(bt);
        __m256d b1 = _mm256_loadu_pd(bt + 4);
        __m256d a;
        a = _mm256_broadcast_sd(a0 + t); c00 = _mm256_fmadd_pd(a, b0, c00); c01 = _mm256_fmadd_pd(a, b1, c01);
        a = _mm256_broadcast_sd(a1 + t); c10 = _mm256_fmadd_pd(a, b0, c10); c11 = _mm256_fmadd_pd(a, b1, c11);
        a = _mm256_broadcast_sd(a2 + t); c20 = _mm256_fmadd_pd(a, b0, c20); c21 = _mm256_fmadd_pd(a, b1, c21);
        a = _mm256_broadcast_sd(a3 + t); c30 = _mm256_fmadd_pd(a, b0, c30); c31 = _mm256_fmadd_pd(a, b1, c31);
    }
    c[0] = c00; c[1] = c01; c[2] = c10; c[3] = c11;
    c[4] = c20; c[5] = c21; c[6] = c30; c[7] = c31;
}

/* Scalar C[i][j] = sum_t A[i][t] B[t][j], for edges. */
static inline double dot_strided(int K, const double *a, const double *b, int ldb)
{
    double s = 0.0;
    for (int t = 0; t < K; t++) s += a[t] * b[(size_t)t * ldb];
    return s;
}

void kernel_k3(int n, int d, int k, const double *X, const double *M, double *D)
{
    double *Y  = dalloc((size_t)n * k);
    double *Yt = dalloc((size_t)k * n);
    double *r  = dalloc((size_t)n);
    const int n4 = n & ~3, n8 = n & ~7, k8 = k & ~7;

    /* ---- Y = X M ---- */
    for (int i = 0; i < n4; i += 4) {
        for (int j = 0; j < k8; j += 8) {
            __m256d c[8];
            mk4x8(d, X + (size_t)i * d, d, M + j, k, c);
            for (int q = 0; q < 4; q++) {
                _mm256_storeu_pd(Y + (size_t)(i + q) * k + j,     c[2 * q]);
                _mm256_storeu_pd(Y + (size_t)(i + q) * k + j + 4, c[2 * q + 1]);
            }
        }
        for (int q = 0; q < 4; q++)
            for (int j = k8; j < k; j++)
                Y[(size_t)(i + q) * k + j] = dot_strided(d, X + (size_t)(i + q) * d, M + j, k);
    }
    for (int i = n4; i < n; i++)
        for (int j = 0; j < k; j++)
            Y[(size_t)i * k + j] = dot_strided(d, X + (size_t)i * d, M + j, k);

    /* ---- row norms and transposed copy ---- */
    for (int i = 0; i < n; i++) {
        const double *yi = Y + (size_t)i * k;
        double s = 0.0;
        for (int c = 0; c < k; c++) { s += yi[c] * yi[c]; Yt[(size_t)c * n + i] = yi[c]; }
        r[i] = s;
    }

    /* ---- upper-triangle Gram with fused epilogue ---- */
    const __m256d zero = _mm256_setzero_pd(), m2 = _mm256_set1_pd(-2.0);
    for (int i = 0; i < n4; i += 4) {
        for (int j = i & ~7; j < n8; j += 8) {   /* first tile straddles the diagonal */
            __m256d c[8];
            mk4x8(k, Y + (size_t)i * k, k, Yt + j, n, c);
            __m256d rj0 = _mm256_loadu_pd(r + j), rj1 = _mm256_loadu_pd(r + j + 4);
            for (int q = 0; q < 4; q++) {
                __m256d ri = _mm256_broadcast_sd(r + i + q);
                __m256d v0 = _mm256_fmadd_pd(m2, c[2 * q],     _mm256_add_pd(ri, rj0));
                __m256d v1 = _mm256_fmadd_pd(m2, c[2 * q + 1], _mm256_add_pd(ri, rj1));
                double *dst = D + (size_t)(i + q) * n + j;
                _mm256_storeu_pd(dst,     _mm256_max_pd(v0, zero));
                _mm256_storeu_pd(dst + 4, _mm256_max_pd(v1, zero));
            }
        }
    }
    /* scalar edges: columns j >= n8 for tiled rows, and all rows i >= n4 */
    for (int i = 0; i < n; i++) {
        int j0 = (i < n4) ? (n8 > i ? n8 : i) : i;
        for (int j = j0; j < n; j++) {
            double v = r[i] + r[j] - 2.0 * dot_strided(k, Y + (size_t)i * k, Yt + j, n);
            D[(size_t)i * n + j] = v > 0.0 ? v : 0.0;
        }
    }
    for (int i = 0; i < n; i++) D[(size_t)i * n + i] = 0.0;

    free(Y); free(Yt); free(r);
}
