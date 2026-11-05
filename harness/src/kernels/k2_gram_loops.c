/*
 * Kernel 2: the algebraic reformulation, written with plain loops.
 *
 *   D_ij = ||y_i||^2 + ||y_j||^2 - 2 y_i . y_j,   Y = X M   (n x k)
 *
 * Steps and costs (see the Step 0 cost model):
 *   1. Y = X M                 2 n d k flops
 *   2. r_i = ||y_i||^2         2 n k   flops (negligible)
 *   3. G_ij = y_i . y_j, j>=i  n (n+1) k flops   (upper triangle only)
 *   4. D_ij = r_i + r_j - 2 G_ij, clamped at 0; diagonal written as exactly 0.
 *
 * Steps 3 and 4 are done in the same loop, so G is never stored: each D_ij is
 * written once.  No blocking, no intrinsics; whatever SIMD the compiler finds
 * on its own (-O3 -march=native) is all there is.
 *
 * Fills only j >= i (fills_full = 0 in registry.c).
 */
#include "harness.h"
#include <stdlib.h>

void kernel_k2(int n, int d, int k, const double *X, const double *M, double *D)
{
    double *Y = dalloc((size_t)n * k);   /* zeroed, 64-byte aligned */
    double *r = dalloc((size_t)n);

    /* 1. Y = X M.  i-t-c order: the inner loop walks a row of M and a row of
     *    Y contiguously (an axpy), so it vectorizes and streams M row by row. */
    for (int i = 0; i < n; i++) {
        const double *xi = X + (size_t)i * d;
        double *yi = Y + (size_t)i * k;
        for (int t = 0; t < d; t++) {
            const double a = xi[t];
            const double *mt = M + (size_t)t * k;
            for (int c = 0; c < k; c++) yi[c] += a * mt[c];
        }
    }

    /* 2. Row norms. */
    for (int i = 0; i < n; i++) {
        const double *yi = Y + (size_t)i * k;
        double s = 0.0;
        for (int c = 0; c < k; c++) s += yi[c] * yi[c];
        r[i] = s;
    }

    /* 3 + 4. Upper-triangle Gram with the epilogue applied in place.
     *    r_i + r_j - 2 y_i.y_j cancels catastrophically for near-duplicate
     *    points and can come out slightly negative; clamp it. */
    for (int i = 0; i < n; i++) {
        const double *yi = Y + (size_t)i * k;
        double *Di = D + (size_t)i * n;
        Di[i] = 0.0;
        for (int j = i + 1; j < n; j++) {
            const double *yj = Y + (size_t)j * k;
            double g = 0.0;
            for (int c = 0; c < k; c++) g += yi[c] * yj[c];
            double v = r[i] + r[j] - 2.0 * g;
            Di[j] = v > 0.0 ? v : 0.0;
        }
    }

    free(Y);
    free(r);
}
