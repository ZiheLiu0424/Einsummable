/*
 * Kernel 0: the formula, transcribed literally.
 *
 * For every pair (i, j) with j >= i: form the difference x_i - x_j, multiply it
 * by M, take the squared norm.  Cost: about n^2 d k flops (half that with the
 * j >= i restriction), which is why the harness refuses to run it beyond
 * n = 1024 without --force.
 *
 * SConstruct compiles this file with -fno-tree-vectorize so that the baseline
 * really is scalar.  Do not "improve" this kernel; copy it to a new file instead.
 */
#include "harness.h"
#include <stdlib.h>

void kernel_naive(int n, int d, int k, const double *X, const double *M, double *D)
{
    double *diff = malloc(sizeof(double) * (size_t)d);
    double *v    = malloc(sizeof(double) * (size_t)k);
    for (int i = 0; i < n; i++) {
        const double *xi = X + (size_t)i * d;
        for (int j = i; j < n; j++) {
            const double *xj = X + (size_t)j * d;
            for (int t = 0; t < d; t++) diff[t] = xi[t] - xj[t];
            for (int c = 0; c < k; c++) v[c] = 0.0;
            for (int t = 0; t < d; t++)
                for (int c = 0; c < k; c++)
                    v[c] += diff[t] * M[(size_t)t * k + c];
            double s = 0.0;
            for (int c = 0; c < k; c++) s += v[c] * v[c];
            D[(size_t)i * n + j] = s;
        }
    }
    free(diff);
    free(v);
}

double kernel_naive_cost(int n, int d, int k)
{
    return (double)n * n * d * k;   /* 2 flops per MAC, over n(n+1)/2 pairs */
}
