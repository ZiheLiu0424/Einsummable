/*
 * Kernel 1: yours.
 *
 * This file is a template.  As shipped it fills D with NaN so that the checker
 * fails loudly; replace the body.  For each further rung of your ladder, copy
 * this file to k2_<idea>.c, k3_<idea>.c, ... and add a line to registry.c.
 * SConstruct picks up every C file in src/kernels/ automatically.
 *
 * Set `fills_full` in registry.c to 0 if you only write D[i][j] for j >= i.
 */
#include "harness.h"
#include <math.h>

void kernel_k1(int n, int d, int k, const double *X, const double *M, double *D)
{
    (void)d; (void)k; (void)X; (void)M;
    for (size_t q = 0; q < (size_t)n * n; q++) D[q] = NAN;
}
