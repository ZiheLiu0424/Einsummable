/*
 * The kernel table.  Add one line per rung of your ladder, in order.
 *
 *   { "name", function, fills_full, cost_estimate_or_NULL, "one-line description" }
 *
 * "name" is what you pass to ./bench --kernel and what appears in the CSV and
 * on the plot's x-axis, so keep names short and ordered (k0, k1, k2, ...).
 */
#include "harness.h"

void   kernel_naive(int, int, int, const double *, const double *, double *);
double kernel_naive_cost(int, int, int);
void   kernel_k1(int, int, int, const double *, const double *, double *);
void   kernel_k2(int, int, int, const double *, const double *, double *);
void   kernel_k3(int, int, int, const double *, const double *, double *);
void   kernel_k4(int, int, int, const double *, const double *, double *);
void   kernel_k5(int, int, int, const double *, const double *, double *);

const kernel_t KERNELS[] = {
    { "k0",   kernel_naive,          0, kernel_naive_cost, "naive direct evaluation, scalar, j >= i only" },
    { "k1",   kernel_k1,             1, NULL,              "YOUR FIRST RUNG (template: fills NaN)" },
    { "k2",   kernel_k2,             0, NULL,              "Y=XM, row norms, upper-tri Gram + fused clamp; plain loops" },
    { "k3",   kernel_k3,             0, NULL,              "k2 + AVX2 4x8 register-blocked micro-kernel for XM and Gram" },
    { "k4",   kernel_k4,             0, NULL,              "k3 + cache blocking (MC/KC/NC) and packed A/B panels" },
    { "k5",   kernel_k5,             0, NULL,              "k4 + row-order tile sweep for small k (C writes sequential)" },
    /* add yours here */

    /* The vendor-library implementation, for comparison.  Keep it last. */
    { "blas", blas_pairwise_sqdist,  0, NULL,              "OpenBLAS dgemm + dsyrk + hand epilogue (reference)" },
};

const int NUM_KERNELS = (int)(sizeof(KERNELS) / sizeof(KERNELS[0]));
