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

const kernel_t KERNELS[] = {
    { "k0",   kernel_naive,          0, kernel_naive_cost, "naive direct evaluation, scalar, j >= i only" },
    { "k1",   kernel_k1,             1, NULL,              "YOUR FIRST RUNG (template: fills NaN)" },
    /* add yours here */

    /* The vendor-library implementation, for comparison.  Keep it last. */
    { "blas", blas_pairwise_sqdist,  0, NULL,              "OpenBLAS dgemm + dsyrk + hand epilogue (reference)" },
};

const int NUM_KERNELS = (int)(sizeof(KERNELS) / sizeof(KERNELS[0]));
