#define _GNU_SOURCE
#include "harness.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

double *dalloc(size_t count)
{
    void *p = NULL;
    size_t bytes = count * sizeof(double);
    if (bytes == 0) bytes = 64;
    bytes = (bytes + 63) & ~(size_t)63;
    if (posix_memalign(&p, 64, bytes) != 0 || !p) {
        fprintf(stderr, "out of memory allocating %zu doubles\n", count);
        exit(2);
    }
    memset(p, 0, bytes);
    return (double *)p;
}

/* --- deterministic PRNG (xorshift64*) so every run sees the same data --- */
static unsigned long long rng_state;
static void rng_seed(unsigned seed) { rng_state = 0x9E3779B97F4A7C15ULL ^ ((unsigned long long)seed * 0xBF58476D1CE4E5B9ULL + 1); }
static double rng_uniform(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    unsigned long long r = rng_state * 0x2545F4914F6CDD1DULL;
    return (double)(r >> 11) * (1.0 / 9007199254740992.0);
}
static double rng_normal(void)
{
    double u1 = rng_uniform(), u2 = rng_uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

int num_near_duplicates(int n) { return n / 64 > 0 ? n / 64 : 1; }

/*
 * X: rows drawn from a handful of Gaussian clusters with different scales so
 *    distances span several orders of magnitude.  The last `num_near_duplicates(n)`
 *    rows are near-copies of the first that many rows (perturbed by ~1e-7),
 *    which is what makes the cancellation issue in the expanded formula visible.
 * M: i.i.d. N(0, 1/d) so that y = xM has entries of order 1.
 */
void gen_data(int n, int d, int k, unsigned seed, double *X, double *M)
{
    rng_seed(seed);
    const int ncl = 8;
    double *centers = dalloc((size_t)ncl * d);
    double scale[8];
    for (int c = 0; c < ncl; c++) {
        scale[c] = pow(10.0, -1.0 + 2.0 * c / (ncl - 1));   /* 0.1 .. 10 */
        for (int t = 0; t < d; t++) centers[(size_t)c * d + t] = 5.0 * rng_normal();
    }
    for (int i = 0; i < n; i++) {
        int c = (int)(rng_uniform() * ncl); if (c >= ncl) c = ncl - 1;
        for (int t = 0; t < d; t++)
            X[(size_t)i * d + t] = centers[(size_t)c * d + t] + scale[c] * rng_normal();
    }
    int nd = num_near_duplicates(n);
    for (int t = 0; t < nd && n - 1 - t > t; t++) {
        const double *src = X + (size_t)t * d;
        double *dst = X + (size_t)(n - 1 - t) * d;
        for (int q = 0; q < d; q++) dst[q] = src[q] + 1e-7 * rng_normal();
    }
    double s = 1.0 / sqrt((double)d);
    for (size_t q = 0; q < (size_t)d * k; q++) M[q] = s * rng_normal();
    free(centers);
}

/*
 * "Useful" work: the operation count of the cheaper of the two obvious
 * matrix-product formulations, counting each unordered pair once.
 *   route A:  Y = X M            2 n d k
 *             G = Y Y^T (sym)    n (n+1) k
 *   route B:  A = M M^T          2 d^2 k
 *             Z = X A            2 n d^2
 *             G = Z X^T (sym)    n (n+1) d
 * The O(n^2) epilogue is omitted (it is lower order in k, d).  GFLOP/s is
 * reported against this number for EVERY kernel, so a kernel that does more
 * work than necessary is correctly penalized.
 */
double useful_flops(int n, int d, int k)
{
    double N = n, Dd = d, K = k;
    double a = 2.0 * N * Dd * K + N * (N + 1.0) * K;
    double b = 2.0 * Dd * Dd * K + 2.0 * N * Dd * Dd + N * (N + 1.0) * Dd;
    return a < b ? a : b;
}

double load_average(void)
{
    FILE *f = fopen("/proc/loadavg", "r");
    if (!f) return -1.0;
    double l1 = -1.0;
    if (fscanf(f, "%lf", &l1) != 1) l1 = -1.0;
    fclose(f);
    return l1;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
void cmp_double_sort(double *a, int cnt) { qsort(a, (size_t)cnt, sizeof(double), cmp_d); }
