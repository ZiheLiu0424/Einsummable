#include "harness.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void compute_rownorms(int n, int d, int k, const double *X, const double *M,
                      double *Y, double *rn)
{
    double *row = Y ? NULL : dalloc((size_t)k);
    for (int i = 0; i < n; i++) {
        double *y = Y ? Y + (size_t)i * k : row;
        for (int c = 0; c < k; c++) y[c] = 0.0;
        const double *x = X + (size_t)i * d;
        for (int t = 0; t < d; t++) {
            double xt = x[t];
            const double *m = M + (size_t)t * k;
            for (int c = 0; c < k; c++) y[c] += xt * m[c];
        }
        long double s = 0.0L;
        for (int c = 0; c < k; c++) s += (long double)y[c] * y[c];
        rn[i] = (double)s;
    }
    free(row);
}

static inline double tol_scale(const double *rn, int i, int j)
{
    double s = rn[i] + rn[j];
    return s > 1.0 ? s : 1.0;
}

/* exact (well, long-double-accumulated) direct form for one pair */
static double direct_pair(int d, int k, const double *xi, const double *xj,
                          const double *M, double *diff, long double *v)
{
    for (int t = 0; t < d; t++) diff[t] = xi[t] - xj[t];
    for (int c = 0; c < k; c++) v[c] = 0.0L;
    for (int t = 0; t < d; t++) {
        long double dt = diff[t];
        if (dt == 0.0L) continue;
        const double *m = M + (size_t)t * k;
        for (int c = 0; c < k; c++) v[c] += dt * m[c];
    }
    long double s = 0.0L;
    for (int c = 0; c < k; c++) s += v[c] * v[c];
    return (double)s;
}

static unsigned long long sstate;
static unsigned rnd(void)
{
    sstate ^= sstate << 13; sstate ^= sstate >> 7; sstate ^= sstate << 17;
    return (unsigned)(sstate >> 32);
}

long check_sampled(int n, int d, int k, const double *X, const double *M,
                   const double *D, int fills_full, const double *rn,
                   int npairs, unsigned seed, double tol,
                   double *max_rel_err, long *neg_count)
{
    double *diff = dalloc((size_t)d);
    long double *v = malloc(sizeof(long double) * (size_t)k);
    long fails = 0, negs = 0;
    double worst = 0.0;
    sstate = 0x1234567887654321ULL ^ seed; if (!sstate) sstate = 1;

    /* planted near-duplicate pairs first, then random pairs */
    int nd = num_near_duplicates(n);
    int total = npairs + nd;
    for (int p = 0; p < total; p++) {
        int i, j;
        if (p < nd) { i = p; j = n - 1 - p; if (j <= i) continue; }
        else { i = (int)(rnd() % (unsigned)n); j = (int)(rnd() % (unsigned)n); }
        if (!fills_full && i > j) { int t = i; i = j; j = t; }
        double ref = direct_pair(d, k, X + (size_t)i * d, X + (size_t)j * d, M, diff, v);
        double got = D[(size_t)i * n + j];
        if (got < 0.0) negs++;
        double err = fabs(got - ref) / tol_scale(rn, i, j);
        if (!(err <= tol)) {           /* also catches NaN */
            if (fails < 5)
                fprintf(stderr, "    sampled mismatch at (%d,%d): got %.17g, ref %.17g, rel %.3e\n",
                        i, j, got, ref, err);
            fails++;
        }
        if (err > worst || err != err) worst = err;
    }
    free(diff); free(v);
    *max_rel_err = worst;
    *neg_count = negs;
    return fails;
}

long check_full(int n, const double *D, const double *Dref, int fills_full,
                const double *rn, double tol, double *max_rel_err)
{
    long fails = 0;
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double ref = Dref[(size_t)i * n + j];
            double sc  = tol_scale(rn, i, j);
            double e1  = fabs(D[(size_t)i * n + j] - ref) / sc;
            if (!(e1 <= tol)) {
                if (fails < 5) fprintf(stderr, "    full mismatch at (%d,%d): got %.17g ref %.17g rel %.3e\n",
                                       i, j, D[(size_t)i * n + j], ref, e1);
                fails++;
            }
            if (e1 > worst || e1 != e1) worst = e1;
            if (fills_full && j != i) {
                double e2 = fabs(D[(size_t)j * n + i] - ref) / sc;
                if (!(e2 <= tol)) {
                    if (fails < 5) fprintf(stderr, "    full mismatch at (%d,%d) [lower]: got %.17g ref %.17g rel %.3e\n",
                                           j, i, D[(size_t)j * n + i], ref, e2);
                    fails++;
                }
                if (e2 > worst || e2 != e2) worst = e2;
            }
        }
    }
    *max_rel_err = worst;
    return fails;
}
