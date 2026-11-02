/*
 * BLAS-backed implementation of pairwise_sqdist, used as
 *   (a) the full-matrix correctness reference, and
 *   (b) the kernel named "blas", so you can see what OpenBLAS achieves on
 *       this exact task (dgemm for Y = XM, dsyrk for the upper triangle of
 *       Y Y^T, then an epilogue pass written by hand).
 * Also provides the square-dgemm ceiling measurement.
 *
 * Built with -DHAVE_BLAS by SConstruct when a CBLAS library was found.  With
 * NOBLAS=1 all of this is replaced by a slow plain-C fallback, so the harness
 * still builds and the sampled check still works; the full-matrix check and the
 * ceiling are then unavailable.
 */
#define _GNU_SOURCE
#include "harness.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef HAVE_BLAS
#  ifdef HAVE_OPENBLAS_CBLAS_H
#    include <openblas/cblas.h>
#  else
#    include <cblas.h>
#  endif
#endif

int blas_available(void)
{
#ifdef HAVE_BLAS
    return 1;
#else
    return 0;
#endif
}

/* Resolve whichever thread-count setter the loaded BLAS exports.  We go through
 * dlsym rather than a direct call so the same binary works against OpenBLAS,
 * FlexiBLAS (RHEL's dispatcher, which hides the OpenBLAS symbol) and MKL.
 * The environment variables are a belt-and-braces measure for backends that
 * read them at load time; the run scripts set them too. */
void blas_set_single_thread(void)
{
    setenv("OPENBLAS_NUM_THREADS", "1", 0);
    setenv("OMP_NUM_THREADS", "1", 0);
    setenv("MKL_NUM_THREADS", "1", 0);
    const char *names[] = { "openblas_set_num_threads", "flexiblas_set_num_threads",
                            "MKL_Set_Num_Threads", "omp_set_num_threads", NULL };
    for (int i = 0; names[i]; i++) {
        void (*f)(int) = (void (*)(int))dlsym(RTLD_DEFAULT, names[i]);
        if (f) f(1);
    }
}

#ifdef HAVE_BLAS

void blas_pairwise_sqdist(int n, int d, int k, const double *X,
                          const double *M, double *D)
{
    double *Y = dalloc((size_t)n * k);
    /* Y = X M */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, k, d, 1.0, X, d, M, k, 0.0, Y, k);
    /* upper(D) = Y Y^T */
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                n, k, 1.0, Y, k, 0.0, D, n);
    /* epilogue: D_ij = G_ii + G_jj - 2 G_ij, clamped at 0 */
    double *rn = dalloc((size_t)n);
    for (int i = 0; i < n; i++) rn[i] = D[(size_t)i * n + i];
    for (int i = 0; i < n; i++) {
        double *row = D + (size_t)i * n;
        double ri = rn[i];
        for (int j = i; j < n; j++) {
            double v = ri + rn[j] - 2.0 * row[j];
            row[j] = v > 0.0 ? v : 0.0;
        }
    }
    free(rn);
    free(Y);
}

double blas_dgemm_ceiling(int N, int reps, double *seconds)
{
    double *A = dalloc((size_t)N * N), *B = dalloc((size_t)N * N), *C = dalloc((size_t)N * N);
    for (size_t q = 0; q < (size_t)N * N; q++) { A[q] = (double)(q % 97) * 0.01; B[q] = (double)(q % 89) * 0.02; }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    double best = 1e300;
    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
        double t = now_sec() - t0;
        if (t < best) best = t;
    }
    free(A); free(B); free(C);
    if (seconds) *seconds = best;
    return 2.0 * (double)N * N * N / best * 1e-9;
}

#else  /* ---------------------------- NOBLAS fallback ---------------------- */

void blas_pairwise_sqdist(int n, int d, int k, const double *X,
                          const double *M, double *D)
{
    /* Plain-C version of the same algorithm.  Slow, but correct. */
    double *Y = dalloc((size_t)n * k), *rn = dalloc((size_t)n);
    compute_rownorms(n, d, k, X, M, Y, rn);
    for (int i = 0; i < n; i++) {
        const double *yi = Y + (size_t)i * k;
        for (int j = i; j < n; j++) {
            const double *yj = Y + (size_t)j * k;
            double g = 0.0;
            for (int c = 0; c < k; c++) g += yi[c] * yj[c];
            double v = rn[i] + rn[j] - 2.0 * g;
            D[(size_t)i * n + j] = v > 0.0 ? v : 0.0;
        }
    }
    free(Y); free(rn);
}

double blas_dgemm_ceiling(int N, int reps, double *seconds)
{
    (void)N; (void)reps;
    if (seconds) *seconds = -1.0;
    return -1.0;
}

#endif
