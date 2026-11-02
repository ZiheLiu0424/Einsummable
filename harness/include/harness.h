/*
 * harness.h -- shared declarations for the pairwise-squared-distance kernel harness.
 *
 * The one function you implement (one per kernel file):
 *
 *     void my_kernel(int n, int d, int k,
 *                    const double *X,   // n x d, row-major, leading dimension d
 *                    const double *M,   // d x k, row-major, leading dimension k
 *                    double *D);        // n x n, row-major, leading dimension n
 *
 * On return D[i*n + j] must equal ||x_i M - x_j M||_2^2 for every (i, j) in the
 * region your kernel claims to fill: the full matrix, or just the upper triangle
 * (j >= i) if the kernel's registry entry sets fills_full = 0.  Anything outside
 * the claimed region is ignored by the checker.
 */
#ifndef HARNESS_H
#define HARNESS_H

#include <stddef.h>

typedef void (*kernel_fn)(int n, int d, int k,
                          const double *X, const double *M, double *D);

/* Optional per-kernel cost estimate, in floating-point operations.  Used only
 * to refuse to run a kernel whose expected runtime is absurd (e.g. the naive
 * O(n^2 d k) kernel at n = 8192).  Return a rough number; it does not need to
 * be exact.  NULL means "always run". */
typedef double (*cost_fn)(int n, int d, int k);

typedef struct {
    const char *name;        /* selected on the command line; also the CSV label  */
    kernel_fn   fn;
    int         fills_full;  /* 1: fills all of D.  0: fills only j >= i.           */
    cost_fn     est_flops;   /* see above; may be NULL                               */
    const char *desc;        /* one line, printed by --list                          */
} kernel_t;

extern const kernel_t KERNELS[];
extern const int      NUM_KERNELS;

/* ---- util.c ------------------------------------------------------------ */
double  now_sec(void);
double *dalloc(size_t count);                  /* 64-byte aligned, zeroed  */
void    gen_data(int n, int d, int k, unsigned seed, double *X, double *M);
int     num_near_duplicates(int n);            /* how many rows gen_data duplicates */
double  useful_flops(int n, int d, int k);     /* canonical work estimate; see README */
double  load_average(void);
void    cmp_double_sort(double *a, int cnt);

/* ---- check.c ----------------------------------------------------------- */
/* Y = X M computed with plain loops; rn[i] = ||y_i||^2.  Y may be NULL. */
void compute_rownorms(int n, int d, int k, const double *X, const double *M,
                      double *Y_or_null, double *rn);

/* Direct evaluation of ||(x_i - x_j) M||^2 on a random sample of pairs
 * (always including the near-duplicate pairs planted by gen_data).
 * Returns the number of failing pairs. */
long check_sampled(int n, int d, int k, const double *X, const double *M,
                   const double *D, int fills_full, const double *rn,
                   int npairs, unsigned seed, double tol,
                   double *max_rel_err, long *neg_count);

/* Compare every claimed entry of D against Dref (which holds the upper
 * triangle).  Returns the number of failing entries. */
long check_full(int n, const double *D, const double *Dref, int fills_full,
                const double *rn, double tol, double *max_rel_err);

/* ---- blas_impl.c ------------------------------------------------------- */
int    blas_available(void);
void   blas_set_single_thread(void);
/* OpenBLAS-based implementation (dgemm + dsyrk + fused-by-hand epilogue).
 * Fills the upper triangle.  Used as the full-matrix reference and listed as
 * a kernel so you can see what a vendor library achieves on this exact task. */
void   blas_pairwise_sqdist(int n, int d, int k, const double *X,
                            const double *M, double *D);
/* Square dgemm N x N x N, min over reps.  Returns GFLOP/s, or -1 if no BLAS. */
double blas_dgemm_ceiling(int N, int reps, double *seconds);

#endif /* HARNESS_H */
