/*
 * bench -- driver for the pairwise-squared-distance kernel exercise.
 *
 *   ./bench --list
 *   ./bench [--kernel k0,k2,blas | all] [--n N] [--d D] [--k K] [--reps R]
 *           [--pairs P] [--seed S] [--tol T] [--no-check] [--no-full-check]
 *           [--force] [--csv FILE]
 *   ./bench --ceiling [--size N] [--reps R] [--csv FILE]
 *
 * For every selected kernel: run once (warm-up + correctness), then time R
 * runs and report min and median.  GFLOP/s is useful work / min time; see
 * useful_flops() in util.c for the definition.
 */
#define _GNU_SOURCE
#include "harness.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0)
{
    fprintf(stderr,
        "usage: %s [options]\n"
        "  --list                 list kernels and exit\n"
        "  --kernel NAMES         comma-separated kernel names, or 'all' (default: all)\n"
        "  --n N --d D --k K      problem size (default 1024 512 128)\n"
        "  --reps R               timed repetitions, min and median reported (default 5)\n"
        "  --pairs P              random pairs for the direct-form check (default 2000)\n"
        "  --seed S               data seed (default 1)\n"
        "  --tol T                relative tolerance (default 1e-9)\n"
        "  --no-check             skip all correctness checks\n"
        "  --no-full-check        skip the full-matrix BLAS comparison (keeps the sampled check)\n"
        "  --force                run kernels whose estimated cost is absurd\n"
        "  --csv FILE             append one CSV row per kernel to FILE\n"
        "  --ceiling [--size N]   time square OpenBLAS dgemm (default N=2048) and exit\n",
        argv0);
}

static int arg_int(int argc, char **argv, int *i, const char *name)
{
    if (*i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", name); exit(1); }
    return atoi(argv[++(*i)]);
}
static double arg_dbl(int argc, char **argv, int *i, const char *name)
{
    if (*i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", name); exit(1); }
    return atof(argv[++(*i)]);
}
static const char *arg_str(int argc, char **argv, int *i, const char *name)
{
    if (*i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", name); exit(1); }
    return argv[++(*i)];
}

static FILE *open_csv(const char *path)
{
    if (!path) return NULL;
    FILE *f = fopen(path, "a");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0)
        fprintf(f, "kernel,n,d,k,check,sampled_maxrel,full_maxrel,negatives,t_min,t_med,gflops_useful,loadavg\n");
    return f;
}

static void print_header(void)
{
    printf("%-8s %6s %5s %5s  %-6s %-9s %-9s %4s  %10s %10s  %8s  %5s\n",
           "kernel", "n", "d", "k", "check", "sampled", "full", "neg", "t_min(s)", "t_med(s)", "GFLOP/s", "load");
}

int main(int argc, char **argv)
{
    int n = 1024, d = 512, k = 128, reps = 5, pairs = 2000, size = 2048;
    unsigned seed = 1;
    double tol = 1e-9;
    int do_list = 0, do_ceiling = 0, do_check = 1, do_full = 1, force = 0;
    const char *kernels = "all", *csv_path = NULL;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--list"))          do_list = 1;
        else if (!strcmp(a, "--ceiling"))       do_ceiling = 1;
        else if (!strcmp(a, "--kernel"))        kernels = arg_str(argc, argv, &i, a);
        else if (!strcmp(a, "--n"))             n = arg_int(argc, argv, &i, a);
        else if (!strcmp(a, "--d"))             d = arg_int(argc, argv, &i, a);
        else if (!strcmp(a, "--k"))             k = arg_int(argc, argv, &i, a);
        else if (!strcmp(a, "--reps"))          reps = arg_int(argc, argv, &i, a);
        else if (!strcmp(a, "--pairs"))         pairs = arg_int(argc, argv, &i, a);
        else if (!strcmp(a, "--size"))          size = arg_int(argc, argv, &i, a);
        else if (!strcmp(a, "--seed"))          seed = (unsigned)arg_int(argc, argv, &i, a);
        else if (!strcmp(a, "--tol"))           tol = arg_dbl(argc, argv, &i, a);
        else if (!strcmp(a, "--no-check"))      do_check = 0;
        else if (!strcmp(a, "--no-full-check")) do_full = 0;
        else if (!strcmp(a, "--force"))         force = 1;
        else if (!strcmp(a, "--csv"))           csv_path = arg_str(argc, argv, &i, a);
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a); usage(argv[0]); return 1; }
    }
    if (reps < 1) reps = 1;

    blas_set_single_thread();

    if (do_list) {
        printf("%-8s %-5s %s\n", "name", "fills", "description");
        for (int i = 0; i < NUM_KERNELS; i++)
            printf("%-8s %-5s %s\n", KERNELS[i].name, KERNELS[i].fills_full ? "full" : "upper", KERNELS[i].desc);
        printf("\nBLAS: %s\n", blas_available() ? "linked" : "NOT linked (NOBLAS build: full check and --ceiling unavailable)");
        return 0;
    }

    if (do_ceiling) {
        if (!blas_available()) { fprintf(stderr, "built with NOBLAS=1; no ceiling available\n"); return 1; }
        double secs = 0.0;
        double gf = blas_dgemm_ceiling(size, reps, &secs);
        double load = load_average();
        printf("OpenBLAS dgemm %d x %d x %d, single thread: %.4f s, %.2f GFLOP/s  (load %.2f)\n",
               size, size, size, secs, gf, load);
        FILE *csv = open_csv(csv_path);
        if (csv) { fprintf(csv, "dgemm_ceiling,%d,%d,%d,PASS,0,0,0,%.6f,%.6f,%.3f,%.2f\n",
                           size, size, size, secs, secs, gf, load); fclose(csv); }
        return 0;
    }

    /* ---- data ---- */
    double *X = dalloc((size_t)n * d), *M = dalloc((size_t)d * k);
    gen_data(n, d, k, seed, X, M);
    double *rn = dalloc((size_t)n);
    double *Dref = NULL;
    if (do_check) {
        compute_rownorms(n, d, k, X, M, NULL, rn);
        if (do_full) {
            if (!blas_available()) {
                fprintf(stderr, "note: NOBLAS build, full-matrix check uses the slow plain-C reference\n");
            }
            Dref = dalloc((size_t)n * n);
            blas_pairwise_sqdist(n, d, k, X, M, Dref);
        }
    }
    double *D = dalloc((size_t)n * n);
    double *times = malloc(sizeof(double) * (size_t)reps);
    FILE *csv = open_csv(csv_path);
    double uflops = useful_flops(n, d, k);

    print_header();
    int all = !strcmp(kernels, "all");
    int ran = 0;
    for (int ki = 0; ki < NUM_KERNELS; ki++) {
        const kernel_t *K = &KERNELS[ki];
        if (!all) {
            /* match name against the comma-separated list */
            const char *p = kernels; int hit = 0; size_t L = strlen(K->name);
            while (*p) {
                const char *e = strchr(p, ','); size_t len = e ? (size_t)(e - p) : strlen(p);
                if (len == L && !strncmp(p, K->name, L)) { hit = 1; break; }
                if (!e) break;
                p = e + 1;
            }
            if (!hit) continue;
        }
        ran++;
        if (!blas_available() && K->fn == blas_pairwise_sqdist) {
            printf("%-8s %6d %5d %5d  skipped (NOBLAS build)\n", K->name, n, d, k);
            continue;
        }
        double est = K->est_flops ? K->est_flops(n, d, k) : 0.0;
        int this_reps = reps;
        if (est > 4e11 && !force) {
            printf("%-8s %6d %5d %5d  skipped: estimated %.2g flops (> 4e11). Use --force if you really mean it.\n",
                   K->name, n, d, k, est);
            continue;
        }
        if (est > 5e10 && this_reps > 1) this_reps = 1;

        /* fill with a sentinel so unwritten entries are caught */
        for (size_t q = 0; q < (size_t)n * n; q++) D[q] = -12345.0;

        /* warm-up run, also the one that is checked */
        double t0 = now_sec();
        K->fn(n, d, k, X, M, D);
        double t_first = now_sec() - t0;

        const char *verdict = "n/a";
        double s_err = 0.0, f_err = 0.0; long negs = 0, fails = 0;
        if (do_check) {
            fails += check_sampled(n, d, k, X, M, D, K->fills_full, rn, pairs, seed + 17, tol, &s_err, &negs);
            if (Dref) fails += check_full(n, D, Dref, K->fills_full, rn, tol, &f_err);
            verdict = fails ? "FAIL" : "PASS";
        }

        /* timed runs */
        int nt = 0;
        if (this_reps == 1 && est > 5e10) { times[nt++] = t_first; }
        else {
            for (int r = 0; r < this_reps; r++) {
                t0 = now_sec();
                K->fn(n, d, k, X, M, D);
                times[nt++] = now_sec() - t0;
            }
        }
        cmp_double_sort(times, nt);
        double tmin = times[0], tmed = times[nt / 2];
        double gf = uflops / tmin * 1e-9;
        double load = load_average();

        printf("%-8s %6d %5d %5d  %-6s %-9.2e %-9.2e %4ld  %10.4f %10.4f  %8.2f  %5.2f%s\n",
               K->name, n, d, k, verdict, s_err, f_err, negs, tmin, tmed, gf, load,
               (this_reps == 1 && reps > 1) ? "  (1 timed run: expensive kernel)" : "");
        if (negs) printf("         note: %ld sampled entries were negative (clamp them, or explain why not)\n", negs);
        if (csv) fprintf(csv, "%s,%d,%d,%d,%s,%.3e,%.3e,%ld,%.6f,%.6f,%.3f,%.2f\n",
                         K->name, n, d, k, verdict, s_err, f_err, negs, tmin, tmed, gf, load);
        fflush(stdout);
    }
    if (!ran) { fprintf(stderr, "no kernel matched '%s'; try --list\n", kernels); return 1; }
    if (csv) fclose(csv);
    printf("\nuseful work = %.3g flops (%s); GFLOP/s above are useful work / t_min\n",
           uflops, "min of the two matrix-product formulations, see util.c");
    free(X); free(M); free(rn); free(Dref); free(D); free(times);
    return 0;
}
