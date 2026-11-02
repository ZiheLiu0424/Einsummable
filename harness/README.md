# Pairwise squared-distance kernel harness

Starter code for the eight-hour kernel exercise. It gives you the plumbing so
your time goes into the kernel: a naive baseline, a BLAS-backed reference, two
correctness checks, a timing loop, a grid runner, and a plot script.

## Build (CLEAR)

    scons            # auto-detects cblas.h and a BLAS library; prints what it found
    ./bench --list

On CLEAR this links `/usr/include/cblas.h` against OpenBLAS (via
`-l:libopenblas.so.0` or FlexiBLAS). Elsewhere: `scons BLAS=/prefix`,
`scons BLASLIB=-lopenblas`, or `scons NOBLAS=1` (then the full-matrix check
and `--ceiling` are unavailable; the sampled check still works).

Requires SCons (`pip install --user scons` if it is missing) and gcc or clang.

## Run

    ./bench --kernel k0 --n 1024 --d 512 --k 32           # one kernel, one size
    ./bench --kernel k0,k1,blas --n 4096 --k 128           # several kernels
    ./bench --ceiling                                      # OpenBLAS dgemm, 2048^3, single core
    scripts/run_grid.sh results.csv k0 k1 k2 blas          # whole grid -> CSV (pinned to one core)
    scripts/plot.py results.csv ladder.png                 # the figure for your write-up

`./bench --help` lists every option.

## Adding a rung

1. Copy `src/kernels/k1_yours.c` to `src/kernels/k2_<idea>.c`, rename the function.
2. Add one line to `src/kernels/registry.c`. Set `fills_full` to 0 if you only
   write `D[i][j]` for `j >= i`.
3. `scons`. Nothing else to edit; the build globs `src/kernels/*.c`.

Keep every rung. The ladder is the deliverable.

## What the checks do

* **sampled**: `||(x_i - x_j) M||^2` evaluated directly (long-double accumulation)
  for 2000 random pairs plus every planted near-duplicate pair. This is the
  ground truth; it does not depend on BLAS or on any reformulation.
* **full**: every entry you claim to fill, compared against the BLAS implementation
  (`dgemm` + `dsyrk` + epilogue). Catches indexing bugs the sample would miss.
* Tolerance is relative, 1e-9, with scale `max(1, ||y_i||^2 + ||y_j||^2)`.
* **neg** counts sampled entries that came back negative. Negative squared
  distances are a symptom, not a failure; the harness reports them and leaves
  the interpretation to you.
* The data deliberately contains clusters at different scales and a few dozen
  near-duplicate rows (`util.c:gen_data`). Look at it before you trust it.

## How GFLOP/s is computed

Every kernel is scored against the **same** operation count, `useful_flops()`
in `src/util.c`: the cheaper of the two obvious matrix-product formulations,
counting each unordered pair once. A kernel that does more work than that
(computes the full matrix, uses the wrong formulation for the shape) gets a
lower number, which is the point. The naive kernel's O(n^2 d k) count is never
used. If you report a different number in your write-up, say exactly what you
divided by.

`t_min` is the minimum over `--reps` runs (default 5) after one untimed
warm-up; `t_med` is the median. Report `t_min`; mention `t_med` if they differ
by more than a few percent, because that means the machine was busy.

## Benchmarking on a shared machine

* `run_grid.sh` pins to CPU 0 (`CORE=`/`NODE=` to change). On CLEAR, even CPUs
  are socket 0, odd are socket 1, and CPU `i` and `i+20` are hyperthreads of
  the same physical core. A busy sibling halves your throughput.
* Check `uptime` before benchmarking and record the load; the CSV has a column for it.
* The naive kernel is refused above an estimated 4e11 flops (`--force` overrides)
  and timed only once above 5e10. Do not try to run it at n = 8192.

## Files

    SConstruct              build; BLAS auto-detection
    include/harness.h       the interface, documented
    src/main.c              driver, CLI, timing
    src/util.c              timer, allocator, data generator, useful_flops()
    src/check.c             correctness checks
    src/blas_impl.c         BLAS reference + ceiling (and NOBLAS fallback)
    src/kernels/k0_naive.c  baseline (compiled scalar-only)
    src/kernels/k1_yours.c  template
    src/kernels/registry.c  the kernel table
    scripts/run_grid.sh     full grid -> CSV
    scripts/plot.py         CSV -> figure
