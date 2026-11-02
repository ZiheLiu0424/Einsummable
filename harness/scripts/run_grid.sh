#!/usr/bin/env bash
# Run the full benchmark grid for a set of kernels, appending to a CSV.
#
#   scripts/run_grid.sh results.csv k0 k2 k3 blas
#   scripts/run_grid.sh results.csv all
#
# Pins to one physical core on NUMA node 0 (CPU 0 on CLEAR).  Override with
#   CORE=4 NODE=0 scripts/run_grid.sh ...
# Grid: d = 512; n in {1024, 4096, 8192}; k in {32, 128, 512}.
# Override with N_LIST="1024 2048" K_LIST="32" etc.
set -euo pipefail

CSV=${1:?usage: $0 out.csv kernel [kernel ...]}; shift
[ $# -ge 1 ] || { echo "need at least one kernel name (or 'all')" >&2; exit 1; }
KERNELS=$(IFS=,; echo "$*")

CORE=${CORE:-0}
NODE=${NODE:-}
D=${D:-512}
N_LIST=${N_LIST:-"1024 4096 8192"}
K_LIST=${K_LIST:-"32 128 512"}
REPS=${REPS:-5}

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

PIN="taskset -c $CORE"
if [ -n "$NODE" ] && command -v numactl >/dev/null 2>&1; then
    PIN="numactl --cpunodebind=$NODE --membind=$NODE taskset -c $CORE"
fi

HERE=$(cd "$(dirname "$0")/.." && pwd)
BENCH="$HERE/bench"
[ -x "$BENCH" ] || { echo "build first: cd $HERE && scons" >&2; exit 1; }

echo "load average now: $(cut -d' ' -f1-3 /proc/loadavg)   (pinned: $PIN)"
for n in $N_LIST; do
  for k in $K_LIST; do
    echo "=== n=$n d=$D k=$k  kernels=$KERNELS"
    # EXTRA="--no-full-check" (etc.) is passed through unquoted on purpose
    # shellcheck disable=SC2086
    $PIN "$BENCH" --kernel "$KERNELS" --n "$n" --d "$D" --k "$k" --reps "$REPS" --csv "$CSV" ${EXTRA:-}
  done
done
echo "=== ceiling"
$PIN "$BENCH" --ceiling --size 2048 --reps 3 --csv "$CSV"
echo "done -> $CSV"
