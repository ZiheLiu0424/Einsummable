#!/usr/bin/env python3
"""
Plot the ladder: useful GFLOP/s per kernel, one panel per k, one line per n,
with the OpenBLAS dgemm ceiling as a dashed reference line.

    scripts/plot.py results.csv ladder.png
    scripts/plot.py results.csv ladder.pdf --order k0,k1,k2,blas

Kernels appear on the x-axis in the order given by --order, or else in the order
they first appear in the CSV.  Rows whose check FAILed are drawn hollow.
If the same (kernel, n, d, k) appears more than once the best (max GFLOP/s) row
is used, so you can re-run noisy points and just append.
"""
import argparse
import csv
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Three categorical hues, fixed order (validated colorblind-safe as a set);
# distinct marker shapes carry identity as well so color is never the only cue.
SERIES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out")
    ap.add_argument("--order", default=None, help="comma-separated kernel order for the x-axis")
    ap.add_argument("--title", default="Pairwise squared distances, single core")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    if not rows:
        sys.exit("empty csv")

    ceiling = None
    data = {}  # (n, d, k) -> {kernel: (gflops, passed)}
    kernel_order = OrderedDict()
    for r in rows:
        if r["kernel"] == "dgemm_ceiling":
            g = float(r["gflops_useful"])
            ceiling = g if ceiling is None else max(ceiling, g)
            continue
        key = (int(r["n"]), int(r["d"]), int(r["k"]))
        g, ok = float(r["gflops_useful"]), r["check"] == "PASS"
        d = data.setdefault(key, {})
        if r["kernel"] not in d or g > d[r["kernel"]][0]:
            d[r["kernel"]] = (g, ok)
        kernel_order.setdefault(r["kernel"], None)

    kernels = a.order.split(",") if a.order else list(kernel_order)
    ks = sorted({key[2] for key in data})
    ns = sorted({key[0] for key in data})
    d_val = sorted({key[1] for key in data})

    fig, axes = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 3.8), sharey=True, squeeze=False)
    axes = axes[0]
    xs = list(range(len(kernels)))
    for ax, k in zip(axes, ks):
        for si, n in enumerate(ns):
            color, marker = SERIES[si % len(SERIES)]
            pts = [data.get((n, dd, k), {}).get(kn) for dd in d_val[:1] for kn in kernels]
            y = [p[0] if p else float("nan") for p in pts]
            ok = [p[1] if p else True for p in pts]
            ax.plot(xs, y, color=color, lw=2, zorder=3, label=f"n = {n}")
            for x, yy, good in zip(xs, y, ok):
                if yy != yy:
                    continue
                ax.plot([x], [yy], marker=marker, ms=7, mfc=(color if good else "white"),
                        mec=color, mew=1.6, ls="none", zorder=4)
        if ceiling is not None:
            ax.axhline(ceiling, color=MUTED, lw=1.2, ls="--", zorder=2)
            ax.text(xs[0] - 0.4, ceiling, f"OpenBLAS dgemm, {ceiling:.0f} GFLOP/s", color=MUTED,
                    fontsize=8, va="bottom", ha="left")
        ax.set_title(f"k = {k}   (d = {d_val[0]})", fontsize=10, color=INK, loc="left")
        ax.set_xticks(xs)
        ax.set_xticklabels(kernels, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK)
    axes[0].set_ylabel("useful GFLOP/s", color=INK)
    axes[0].set_ylim(bottom=0)
    axes[-1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(a.title + "   (hollow marker = failed correctness check)", fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
