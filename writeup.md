# Pairwise Distances Under a Learned Metric — Write-up (DRAFT)

> Draft prepared with Claude. Leo: every claim below must be one you can defend in the meeting.
> Rewrite it in your own words, and fill in sections (e) and (f) yourself.

All numbers are from `harness/results.csv`, produced by `scripts/run_grid.sh` on CLEAR node
**agate** (Xeon E5-2640 v4, AVX2 + FMA, no AVX-512), pinned to CPU 0. All kernels except k1
fill only the upper triangle (j ≥ i). Every rung passes both checks at all 9 grid points.
Figure: `harness/ladder.png`.

## (a) Algorithm and cost model

Naive: for each of n(n+1)/2 pairs form x_i − x_j (d flops), multiply by M (2dk), and take the norm (2k): **≈ n²dk**.
The waste is that x_i M is recomputed n times, because (x_i − x_j)M = y_i − y_j.

- **A: Y = XM, then Gram.** D_ij = r_i + r_j − 2 y_i·y_j with r_i = ‖y_i‖². The cost is 2ndk (XM) + n(n+1)k (upper-triangle Gram), plus an O(n²) epilogue.
- **B: A = MMᵀ, then XAXᵀ.** The cost is 2d²k + 2nd² + n(n+1)d.

The dominant terms are n²k for A and n²d for B, so A wins whenever k ≤ d. That covers the whole grid: 16× at k=32, and only 1–25% at k=512 (where C_B = C_A + 2d³).
A is 250–460× cheaper than naive here. Within A, Gram/XM ≈ n/(2d), so at n=1024 the XM product is half the work and has to be fast too.

Predicted k0 behaviour: the harness refuses runs with n²dk > 4e11, so k0 should run at (1024, all k) and (4096, 32) and be skipped at the other 5 points. Measured: exactly that.
Its time scales as predicted: 3.97 → 16.9 → 59.8 s for k = 32 → 128 → 512, and ×16.1 going from n=1024 to n=4096.

Numerics: A and B both subtract large, nearly equal numbers. For the planted near-duplicates (true D ≈ 1e-13), the rounding error ε(r_i + r_j) is larger than the answer.
With the clamp removed, k2 returns 1–22 negative sampled entries, depending on size. These still PASS, because the tolerance is relative to ‖y_i‖² + ‖y_j‖².
All kernels clamp with max(0, ·) and write the diagonal as exactly 0, so `neg` = 0. The clamp makes the value a legal distance; it does not make it more accurate.

## (b) The ladder

| rung | idea | GFLOP/s range on the grid |
|---|---|---|
| k0 | naive, scalar | 0.01–0.02 |
| k2 | algorithm A, plain loops, epilogue fused in the same loop | 2.1–5.4 |
| k3 | AVX2 4×8 register tile for both XM and Gram; Yᵀ copy | 6.9–24.1 |
| k4 | + MC/KC/NC cache blocking, packed A/B panels | 14.1–34.0 |
| k5 | + row-order tile sweep when kc ≤ 64 | 20.4–33.5 |
| blas | OpenBLAS dgemm + dsyrk + separate epilogue | 13.4–36.6 |

**k2.** I expected a speedup of hundreds over k0, and measured 210–360× in time. It runs at only 2–5 GFLOP/s, because every Gram entry is a k-long reduction into one scalar, so the compiler's SIMD helps little.

**k3.** The micro-kernel holds a 4×8 C tile in 8 YMM accumulators. Each K step does 2 loads of B, 4 broadcasts of A and 8 FMAs. This gave 3–6× over k2.
It collapses to 6.9 at n=8192, k=512. There, the 8-column strip of Yᵀ that one tile walks spans all of Yᵀ: 512 rows, 64 KB apart, 32 MB in total. That is more than the 25 MB L3, so the strip is streamed from DRAM for every row block. At n=4096 Yᵀ is 16 MB, fits in L3, and k3 holds 14.7.

**k4.** Blocking and packing follow the GotoBLAS structure: KC=256, MC=96, NC=2048. A B micro-panel (KC×8 = 16 KiB) stays in L1 and the A block stays in L2.
It nearly doubled k3 at k ≥ 128 (n=8192, k=512: 6.9 → 29.0). Unexpectedly, it was *no better or slightly worse* at k=32 (n=4096: 14.8 → 14.1).
Explanation: the BLIS jr-outer/ir-inner order writes C in 4×8 tiles going *down* a column strip. At small k each tile has little compute (2·32·32 flops) per 256 B written, so the scattered write pattern dominates.

**k5.** When kc ≤ 64, k5 sweeps tiles along rows instead, so stores are sequential; above that it is identical to k4. At k=32 it went from 14.1 to 20.4 (n=4096) and from 15.6 to 20.9 (n=8192). It is unchanged at k ≥ 128, as expected.
At k=32, k5 beats OpenBLAS at every n (25.1/20.4/20.9 vs 20.6/15.3/13.4). OpenBLAS writes G and then makes a second pass over n² doubles for the epilogue; we fuse that pass away.

## (c) Where we are and why

Theoretical peak: 2 FMA ports × 4 doubles × 2 flops = 16 flops/cycle. That is 38.4 GFLOP/s at the 2.4 GHz base and 54.4 at the 3.4 GHz max turbo.
Measured dgemm ceiling: 39.2–40.0 GFLOP/s, slightly below the 41–43 quoted for an idle node. It corresponds to ~2.5 GHz sustained, which suggests the core runs AVX-heavy code well below max turbo. The gap is not all OpenBLAS's.

Best kernel (k5): 70–76% of the dgemm ceiling at k=512, 59–84% at k=128, and 51–63% at k=32.

- **k = 512, compute-bound by the micro-kernel shape.** An FMA on Broadwell has 5-cycle latency and 2 ports, so 10 independent accumulators are needed to keep both ports busy. The 4×8 tile has only 8, which caps it at about 8/10 of peak, i.e. ~32 GFLOP/s at the ceiling's clock. We measure 28–30; OpenBLAS's kernel reaches 35–36.6.
  Caveat: n=1024, k=128 hits 34, slightly above this cap, so the cap is approximate (clock variation, or out-of-order overlap between tiles).
- **k = 128 and k = 32, the output write matters.** The upper triangle of D is 4 MB at n=1024, which fits in L3, but 64–256 MB at n ≥ 4096. Both k5 and OpenBLAS drop by ~25% going from n=1024 to n≥4096 at k=128 (34 → 24–25, 35 → 28). At k=32 the drop is 25 → 20.
  At k=32 each output entry carries 64 flops and costs ~16 B of DRAM traffic (store + write-allocate read), i.e. 4 flops/B. At n=8192, k5 takes 0.116 s; the micro-kernel alone at ~32 GFLOP/s would need ~0.075 s, and the remaining ~0.04 s is consistent with writing ~0.54 GB.
  I have not measured the node's single-core store bandwidth, so this is an argument, not a proof.
- Caveat on data quality: the n=8192, k ∈ {32, 128} rows ran while another user pushed the node load to 8–12. t_min and t_med still agree within 1–8%, so core 0 itself was not heavily contended.

## (d) With eight more hours (in order of expected payoff)

1. **A 12-accumulator micro-kernel (6×8 or 4×12).** It hides FMA latency, and by the argument in (c) should move k ≥ 128 from ~80% toward ~90% of the ceiling.
2. **Non-temporal stores (`_mm256_stream_pd`) for D when k is small.** They remove the write-allocate read, which could cut DRAM traffic at k=32 by up to half. A STREAM/memset measurement first would confirm the bandwidth bound.
3. **Tune MC/KC/NC per k.** In particular a larger KC (=k) at k=512 avoids the second pass over partial sums held in D.
4. **Re-run n=8192 when the node is idle**, and pin to an isolated core with an idle hyperthread sibling.

## (e) How I used AI

TODO (Leo): describe this in your own words. Facts to start from: Claude summarized the spec and derived the cost model (Step 0 doc). It wrote k2–k5 and set up SSH-key access to CLEAR. It ran the grid on CLEAR from my Mac and drafted this write-up.
Say what you checked, what you changed, and what you would do differently.

## (f) Time log

TODO (Leo). Rough record from the project: 11-02 repo and spec read · 11-05 cost model doc · ~0.5 h k2–k5 written and tested in a container · ~2 h CLEAR SSH setup plus the full grid run · write-up.
Add your own reading and study time.
