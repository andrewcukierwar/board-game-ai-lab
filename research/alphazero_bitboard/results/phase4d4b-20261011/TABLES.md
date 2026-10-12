# Phase 4D.4B complete-search measurements

Median latency = median of eight checkpoint medians, ten trials per checkpoint.
P95/min/max and CPU distributions describe the fixed 80-observation checkpoint mixture.
Complete uninstrumented search includes root conversion/expansion and result construction.

| Position | Sims | Engine median ms | Bitboard median ms | Speedup | Engine p95 ms | Bitboard p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| empty | 256 | 114.19 | 59.26 | 1.927× | 120.17 | 62.03 |
| empty | 512 | 227.98 | 119.20 | 1.913× | 236.10 | 123.20 |
| early_opening | 256 | 113.29 | 58.81 | 1.927× | 116.27 | 60.13 |
| early_opening | 512 | 226.84 | 118.21 | 1.919× | 232.60 | 120.88 |
| quiet_midgame | 256 | 100.72 | 57.22 | 1.760× | 109.35 | 60.09 |
| quiet_midgame | 512 | 199.04 | 112.18 | 1.774× | 215.63 | 115.00 |
| tactical_midgame | 256 | 100.71 | 52.94 | 1.902× | 105.39 | 56.67 |
| tactical_midgame | 512 | 202.09 | 107.32 | 1.883× | 211.69 | 114.34 |
| late_game | 256 | 31.78 | 13.69 | 2.322× | 38.20 | 18.41 |
| late_game | 512 | 48.88 | 18.76 | 2.606× | 53.37 | 21.51 |
| immediate_win | 256 | 18.03 | 5.34 | 3.378× | 23.13 | 8.29 |
| immediate_win | 512 | 31.23 | 7.88 | 3.962× | 37.88 | 11.55 |
| mandatory_block | 256 | 85.78 | 42.41 | 2.023× | 100.09 | 50.11 |
| mandatory_block | 512 | 179.36 | 89.19 | 2.011× | 204.53 | 103.43 |

| Position | Sims | Backend | Wall min–max ms | CPU median ms | CPU p95 ms | Sims/s median | NN calls mean | Terminal leaves mean |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| empty | 256 | engine | 111.91–160.10 | 114.39 | 119.87 | 2237 | 257.0 | 0.0 |
| empty | 256 | bitboard | 56.75–85.15 | 59.46 | 61.81 | 4305 | 257.0 | 0.0 |
| empty | 512 | engine | 223.62–246.11 | 228.18 | 235.94 | 2244 | 513.0 | 0.0 |
| empty | 512 | bitboard | 113.89–133.26 | 119.51 | 123.16 | 4284 | 513.0 | 0.0 |
| early_opening | 256 | engine | 111.25–118.15 | 113.41 | 115.39 | 2257 | 255.9 | 1.1 |
| early_opening | 256 | bitboard | 56.73–64.31 | 58.91 | 60.09 | 4344 | 255.9 | 1.1 |
| early_opening | 512 | engine | 217.98–251.30 | 227.05 | 232.39 | 2255 | 508.5 | 4.5 |
| early_opening | 512 | bitboard | 112.36–121.57 | 117.79 | 120.86 | 4345 | 508.5 | 4.5 |
| quiet_midgame | 256 | engine | 96.92–126.66 | 101.10 | 108.97 | 2531 | 253.8 | 3.2 |
| quiet_midgame | 256 | bitboard | 54.08–84.48 | 57.03 | 59.32 | 4488 | 253.8 | 3.2 |
| quiet_midgame | 512 | engine | 189.46–229.26 | 199.89 | 215.43 | 2561 | 498.9 | 14.1 |
| quiet_midgame | 512 | bitboard | 106.89–130.37 | 112.17 | 114.95 | 4564 | 498.9 | 14.1 |
| tactical_midgame | 256 | engine | 82.83–130.53 | 101.38 | 105.26 | 2523 | 229.1 | 27.9 |
| tactical_midgame | 256 | bitboard | 43.70–249.52 | 53.10 | 56.63 | 4821 | 229.1 | 27.9 |
| tactical_midgame | 512 | engine | 175.19–246.26 | 203.24 | 210.48 | 2519 | 460.9 | 52.1 |
| tactical_midgame | 512 | bitboard | 94.38–124.08 | 107.31 | 114.25 | 4771 | 460.9 | 52.1 |
| late_game | 256 | engine | 22.73–39.06 | 31.91 | 38.20 | 8003 | 55.8 | 201.2 |
| late_game | 256 | bitboard | 7.28–18.61 | 13.78 | 18.41 | 18541 | 55.8 | 201.2 |
| late_game | 512 | engine | 38.19–54.87 | 48.91 | 53.35 | 10459 | 66.6 | 446.4 |
| late_game | 512 | bitboard | 10.40–21.56 | 18.82 | 21.49 | 27215 | 66.6 | 446.4 |
| immediate_win | 256 | engine | 15.70–23.35 | 18.38 | 23.13 | 13929 | 22.5 | 234.5 |
| immediate_win | 256 | bitboard | 4.10–8.37 | 5.44 | 8.29 | 47079 | 22.5 | 234.5 |
| immediate_win | 512 | engine | 28.63–38.39 | 31.16 | 37.86 | 16434 | 30.8 | 482.2 |
| immediate_win | 512 | bitboard | 6.54–11.82 | 7.97 | 11.55 | 64285 | 30.8 | 482.2 |
| mandatory_block | 256 | engine | 68.36–247.72 | 85.71 | 100.02 | 2987 | 185.5 | 71.5 |
| mandatory_block | 256 | bitboard | 33.15–50.52 | 42.64 | 50.09 | 6004 | 185.5 | 71.5 |
| mandatory_block | 512 | engine | 163.11–209.51 | 178.53 | 204.41 | 2869 | 389.0 | 124.0 |
| mandatory_block | 512 | bitboard | 80.21–132.46 | 89.34 | 103.41 | 5734 | 389.0 | 124.0 |

The original 42/g10 tactical-midgame 256 cell retains its per-checkpoint p95 flag.
The separate 40-pair follow-up reports 1.921× median speedup and p95 ratio 0.528.
No trial is excluded and the original primary summary remains unchanged.

## Separate Python allocation probes

| Position | Sims | Engine median peak bytes | Bitboard median peak bytes | Median paired ratio |
|---|---:|---:|---:|---:|
| empty | 256 | 2941138 | 879921 | 0.300 |
| empty | 512 | 5721144 | 1730668 | 0.303 |
| early_opening | 256 | 2941523 | 879204 | 0.299 |
| early_opening | 512 | 5691198 | 1723318 | 0.303 |
| quiet_midgame | 256 | 1604452 | 498114 | 0.307 |
| quiet_midgame | 512 | 2953013 | 930100 | 0.315 |
| tactical_midgame | 256 | 2209847 | 666208 | 0.302 |
| tactical_midgame | 512 | 4160060 | 1274370 | 0.306 |
| late_game | 256 | 187324 | 69346 | 0.385 |
| late_game | 512 | 227666 | 86732 | 0.381 |
| immediate_win | 256 | 242026 | 72888 | 0.300 |
| immediate_win | 512 | 348469 | 104011 | 0.300 |
| mandatory_block | 256 | 2118996 | 630728 | 0.297 |
| mandatory_block | 512 | 4266997 | 1286488 | 0.301 |

Native tensor allocations are largely outside tracemalloc. RSS is distinct.

| Fresh process: 42/g10 quiet 512 | Traced peak bytes | Lifetime peak RSS MiB |
|---|---:|---:|
| engine | 2980121 | 271.31 |
| bitboard | 950324 | 267.36 |

RSS includes imports, model loading, warm-up and tracing; one fresh process per backend.
Main-run process high-water RSS ranges are retained per allocation probe in memory.json.

## Complete frozen-network self-play

| Backend | Games | Plies | Total wall s | Total CPU s | Median game s | Games/hour | Positions/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| engine | 4 | 119 | 11.762 | 11.754 | 3.065 | 1224 | 10.12 |
| bitboard | 4 | 119 | 5.617 | 5.617 | 1.489 | 2564 | 21.19 |

## Exclusive profile attribution

| Category | Engine % | Bitboard % |
|---|---:|---:|
| backup | 0.23 | 0.35 |
| encoding_tensor | 4.26 | 2.61 |
| expansion | 1.65 | 1.89 |
| inference_pipeline | 8.12 | 14.21 |
| legal_moves | 1.61 | 0.65 |
| neural_forward | 36.40 | 68.71 |
| other | 1.08 | 1.91 |
| selection | 3.13 | 5.64 |
| state_copy | 16.00 | 0.00 |
| state_transition | 1.51 | 3.59 |
| terminal | 26.00 | 0.44 |

Exclusive categories plus other are additive. Neural forward includes contract checks.
Inclusive timings overlap; cProfile corroboration and counts are in paired/*cprofile.json and profile.json.
