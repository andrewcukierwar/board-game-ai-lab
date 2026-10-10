# Phase 3B.3: iterative deepening and cross-depth move-only hints

October 9, 2026. Existing laptop checkout, branch `search/negamax-v2`; immutable baseline [2e79b15ecdb1967345a1e66593201f9803c89757](https://github.com/andrewcukierwar/board-game-ai-lab/commit/2e79b15ecdb1967345a1e66593201f9803c89757).

**Reject B/hints, C/iterative and D/combined as production defaults. Keep A/direct unchanged.** Cross-depth hints reduce work on several expensive trees, including the prior tail, but do not meet the predeclared distribution, regression and memory requirements. Iterative deepening without reuse adds work and gives identical final-iteration counters. The experiments remain research scripts; no production module imports them.

## Design and implementation

[DESIGN.md](DESIGN.md) and [manifest.json](manifest.json) were frozen before performance measurement. A runs the byte-identical baseline source loaded with `git show`. B searches target−2, then target; C searches 2/4/…/target without cross-depth hints; D uses that schedule with hints from its immediately preceding completed iteration. Odd targets use 1/3/…/target; targets ≤2 have one iteration. Two-ply increments align horizon parity and limit preparatory overhead. No schedules were tuned after observing results.

B/D export a new dictionary of **full X board, full O board and mover → integer column only** from the previous TT. The packed identity is `X | (O << 49) | (mover << 98)`. Export decodes only the move field and discards every score, bound and horizon. Both exact and bound entries may suggest a searched move. The next score TT is fresh and remains keyed by full identity plus remaining depth; its EXACT/LOWER/UPPER interpretation and original-window classification are unchanged. No previous horizon can return a value or close a window. A same-depth TT move has priority over a cross-depth suggestion; immediate wins remain first. Strict integer/range/non-full-column checks reject invalid hints. Replacing snapshots avoids accumulating older horizons. Export and release costs are included in decision timings.

All legal root moves are searched with full windows in historical `(3,2,4,1,5,0,6)` order at every iteration, preserving exact vectors and center-first root ties. No anytime returns, aspiration, new pruning, skipped roots, time limits or changed final evaluations were introduced. Terminal-before-leaf ±(1,000,000 + remaining depth), the exact incremental 69-window heuristic, play/undo and exception restoration retain the baseline bodies. Generated source snapshots for all normal and diagnostic variants are included here.

Victor inspection: `victor/exact.py` uses terminal-only WDL, game-rule forced-move pruning, mirror identities and bounds; the native solver also uses bounded proof intervals. Its bitboard geometry informs the existing immediate-win ordering. WDL score reuse, proof pruning and budget interruptions are incompatible with this heuristic horizon contract and were not transferred.

## Distribution, timing and resource preflight

All 24 Phase 3B.2 histories are reused unchanged: eleven established Phase 3A boards, eight Phase 3B.1 seeded boards, four Phase 3B.2 additions and the separately labeled POST HOC `[1,4,6,0,6]` tail. Four new legal nonterminal histories use seed 20261011 at 6/10/18/26 plies, uniform legal moves, terminal/duplicate rejection only. No position selection uses outcomes. There are 112 depth/position conditions, 108 non-tail conditions. Broad-tree classification uses A alone: depth 8/10, ≥2,000 nodes and ≥35% heuristic leaves. The inherited `double-threat-loss` fixture permits an own immediate win; a separate test history `[1,0,3,0,5,0,1,6,3,6,5,6]` verifies an actual forced root loss.

Initial preflight: 62.3s, largest untraced decision 0.885s, largest traced peak 13.33 MiB. Conservative 2× main projection 5247.5s was within the declared 7,200s main allowance. Actual main runtime 415.8s. Preflight observations are excluded from final estimates. The declared overall compute allowance is 8,400s (7,200 main + 600 diagnostics/audit + 600 validation/report).

Each condition has one discarded warm-up and seven serial timed decisions per variant, rotating order by repetition and condition. Wall and process CPU cover complete choose_move, state initialization, every intermediate/root search, hint export, table construction and release. Game creation, variant loading and diagnostics are outside timing. Tables receive 64 constructor warmups to stabilize CPython shared instance dictionaries. Separate counted and two traced decisions measure leaves/ordering and memory. The final table alone is retained for memory sizing; no artificial accumulation of intermediate tables inflates peak.

Runtime: Model Name: MacBook Pro; Model Identifier: Mac17,2; Chip: Apple M5; Total Number of Cores: 10 (4 Performance and 6 Efficiency); Memory: 32 GB; 3.11.17 (main, Oct  1 2026, 00:48:48) [Clang 21.0.0 (clang-2100.3.34.2)]; macOS-26.6.2-arm64-arm-64bit; initial load averages [3.2900390625, 3.25341796875, 3.22314453125]. Other host activity was not stopped. These are finite laptop engineering comparisons, not randomized population estimates, game-strength results, public endpoint throughput or worst-case guarantees.

## Four-way complete-decision comparison

| Variant | All-condition geometric speedup | Broad geometric speedup | Sum median wall / A | Sum median CPU / A | Decision |
| --- | --- | --- | --- | --- | --- |
| A/direct | 1.000 | 1.000 | 1.000 | 1.000 | reference |
| hints | 0.922× | 1.131× | 0.853 | 0.853 | reject |
| iterative | 0.737× | 0.783× | 1.287 | 1.287 | reject |
| combined | 0.861× | 1.124× | 0.860 | 0.860 | reject |

Speedup is A/new: above 1 is faster. Ratios new/A: below 1 is cheaper. Tail and D12 are excluded from acceptance aggregates. Summed medians give each declared condition one complete decision; geometric means give each condition equal log weight.

| Depth | A summed medians ms | B ms / geometric speedup | C ms / geometric speedup | D ms / geometric speedup |
| --- | --- | --- | --- | --- |
| 4 | 19.096 | 20.108 / 0.899× | 20.922 / 0.882× | 20.082 / 0.899× |
| 6 | 122.164 | 112.236 / 0.912× | 142.965 / 0.767× | 114.053 / 0.874× |
| 8 | 604.954 | 520.349 / 0.930× | 747.812 / 0.694× | 521.607 / 0.853× |
| 10 | 2434.014 | 2061.416 / 0.948× | 3180.158 / 0.630× | 2079.201 / 0.818× |

| Variant | Broad conditions faster | New generalization geometric speedup | Broad retained bytes / A | Failed declared gates |
| --- | --- | --- | --- | --- |
| hints | 70.4% of 27 | 1.001× | 0.807 | broad_speedup, broad_fraction, distribution_speedup, per_condition, memory |
| iterative | 0.0% of 27 | 0.751× | 1.000 | broad_speedup, broad_fraction, distribution_speedup, summed_latency, summed_cpu, per_condition, expensive |
| combined | 59.3% of 27 | 0.974× | 0.777 | broad_speedup, broad_fraction, distribution_speedup, per_condition, expensive, memory |

Eligibility required all exact checks; broad ≥1.15× and ≥75% faster; overall ≥1.05× and wall sum ≤90%/CPU sum ≤95%; each condition ≤20% slowdown or ≤0.25ms added; expensive conditions including tail ≤20% slowdown and ≤25% extra total nodes; every retained/peak ≤25% growth or ≤64KiB added; broad retained sum ≤15% growth. No thresholds were relaxed.

## Every position and depth

Each cell is complete-decision median milliseconds; B/C/D include A/new speedup. Full CPU times, sample ranges, counters, leaves, hint metrics and memory are in [performance.csv](performance.csv); all raw samples are in [results.jsonl](results.jsonl).

| Position | Depth | A ms | B ms / speedup | C ms / speedup | D ms / speedup |
| --- | --- | --- | --- | --- | --- |
| empty | 4 | 0.821 | 0.932 / 0.881× | 0.901 / 0.912× | 0.934 / 0.879× |
| empty | 6 | 7.505 | 7.935 / 0.946× | 8.406 / 0.893× | 8.028 / 0.935× |
| empty | 8 | 46.812 | 55.620 / 0.842× | 55.285 / 0.847× | 56.220 / 0.833× |
| empty | 10 | 217.484 | 257.752 / 0.844× | 273.529 / 0.795× | 272.338 / 0.799× |
| near-opening | 4 | 0.930 | 0.987 / 0.942× | 1.008 / 0.923× | 0.984 / 0.945× |
| near-opening | 6 | 7.243 | 6.993 / 1.036× | 8.276 / 0.875× | 7.128 / 1.016× |
| near-opening | 8 | 41.643 | 42.717 / 0.975× | 49.734 / 0.837× | 42.239 / 0.986× |
| near-opening | 10 | 176.287 | 176.456 / 0.999× | 225.966 / 0.780× | 181.496 / 0.971× |
| midgame | 4 | 0.548 | 0.650 / 0.842× | 0.624 / 0.878× | 0.653 / 0.838× |
| midgame | 6 | 1.985 | 2.656 / 0.747× | 2.627 / 0.756× | 2.748 / 0.723× |
| midgame | 8 | 4.975 | 7.318 / 0.680× | 7.574 / 0.657× | 8.028 / 0.620× |
| midgame | 10 | 10.117 | 15.833 / 0.639× | 17.833 / 0.567× | 18.822 / 0.538× |
| midgame-wide | 4 | 1.221 | 1.364 / 0.895× | 1.304 / 0.937× | 1.337 / 0.914× |
| midgame-wide | 6 | 5.582 | 6.478 / 0.862× | 6.890 / 0.810× | 6.546 / 0.853× |
| midgame-wide | 8 | 15.647 | 16.965 / 0.922× | 22.450 / 0.697× | 18.018 / 0.868× |
| midgame-wide | 10 | 39.491 | 36.651 / 1.077× | 60.767 / 0.650× | 38.748 / 1.019× |
| late | 4 | 0.089 | 0.116 / 0.771× | 0.106 / 0.841× | 0.116 / 0.772× |
| late | 6 | 0.241 | 0.288 / 0.838× | 0.334 / 0.722× | 0.334 / 0.722× |
| late | 8 | 0.593 | 0.623 / 0.952× | 0.922 / 0.643× | 0.757 / 0.784× |
| late | 10 | 1.142 | 1.252 / 0.912× | 2.043 / 0.559× | 1.485 / 0.769× |
| immediate-win | 4 | 0.557 | 0.525 / 1.060× | 0.628 / 0.886× | 0.525 / 1.060× |
| immediate-win | 6 | 3.417 | 2.682 / 1.274× | 4.052 / 0.843× | 2.633 / 1.298× |
| immediate-win | 8 | 21.473 | 15.524 / 1.383× | 25.465 / 0.843× | 12.851 / 1.671× |
| immediate-win | 10 | 106.670 | 86.321 / 1.236× | 131.653 / 0.810× | 78.987 / 1.350× |
| forced-reply | 4 | 0.566 | 0.619 / 0.915× | 0.640 / 0.884× | 0.617 / 0.917× |
| forced-reply | 6 | 2.655 | 2.599 / 1.022× | 3.291 / 0.807× | 2.591 / 1.025× |
| forced-reply | 8 | 14.427 | 10.597 / 1.361× | 17.798 / 0.811× | 10.558 / 1.366× |
| forced-reply | 10 | 54.162 | 45.629 / 1.187× | 71.815 / 0.754× | 50.080 / 1.081× |
| double-threat-loss | 4 | 0.198 | 0.271 / 0.731× | 0.265 / 0.748× | 0.270 / 0.734× |
| double-threat-loss | 6 | 0.198 | 0.408 / 0.485× | 0.446 / 0.443× | 0.470 / 0.421× |
| double-threat-loss | 8 | 0.196 | 0.391 / 0.501× | 0.635 / 0.309× | 0.660 / 0.297× |
| double-threat-loss | 10 | 0.196 | 0.397 / 0.495× | 0.807 / 0.243× | 0.855 / 0.230× |
| dense-endgame | 4 | 0.021 | 0.026 / 0.797× | 0.026 / 0.800× | 0.026 / 0.789× |
| dense-endgame | 6 | 0.022 | 0.033 / 0.656× | 0.035 / 0.608× | 0.037 / 0.575× |
| dense-endgame | 8 | 0.021 | 0.033 / 0.643× | 0.045 / 0.474× | 0.047 / 0.451× |
| dense-endgame | 10 | 0.022 | 0.034 / 0.638× | 0.056 / 0.387× | 0.059 / 0.363× |
| quiet-5 | 4 | 0.647 | 0.737 / 0.878× | 0.710 / 0.911× | 0.734 / 0.882× |
| quiet-5 | 6 | 3.919 | 4.046 / 0.969× | 4.637 / 0.845× | 4.146 / 0.945× |
| quiet-5 | 8 | 12.140 | 11.970 / 1.014× | 16.733 / 0.726× | 12.271 / 0.989× |
| quiet-5 | 10 | 20.886 | 24.592 / 0.849× | 37.638 / 0.555× | 23.277 / 0.897× |
| supplement-main-003 | 4 | 1.170 | 1.146 / 1.021× | 1.239 / 0.944× | 1.146 / 1.021× |
| supplement-main-003 | 6 | 9.621 | 7.378 / 1.304× | 10.889 / 0.884× | 7.300 / 1.318× |
| supplement-main-003 | 8 | 44.085 | 32.841 / 1.342× | 54.989 / 0.802× | 30.487 / 1.446× |
| supplement-main-003 | 10 | 163.379 | 116.233 / 1.406× | 218.228 / 0.749× | 111.468 / 1.466× |
| seeded-00 | 4 | 1.438 | 1.495 / 0.962× | 1.503 / 0.957× | 1.481 / 0.971× |
| seeded-00 | 6 | 11.838 | 10.972 / 1.079× | 13.284 / 0.891× | 11.058 / 1.070× |
| seeded-00 | 8 | 62.005 | 56.960 / 1.089× | 75.906 / 0.817× | 55.764 / 1.112× |
| seeded-00 | 10 | 240.105 | 191.672 / 1.253× | 309.156 / 0.777× | 216.461 / 1.109× |
| seeded-01 | 4 | 0.278 | 0.353 / 0.787× | 0.345 / 0.806× | 0.352 / 0.789× |
| seeded-01 | 6 | 0.845 | 1.152 / 0.733× | 1.167 / 0.724× | 1.223 / 0.691× |
| seeded-01 | 8 | 3.219 | 4.106 / 0.784× | 4.383 / 0.734× | 4.474 / 0.719× |
| seeded-01 | 10 | 9.818 | 12.443 / 0.789× | 14.160 / 0.693× | 13.732 / 0.715× |
| seeded-02 | 4 | 0.460 | 0.516 / 0.891× | 0.533 / 0.863× | 0.520 / 0.884× |
| seeded-02 | 6 | 1.046 | 1.439 / 0.727× | 1.578 / 0.663× | 1.508 / 0.693× |
| seeded-02 | 8 | 2.602 | 3.667 / 0.710× | 4.138 / 0.629× | 4.149 / 0.627× |
| seeded-02 | 10 | 4.865 | 7.693 / 0.632× | 9.040 / 0.538× | 9.116 / 0.534× |
| seeded-03 | 4 | 0.306 | 0.367 / 0.833× | 0.378 / 0.808× | 0.372 / 0.822× |
| seeded-03 | 6 | 0.939 | 1.121 / 0.837× | 1.278 / 0.735× | 1.186 / 0.792× |
| seeded-03 | 8 | 3.419 | 3.910 / 0.874× | 4.665 / 0.733× | 4.191 / 0.816× |
| seeded-03 | 10 | 11.408 | 9.291 / 1.228× | 16.073 / 0.710× | 10.100 / 1.130× |
| seeded-04 | 4 | 1.542 | 1.333 / 1.157× | 1.621 / 0.951× | 1.332 / 1.158× |
| seeded-04 | 6 | 13.874 | 9.701 / 1.430× | 15.515 / 0.894× | 9.978 / 1.390× |
| seeded-04 | 8 | 89.104 | 59.927 / 1.487× | 104.511 / 0.853× | 62.189 / 1.433× |
| seeded-04 | 10 | 454.223 | 349.920 / 1.298× | 558.805 / 0.813× | 315.435 / 1.440× |
| seeded-05 | 4 | 0.781 | 0.899 / 0.868× | 0.866 / 0.902× | 0.898 / 0.869× |
| seeded-05 | 6 | 4.529 | 5.086 / 0.890× | 5.376 / 0.843× | 5.205 / 0.870× |
| seeded-05 | 8 | 15.245 | 20.287 / 0.751× | 20.615 / 0.740× | 20.658 / 0.738× |
| seeded-05 | 10 | 53.630 | 53.562 / 1.001× | 74.414 / 0.721× | 64.598 / 0.830× |
| seeded-06 | 4 | 0.512 | 0.616 / 0.830× | 0.592 / 0.864× | 0.610 / 0.838× |
| seeded-06 | 6 | 1.728 | 2.341 / 0.738× | 2.336 / 0.740× | 2.448 / 0.706× |
| seeded-06 | 8 | 4.640 | 6.662 / 0.697× | 6.965 / 0.666× | 7.294 / 0.636× |
| seeded-06 | 10 | 11.237 | 16.499 / 0.681× | 18.190 / 0.618× | 19.238 / 0.584× |
| seeded-07 | 4 | 0.284 | 0.375 / 0.757× | 0.346 / 0.818× | 0.375 / 0.756× |
| seeded-07 | 6 | 0.593 | 0.904 / 0.656× | 0.928 / 0.639× | 1.238 / 0.479× |
| seeded-07 | 8 | 1.020 | 1.694 / 0.602× | 1.964 / 0.519× | 3.704 / 0.275× |
| seeded-07 | 10 | 1.509 | 2.620 / 0.576× | 3.423 / 0.441× | 8.486 / 0.178× |
| post-hoc-tail | 4 | 1.834 | 1.831 / 1.002× | 1.913 / 0.959× | 1.839 / 0.998× |
| post-hoc-tail | 6 | 17.291 | 13.884 / 1.245× | 19.279 / 0.897× | 13.982 / 1.237× |
| post-hoc-tail | 8 | 131.103 | 84.098 / 1.559× | 150.403 / 0.872× | 80.925 / 1.620× |
| post-hoc-tail | 10 | 734.887 | 470.574 / 1.562× | 885.264 / 0.830× | 452.068 / 1.626× |
| additional-00 | 4 | 1.325 | 1.348 / 0.983× | 1.405 / 0.943× | 1.337 / 0.991× |
| additional-00 | 6 | 11.135 | 9.852 / 1.130× | 12.504 / 0.891× | 10.082 / 1.104× |
| additional-00 | 8 | 65.421 | 53.514 / 1.222× | 77.893 / 0.840× | 53.115 / 1.232× |
| additional-00 | 10 | 278.144 | 221.201 / 1.257× | 362.279 / 0.768× | 219.028 / 1.270× |
| additional-01 | 4 | 1.416 | 1.323 / 1.070× | 1.499 / 0.945× | 1.332 / 1.063× |
| additional-01 | 6 | 13.062 | 10.355 / 1.261× | 14.493 / 0.901× | 10.344 / 1.263× |
| additional-01 | 8 | 66.710 | 43.428 / 1.536× | 81.255 / 0.821× | 43.264 / 1.542× |
| additional-01 | 10 | 221.029 | 184.816 / 1.196× | 302.261 / 0.731× | 189.190 / 1.168× |
| additional-02 | 4 | 0.702 | 0.705 / 0.996× | 0.761 / 0.923× | 0.712 / 0.986× |
| additional-02 | 6 | 2.530 | 2.298 / 1.101× | 3.287 / 0.770× | 2.335 / 1.083× |
| additional-02 | 8 | 7.025 | 6.151 / 1.142× | 10.295 / 0.682× | 6.021 / 1.167× |
| additional-02 | 10 | 16.142 | 15.204 / 1.062× | 26.440 / 0.611× | 13.862 / 1.165× |
| additional-03 | 4 | 0.426 | 0.521 / 0.817× | 0.499 / 0.854× | 0.524 / 0.813× |
| additional-03 | 6 | 1.098 | 1.456 / 0.754× | 1.604 / 0.684× | 1.435 / 0.765× |
| additional-03 | 8 | 2.247 | 3.793 / 0.592× | 3.868 / 0.581× | 3.796 / 0.592× |
| additional-03 | 10 | 5.511 | 5.840 / 0.944× | 9.319 / 0.591× | 7.781 / 0.708× |
| generalization-00 | 4 | 1.324 | 1.150 / 1.151× | 1.404 / 0.943× | 1.150 / 1.151× |
| generalization-00 | 6 | 10.894 | 7.654 / 1.423× | 12.318 / 0.884× | 7.464 / 1.460× |
| generalization-00 | 8 | 57.293 | 38.435 / 1.491× | 69.446 / 0.825× | 36.699 / 1.561× |
| generalization-00 | 10 | 243.941 | 140.519 / 1.736× | 313.458 / 0.778× | 118.424 / 2.060× |
| generalization-01 | 4 | 0.510 | 0.575 / 0.886× | 0.579 / 0.880× | 0.576 / 0.884× |
| generalization-01 | 6 | 2.636 | 2.840 / 0.928× | 3.263 / 0.808× | 2.915 / 0.904× |
| generalization-01 | 8 | 14.807 | 14.751 / 1.004× | 17.959 / 0.824× | 14.980 / 0.988× |
| generalization-01 | 10 | 74.230 | 71.497 / 1.038× | 92.104 / 0.806× | 77.718 / 0.955× |
| generalization-02 | 4 | 0.794 | 0.883 / 0.900× | 0.856 / 0.928× | 0.891 / 0.891× |
| generalization-02 | 6 | 2.562 | 2.949 / 0.869× | 3.418 / 0.750× | 3.019 / 0.849× |
| generalization-02 | 8 | 7.280 | 7.247 / 1.005× | 10.682 / 0.682× | 7.743 / 0.940× |
| generalization-02 | 10 | 16.862 | 15.248 / 1.106× | 27.571 / 0.612× | 15.671 / 1.076× |
| generalization-03 | 4 | 0.231 | 0.276 / 0.838× | 0.286 / 0.807× | 0.277 / 0.835× |
| generalization-03 | 6 | 0.470 | 0.620 / 0.758× | 0.733 / 0.641× | 0.653 / 0.720× |
| generalization-03 | 8 | 0.903 | 1.217 / 0.742× | 1.636 / 0.552× | 1.429 / 0.632× |
| generalization-03 | 10 | 1.523 | 2.243 / 0.679× | 3.130 / 0.487× | 2.746 / 0.555× |

## Final iteration versus total work and ordering effectiveness

A’s node count is its single final iteration. Other cells show final / TOTAL nodes; total includes every preparatory search. Iteration TT entry sums count entries retained at each iteration end and are not simultaneous cache occupancy. C’s final nodes, hits, entries and cutoffs equal A exactly in every condition; its extra work buys no final ordering.

| Position D10 | A nodes | B final / total | C final / total | D final / total |
| --- | --- | --- | --- | --- |
| empty | 128308 | 121180 / 150605 | 128308 / 163435 | 124698 / 159259 |
| near-opening | 99441 | 73781 / 97835 | 99441 / 128378 | 76139 / 99992 |
| midgame | 5389 | 5389 / 8173 | 5389 / 9734 | 5389 / 9734 |
| midgame-wide | 19097 | 9444 / 17513 | 19097 / 30984 | 9451 / 18312 |
| late | 506 | 251 / 527 | 506 / 953 | 287 / 617 |
| immediate-win | 59934 | 35182 / 47652 | 59934 / 74784 | 36436 / 43832 |
| forced-reply | 30407 | 16923 / 25216 | 30407 / 40641 | 21826 / 27823 |
| double-threat-loss | 81 | 81 / 162 | 81 / 373 | 81 / 373 |
| dense-endgame | 4 | 4 / 8 | 4 / 18 | 4 / 18 |
| quiet-5 | 10140 | 5764 / 12327 | 10140 / 19398 | 5156 / 11780 |
| supplement-main-003 | 87810 | 36142 / 61039 | 87810 / 119154 | 41406 / 58042 |
| seeded-00 | 134238 | 73095 / 108144 | 134238 / 177329 | 86061 / 117499 |
| seeded-01 | 5221 | 4677 / 6512 | 5221 / 7674 | 4677 / 7067 |
| seeded-02 | 2290 | 2240 / 3568 | 2290 / 4426 | 2240 / 4259 |
| seeded-03 | 5772 | 2737 / 4538 | 5772 / 8247 | 2747 / 4829 |
| seeded-04 | 253274 | 139458 / 190378 | 253274 / 313546 | 136557 / 171547 |
| seeded-05 | 28891 | 19175 / 27650 | 28891 / 40596 | 22250 / 33318 |
| seeded-06 | 6129 | 6129 / 8795 | 6129 / 10163 | 6129 / 10163 |
| seeded-07 | 730 | 730 / 1229 | 730 / 1680 | 2209 / 3939 |
| post-hoc-tail | 423857 | 190958 / 269620 | 423857 / 514601 | 211583 / 259977 |
| additional-00 | 153647 | 81273 / 116705 | 153647 / 196159 | 86324 / 114765 |
| additional-01 | 117199 | 58109 / 94609 | 117199 / 162164 | 73061 / 96420 |
| additional-02 | 8271 | 3973 / 7557 | 8271 / 13647 | 3826 / 6816 |
| additional-03 | 2762 | 1609 / 2681 | 2762 / 4626 | 1780 / 3517 |
| generalization-00 | 139988 | 44757 / 78515 | 139988 / 181096 | 44778 / 65630 |
| generalization-01 | 41370 | 30743 / 39109 | 41370 / 51522 | 33960 / 42164 |
| generalization-02 | 8193 | 3331 / 6957 | 8193 / 13579 | 3370 / 6930 |
| generalization-03 | 754 | 625 / 1066 | 754 / 1567 | 625 / 1297 |

| Variant D8/D10 pooled | Lookups | Legal cross hints | Cross hint first | Cross hint changes first | Cutoff on cross hinted first |
| --- | --- | --- | --- | --- | --- |
| hints | 313649 | 61786 | 60259 | 18009 | 36679 |
| combined | 433329 | 76921 | 74936 | 20584 | 44143 |

The supplemental [cross-hint-effectiveness.json](cross-hint-effectiveness.json) distinguishes cross-depth suggestions from same-depth TT hints. Its source is archived and its complete scores/counters match primary evidence. All instrumentation is excluded from timing. A legal hint or first-move cutoff is not itself causal proof of saved work; compare final and total nodes/leaves and complete time. Primary changed-first/hinted-cutoff counters include both kinds of ordering hint; supplemental cross counters disambiguate them.

| Variant D8/D10 | Total nodes / A | Final nodes / A | Total leaves / A | Final leaves / A |
| --- | --- | --- | --- | --- |
| direct | 1.000 | 1.000 | 1.000 | 1.000 |
| hints | 0.832 | 0.588 | 0.891 | 0.624 |
| iterative | 1.301 | 1.000 | 1.336 | 1.000 |
| combined | 0.832 | 0.624 | 0.908 | 0.664 |

## Memory and expensive searches

Retained memory sums unique reachable Python objects, attributing score TT first, hint dictionary/keys/moves second, then table overhead. Shared objects count once; dictionary allocated capacity is included. Two repetitions match retained sizes exactly. Traced peak includes intermediate snapshot export overlap; traced current includes result objects. These are different from fresh-process RSS, which includes interpreter, imports and allocator. Production releases all caches after each decision; diagnostic retention is for sizing only.

| Variant non-tail | Sum retained / A | Maximum retained growth / A | Maximum peak growth / A | Largest peak MiB |
| --- | --- | --- | --- | --- |
| direct | 1.000 | 1.000 | 1.000 | 7.693 |
| hints | 0.819 | 1.634 | 1.486 | 7.787 |
| iterative | 1.000 | 1.061 | 1.310 | 7.694 |
| combined | 0.792 | 4.594 | 4.066 | 4.757 |

| Variant | Memory gate failures (condition / retained or peak) |
| --- | --- |
| hints | midgame/D10/retained, midgame/D10/peak, seeded-06/D10/retained, seeded-06/D10/peak |
| iterative | none |
| combined | midgame/D10/retained, midgame/D10/peak, seeded-06/D10/retained, seeded-06/D10/peak, seeded-07/D10/retained, seeded-07/D10/peak |

| Position D10 | Variant | Wall ms / speedup | Retained MiB | Hint KiB | Peak MiB | Fresh RSS MiB |
| --- | --- | --- | --- | --- | --- | --- |
| empty | direct | 217.484 / 1.000× | 3.624 | 0.0 | 3.758 | 47.02 |
| empty | hints | 257.752 / 0.844× | 4.134 | 607.0 | 4.272 | 47.91 |
| empty | iterative | 273.529 / 0.795× | 3.624 | 0.0 | 3.760 | 47.16 |
| empty | combined | 272.338 / 0.799× | 4.161 | 616.4 | 4.300 | 48.03 |
| near-opening | direct | 176.287 / 1.000× | 3.094 | 0.0 | 3.285 | 46.25 |
| near-opening | hints | 176.456 / 0.999× | 3.225 | 542.8 | 3.822 | 46.62 |
| near-opening | iterative | 225.966 / 0.780× | 3.094 | 0.0 | 3.287 | 46.48 |
| near-opening | combined | 181.496 / 0.971× | 3.264 | 514.4 | 3.795 | 47.06 |
| seeded-04 | direct | 454.223 / 1.000× | 7.415 | 0.0 | 7.693 | 52.41 |
| seeded-04 | hints | 349.920 / 1.298× | 6.443 | 1165.7 | 7.787 | 51.47 |
| seeded-04 | iterative | 558.805 / 0.813× | 7.415 | 0.0 | 7.694 | 52.67 |
| seeded-04 | combined | 315.435 / 1.440× | 4.599 | 627.5 | 4.757 | 48.5 |
| post-hoc-tail | direct | 734.887 / 1.000× | 12.574 | 0.0 | 13.333 | 60.53 |
| post-hoc-tail | hints | 470.574 / 1.562× | 7.313 | 1349.0 | 8.027 | 52.38 |
| post-hoc-tail | iterative | 885.264 / 0.830× | 12.574 | 0.0 | 13.334 | 60.84 |
| post-hoc-tail | combined | 452.068 / 1.626× | 6.890 | 673.9 | 7.339 | 52.02 |

Fresh RSS shows after-search retained snapshots when `ps` is available; otherwise the table uses OS high water. [rss.json](rss.json) records before/after/high-water fields and parity. RSS observations are single fresh-process diagnostics, not replicated memory claims.

In this run `ps` process inspection was unavailable in the sandbox, so all sixteen fresh-process measurements report OS `ru_maxrss` high water. Before/after RSS fields are null in the raw evidence; no before/after RSS reduction is claimed.

| Variant | Per-condition latency gate failures | All failed conditions |
| --- | --- | --- |
| hints | 20 | midgame/D6, midgame/D8, midgame/D10, seeded-01/D6, seeded-01/D8, seeded-01/D10, seeded-02/D6, seeded-02/D8, seeded-02/D10, seeded-05/D8, seeded-06/D6, seeded-06/D8, seeded-06/D10, seeded-07/D6, seeded-07/D8, seeded-07/D10, additional-03/D6, additional-03/D8, generalization-03/D8, generalization-03/D10 |
| iterative | 60 | empty/D10, near-opening/D10, midgame/D6, midgame/D8, midgame/D10, midgame-wide/D6, midgame-wide/D8, midgame-wide/D10, late/D8, late/D10, immediate-win/D10, forced-reply/D6, forced-reply/D8, forced-reply/D10, double-threat-loss/D8, double-threat-loss/D10, quiet-5/D8, quiet-5/D10, supplement-main-003/D8, supplement-main-003/D10, seeded-00/D8, seeded-00/D10, seeded-01/D6, seeded-01/D8, seeded-01/D10, seeded-02/D6, seeded-02/D8, seeded-02/D10, seeded-03/D6, seeded-03/D8, seeded-03/D10, seeded-04/D10, seeded-05/D8, seeded-05/D10, seeded-06/D6, seeded-06/D8, seeded-06/D10, seeded-07/D6, seeded-07/D8, seeded-07/D10, additional-00/D10, additional-01/D8, additional-01/D10, additional-02/D6, additional-02/D8, additional-02/D10, additional-03/D6, additional-03/D8, additional-03/D10, generalization-00/D8, generalization-00/D10, generalization-01/D6, generalization-01/D8, generalization-01/D10, generalization-02/D6, generalization-02/D8, generalization-02/D10, generalization-03/D6, generalization-03/D8, generalization-03/D10 |
| combined | 29 | empty/D8, empty/D10, midgame/D6, midgame/D8, midgame/D10, late/D10, double-threat-loss/D6, double-threat-loss/D8, double-threat-loss/D10, seeded-01/D6, seeded-01/D8, seeded-01/D10, seeded-02/D6, seeded-02/D8, seeded-02/D10, seeded-03/D8, seeded-05/D8, seeded-05/D10, seeded-06/D6, seeded-06/D8, seeded-06/D10, seeded-07/D6, seeded-07/D8, seeded-07/D10, additional-03/D6, additional-03/D8, additional-03/D10, generalization-03/D8, generalization-03/D10 |

Improvements on the prior high-cost tail are useful research evidence, but tail success cannot override broad-board or memory failures. Shallow, tactical and terminal-heavy searches can finish before hints amortize preparation. The empty opening is a material counterexample to treating iterative deepening as universally faster. No adaptive fixture-specific strategy was added.

## Experimental depth 12

Empty direct D12 preflight completed in 0.704s, high-water RSS 60.45 MiB, conservative diagnostic projection 349.3s ≤600s. All four declared D12 diagnostic positions completed, with seven matched samples and one traced run per variant. Depth 12 remains unexposed by public API/caps.

| Position D12 | A ms | B ms / speedup | C ms / speedup | D ms / speedup |
| --- | --- | --- | --- | --- |
| empty | 669.104 | 858.905 / 0.779× | 943.374 / 0.709× | 948.959 / 0.705× |
| near-opening | 620.651 | 560.077 / 1.108× | 849.375 / 0.731× | 596.116 / 1.041× |
| seeded-04 | 1996.677 | 1378.172 / 1.449× | 2555.757 / 0.781× | 1275.790 / 1.565× |
| post-hoc-tail | 3279.200 | 1969.897 / 1.665× | 4162.320 / 0.788× | 1707.602 / 1.920× |

| Position D12 | Variant | Final / total nodes | Retained / peak MiB |
| --- | --- | --- | --- |
| empty | direct | 380468 / 380468 | 12.35 / 13.25 |
| empty | hints | 355926 / 484234 | 14.44 / 15.96 |
| empty | iterative | 380468 / 543903 | 12.35 / 13.25 |
| empty | combined | 377792 / 537051 | 14.68 / 15.92 |
| near-opening | direct | 340974 / 340974 | 11.61 / 13.39 |
| near-opening | hints | 203223 / 302664 | 9.00 / 9.27 |
| near-opening | iterative | 340974 / 469352 | 11.61 / 13.40 |
| near-opening | combined | 220866 / 320858 | 9.13 / 9.41 |
| seeded-04 | direct | 1089090 / 1089090 | 42.06 / 53.24 |
| seeded-04 | hints | 485487 / 738761 | 20.90 / 21.49 |
| seeded-04 | iterative | 1089090 / 1402636 | 42.06 / 53.24 |
| seeded-04 | combined | 514229 / 685776 | 18.58 / 19.18 |
| post-hoc-tail | direct | 1820265 / 1820265 | 54.52 / 56.49 |
| post-hoc-tail | hints | 663738 / 1087595 | 32.11 / 36.26 |
| post-hoc-tail | iterative | 1820265 / 2334866 | 54.52 / 56.49 |
| post-hoc-tail | combined | 695527 / 955504 | 27.50 / 31.42 |

## Correctness, validation and evidence audit

All 4,705 saved main/preflight/RSS/D12 decisions have identical ordered final scores and selected moves across variants and deterministic counters within each variant. [root-parity.json](root-parity.json) records every complete comparison vector and final counters. All 96 prior Phase 3B.2 score/move/counter/leaf vectors match direct A. Independent unpruned array minimax verified 99 vectors and 396 variant decisions: every manifest board at 1/2/3 plus nearly full draw prefixes at 4/6/8/10/12. [oracle-vectors.json](oracle-vectors.json) preserves expected scores. Deeper broad checks use the immutable validated baseline, not an unpruned depth-10 oracle.

The 109 added tests cover true forced losses and faster wins, required replies, Phase 4 unsafe-cache reproduction, transpositions, differing mover/horizon semantics, every retained bound against array truth, full draws and near-full boards, booleans/floats/strings/foreign/full-column hints, ties, repeated decisions, incremental state, recursive/root/final-iteration/export exception rollback, source scope and exclusive evidence writes. Primary and supplemental instruments preserve self counters; no invalid generated hints occurred. C has exactly A’s final counters, while B/D counters legitimately change.

Full backend: **2723 passed, 15 skipped in 114.15s (0:01:54)**. [backend-tests.txt](backend-tests.txt) contains full output. The suite preserves prior Negamax/MCTS tests and checks public API validation/caps, search concurrency, rate limits, history/replay and provenance. Runs use `PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY=`; the global fixture forbids live HTTPS providers. PyTorch skips retain the existing dependency limitation. No paid API calls, public interface or frontend changes; frontend tests were unnecessary. Python compilation and Git whitespace checks pass. Production agent bytes equal baseline; previous phase reports, canonical datasets and protected services/models are untouched.

[audit.json](audit.json) verifies exact coverage, source/design/runtime hashes, all saved score/counter vectors, repeated retained sizes, previous-phase parity and oracle results. The report is generated from complete evidence. Negative findings apply to these concrete target−2/even-increment schedules and previous-iteration snapshots, not every possible iterative/hint design. No sampling uncertainty or universal strength improvement is claimed.

## Reproduction and next phase

Use this checkout and immutable baseline objects; choose a fresh output directory. Runners refuse to overwrite evidence. Copy DESIGN first, then declare. Run serially so benchmarks do not compete with tests. Source/hash drift fails closed. `ps` RSS is optional; high water is retained. Run depth12 only when its preflight saved `passed: true`.

```sh
mkdir -p /tmp/negamax-iterative-replay
cp docs/search-negamax-v2/iterative-deepening/DESIGN.md /tmp/negamax-iterative-replay/
export PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY=
.venv/bin/python -m scripts.benchmark_negamax_iterative declare --directory /tmp/negamax-iterative-replay
.venv/bin/python -m scripts.benchmark_negamax_iterative preflight --directory /tmp/negamax-iterative-replay
.venv/bin/python -m scripts.benchmark_negamax_iterative run --directory /tmp/negamax-iterative-replay
.venv/bin/python -m scripts.benchmark_negamax_iterative rss --directory /tmp/negamax-iterative-replay
.venv/bin/python -m scripts.benchmark_negamax_iterative depth12-preflight --directory /tmp/negamax-iterative-replay
# If the depth12 gate passes, run its declared four-way diagnostic:
# .venv/bin/python -m scripts.benchmark_negamax_iterative depth12 --directory /tmp/negamax-iterative-replay
.venv/bin/python -m scripts.diagnose_negamax_iterative --directory /tmp/negamax-iterative-replay
.venv/bin/python -m scripts.audit_negamax_iterative --directory /tmp/negamax-iterative-replay
.venv/bin/python -m pytest tests -q -rs > /tmp/negamax-iterative-replay/backend-tests.txt
.venv/bin/python -m scripts.summarize_negamax_iterative --directory /tmp/negamax-iterative-replay
```

Recommended next phase: evaluate collision-free mirror canonicalization of the score TT as a separately predeclared experiment, including exact column remapping for hints, stable root ties, all bound semantics and complete-decision overhead. Symmetry can reduce duplicate work without paying for preparatory horizons; bit manipulation and canonicalization costs must still earn their place on this same distribution. Keep iterative deepening and cross-depth hints experimental unless a new independently declared policy meets correctness and broad regression/memory gates.
