Phase 1: Negamax move ordering and TT move hints — 2026-10-09

Work is confined to the dedicated development checkout on `search/negamax-v2`, based on `b3ce5b7e490b871ae85328698902a5e92d0490ef`. The selected internal default is immediate wins first, then a valid TT move hint, then stable center order. Threat-count ordering remains an internal ablation option and is disabled for ordinary agent decisions. No public API, configured depth, deployed agent configuration, pruning rule, or search horizon changed. No commit, push, PR, merge, or deployment was performed.

The audit confirmed that the existing engine already uses two 49-bit boards, six playable bits and one sentinel per column, shift-based four detection, detached heights, and protected play/undo. It checks terminal outcomes before heuristic leaves: wins/losses are mover-relative ±(1,000,000 + remaining depth), and draws are zero. The 69-window, 1/3/9/81 heuristic is unchanged. Every root child is searched with a full window and a fresh per-decision shared table; root iteration stays in `(3, 2, 4, 1, 5, 0, 6)` order, so all reported root scores remain exact for their depth and root ties retain their historical choice.

The TT key remains `(X bitboard, O bitboard, mover, remaining depth)`. Both mover and depth are necessary: different horizons have different heuristic values and terminal magnitudes. EXACT returns a value, LOWER raises alpha, UPPER lowers beta, and a closed window returns a bound. Storage still classifies against the original incoming alpha/beta, before TT tightening. Values from narrow windows are not promoted to exact merely because a hint exists. Entries now contain `(flag, value, searched_move)`; a bound entry's move is only an ordering hint, not proof of an optimal move. Illegal hints are ignored. Existing node/hit/cutoff accounting is preserved, including the fact that the cutoff counter counts search-loop cutoffs rather than TT-only returns.

The ordering path scores no tactical moves at remaining depth one. Immediate-win scoring is also skipped when the mover has fewer than three stones, and sorting is skipped when no winning square is currently playable. All legal moves remain eligible. The optional threat mode counts empty own winning squares after a candidate drop, including unsupported future squares, using the geometry already present in Victor's Python/C search. Those squares affect ordering only; Victor's terminal-only forced-move rules, endgame shortcuts, mirror keys, and theorem bounds were not transferred to this depth-limited search. TT hints have limited opportunity here because many same-depth hits already return exact values or close the window; no iterative deepening or cross-depth score reuse was added.

Measurements use the five histories imported directly from the existing `scripts/benchmark_public_agents.py`, including its varied midgame. The original script and historical evidence are untouched. Raw measurements are in [baseline.json](baseline.json), [ordering.json](ordering.json), and [seeded-ordering.json](seeded-ordering.json). They include histories, every root score, chosen moves, raw elapsed samples, nodes, table entries, TT hits, search-loop cutoffs, and source hashes. `baseline.json` was collected before editing the agent; the stronger comparison below uses the interleaved baseline in `ordering.json`.

Hardware was a MacBook Pro Mac17,2, Apple M5, 10 cores (4 performance / 6 efficiency), 32 GB RAM; CPython 3.11.17, native arm64, macOS 26.6.2. Host-visible Python processes other than the benchmark and Docker host processes showed no significant CPU activity before the heavy runs; the two pre-existing Python processes were at 0.0%. Browser/system activity remained, with host load averages around 2.7–4.2. No processes were stopped, reprioritized, or altered. These are local measurements, not measurements of the remote Mac Mini production service or a claim of quiet-machine isolation.

Each variant/position/depth has one discarded warm-up and three fresh-table samples. Variant order rotates per repetition in one interpreter; game construction is excluded. Root scores and moves are asserted identical between variants, and counters must repeat exactly. Memory sizing is outside latency measurement, after each variant's final sample. It sums unique Python object sizes reachable from the retained table, including keys, entries, and options; it excludes allocator overhead, temporary allocations, process RSS, and peak memory. Three samples are descriptive medians/maxima, not confidence intervals, p95 estimates, or production latency guarantees.

For the selected wins-plus-TT-hints default, the following medians and node counts are paired within the same run. Columns are zero-based. Every before/after choice and entire root-score vector agreed; exact vectors are in the JSON, rather than only the best score.

| Position | Depth | Move (both) | Median ms before → after | Nodes before → after |
| --- | ---: | ---: | ---: | ---: |
| empty | 4 | 3 | 2.317 → 2.322 | 594 → 594 |
| empty | 6 | 3 | 18.008 → 17.712 | 5,059 → 5,052 |
| empty | 8 | 3 | 110.100 → 110.197 | 31,207 → 29,425 |
| empty | 10 | 2 | 545.293 → 422.695 | 172,590 → 128,308 |
| near-opening | 4 | 3 | 2.917 → 2.096 | 768 → 569 |
| near-opening | 6 | 3 | 26.439 → 14.611 | 8,347 → 4,258 |
| near-opening | 8 | 3 | 217.928 → 78.916 | 80,742 → 24,054 |
| near-opening | 10 | 3 | 1575.150 → 301.708 | 646,784 → 99,441 |
| midgame | 4 | 3 | 1.715 → 0.974 | 610 → 339 |
| midgame | 6 | 3 | 12.378 → 2.978 | 4,583 → 1,166 |
| midgame | 8 | 3 | 58.545 → 6.542 | 22,981 → 2,784 |
| midgame | 10 | 3 | 196.160 → 12.107 | 82,766 → 5,389 |
| midgame-wide | 4 | 3 | 2.767 → 2.576 | 784 → 726 |
| midgame-wide | 6 | 3 | 12.508 → 9.113 | 4,286 → 3,036 |
| midgame-wide | 8 | 3 | 43.766 → 21.780 | 17,450 → 8,069 |
| midgame-wide | 10 | 3 | 114.605 → 45.574 | 52,881 → 19,097 |
| late | 4 | 4 | 0.102 → 0.111 | 41 → 41 |
| late | 6 | 4 | 0.272 → 0.310 | 120 → 119 |
| late | 8 | 4 | 0.541 → 0.636 | 284 → 276 |
| late | 10 | 6 | 0.984 → 1.098 | 600 → 506 |

Depth-10 latency fell by 22.5% on empty, 80.8% on near-opening, 93.8% on tactical midgame, and 60.2% on varied midgame. The late board regressed by 11.6% (about 0.114 ms) despite fewer nodes. Late-board regressions across all depths were approximately 0.009–0.114 ms / 9.0–17.6%. Empty depths 4 and 8 were effectively unchanged in latency, even though depth 8 visited 5.7% fewer nodes. Those small timing differences should not be presented as gains. Tactical overhead and the fixed costs of tiny trees can outweigh work saved.

TT and retained-memory measurements for the selected default follow. Full entry counts and all per-sample counters are in the JSON. The extra move field increases storage per entry; memory can grow where node savings are small (for example empty depth 8), even though larger tactical searches retain much less memory overall.

| Position | Depth | TT hits before → after | Cutoffs before → after | Retained TT MiB before → after |
| --- | ---: | ---: | ---: | ---: |
| empty | 4 | 15 → 15 | 71 → 71 | 0.029 → 0.030 |
| empty | 6 | 396 → 396 | 914 → 914 | 0.314 → 0.325 |
| empty | 8 | 2,875 → 2,725 | 5,886 → 5,639 | 1.787 → 1.793 |
| empty | 10 | 21,038 → 15,828 | 35,485 → 27,477 | 10.970 → 8.259 |
| near-opening | 4 | 24 → 19 | 114 → 91 | 0.045 → 0.035 |
| near-opening | 6 | 1,182 → 394 | 1,775 → 806 | 0.483 → 0.257 |
| near-opening | 8 | 18,667 → 2,903 | 19,722 → 4,939 | 5.353 → 1.537 |
| near-opening | 10 | 188,329 → 14,340 | 162,963 → 22,095 | 42.816 → 6.746 |
| midgame | 4 | 16 → 0 | 73 → 37 | 0.030 → 0.019 |
| midgame | 6 | 390 → 61 | 630 → 167 | 0.215 → 0.076 |
| midgame | 8 | 3,311 → 284 | 3,523 → 452 | 1.071 → 0.209 |
| midgame | 10 | 16,073 → 700 | 14,087 → 993 | 4.169 → 0.437 |
| midgame-wide | 4 | 25 → 25 | 99 → 100 | 0.043 → 0.039 |
| midgame-wide | 6 | 350 → 242 | 733 → 582 | 0.238 → 0.202 |
| midgame-wide | 8 | 2,312 → 796 | 3,393 → 1,950 | 1.031 → 0.608 |
| midgame-wide | 10 | 9,163 → 2,759 | 11,434 → 5,110 | 3.422 → 1.632 |
| late | 4 | 3 → 3 | 4 → 4 | 0.005 → 0.005 |
| late | 6 | 6 → 6 | 17 → 18 | 0.014 → 0.015 |
| late | 8 | 11 → 11 | 72 → 72 | 0.035 → 0.036 |
| late | 10 | 40 → 22 | 195 → 166 | 0.085 → 0.070 |

Ablations below show depth-10 median milliseconds / nodes. `center` uses the new entry representation with both ordering features disabled, separating structural changes from ordering. `tt` adds only hints; `wins` adds only immediate wins; `tt-wins` is the selected default; `threats` uses only future threat counts; `tactical` uses wins plus threats; `combined` adds TT hints to tactical ordering. All eight variants were measured at all four depths, not only these displayed depth-10 rows.

| Variant | Empty | Near-opening | Midgame | Midgame-wide | Late |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 545.293 / 172,590 | 1575.150 / 646,784 | 196.160 / 82,766 | 114.605 / 52,881 | 0.984 / 600 |
| center | 547.148 / 172,590 | 1588.550 / 646,784 | 195.531 / 82,766 | 116.400 / 52,881 | 0.973 / 600 |
| tt | 541.126 / 171,530 | 1554.177 / 631,538 | 190.199 / 80,424 | 110.550 / 50,976 | 0.978 / 598 |
| wins | 424.629 / 128,737 | 315.886 / 101,598 | 12.089 / 5,389 | 45.930 / 19,235 | 1.093 / 506 |
| tt-wins | 422.695 / 128,308 | 301.708 / 99,441 | 12.107 / 5,389 | 45.574 / 19,097 | 1.098 / 506 |
| threats | 706.425 / 173,755 | 2252.185 / 678,891 | 457.029 / 142,792 | 244.431 / 69,753 | 1.489 / 541 |
| tactical | 455.005 / 104,950 | 295.952 / 69,162 | 26.048 / 6,549 | 44.504 / 9,938 | 1.190 / 365 |
| combined | 456.188 / 104,558 | 289.596 / 67,509 | 25.953 / 6,549 | 44.457 / 9,904 | 1.180 / 365 |

TT hints alone reduced depth-10 nodes by about 0.3–3.6% across these positions. On top of wins, their node reduction ranges from zero to 2.1%; near-opening changes from 101,598 to 99,441 nodes, while the tactical midgame is identical. Keep the move field and same-depth hint use: the measured node reductions are modest but repeatable, and the change does not weaken bound semantics. Do not claim a broad latency win for hints alone from these small samples.

Keep immediate-win ordering as the default: it supplies the main measured gains across several distinct positions, with explicit late-board regressions. Keep the low-overhead gates; they use general board/horizon facts, not fixture names. Revert the initially tested threat-enabled default. Retain threat modes disabled for reproducible ablation: threat-only was substantially slower at depth 10 on every existing position, and wins-plus-threats can produce fewer nodes while still being slower. For example, empty drops from 128,308 to 104,558 nodes when threats are added to the selected default, yet latency rises from 422.695 to 456.188 ms. Near-opening benefits slightly from adding threats, but that single case does not justify enabling them broadly.

A separately measured generalization set contains eight deterministic legal nonterminal histories at 8, 12, 16, and 20 plies (two each), generated without filtering on search performance using seed 20261009. Wins plus hints was faster and visited fewer nodes in all 16 comparisons at depths 6 and 8. Speedups range from 1.15–21.75× at depth 6 and 1.14–53.83× at depth 8; the largest gains are tactical, not a strength claim. Adding threats was slower than wins plus hints in 11/16 comparisons (6/8 at depth 6, 5/8 at depth 8), although it helps some broader trees. Raw histories allow independent reruns; these positions were not used to choose the gates.

| Seeded position | Depth | Median ms before → after | Nodes before → after |
| --- | ---: | ---: | ---: |
| seeded-00 | 6 | 38.652 → 25.105 | 10,838 → 7,103 |
| seeded-00 | 8 | 326.600 → 115.167 | 98,729 → 35,049 |
| seeded-01 | 6 | 3.841 → 1.223 | 1,379 → 443 |
| seeded-01 | 8 | 21.409 → 5.341 | 7,559 → 1,835 |
| seeded-02 | 6 | 32.071 → 1.604 | 11,766 → 527 |
| seeded-02 | 8 | 202.593 → 3.763 | 83,930 → 1,328 |
| seeded-03 | 6 | 5.630 → 1.253 | 2,211 → 478 |
| seeded-03 | 8 | 25.415 → 4.596 | 10,732 → 1,801 |
| seeded-04 | 6 | 37.656 → 30.033 | 10,446 → 8,304 |
| seeded-04 | 8 | 246.590 → 170.386 | 76,210 → 50,920 |
| seeded-05 | 6 | 14.261 → 8.698 | 4,662 → 2,701 |
| seeded-05 | 8 | 75.072 → 23.952 | 26,583 → 8,475 |
| seeded-06 | 6 | 3.029 → 2.637 | 1,261 → 1,005 |
| seeded-06 | 8 | 7.695 → 6.728 | 3,376 → 2,666 |
| seeded-07 | 6 | 11.371 → 0.523 | 4,098 → 284 |
| seeded-07 | 8 | 42.012 → 0.863 | 18,600 → 499 |

Correctness checks use the existing independent array/engine minimax oracle, not Victor or the optimized search as its own oracle. Every root move is checked across all seven candidate ordering configurations at tractable depths, including immediate/slower wins, forced blocks, the unsafe-cache regression, exact ties, varied midgames, and near-draw boards. Added tests independently check winning-square geometry and stable tactical ordering, inspect every stored bound against minimax values on a tractable tree, exercise exact/lower/upper entries with valid/invalid hints and several windows, and verify detached-state/caller restoration even when evaluation raises. Existing seeded positions, mover/depth key distinctions, transposition reuse, terminal perspectives, full draws, and fresh-decision determinism remain covered.

Final verification results: the selected backend regression command passed **525 tests**, with no failures or skips. It covers Negamax, engine rules, public agent API validation and depth caps, MCTS, seeded/history/evidence APIs, grounding/explanations, provenance, and concurrency/rate limits. Negamax contributes **78 tests**. The original pre-change Negamax suite passed 15 tests. All 160 main ablation rows and 72 supplemental rows completed their score/move/counter assertions. The full repository suite and frontend/browser suites were not run; this change has no frontend code, and the selected backend checks cover its integration. `git diff --check` passed.

Reproduce after checking host contention, from this checkout, with no service running or provider access needed:

```sh
.venv/bin/python -m scripts.benchmark_negamax_ordering --baseline-ref b3ce5b7e490b871ae85328698902a5e92d0490ef --output /tmp/negamax-ordering.json
.venv/bin/python -m scripts.benchmark_negamax_ordering --baseline-ref b3ce5b7e490b871ae85328698902a5e92d0490ef --depths 6 8 --variants baseline wins tt-wins combined --positions midgame-wide --seeded-positions 8 --output /tmp/negamax-seeded.json
EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m pytest tests/test_connect4_negamax.py tests/test_connect4_engine.py tests/test_connect4_api.py tests/test_connect4_mcts.py tests/test_connect4_mcts_api.py tests/test_connect4_seeded_api.py tests/test_connect4_history_api.py tests/test_connect4_evidence.py tests/test_connect4_grounding.py tests/test_connect4_explanations.py tests/test_connect4_provenance.py tests/test_public_api_limits.py -q -rs
```

Files changed:

- `games/connect4/agents/negamax_agent.py`: internal move ordering, winning-square geometry, and TT move field; public agent constructor and root-score interface unchanged.
- `tests/test_connect4_negamax.py`: independent oracle/ordering/bound/restoration coverage and the extended entry representation.
- `scripts/benchmark_negamax_ordering.py`: original-source loading by Git commit, interleaved ablations, score/move assertions, counters, retained-memory estimates, and seeded histories.
- `docs/search-negamax-v2/REPORT.md`, `baseline.json`, `ordering.json`, `seeded-ordering.json`: this report and new development evidence, separate from canonical/frozen evidence.

The Services checkout, Docker containers, Tailscale Funnel, Render, deployment configuration, AlphaZero worktree/processes/models/evidence, `canonical-season-v1`, and existing frozen benchmark artifacts were not modified. Only development files are pending review; no Git write operation was used.

Proposed next phase, not implemented: profile leaf evaluation and TT work on a larger predeclared legal-position set, including quiet endgames and loss trees, with CPU time as well as elapsed time. Evaluate incremental window scoring separately while proving the same 69-window heuristic with independent tests. Evaluate compact full-key storage and deterministic bounded-table replacement to reduce memory, preserving depth/mover separation and typed bounds; benchmark hint use before considering additional hint reuse across depths (never reuse a value from another horizon). Mirror canonicalization would need move remapping and explicit tie tests. Revisit cheap threat ordering only with broader evidence. Preserve full-window values for every root move throughout; root aspiration, PVS, iterative deepening, forced-move pruning, and depth/cap changes are outside this iteration.
