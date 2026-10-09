# Phase 3A: deeper Connect Four Negamax evaluation

October 9, 2026. Application baseline: [babaac798c8071fe60985ed41f0465442dee0fc7](https://github.com/andrewcukierwar/board-game-ai-lab/commit/babaac798c8071fe60985ed41f0465442dee0fc7), branch `search/negamax-v2`. Production Negamax/MCTS, API limits, UI presets, deployment and all earlier evidence are unchanged.

**All 512 declared games completed. Neither primary comparison establishes a strength improvement:** N8 scores 52.3% against N6 and N10 scores 54.7% against N8, with both family-adjusted intervals crossing parity. Depth 10 is materially more expensive on broad quiet trees, reaching 1.41 seconds and 423,857 nodes in a played game. Heuristic evaluation is the leading CPU target; Python TT objects dominate retained search memory. These results support optimization of the existing semantics before considering a public depth change.

## Frozen experiment and runtime

[DESIGN.md](DESIGN.md) predates the experiment; [candidate.json](candidate.json) preserves the target manifest. [experiment.json](experiment.json) freezes 64 pairs for each primary and 32 for each secondary comparison, both agent colors: 512 games. The 96-game independent preflight completed in 72.99 seconds; its timing-only projection was 379.18 seconds. The declared 1.75× multiplier gives 663.57 seconds, within the 1,500-second planning allowance plus 300-second reserve. Thus the full target was feasible; no reduced design, cap increase, or outcome-based extension occurred. Main cumulative wall cost was **344.15 seconds / 1,800 seconds (19.1%)**, including the real [12-game checkpoint](checkpoint-status.json) and validated resume. There were no errors, incomplete games, or unmatched colors. Preflight and audit repetitions never enter inference.

Preflight maxima were 8,682 / 53,397 / 256,169 nodes at depths 6/8/10, with traced peaks 0.51 / 3.41 / 17.17 MiB and retained TT 0.49 / 3.34 / 16.86 MiB. These are observed maxima, not worst-case guarantees. Raw preflight games, memory and freeze decision are in [preflight.jsonl](preflight.jsonl), [preflight-memory.json](preflight-memory.json), and [freeze-record.json](freeze-record.json).

Hardware: MacBook Pro Mac17,2, Apple M5, ten cores (four performance, six efficiency), 32 GB; native arm64 CPython 3.11.17, macOS 26.6.2. All search measurements, profiles, audits and tests ran serially. [Host snapshots](host-start.json) show load averages around 3.6–4.0 at the start; process CPU inspection was sandbox-denied. Other host work was neither isolated nor stopped. These are laptop measurements, not endpoint or production Mac Mini performance. Existing deployment documentation describes a one-worker/four-thread service and a shared two-search admission gate; no production concurrency or hardware extrapolation is measured here.

## Opening cohorts and stage results

The 64-opening primary pool comprises 32 random boards and 32 snapshots from eight agent-play trajectories. Secondary comparisons share its first 32 openings: 16 random boards and four trajectories. The random cohort retains Phase 2B's 2/5/8/11/14/17/20/23-ply distribution, with board-plus-mover deduplication. The agent cohort uses Negamax 4 and MCTS 400 with 20% legal exploration after two random initial moves; snapshots are at 5/14/23/32 plies. Complete surviving trajectories are regenerated only for early termination or duplicate boards. This is survival-conditioned exploratory agent play, not typical human or perfect play. All main and preflight board identities are unique and disjoint. Related snapshots share one inferential cluster.

The following cohort × stage scores precede the pooled estimates. Parentheses are game counts; every cell includes both agent colors. Early ≤8 plies, midgame 9–23, late ≥24. Random late positions are absent by design.


| Comparison | Random early | Random midgame | Agent early | Agent midgame | Agent late |
| --- | --- | --- | --- | --- | --- |
| N8 vs N6 | 47.9% (24) | 48.8% (40) | 56.2% (16) | 59.4% (32) | 50.0% (16) |
| N10 vs N8 | 60.4% (24) | 50.0% (40) | 59.4% (16) | 56.2% (32) | 50.0% (16) |
| N10 vs N6 | 50.0% (12) | 52.5% (20) | 50.0% (8) | 68.8% (16) | 50.0% (8) |
| N8 vs MCTS 2,000 | 75.0% (12) | 42.5% (20) | 81.2% (8) | 50.0% (16) | 50.0% (8) |
| N10 vs MCTS 2,000 | 83.3% (12) | 45.0% (20) | 62.5% (8) | 56.2% (16) | 50.0% (8) |
| N10 vs MCTS 5,000 | 58.3% (12) | 47.5% (20) | 75.0% (8) | 40.6% (16) | 50.0% (8) |


Cohort-specific 95% trajectory-cluster intervals are exploratory; the agent cohort has only eight primary or four secondary independent families.


| Comparison | Random score / 95% interval | Agent score / 95% interval |
| --- | --- | --- |
| N8 vs N6 | 48.4% / 40.6–56.2% | 56.2% / 49.2–63.3% |
| N10 vs N8 | 53.9% / 46.9–60.9% | 55.5% / 48.4–63.3% |
| N10 vs N6 | 51.6% / 46.9–56.2% | 59.4% / 43.8–75.0% |
| N8 vs MCTS 2,000 | 54.7% / 39.1–68.8% | 57.8% / 48.4–65.6% |
| N10 vs MCTS 2,000 | 59.4% / 48.4–71.9% | 56.2% / 46.9–65.6% |
| N10 vs MCTS 5,000 | 51.6% / 45.3–59.4% | 51.6% / 40.6–62.5% |


Tactical labels indicate an immediate own win or at least one move permitting an immediate opponent win; quiet is the complement. These tags overlap chronological stages.


| Comparison | Quiet score (games) | Tactical score (games) |
| --- | --- | --- |
| N8 vs N6 | 53.7% (68) | 50.8% (60) |
| N10 vs N8 | 57.4% (68) | 51.7% (60) |
| N10 vs N6 | 58.3% (30) | 52.9% (34) |
| N8 vs MCTS 2,000 | 61.7% (30) | 51.5% (34) |
| N10 vs MCTS 2,000 | 65.0% (30) | 51.5% (34) |
| N10 vs MCTS 5,000 | 55.0% (30) | 48.5% (34) |


N8 vs N6 changes from 48.4% on random boards to 56.2% on agent boards; it is not a distribution-independent improvement. N10 vs MCTS 2,000 scores 83.3% on twelve random early games but 45.0% on twenty random midgame games. Such small subgroup contrasts are descriptive, not separate discoveries. All eight primary agent late positions are tactical and every late color pair scores 0.5 in every comparison. This cohort has little discriminatory information; its degenerate bootstrap interval must not be mistaken for certainty about broader endgames. Quiet late fixtures are present in performance profiling, not the strength pool. Subgroup intervals and every pair score are preserved in [analysis.json](analysis.json).

## Pooled strength and uncertainty

Wins/draws/losses and scores below refer to the named Negamax challenger. Draws score 0.5. Red is player 0; yellow is player 1. Both colors of an opening are averaged first. Resample trajectory families within cohort, preserving all related stages and shared-comparison observations, using 10,000 fixed-seed draws. The primary comparisons each have **64 openings but only 40 clusters** (32 random + eight trajectories); secondary comparisons have 32 openings and 20 clusters. Treating 128/64 games or 64/32 snapshots as independent would overstate information. Equal-cohort pooling estimates this declared mixture, not an arbitrary tournament population.


| Comparison | Pairs | W / D / L | Score | 95% cluster interval | Primary 97.5% interval | Red / yellow |
| --- | --- | --- | --- | --- | --- | --- |
| N8 vs N6 | 64 | 59 / 16 / 53 | 52.3% | 47.3–57.4% | 46.5–58.2% | 47.7% / 57.0% |
| N10 vs N8 | 64 | 65 / 10 / 53 | 54.7% | 49.6–59.8% | 48.8–60.5% | 51.6% / 57.8% |
| N10 vs N6 | 32 | 33 / 5 / 26 | 55.5% | 46.9–64.1% | exploratory | 57.8% / 53.1% |
| N8 vs MCTS 2,000 | 32 | 34 / 4 / 26 | 56.2% | 47.7–64.8% | exploratory | 57.8% / 54.7% |
| N10 vs MCTS 2,000 | 32 | 35 / 4 / 25 | 57.8% | 50.0–65.6% | exploratory | 56.2% / 59.4% |
| N10 vs MCTS 5,000 | 32 | 32 / 2 / 30 | 51.6% | 45.3–57.8% | exploratory | 51.6% / 51.6% |


Bonferroni adjustment uses 97.5% intervals for the two predeclared primary claims. Their lower limits, 46.5% and 48.8%, do not exceed 50%. Neither nominal 95% primary interval excludes parity either. No secondary matchup establishes superiority; N10 vs MCTS 2,000 has a nominal lower limit exactly 50%, not above it. Positive point estimates should not be promoted to proven rankings.

Percentile intervals are approximate and can under-cover with few families or constant outcomes. Supplementary conservative bounded-variable 97.5% intervals are **23.1–81.6%** for N8 vs N6 and **25.4–83.9%** for N10 vs N8. They are intentionally wide and depend on independent-family sampling; uniqueness conditioning and trajectory survival still limit interpretation. Neither method supports a primary improvement claim. Secondary/subgroup intervals have no simultaneous discovery guarantee.

The paired N10-minus-N8 score difference against shared MCTS 2,000 openings is **+1.56 percentage points**, 95% interval **−1.56 to +5.47**. Changing the shared N10 opponent from MCTS 2,000 to 5,000 changes its score by **−6.25 points**, interval **−15.62 to +3.12**. These contrasts preserve both color and trajectory dependence and are exploratory; shared-opponent differences are not substitute head-to-head matches. One independently seeded MCTS stream pair per opening mixes seed and position variability. Reused role seeds permit pairing but divergent games consume different RNG trajectories; seed variance is not separately estimated. Deterministic Negamax depth-only comparisons contain no agent RNG variance. No Elo or universal strength estimates are made.

## Search latency, growth and memory

The fixed performance set has eleven positions: ten base fixtures plus the explicitly declared extra quiet midgame, for **44 depth/position conditions** at 4/6/8/10. Each condition uses one discarded warm-up and three rotating-depth, fresh-table production samples. Profile and allocation runs are separate and not latency samples. The supplement was selected before main by a fixed board-label rule and retains its own manifest. Every diagnostic move, complete root-score vector and nodes/TT/cutoff vector matches its corresponding uninstrumented run.

Median elapsed milliseconds by fixture; maximum is the largest of its three depth-10 samples. Full samples and CPU times are in [profiling.json](profiling.json) and [supplement profiling](profiling-supplement/profiling.json). [performance.csv](performance.csv) contains all 44 rows of medians, maxima, CPU, nodes, TT entries/hits, cutoffs, leaf evaluations, retained/peak bytes, nodes/s and chosen move.


| Position | D4 ms | D6 ms | D8 ms | D10 ms | D10 max ms | D10 / D6 |
| --- | --- | --- | --- | --- | --- | --- |
| empty | 2.302 | 17.531 | 100.398 | 411.585 | 415.822 | 23.48× |
| near-opening | 2.122 | 14.469 | 78.286 | 300.857 | 303.196 | 20.79× |
| midgame | 0.972 | 2.944 | 6.522 | 12.046 | 12.048 | 4.09× |
| midgame-wide | 2.588 | 9.172 | 21.490 | 45.432 | 45.754 | 4.95× |
| late | 0.113 | 0.309 | 0.633 | 1.080 | 1.087 | 3.50× |
| immediate-win | 1.033 | 6.753 | 40.361 | 182.799 | 185.353 | 27.07× |
| forced-reply | 1.234 | 4.986 | 26.363 | 89.844 | 109.992 | 18.02× |
| double-threat-loss | 0.140 | 0.141 | 0.139 | 0.140 | 0.140 | 0.99× |
| dense-endgame | 0.013 | 0.014 | 0.015 | 0.014 | 0.017 | 1.04× |
| quiet-5 | 1.241 | 6.021 | 15.552 | 21.997 | 22.307 | 3.65× |
| supplement-main-003 | 2.711 | 19.001 | 70.839 | 235.447 | 248.459 | 12.39× |


Depth-10 diagnostics below show production median CPU and separately measured TT/leaf statistics. TT bytes sum unique reachable Python objects; traced peak also includes temporary Python allocations. Neither is process RSS or a hard memory ceiling.


| Position | CPU ms | Nodes | TT entries / hits | Cutoffs / leaves | TT / peak MiB | Nodes/s |
| --- | --- | --- | --- | --- | --- | --- |
| empty | 411.282 | 128308 | 38395 / 15828 | 27477 / 68511 | 8.259 / 8.414 | 311,741 |
| near-opening | 300.339 | 99441 | 28999 / 14340 | 22095 / 49039 | 6.746 / 6.868 | 330,526 |
| midgame | 12.046 | 5389 | 1881 / 700 | 993 / 1543 | 0.437 / 0.447 | 447,376 |
| midgame-wide | 45.422 | 19097 | 6730 / 2759 | 5110 / 5850 | 1.632 / 1.655 | 420,341 |
| late | 1.080 | 506 | 328 / 22 | 166 / 78 | 0.070 / 0.074 | 468,428 |
| immediate-win | 182.721 | 59934 | 18565 / 8028 | 14178 / 30391 | 3.998 / 4.084 | 327,868 |
| forced-reply | 89.834 | 30407 | 10297 / 3522 | 7563 / 13904 | 2.200 / 2.259 | 338,442 |
| double-threat-loss | 0.140 | 81 | 38 / 0 | 32 / 0 | 0.010 / 0.013 | 578,572 |
| dense-endgame | 0.014 | 4 | 3 / 0 | 0 / 0 | 0.001 / 0.004 | 281,531 |
| quiet-5 | 21.994 | 10140 | 4174 / 1539 | 3124 / 2152 | 0.984 / 1.001 | 460,975 |
| supplement-main-003 | 235.305 | 87810 | 28395 / 16340 | 22427 / 36833 | 6.681 / 6.736 | 372,950 |


From depth 6 to 10, empty-board latency grows 23.5× and near-opening 20.8×. Quiet midgames vary: midgame-wide grows 5.0×, the supplementary 14-ply agent board 12.4×, and the 23-ply quiet fixture 3.7×. Depth 8→10 costs 4.10× on empty, 3.84× near-opening, and 2.11–3.32× on two quiet midgames. This added computation accompanies small, uncertain strength gains, rather than an established payoff per unit compute.

Full-window values for every root move matter: an immediate winning move does not let Negamax skip all alternative root searches. The immediate-win fixture still costs 182.8 ms at depth 10, while MCTS's win guard can return immediately. The double immediate-opponent-threat loss is the opposite: all horizons visit just 81 nodes and no heuristic leaves, so depth 10 remains about 0.14 ms. The dense endgame has four nodes at every tested depth. Losses become expensive when the search must explore many alternatives before discovering a distant forced result or reaches heuristic leaves without resolution, not merely because the score is losing. The measured simple forced loss is cheap; this study does not establish a worst-case deep forced-loss bound.

Different depths legitimately change scores and choices: empty chooses column 3 at 4/6/8 and column 2 at 10. Equality assertions apply within a fixed depth across measurement methods, never across different horizons.

Real-game move costs are descriptive and visit different positions at different depths; they are not matched causal scaling estimates or endpoint percentiles.


| Agent | Calls | Median / p95 / max ms | Wall / CPU total s | Max nodes / TT entries |
| --- | --- | --- | --- | --- |
| m2000 | 998 | 12.97 / 22.17 / 35.57 | 12.06 / 12.03 | — |
| m5000 | 470 | 32.35 / 57.29 / 69.75 | 14.21 / 14.19 | — |
| n10 | 2665 | 7.39 / 444.03 / 1413.43 | 227.75 / 228.53 | 423857 / 119958 |
| n6 | 1705 | 1.77 / 23.49 / 43.86 | 10.06 / 10.07 | 13032 / 2931 |
| n8 | 2837 | 4.44 / 119.21 / 278.99 | 74.11 / 75.38 | 79711 / 20755 |


There were 8,675 main search calls and 338.19 seconds inside choose_move. Process CPU counters cover all process threads and can slightly exceed elapsed intervals; they are not exclusive per-thread instruction time.

The maximum main depth-10 nodes and maximum wall time select the same quiet five-ply position, history `[1,4,6,0,6]`, with 423,857 nodes, 119,958 entries, 47,944 hits and 93,664 search-loop cutoffs. A **post hoc, result-excluded** diagnostic replay matches those counters and the move, and measures **27.61 MiB retained TT / 27.86 MiB traced peak**. All root values are negative heuristic values (best −4), not a proof of forced loss. This tail position was selected by cost alone and is excluded from predeclared fixture aggregates. See [tail-diagnostic.json](tail-diagnostic.json).

Live allocation sites in that tail are dominated by TT dictionary/entry insertion (12.32 MiB), key tuples (8.24 MiB), and the retained bitboard integers created by play/undo (about 6.15 MiB). Reachable TT storage is roughly 241 bytes per entry. The TT is unbounded **within one decision**, shared across root moves, then released by production. Diagnostic retention deliberately holds a table reference; it does not demonstrate a production leak. A larger unmeasured tree can exceed all observed memory maxima.

## Runtime attribution

cProfile percentages below are disjoint inclusive heuristic, terminal, ordering and play/undo subtrees at depth 10. The residual combines the recursive body, TT/bound work, root work and instrumentation glue; it must not be labeled pure recursion. Builtin work inside a subtree is included. Profiled absolute seconds are not ordinary latency.


| Position | 69-window heuristic | Terminal detection | Legal / ordering | Play / undo | TT + recursive residual |
| --- | --- | --- | --- | --- | --- |
| empty | 65.7% | 10.7% | 5.8% | 5.7% | 12.1% |
| near-opening | 59.4% | 12.4% | 7.7% | 6.6% | 13.9% |
| midgame | 34.2% | 19.7% | 15.7% | 10.0% | 20.4% |
| midgame-wide | 37.1% | 17.8% | 16.5% | 9.3% | 19.4% |
| late | 12.7% | 22.0% | 28.7% | 10.9% | 25.7% |
| immediate-win | 61.7% | 11.8% | 7.0% | 6.2% | 13.2% |
| forced-reply | 58.1% | 12.6% | 8.5% | 6.7% | 14.0% |
| double-threat-loss | 0.0% | 21.0% | 40.6% | 11.8% | 26.6% |
| dense-endgame | 0.0% | 14.5% | 20.4% | 6.5% | 58.6% |
| quiet-5 | 22.0% | 20.6% | 23.5% | 10.8% | 23.1% |
| supplement-main-003 | 44.8% | 16.8% | 11.5% | 8.6% | 18.4% |


Separate exclusive line-event tracing at depth 6 provides approximate TT/bound and recursive-body attribution. It changes relative costs, especially the 69-window Python loop, and is not a claim that these percentages apply to uninstrumented depth 10. TT/bounds includes key/lookup/insertion and bound/cutoff bookkeeping. Raw per-line data identify exactly what was charged.


| Position | Heuristic | Terminal | Legal / ordering | TT / bounds | Play / undo | Recursive / other |
| --- | --- | --- | --- | --- | --- | --- |
| empty | 75.1% | 10.8% | 2.0% | 1.9% | 4.2% | 5.9% |
| near-opening | 70.2% | 12.1% | 4.2% | 2.2% | 4.7% | 6.5% |
| midgame | 56.4% | 17.7% | 7.3% | 2.7% | 6.6% | 9.2% |
| midgame-wide | 63.9% | 14.6% | 5.9% | 2.4% | 5.5% | 7.7% |
| late | 47.1% | 18.6% | 12.0% | 3.7% | 6.7% | 11.9% |
| immediate-win | 70.5% | 11.9% | 4.3% | 2.2% | 4.6% | 6.6% |
| forced-reply | 67.8% | 12.5% | 5.7% | 2.1% | 4.9% | 6.9% |
| double-threat-loss | 0.0% | 28.2% | 36.6% | 5.0% | 11.5% | 18.7% |
| dense-endgame | 0.0% | 21.2% | 22.9% | 3.8% | 6.4% | 45.7% |
| quiet-5 | 57.4% | 17.0% | 7.1% | 2.9% | 6.5% | 9.1% |
| supplement-main-003 | 65.9% | 14.1% | 4.5% | 2.6% | 5.3% | 7.5% |


Early broad-tree depth-10 cProfile time is 59–66% heuristic; the quiet cost-tail replay is 64.8% heuristic. Terminal checks are about 11–12% on those fixtures. Some midgames spend only 22–45% in the heuristic, with ordering/terminal/state work larger, and resolved loss/endgame trees have zero heuristic leaves. Therefore an incremental evaluator targets the dominant broad-tree work, but cannot promise the same gain on tactical/endgame positions. TT direct CPU attribution is modest in the line diagnostic, despite its dominant memory cost. Python recursive and allocation overhead remain relevant if leaf scoring becomes faster.

## Optimization feasibility and proposed Phase 3B

The following ranking is a measured-baseline implementation order, not predicted numerical speedups. Preserve exact depth-sensitive terminals, the same 69-window heuristic, full-window exact values for every root move, typed EXACT/LOWER/UPPER bounds keyed correctly by depth and mover, and stable center-first root ties. Public caps/presets remain unchanged.


| Rank / technique | Expected benefit and relevance | Difficulty | Correctness risk |
| --- | --- | --- | --- |
| 1 — A: incremental windows | High CPU opportunity on broad quiet trees; less on resolved tactics | Medium | Medium: cell/window updates, mover sign, rollback |
| 2 — B: compact TT keys/entries | High memory benefit; CPU benefit unmeasured | Medium | Medium: injective identity, depth/mover and full score width |
| 3 — C: bounded TT replacement | Predictable peak memory; CPU may regress through misses | Medium | Low–medium with full-key collision checks and exact completion |
| 4 — D: cross-iteration move hints | Potentially reduce nodes; current same-depth hints have modest prior gains | Medium | Low if hints alone; high if cross-depth values are reused |
| 5 — E: iterative deepening | Enables better hints and predictable checkpoints; adds work | Medium | Medium: final horizon and all-root-score completion |
| 6 — F: PVS / aspiration | Potentially reduce interior work after ordering improves | High | High: re-search failed windows to recover every exact root value |
| 7 — G: conservative tactical pruning | Potential tactical reduction; simple loss fixture already cheap | High | High: horizon and terminal-distance semantics |
| 8 — H: optional native kernel | Potentially large CPU/GIL benefit across loops | High | High: independent parity, ABI, score width and cancellation |


**Phase 3B.1:** implement incremental scoring as an isolated ablation. Precompute cell-to-window membership; maintain exact per-window red/yellow counts and the existing 1/3/9/81 contribution, reversing only mover perspective. Update only windows touching a dropped/undone cell. Compare against an independent array evaluator over random legal paths and complete undo paths; include exception restoration, terminals-before-leaves and every root score at tractable depths. Rebenchmark all declared positions and the separately labeled cost tail with full score/counter checks; measure incremental update overhead on cheap terminal-heavy trees. Enable it only if correctness holds and the mixed-position cost result warrants it.

**Phase 3B.2:** separately compact TT representation with collision-free full identity, explicit remaining depth/mover semantics, typed bounds and sufficient signed score width for ±(1,000,000 + depth). Keep fresh per-decision scope. Compare retained/peak memory and timing without combining this with new search rules. Then add an optional deterministic entry budget/replacement ablation. Replacement may create a cache miss, never a false hit or partial root value. Eviction should continue exact search rather than aborting with unknown results. Measure increased nodes as well as memory.

**Phase 3B.3:** prototype move-only cross-iteration hints and iterative deepening together behind an internal switch, while reporting their separate costs. Store hints independently of value entries; never import another depth's score or bound. Every final-depth root move must still obtain a full exact value with historical tie-breaking. Only after these baselines should PVS/aspiration be considered, with explicit re-searches of failed windows. Keep tactical pruning and native migration as later independently tested projects.

Victor provides reusable geometry and implementation patterns in [exact.py](../../../games/connect4/victor/exact.py), [native_search.c](../../../games/connect4/victor/native_search.c), and [native.py](../../../games/connect4/victor/native.py): playable-bit arithmetic, winning-square geometry, full-key collision checking, a compact fixed-size replacement table, explicit local search state and optional ctypes loading. Its existing native entry holds WDL-sized int8 bounds; depth-sensitive heuristic Negamax needs wider scores and explicit horizon identity. The Python Victor table cap terminates proof search; that behavior is unsuitable for a contract requiring all exact root values. Mirror keys require root-move remapping and tie tests. Victor's WDL-only forced moves, endgame draws and Claimeven bounds cannot be copied into this heuristic/horizon search without new proofs. Reuse its ABI/build/fallback pattern rather than treating its terminal-only kernel as a drop-in Negamax evaluator. No speculative optimization or native Negamax kernel was implemented. The complete backend suite runs its existing Victor native-build fixtures in the laptop checkout; these rebuild the existing ignored, source-keyed Victor library, not a new Negamax kernel or production artifact.

## Validation and limitations

[audit.json](audit.json) independently replays **608 saved games** (512 main + 96 preflight), covering **10,300 searched moves**, and passes all **12 deterministic game repeats** (both colors, all six comparisons, first main opening). Repeated outcomes, histories, search counters and MCTS RNG fingerprints match exactly; repetitions are excluded from strength. Source hashes still match declaration. Resume is exercised in real evidence and tests; corrupted/foreign/duplicate rows, wrong histories/winners/roles/times/counters/seeds, source drift, errors and an in-search deadline are tested. The deadline produces an explicit unscored incomplete attempt. Deep 6/8/10 near-endgame root vectors are checked against an independent unpruned array oracle; all existing Negamax bound/heuristic/tie contracts remain covered.

The full backend suite passes **1,854 tests**, with **15 skips because PyTorch is unavailable**, in 89.76 seconds; see [backend-tests.txt](backend-tests.txt). This includes 18 new evaluation/profiling/inference/deep-oracle tests and all existing engine, Negamax, MCTS, API and integration checks. The initial run passed 1,853 tests but failed the existing Victor default-off test because create_app reloaded the laptop's `.env` after the test removed its flag; [the initial log](backend-tests-initial.txt) is preserved. The isolated failing test passes with `PYTHON_DOTENV_DISABLED=1`, and the complete suite then passes with that setting. Neither application code nor local configuration was changed. Python compilation, deterministic analysis reproduction, source/evidence checks and final diff whitespace validation pass. No frontend code or behavior changed, so frontend/browser suites were not rerun. No OpenAI requests, paid services, application changes, public cap changes, production services, Docker/container operations, Tailscale/Render configuration, training/model artifacts, canonical evidence, merge or deployment occur.

Limitations: modest numbers of independent agent families; one MCTS seed pair per opening; survival/uniqueness conditioning; unequal cohort stage mixtures; no random late or quiet agent late strength positions; approximate bootstrap coverage with degenerate late pairs; only three latency samples; instrumentation perturbs attribution; no worst-case search/memory bound, traffic/concurrency stress, endpoint latency or Mac Mini measurements. A hard process kill can lose time since the latest checkpoint or damage a trailing JSONL; resume fails closed rather than silently dropping records. The frozen compute cap was not reached, so no runtime-selected incomplete sample affects this main analysis.

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) supplies local commands and artifact definitions, including PYTHONHASHSEED=0 for reproducible shared-opening contrast bootstrap order. New scripts and evidence are confined to this checkout and feature branch. Prior Phase 1/2/2B evidence is unchanged. The final Git commit identifies the complete hashed baseline and report.
