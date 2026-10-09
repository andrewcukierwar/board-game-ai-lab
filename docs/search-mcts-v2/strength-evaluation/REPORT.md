# Optimized Connect Four MCTS: strength and simulation budgets

October 9, 2026. Laptop development evaluation on `search/negamax-v2`, starting
at [fadd61cad27f92673cfeef85164c27cca6cce21a](https://github.com/andrewcukierwar/board-game-ai-lab/commit/fadd61cad27f92673cfeef85164c27cca6cce21a).
The MCTS, bitboard, and Negamax agents are unchanged. No production presets,
deployment configuration, canonical evidence, or prior Phase 1/2 artifacts changed.

The declared experiment completed **1,664 games over 128 primary opening/seed pairs** per matchup, with no errors or incomplete matches. Budgets of **2,000, 5,000, and 10,000 show improvement over optimized 400** after the primary four-comparison adjustment. The 800 result is suggestive at the nominal 95% level but inconclusive after that adjustment. **10,000 is not clearly better than 5,000** on the paired shared-opponent contrast. No strength inference uses the preflight or deterministic audit repeats.

## Strength results

Each primary row contains 256 games and 128 clusters. Scores refer to the challenger; red is player 0 and yellow player 1. Intervals resample opening pairs, not individual games.

| Simulations vs MCTS 400 | W / D / L | Score | 95% interval | 98.75% interval | Red / yellow score |
| --- | --- | ---: | --- | --- | --- |
| 800 | 130 / 19 / 107 | 54.5% | 50.2–58.8% | 49.0–60.0% | 58.6% / 50.4% |
| 2,000 | 148 / 16 / 92 | 60.9% | 56.8–65.0% | 55.7–66.0% | 66.0% / 55.9% |
| 5,000 | 163 / 11 / 82 | 65.8% | 61.7–69.9% | 60.7–70.9% | 68.0% / 63.7% |
| 10,000 | 175 / 10 / 71 | 70.3% | 66.4–74.0% | 65.2–75.2% | 70.3% / 70.3% |

Mean challenger-minus-opponent pair advantages are +9.0, +21.9, +31.6, and +40.6 percentage points respectively. The first adjusted score interval crosses 50%; the other three remain above it. Even at 10,000, MCTS loses 71 of 256 games to 400. Increased compute improves aggregate play on this distribution, not every move or opening.

Each Negamax row below has only **64 games / 32 independent opening/seed clusters**, balanced over the same eight lengths. These intervals are unadjusted exploratory estimates.

| MCTS simulations | Opponent | W / D / L | Score | 95% interval | Red / yellow score |
| ---: | --- | --- | ---: | --- | --- |
| 400 | Negamax 4 | 24 / 1 / 39 | 38.3% | 28.1–48.4% | 39.1% / 37.5% |
| 800 | Negamax 4 | 28 / 4 / 32 | 46.9% | 37.5–56.2% | 48.4% / 45.3% |
| 2,000 | Negamax 4 | 30 / 5 / 29 | 50.8% | 45.3–56.2% | 48.4% / 53.1% |
| 5,000 | Negamax 4 | 35 / 3 / 26 | 57.0% | 50.0–63.3% | 51.6% / 62.5% |
| 10,000 | Negamax 4 | 36 / 5 / 23 | 60.2% | 53.9–66.4% | 57.8% / 62.5% |
| 400 | Negamax 6 | 24 / 4 / 36 | 40.6% | 34.4–46.9% | 39.1% / 42.2% |
| 800 | Negamax 6 | 22 / 5 / 37 | 38.3% | 30.5–46.1% | 34.4% / 42.2% |
| 2,000 | Negamax 6 | 28 / 4 / 32 | 46.9% | 39.1–54.7% | 48.4% / 45.3% |
| 5,000 | Negamax 6 | 32 / 5 / 27 | 53.9% | 46.9–60.2% | 46.9% / 60.9% |
| 10,000 | Negamax 6 | 30 / 8 / 26 | 53.1% | 46.1–60.2% | 48.4% / 57.8% |

MCTS 400 scores below parity against both Negamax depths at nominal 95%. MCTS 800 also scores below parity against depth 6 and actually scores lower than 400 there (38.3% versus 40.6%); this is not proof that 800 is weaker. MCTS 10,000 exceeds parity against depth 4 at nominal 95%, but neither 5,000 nor 10,000 clearly beats depth 6. The depth-6 score falls slightly from 53.9% to 53.1% at the last budget step. These smaller, multiple secondary comparisons cannot establish a universal ranking.

Paired changes against the common MCTS 400 opponent:

| Budget change | Score difference, percentage points | 95% paired interval, percentage points |
| --- | ---: | --- |
| 800 → 2,000 | +6.4 | +1.4 to +11.7 |
| 800 → 5,000 | +11.3 | +6.4 to +16.2 |
| 800 → 10,000 | +15.8 | +10.9 to +20.9 |
| 2,000 → 5,000 | +4.9 | +1.0 to +8.8 |
| 2,000 → 10,000 | +9.4 | +4.3 to +14.6 |
| 5,000 → 10,000 | +4.5 | -0.4 to +9.4 |

These six contrasts are exploratory, with no simultaneous significance claim. The 5,000 → 10,000 contrast is +4.5 points with a −0.4 to +9.4 interval: doubling compute has an uncertain marginal gain. The 2,000 → 5,000 contrast is +4.9 points with a +1.0 to +8.8 nominal interval at 2.5× the simulation budget. The strongest supported economical step is 400 → 2,000: +10.9 points against 400, with about five times the search compute. This is a practical tradeoff, not an optimized universal budget or an Elo estimate.

## Runtime, memory, and scaling

The preflight completed 112 games in **18.73 s**, projected **286.31 s**, and the full run took **455.09 s** including the 28-game checkpoint invocation. The realized cost is 1.59× the coarse projection and uses 25.3% of the fixed 1,800 s main cap. Search calls total **27,814**, with **452.31 s wall / 448.64 s CPU** in choose_move. No sample or compute cap was enlarged.

Matched fresh-tree fixture timings below are medians of seven uninstrumented samples. Traced peak memory comes from a separate seed/run. Guard bypasses are not assigned fictional simulation throughput.

| Position | Simulations | Wall / CPU ms | Simulations/s | Tree nodes | Traced peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| empty | 400 | 4.927 / 4.923 | 81,191 | 401 | 0.207 |
| empty | 2,000 | 25.024 / 25.009 | 79,923 | 2,001 | 1.015 |
| empty | 5,000 | 63.640 / 63.623 | 78,567 | 5,001 | 2.550 |
| empty | 10,000 | 136.302 / 136.254 | 73,367 | 10,001 | 5.091 |
| near-opening | 400 | 3.932 / 3.932 | 101,724 | 393 | 0.201 |
| near-opening | 2,000 | 20.680 / 20.648 | 96,711 | 1,883 | 0.952 |
| near-opening | 5,000 | 53.485 / 53.441 | 93,483 | 4,786 | 2.431 |
| near-opening | 10,000 | 117.049 / 117.008 | 85,435 | 9,657 | 4.898 |
| late | 400 | 2.448 / 2.448 | 163,421 | 360 | 0.184 |
| late | 2,000 | 9.798 / 9.797 | 204,115 | 837 | 0.422 |
| late | 5,000 | 21.731 / 21.731 | 230,086 | 1,051 | 0.530 |
| late | 10,000 | 40.994 / 40.991 | 243,935 | 1,181 | 0.596 |

All **76 profiling conditions** complete; three random positions bypass search at every budget and remain included in raw evidence. On the sixteen non-bypassed positions, maximum sampled latency is 13.13 / 36.09 / 85.82 / 140.23 ms at the four budgets. Maximum traced peaks are 0.209 / 1.024 / 2.550 / 5.095 MiB. These are descriptive extrema, not production tail bounds.

For 25× more simulations, empty latency grows 27.67× and near-opening 29.77×: time per simulation rises **10.7% and 19.1%**. Selection calls per simulation grow from 2.07 to 4.00 and 2.02 to 3.93. Average profiled selection-call cost does not grow (2.60 → 2.46 μs and 2.61 → 2.44 μs). Thus the measured nonlinear cost is mainly longer selection paths, not a scan proportional to the whole tree. Node and memory growth on these broad trees remains approximately linear, with at most one new node per simulation.

Late-board latency grows only 16.75× for 25× simulations, because the tree approaches its finite useful size (360 → 1,181 nodes), and terminal revisits replace rollouts. Selection calls rise from 4.72 to 7.48 per simulation. This produces a different scaling curve; do not assume a single simulations/second conversion for all positions.

Real priority-game challenger move times include tactical bypasses and many correlated moves per game:

| Simulations | Calls | Mean / median ms | Observed p95 / maximum ms | Total wall s |
| ---: | ---: | ---: | ---: | ---: |
| 800 | 2,130 | 5.70 / 5.99 | 9.51 / 28.04 | 12.15 |
| 2,000 | 2,096 | 13.80 / 15.01 | 23.85 / 63.73 | 28.92 |
| 5,000 | 2,080 | 35.30 / 39.00 | 60.64 / 161.86 | 73.42 |
| 10,000 | 2,000 | 72.21 / 81.34 | 123.99 / 264.38 | 144.43 |

The game maximum exceeds fixture maxima: 161.86 ms at 5,000 and 264.38 ms at 10,000. The 400 opponent also has an isolated 93.55 ms outlier during the 10,000 condition. Host scheduling/GC effects are not separated by these measurements. Observed quantiles are descriptive, contain opening/move dependence, and are not confidence bounds or endpoint latency guarantees. Different budgets also visit different game positions; use matched fixtures to compare pure runtime scaling.

## Bottlenecks and measured loop ablation

| Position | Budget | Selection % | Rollout % | Expansion components % | Backpropagation % |
| --- | ---: | ---: | ---: | ---: | ---: |
| empty | 400 | 15.0 | 63.4 | 9.8 | 1.3 |
| empty | 10,000 | 24.8 | 54.0 | 8.7 | 1.6 |
| near-opening | 400 | 18.1 | 56.3 | 11.6 | 1.5 |
| near-opening | 10,000 | 28.4 | 46.9 | 10.3 | 1.8 |
| late | 400 | 35.5 | 26.9 | 13.7 | 3.0 |
| late | 10,000 | 61.8 | 6.5 | 2.3 | 4.5 |

Random rollouts still dominate broad early trees (46.9–54.0% at 10,000); selection reaches 24.8–28.4%. In the late fixture selection reaches 61.8% and rollout only 6.5%. The measured remaining opportunities are reducing work per selection step and per rollout move while preserving UCT arithmetic and the random policy. Rollout four detection, RNG calls, and legal-list work remain in full caller profiles. They are targets for future separately measured implementations, not proven new speedups.

The empty 10,000 tree retains about 5.09 MiB. Largest live allocation groups include legal-move lists (~1.08 MiB), nodes (~0.92 MiB), compact snapshots (~0.92 MiB), empty child dictionaries (~0.61 MiB), populated child dictionaries (~0.58 MiB), and bitboard tuples/integers. These describe retained blocks rather than allocation churn. RSS grows by about 4.75 MiB on empty at 10,000. There is no evidence here of disproportionate tree-memory growth, but cyclic GC and long-service behavior need endpoint measurements.

[loop-ablation.json](loop-ablation.json) measures a low-risk prototype caching math.sqrt/RNG choice in selection and bitboard constants/four detection in rollout locals. The candidate source, eleven predetermined existing profile positions, four budgets, seven interleaved seeds, warm-up, and acceptance thresholds are written before observations. All **44 conditions** match selected moves and final RNG states at all timed seeds; complete diagnostic tree fingerprints match at seed 17001.

The prototype has median speedup **0.9948×** (about 0.52% slower), improves only **34.1%** of conditions, and ranges from 0.882× to 1.183×. It fails the predeclared criteria: at least 5% aggregate latency reduction, improvement in at least 75% of conditions, and no greater than 5% condition regression. **Reject the prototype; production agents remain unchanged.** Favorable isolated rows do not justify it. No speculative loop optimization, tactical rollout policy, compiled kernel, or UCT semantic change is implemented.

## Design and prior-experiment review

The previous [Phase 2 report](../REPORT.md) establishes a large throughput gain
and equal-budget seeded behavior, but its strength comparison has only eight
opening/seed clusters, sixteen dependent color games per condition, one RNG pair
per opening, reused seeds/positions across conditions, and an immediate-win
opening. The 71.9% higher-budget score is descriptive, with no opening-cluster
interval, broad budget curve, or Negamax comparison. It cannot support an Elo
claim or a production strength recommendation. Increasing simulations changes
RNG consumption and can lose games; equal-budget control is not evidence that
higher-budget play improves uniformly.

Existing tournament and season code provides deterministic seed derivation,
color-balanced schedules, compact confirmed histories, replay validation,
standings, and resumable UI state. It executes through public APIs with capped
budgets, uses empty-board games, and the season interval bootstraps individual
games. Its sequential Elo and draw-advancement tiebreaks are unsuitable for this
paired-opening experiment. We reuse the local fixture builder and production
agents, and leave the UI and frozen canonical season untouched.

[DESIGN.md](DESIGN.md) and [experiment.json](experiment.json) fix the experiment
before measurements. Four priority matchups each use the same 128 distinct
opening/seed pairs, both agent colors. Five MCTS budgets each face Negamax 4
and 6 on the first 32 openings (balanced across lengths). The main schedule has
1,664 games: 1,024 primary and 640 exploratory. No opening is selected by result,
latency, or speedup, and the sample is not enlarged after seeing outcomes.

Uniform legal-column histories are drawn at 2/5/8/11/14/17/20/23 plies, sixteen
per length, rejecting only terminal or duplicate histories. All 128 main histories
also have distinct board arrays in this realized sample. Eighteen include a full
column; 26 have an immediate winning move for the current player. Keep these
tactical positions: removing them would change the declared estimand. Random
nonterminal long histories condition on survival and differ from positions
reached by competent agents. This is a balanced artificial position mixture,
not empty-board, human-opponent, solved-game, or general tournament strength.

Domain-separated 128-bit challenger/opponent seeds create independent Random
objects per game, with streams persisting through that game. Role seeds follow
the agent across color swaps and are reused across budgets/opponents within an
opening. Color games and condition results are therefore dependent and are never
counted as independent samples. Negamax uses deterministic ties and no RNG;
its assigned unused seed is recorded. A larger budget consumes more randomness;
pairing does not imply synchronized rollout trajectories after games diverge.
Public seeded sessions use per-ply initialization instead of persistent streams.

The four separately seeded preflight pairs are outside inference. One short
history `[0, 5]` coincides with a main history, but its agent seeds differ and
none of its observations enter the main estimates. Documentation clarifies
"excluded pairs" rather than implying disjoint board histories; the declaration,
schedule, seeds, and sample sizes are unchanged.

## Reproducibility and inference

[results.jsonl](results.jsonl) contains all main observations and binds each row
to the manifest/source hashes: full opening and move histories, configurations,
agent source commit, role seeds, colors, winner, score, per-move wall/CPU timings,
simulation accounting, search totals, and final RNG fingerprints. Guard-based
simulation accounting is outside timed choose_move; separate profiling captures
actual root visits. Errors and interruptions are separate incomplete attempts,
never scored losses. Game-boundary resume validates source/runtime and replays
saved results before skipping them. A real 28-game checkpoint is preserved in
[checkpoint-status.json](checkpoint-status.json).

[analysis.json](analysis.json) uses 10,000 deterministic stratified bootstrap
replicates. Average both color scores per opening, then resample entire opening
pairs within fixed length strata. Draws score 0.5. The per-opening advantage is
challenger score minus opponent score, `2 * pair_score - 1`; all values are saved.
Cross-budget differences resample jointly on shared openings and measure score
against the same opponent, not direct higher-budget head-to-head strength.

Report approximate 95% percentile intervals and a 98.75% interval for each of
the four primary comparisons (Bonferroni family adjustment). A primary claim
of improvement requires its adjusted lower bound above 50%. These are approximate
bootstrap coverage intervals rather than exact tests. Color scores, secondary
comparisons, and cross-budget contrasts are exploratory. One seed pair per
opening mixes position and seed variability rather than estimating them separately.
Uniqueness conditioning weakly couples short-history sampling; this realization
has no duplicate boards, but the bootstrap is still a model of opening clusters,
not a proof of independence or universal strength. No Elo conversion is used.

## Performance methodology

Hardware matches Phase 2: MacBook Pro Mac17,2, Apple M5, ten cores (four performance,
six efficiency), 32 GB, native arm64 CPython 3.11.17, macOS 26.6.2. Search, profile,
capacity, audit, and test runs execute sequentially. No services/processes are
stopped or reprioritized. Read-only host inspection found the pre-existing Python
processes idle and Docker host CPU near zero; browser/system activity remains.
A mid-run load-average snapshot was 4.09/3.44/3.07, so this is not quiet-machine
isolation. Hardware metadata is in the manifest; do not transfer laptop numbers
to the Mac Mini or Docker unchanged.

[profiling.json](profiling.json) uses sixteen independently declared random
positions plus the existing empty, near-opening, and late fixtures. Each budget
400/2,000/5,000/10,000 has one discarded warm-up and seven interleaved fresh-seed
samples. Actual production choose_move is timed without subclass instrumentation.
Memory capture and cProfile are separate runs. Full caller profiles and raw samples
remain available; profiled absolute time is not normal latency.

Selection and rollout are inclusive method subtrees. Expansion components sum
drop and Node construction called directly by choose_move, including root
construction; this omits untried-list removal, RNG choice, and loop checks.
Backpropagation is a separate subtree. Root tactical checks use only direct
choose_move callers to avoid double-counting nested winning-move checks.
The residual includes choose_move/loop machinery and uncategorized work. This
approximates a disjoint accounting, not a precise line-level expansion stopwatch.

Tracemalloc current/peak values retain the complete tree via a diagnostic hook,
including agent/RNG/temporary Python allocations, and exclude interpreter/native
overhead. Largest retained allocation sites describe live blocks, not all
allocation churn. Fresh-process RSS in [capacity.json](capacity.json) is a separate
high-water measurement with interpreter/import overhead; it does not isolate tree
bytes. No allocator failure, swap-pressure, or long-service leak claim follows
from these short runs. Parent/child references form cycles, so GC behavior can
affect timing and temporary retention.

## Production readiness and recommended future presets

| Position | Budget | Fresh-process RSS before → after MiB | Isolated → contended MCTS ms |
| --- | ---: | ---: | ---: |
| empty | 400 | 37.78 → 37.78 | 4.84 → 4.87 |
| empty | 2,000 | 37.86 → 38.34 | 24.88 → 43.00 |
| empty | 5,000 | 37.95 → 40.02 | 62.95 → 80.95 |
| empty | 10,000 | 37.73 → 42.48 | 133.55 → 153.18 |
| near-opening | 400 | 38.59 → 38.59 | 3.94 → 4.06 |
| near-opening | 2,000 | 37.73 → 38.12 | 20.07 → 34.98 |
| near-opening | 5,000 | 38.00 → 39.97 | 53.32 → 67.91 |
| near-opening | 10,000 | 37.75 → 42.30 | 118.52 → 132.11 |

RSS is an isolated-process high-water mark, not live reachable memory; baseline imports consume about 38 MiB. At 10,000 the high-water mark is about 42.5 MiB. The traced worst tree remains about 5.1 MiB. These small short-run values do not establish a steady-state service memory bound.

Synthetic contention runs one MCTS and one Negamax-depth-6 search on separate copies of the same fixture using two Python threads, with one discarded warm-up and five interleaved repetitions. Returned moves match isolated searches. MCTS 2,000 grows from roughly 20–25 ms to 35–43 ms; 10,000 grows from 119–134 ms to 132–153 ms. Depth-6 work finishes sooner than the larger MCTS search, limiting its interference. This does not model deeper Negamax/Victor work, endpoint overhead, a traffic burst, cold start, Docker, or remote network latency.

Current [Mac Mini deployment documentation](../../mac-mini-deployment.md) describes one Gunicorn worker, four threads, PUBLIC_SEARCH_CONCURRENCY=2, and a separate nonblocking **one-at-a-time MCTS reservation**. The source confirms both gates. A second simultaneous MCTS request is rejected as agent_busy/503 rather than queued; MCTS plus another search can contend. Threads share the CPython GIL, so additional cores do not imply parallel throughput for these Python loops. Request-rate limits are availability bounds, not reserved search capacity. Higher budgets keep the MCTS reservation occupied longer and increase busy responses under concurrent public traffic.

A useful future ladder is **400 quick / 2,000 balanced / 5,000 strong**, with **2,000 as the candidate default** after an optimized-build deployment and capacity validation. The 2,000 budget has clear primary improvement at roughly 20–25 ms on early laptop fixtures; 5,000 offers another nominal score gain at roughly 53–64 ms. Keep 10,000 for local/advanced experiments initially: it roughly doubles 5,000 latency and memory, while the additional primary score gain remains uncertain and the depth-6 score does not improve. MCTS 800 can remain a compatibility setting, but this experiment provides weaker evidence for it as a new strength tier.

The current public UI remains **100/400/800, default 100**, and the API remains capped at 800 with its existing accepted values. This study does not compare 100 directly, so it cannot quantify a proposed default change relative to the current default. The recommendation is conditional on this optimized implementation and tested position mixture, not evidence that deployment has occurred.

The Mac Mini hardware/model and effective Python/container performance are not measured in this task; the supplied deployment documentation does not establish a laptop-to-Mini speed ratio. CPU generation, performance/efficiency cores, architecture, container runtime, cooling, host services, and GC can differ. As a planning sensitivity only, a 2× slower host would turn early-fixture 2,000/5,000 medians into roughly 40–50/107–127 ms; this is neither a prediction nor an upper bound. First measure warm/cold endpoint p50/p95/max, CPU and RSS/GC over repeated games, and MCTS plus the highest allowed other searches on the actual host during a separately authorized phase. Preserve the gates and measure busy-response rates; no cap/default/deployment change is made now.

## Validation, scope, and next phase

The full backend suite passes **1,836 tests**, with **15 skips because PyTorch is not installed**, recorded in [backend-tests.txt](backend-tests.txt). This includes all existing MCTS, bitboard, Negamax, engine/API, seeded history/provenance, public search contention, and rate-limit tests, plus six new harness tests. Torch-dependent research modules remain unvalidated in this venv. The audit passes **1,776 saved games / 29,044 search calls** and all **28 deterministic repeats**; see [audit.json](audit.json). All **431 frontend unit tests** pass; see [frontend-tests.txt](frontend-tests.txt). Browser/E2E tests are not run for this backend-only change. Python compilation and final `git diff --check` pass.

The added integrity tests exercise legal and diverse opening generation, isolated
RNGs, exact seeded game-boundary resume, duplicated/corrupt/foreign-result
rejection, source drift, incomplete failures, cluster inference, and actual
diagnostic simulation counts. The saved-game audit independently replays histories,
checks guard accounting and totals, and repeats both colors of all fourteen
conditions on the first declared opening. Repeated audit games are excluded
from strength observations.

All work stays in the laptop repository. The Mac Mini Services checkout, Docker
deployment, Tailscale, Render, AlphaZero worktree/training, frozen canonical
benchmarks, and main branch are unchanged. No paid API calls or deployment occur.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for commands and artifact semantics.

Proposed next phase: evaluate deeper optimized Negamax search (depths 8 and 10)
with a new declared, paired strength/cost plan against the useful MCTS budgets
identified here. Preflight search-node and memory costs first, including quiet
positions, tactical loss trees, and endgames. Profile leaf scoring, terminal
geometry, and TT work before choosing an implementation. Incremental 69-window
evaluation or compact TT storage should be separate measured ablations with an
independent oracle; preserve full-window root values, typed bounds, depth/mover
separation, deterministic ties, and public depth caps until separate approval.
