# Connect Four MCTS Phase 2

Phase 2 replaces engine-object simulations with detached bitboards and reduces
tree overhead. The public constructor, agent name, legal-column return type,
simulation presets (100/400/800), API contract, resource limits, and deployment
configuration are unchanged. Phase 1 Negamax source and evidence are untouched.

## Architecture

Before: UCT nodes retain full `Connect4` objects; root checks, expansion, and
each rollout deep-copy those objects. Rollouts repeatedly generate legal moves,
scan the array board for wins, and invoke the engine's guarded move operation.
Rewards belong to the player making the incoming move; final selection uses
root-child visit counts with injected-RNG tie breaking.

After: the engine board is converted once to an immutable slotted
`BitboardState`. Its two player bitboards use seven bits per column: six
playable cells, bottom first, and an empty sentinel. Win shifts are 1/7/6/8,
matching existing Negamax/Victor geometry without importing their search code.
A drop locates the next cell with column-local integer addition, computes only
the mover's new win status, and flips the turn even on a winning move.
Occupied bits identify full columns and full-board draws. Legal boards with
gravity are the input contract; this internal state is not a new board parser
or public API.

Nodes use slots and retain these compact snapshots. Terminal status is cached
in the state. Rollouts use local player/occupied integers and one legal-column
list, removing a column only when it fills; they allocate no engine objects or
per-move state snapshots. Selection computes the parent's logarithm once and
builds only the best-score ties. The UCT arithmetic, exploration coefficient,
previous-player reward perspective, draw half-points, expansion accounting,
root visit rule, and RNG calls are preserved. The immediate-win guard still
bypasses simulations; otherwise every decision performs its requested budget.
The immediate-opponent-win filter retains the existing all-unsafe fallback.

The center-first legal order `(3, 2, 4, 1, 5, 0, 6)` is deliberately retained,
including singleton random choices. There is no heuristic rollout policy,
transposition sharing, neural evaluation, or NumPy/PyTorch dependency in the
MCTS implementation. A read-only array view exists for private-node diagnostic
compatibility; production search never constructs it. `Node(Connect4(...))`
remains usable by legacy unit tests; production nodes contain bitboards.

## Methodology and evidence

The original MCTS source is loaded directly from commit
`81aeb764eb3ab29d6b26ecd02a0378f7c5c9c0d9`, never from a mutable copy.
[baseline.json](baseline.json) records measurements started before editing the
agent; the benchmark subsequently gained tree/RNG fingerprints and caller-level
profile output before the main ablation run. The primary comparisons use
[ablations.json](ablations.json), with all
four variants interleaved in one interpreter. The five existing histories are
imported directly from `scripts/benchmark_public_agents.py`. Eight additional
legal, nonterminal histories were declared before optimization: two each at
8/12/16/20 plies, generated with seed 20261010, rejecting only terminal games.
No position was selected or excluded using timing or MCTS performance. One
supplemental history has an immediate win; its zero-simulation bypass is
reported separately rather than assigned fictional simulation throughput.

Every position/budget/variant gets one discarded warm-up and three fresh-tree
samples at seeds 701/702/703, with variant order rotating across repetitions.
Budgets are 100/400/800/1000. Wall (`perf_counter`) and CPU (`process_time`)
measure only `choose_move`, excluding fixture setup, assertions, profiling, and
memory tracing. Reported throughput uses executed simulations, not the budget
when a guard bypasses search. Raw timings, returned moves, histories, source
hashes, and hardware metadata are retained. Three samples give descriptive
medians, not confidence intervals or p95/production latency guarantees.

An untimed seed-701 run uses `tracemalloc` while retaining the root through a
backpropagation hook. Current retained bytes and peak traced allocations
include the agent, complete tree, and search-induced cached temporary Python
allocations; these are neither pure reachable-tree size, process RSS, nor
serialized object sizes. Pre-existing interpreter allocations, native memory,
and allocator overhead are excluded. GC runs before tracing, not after search;
in particular C's repeated array-view conversions can leave tuple freelists
in this snapshot. Complete tree fingerprints cover boards, path/move
order, untried moves, mover perspective, visits, and rewards. Fingerprints and
final RNG states are compared across variants, independently of timed runs.

A: original. B: original tree/guards/selection with bitboard rollouts, including
engine-to-bitboard conversion per simulation. C: compact tree/guards/selection
with original engine rollouts, including compact-to-engine conversion per
simulation. D: combined production implementation. C includes state operations,
cached terminals, slots, compact root guards, and selection overhead reductions;
it does not isolate raw object-storage size from those associated tree changes.
Conversion costs are included rather than hidden, so the ablation speedups are
not additive. The policy is identical in all four variants.

Hardware: MacBook Pro Mac17,2, Apple M5, 10 cores (4 performance/6 efficiency),
32 GB RAM, CPython 3.11.17 native arm64, macOS 26.6.2. Baseline collection
overlapped backend tests for part of the supplemental set; it is preliminary
evidence, not the primary timing comparison. The matched ablations run after
tests finish. Host browser/widget/Bluetooth activity remains; no process or
service is stopped or reprioritized. These are laptop measurements.

## Results

All **208 ablation rows** (13 histories × 4 budgets × 4 variants) completed
their legality, caller-isolation, simulation-count, move, tree-fingerprint,
and final-RNG assertions. The combined implementation is **53.6–72.3× faster**
over all 48 non-bypassed position/budget comparisons. The seven supplemental
positions that perform search improve by **53.6–67.6×** across all budgets.
The immediate-win supplemental position is excluded from those ranges.

### Wall and CPU time

Paired medians in milliseconds, original → combined. CPU time closely follows
wall time. Columns remain zero-based; complete per-seed moves are in JSON.

| Position | Simulations | Wall ms | CPU ms | Wall speedup |
| --- | ---: | ---: | ---: | ---: |
| empty | 100 | 84.851 → 1.251 | 84.826 → 1.251 | 67.8× |
| empty | 400 | 331.337 → 4.871 | 331.190 → 4.871 | 68.0× |
| empty | 800 | 659.853 → 9.767 | 658.100 → 9.766 | 67.6× |
| empty | 1000 | 817.250 → 12.527 | 816.669 → 12.525 | 65.2× |
| near-opening | 100 | 67.892 → 1.050 | 67.835 → 1.049 | 64.7× |
| near-opening | 400 | 260.508 → 3.923 | 260.339 → 3.920 | 66.4× |
| near-opening | 800 | 507.654 → 8.088 | 507.392 → 8.085 | 62.8× |
| near-opening | 1000 | 630.894 → 9.946 | 630.469 → 9.940 | 63.4× |
| midgame | 100 | 43.706 → 0.673 | 43.670 → 0.672 | 64.9× |
| midgame | 400 | 152.377 → 2.371 | 152.302 → 2.363 | 64.3× |
| midgame | 800 | 295.877 → 4.505 | 295.747 → 4.505 | 65.7× |
| midgame | 1000 | 365.571 → 5.586 | 365.450 → 5.578 | 65.4× |
| midgame-wide | 100 | 47.869 → 0.782 | 47.859 → 0.781 | 61.2× |
| midgame-wide | 400 | 181.309 → 2.889 | 181.219 → 2.888 | 62.8× |
| midgame-wide | 800 | 349.661 → 6.218 | 349.610 → 6.218 | 56.2× |
| midgame-wide | 1000 | 437.123 → 7.311 | 437.073 → 7.311 | 59.8× |
| late | 100 | 43.968 → 0.646 | 43.956 → 0.646 | 68.0× |
| late | 400 | 166.602 → 2.816 | 166.543 → 2.815 | 59.2× |
| late | 800 | 322.165 → 4.815 | 322.097 → 4.810 | 66.9× |
| late | 1000 | 403.231 → 5.579 | 403.094 → 5.570 | 72.3× |

At 1,000 simulations, throughput changes as follows:

| Position | Original sims/s | Combined sims/s | Seed-701 move (both) |
| --- | ---: | ---: | ---: |
| empty | 1,224 | 79,826 | 1 |
| near-opening | 1,585 | 100,546 | 6 |
| midgame | 2,735 | 179,014 | 0 |
| midgame-wide | 2,288 | 136,776 | 3 |
| late | 2,480 | 179,229 | 6 |

### Ablations

Median wall milliseconds at 1,000 simulations:

| Position | A original | B rollouts only | C compact tree only | D combined |
| --- | ---: | ---: | ---: | ---: |
| empty | 817.250 | 168.882 | 677.925 | 12.527 |
| near-opening | 630.894 | 164.856 | 481.717 | 9.946 |
| midgame | 365.571 | 143.617 | 243.978 | 5.586 |
| midgame-wide | 437.123 | 167.401 | 290.764 | 7.311 |
| late | 403.231 | 273.323 | 147.070 | 5.579 |

B alone improves these five rows by 1.5–4.8×, while C improves them by
1.2–2.7×. Keeping array-based terminal checks anywhere in the main tree or
rollout path leaves substantial cost. The combined result removes both
costs and the per-rollout representation conversion; it is substantially
faster than either isolated variant. C is particularly useful on the late
board, where visits to terminal leaves make repeated tree scans expensive.

### Memory

Retained traced-allocation snapshots at 1,000 simulations, MiB. Peak values
are in JSON and are within about 0.004 MiB of these snapshots. The trees
contain exactly the same nodes across variants, so this comparison does
not depend on exploring fewer nodes.

| Position | A original | B rollouts only | C compact tree only | D combined | D reduction vs A |
| --- | ---: | ---: | ---: | ---: | ---: |
| empty | 1.716 | 1.715 | 0.881 | 0.512 | 70.1% |
| near-opening | 1.655 | 1.654 | 0.865 | 0.496 | 70.0% |
| midgame | 1.156 | 1.129 | 0.708 | 0.339 | 70.7% |
| midgame-wide | 1.680 | 1.679 | 0.871 | 0.502 | 70.1% |
| late | 1.070 | 1.029 | 0.675 | 0.306 | 71.4% |

Combined retained allocations fall by **70.0–71.4%** on the existing five
1,000-simulation positions. B retains the full engine tree, so its memory
changes are small. C retains compact nodes but its repeated diagnostic
array conversions leave extra traced temporary allocations/freelists.
Do not interpret C minus D as additional permanent tree storage.

### Profile

A separate near-opening/1,000-simulation cProfile run at seed 9102 measures
1,626.15 ms original versus 30.02 ms combined. Profiling changes absolute
latency; use the uninstrumented tables above for speed claims. These are
inclusive categories with overlap, not an additive partition:

| Work | Original profiled ms | Combined profiled ms |
| --- | ---: | ---: |
| Rollouts (`_simulate`) | 1,229.35 | 16.18 |
| Engine winner scanning (`check_winner`) / bitboard `has_four` | 1,192.99 | 4.48 |
| Engine pre-move terminal guard (`_has_terminated`) | 141.53 | 0 |
| Copying (`deepcopy`, 2,008 top-level calls) | 116.74 | 0 |
| Move generation (`get_valid_moves`) | 95.67 | 1.12 |
| Child selection (`_select_child`, 2,586 calls) | 17.76 | 6.56 |
| Expansion/root construction components | 136.09 | 3.23 |
| Backpropagation (1,000 calls) | 0.71 | 0.49 |
| Root tactical checks, direct calls only | 8.08 | 0.16 |

Expansion/root construction sums the direct `choose_move` callers of
`deepcopy`, `make_move`, and the node constructor in A, versus `drop` and
the node constructor in D. It includes initial node construction and is
an estimate of those components, not an isolated expansion stopwatch.
It excludes shared selection-loop checks, container/RNG work, and D’s
one initial board conversion. Root tactical checks sum the direct root
winning/safe checks without double counting winning checks inside the
safe filter. Full caller-level profiles for all four variants are in JSON.

Terminal winner scans consume about 73% of original profile time; the
additional guarded move scan is another 9%. In the optimized profile,
rollouts are about 54% and child selection about 22%, making both useful
targets for the next phase.

### Stochastic behavior and regressions

There is **no observed RNG-consumption change** at equal budgets. All three
timed seeds agree on moves in every matched group, and seed 701 agrees on
complete tree boards, untried move order, visits, rewards, incoming movers,
and final RNG state. Independent engine rollout tests also agree on final
RNG state. This preserves the existing random policy, rather than adjusting
outcomes to force equality. This is measured coverage, not a proof about
every seed or Python version. Increasing the budget legitimately changes
RNG consumption and can change moves; those differences are not regressions.

No combined wall-time regression was measured on this declared set, and
there is no unresolved correctness regression. Limitations include three
timing samples, host activity, one supplemental tactical bypass, traced
Python memory rather than RSS, internal legal-board assumptions, and small
playing-strength sample size. Public budgets remain unchanged.

### Higher budgets and paired play

[strength.json](strength.json) records one warm-up and three interleaved
fresh-seed latency samples at original 400 versus combined 10,000 simulations.
The latter executes **25× more simulations** in **24.1–45.2% of the original
latency** across these four positions. This demonstrates substantial latency
headroom directly at higher budgets rather than extrapolating from a fixed
simulations/second ratio:

| Position | Original 400 wall ms | Combined 10,000 wall ms | Original → combined CPU ms |
| --- | ---: | ---: | ---: |
| empty | 331.938 | 127.318 | 331.779 → 127.261 |
| near-opening | 258.403 | 116.684 | 258.354 → 116.672 |
| midgame-wide | 175.514 | 72.069 | 175.474 → 72.052 |
| late | 167.783 | 40.515 | 167.762 → 40.506 |

Paired play uses all eight supplemental openings, both optimized colors per
opening, and fresh agent seed streams 950000–950015. Each seed pair is reused
with reversed colors and across budget conditions. Agents retain their RNG
streams throughout each game. Histories, complete move sequences, winners,
colors, seeds, and per-move wall times are recorded; no provider, API,
training, or production service is involved. There are 16 games per condition:

| Optimized vs original budgets | Optimized wins / draws / losses | Score with half-points for draws |
| --- | ---: | ---: |
| 400 vs 400 | 7 / 2 / 7 | 50.0% |
| 10,000 vs 400 | 11 / 1 / 4 | 71.9% |

The equal-budget control scores 50%; it does not establish a strength change,
consistent with the preserved seeded behavior. The higher-budget condition
scores better in this small sample, but 16 games over eight reused openings
are insufficient for a reliable strength estimate or Elo claim. One opening
has an immediate win and largely tests color balance. Throughput improvement
is established much more strongly than playing-strength improvement.
Increasing simulations can still lose games; root guards protect only the
next reply. Keep public presets unchanged and use a larger independent
opening/seed set before making strength or production-budget decisions.

## Correctness and integration

The new suite has **277 passing tests**. Geometry tests independently enumerate
all 69 array-board four-cell windows for both players, checking each complete
window and each single missing cell, plus sentinel-boundary false positives.
160 seeded legal games compare every visited state and every alternative legal
drop against `Connect4`; fixtures cover empty/full columns, both turn parities,
horizontal/vertical/both diagonal wins, nearly full boards, full draws, and
terminal rejection. Rollout tests compare outcomes and complete RNG states
with engine rollouts. Tests enforce immutable snapshots, caller isolation,
exact 100/400/800/1000 accounting, compact node state, previous-player rewards,
and selection equivalence to independently computed UCB scores.

All **116 existing MCTS and MCTS API tests** pass unchanged. They cover
immediate wins, forced blocks, win-before-block precedence, all-unsafe fallback,
floating opponent threats, terminal draw revisits, legal moves, injected RNG,
root visit selection, reproducible tree statistics, atomic failure handling,
process-wide search contention, reservations, history, and replay.

The complete backend command passes **1,830 tests**, with **15 skips** because
PyTorch is not installed in this development venv; research/neural modules and
two individual torch-dependent checks are not fully validated here. No new
dependency is installed and no training resources are used. A focused API,
MCTS, Negamax, history/seeded API, and public rate-limit run passes **592 tests**.
The extra frontend unit run passes **431 tests** despite no frontend change.
Browser/E2E tests were not run. See [backend-tests.txt](backend-tests.txt),
[integration-tests.txt](integration-tests.txt), and
[frontend-tests.txt](frontend-tests.txt).

The initial full-suite invocation had one configuration failure (1,829 passed,
15 skipped): a local dotenv research opt-in overrode the test's expected
default-disabled Victor flag. It is recorded in
[backend-initial.txt](backend-initial.txt). Rerunning with
`PYTHON_DOTENV_DISABLED=1` resolves it without changing application code or
environment files. MCTS import and an actual 100-simulation search also succeed
with NumPy and PyTorch imports explicitly blocked. `git diff --check` passes.

## Reproduction

```sh
.venv/bin/python -m scripts.benchmark_mcts_bitboards --variants original --profile --output /tmp/mcts-baseline.json
.venv/bin/python -m scripts.benchmark_mcts_bitboards --profile --output /tmp/mcts-ablations.json
.venv/bin/python -m scripts.evaluate_mcts_bitboards --output /tmp/mcts-strength.json
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m pytest tests -q -rs
cd ui
npm test
```

The dotenv switch applies only to the test process. It prevents a laptop `.env`
research opt-in from overriding the default-disabled configuration test. No
local environment file or production configuration is changed.

## Files and next phase

- `games/connect4/agents/mcts_agent.py`: compact tree, guards, rollouts, and UCT loop.
- `games/connect4/agents/mcts_bitboard.py`: stdlib-only detached state and random rollout core.
- `tests/test_connect4_mcts_bitboard.py`: independent array-engine correctness tests.
- `scripts/benchmark_mcts_bitboards.py`: original-source loading, matched ablations, profiling, memory and behavior evidence.
- `scripts/evaluate_mcts_bitboards.py`: paired/color-balanced play and increased-budget latency measurements.
- `docs/search-mcts-v2/`: report, raw measurements, and validation evidence.

Recommend profiling the optimized selection/rollout split on a larger
predeclared set, then testing reduced Python allocations or a compiled rollout
kernel as separate ablations. Any tactical rollout policy should be a separate
strength experiment with fresh seeds, color-balanced pairs, and many more
games. Retain the current public budgets pending API-capacity and strength
evaluation; throughput gains alone do not establish stronger play.

The Services checkout, production API, Tailscale, Docker production, Render,
AlphaZero research/training resources, canonical-season-v1, frozen evidence,
and main branch were not modified. No deployment is part of this phase.
