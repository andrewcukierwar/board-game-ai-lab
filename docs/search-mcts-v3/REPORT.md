# Connect Four MCTS v3: strength per unit of computation

October 10, 2026. Laptop research on branch `research/mcts-v3`, starting from
`main` at `60b99b0d42145da149f433907585b809060c946d`. This is a research
result. The production `MCTSAgent`, its bitboard module, the public presets
(100 / 400 / 800), API caps, provenance versions and all deployment
configuration are unchanged, and nothing was deployed.

## Result

A research-only MCTS variant, **F1**, beats the current optimized MCTS while
using no more search time. F1 combines three changes: tactical rollouts that
take wins, block single threats and avoid handing over a win; a solver that
proves and prunes tactical lines inside the tree; and an exploration constant
of 0.5 instead of 1.41.

On 256 held-out openings (512 games per comparison, both colours):

| Comparison | F1 simulations | Baseline simulations | F1 score | Family-adjusted interval | F1 time ÷ baseline time |
| --- | ---: | ---: | ---: | --- | ---: |
| Equal simulations | 400 | 400 | 74.2% | 70.4–78.1% | 1.95 |
| Equal time | 187 | 400 | 70.8% | 66.3–75.1% | 0.97 |
| Equal simulations | 2,000 | 2,000 | 68.3% | 64.3–72.2% | 1.81 |
| Equal time | 1,023 | 2,000 | 65.7% | 61.6–69.6% | 0.96 |

For scale: in the v2 study, on a slightly different position mixture, giving
the baseline 25 times more simulations (10,000 versus 400) produced a 70.3%
score. F1 gets a comparable score against the baseline at 400 with fewer than
half the simulations and slightly less time.

Supporting results, all at equal time or less:

- Against Negamax depth 4, 6 and 8, F1 scores 21.6 to 26.0 points higher than
  the baseline does against the same opponent.
- The gain holds from 100 to 5,000 baseline simulations (66–71%).
- From the empty board, with the budget cut further so F1 is no slower in the
  opening, F1 scores 75.4% and 73.4%.
- F1 at 187 simulations (2.5 ms per move) beats the baseline at 2,000
  (13.0 ms) 58.8% to 41.2%; F1 at 1,023 beats the baseline at 10,000 63.5% to 36.5%.

A second finalist, **F2** (plainer rollouts, solver, default constant), is
also a confirmed improvement but is weaker than F1 in every comparison.

Classification, by the gates declared before the games:

| Variant | Status |
| --- | --- |
| F1 — R2 rollouts + solver, c = 0.5 | **Accepted research improvement** at both budget tiers |
| F2 — R1 rollouts + solver | **Accepted research improvement** at both tiers; superseded by F1 |
| R1, R2, solver, c = 0.5 or 0.7, each alone | Promising but inconclusive: positive on the development set, not individually confirmed |
| Centre-first expansion | Rejected at the pilot stage: 51.4%, indistinguishable from the A/A control |
| c = 2.0 | Rejected: below 50% in both sweeps |
| c = 1.0 | No evidence of a difference from 1.41 |

"Accepted" means validated on this position mixture on this laptop. It is not
a production change; see [Integration recommendation](#integration-recommendation).

## What was tested

The baseline is the production agent: UCB1 with c = 1.41, random expansion,
uniform random rollouts, most-visited root move, and immediate-win and
safe-reply guards at the root only.

Candidates are switches on a research-only class, `ResearchMCTSAgent`
(`games/connect4/agents/mcts_research_agent.py`), which the agent factory and
API never import. With every switch at its default it reproduces the
production agent's moves, tree and random-number consumption exactly.

| Id | Change |
| --- | --- |
| R1 | Rollouts play an immediate win, otherwise block a single opponent win; two simultaneous opponent wins end the rollout as a loss. Otherwise uniform. |
| R2 | R1, and the uniform choice skips columns that let the opponent win directly on top, unless every column does. |
| S | Tree nodes carry a proven win / draw / loss. One-move tactics are resolved when a node is created, proofs back up minimax-style, proven-lost moves are never selected, the search stops once the root is proven, and a proven win is always played. |
| E | Expand untried moves centre-first instead of at random. |
| C | Exploration constant 0.5, 0.7, 1.0 or 2.0 instead of 1.41. |

R1 and R2 follow the decisive-move idea of Teytaud & Teytaud (2010); S is
MCTS-Solver (Winands, Björnsson & Saito, 2008). Both are knowledge-free: they
use only the rules of the game.

## Method

The full plan is in [DESIGN.md](DESIGN.md). It was pushed before any strength
game (`65f4211`); three dated amendments and the finalist freeze were each
pushed before the games they govern.

- **Positions.** Uniform random legal histories at 2, 5, 8, 11, 14, 17, 20 and
  23 plies in equal numbers. Terminal positions, duplicate boards, and
  positions the shared root guards decide within two plies are excluded; the
  last group always scores exactly 50% and carries no information. This
  differs from v2, which kept them, so percentages here are not directly
  comparable with v2's.
- **Two disjoint sets.** 96 development openings for every pilot, tuning and
  timing calibration; 256 held-out openings touched only after the finalists
  and their budgets were frozen in Git (`9e98cac`).
- **Pairing.** Each opening is played twice with colours swapped. The two
  games are one cluster. Seeds are 128-bit, domain-separated, and follow the
  agent across colours.
- **Equal time.** A wall-clock deadline would make games irreproducible, so
  the candidate instead gets a reduced fixed budget
  `B' = floor(min(1, 0.90·r)·B)`, where `r` is baseline time over candidate
  time measured in development games. Realised time is then checked in the
  held-out games themselves.
- **Statistics.** Bootstrap over whole openings within length strata, 10,000
  replicates. Primary comparisons use a Bonferroni interval for a family of 8
  (99.375%). A sign-flip randomisation test is a sensitivity check. No Elo.
- **Execution.** All games serial, under the shared benchmark lock. Every game
  is one fsynced JSON line; studies resume at game boundaries and refuse to
  resume if their pinned sources changed.

Volume: 20,288 games and 4,013 s of game time across 16 studies, inside every
declared cap (pilots 1,808 s of 3,600; primary 719 s of 4,500; secondaries
757 s of 3,600; follow-ups 665 s of 2,700).

## Pilots (development set, exploratory)

192 games per condition per budget. Scores are pooled over baseline budgets
400 and 2,000. These numbers chose what to confirm and support no claim.

| Configuration | Equal simulations | Equal time | Time per decision vs baseline |
| --- | ---: | ---: | ---: |
| A/A control (research class at defaults) | 50.7% | — | 0.99 |
| R1 | 60.3% | 59.6% | 1.31–1.35 |
| R2 | 64.7% | 59.2% | 1.70–1.78 |
| S | 58.3% | 57.7% | 0.95–0.98 |
| E | 51.4% | not run | 0.97–0.98 |
| C = 0.5 / 0.7 / 1.0 / 2.0 | 57.9% / 57.8% / 52.6% / 47.7% | — | 0.99–1.01 |
| R1+S | 65.1% | 61.0% | 1.27–1.41 |
| R2+S | 68.2% | 61.6% | 1.78–1.96 |
| R2+S, c = 0.5 | 65.8% | 63.3% | 1.76–1.92 |

On R2+S, head-to-head against R2+S at 1.41, the constants scored 53.6% (0.5),
52.4% (0.7), 51.4% (1.0) and 46.7% (2.0).

What the pilots showed:

- The A/A control sits at 50.7%, so the harness is not biased toward the
  research class.
- The components add up: rollouts and solver together beat either alone at
  equal simulations.
- R2 is clearly stronger than R1 per simulation but costs more per
  simulation; at equal time the two were indistinguishable. Amendment 1
  therefore piloted both combinations instead of dropping R2 on a 0.4-point
  difference.
- The top four configurations are within about four points at equal time,
  inside pilot noise. The declared rule picked F1 and F2.

## Confirmatory results (held-out set)

### Primary family of eight

512 games per row. The sign-flip p-value is 0.0001 for every row, the smallest
value 10,000 replicates can produce.

| Finalist | Mode | Finalist sims | Baseline sims | W / D / L | Score | 95% interval | Family interval | Time ratio |
| --- | --- | ---: | ---: | --- | ---: | --- | --- | ---: |
| F1 | equal sims | 400 | 400 | 373 / 14 / 125 | 74.2% | 71.4–77.1% | 70.4–78.1% | 1.949 |
| F1 | equal time | 187 | 400 | 352 / 21 / 139 | 70.8% | 67.6–73.8% | 66.3–75.1% | 0.970 |
| F1 | equal sims | 2,000 | 2,000 | 339 / 21 / 152 | 68.3% | 65.3–71.1% | 64.3–72.2% | 1.808 |
| F1 | equal time | 1,023 | 2,000 | 326 / 21 / 165 | 65.7% | 62.8–68.7% | 61.6–69.6% | 0.963 |
| F2 | equal sims | 400 | 400 | 341 / 18 / 153 | 68.4% | 65.2–71.5% | 64.0–72.7% | 1.407 |
| F2 | equal time | 255 | 400 | 326 / 25 / 161 | 66.1% | 62.9–69.2% | 61.6–70.6% | 0.935 |
| F2 | equal sims | 2,000 | 2,000 | 314 / 29 / 169 | 64.2% | 61.6–66.7% | 60.7–67.7% | 1.334 |
| F2 | equal time | 1,415 | 2,000 | 299 / 23 / 190 | 60.6% | 57.6–63.7% | 56.4–64.7% | 0.965 |

Every family-adjusted lower bound is above 56%, and every equal-time match
used slightly less total search time than the baseline. The advantage is
somewhat smaller at 2,000 than at 400: more simulations let the baseline find
some of the tactics on its own.

F1 scores above F2 in all four pairings by 4.1 to 5.9 points (nominal 95%
intervals exclude zero). That contrast was computed after the fact from
shared openings against the common baseline. It was not a declared comparison
and the two were never played against each other.

Colour balance is unremarkable: F1's equal-time score is 72.7% as first
player and 68.9% as second at 400, and 64.6% / 66.8% at 2,000.

### Where the gain comes from

F1's equal-time score by opening length (32 openings per cell, so each cell is
noisy):

| Opening plies | 2 | 5 | 8 | 11 | 14 | 17 | 20 | 23 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vs baseline 400 | 77.3% | 76.6% | 75.0% | 79.7% | 72.7% | 65.6% | 58.6% | 60.9% |
| vs baseline 2,000 | 82.0% | 75.8% | 74.2% | 56.2% | 64.1% | 62.5% | 53.9% | 57.0% |

The gain is largest from early positions, where a long game gives better
search more decisions to matter, and smallest from positions that are already
20 or more plies deep. It stays above 50% in every cell.

### Against Negamax (secondary, exploratory)

First 96 held-out openings, 192 games per cell. Each cell is the MCTS agent's
score against that Negamax depth; MCTS variants run at their equal-time budgets.

| Opponent | Baseline 400 | F1 187 | F2 255 | Baseline 2,000 | F1 1,023 | F2 1,415 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Negamax 4 | 37.5% | 61.5% | 52.9% | 46.9% | 68.8% | 62.8% |
| Negamax 6 | 33.3% | 55.7% | 48.2% | 41.4% | 67.4% | 60.7% |
| Negamax 8 | 29.4% | 51.0% | 38.5% | 40.1% | 63.8% | 56.0% |

Paired difference from the baseline against the same opponent, on shared
openings: F1 gains +21.6 to +26.0 points (every 95% interval above +14.8);
F2 gains +9.1 to +19.3 (every interval above +3.4). The baseline loses to
every Negamax depth at both budgets on this mixture. F1 at about 12 ms per
move beats all three, including depth 8, which takes about 16 ms per move.

### Budget scaling (secondary, exploratory)

F1 at equal time against the baseline across the budget range. 400 and 2,000
are the primary rows above (256 openings); the others use the first 128
held-out openings.

| Baseline sims | 100 | 400 | 800 | 2,000 | 5,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F1 sims | 44 | 187 | 386 | 1,023 | 2,702 |
| F1 score | 71.3% | 70.8% | 66.8% | 65.7% | 66.2% |
| 95% interval | 67.0–75.4% | 67.6–73.8% | 62.3–71.3% | 62.8–68.7% | 61.9–70.3% |
| Time ratio | 0.945 | 0.970 | 0.960 | 0.963 | 0.947 |

The improvement does not fade over a 50-fold range of budgets, which covers
all three public presets.

### Empty board (secondary, exploratory)

64 independent seed pairs from the real starting position, 128 games per row.

| Study | F1 sims | Baseline sims | W / D / L | Score | 95% interval | Time ratio |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| Declared equal-time budgets | 187 | 400 | 99 / 5 / 24 | 79.3% | 72.7–85.5% | **1.109** |
| Declared equal-time budgets | 1,023 | 2,000 | 92 / 1 / 35 | 72.3% | 65.2–79.3% | **1.121** |
| Strict budgets, fresh seeds | 151 | 400 | 95 / 3 / 30 | 75.4% | 67.6–82.8% | 0.900 |
| Strict budgets, fresh seeds | 821 | 2,000 | 93 / 2 / 33 | 73.4% | 66.0–80.9% | 0.918 |

The first two rows **fail the declared time-parity condition**: F1 took 11–12%
longer than the baseline, so they are not per-millisecond evidence. The last
two rows are the follow-up that fixes this; see the next section.

## Latency and resources

### Time is redistributed, not just scaled

"Equal time" was calibrated on total search time over mixed-length games. It
held on the held-out set, but F1 spends that time differently. Its rollouts
cost more in broad early positions, and its solver stops early in late ones:
across the held-out games F1 executed on average 139 of its 187 budgeted
simulations.

Per-decision wall time in the held-out equal-time games, in milliseconds:

| Agent | Mean | Median | 95th percentile | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Baseline 400 | 2.71 | 2.78 | 4.33 | 19.99 |
| F1 187 | 2.57 | 2.86 | 5.42 | 14.97 |
| F1 151 (strict) | 2.14 | 2.37 | 4.45 | 13.1 |
| Baseline 2,000 | 13.28 | 13.90 | 23.12 | 41.39 |
| F1 1,023 | 12.59 | 13.65 | 29.68 | 52.00 |
| F1 821 (strict) | 10.24 | 11.12 | 24.10 | 50.7 |

At the declared equal-time budgets F1 matches the baseline's mean and median
but its 95th percentile is about 25% higher, and from the empty board it is
11–12% slower overall.

**Follow-up A (amendment 2)** cut F1's budget by the measured empty-board
overrun, to 151 and 821, and evaluated once on fresh empty-board seeds and on
the held-out set:

| Set | F1 sims | Baseline sims | W / D / L | Score | Family interval | Time ratio |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| Held-out | 151 | 400 | 342 / 25 / 145 | 69.2% | 64.9–73.3% | 0.789 |
| Held-out | 821 | 2,000 | 334 / 24 / 154 | 67.6% | 63.6–71.6% | 0.783 |
| Empty board | 151 | 400 | 95 / 3 / 30 | 75.4% | 64.1–85.5% | 0.900 |
| Empty board | 821 | 2,000 | 93 / 2 / 33 | 73.4% | 62.9–83.2% | 0.918 |

At these budgets F1 is faster than the baseline from the empty board, uses
about 21% less total time on the mixed set, and has a 95th-percentile latency
within 1–6% of the baseline's, while keeping essentially the same score. The
held-out set is reused here; nothing was selected from it and the budgets came
from other data.

### Compute equivalence (follow-up C, exploratory)

F1 at its frozen equal-time budgets against much larger baseline budgets,
first 128 held-out openings, 256 games per row:

| F1 sims | Baseline sims | W / D / L | F1 score | 95% interval | F1 time ÷ baseline time |
| ---: | ---: | --- | ---: | --- | ---: |
| 187 | 2,000 | 146 / 9 / 101 | 58.8% | 54.1–63.5% | 0.197 |
| 187 | 5,000 | 132 / 18 / 106 | 55.1% | 51.0–59.2% | 0.077 |
| 1,023 | 5,000 | 162 / 11 / 83 | 65.4% | 61.5–69.3% | 0.379 |
| 1,023 | 10,000 | 157 / 11 / 88 | 63.5% | 59.4–67.8% | 0.190 |

F1 is ahead of a baseline that spends five times as long, at both tiers. The
187-versus-5,000 row (13 times the time) is ahead only at the nominal level;
its family-adjusted interval, 49.4–60.7%, includes 50%.

### Throughput and memory

- In games, the baseline searches about 143,000 simulations per second, F2
  about 67,000 and F1 about 49,000. Tactical rollouts cost more per move and
  run longer, because blocked wins extend the simulated game.
- Per decision at equal simulations F1 costs 1.8–1.95 times the baseline and
  F2 1.33–1.41 times, less than the per-simulation ratio because the solver
  ends many searches early.
- On the empty-board fixture, the worst case, F1 takes 12.9 ms at 400
  simulations against the baseline's 4.8 ms, and 64 ms at 2,000 against 25 ms.
- Traced peak memory at equal simulations is unchanged in the median and at
  most 1.28 times the baseline on any fixture; at equal-time budgets it is
  0.47–0.91 times. The tree still adds at most one node per simulation. The
  declared 1.5× memory gate is met.

Fixture numbers are in [finalists/throughput.json](finalists/throughput.json).

## Tactical audit (follow-up B, diagnostic)

Every decision in the first 128 held-out pairs of the two F1 equal-time
matches was re-scored with exact depth-8 Negamax. A "proven blunder" is a move
that turns a position not proven lost within 8 plies into one that is.

| Agent | Decisions | Proven blunders | Loss lands after 3 / 5 / 7 plies | Forced wins missed |
| --- | ---: | ---: | --- | ---: |
| Baseline 400 | 2,424 | 52 (2.15%) | 15 / 14 / 23 | 20 of 308 (6.5%) |
| F1 187 | 2,467 | 12 (0.49%) | 0 / 4 / 8 | 14 of 563 (2.5%) |
| Baseline 2,000 | 2,468 | 30 (1.22%) | 3 / 9 / 18 | 10 of 305 (3.3%) |
| F1 1,023 | 2,512 | 4 (0.16%) | 0 / 1 / 3 | 4 of 575 (0.7%) |

With fewer than half the simulations F1 makes roughly a quarter as many
provable blunders, and none of the shortest kind, where the opponent's forced
win lands three plies later. The baseline still makes those at 400.

Examples, as zero-based column histories from the empty board:

- Baseline, 400 simulations, history `601443052153445`: played column 2;
  column 1 was not proven lost. The opponent's win lands three plies later.
- Baseline, 2,000 simulations, history `4643244036366`: played 3, column 2
  holds; loss in three plies.
- F1, 187 simulations, history `44430011303`: played 2, column 4 holds; loss
  in five plies. This is F1's shortest kind of provable error.

F1 lost 139 of 512 games in that match but made only 12 provable blunders in
half of them. Most of its losses therefore contain no error visible within
eight plies. Its remaining weakness is positional or long-range, not
short-range tactics. The audit sees only outcomes forced within eight plies
and says nothing about positional quality; it is a description, not a
strength estimate. Full tallies and examples are in
[tactical-audit/audit.json](tactical-audit/audit.json).

## Decision gates

Applied to each finalist at each budget tier, as declared.

| Gate | F1 at 400 | F1 at 2,000 | F2 at 400 | F2 at 2,000 |
| --- | --- | --- | --- | --- |
| Correctness tests pass, seeded runs reproduce | yes | yes | yes | yes |
| Equal-time family lower bound above 50% | 66.3% | 61.6% | 61.6% | 56.4% |
| Equal-time point estimate at least 55% | 70.8% | 65.7% | 66.1% | 60.6% |
| Realised time ratio at most 1.00 | 0.970 | 0.963 | 0.935 | 0.965 |
| No significant loss to baseline versus Negamax | gains | gains | gains | gains |
| Memory at most 1.5× baseline | yes | yes | yes | yes |

All gates pass for both finalists at both tiers.

## Correctness

- **130 focused tests** in `tests/test_connect4_mcts_research.py` and
  `tests/test_mcts_v3_harness.py`.
- **Parity:** the research class at defaults matches production in move, final
  RNG state and complete tree fingerprint on 5 fixtures × 4 budgets × 2 seeds.
- **Threat detection:** agrees with four-detection on every empty cell of
  random positions from 4 to 36 plies.
- **Rollouts:** both tactical policies match an independent array-engine
  statement of the policy in outcome and RNG consumption.
- **Solver:** every proven node at any depth matches an exhaustive solve of
  late positions; a proven root plays a move achieving the exact value; a
  proven-lost move is never chosen when an alternative was searched.
- **All configurations:** legal moves, no caller mutation, seeded
  reproducibility, no tree kept between decisions, terminal rejection, and
  the production root guards.
- **Replay audit:** 440 recorded games sampled from all 16 studies were
  replayed and reproduced moves, winners, simulation counts and RNG
  fingerprints with zero mismatches
  ([validation/replay-audit.json](validation/replay-audit.json)).
- **Full backend suite:** 2,882 passed, 15 skipped
  ([validation/backend-tests-final.txt](validation/backend-tests-final.txt)).
  The skips are the torch-dependent research modules; PyTorch is not installed
  in this environment, as in earlier phases.

One real defect was found, by the rollout oracle, before any game was played:
R2's gift detection shifted sentinel-row bits back onto the board and wrongly
excluded some top-row moves. It was fixed by masking with the board mask.

## Limitations

- **One laptop.** Apple M5, CPython 3.11.17. A pre-existing `BTLEServer`
  process held about one core throughout (load average 3–5). Fixed-budget
  results do not depend on load; timings are descriptive and are not Mac Mini,
  Docker or endpoint latencies.
- **Cost ratios are specific to this Python implementation.** A compiled
  rollout kernel would change the price of tactical rollouts and therefore the
  equal-time budgets.
- **Artificial position mixture.** Random legal histories are not positions
  competent players reach. The empty-board study is the closest to real play
  and is small (64 seed pairs).
- **Held-out reuse.** The primary family used the held-out set once. The
  Negamax, scaling, strict-latency and equivalence studies reuse subsets of
  it. Nothing was tuned or selected on it, but those results are not
  independent samples and carry no adjusted significance claim.
- **No component attribution on held-out data.** Only the two combinations
  were confirmed. How much each of R2, S and c = 0.5 contributes is known only
  from the pilots. In particular c = 0.5 was chosen on a 3.6-point pilot
  difference and never confirmed separately.
- **F1 versus F2** is a post hoc contrast through a common opponent.
- **Latency shape.** At the declared equal-time budgets F1's 95th-percentile
  latency is about 25% above the baseline's and it is slower from the empty
  board. The strict budgets remove this at a small cost in simulations.
- **The simulation budget becomes a cap.** With the solver, a search can stop
  before its budget is spent. Anything that reports "simulations performed"
  would need to report the executed count.
- **Seeding differs from the public API**, which re-initialises its RNG per
  ply; these games use one persistent stream per agent per game.
- **Bootstrap intervals are approximate**, with one seed pair per opening.
- **Evidence size.** Raw game logs add about 19 MB to the branch.

## Integration recommendation

Recommend F1 for a future, separately authorized integration phase. Do not
change production from this report alone.

F1 meets every condition set for a production recommendation: tests pass,
gains are large and reproduced on positions never used for selection, memory
is bounded, and latency is acceptable with the caveat above. What a later
phase would need to decide and verify:

1. **Budget policy.** Two reasonable options. Keep the public simulation
   presets and accept roughly 2× the time per early move, which is still
   under 15 ms at 400 simulations on this laptop, for about a 74% score
   against today's agent. Or keep today's latency and scale the internal
   budget by about 0.38–0.41 (the strict budgets), for about a 69% score.
2. **Host measurement.** Repeat the latency and capacity measurements on the
   Mac Mini under the existing concurrency gates before choosing.
3. **Contract details.** Report executed rather than requested simulations,
   bump the MCTS provenance version, and check behaviour under the API's
   per-ply seeding.
4. **Keep the baseline.** Retain the current agent as a selectable reference
   so later comparisons stay possible.

## Next research

In rough order of expected value:

1. **Positional knowledge.** The audit says F1's remaining losses are not
   short-range tactical. Candidates: implicit minimax backups with a cheap
   threat-parity evaluation, or odd/even threat awareness in rollouts.
2. **Component ablation on a fresh held-out set**, to learn whether c = 0.5
   and gift avoidance each earn their place, and a direct F1 versus F2 match.
3. **Cheaper tactical rollouts.** The rollout recomputes one player's threat
   set per move; an incremental update or a compiled kernel would raise the
   equal-time budget without changing the policy, and the existing RNG-parity
   oracle can verify it.
4. **Latency-aware budgeting**, giving fewer simulations to broad early
   positions and more to narrow ones, since F1's cost varies with position
   more than the baseline's.
5. **Stronger opponents.** Compare F1 with Negamax 10 and the Victor research
   solver once the Negamax v3 work lands.

## Reproduction

```sh
/opt/homebrew/bin/python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

# Tests
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m pytest tests -q -rs

# Re-analyse any recorded study (writes analysis.json, prints the table)
.venv/bin/python -m scripts.mcts_v3.analysis --study confirm_primary

# Verify recorded games reproduce exactly
scripts/mcts_v3/with_benchmark_lock.sh .venv/bin/python -m scripts.mcts_v3.replay_audit \
    --pairs 2 --output /tmp/replay-audit.json

# Re-run a study from scratch: add a new function in scripts/mcts_v3/studies.py, then
.venv/bin/python -m scripts.mcts_v3.harness declare --study <name>
scripts/mcts_v3/with_benchmark_lock.sh .venv/bin/python -m scripts.mcts_v3.harness run --study <name> --max-seconds 560

# Fixture latency and memory; tactical audit
scripts/mcts_v3/with_benchmark_lock.sh .venv/bin/python -m scripts.mcts_v3.throughput --finalists --samples 7 --output /tmp/throughput.json
scripts/mcts_v3/with_benchmark_lock.sh .venv/bin/python -m scripts.mcts_v3.tactical_audit --study confirm_primary \
    --matchups f1-time-400 f1-time-2000 --pairs 128 --depth 8 --output /tmp/audit.json
```

F1 in code:

```python
from random import Random
from games.connect4.agents.mcts_research_agent import ResearchConfig, ResearchMCTSAgent

agent = ResearchMCTSAgent(187, rng=Random(1),
                          config=ResearchConfig(rollout='safe', solver=True, exploration=0.5))
```

## Evidence map

| Directory | Set | Role |
| --- | --- | --- |
| `preflight/` | preflight | Smoke run and fixture throughput; excluded from inference |
| `pilot1/` … `pilot3e/`, `pilot4/` | development | Exploratory pilots and timing calibration |
| `confirm_primary/` | held-out | Primary family of eight |
| `confirm_negamax/`, `confirm_scaling/`, `confirm_empty/` | held-out, empty | Declared secondaries |
| `followup_strict_empty/`, `followup_strict_holdout/` | empty2, held-out | Follow-up A |
| `tactical-audit/` | held-out games | Follow-up B |
| `followup_equivalence/` | held-out | Follow-up C |
| `finalists/`, `validation/` | fixtures | Latency, memory, test logs, replay audit |

Each study directory holds `study.json` (the frozen plan, with source hashes
and commit), `results.jsonl` (one line per game), `run-log.jsonl` (batches and
load averages) and `analysis.json`. Studies declared before the `empty2` seed
set was appended pin an earlier hash of `scripts/mcts_v3/harness.py`; only the
opening generator changed, and the replay audit confirms their games still
reproduce.

## Checkpoint commits

| Commit | Content |
| --- | --- |
| `337c214` | Progress log and benchmark lock wrapper |
| `65f4211` | Predeclared design |
| `6847229` | Research agent, harness, frozen openings, tests (suite: 2,878 passed, 15 skipped) |
| `2136a35`, `cefc677` | Preflight declared and recorded; pilot 1 declared |
| `a0d3438` | Pilot 1 results; pilot 2 declared |
| `0217e61` | Pilot 2 results; amendment 1; pilot 3a declared |
| `96f33be`, `448f16e`, `4ea9fcd`, `44f8ca7` | Pilots 3a–3d results, each declaring the next |
| `9e98cac` | Pilot 3e results; finalists frozen; confirmatory studies declared |
| `b1c0de1`, `64f9a1c` | Confirmatory primary (WIP batch, then complete) |
| `75dd02b` | Negamax secondary |
| `add3101`, `e050317`, `b6ebd7c` | Empty-board and scaling secondaries, finalist fixtures |
| `944b581`, `294a1d0` | Amendment 2; strict-latency follow-up |
| `9063217`, `513138c` | Tactical audit; amendment 3; equivalence follow-up; replay audit |

Later commits add this report and the final validation log; see
`git log origin/research/mcts-v3`.

## Scope

All work stayed in this worktree and on `origin/research/mcts-v3`. `main`, the
`research/negamax-v3` worktree and branch, AlphaZero worktrees, the external
SSD, Mac Mini services, Tailscale, Render, Docker deployment, live APIs,
`canonical-season-v1` and all earlier evidence directories were not modified.
No paid API was called. The shared benchmark lock was held for every timed
run and full test suite and was never taken from another owner.
