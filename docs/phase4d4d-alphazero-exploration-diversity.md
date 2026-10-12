# Phase 4D.4D — Frozen-checkpoint self-play exploration and diversity

Longer or smoother action-temperature schedules make generation-20 self-play
far more diverse. All three tested schedules also regress on the one-reply
tactical endpoints by margins declared before collection, so under the
predeclared rule no schedule is eligible: **keep the original eight-ply
schedule for now; the rule does not justify a temperature-only training
ablation on this evidence.**

That verdict is weaker than its label, for two reasons found by the
independent review and confirmed with the project's exact solver (post hoc,
not part of the rule):

- **The tactical endpoints were a poor proxy.** 86–90% of the counted unsafe
  moves were played by a side already lost under perfect play, and 90–98% of
  the missed immediate wins still won. Errors that change the game-theoretic
  result are 0.043 per game originally and 0.059 / 0.051 / 0.086 under the
  three schedules; no paired difference excludes zero.
- **`smoothed` is knife-edge on both labels.** Its regression label comes from
  one endpoint and disappears if a single game identity is removed; its
  diversity label clears the 25% threshold at 25.5% for seed 42.

| Schedule | Duplicate positions at plies ≥ 8 (seed 42 / seed 314159) | One-reply errors per game (predeclared T3) | Value-changing errors per game (post hoc) | Predeclared diversity label | Predeclared tactics label | Robust to removing one game |
|---|---:|---:|---:|---|---|---|
| A original (reference) | 31.1% / 42.7% | 0.391 | 0.043 | — | — | — |
| B extended | 9.7% / 15.2% | 0.652 | 0.059 | MATERIAL | MATERIAL REGRESSION | Yes |
| C smoothed | 23.2% / 27.0% | 0.578 | 0.051 | MATERIAL | MATERIAL REGRESSION | **No** |
| D extended + smoothed | 8.7% / 12.6% | 0.672 | 0.086 | MATERIAL | MATERIAL REGRESSION | Yes |

The experiment executed correctly: every correctness and execution criterion
passes, 1,024 complete primary games were collected in 16.0 minutes, and the
official campaign is byte-identical. That is a statement about execution. It
is separate from the findings above, and none of it says anything about
trained playing strength: no network was trained.

Machine-readable evidence is in
`research/alphazero_exploration/results/phase4d4d-20261011/`; tools and
reproduction steps are in `research/alphazero_exploration/README.md`.

## Isolation and provenance

Work lives in the `phase4d4d-alphazero-exploration-diversity` worktree at
`experiment-output/phase4d4d-worktree`, created from Phase 4D.4C commit
`99cec1c5f45584145d546b28f19b3d2097b37421`. The historical checkout stays
clean at `1894bc4d8f035cf738df8e3fc0ce234f1ce2c8be`; no other worktree, main,
service or the canonical benchmark was touched. Nothing was trained, resumed,
merged, promoted or deployed.

The official campaign remains
`experiment-output/phase4d3c-alphazero-v2-official-m4-20261008-110950/` with
declaration SHA-256
`c8ab2227d4b2c707b3185da8695a991f0914faede88ecd5f1c7fc84fb017b716`. Every
collection invocation verified the 150-file inventory, the declaration and all
eight inference-checkpoint hashes before inference and re-hashed all 150 files
in `finally`; the analysis did the same around its read-only use of the
archived generation-20 game log and replay storage. Every preservation
record (five collections, two analyses) reports 150 unchanged files.

Runtime: Apple M4 Mac mini (10 cores, 16 GiB), macOS 26.6.2 arm64, CPython
3.11.17, Torch 2.10.0, NumPy 1.26.4, CPU float32, deterministic algorithms,
one intra-op and one inter-op thread, five vector thread variables pinned to
one, existing virtualenv `/Users/andrewcukierwar/.venvs/board-game-alphazero-m4`.
No package was installed.

Checkpoints: seed 42 generation 20 (weights
`f58b6d42…82ac2d`) and seed 314159 generation 20 (`9d4c5a6a…52ef8e`) are
primary; seed 42 generation 10 (`771b577d…2864e5`) is the supplement. Weights
hashes were equal before and after every one of the 52 collection cells.

## Implementation

`collect_batched` gained one optional argument, `temperature_schedule`, a
validated `TemperatureSchedule` of `(first zero-based ply, temperature)` steps.
`None`, the default, takes the unchanged
`action_temperature(ply, config.exploratory_plies)` path. Nothing else in
`batching.py` changed, and `config.py`, `search.py`, `selfplay.py`, `data.py`,
`bitboard.py`, `network.py`, `mcts_nn_agent.py` and `connect4.py` are
byte-identical to `99cec1c`. `V2Config` defaults, PUCT, the 256-simulation
budget, root noise (ε 0.25, α 1.0) and the visit targets are untouched. No
campaign, serial collector or production path can pass a schedule.

A schedule only changes which visited action is executed. Temperature goes
through the existing `select_action` / `visit_policy` mathematics
(`N^(1/τ)`, uniform over maximum-visit actions at τ = 0). `select_action`
makes exactly one `rng.choices(k=1)` call per move at every temperature, so
no schedule-dependent draw exists: matched identities consume identical
search, action and noise streams until their trajectories differ.

| Schedule | τ = 1.0 | τ = 0.3 | τ = 0.0 | Segments |
|---|---|---|---|---|
| A `original` | plies 0–7 | — | 8+ | `((0,1.0),(8,0.0))` |
| B `extended` | plies 0–15 | — | 16+ | `((0,1.0),(16,0.0))` |
| C `smoothed` | plies 0–7 | 8–29 | 30+ | `((0,1.0),(8,0.3),(30,0.0))` |
| D `extended_smoothed` | plies 0–15 | 16–29 | 30+ | `((0,1.0),(16,0.3),(30,0.0))` |

The collection harness records, for every move, the root visits, executed
action, execution temperature, execution distribution, neural root prior and
root value, and for every game a record hash and a final RNG-state hash.

## Prospective design

`research/alphazero_exploration/DESIGN.md` (SHA-256 `42ba03dd32ea…`) was
committed in `9f8db74` at 02:43:46Z and pushed at 02:43:54Z, before the first
measured invocation started (02:43:56Z). It has not changed since, and every
run's provenance records the same hash.

Three things happened that a strict reading of "before any results" must know:

- A harness smoke test ran before the design commit: 10 games and one
  four-second deadline-stop test with a **randomly initialized** network, not
  a campaign checkpoint, written to the session scratch directory. It is not
  part of the evidence or of the measured time.
- The pilot was read for cost only, but the cost script also printed each
  pilot cell's ply total and mean batch occupancy, so game length by schedule
  was visible for 64 pilot games. The design was already committed and pushed.
- The analysis code was committed in `83927d5` at 02:56:18Z, when 9 of 16 main
  cells were on disk. No collected game was opened before that commit; that
  statement cannot be verified from the repository. The decision thresholds in
  the code equal the design clause by clause (checked by the reviewer).

After results were read, two things were added in response to the review and
are labeled post hoc wherever they appear: the exact-value breakdown
(`exact.py`) and the single-identity robustness check. Neither feeds the
decision rule.

The declared design, in brief:

- **Identities.** RNG protocol `connect4-az-independent-games-v1`,
  experimental generation id 21. Each checkpoint × schedule has two
  independently seeded cohorts of game ids 0–63: base seeds 4404411/4404412
  (seed 42) and 4404421/4404422 (seed 314159). All four schedules reuse the
  same identities. Cohorts are independent games, not repeats.
- **Budget rule.** The pilot (base seed 4404400, 64 games, cost only) measured
  0.991 s per game, projecting 1,015 s for the main run against 1,615 s
  available, so the full 64-game cohorts ran unreduced. No condition was
  expanded after results were seen. The pilot's games were never analyzed.
- **Diversity endpoints** (per checkpoint, 128 matched games). D1: duplicate
  fraction of exact positions at plies ≥ 8. D2: the mirror-canonical version.
  D3: pairwise collision probability at plies ≥ 8, which does not depend on
  the number of games. MATERIAL needs, in both checkpoints, a D1 reduction of
  at least 25% relative and 5 points, a paired interval excluding zero, a
  lower D1 in all four cohorts, and lower D2 and D3.
- **Tactical endpoints** (both checkpoints pooled, 256 matched games, executed
  actions) with regression margins: T1 missed-immediate-win rate +5.0 points,
  T2 unsafe-move rate per choice state +2.0 points, T3 errors per game +0.15,
  T4 share of games ended by an avoidable one-ply loss +5.0 points. MATERIAL
  REGRESSION: any endpoint rises by at least its margin with the paired
  interval's lower bound above zero.
- **Recommendation rule.** No composite score. Eligible means MATERIAL
  diversity and not MATERIAL REGRESSION; the smallest eligible change wins;
  with no eligible schedule the original is kept and a temperature training
  ablation is not justified.

Games are the sampling units. Intervals are delete-one-game jackknife 95%
intervals, paired by game identity for differences.

## Execution and budget

| Invocation | Cells | Complete games | Seconds | Result |
|---|---:|---:|---:|---|
| Regression | 10 | 64 | 60.2 | complete |
| Pilot (cost only) | 8 | 64 | 64.4 | complete |
| Main | 16 | 1,024 | 961.4 | complete |
| Supplement (42/g10) | 8 | 128 | 108.1 | complete |
| Regression, delivered source (post-review) | 10 | 64 | 61.0 | complete |
| **Total** | **52** | **1,344** | **1,255.0 (20.9 min)** | under the 30-minute limit |

Seconds are whole invocations, including both 150-file hash passes. No
measured invocation had an error, a deadline stop or an incomplete game. The
supplement ran because the pre-declared rule was met: 714 s remained against
a 150 s requirement. The last row re-ran the regression after the review
hardened `TemperatureSchedule` validation (name characters; `from_dict` error
type): all ten cells equal the first regression in records and RNG hashes.
The change is validation only, and the delivered collector behaves identically
on those 64 games; the main identities were not re-collected with it.

Main collection performance, pool eight:

| Schedule | Games | Plies | Simulations | Forward calls | Mean batch | Games/s | Positions/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 256 | 8,448 | 2,162,688 | 218,036 | 7.57 | 1.035 | 34.17 |
| extended | 256 | 7,695 | 1,969,920 | 205,162 | 7.46 | 1.102 | 33.13 |
| smoothed | 256 | 8,391 | 2,148,096 | 218,057 | 7.53 | 1.036 | 33.95 |
| extended_smoothed | 256 | 7,761 | 1,986,816 | 210,480 | 7.36 | 1.098 | 33.27 |
| **All** | **1,024** | **32,295** | **8,267,520** | **851,735** | **7.48** | **1.067** | **33.64** |

The main run evaluated 6,373,168 neural positions in 959.95 s of collection
wall time. Process peak RSS was 294.5 MiB in the main run and 330.0 MiB at
most (regression). Throughput per position is the same across schedules; the
extended schedules finish more games per second only because their games are
shorter.

## Correctness and reproducibility

| Criterion | Result | Evidence |
|---|---|---|
| Original schedule matches Phase 4D.4C exactly | PASS | Default path and explicit `original` reproduce the retained 4C pool-eight trial (seed 314159, 16 games, records and RNG hashes) and the 4C validation games (seed 42, 4 games, records only; default and explicit RNG states equal) |
| Delivered source behaves as the measured source | PASS | Post-review regression: all ten cells equal the first regression in records and RNG hashes; the diff is validation only |
| Same checkpoint/config/identity reproduces | PASS | Each new schedule collected twice: identical records and final RNG states |
| No schedule-dependent RNG draws | PASS | 1,024 matched prefix comparisons, 0 mismatches; unit test of one action draw per move |
| Exactly 256 simulations per search | PASS | Checked per decision online; all 32,295 stored root-visit vectors sum to 256 |
| Network frozen | PASS | Weights hash equal before and after all 52 cells; checkpoint file hashes verified |
| Search, noise, defaults and target code unchanged | PASS | `git diff --quiet 99cec1c` for eight source files |
| Targets are normalized raw visits; temperature affects execution only | PASS | Every recorded execution distribution equals `visit_policy(visits, τ)`; every τ = 0 action is maximum-visit |
| Pre-move positions, actor-relative outcomes, legal replay | PASS | All 1,024 main games replayed through the engine; 1,152 main and supplement games rebuilt and matched to the collector's record hashes |
| No incomplete game in the dataset | PASS | 4 of 4 checkpoint × cohort blocks complete; 0 cells excluded |
| Every schedule applied correctly by ply | PASS | Every recorded temperature compared with its schedule |
| Mirror and canonical counting correct | PASS | Reproduces Phase 4D.4A's generation-20 unique positions, mirror families, duplicate fractions, trajectories, openings and tactical counts for both seeds, and the campaign's own replay-window unique-state diagnostics |
| Cancellation and partial failure safe | PASS | Unit tests: a deadline stop leaves only complete labeled games; the stopped cohort is excluded whole |
| Official campaign preserved | PASS | 150 files byte-identical and declaration hash unchanged in every invocation; re-hashed independently by the reviewer |
| All declared primary cells completed | PASS | 16 of 16 cells, 1,024 complete games, no deadline stop |
| Measured execution under 30 minutes | PASS | 1,255.0 s |

**Negative control.** Schedules agree before ply 8, so matched identities must
share their first eight searched examples; B and D must share sixteen. All
1,024 comparisons match exactly (positions, root visits and actions). No
batch-composition rounding difference reached a discrete decision in these
prefixes, so none is misattributed to exploration. After ply 8 the schedules
change play as intended: only 24 / 41 / 14 of 128 seed-42 games and 8 / 16 / 3
of 128 seed-314159 games remain identical to the original under B / C / D.

## Diversity results

128 matched games per cell. "Duplicates" is 1 − unique / positions.

| Checkpoint | Schedule | Positions | Mean length | Unique trajectories | Unique positions | Duplicates, all plies | Mirror-canonical duplicates | **D1: duplicates, plies ≥ 8** | D3: collision, plies ≥ 8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42/g20 | original | 3,922 | 30.64 | 110 | 2,240 | 42.9% | 45.8% | **31.1%** | 0.0169 |
| | extended | 3,715 | 29.02 | 127 | 2,674 | 28.0% | 31.0% | **9.7%** | 0.0046 |
| | smoothed | 3,919 | 30.62 | 122 | 2,468 | 37.0% | 41.4% | **23.2%** | 0.0130 |
| | extended_smoothed | 3,736 | 29.19 | 127 | 2,721 | 27.2% | 30.0% | **8.7%** | 0.0043 |
| 314159/g20 | original | 4,526 | 35.36 | 103 | 2,181 | 51.8% | 55.1% | **42.7%** | 0.0373 |
| | extended | 3,980 | 31.09 | 126 | 2,683 | 32.6% | 35.1% | **15.2%** | 0.0183 |
| | smoothed | 4,472 | 34.94 | 124 | 2,691 | 39.8% | 44.0% | **27.0%** | 0.0259 |
| | extended_smoothed | 4,025 | 31.45 | 126 | 2,799 | 30.5% | 33.0% | **12.6%** | 0.0172 |

Paired differences from the original, percentage points, with jackknife 95%
intervals:

| Checkpoint | Schedule | D1 | Relative D1 change | D2 | D3 | D1 by cohort (schedule vs original) |
|---|---|---:|---:|---:|---:|---|
| 42/g20 | extended | −21.4 [−28.2, −14.5] | −68.7% | −21.3 [−27.8, −14.8] | −1.23 [−1.90, −0.56] | 3.6 vs 18.7; 8.9 vs 24.2 |
| | smoothed | −7.9 [−12.7, −3.2] | −25.5% | −6.0 [−11.6, −0.5] | −0.39 [−0.72, −0.06] | 16.0 vs 18.7; 15.6 vs 24.2 |
| | extended_smoothed | −22.4 [−29.5, −15.3] | −72.0% | −22.5 [−29.3, −15.8] | −1.26 [−1.95, −0.58] | 3.6 vs 18.7; 7.0 vs 24.2 |
| 314159/g20 | extended | −27.6 [−35.3, −19.8] | −64.5% | −28.7 [−36.3, −21.1] | −1.90 [−2.92, −0.87] | 10.5 vs 30.5; 12.9 vs 34.1 |
| | smoothed | −15.7 [−22.2, −9.2] | −36.7% | −14.5 [−20.7, −8.3] | −1.13 [−2.07, −0.19] | 18.9 vs 30.5; 22.6 vs 34.1 |
| | extended_smoothed | −30.2 [−37.1, −23.2] | −70.6% | −31.2 [−37.9, −24.4] | −2.00 [−3.01, −1.00] | 10.0 vs 30.5; 10.9 vs 34.1 |

All three schedules meet every MATERIAL condition. `smoothed` clears the 25%
relative threshold narrowly at seed 42 (25.5%).

Stage-conditioned duplicate fraction (unique / positions):

| Checkpoint | Schedule | Plies 0–7 | Plies 8–15 | Plies 16–23 | Plies 24+ |
|---|---|---:|---:|---:|---:|
| 42/g20 | original | 76.5% (239/1,017) | 36.7% (594/939) | 32.7% (534/794) | 25.5% (873/1,172) |
| | extended | 76.5% (239/1,017) | 21.3% (741/942) | 5.8% (695/738) | 1.9% (999/1,018) |
| | smoothed | 76.5% (239/1,017) | 35.3% (609/941) | 25.8% (601/810) | 11.5% (1,019/1,151) |
| | extended_smoothed | 76.5% (239/1,017) | 21.3% (741/942) | 3.8% (717/745) | 0.8% (1,024/1,032) |
| 314159/g20 | original | 82.9% (175/1,024) | 50.8% (496/1,008) | 42.5% (521/906) | 37.7% (989/1,588) |
| | extended | 82.9% (175/1,024) | 31.8% (686/1,006) | 7.9% (723/785) | 5.7% (1,099/1,165) |
| | smoothed | 82.9% (175/1,024) | 46.7% (538/1,009) | 28.5% (640/895) | 13.3% (1,338/1,544) |
| | extended_smoothed | 82.9% (175/1,024) | 31.8% (686/1,006) | 6.2% (744/793) | 0.7% (1,194/1,202) |

Three things follow from the stage view.

- **The gain is not a length artifact.** The extended schedules produce fewer
  positions (shorter games) yet more unique ones, and the gain holds inside
  every ply bin and in the count-free D3.
- **Extension does most of the work.** The two-factor contrasts on D1 are an
  extension main effect of −17.9 [−23.8, −12.1] and −21.0 [−26.6, −15.4]
  points, a smoothing main effect of −4.5 [−7.3, −1.7] and −9.1
  [−12.7, −5.6], and a positive interaction of +6.9 [+2.5, +11.3] and +13.1
  [+6.0, +20.2]: once extended, adding τ = 0.3 changes D1 by about one to
  three points.
- **No schedule touches the opening.** Plies 0–7 are 76.5% and 82.9%
  duplicates under every schedule, by construction.

Prefixes, families and repeated trajectories:

| Checkpoint | Schedule | Unique 8-ply | 12-ply | 16-ply | 20-ply | 24-ply prefixes / eligible games | Families before ply 16 (top share) | Most repeated trajectory |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42/g20 | original | 79/121 | 79/119 | 76/110 | 75/101 | 69/92 | 67 (8.2%) | 4× |
| | extended | 79/121 | 101/119 | 105/111 | 89/92 | 81/84 | 96 (2.8%) | 2× |
| | smoothed | 79/121 | 81/119 | 82/112 | 85/103 | 78/91 | 72 (9.9%) | 3× |
| | extended_smoothed | 79/121 | 101/119 | 105/111 | 92/93 | 85/85 | 96 (2.8%) | 2× |
| 314159/g20 | original | 64/128 | 69/126 | 75/121 | 75/112 | 80/110 | 62 (21.5%) | 5× |
| | extended | 64/128 | 95/126 | 114/120 | 94/97 | 85/87 | 104 (5.1%) | 2× |
| | smoothed | 64/128 | 70/126 | 85/121 | 89/111 | 93/107 | 72 (11.6%) | 2× |
| | extended_smoothed | 64/128 | 95/126 | 114/120 | 95/97 | 89/90 | 104 (5.1%) | 2× |

Opening-family concentration before ply 8 is identical across schedules
(negative control) and is itself narrow: 64 mirror-canonical families with a
14.2% top family at seed 42, and 51 families with a **42.2%** top family at
seed 314159. Every trajectory that repeats most begins with five moves in the
centre column. Under the original schedule the most repeated complete games
are 42-ply games played four and five times in 128.

Policy statistics at plies ≥ 8 (means over examples):

| Checkpoint | Schedule | Target entropy (nats) | Max target probability | Execution entropy | Executed = top visit | Prior entropy | Max prior |
|---|---|---:|---:|---:|---:|---:|---:|
| 42/g20 | original | 0.696 | 0.749 | 0.001 | 100.0% | 1.025 | 0.598 |
| | extended | 0.726 | 0.735 | 0.287 | 91.2% | 1.056 | 0.586 |
| | smoothed | 0.704 | 0.745 | 0.161 | 94.3% | 1.031 | 0.596 |
| | extended_smoothed | 0.739 | 0.729 | 0.397 | 86.3% | 1.064 | 0.581 |
| 314159/g20 | original | 0.727 | 0.731 | 0.001 | 100.0% | 1.029 | 0.584 |
| | extended | 0.801 | 0.703 | 0.318 | 89.2% | 1.097 | 0.565 |
| | smoothed | 0.747 | 0.722 | 0.186 | 92.3% | 1.044 | 0.582 |
| | extended_smoothed | 0.805 | 0.702 | 0.440 | 84.3% | 1.109 | 0.559 |

Targets become slightly less peaked under the extended schedules (target
entropy +0.031 [+0.002, +0.060] and +0.075 [+0.041, +0.109] nats for B) and
the network's priors are slightly flatter on the positions reached, consistent
with less familiar positions. Under `smoothed` the target-entropy change is
not distinguishable from zero. Full quantiles, per-stage values and the
distributions of priors are in `results.json`.

Game length falls under extension: −1.62 [−3.38, +0.14] plies at seed 42 and
−4.27 [−6.39, −2.14] at seed 314159 for B; `smoothed` leaves it unchanged
(−0.02 and −0.42).

## Tactical quality

### Predeclared one-reply endpoints

Tactics are measured on the actions actually executed, using the project's
one-reply oracle scans. A *win opportunity* has an immediately winning move.
A *choice state* has no immediate win, at least one safe move and at least
one move that hands the opponent an immediate win. Both primary checkpoints
are pooled: 256 matched games per schedule.

| Measure | original | extended | smoothed | extended_smoothed |
|---|---:|---:|---:|---:|
| Positions | 8,448 | 7,695 | 8,391 | 7,761 |
| Tactical states (share of positions) | 1,934 (22.9%) | 1,867 (24.3%) | 2,028 (24.2%) | 1,951 (25.1%) |
| States with at least one safe move | 8,306 | 7,552 | 8,252 | 7,621 |
| Immediate-win opportunities | 213 | 261 | 238 | 262 |
| **T1 missed immediate wins** | 29 (13.6%) | 49 (18.8%) | 49 (20.6%) | 50 (19.1%) |
| Choice states | 1,504 | 1,412 | 1,569 | 1,490 |
| **T2 unsafe moves with a safe alternative** | 71 (4.72%) | 118 (8.36%) | 99 (6.31%) | 122 (8.19%) |
| Mandatory-block situations (blockable) | 575 (495) | 601 (513) | 661 (573) | 631 (542) |
| Missed mandatory blocks | 40 (8.1%) | 79 (15.4%) | 59 (10.3%) | 77 (14.2%) |
| **T3 errors per game** | 0.391 | 0.652 | 0.578 | 0.672 |
| **T4 games ended by an avoidable one-ply loss** | 57 (22.3%) | 89 (34.8%) | 67 (26.2%) | 95 (37.1%) |

Unique tactical positions rise with diversity (seed 42: 780 / 926 / 869 / 954;
seed 314159: 737 / 886 / 948 / 968): the new schedules expose more distinct
tactical states.

Paired differences from the original and the predeclared classification:

| Schedule | T1 (margin +5.0) | T2 (margin +2.0) | T3 (margin +0.15) | T4 (margin +5.0) | Classification |
|---|---:|---:|---:|---:|---|
| extended | +5.2 [−2.2, +12.6] | **+3.64 [+1.72, +5.56]** | **+0.262 [+0.122, +0.402]** | **+12.5 [+5.6, +19.4]** | MATERIAL REGRESSION |
| smoothed | +7.0 [−1.7, +15.7] | +1.59 [+0.03, +3.15] | **+0.188 [+0.027, +0.348]** | +3.9 [−1.7, +9.5] | MATERIAL REGRESSION |
| extended_smoothed | +5.5 [−2.6, +13.5] | **+3.47 [+1.41, +5.53]** | **+0.281 [+0.116, +0.446]** | **+14.8 [+7.4, +22.3]** | MATERIAL REGRESSION |

Bold cells meet both parts of the regression rule. T1 is inconclusive for
every schedule; roughly 240 opportunities per schedule cannot resolve a
five-point change.

Pooling was predeclared, and the two checkpoints differ. At seed 42 no
tactical endpoint excludes zero for any schedule; the pooled regression of
the two extended schedules is carried by seed 314159.

| Checkpoint | Schedule | T1 | T2 | T3 | T4 |
|---|---|---:|---:|---:|---:|
| 42/g20 | extended | +4.6 [−4.1, +13.2] | +2.28 [−0.48, +5.03] | +0.188 [−0.014, +0.389] | +6.3 [−3.7, +16.2] |
| | smoothed | +7.4 [−1.5, +16.2] | +1.66 [−0.87, +4.20] | +0.188 [−0.029, +0.404] | +2.3 [−5.0, +9.7] |
| | extended_smoothed | +6.6 [−3.1, +16.2] | +1.84 [−1.29, +4.96] | +0.203 [−0.050, +0.456] | +6.3 [−4.3, +16.8] |
| 314159/g20 | extended | +6.0 [−7.0, +19.1] | +4.85 [+2.20, +7.50] | +0.336 [+0.141, +0.531] | +18.8 [+9.3, +28.2] |
| | smoothed | +6.7 [−10.2, +23.5] | +1.62 [−0.29, +3.53] | +0.188 [−0.051, +0.426] | +5.5 [−3.0, +14.0] |
| | extended_smoothed | +4.3 [−9.8, +18.3] | +4.98 [+2.30, +7.67] | +0.359 [+0.147, +0.572] | +23.4 [+13.1, +33.8] |

### Search preference versus stochastic selection

Each executed error is attributed by the maximum-visit action set: *search*
if every maximum-visit action is also an error (a τ = 0 choice would have
erred too), *sampling* if none is. "Search" here is self-play search with
root Dirichlet noise on, not evaluation-mode PUCT.

| Schedule | One-reply errors | Search-attributable | Sampling-attributable | Tie | Game-ending errors: search / sampling |
|---|---:|---:|---:|---:|---:|
| original | 100 | 99 | 1 | 0 | 56 / 1 |
| extended | 167 | 126 | 41 | 0 | 61 / 28 |
| smoothed | 148 | 131 | 16 | 1 | 58 / 8 (+1 tie) |
| extended_smoothed | 172 | 112 | 60 | 0 | 59 / 36 |

- **Sampling.** Under B, 41 of 167 errors are sampling-attributable against 1
  of 100 originally, and they sit where τ = 1 was added: unsafe moves at plies
  8–15 rise from 12 to 45 and missed wins there from 2 to 15. In B's τ = 1
  choice states 46 of 221 executed moves (20.8%) were one-reply unsafe.
  τ = 0.3 samples far fewer: 16 sampling-attributable errors under C.
- **Search preference.** Search-attributable errors also rise (99 → 126 / 131
  / 112), and the greedy-counterfactual rates, which ask what a maximum-visit
  choice would have done at every opportunity, move the same way: unsafe
  4.69% → 6.30% / 6.05% / 5.91% (paired +1.62 [+0.02, +3.21], +1.37
  [−0.11, +2.84], +1.22 [−0.53, +2.96] points) and missed win 13.6% → 16.5% /
  18.9% / 13.7% (all intervals include zero). This is suggestive only.
  Temperature does not change the search, so if real it is a shift in which
  positions are reached.

### Post hoc: what the one-reply events mean for the game result

This subsection was added after the independent review. It is not a
predeclared endpoint and does not enter the decision rule.

The one-reply scan counts an unsafe move even when every move loses under
perfect play, and a missed win even when the executed move still wins. The
project's exact solver (`oracle.BitboardSolver`, 6,000,000-node budget; all
370 positions solved; the reviewer cross-checked 60 late positions against
the independent exhaustive solver) separates those cases.

| Pooled, 256 games | original | extended | smoothed | extended_smoothed |
|---|---:|---:|---:|---:|
| Unsafe moves (T2 numerator) | 71 | 118 | 99 | 122 |
| …mover already lost before the move | 62 (87%) | 104 (88%) | 89 (90%) | 105 (86%) |
| …value-changing | 9 | 14 | 10 | 17 |
| Missed immediate wins (T1 numerator) | 29 | 49 | 49 | 50 |
| …executed move still wins | 27 (93%) | 48 (98%) | 46 (94%) | 45 (90%) |
| Games ended by an avoidable one-ply loss (T4) | 57 | 89 | 67 | 95 |
| …mover already lost before the move | 50 | 76 | 59 | 82 |
| …value-changing | 7 (2.7%) | 13 (5.1%) | 8 (3.1%) | 13 (5.1%) |
| …value-changing: search / sampling | 7 / 0 | 5 / 8 | 7 / 1 | 5 / 8 |
| **Value-changing errors per game** | **0.043** | **0.059** | **0.051** | **0.086** |
| Paired difference from original | — | +0.016 [−0.029, +0.060] | +0.008 [−0.031, +0.047] | +0.043 [−0.027, +0.113] |
| Value-changing game endings, paired (points) | — | +2.3 [−0.9, +5.6] | +0.4 [−1.3, +2.1] | +2.3 [−0.9, +5.6] |

Read with the exact values, the predeclared regression is mostly more
value-neutral events: delayed wins, and already-lost sides losing one move
sooner. Search backs up undiscounted values, so it is indifferent to the
length of a win or of a loss. T4 therefore does not measure games decided by
a blunder: 7 of the 57 original-schedule endings changed the result, and 13
of 89 under B.

Three cautions keep this from being an all-clear.

- **Neutral under perfect play is not neutral in practice.** A theoretically
  lost position may not be lost against a 256-simulation opponent, so giving
  it up one move early can still change the realized outcome and its labels. Under
  B, 36 of the 104 already-lost unsafe moves fall at plies 8–15, against 6 of
  62 originally.
- **Sampling can discard a result.** Sampled moves turned a not-lost position
  into an immediate loss in 8 of 256 games under B and D and 1 under C. The
  original schedule cannot sample after ply 8, so its 0 is partly definitional;
  the net change in value-changing endings is the +2.3 points [−0.9, +5.6]
  in the table.
- **Only one-reply-flagged events were solved.** Whether sampled moves change
  exact values through deeper errors is unmeasured. Outcome balance does shift
  under extension (next section), which suggests they do.

### Robustness of the labels to single games

Also post hoc. Removing the one identity with the largest error increase:

| Schedule | Predeclared tactics label | Label without that identity | T3 without it | Delete-one replicates below the 25% D1 threshold (42 / 314159) |
|---|---|---|---:|---:|
| extended | MATERIAL REGRESSION | MATERIAL REGRESSION | +0.239 | 0 of 128 / 0 of 128 |
| smoothed | MATERIAL REGRESSION | **INCONCLUSIVE** | +0.141 | **14 of 128** / 0 of 128 |
| extended_smoothed | MATERIAL REGRESSION | MATERIAL REGRESSION | +0.243 | 0 of 128 / 0 of 128 |

`smoothed` trips only T3, and one game identity (seed 314159, cohort 0, game
23: 12 errors against 0, all of them value-neutral) carries it over the
margin. Without that game `smoothed` would be eligible and the rule would
recommend it conditionally. Its diversity label is equally thin at seed 42:
relative D1 reduction 25.5% [13.0%, 37.9%], and the absolute interval's lower
bound (3.2 points) is under the 5-point threshold. The predeclared rule tests
point estimates against thresholds, so both labels stand as declared, but
neither would be expected to replicate reliably. The two extended schedules'
labels do not depend on any single game.

## Replay-distribution analysis

No training was run. These figures describe what a replay buffer would be
sampling from; they do not show better gradient updates, generalization or
playing strength. A duplicated position still carries a separately searched
target and outcome, so "distinct exposure" is a diversity proxy only.

Per 128-game dataset. Rarefaction takes whole games in random order up to a
fixed example count (the smallest schedule total at that checkpoint), so
schedules are compared at equal collected examples. The equal sampling budget
is four draws per example of that same smallest total.

| Checkpoint | Dataset | Examples | Unique mirror-canonical states | Examples in duplicated states | Unique positions at equal examples | Expected distinct states at equal draws | Effective states, plies ≥ 8 (Shannon) |
|---|---|---:|---:|---:|---:|---:|---:|
| 42/g20 | original | 3,922 | 2,125 | 56.9% | 2,144 | 2,087 | 1,498 |
| | extended | 3,715 | 2,562 | 37.3% | 2,674 | 2,519 | 2,141 |
| | smoothed | 3,919 | 2,298 | 51.1% | 2,356 | 2,255 | 1,670 |
| | extended_smoothed | 3,736 | 2,614 | 35.5% | 2,704 | 2,569 | 2,198 |
| | archived g20, 128-game halves | 4,303 / 4,290 | 2,073 / 2,241 | 64.4% / 57.9% | 1,971 / 2,092 | 2,024 / 2,184 | 1,386 / 1,335 |
| 314159/g20 | original | 4,526 | 2,032 | 66.9% | 1,979 | 1,987 | 1,111 |
| | extended | 3,980 | 2,585 | 41.1% | 2,683 | 2,542 | 1,903 |
| | smoothed | 4,472 | 2,503 | 53.6% | 2,432 | 2,444 | 1,602 |
| | extended_smoothed | 4,025 | 2,696 | 38.1% | 2,767 | 2,648 | 2,019 |
| | archived g20, 128-game halves | 4,456 / 4,374 | 2,382 / 2,192 | 55.7% / 59.9% | 2,254 / 2,133 | 2,326 / 2,146 | 1,380 / 1,149 |

Equal examples are 3,715 (seed 42) and 3,980 (seed 314159); equal draws are
14,860 and 15,920. At equal collected examples the extended schedules yield
25–40% more unique positions than the original.

The archived generation-20 collection is the historical reference. It was
played by the generation-19 learner (verified from the campaign's weight
hashes) with the campaign's shared sequential RNG, so it is a reference and
not a matched control. Its 256-game duplicate fractions at plies ≥ 8 are 48.2%
and 43.7%; at the matched size of 128 games they are 37.2% and 36.6%, the same
range as this experiment's original-schedule 31.1% and 42.7%. The duplicate
fraction grows with the number of games, so the 256-game historical figures
must not be compared directly with the 128-game figures here; the count-free
collision probability is 0.024 and 0.038 historically against 0.017 and 0.037
here.

Other replay-relevant properties:

| Property | original | extended | smoothed | extended_smoothed |
|---|---:|---:|---:|---:|
| Tactical-state share of examples (42 / 314159) | 24.1% / 21.8% | 25.3% / 23.3% | 24.6% / 23.7% | 25.9% / 24.4% |
| Win-opportunity share (42 / 314159) | 3.0% / 2.1% | 3.7% / 3.1% | 3.3% / 2.4% | 3.8% / 3.0% |
| Mean target entropy, all plies (42 / 314159) | 0.701 / 0.699 | 0.724 / 0.751 | 0.707 / 0.715 | 0.733 / 0.754 |
| Near-one-hot targets, max ≥ 0.9 (42 / 314159) | 36.4% / 33.6% | 35.4% / 30.6% | 36.6% / 32.3% | 35.0% / 30.6% |
| Positions with both mirror orientations present (42 / 314159) | 115 / 149 | 112 / 98 | 170 / 188 | 107 / 103 |
| Drawn games (42 / 314159) | 21.1% / 35.2% | 14.1% / 20.3% | 21.9% / 30.5% | 12.5% / 21.9% |
| First-player wins (42 / 314159) | 54.7% / 48.4% | 58.6% / 65.6% | 53.9% / 53.1% | 59.4% / 64.8% |
| Draw-labeled examples (42 / 314159) | 28.9% / 41.8% | 20.3% / 27.4% | 30.0% / 36.6% | 18.0% / 29.2% |

**Outcome labels shift under extension.** Draws fall by roughly a third to
two-fifths and, at seed 314159, first-player wins rise from 48.4% to about 65%.
Games are also decided earlier. `smoothed` leaves the outcome balance close to
the original. A changed label distribution is a change in what the value head
would be trained on. This experiment cannot say whether it is closer to or
further from correct values. The value-changing one-reply events are too few
to account for the shift; the roles of value-neutral early concessions and of
deeper errors are unmeasured.

**Counterfactual replay sampling.** For each seed the archived window
(generations 13–20) was loaded read-only into the real `GenerationReplay`
(window 8, at most 2,048 games) and each 128-game candidate was added as a
**partial** generation 21, which evicts generation 13 exactly as the runner
would. This is counterfactual: the declared generation size is 256 games, and
the real generation 21 was never collected. Updates are equal for all
schedules at `ceil(4 × schedule A's new positions / 128)`: 123 batches (seed
42) and 142 (seed 314159) of 128 uniform draws, over twenty fixed sampling
seeds.

| Checkpoint | Schedule | Candidate states not already in the window | Per 1,000 candidate examples | Distinct states among all draws (mean ± sd) | Novel states drawn (mean ± sd) |
|---|---|---:|---:|---:|---:|
| 42/g20 | original | 1,448 | 369 | 8,346 ± 70 | 353 ± 15 |
| | extended | 1,952 | 525 | 8,417 ± 77 | 429 ± 12 |
| | smoothed | 1,581 | 403 | 8,367 ± 65 | 371 ± 16 |
| | extended_smoothed | 2,034 | 544 | 8,424 ± 71 | 448 ± 19 |
| 314159/g20 | original | 1,612 | 356 | 9,438 ± 47 | 532 ± 25 |
| | extended | 2,256 | 567 | 9,556 ± 48 | 620 ± 23 |
| | smoothed | 2,062 | 461 | 9,513 ± 49 | 598 ± 23 |
| | extended_smoothed | 2,361 | 587 | 9,567 ± 49 | 633 ± 18 |

Under the extended schedules a candidate generation contributes 35–46% more
states that the window does not already hold (42–65% more per 1,000 candidate
examples). Seen from the whole window the effect is small, about one percent
more distinct states among a generation's draws, because a 128-game
generation is 5.8–7.4% of a 61,000–65,000-position window. A sustained schedule
change would act over all eight generations; that compounding was not
simulated and would itself be a counterfactual about networks that do not
exist. Under the existing rule (updates proportional to
new positions) the shorter extended games would also receive fewer updates:
117 against 123, and 125 against 142.

## Supplement: seed 42, generation 10

Descriptive only and excluded from every decision rule: 32 games per schedule.
D1 is 9.4% / 1.5% / 3.3% / 1.6% for A / B / C / D, so the earlier checkpoint
has far less repetition to remove. Unsafe moves are 3/257, 9/187, 7/249 and
9/168, and games ended by an avoidable one-ply loss are 3, 9, 7 and 9 of 32.
The directions match the primary result. With 32 games per schedule the
diversity intervals include zero (D1 −7.9 [−17.2, +1.4] points for B), while
the extended schedules' increase in game-ending errors does not (+18.8
[+2.4, +35.1] points).

## Decision under the predeclared rules

| Schedule | Diversity | Tactics | Eligible |
|---|---|---|---|
| B extended | MATERIAL | MATERIAL REGRESSION (T2, T3, T4) | No |
| C smoothed | MATERIAL | MATERIAL REGRESSION (T3 only) | No |
| D extended + smoothed | MATERIAL | MATERIAL REGRESSION (T2, T3, T4) | No |

No schedule is eligible, so the rule returns the original schedule and states
that a temperature training ablation is not justified by this evidence. The
thresholds were not revisited, and the post hoc analyses above do not change
this output. They change how much weight it deserves.

The tradeoff for each schedule, with both readings:

- **B extended.** The largest diversity gain per unit of change (D1 down
  two-thirds, robust to any single game). A clear and robust regression on
  the one-reply proxy, concentrated at seed 314159. On exact values: +0.016
  value-changing errors per game [−0.029, +0.060], with eight games in 256
  where a sampled move threw away a not-lost position.
- **C smoothed.** The smallest diversity gain (D1 down a quarter to a third,
  mostly at plies 16+) and the smallest tactical movement. Both of its labels
  are knife-edge.
- **D extended + smoothed.** Almost the same diversity as B, slightly more
  errors on both the proxy and the exact reading. The added τ = 0.3 buys one
  to three points of D1.

## Statistical limitations

- Two checkpoints from two training seeds, both generation 20. Results need
  not transfer to other generations, budgets or architectures; the 32-game
  generation-10 supplement suggests much smaller effects earlier in training.
- The predeclared tactical endpoints are one-reply proxies. They were chosen
  for comparability with Phase 4D.4A and turned out to count mostly
  value-neutral events. The exact-value reading that corrects this is post
  hoc: it was chosen after seeing the results, covers only one-reply-flagged
  events, and needs prospective confirmation on fresh games.
- Jackknife intervals are descriptive. They are approximate for uniqueness
  statistics and for rare-event rates. Three schedules were compared on four
  tactical endpoints under an "any endpoint" rule with no multiplicity
  adjustment, which makes a regression label easier to trip; `smoothed`'s
  single tripped endpoint (z ≈ 2.3 unadjusted; sign test 40 of 65 nonzero
  paired differences positive, p ≈ 0.08) is the case this matters for.
- The rule compares point estimates with thresholds. A label can therefore
  rest on an estimate that sits on a threshold, as `smoothed`'s do.
- Games sharing an identity share their first eight plies, which the paired
  analysis exploits; the two cohorts per checkpoint are the only independent
  replication of each contrast, and cohort values are point comparisons.
- Duplicate fractions depend on the number of games. Every comparison here is
  at 128 games per cell; the count-free collision probability is reported
  alongside.
- T1 is underpowered: about 240 win opportunities per schedule.
- The search/sampling attribution describes each error locally. It does not
  apportion the downstream consequences of an earlier sampled move.
- All collection used pool eight. Batch composition differs between schedules
  after ply 8, so numerical rounding may differ at the 1e-5 level measured in
  Phase 4D.4C; no such difference reached a discrete decision in the shared
  prefixes, and none can be separated from exploration after them.
- Forking paths, disclosed in the design section: a pre-design smoke test with
  an untrained network, pilot game lengths visible in the cost printout, and
  analysis code committed while the main collection was running.
- Diversity is not strength. Nothing here measures learning.

## Tests and independent review

The complete relevant regression suite passes: **854 passed, 2
optional skips** in 277.08 seconds, including **78 new tests** (49 for
the scheduled collector, 29 for the analysis, replay, exact-value, decision
and packaging tools). The skips remain the optional retained-v1-checkpoint
checks absent from the isolated worktree. The command is in the tooling
README and the complete output in `tests-final.txt`.

A separate reviewer agent, which did not write any of the code, inspected the
collector diff and tools, loaded the `99cec1c` collector beside the current
one, recomputed every headline number from the raw game files with its own
code, re-hashed the official campaign twice and ran the focused tests. It was
not allowed to edit the repository. Its full review and the disposition of
each finding are in `REVIEW.md`. In short:

- **Verified with no discrepancy:** default-path equality with Phase 4D.4C;
  one action draw and one noise draw per move at every temperature; all raw
  data contracts over 32,295 positions; every record and RNG hash of all 1,280
  games then collected; the negative control; game counts, D1, D2, tactical
  counts, all twelve pooled paired intervals, the replay window figures; the
  design predating collection and matching the coded thresholds; the 150-file
  inventory; the budget.
- **Material finding 1 — overclaim.** Most counted tactical errors are
  value-neutral under the exact solver. Addressed: `exact.py` reproduces the
  reviewer's figures independently, the breakdown is in the report and
  tables, and the interpretation, recommendation and next experiment were
  rewritten. The predeclared decision output is unchanged.
- **Material finding 2 — robustness.** `smoothed`'s regression label rests on
  one game and its diversity label on a 25.5% estimate against a 25%
  threshold. Addressed: a single-identity robustness check is in the tools and
  the report, and the headline table marks it.
- **Minor findings addressed:** per-checkpoint tactical differences shown
  beside the pooled ones; the pre-design smoke test, pilot printout and
  analysis-commit timing disclosed; the ablation plan corrected for what the
  existing runner can and cannot store; direct tests added for the diversity thresholds,
  the escalation rule, the compressed-evidence loader and packaging; record
  hashes now verified by the analysis; schedule validation hardened and the
  delivered source re-verified against Phase 4D.4C evidence.

The reviewer then checked the fixes and this rewritten report in a second
read-only pass. It found the final decision identical to the pre-review one,
the post hoc figures equal to its own, and the delivered analysis reproducing
`decision.json`, `acceptance.json` and `TABLES.md` exactly from the committed
compressed evidence. **No material finding remains unresolved.** Its remaining
minor item, that a warm-started `extended` arm cannot be loaded by the
existing boundary code, and five wording notes are addressed in this text.

This is an independent engineering review, not third-party certification or
replication on separate hardware. No threshold, margin or rule was changed
after results.

## Recommendation and next experiment

**Temperature.** Keep the original schedule (τ = 1 for plies 0–7, then 0) for
any collection that feeds training now. That is the predeclared rule's output
and nothing here overturns it.

**Training ablation.** Not justified by the predeclared rule. The rule's
verdict rests on a proxy that the exact-value check shows to be mostly
value-neutral, so the honest position is that the evidence neither shows
exploration to be tactically harmful nor shows it to be safe. Whether to
proceed anyway is a judgment for the project owner. The recommended route is
to settle it cheaply first.

**Next: one confirmatory frozen-checkpoint run, no training.** Fresh game
identities, the same two checkpoints and budget, thresholds declared in
advance as here, comparing `original` with `extended`.

- *Why `extended`.* It has the largest diversity gain, its labels do not
  depend on any single game, and it is the only tested schedule the existing
  `V2Config` can express: it is `exploratory_plies = 16`. `smoothed` and the
  combined schedule need a temperature the config cannot represent.
- *Primary tactical endpoint.* Value-changing moves per game from the exact
  solver, over every executed move at plies ≥ 8 that the solver can decide
  within a declared node budget, with unsolved positions reported and handled
  by a rule fixed in advance. The Python solver decided the 370 flagged
  positions here in under two minutes; unflagged early positions are harder,
  so the ply range or budget must be piloted for cost first.
- *Secondary.* The one-reply endpoints, for continuity; the realized
  outcome balance; D1–D3.
- *If the budget allows,* a milder extension (τ = 1 through ply 11, or τ = 0.5
  on plies 8–15) as a third arm, since the sampled value-changing losses sit
  at plies 8–15.

Opening concentration is a separate lever that no tested schedule touches: at
seed 314159, 42% of games share one family before ply 8.

**The training ablation to run if the confirmatory run finds no material
value-changing regression.** Two arms, `original` versus `extended`, nothing
else varied:

- Identical initialization: both arms start from the same archived
  generation-20 development boundary of each seed, with identical optimizer
  state and replay window, in fresh output directories under their own
  declaration. The official campaign is never resumed.
- **Prerequisite runner change.** The existing boundary loader takes one
  `exploratory_plies` from the stored config and rebuilds every replay-window
  game against it (`generation.load_resume_boundary`, `rebuild_game`). A warm
  start therefore cannot switch an arm to 16 plies, and a window mixing 8-ply
  and 16-ply generations cannot be reloaded: all archived generation-20 games
  are rejected under 16 and 248 of 256 `extended` games under 8. The ablation
  needs the schedule stored per generation and validated per generation, as a
  separately tested change. Only a from-scratch run avoids it, and from
  scratch the narrowing this phase is about has not yet developed.
- Equal collected data, reported both ways: equal games per generation and
  equal positions per generation, since extension shortens games.
- Equal wall-clock budget per arm. The campaign runner uses the serial
  collector; using pool eight for both arms means wiring the batched collector
  into an exploratory runner first, which is its own small, separately
  verified change.
- Explicit training-update budgets: the existing
  `ceil(4 × new positions / 128)` rule and a fixed-updates variant, because
  the existing rule gives shorter games fewer updates.
- Three to five generations per arm, both seeds, resumable at generation
  boundaries with the source- and runtime-checked boundary format extended as
  above.
- Development evaluation on matched positions and opponents for every arm and
  generation: network-only and full-PUCT accuracy on the 200 tactical rows,
  solved-position value error, and paired-colour arena games against the
  established baselines (generation 0, Negamax depths, classic and tactical
  MCTS) on a fixed opening set.
- Self-play diagnostics per generation with the tools from this phase: D1–D3,
  the one-reply and exact-value endpoints, and the outcome-label balance.
- Decision thresholds declared before launch, with tactical accuracy as a
  guard that diversity cannot buy back.

Replay sampling frequency, tactical search changes and new architectures stay
out of that ablation and are tested separately.

## Summary

1. **Diversity.** All three schedules materially reduce repeated positions
   after ply 8: from 31.1% / 42.7% to 9.7% / 15.2% (extended), 23.2% / 27.0%
   (smoothed) and 8.7% / 12.6% (extended + smoothed). Paired intervals exclude
   zero in both checkpoints and the reduction appears in all four cohorts.
   Extension to ply 16 does most of it. The first eight plies stay as
   concentrated as before.
2. **Tactical quality.** One-reply errors rise from 0.39 to 0.65, 0.58 and
   0.67 per game, a material regression under the predeclared margins for all
   three. Post hoc, 86–90% of the unsafe moves were played from already-lost
   positions and 90–98% of missed wins still won; value-changing errors move
   from 0.043 to 0.059, 0.051 and 0.086 per game with every paired interval
   including zero. Sampling did discard a not-lost position in 8 of 256 games
   under the extended schedules.
3. **Temperature recommendation.** Keep the original schedule for now.
4. **Training experiment.** Not justified by the predeclared rule. Run one
   confirmatory frozen-checkpoint comparison of `original` and `extended` with
   an exact-value endpoint declared in advance; if it shows no material
   value-changing regression, run the two-arm ablation specified above.
5. **Unresolved.** Whether more varied self-play with more value-neutral
   one-reply errors helps or harms learning cannot be answered without
   training. Deeper value-changing errors were not measured, and the outcome
   balance does shift under extension. `smoothed`'s labels would not be
   expected to replicate. Everything rests on two generation-20 checkpoints
   at 128 games per cell.
