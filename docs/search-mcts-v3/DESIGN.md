# MCTS v3 — predeclared experimental design

Declared on October 10, 2026, at starting SHA
`60b99b0d42145da149f433907585b809060c946d`, before any v3 strength game was
played. Later amendments are appended under "Amendments" with a date and a
reason; the text above them is not rewritten after results exist.

## Question

Can Connect Four MCTS choose better moves for the same amount of computation?
The v2 work made each simulation 53–72× cheaper without changing the
algorithm. This study changes the algorithm and asks whether strength per
millisecond improves, not whether simulations per second do.

## Baseline

The production `MCTSAgent` at this SHA, unmodified and imported directly:
UCB1 with `c = 1.41`, uniformly random expansion, uniformly random rollouts,
reward for the player who made the incoming move, most-visited root child, and
root-only immediate-win / safe-reply guards. It stays the opponent in every
primary comparison.

## Candidate variants

All candidates live in a research-only agent class that is not registered in
the agent factory or the API. Each component is an explicit configuration
switch. With every switch at its default the research agent must reproduce the
baseline's moves, tree statistics, and RNG consumption exactly; that parity is
a test, and an A/A match is run as a harness control.

| Id | Component | Change | Basis |
| --- | --- | --- | --- |
| R1 | Decisive rollouts | In a rollout, play an immediate win; otherwise block the opponent's single immediate win; two simultaneous opponent wins end the rollout as a loss. Otherwise uniform. | Teytaud & Teytaud 2010, decisive and anti-decisive moves |
| R2 | R1 + gift avoidance | As R1, and uniform choice excludes columns whose move would let the opponent win directly on top, unless every column does. | Common Connect Four rollout heuristic |
| S | Solver + tactical expansion | Nodes carry a proven win/draw/loss value. Terminal children, nodes with an immediate win for the mover, and nodes with two immediate opponent wins are proven on creation; a single opponent threat restricts expansion to the block. Proofs back up minimax-style, proven-lost children are never selected, the search stops when the root is proven, and a proven-winning root move is always played. | Winands, Björnsson & Saito 2008, MCTS-Solver |
| E | Centre-first expansion | Expand untried moves in the existing centre-first order instead of uniformly at random. | Expansion-policy prior |
| C | Exploration constant | `c` ∈ {0.5, 0.7, 1.0, 2.0} instead of 1.41. | UCT tuning |

Supporting literature: Baier & Winands 2013 ("Monte-Carlo Tree Search and
Minimax Hybrids") report in Connect Four that shallow minimax in rollouts and
in backpropagation both beat MCTS-Solver; R1/R2 and S are the cheapest
knowledge-free members of that family. Figures from that paper were read from
a search summary of its slides and are motivation only, not evidence here.

Not candidates in this study: neural networks, transposition tables, RAVE
(column-indexed AMAF is poorly matched to gravity moves), tree reuse between
moves (changes the stateless agent contract), and compiled kernels.

## Hypotheses

- **H-R:** tactical rollouts give higher score per simulation, and the gain
  survives their lower throughput at equal wall-clock time.
- **H-S:** proving and pruning tactical lines in the tree gives higher score at
  near-zero cost per simulation.
- **H-E:** centre-first expansion is a small free gain at low budgets.
- **H-C:** some `c` other than 1.41 scores above 50% against the baseline.
- **H-comb:** the accepted components combine without cancelling.

## Positions

Uniform legal-column random histories, as in v2, at 2, 5, 8, 11, 14, 17, 20
and 23 plies in equal numbers, generated from master seed `2026101003` with
SHA-256 domain separation. Rejected only when: the history is terminal; the
resulting board duplicates one already drawn in either set; or the position is
decided within two plies by the root guards every agent shares (the mover has
an immediate win, or the opponent has two immediate wins and the mover none).
Those decided positions always give a pair score of exactly 0.5, so they carry
no information; this is a deliberate change of estimand from v2, which kept
them. No rejection uses any agent's search, result, or timing.

- **Development set** (`dev`): 96 openings, 12 per length. Used for all
  pilots, tuning, throughput calibration, and candidate selection.
- **Held-out set** (`holdout`): 256 openings, 32 per length. Used only for the
  confirmatory phase, only after finalists and their budgets are frozen and
  committed. Never used for tuning or selection.
- **Empty board:** a separate secondary study from the real starting position
  with 64 independent seed pairs (domain `empty`).

This is an artificial balanced mixture of early, middle, and late positions.
It is not human play, solved play, or tournament strength.

## Seeds, pairing, colours

Each opening carries two domain-separated 128-bit role seeds (challenger,
opponent). Each opening is played twice with colours swapped; role seeds follow
the agent, and a fresh `random.Random` per agent persists through the game.
The two colour games of an opening form one cluster and are never counted as
independent. Role seeds are reused across conditions on the same opening.
Negamax is deterministic and ignores its seed.

## Budgets and the equal-time comparison

Baseline budgets: **400** (public preset) and **2,000** (v2's recommended
balanced tier) are primary. 100, 800 and 5,000 are secondary.

Two comparisons are made at each baseline budget `B`:

- **Equal simulations:** candidate at `B` versus baseline at `B`.
- **Equal time:** candidate at `B' = floor(min(1, 0.90 · r_B) · B)` versus
  baseline at `B`, where `r_B` is baseline total search wall time divided by
  candidate total search wall time over the development-set equal-simulation
  games at budget `B`. The 0.90 factor is a deliberate handicap so the
  candidate ends up no slower. Fixed simulation counts keep games reproducible,
  which a wall-clock deadline would not.

Time parity is then verified, not assumed: in every equal-time match the
realised ratio (candidate total search wall ÷ baseline total search wall, same
games) is reported. A per-millisecond claim requires that ratio to be ≤ 1.00.

All timed games run serially under the shared benchmark lock. Fixed-budget
results (W–D–L) are deterministic given seeds and do not depend on load;
timings do, so timings from any run that overlapped a competing heavy workload
are discarded and rerun.

## Phases

**Pilot 1 — single components, equal simulations (dev).** A/A control, R1, R2,
S, E, and C at four values, each versus the baseline at 400 and 2,000; 96
openings × 2 colours = 192 games per condition per budget.

**Pilot 2 — single components, equal time (dev).** Every structural component
(R1, R2, S, E) whose pooled pilot-1 score is ≥ 52% with neither budget below
48% is replayed at its `B'`. If both rollout levels pass, the one with the
higher pooled equal-time score becomes "R".

**Pilot 3 — combination and constant (dev).** The combination of all
components that scored above 50% pooled at equal time is played against the
baseline at equal simulations and at equal time. The exploration constant is
re-swept on that combination ({0.5, 0.7, 1.0, 2.0} head-to-head against the
combination at 1.41, equal simulations), because the best constant can depend
on the rollout policy. A constant replaces 1.41 only if its pooled head-to-head
score is ≥ 52%.

Pilots eliminate weak variants. They are exploratory, share one small
development set, involve many looks, and support no strength claim.

**Finalist selection.** At most two finalists: the two highest pooled
equal-time pilot scores among configurations that differ in at least one
structural component (rollout, solver, expansion) and whose point estimate
exceeds 50% at both budgets. Constant variants of one structure count once,
represented by their best pilot constant. Finalist configurations and their
`B'` values are committed before any held-out game.

**Confirmatory — primary (holdout).** For each finalist and each
`B` ∈ {400, 2,000}: equal simulations and equal time versus the baseline, 256
openings × 2 colours = 512 games per comparison. At most 8 primary comparisons.

**Confirmatory — secondary (holdout, exploratory).**
- Versus Negamax depth 4 and 6 (and 8 if preflight cost permits): baseline at
  `B` and each finalist at `B'`, first 96 held-out openings (12 per length).
  Reported as each agent's score against the common opponent and the paired
  difference between candidate and baseline.
- Budget scaling at 100, 800 and 5,000 for the best finalist, equal time,
  first 128 held-out openings.
- Empty-board study: best finalist at `B'` versus baseline at 400 and 2,000.

## Analysis

Per opening, average the two colour games into one pair score. Resample whole
openings within opening-length strata, 10,000 deterministic bootstrap
replicates, percentile intervals. Report W–D–L, score, per-colour score,
unadjusted 95% intervals, and for primary comparisons a Bonferroni-adjusted
interval for a family of 8 (99.375%), fixed at 8 even if fewer comparisons
run. A two-sided sign-flip randomisation test on per-opening advantages is
reported as a sensitivity check. No Elo conversion. Secondary results carry
unadjusted intervals and no significance claim. No opening, game, or loss is
dropped; errors are recorded as incomplete attempts, never as losses.

## Decision gates

Per finalist and budget tier:

- **Accepted research improvement** — all of: correctness tests pass and seeded
  runs reproduce; equal-time score has adjusted lower bound above 50% and point
  estimate ≥ 55%; realised time ratio ≤ 1.00; no nominally significant (95%)
  loss to the baseline in the paired Negamax comparison; memory per decision
  no more than 1.5× the baseline's at the same budget.
- **Promising but inconclusive** — point estimate above 50% at equal time but a
  gate above is not met, or time parity was not achieved.
- **Rejected** — equal-time point estimate ≤ 50%, or a correctness failure that
  is not fixable within the variant's definition.
- **Incomplete** — the declared games did not finish.

Acceptance here means "validated research result on this position mixture and
this laptop". It is not a production change. Public presets, API caps, the
production agent, and provenance versions are not modified in this study.

## Compute budget

Hard caps on cumulative search wall time: pilots 60 minutes, primary
confirmatory 75 minutes, secondary 60 minutes, any follow-up experiment 45
minutes. Each lock hold targets ≤ 12 minutes; runs are resumable at game
boundaries and the lock is released between batches. Sample sizes are not
enlarged after seeing outcomes; if a cap is hit the phase is reported as
incomplete.

## Preflight

Before pilot 1: unit and property tests for every variant; a throughput and
Negamax-cost measurement on fixed fixtures to project run time; and a 4-opening
smoke run of the harness on a separately seeded `preflight` domain excluded
from all inference.

## Amendments

### Amendment 1 — 2026-10-10, after pilot 2, before pilot 3

Pilot 2 gave pooled equal-time scores of 59.65% for R1 and 59.25% for R2, a
difference far inside the noise of 96 openings, while pilot 1 showed R2
clearly ahead per simulation (64.7% versus 60.3% pooled). The declared rule
names R1 as "R". Discarding R2 on a 0.4-point pilot difference would throw
away information, so pilot 3 is widened, on the development set only:

- 3a: both R1+S and R2+S versus the baseline at equal simulations.
- 3b: both at their equal-time budgets.
- 3c: the exploration-constant sweep is run on whichever combination has the
  higher pooled equal-time score in 3b, instead of on R1+S unconditionally.
- 3d: if a constant replaces 1.41, that configuration is replayed against the
  baseline at equal simulations and then at equal time.

Unchanged: the finalist rule, the cap of two finalists, the held-out set, the
family of 8, the gates, and the compute caps. The cost is more looks at the
development set, which is why only held-out results support claims.
