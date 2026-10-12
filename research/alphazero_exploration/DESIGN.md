# Prospective Phase 4D.4D design (written and committed before any results)

Question: does a different self-play action-temperature schedule give more
diverse, informative frozen-checkpoint self-play data without a material
increase in tactical mistakes? No network is trained. Nothing here can show
that a schedule produces a stronger trained agent.

## Fixed conditions

CPU float32, deterministic Torch, intra/inter-op 1/1, five vector thread
variables pinned to 1, the existing M4 virtualenv. Frozen official inference
checkpoints, hash-verified. Pool-eight `collect_batched` from Phase 4D.4C,
256 simulations per move, root noise 0.25 / alpha 1.0, PUCT 1.41, unchanged.
The only varied input is the explicit `TemperatureSchedule`.

| Schedule | Plies at 1.0 | Plies at 0.3 | Plies at 0.0 | Segments |
|---|---|---|---|---|
| A `original` | 0-7 | none | 8+ | `((0,1.0),(8,0.0))` |
| B `extended` | 0-15 | none | 16+ | `((0,1.0),(16,0.0))` |
| C `smoothed` | 0-7 | 8-29 | 30+ | `((0,1.0),(8,0.3),(30,0.0))` |
| D `extended_smoothed` | 0-15 | 16-29 | 30+ | `((0,1.0),(16,0.3),(30,0.0))` |

Plies are zero-based pre-move occupancies. Temperature goes through the
existing `select_action` / `visit_policy` (N^(1/tau), tau=0 uniform over
maximum-visit actions). Policy targets remain normalized raw visits.
`select_action` makes one `rng.choices(k=1)` call per move at every
temperature, so no schedule-dependent draw exists and matched identities share
the same action uniforms (common random numbers).

## Identities and sample size

RNG protocol `connect4-az-independent-games-v1`, experimental generation id 21.
Primary checkpoints: seed 42 generation 20 and seed 314159 generation 20.
Each checkpoint x schedule: two independently seeded cohorts of 64 game ids
0-63 (128 games; 1,024 primary games). Cohort base seeds: 42/g20 4404411 and
4404412; 314159/g20 4404421 and 4404422. All four schedules reuse the same
identities. Cohorts are independent games, not repeated trials. One
`collect_batched` call per checkpoint x schedule x cohort; calls run
cohort-major with rotated schedule order.

Aggregate measured execution (regression + pilot + main + optional supplement,
whole invocations including campaign hashing) stays under 30 minutes.

1. Regression (about 64 games, base seed 4404003): default path and explicit
   `original` must reproduce the retained Phase 4D.4C pool-eight records and
   RNG hashes exactly; each new schedule must reproduce itself exactly.
2. Pilot (base seed 4404400, ids 0-7, both primary checkpoints, all four
   schedules, 64 games): used ONLY for cost. Its games are never analyzed for
   diversity or tactics and never pooled with the main sample.
3. Main size rule: projected main seconds = pilot seconds per game x 1,024
   (conservative: eight-game calls have lower batch occupancy). If the
   projection exceeds 1,800 minus regression minus pilot minus a 60 s margin,
   the cohort size drops to the largest of 48, 32, 24, 16 that fits, for every
   condition alike. No condition is ever expanded after results are seen.
4. Supplement (optional, descriptive only, excluded from every decision rule):
   seed 42 generation 10, two cohorts of 16 (base seeds 4404431/4404432), run
   only if the budget left after the main run is at least 1.25 x its projected
   cost at the main run's measured seconds per game.

A cooperative deadline bounds each invocation. Each complete game is written
when it finishes. A cell interrupted by the deadline is retained as evidence
but excluded: only checkpoint x cohort blocks complete for all four schedules
with identical identity sets are analyzed. Incomplete games are never scored.

## Definitions

A position is a pre-move training example state (both piece sets; the player
to move follows from the piece count). Mirror-canonical identity is the
smaller of a position and its horizontal reflection. Stages are plies 0-7,
8-15, 16-23 and 24+.

- Duplicate fraction over a ply range: 1 - unique positions / positions.
- Pairwise collision probability over a ply range: sum over positions of
  c(c-1) divided by sum over plies of T(T-1), where T is the number of games
  with an example at that ply. It does not depend on the number of games.
- k-ply prefixes: distinct move sequences of length k among games of length
  at least k. Opening family: mirror-canonical position before ply 8.
- Tactics use the project's one-reply oracle scans on the pre-move history.
  W = immediately winning moves, S = safe moves (no immediate opponent win
  afterwards; a winning move is safe), T = opponent immediate threats, L =
  legal moves. Win opportunity: W nonempty; missed if the action is not in W.
  Choice state (the Phase 4D.4A "mandatory response" definition): W empty, S
  nonempty, S smaller than L; unsafe if the action is not in S. Mandatory
  block: W empty and T nonempty; missed if S nonempty and the action is not in
  S. Avoidable one-ply loss: the action is not in S although S is nonempty and
  the opponent wins on the next ply; this is also the game-ending tactical
  error. Tactical state: W nonempty, T nonempty or S smaller than L.
- Attribution of an executed error, with M the maximum-visit actions:
  search-attributable if every action in M is the same kind of error (a
  temperature-0 choice would also err), sampling-attributable if no action in
  M is, tie-ambiguous otherwise. The greedy-counterfactual error rate counts,
  at every opportunity, the share of M that is an error.

## Predeclared endpoints, thresholds and decision rules

Reference is always schedule A at the same checkpoint and identities.
Sampling units are games. Uncertainty is a delete-one-game jackknife (paired
by identity for differences), 95% interval = estimate +/- 1.96 SE. It is
descriptive: three schedules and many endpoints are compared without
multiplicity adjustment, and jackknife intervals for uniqueness statistics
and rare-event rates are approximate.

**Diversity (per checkpoint, 128 matched games).** Primary endpoint D1 is the
exact duplicate-position fraction at plies >= 8. Robustness endpoints: D2 the
mirror-canonical duplicate fraction at plies >= 8, D3 the pairwise collision
probability at plies >= 8.

- MATERIAL: in both primary checkpoints D1 falls by at least 25% relative AND
  at least 5 percentage points, the paired interval of the D1 difference
  excludes zero, D1 is lower in all four checkpoint x cohort blocks, and D2
  and D3 are lower in both checkpoints.
- SUGGESTIVE: D1 falls by at least 10% relative in both checkpoints but the
  MATERIAL conditions are not all met.
- NONE: otherwise.

More examples from longer games are not an improvement: counts are reported,
but classification uses only the normalized endpoints above.

**Tactics (both primary checkpoints pooled, 256 matched games, executed
actions).** Endpoints and regression margins:

- T1 missed-immediate-win rate per opportunity: +5.0 percentage points.
- T2 unsafe-move rate per choice state: +2.0 percentage points.
- T3 executed errors (T1 + T2 numerators) per game: +0.15.
- T4 share of games ended by an avoidable one-ply loss: +5.0 points.

- MATERIAL REGRESSION: for any endpoint the increase is at least its margin
  and the paired interval's lower bound is above zero.
- NO MATERIAL REGRESSION: for every endpoint the interval's upper bound is
  below its margin.
- INCONCLUSIVE: otherwise.

Search-attributable and sampling-attributable errors and the
greedy-counterfactual rates are reported beside the executed totals. A
regression that is purely sampling-attributable is described as a cost of
stochastic execution, not as worse search or a worse network, but it still
counts toward T1-T4 because it changes the collected games and value labels.

**Recommendation.** No composite score. A schedule is eligible only if its
diversity is MATERIAL and its tactics are not MATERIAL REGRESSION. Among
eligible schedules the smallest change wins, ordered B (one existing
parameter, 8 to 16) < C (a new temperature level) < D (both); a larger change
replaces a smaller eligible one only if its D1 is lower by at least 10
percentage points in both checkpoints with paired intervals excluding zero.
If the chosen schedule's tactics are INCONCLUSIVE, the recommendation is
conditional and the follow-up ablation must monitor tactics as a guard. If no
schedule is eligible, the original schedule is retained and a temperature
training ablation is not justified by this evidence. The two-factor main
effects and interaction on D1 are reported descriptively.

**Negative control.** Schedules agree before ply 8, so matched identities
should share their first eight searched examples. Mismatches are counted and
investigated as possible batch-composition rounding, never read as diversity.

## Replay analysis (no training)

Per dataset: unique positions per fixed number of examples (game-order
rarefaction), multiplicity distribution, concentration, mirror-equivalent
positions, tactical-example share, target-entropy distribution, outcome-label
balance, and the expected number of distinct mirror-canonical states in B
uniform draws, sum over states of 1 - (1 - c/T)^B, at an equal B for all
schedules. The archived generation-20 collection (played by the generation-19
network, shared sequential RNG, 256 games) is the historical reference, also
at matched 128-game halves.

Counterfactual replay: the archived seed-matched window (generations 14-20)
plus each 128-game candidate as a PARTIAL generation 21, through the real
`GenerationReplay` (window 8, at most 2,048 games) and its uniform
without-replacement batches of 128. The number of updates is equal for all
schedules: ceil(4 x schedule A's new positions / 128), twenty fixed sampling
seeds. This is a counterfactual about sampling exposure only. It says nothing
about gradients, generalization or playing strength.

## Correctness gates (all mandatory)

Original-schedule exactness against retained 4C evidence; repeat exactness;
exactly 256 simulations per search; frozen weights; legal engine replay of
every trajectory; pre-move example positions; actor-relative outcome signs;
targets equal to normalized raw visits; recorded execution distribution equal
to `visit_policy(visits, temperature)`; temperature equal to the schedule at
every ply; tau-0 actions maximum-visit; no incomplete game in any dataset;
official 150 files byte-identical before and after every invocation.
