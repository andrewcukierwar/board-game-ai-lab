# Phase 4D.4J-B prospective design — tactical root guard confirmation

Confirmation study only. The Phase 4D.4J-A guard
(`research/alphazero_tactical_guard/guard.py`, SHA256
`d2ee54ba1fe9ddbcc4641538e2f5dd89a2435234ff194d6909bed72096632d04`, byte-identical
to its prospective freeze commit `ffb8009`) is imported, never copied or edited.
The 4D.4J-A decision wrapper (`evaluate.decision`/`Agent`) and regression-suite
runner (`evaluate.positions`) are also imported unchanged. No training, self-play,
merge, deployment or promotion. The 4D.4J-A INCONCLUSIVE verdict (+5pp target) is
not reinterpreted; 4D.4J-A results are compared descriptively only, never pooled.

Base `1420cc803bdfa882a3b68a7ae93017f1f0081dbd`, branch/worktree
`phase4d4j-b-tactical-guard-confirmation`. Task origin 2026-10-14 03:57:54 UTC.

## Agents

Phase 4D.4H-C generation-6 CNN seeds 42 (`ba2bde2e…f58b`) and 314159
(`2646d7cd…401d`), loaded read-only through the same strict 4D.4J-A loader.
CONTROL: original CNN PUCT. TREATMENT: identical PUCT plus the frozen root guard.
256 simulations, c=1.41, no root noise, temperature 0, learned policy/value
unchanged, no scaling, no proof propagation. ResNet and value-regularized
checkpoints excluded. Optional generation-20 official CNN OMITTED before freeze:
it needs the official-campaign loader (strict compatibility with this runtime not
verified) and would add ~768 games (~470 CPU-s), competing with the primary
evaluation and review budget.

## Confirmation openings (results/openings.json)

`openings.py`: uniform random legal drops from the empty board with
`seeded_rng('phase4d4j-b-confirmation-openings:ply{L}:attempt{k}', 4420001)`.
Strata L = 4, 6, 8, 10; first 24 accepted per stratum → 96 families.
Rejected: a history that ends the game early, terminal final position, any
immediate winning drop for the side to move OR for the opponent, a self-symmetric
board, and any (actor, mirror-canonical board) duplicate of the 100 development
package rows (40 base families, their mirrors, empty-board rows) or of an earlier
accepted row (this also removes transpositions). Rules come from the independent
matrix reference; the engine replays each prefix. No model, guard, opponent or
outcome input. Confirmation hash `416bfb10917849eb55402a55690d5f3871aa4eee4f36c2c459b0e9fd75059ff5`.
Attempts per stratum: 26/27/37/51.

Ancestry is documented, not hidden: 22 of 96 families pass through a position
equal (mirror-canonically) to a development opening at a shorter ply, and 54
shorter prefix positions are shared between two or more new families. Even-length
prefixes always leave X to move; both agent colors are played from every family.
Full sequences differ from all development rows, but this is not complete
independence. The sealed package is never opened; because it cannot be read,
an accidental coincidence with a sealed opening cannot be excluded, and
generation never used sealed information.

A separate 16-family calibration set (4 per stratum, same rule, namespace
`calibration`) excludes all confirmation and development positions.

## Opponents

A: reference Negamax depth 3 (continuity). B: chosen by the rule in
`results/opponent-selection-rule.json`, written BEFORE calibration. Candidates
in preference order 5, 4. Eligible if (i) ≥0.55 score vs Negamax3 over 32
calibration games, (ii) control-CNN (no guard) score vs candidate in [0.10,0.90]
over 32 calibration games (8 families × 2 colors × 2 seeds), (iii) projected full workload ≤1800 CPU-s.
Calibration (`calibrate.py`, results/calibration.json, 68 s) never ran the guard
or a confirmation opening. Result: Negamax5 vs Negamax3 0.656, control vs
Negamax5 0.266 (vs Negamax3 0.469, vs Negamax4 0.266), projected 964 CPU-s:
**Opponent B = Negamax depth 5.** Negamax5 per-decision cost mean 8.4 ms, worst
24.8 ms (heaviest at 10–16 occupied cells). Deeper nominal depth is not assumed
stronger in every position; the calibration only shows higher aggregate score.

## Match design

2 seeds × 2 arms × 2 opponents × 96 families × 2 colors = **1,536 games**, no
reduction (projection fits). Pair identity: same opening board, agent color,
opponent, namespace `phase4d4j-b-confirmation:negamax{d}`, seed 4420003, and
role-specific `agent_rng` streams for both arms. Trajectories that diverge may
consume later random numbers differently (documented limitation). Four
single-thread tasks run in parallel: (seed, opponent); within a task, control then
guarded matches. The two Negamax3 tasks first rerun the 4D.4J-A
tactical200/solved300/deep128 suites with their original streams (2,512 searches)
as regression checks; they are not new confirmation evidence and must reproduce
4D.4J-A's saved decisions exactly (timings excepted).

## Endpoints and statistics (analyze.py, frozen)

Score win 1, draw 0.5, loss 0. Paired difference per (seed, opponent, family,
color) = guarded − control. Pooled estimate: equal-weight mean of all 768 paired
differences (balanced). Also per seed, per opponent, seed×opponent, color,
stratum (overall and by opponent), opponent-B-minus-A and seed contrasts.
Uncertainty: 10,000 percentile bootstrap replicates, PCG64 seed 4420004,
resampling whole opening families with ALL seeds, opponents, colors and both arms
kept together (shared indices across all conditions); strata resample within
stratum. Games are not treated as independent. Intervals condition on these two
fixed checkpoints, these opponents and this generation process; two seeds do not
estimate training-seed variance; no multiplicity correction; no universal claim.

## Decision hierarchy (margin m = 2pp)

1. FAIL: any reproducible correctness violation (a guarded match decision that
   disagrees with the independent reference, a guarded tactical200 avoidable
   error or missed win, or a first paired decision differing between arms); any
   solved300/deep128 row where guarded exact WDL is below control; non-reproduction
   of 4D.4J-A suite decisions; integrity failure; latency failure; or demonstrated
   material regression (pooled ≤ −2pp with 95% upper < 0, or either opponent
   ≤ −2pp with its upper < 0).
2. CONFIRMED POSITIVE EFFECT: no FAIL, pooled > 0, pooled 95% lower > 0, each
   seed > −2pp, no opponent material regression.
3. Otherwise INCONCLUSIVE. A positive estimate with an interval spanning zero is
   never "confirmed".

Separately: PRACTICALLY MEANINGFUL iff pooled point estimate ≥ +2pp; a confirmed
effect below it is reported as SMALL. Sign disagreement across seeds or opponents
→ qualified interpretation. Latency limits retained from 4D.4J-A: guard p95 ≤ 1 ms,
worst ≤ 5 ms, mean guard/search ≤ 5%.

## Correctness and the non-regression argument

`validate.py` (PASS before freeze) checks the frozen guard on 9,111 decisions from
rules-only continuations of the new openings (plus mirrors and color swaps) and
designed edge cases against the independent matrix reference: immediate wins,
avoid-loss overrides, unchanged safe originals, all-unsafe preservation, legal
returns, both actors, terminal and malformed-input rejection, unchanged visit
vectors, immutable root, guard imports only `numbers`, `time` and the bitboard,
and for both seeds identical visits/priors/logits/value/RNG state/predictions/model
and learner state with the guard on or off.

A. *A replacement never worsens the selected move's optimal-play WDL value.* The
guard changes the original o only in two branches. IMMEDIATE_WIN: the chosen move
wins at once, the maximum value. AVOID_IMMEDIATE_LOSS: no immediate win exists and
o is unsafe, i.e. o does not end the game (a winning o would have triggered the
first branch; a final-cell draw has no replies and is classified safe) and the
opponent then has a drop that completes four. Under optimal play o is therefore a
loss, the minimum value, so any replacement is at least as good. UNCHANGED and
NO_SAFE_MOVE return o. This holds for every position, not only fixtures, provided
the bitboard rules equal Connect 4 rules — checked against the independent
reference above, and each replaced original's witnessed losing reply is replayed.

B. This does not guarantee stronger play against imperfect opponents: an opponent
may miss the immediate win after o, and the replacement may lose later.

C. "Safe" means only that no immediate opponent win exists; a safe move can still
be theoretically lost (e.g. deeper forced sequences).

## Activation, latency, integrity, resources

Every guarded decision's diagnostic is retained. Report calls, immediate-win
activations, avoid-loss overrides, no-safe cases, changes, change rate,
zero-visit selections, by opponent, stratum and phase (occupancy 0–13 / 14–27 /
28–41), and distinct games/families affected. Activation–outcome associations are
descriptive; the paired arms provide the causal contrast. Latency: guard,
classification, search and total clocks; relative overhead; candidates; replies;
by occupancy. Tasks run four-way parallel, so wall timings include contention.

Before measurement: guard/source/runtime/checkpoint freeze (`freeze.json`),
openings, opponents, RNG identities, counts, statistics, rules, resource limits,
validation PASS, independent prelaunch review PASS, commit and push. `manage.py`
requires clean pushed branch, review PASS and frozen identities; one attempt,
no retries, exclusive task directories, compute deadline 04:52:54 UTC (hard
maximum 75 min = 05:12:54 UTC, including review and publication). Cancel: create
`experiment-output/phase4d4j-b/CANCEL`. Inspect: `experiment-output/phase4d4j-b/status.json`
and `tasks/*/progress.json`. An interrupted attempt is not restarted; its evidence
is preserved and reported. After completion: preservation audit, analysis,
independent results review with its own verifier, regression tests, report,
push. No further experiment is launched.

Prelaunch review M2: integrity failures are enforced by `enforce()`/preservation
assertions that abort the run rather than by an analysis branch; an aborted or
integrity-failed run is reported as FAIL / not evaluable, never INCONCLUSIVE.
