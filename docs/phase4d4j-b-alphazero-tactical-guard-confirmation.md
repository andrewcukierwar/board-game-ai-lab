# Phase 4D.4J-B — Tactical root guard: independent strength confirmation

**Strength verdict (frozen rules): CONFIRMED POSITIVE EFFECT — SMALL.** On 96 newly generated development opening families and two opponents, the unchanged 4D.4J-A root guard raised match score by **+1.302 percentage points**, paired 95% interval **[+0.521, +2.344]**. The interval excludes zero, both seeds and both opponents point positive, and no paired game got worse. The estimate is **below the prospectively declared +2pp practical margin**, so this is a small effect, not a practically meaningful strength gain.

| Question | Result |
|---|---|
| Guard unchanged from 4D.4J-A | YES — SHA256 `d2ee54ba…2d04`, imported, never edited |
| Correctness (8,933 match decisions, regression suites) | PASS — 0 reference mismatches; tactical200 400/400 |
| Exact-WDL non-regression (solved300, deep128) | PASS — 0 regressions; 8 improved rows (4 per seed) |
| Strength on new openings | CONFIRMED POSITIVE, SMALL (+1.30pp; practical margin +2pp not reached) |
| Latency | PASS — mean 0.065 ms, p95 0.097 ms, worst 0.171 ms |
| Integrity / preservation | PASS (see below; results review in [independent-results-review.md](../research/alphazero_tactical_guard_confirmation/independent-results-review.md)) |

## What was tested

This phase is a confirmation study, not algorithm development. The guard runs after unchanged 256-simulation PUCT. It takes an immediate win if one exists, otherwise keeps a safe PUCT choice, otherwise picks the most-visited safe move, and keeps PUCT's move when nothing is safe. It does not modify network outputs, priors, tree, backups, simulations, exploration, root noise or terminal handling. The 4D.4J-A INCONCLUSIVE verdict (+5pp target) stands as recorded; its results are compared descriptively only and are not pooled here.

- **Agents:** Phase 4D.4H-C generation-6 CNN, seeds 42 (`ba2bde2e…f58b`) and 314159 (`2646d7cd…401d`). Both use the strict 4D.4J-A loader, read-only, with model and learner state verified unchanged in every task. Search settings: PUCT 256 simulations, c=1.41, no root noise, deterministic selection. The control arm uses the identical wrapper with the guard off.
- **Excluded before freeze:** ResNet, value-regularized checkpoints, and the optional generation-20 official CNN. The generation-20 checkpoint needs the official-campaign loader, which was not verified strict-compatible, and would have added about 768 games.
- **Openings:** 96 new families, 24 each after plies 4/6/8/10 ([openings.json](../research/alphazero_tactical_guard_confirmation/results/openings.json), hash `416bfb10…59ff5`).
  - Generation is uniform random legal play under a frozen domain-separated RNG (seed 4420001). It rejects: games that end early, terminal positions, any immediate win for either side, self-symmetric boards, and mirror-canonical duplicates of the 100 development rows or of earlier accepted rows.
  - Generation used no model, guard, opponent or outcome input.
  - **Shared ancestry is disclosed:** 22 families pass through a development-opening position at a shorter ply, and 54 shorter prefixes are shared between new families. These families are new, but not fully independent.
  - The sealed package was never opened, so an accidental coincidence with a sealed opening cannot be ruled out. No sealed information was used.
- **Opponents:** A is reference Negamax depth 3 (continuity). B is **Negamax depth 5**, picked by a rule declared before calibration ([rule](../research/alphazero_tactical_guard_confirmation/results/opponent-selection-rule.json), [calibration](../research/alphazero_tactical_guard_confirmation/results/calibration.json)).
  - Calibration used 16 separate calibration families and the control CNN only; the guard never ran.
  - Negamax5 scored 0.656 against Negamax3. The control CNN scored 0.266 against Negamax5, which is not saturated (0.469 against Negamax3).
  - Negamax5 cost per decision: mean 8.4 ms, worst 24.8 ms, heaviest at 10–16 occupied cells.
- **Design:** 2 seeds × 2 arms × 2 opponents × 96 families × 2 colors = **1,536 games**, with no workload reduction.
  - Both arms share namespace, seed 4420003, opening, color and role-specific RNG streams.
  - Every first paired decision was identical across arms (768/768).
  - Once trajectories diverge, later random numbers may be consumed differently.

## Prospective controls

- **Freeze and launch order:** everything was frozen and pushed at **`ae4715ff30d6796b010f115185ad90514e346141`** before any measured game: [DESIGN.md](../research/alphazero_tactical_guard_confirmation/DESIGN.md), [design.json](../research/alphazero_tactical_guard_confirmation/results/design.json), [freeze.json](../research/alphazero_tactical_guard_confirmation/results/freeze.json) (82 source/evidence hashes, runtime, checkpoints), and the focused [validation](../research/alphazero_tactical_guard_confirmation/results/validation.json).
- **Analysis dry run:** a synthetic, model-free dry run of the frozen analysis preceded the freeze.
- **Independent prelaunch review:** [review](../research/alphazero_tactical_guard_confirmation/independent-prelaunch-review.md) returned PASS with no blockers.
  - Minor M1, a calibration game count misstated in DESIGN.md, was corrected before freeze.
  - Minor M2: integrity is enforced by run-aborting checks, so an aborted run would have been FAIL, not INCONCLUSIVE.
- **Supervisor:** one detached managed attempt with four parallel single-thread tasks, an absolute compute deadline, cooperative CANCEL, no retries and exclusive task directories. All four tasks completed in 293 s, then analysis took 11 s. No undeclared games were run.

## Primary result — paired match score

Win 1, draw 0.5, loss 0. Uncertainty: 10,000 percentile bootstrap replicates (PCG64 seed 4420004), resampling whole families with all seeds, opponents, colors and arms kept together; strata resample within stratum.

| Comparison | Control | Guarded | Guarded − control (pp) | 95% interval |
|---|---:|---:|---:|---:|
| **Pooled (768 pairs, 96 families)** | 43.685% | 44.987% | **+1.302** | **[+0.521, +2.344]** |
| Seed 42 | 45.312% | 46.094% | +0.781 | [+0.000, +1.823] |
| Seed 314159 | 42.057% | 43.880% | +1.823 | [+0.521, +3.385] |
| vs Negamax3 | 50.521% | 52.344% | +1.823 | [+0.521, +3.516] |
| vs Negamax5 | 36.849% | 37.630% | +0.781 | [+0.130, +1.562] |
| 42 × Negamax3 | 54.427% | 55.469% | +1.042 | [+0.000, +2.604] |
| 42 × Negamax5 | 36.198% | 36.719% | +0.521 | [+0.000, +1.562] |
| 314159 × Negamax3 | 46.615% | 49.219% | +2.604 | [+0.781, +4.948] |
| 314159 × Negamax5 | 37.500% | 38.542% | +1.042 | [+0.000, +2.344] |
| Agent as X | 51.302% | 52.604% | +1.302 | [+0.260, +2.865] |
| Agent as O | 36.068% | 37.370% | +1.302 | [+0.260, +2.865] |

W/D/L per seed × opponent (control → guarded):

| Seed × opponent | Control W/D/L | Guarded W/D/L |
|---|---|---|
| 42 × N3 | 98/13/81 | 100/13/79 |
| 42 × N5 | 62/15/115 | 63/15/114 |
| 314159 × N3 | 86/7/99 | 90/9/93 |
| 314159 × N5 | 63/18/111 | 64/20/108 |

By opening stratum (pooled; N3 / N5 in brackets):

| Stratum | Difference (pp) | 95% interval | N3 | N5 |
|---|---:|---:|---:|---:|
| ply 4 | +0.521 | [+0.000, +1.562] | +1.04 | +0.00 |
| ply 6 | +1.562 | [+0.000, +4.167] | +3.13 | +0.00 |
| ply 8 | +1.042 | [+0.000, +2.604] | +0.52 | +1.56 |
| ply 10 | +2.083 | [+0.000, +4.688] | +2.60 | +1.56 |

**How concentrated the evidence is.**
- Only **12 of 768 paired games** had different outcomes, and **all 12 favored the guarded arm** (8 loss→win, 4 loss→draw).
- They come from **8 of 96 families**, every one net positive.
- Games without an override are move-for-move identical across arms, so the whole contrast comes from the 78 games that contained a changed decision.
- The interval excludes zero because no family went the other way. With so few informative clusters, the percentile interval rests on a handful of families and should be read as approximate.

**Heterogeneity.**
- Negamax5 minus Negamax3: −1.04pp [−2.60, +0.26]. The guard helped less against the stronger opponent, but the opponents are not clearly different.
- Seed 314159 minus seed 42: +1.04pp [−0.52, +2.86].
- Every seed, opponent, cell and stratum estimate is ≥ 0, so there is no sign disagreement and no qualified interpretation is triggered.
- The gain is identical as X and O (the colors are paired within families).

**Frozen decision rules.**
- Pooled lower bound > 0. Neither seed ≤ −2pp. No opponent shows a material regression. No correctness, WDL, reproduction, integrity or latency failure. Verdict: **CONFIRMED_POSITIVE_EFFECT**.
- Pooled +1.302 < +2.0pp: **not practically meaningful; reported as SMALL.**
- Intervals condition on these two fixed checkpoints, these opponents and this generation process. Two seeds do not estimate training-seed variance. No multiplicity correction was applied. No universal strength claim is made.

**Descriptive comparison with 4D.4J-A** (not pooled): 4D.4J-A measured +2.19pp [+0.63, +4.38] against Negamax3 on 40 development families. Here the Negamax3-only estimate is +1.82pp [+0.52, +3.52]. The direction repeats and the size is similar or slightly smaller; the gain against the stronger Negamax5 is smaller (+0.78pp).

## Tactical correctness and exact-value regression

- **Validation before freeze:** 9,111 decisions on rules-only continuations of the new openings (plus mirrors and color swaps) and designed edge cases, all matching the independent matrix rules reference. Coverage includes immediate wins, avoid-loss overrides, unchanged safe originals, all-unsafe positions, legal returns, both actors, terminal and malformed-input rejection, unchanged visits, an immutable root, and guard imports limited to `numbers`/`time`/bitboard.
- **Preservation:** for both seeds, visits, priors, logits, value, RNG state, predictions, weights and learner state are identical with the guard on or off. The guard draws no RNG and makes no inference.
- **In the 1,536 matches:** all 8,933 guarded decisions were re-derived from the actual reconstructed board by the independent reference, with 0 mismatches. Every played move equals the guard's final action, and every original visit vector is unchanged.
- **Regression suites** (reused, regression only, not new evidence): all 2,512 tactical200/solved300/deep128 control and guarded decisions reproduced 4D.4J-A's saved records exactly (non-timing fields).

| Suite | Seed 42 control → guarded | Seed 314159 control → guarded |
|---|---|---|
| tactical200 correct | 187 → 200 (avoidable 10→0, missed wins 3→0) | 186 → 200 (13→0, 1→0) |
| solved300 optimal | 229 → 233, 0 regressions | 233 → 237, 0 regressions |
| deep128 optimal | 98 → 98, unchanged | 96 → 96, unchanged |

**The non-regression argument (A), from the rules rather than from fixtures.** The guard replaces PUCT's move o only in two cases. If it takes an immediate win, the new move has the maximum value. If o is unsafe with no win available, then o does not end the game: a winning o would have hit the first branch, and a final-cell draw has no replies so it counts as safe. The opponent then has a drop completing four, so o is a forced loss under optimal play, the minimum value, and any replacement is at least as good. The UNCHANGED and NO_SAFE_MOVE branches return o. Validation replayed the witnessed losing reply for all 1,128 replacements, and the bitboard rules matched the independent reference.

**(B)** This does not guarantee stronger play against imperfect opponents: an opponent can miss the reply that punishes o, and the replacement can still lose later.

**(C)** A "safe" move only avoids an immediate loss; it can still be theoretically lost. For example, 394 match decisions were NO_SAFE_MOVE, where every move already loses at once.

## Guard activation in the new games

| Scope | Calls | Immediate-win | Avoid-loss overrides | No safe move | Changed | Changed rate |
|---|---:|---:|---:|---:|---:|---:|
| All | 8,933 | 317 | 76 | 394 | 94 | 1.052% |
| vs Negamax3 | 4,333 | 190 | 45 | 172 | 55 | 1.269% |
| vs Negamax5 | 4,600 | 127 | 31 | 222 | 39 | 0.848% |
| Seed 42 | 4,383 | 163 | 37 | 193 | 46 | 1.050% |
| Seed 314159 | 4,550 | 154 | 39 | 201 | 48 | 1.055% |
| Opening (0–13 occupied) | 2,615 | 12 | 32 | 43 | 32 | 1.224% |
| Middle (14–27) | 4,245 | 90 | 27 | 162 | 32 | 0.754% |
| Late (28–41) | 2,073 | 215 | 17 | 189 | 30 | 1.447% |
| Stratum ply 4 / 6 / 8 / 10 | 2,548 / 2,334 / 2,103 / 1,948 | 100 / 65 / 66 / 86 | 5 / 13 / 33 / 25 | 76 / 108 / 112 / 98 | 9 / 20 / 38 / 27 | 0.35 / 0.86 / 1.81 / 1.39% |

- 94 changed decisions: 76 avoided an immediate losing reply and 18 recovered a missed immediate win.
- They fell in **78 distinct games across 44 families**. No zero-visit move was ever selected.
- Changed games went from 17W/0D/61L (control) to 25W/4D/49L (guarded).
- These activation–outcome counts are descriptive; the paired arms give the causal contrast.
- The change rate (1.05%) is slightly higher than 4D.4J-A's 0.78% (15/1,932). Against Negamax5 the guard intervened less often and the opponent more often left no safe move.

## Latency

Guard clock (nanosecond, CPU-local) is separate from inference. Four tasks ran in parallel, so these figures include contention.

| Measure (ms) | Mean | Median | p95 | Worst |
|---|---:|---:|---:|---:|
| Original PUCT search | 49.222 | 59.497 | 68.607 | 141.980 |
| Guard classification | 0.057 | 0.064 | 0.089 | 0.154 |
| Whole guard | **0.065** | 0.072 | **0.097** | **0.171** |
| Total decision | 49.289 | 59.574 | 68.696 | 142.065 |

- Mean guard/search overhead was 0.229% (median 0.135%, p95 0.837%). The single worst ratio, 11.5%, was one decision with a very fast search.
- Limits (p95 ≤ 1 ms, worst ≤ 5 ms, mean relative ≤ 5%): **PASS**.
- Per decision the guard inspected 5.46 candidate moves on average (max 7) and 32.4 opponent replies on average (max 49).
- Guard time by occupancy (mean ms): 0–6: 0.089; 7–13: 0.088; 14–20: 0.080; 21–27: 0.055; 28–34: 0.031; 35–41: 0.016.
- Overhead is slightly above 4D.4J-A's sequential run (0.050 / 0.072 / 0.103 ms), consistent with the parallel tasks, and remains negligible.

## Independent review, preservation and resources

- **Independent review:** a separate prelaunch reviewer and a separate results reviewer, neither of whom wrote the runner or analyzer. The results reviewer wrote its own rules engine and verifier ([independent_verify.py](../research/alphazero_tactical_guard_confirmation/independent_verify.py), [output](../research/alphazero_tactical_guard_confirmation/results/independent-verification.json), [review](../research/alphazero_tactical_guard_confirmation/independent-results-review.md)). This is same-host verification by separate agents, not external replication.

  **Results review disposition: PASS WITH NOTES.**
  - The verifier replayed all 1,536 games and independently re-derived all 8,933 guarded and 8,649 control decisions and all 94 overrides, with 0 mismatches.
  - It confirmed that the 78 divergent pairs are exactly the 78 games with a guard change, and reproduced every point estimate exactly.
  - With its own bootstrap stream (PCG64 99020001) the pooled interval was again [+0.521, +2.344]. Only the Negamax5 upper bound differed (+1.693 vs +1.562), which is sampling variation.
  - It also reproduced activation, latency, suite and hash checks, and confirmed exactly 4 declared tasks with 0 retries. Planted errors (wrong override, wrong score, unsafe move marked safe, altered move) were all caught.

  **Retained notes** (not defects):
  - Nine of the 12 outcome-changing pairs are seed 314159 and nine are against Negamax3. The positive lower bound rests on 8 families (family sign test p ≈ 0.008), so it confirms the direction more firmly than the size.
  - Seed 42's interval lower bound is exactly 0, and the Negamax5 effect is zero in the ply-4 and ply-6 strata.
  - Most of the 78 changed games are still lost.
  - The design limits still apply: two checkpoints, two opponents, partly shared opening ancestry, parallel-task latency.
- **Preservation:** [preservation.json](../research/alphazero_tactical_guard_confirmation/results/preservation.json) and [final-audit.json](../research/alphazero_tactical_guard_confirmation/results/final-audit.json) cover:
  - all 150 official campaign files;
  - hash-only inventories of G/H/I/J-A checkpoints, outputs, results and docs;
  - every canonical tracked file at the base commit;
  - the 4D.4J-A guard;
  - both checkpoint files.

  No historical file, checkpoint, sealed fixture, production agent, API, UI or deployment was modified. Only the new package and this report are committed; no model files.
- **Regression tests:** 113 passed, 2 self-play tests deselected, both before launch and at the final audit.
- **Resources:**
  - Task origin was 03:57:54 UTC.
  - Cost-only pilot plus calibration took 68 s, validation 19 s, measured tasks 293 s wall (4 × ~285 CPU-s; peak RSS ~300 MiB each), and analysis 11 s.
  - Execution finished 18.3 minutes into the task. The final audit ran at 24.3 minutes (1,456 s), well inside the 75-minute hard maximum; the commit and push followed within minutes.
  - No retry and no additional experiment.

## Integration recommendation

This is **Outcome A in qualified form**. The benefit is confirmed but small, and safety is strong:
- the improvement repeats on new openings, both seeds and both opponents;
- no paired game got worse;
- correctness and game-theoretic non-regression hold;
- latency is negligible.

The improvement is +1.3pp, below the +2pp practical margin and smaller against the stronger Negamax5.

**Recommendation:** proceed to a **separate, prospectively planned production-integration phase** for an **optional, clearly labelled tactical-safeguard toggle**. It must keep the original AlphaZero baseline available and unchanged. Present it as one-ply blunder protection with a small measured strength gain, **not** as a stronger agent. Whether it is on by default is for that phase to decide with product and UX review. Nothing was integrated, merged, deployed or promoted here, and no further experiment was launched.

**Answer to the final question:** the existing guard gives a **repeatable but small** improvement on newly generated development openings against two opponents (+1.3pp, 95% [+0.5, +2.3]). It is **not** practically meaningful by the pre-declared +2pp standard. Its main value is reliable, essentially free protection against one-move blunders.
