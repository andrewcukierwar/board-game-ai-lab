# Phase 4D.4J-B independent results review

**Disposition: PASS WITH NOTES.** I reproduced the author's verdict, CONFIRMED POSITIVE
EFFECT (SMALL, not practically meaningful), from the raw records. I found no
correctness, integrity or reproduction discrepancy. The notes concern how to read
the result, not errors.

This is a same-host verification by a separate agent. It is not an external
replication: no match was replayed by a model, no checkpoint was loaded, and the
records come from the same machine and run.

## Scope

Inputs: `DESIGN.md`, `results/{design,openings,freeze,analysis,preservation,execution}.json`,
the 4 task directories (8 match files, 4 `identity.json`, 6 regression-suite files)
and the 4D.4J-A suite files `research/alphazero_tactical_guard/results/tasks/{42,314159}/`.
No file with "sealed" in its name was opened, and no repo-wide search was run.

## Method

`independent_verify.py` was written from scratch. It does not import or read
`analyze.py`, `guard.py` or the 4D.4J-A reference board. Its pieces:
- Connect 4 rules: 7 column lists, with explicit 4-direction line scans through the last drop.
- Its own classifier for immediate wins, safe/unsafe moves and winning replies, plus the frozen guard rule re-derived from the DESIGN text.
- Its own paired-difference tensor (seed x opponent x family x color).
- Its own percentile family bootstrap: numpy PCG64 seed **99020001**, a different stream from the author's 4420004. It runs 10,000 reps and resamples whole families with all conditions kept together; strata are resampled within stratum.

Output: `results/independent-verification.json` (`all_checks_pass: true`, 0 issues).
Run time is about 1.4 s, torch-free. Mutation tests run in memory with output sent
to scratch. Each one was detected and flipped the verdict to FAIL:
- changing an override's final action
- changing a recorded score
- marking an unsafe move as safe
- altering a played move

## Checks and numbers

1. **Replay (1,536 games).** Every game was replayed from the empty board:
   - All moves are legal. Each game ends exactly at the first four or at a full board. Recorded winner (-1 = draw), result and score are all correct.
   - The prefix equals `openings.json` for the family, and the stratum label matches the prefix length.
   - 96 families (24 per stratum). 192 games per seed x opponent x arm. Each family x seed x opponent x arm has agent colors 0 and 1 exactly once.
   - 0 abandoned games. Opponent labels and document metadata match the task. The `openings.json` file hash matches design.
   - Every opening is non-terminal and has no immediate win for either side.
2. **Guard decisions.** All 8,933 guarded agent decisions were checked against my own classification:
   - Every guard field matches: legal, immediate wins, safe, unsafe, the winning-reply map, reason, final action, changed, selected visits, zero-visit flag, occupancy, no-safe flag, candidate count, and the opponent-replies-inspected count.
   - The final action equals both the decision's `final_action` and the move actually played.
   - `original_visits == visits` and visits sum to 256 everywhere. The original action always has maximal visits.
   - Each AVOID_IMMEDIATE_LOSS witness reply was replayed and does complete four.
   - All 8,649 control agent decisions have `guard: null` and play `move == original_action`.
   - **0 mismatches.** Reasons: IMMEDIATE_WIN 317, AVOID_IMMEDIATE_LOSS 76, NO_SAFE_MOVE 394, UNCHANGED 8,146.
   - 94 overrides were each verified: 76 avoid-loss and 18 immediate-win (missed wins the PUCT did not select).
3. **Paired fairness.**
   - In all 768 pairs, the first agent decision is identical across arms (original action and visit vector).
   - 690 pairs have identical move sequences. The 78 that diverge are *exactly* the 78 guarded games with a changed decision, so no divergence comes from anything else (such as RNG drift).
4. **Scores (guarded − control).** Every point estimate equals `analysis.json` exactly (|Δ| < 1e-12). The intervals come from my separate RNG stream:

   | slice | diff | my 95% CI | author CI |
   |---|---|---|---|
   | pooled (768 pairs) | **+1.302pp** | [+0.521, +2.344] | identical |
   | seed 42 / 314159 | +0.781 / +1.823 | [0, 1.823] / [0.521, 3.385] | identical |
   | negamax3 / negamax5 | +1.823 / +0.781 | [0.521, 3.516] / [0.130, **1.693**] | 5th: upper 1.563 |
   | 42-nm3 / 42-nm5 | +1.042 / +0.521 | [0, 2.604] / [0, 1.563] | identical |
   | 314159-nm3 / 314159-nm5 | +2.604 / +1.042 | [0.781, 4.948] / [0, 2.344] | identical |
   | X / O | +1.302 / +1.302 | [0.260, 2.865] each | identical |
   | ply04/06/08/10 | +0.521/+1.563/+1.042/+2.083 | lower 0 in each | identical |
   | B − A | −1.042 | [−2.604, +0.260] | identical |
   | seed 314159 − 42 | +1.042 | [−0.521, +2.865] | identical |

   Arm scores: control 0.43685, guarded 0.44987. The bootstrap is discrete, so one endpoint differing by 0.13pp (negamax5 upper) is Monte Carlo variation.
5. **Activation and latency.**
   - Every activation bucket in `analysis.json` matches exactly (all/seed/opponent/stratum/phase × calls/reasons/changed/zero-visit). Totals: changed 94, games 78, families 44, rate 1.052%, zero-visit selections 0. The per-game change records also match.
   - guard_seconds: mean 64.8 µs, median 72.5 µs, p95 97.1 µs, worst 171 µs. Mean guard/search ratio 0.229%. These match the author's figures and pass the limits (p95 ≤ 1 ms, worst ≤ 5 ms, ratio ≤ 5%).
6. **Regression suites.**
   - All 1,256 rows (2,512 searches) are identical to the 4D.4J-A rows on every non-timing field.
   - solved/deep: guarded chosen_value is never below control (0 rows). Guarded is above control in 4 solved rows per seed.
   - tactical200: guarded avoidable 0 and missed_win 0 for both seeds. Control had 10+3 (seed 42) and 13+1 (seed 314159).
7. **Hashes.**
   - guard.py = `d2ee54ba…2d04`. This matches freeze.json and design.json, and `git show ffb8009:` of the file hashes identically.
   - The checkpoint files at the freeze paths hash (as raw bytes) to `ba2bde2e…f58b` (42) and `2646d7cd…401d` (314159). These match identity.json and freeze.json.
   - All 82 freeze-listed source files are unchanged.
   - identity before == after (model/optimizer/RNG/replay/counters) in all 4 tasks, and the weights hash is constant per seed.
8. **No undeclared experiment.**
   - The task directories are exactly the 4 declared tasks, with 384 matches per task and 1,536 total.
   - execution.json: status COMPLETE_PENDING_RESULTS_REVIEW, retries 0, 4 tasks, finished before the compute deadline. preservation intact.
   - HEAD = prospective commit `ae4715ff…6141`. No tracked package or guard file differs from it, and the results are untracked (not yet committed, as expected).
9. **Verdict mechanics.** I applied DESIGN.md's hierarchy to my reproduced numbers:
   - No FAIL condition holds (correctness, tactical, pairing, WDL, reproduction, integrity, latency, material regression).
   - pooled > 0 with lower bound +0.52pp > 0, and each seed > −2pp. No opponent shows a material regression.
   - Result: CONFIRMED_POSITIVE_EFFECT. The pooled +1.30pp is below +2pp, so it is SMALL and not practically meaningful.
   - Seed and opponent estimates have the same sign, so the interpretation is not qualified.
   - All of this matches `analysis.json` field by field.

## Heterogeneity commentary

- **Concentration.** Only 12 of 768 pairs changed outcome, spread over 8 of 96 families. All 12 favour the guarded arm, and none favours control. Most of the effect sits in seed 314159 (9 of 12 pairs) and in Negamax3 (9 of 12).
- **Seed 42.** Its estimate is +0.78pp with lower bound exactly 0. Of its 3 nonzero pairs, 2 are vs Negamax3 and 1 vs Negamax5.
- **Negamax5.** The effect is +0.78pp and is exactly zero in strata ply04 and ply06. The B−A contrast (−1.04pp, CI spanning 0) is consistent with a smaller benefit against the stronger opponent, but does not establish one.
- **Strata.** Estimates range from +0.52 to +2.08pp. Every stratum lower bound is 0. No stratum or stratum × opponent cell is individually distinguishable from zero.
- **Color.** X and O are identical (+1.30pp each).
- **Changed games.** Across the 78 changed games, control went 17W/61L and guarded 25W/4D/49L: most changed games are still lost (an avoided immediate loss is often followed by a later loss). These associations are descriptive.

## Discrepancies

None. The only differences are bootstrap endpoint wobble from the separate RNG stream (largest 0.13pp, negamax5 upper bound). No point estimate, count, activation figure, latency statistic or verdict field differs.

## Unresolved findings retained (notes, not defects)

1. **The interval rests on a few families.** The positive lower bound (+0.52pp) depends on 8 nonzero families that are all non-negative. Exact-sign reasoning gives the same direction: 8/8 families positive, two-sided sign p ≈ 0.008. But a percentile bootstrap with this few nonzero clusters may be optimistic. "Confirmed" means confirmed *direction* under these fixed checkpoints, openings and opponents. It is not a precise magnitude.
2. **Scope is narrow.** There are two checkpoints, and two seeds do not estimate training-seed variance. Negamax depths 3 and 5 are the only opponents. There is no multiplicity correction.
3. **Openings are not fully independent.** 22 of 96 families pass through development-opening positions (declared in DESIGN). A coincidence with the sealed set cannot be excluded, by design.
4. **The 4D.4J-A verdict stands.** 4D.4J-A was INCONCLUSIVE against a +5pp target. That is not reinterpreted, and the two phases are not pooled.
5. **Latency was measured under 4-way parallel contention on one host.** No external-hardware timing was done.
6. **Rules agreement is limited to these games.** My rules and the bitboard agree on these 1,536 games and 8,933 positions. That is an agreement on this data, not a proof of the engine.
