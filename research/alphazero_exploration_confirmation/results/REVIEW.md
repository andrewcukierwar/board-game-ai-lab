# Independent review — Phase 4D.4E

A separate reviewer agent examined the prospective design, unchanged collector,
new exact-analysis implementation, raw records, proof outputs and phase report.
It authored no experiment implementation or statistical thresholds. It wrote
only independent validation evidence and this review. This is an independent
engineering review on the same host, not replication by another laboratory.

**Verdict: no material implementation or interpretation finding remains
unresolved. The prespecified safety result is INCONCLUSIVE. Diversity replicates.
The literal full-record prefix requirement has a disclosed diagnostic deviation;
it is not silently relabeled as passed.**

## Design and implementation

The reviewer read the previous phase's prospective design, exact post hoc tools,
raw-evidence contract and independent review, plus the 4A–4C reports and relevant
collector, solver and generation-boundary code. It reviewed the new design before
its freeze and the analyzer before opening new game outcomes.

- Frozen design commit `89c8822` precedes fresh main collection. Its SHA-256 is
  `e90aecd6f0deed6cec2c2034d2be12c9a1a3dd51f18c8bdc1472f05e37b32713`, unchanged
  in collection/proof provenance. The local origin tracking branch also pointed
  to that freeze commit. Analyzer commit `c549021` preceded main collection.
- Fresh primary identities use generation identity 22 and new cohort seeds.
  Two checkpoints, two cohorts and two schedules yield 512 complete games and
  256 matched statistical units. Regression identities are separately retained
  controls and excluded from primary analysis.
- No source under `games/` changed from starting commit `08656fa`. PUCT,
  root noise, network architecture, training runner and production defaults are
  unchanged. The optional collector retains independently owned game RNGs.
- The +0.05 margin is a permissive research screen, not universal safety. The
  reviewer checked that the fixed rule uses conservative game-level missingness
  bounds, both checkpoint upper bounds and stage-specific coverage for a
  no-material-harm conclusion. No threshold changed after results.
- Native C source SHA-256
  `b60dc839edebdcc92cee1a210b795956125bf10a5989dfa346d4e18e9525c942` equals the
  historical project source at `8704142:tests/victor_validation/c4_oracle.c`.
  It remains an analysis oracle and never participates in AlphaZero search.
- The proof collector respects the two declared node passes, per-query wall
  limits and total launch ceiling. Completed proofs survive node aborts; an
  aborted root exposes no partial vector. Cache keys preserve player perspective
  and reflection equivalence. Successful vectors establish exact successor
  values, with terminal wins and draws checked separately.
- Exact deductions are sound: a legal winning action proves state value +1;
  every action from an exactly lost state has value -1. Unknown values remain
  explicit. Adjacent contradictory values are rejected before applying these
  deductions.

## Review findings and fixes

1. **MINOR — partial collection resilience.** The initial analyzer aborted on
   any incomplete cell, instead of retaining complete matched checkpoint/cohort
   blocks. Initial fixes also attempted hashes for excluded arms and statistics
   on empty cohorts. These paths now exclude whole unmatched blocks, preserve
   explicit exclusion reasons, skip their hash checks, mark absent cohorts, and
   write INCONCLUSIVE for insufficient matching. Actual primary collection has
   no incomplete cell or exclusion. Tests cover the filtering and decision rule.
2. **MINOR — frozen driver limits.** Initial command arguments could exceed
   the prospective ceiling/sample size. The phase drivers now enforce the
   frozen bounds. Actual main invocation already used exactly 64 games per cell
   and 650 seconds. This enforcement did not alter its collection behavior.
3. **MINOR — exact consistency.** A scalar neutral deduction could overwrite
   an inconsistent known adjacent value. The analyzer now rejects any known
   action value exceeding the known optimal state value before deductions.
   Independent reconstruction found no actual contradiction.
4. **MINOR — literal prospective prefix equality.** The first summary attempt
   stopped on three differing root-value diagnostics, despite identical
   discrete decisions. Independent raw inspection found exactly three float32
   value differences, at most `5.960464477539063e-08`, all for index 63. Every
   other stored pre-ply-8 field is identical over 256 matched games / 2,040
   decisions: actions, visits, states, temperatures, priors, distributions,
   depths and terminal-leaf counts. Batch-tail rounding is a supported
   explanation; it is not a separately proved hardware cause. Differences are
   below the inherited 4C value tolerance of `2e-6`.

   The user's identical-decision criterion passes. The 4E design's stricter
   literal full-record gate remains false in the evidence and prevents a
   no-material-harm label. The failed summary and original exact proofs were
   retained. Reanalysis changes no primary endpoint, threshold, identity or
   observation and executes no new solving. The primary result is independently
   INCONCLUSIVE from inadequate coverage and uncertainty, so this deviation
   cannot improve its conclusion.
5. **NOTE — interpretation and presentation.** Confirmed-count differences
   must not be called full-population error differences; selected non-max actions
   should be reported separately from all extended decisions. The report makes
   both distinctions and describes the training recommendation as a guarded
   learning experiment rather than a finding of safety.

## Independent evidence checks

The evidence-only `validation/independent-recompute.py` imports no experiment
analyzer. Its execution took 8.999 seconds and performed no new collection or
solving. It independently:

- Replayed all 512 games through the engine, checked every move legal, complete
  termination and winner labels, actor-relative outcomes, 256 root visits,
  schedule application and mathematical temperature distributions.
- Rebuilt every cached scalar proof from native exact vectors, immediate legal
  wins and completed successor proofs; checked each native vector's legal
  columns, state/action maximum and terminal perspective.
- Recomputed all 11,884 decision classifications, sampling/search attribution,
  explicit max-visit ties and same-search immediate regret from the reconstructed
  states and recorded proofs.
- Recomputed checkpoint and pooled paired means, standard errors,
  identification bounds, missingness-expanded uncertainty and the safety rule.
- Recomputed headline duplicate fractions, mirror duplicates, collision
  probabilities and actual outcome distributions directly from raw records.

There are **zero numerical discrepancies** in those comparisons.

The reviewer also cross-checked every legal native action value at eight prior
late positions against both existing Python BitboardSolver and independent
exhaustive column-stack minimax; eight reflected vectors matched. Exact execution
was 0.00956 seconds, with zero discrepancies. On twelve newly confirmed errors
that the one-reply oracle regarded as safe, spanning both checkpoints, schedules
and stages, it made 24 Python reference state queries at 20,000 nodes each.
Eleven resolved exactly and matched; thirteen remained explicitly unknown.
There were no discrepancies, and solver execution was 1.537 seconds.

An independent focused run passed 29 tests before review fixes. The root's final
relevant inference/analysis run reports 245 passed, including 38 new focused
contracts, without optimizer steps. The reviewer checked the added focused tests
and their relation to the fixes; it did not independently rerun all 245 tests.

Independent preservation rehashed all 150 official files against the retained
inventory: byte-identical, including all eight inference checkpoints, with the
required declaration hash. The historical checkout remains clean at
`1894bc4d8f035cf738df8e3fc0ce234f1ce2c8be`. Production/game sources and previous
research results are unchanged. No merge, deployment or service work occurred.

## Findings and limits of interpretation

Original versus extended confirmed errors are 390 versus 358 over 256 games
per arm: **at least 1.523 versus at least 1.398 per game**. Type counts are
141/130/119 versus 115/118/125 for win→draw / win→loss / draw→loss.
The confirmed-count contrast is -0.125, approximate 95% interval
[-0.417, +0.167]. This contrast is coverage-dependent and is neither a full
population estimate nor a lower bound on the true effect.

10,619 of 11,884 eligible actions are adjudicated exactly (89.36%); 1,265 remain
unknown. Almost all unknowns occur at plies 8–15, whose checkpoint/schedule
coverage is only 54.5–77.4%. The pooled full-population identification interval
is [-2.773, +2.168], expanding for paired RNG uncertainty to
**[-3.160, +2.548]**. These results cannot screen out the +0.05 material margin.
The observed confirmed-count half-width, 0.292, is itself substantially larger
than that margin. The previous rare flagged-event endpoint did not forecast the
much larger burden and variability of all-decision deep errors.

Diversity is independently replicated: duplicates after ply 8 fall from
33.97% to 10.75% for checkpoint 42 and from 37.37% to 12.51% for checkpoint
314159. Mirror duplicates and collision probabilities also fall. This is about
collected data, not improved neural learning.

Of 508 non-max selections, 69 are exactly worse than all maximum-visit
alternatives, 23 better, 286 equal and 130 unresolved. Fifty-seven extended
confirmed errors are strictly sampling-attributable, 299 search-attributable,
one mixed-tie and one with unresolved attribution. Immediate sampling cost is
real, but it is not the paired full-trajectory effect or a learning result.

The INCONCLUSIVE conclusion is applied correctly. Absence of a positive
confirmed-count contrast is not equivalence. The broader error counts are
plausible and independently checked; they must not be compared as though they
shared the prior post hoc one-reply-flagged denominator.

A small original-versus-extended training ablation is defensible as a bounded
research decision because diversity replicates and learning is the remaining
unmeasured benefit. It carries acknowledged risk from direct sampling
regressions and unresolved early positions. Keep the original default; require
separate tactical and strength guards. First implement an explicitly experimental
batched generation runner with per-generation schedule provenance, mixed-schedule
replay validation and faithful boundary resume. Warm-start paired arms with
identical model, optimizer and replay states and match computation/update budgets.
No such engineering or training was performed in this phase. Another automatic
frozen-checkpoint confirmation is not recommended.

The reviewer did not independently recollect the 512 games, prove every native
output with a different solver, resolve unknown positions, or test neural learning.
Stored proofs make analysis exactly reproducible; rerunning bounded solvers can
have timing-dependent coverage. All claims are limited to these two frozen
checkpoints and the declared RNG sample.

## Independent artifacts and reproduction

`validation/` contains:

- `independent-native-validation.json`: historical-source equality and late
  native/Python/exhaustive/reflection validation.
- `independent-new-error-solver-validation.json`: bounded new-error reference
  checks, including every reference unknown.
- `independent-prefix-validation.json`: all matched prefixes and three retained
  root-value diagnostic differences.
- `independent-preservation.json`: independent campaign and checkout verification.
- `independent-recompute.py`, `independent-recompute.json` and
  `independent-final-recompute.json`: independent arithmetic, classified-proof
  reconstruction and exact headline comparison before and after final packaging.

From the worktree, with the existing M4 environment and one-thread variables set:

```sh
PYTHONPATH=. /Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python \
  research/alphazero_exploration_confirmation/results/validation/independent-recompute.py \
  --analysis research/alphazero_exploration_confirmation/results/analysis-final \
  --output /tmp/phase4d4e-independent-review.json
```

The script requires a fresh output filename and refuses to overwrite evidence.
The example uses a new `/tmp` JSON file; choose another fresh filename if it exists. The primary analyzer
also requires fresh output directories, as documented in the tools README.

## Final delivered-source pass

The reviewer reran the independent script on `analysis-final` and the compressed
committed game records, using a fresh output file. It reproduced all 512 games,
11,884 action classifications, 2,040 matched prefix decisions, the three retained
root-value diagnostic differences, every scalar proof, headline diversity,
outcomes, sampling classifications and paired primary uncertainty. Zero numeric
discrepancies; 9.061 seconds; no new solving or self-play. Eight matched identities
finish after seven plies, explaining why the prefix count is 2,040 rather than
2,048.

All common fields of initial `analysis/results.json` and final
`analysis-final/results.json` are exactly identical. The final file adds only the
prospectively defined pooled secondary statistics. The phase report was reviewed
after spacing, move-count and conditional-precision wording corrections.

**Final disposition: no material finding remains unresolved; INCONCLUSIVE safety
and replicated diversity are correctly reported. The literal diagnostic equality
deviation and limited cross-solver validation remain explicit limitations.**
