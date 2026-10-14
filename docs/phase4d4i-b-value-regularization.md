# Phase 4D.4I-B — Value prediction regularization

**This experiment does not demonstrate improved held-out outcome prediction with preserved search guidance.**

Both seeds meet the prespecified novel-outcome MSE improvement rule versus final control. The combined decision remains inconclusive because seed42's match-score safeguard is unresolved: treatment scores8.125 percentage points below control, with95% interval [−19.375,+2.500]. This interval permits both a material strength loss and a small gain; it establishes neither preserved strength nor a confirmed loss.

The final milestone was selected prospectively, not from the measured curves. Both seeds are reported separately. The treatment adds `0.2 * mean(predicted_value ** 2)` to unchanged policy CE and realized-outcome value MSE. The added gradient reaches the shared trunk; this is full-network objective regularization, not a value-head-only change. AdamW settings, architecture, target definitions and training data are unchanged.

Prospective protocol: [DESIGN.md](../research/alphazero_value_regularization/DESIGN.md). Fixed machine-readable design/membership/source-runtime contracts: [design.json](../research/alphazero_value_regularization/results/design.json), [membership.json](../research/alphazero_value_regularization/results/membership.json), [freeze.json](../research/alphazero_value_regularization/results/freeze.json). Independent prelaunch review: [review](../research/alphazero_value_regularization/independent-prelaunch-review.md). Final [independent results review](../research/alphazero_value_regularization/independent-results-review.md) passed for execution and qualified interpretation, not treatment adoption.

Prospective pushed commit: `3bf0bf5ce70cab994618ed1f046168eb3bcaf1e4`. Base: `9b3e171b239ba6f25272f1cc5abc47907ad66249`. Milestones: [0, 1024, 2048] additional optimizer updates per arm/seed. No self-play games, additional arms, sweeps, generations, merge, deployment or promotion.

## Why increasing AdamW decay is a weak direct intervention

At LR3e-4, decay-only multiplication is `(1 − lr*wd)^U`. Changing decay1e-4→1e-3 yields only about2.7e-7 additional shrink per step. At2048 steps, the two cumulative shrinkages are0.0061438% and0.0614211%, an additional0.0552807% relative shrink. These figures isolate direct decoupled decay, not all possible optimizer/training feedback. Biases would not decay. The explicit penalty instead adds derivative `0.4*v/N` at each predicted output; useful values may also be suppressed, so magnitude reduction alone cannot establish success.

## Frozen data, novelty and uncertainty

| Seed | G5 train games/positions | G6 heldout games/positions | Exact overlap | Mirror overlap | Novel positions/games/families |
|---|---|---|---|---|---|
| 42 | 720/20049 | 144/4149 | 1099 | 1210 | 2939/138/99 |
| 314159 | 720/20505 | 144/4514 | 1545 | 1774 | 2740/132/75 |

Train only replay retained at the original generation5 checkpoint (generations1–5). Heldout is precisely each seed’s144 frozen generation6 new games, which never enter training. Independent history reconstruction found zero label errors. Exact and mirror-equivalent overlaps are audited against every retained training position. Primary membership excludes mirror overlap. Novel repeated heldout boards are retained with position weighting; they are not independent observations.

An additive provenance correction preserves the historical I-A report: its seen/novel subgroup labels were reversed. The saved I-A raw rows and this phase's initial rows have identical IDs, values, targets and overlap flags. Their correct initial novel MSEs are0.6840804 (seed42,2939 rows) and0.5442664 (seed314159,2740 rows); the seen MSEs are0.7428708 and0.8600786. This correction changes neither the frozen datasets nor the new measured results.

Intervals are paired10000-resample95% percentile bootstrap, RNG4411403. Entire opening-family clusters are resampled, preserving complete-game rows and paired colors; fixed development ancestry/reflection clusters are resampled for position suites. Means retain row weighting. Each seed’s own game distribution is evaluated separately. Intervals condition on two fixed networks and reused development families, exclude training-seed population uncertainty and have no multiplicity correction. The final novel-outcome MSE is the only primary endpoint.

## Value prediction across all declared milestones

| Seed | Updates | Arm | Train MSE | All heldout MSE | Novel MSE | Novel zero MSE | Novel−train gap | Novel bias | Mean abs value | Saturated >.95 | Calibration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 314159 | 0 | shared initial | 0.5803 | 0.6684 | 0.5443 | 0.6106 | -0.0360 | 0.0550 | 0.3756 | 0.51% | 0.1399 |
| 314159 | 1024 | control | 0.4217 | 0.5816 | 0.4841 | 0.6106 | 0.0624 | -0.0215 | 0.5123 | 3.61% | 0.2073 |
| 314159 | 1024 | treatment | 0.4277 | 0.5541 | 0.4592 | 0.6106 | 0.0315 | -0.0166 | 0.4253 | 0.99% | 0.1213 |
| 314159 | 2048 | control | 0.3644 | 0.5555 | 0.4642 | 0.6106 | 0.0998 | 0.0246 | 0.5650 | 10.00% | 0.2121 |
| 314159 | 2048 | treatment | 0.3667 | 0.5203 | 0.4230 | 0.6106 | 0.0563 | -0.0172 | 0.4677 | 2.34% | 0.1206 |
| 42 | 0 | shared initial | 0.5591 | 0.7012 | 0.6841 | 0.8054 | 0.1249 | 0.0245 | 0.3403 | 0.00% | 0.0829 |
| 42 | 1024 | control | 0.4290 | 0.7670 | 0.7749 | 0.8054 | 0.3459 | -0.0319 | 0.5097 | 2.28% | 0.2378 |
| 42 | 1024 | treatment | 0.4390 | 0.7341 | 0.7260 | 0.8054 | 0.2869 | -0.0318 | 0.4278 | 0.14% | 0.1511 |
| 42 | 2048 | control | 0.3540 | 0.7690 | 0.7824 | 0.8054 | 0.4285 | 0.0428 | 0.6025 | 9.05% | 0.2905 |
| 42 | 2048 | treatment | 0.3603 | 0.7321 | 0.7286 | 0.8054 | 0.3683 | 0.0284 | 0.5095 | 1.50% | 0.2084 |

Zero predicts0 everywhere; its MSE is the empirical mean squared target, not a favorable default due to class balancing. All datasets’ zero excess, means, fixed calibration bins, magnitude, saturation and per-target strata are in analysis.json. Realized imperfect-play outcomes and exact game-theoretic labels remain distinct. The gaps include policy/data distribution differences and cannot identify memorization alone.

| Seed | Updates | Novel treatment−control MSE [95%] | All-heldout Δ [95%] | Training Δ [95%] |
|---|---|---|---|---|
| 314159 | 1024 | -0.0248 [-0.0603, +0.0112] | -0.0275 [-0.0528, -0.0039] | +0.0060 [+0.0005, +0.0115] |
| 314159 | 2048 | -0.0412 [-0.0809, -0.0025] | -0.0351 [-0.0638, -0.0093] | +0.0023 [-0.0028, +0.0071] |
| 42 | 1024 | -0.0490 [-0.0871, -0.0144] | -0.0329 [-0.0641, -0.0061] | +0.0100 [+0.0052, +0.0148] |
| 42 | 2048 | -0.0538 [-0.0844, -0.0238] | -0.0369 [-0.0623, -0.0124] | +0.0063 [+0.0014, +0.0112] |

The prespecified practical margin is at least0.02 absolute novel MSE reduction in **each seed**, with each95% interval upper bound below0.

Seed314159: primary rule **PASS**; final novel MSE difference -0.0412 [-0.0809, -0.0025].
Seed42: primary rule **PASS**; final novel MSE difference -0.0538 [-0.0844, -0.0238].

Improvement over additional unregularized training does not establish improvement over the initial model. Final treatment minus initial novel MSE is+0.0445 [−0.0594,+0.1840] for42 and−0.1213 [−0.2403,−0.0088] for314159. Treatment's novel MSE is below zero's point baseline in both seeds, but its excess-zero interval includes0 for42. Confidence and calibration improve versus final control; that alone cannot establish useful search guidance.

## Search guidance, exact labels and playing strength

| Seed | Updates | Arm | Raw tactical accuracy | PUCT tactical accuracy | Solved300 exact value MSE | Solved PUCT accuracy | Deep128 mistakes | Negamax3 W/D/L | Score |
|---|---|---|---|---|---|---|---|---|---|
| 314159 | 0 | shared initial | 34.50% | 92.00% | 0.6963 | 74.67% | 37/128 | 36/5/39 | 48.12% |
| 314159 | 1024 | control | 36.00% | 92.00% | 0.7345 | 76.67% | 38/128 | 36/5/39 | 48.12% |
| 314159 | 1024 | treatment | 38.00% | 94.50% | 0.6837 | 77.33% | 35/128 | 35/7/38 | 48.12% |
| 314159 | 2048 | control | 39.50% | 92.50% | 0.7573 | 77.33% | 39/128 | 34/5/41 | 45.62% |
| 314159 | 2048 | treatment | 40.00% | 94.50% | 0.7049 | 76.33% | 37/128 | 36/7/37 | 49.38% |
| 42 | 0 | shared initial | 37.00% | 91.00% | 0.5926 | 74.67% | 40/128 | 36/7/37 | 49.38% |
| 42 | 1024 | control | 39.00% | 91.00% | 0.7090 | 78.00% | 30/128 | 30/7/43 | 41.88% |
| 42 | 1024 | treatment | 40.00% | 93.00% | 0.6555 | 79.00% | 28/128 | 32/9/39 | 45.62% |
| 42 | 2048 | control | 45.00% | 93.00% | 0.7962 | 78.67% | 23/128 | 35/10/35 | 50.00% |
| 42 | 2048 | treatment | 45.50% | 94.50% | 0.7384 | 80.33% | 23/128 | 29/9/42 | 41.88% |

Solved300 has100 each exact win/draw/loss, with zero MSE2/3; deep128 has70 wins,58 draws and no losses, with zero MSE70/128. Exact value strata and W→D/W→L/D→L downgrade counts are retained for each milestone in analysis.json. All actions on an exact lost board preserve WDL automatically, so solved accuracy on those rows is not evidence of useful tactics. Raw policy and PUCT are separate quantities. All milestones use unattenuated learned values, PUCT256 and frozen Negamax3, on all40 base families with both colors (80 games/task).

Final treatment's solved exact-value MSE remains above the zero baseline in both seeds, despite improving over final control. Seed42's exact-value MSE also increases from initial on solved and deep suites while its deep search mistake rate improves. Raw value error and resulting search action quality can move differently; neither target type can substitute for the other.

| Seed | Final comparator | Tactical category differences pp [95%] | Deep mistake difference pp [95%] | Match-score difference pp [95%] | Safeguards |
|---|---|---|---|---|---|
| 314159 | control | unique_safe: +4.0000 [+0.0000, +9.7826]; unique_win: +0.0000 [+0.0000, +0.0000] | -1.5625 [-5.6338, +2.7778] | +3.7500 [-6.8750, +14.3750] | deep: NO_DEMONSTRATED_REGRESSION; strength: SUPPORTIVE_UNDERPOWERED; tactical_puct_unique_safe: SUPPORTIVE; tactical_puct_unique_win: SUPPORTIVE |
| 314159 | initial | unique_safe: +7.0000 [+1.9608, +13.6364]; unique_win: -2.0000 [-7.2923, +2.2222] | +0.0000 [-5.2239, +5.5556] | +1.2500 [-10.6250, +13.1250] | deep: NO_DEMONSTRATED_REGRESSION; strength: SUPPORTIVE_UNDERPOWERED; tactical_puct_unique_safe: SUPPORTIVE; tactical_puct_unique_win: SUPPORTIVE |
| 42 | control | unique_safe: +1.0000 [+0.0000, +3.3333]; unique_win: +2.0000 [+0.0000, +5.0000] | +0.0000 [-5.5556, +4.6672] | -8.1250 [-19.3750, +2.5000] | deep: NO_DEMONSTRATED_REGRESSION; strength: UNRESOLVED_UNDERPOWERED; tactical_puct_unique_safe: SUPPORTIVE; tactical_puct_unique_win: SUPPORTIVE |
| 42 | initial | unique_safe: +6.0000 [-1.9608, +14.8936]; unique_win: +1.0000 [-3.3333, +5.3191] | -13.2812 [-25.0000, -2.3810] | -7.5000 [-21.2500, +5.6250] | deep: NO_DEMONSTRATED_REGRESSION; strength: UNRESOLVED_UNDERPOWERED; tactical_puct_unique_safe: SUPPORTIVE; tactical_puct_unique_win: SUPPORTIVE |

Tactical pass margins are point>=−3pp and lower>=−10pp; regression requires point<=−5pp and upper<0. Deep regression requires mistake increase>=5pp and lower>0. Supporting strength pass requires point>=−5pp and lower>=−15pp; regression requires point<=−5pp and upper<0. These compare treatment with both final control and initial state in both seeds. Strength evidence from40 pairs is underpowered for small differences; supportive intervals cannot establish noninferiority or prove general playing-strength preservation. Favorable MSE cannot override search regression.

## Correctness, independent review and preservation

Focused correctness tests: [validation.json](../research/alphazero_value_regularization/results/validation.json). Both seeds’ control loss and entire optimizer update matched the historical trainer exactly. Analytic penalty/gradient, unchanged policy CE, finite gradients and nonzero shared-trunk penalty gradients passed. Both arms started with identical model/optimizer/full state, saw identical sampled addresses and reflections, and matched fresh-process continuation exactly. Strict rejection checks cover altered identities and overwrite paths. Every evaluation checked complete learner/RNG preservation. New checkpoints use an explicitly versioned fixed-replay contract; original H-C checkpoints were strictly loaded read-only.

Independent arithmetic, cluster intervals, membership/action/result identities and all task completion markers: [independent-verification.json](../research/alphazero_value_regularization/results/independent-verification.json). This same-host saved-evidence audit makes no new model query and is not an external replication or fresh exact-game solver proof. The final independent reviewer additionally checked 40 saved means, all 63 frozen source identities and four original checkpoint hashes. Both scientific reviews passed; the original finalization failure and its saved-evidence recovery are retained separately.

All150 official campaign files and required historical evidence inventories passed [preservation.json](../research/alphazero_value_regularization/results/preservation.json). The Phase4D.4H architecture-only phase has no experiment-output tree; its actual source/evidence/report paths are inventoried. No historical files or frozen training sources were changed.

The [prior sealed-access incident](../research/alphazero_value_diagnostics/integrity-resolution.md) remains disclosed historical research provenance. No sealed fixture content or raw exposed response was opened/copied for this phase. Hash-only preservation checks do not inspect labels. This study measures novel positions within known development/self-play distributions, not sealed independent generalization.

## Resource accounting and recommendation

Cost-only pilot: 14.55s. Successful focused validation: 105.23s. Complete measured computation, independent arithmetic and report generation took 1549.9982s; with the disclosed conservative 240s prelaunch charge, this is 1789.9982s (29.83 minutes), below 2700s. All 8192 updates, 6280 fixture roots and 800 matches completed without measured retries or new self-play. The prelaunch charge includes estimated allocations for two disposable failed checks whose exact timers were not retained; those allocations are not exact measurements.

The original automated finalization failed immediately at CLI exit 2 because `--approve-for-me` and `--sandbox` cannot be combined. It did not run its final AI reviewer or publish. The frozen launcher remains unchanged, and [failure evidence](../research/alphazero_value_regularization/results/finalization-failure.json) is preserved. After the user requested status, the independent reviewer completed a saved-evidence review with zero training, model queries or bootstrap reruns. This editorial recovery uses the original absolute deadline; it does not retry or extend the experiment. [Final publication accounting](../research/alphazero_value_regularization/results/final-publication-audit.json) charges the original-start elapsed time, including the interruption gap, review and audits. Publication completion is separately recorded under the ignored local output root. Detailed per-task CPU/RSS and immutable original resource records are retained.

Prespecified combined decision: **NO_CONFIRMED_PRESERVED_GUIDANCE_IMPROVEMENT**. Primary pass: True; demonstrated safeguard regression: False; unresolved safeguards: True.

Retain the CNN baseline. Recommend one next value-learning hypothesis: weight only the realized-outcome value loss inversely by mirror-equivalent training-position frequency, giving each board equal total value-loss weight while leaving policy CE unchanged. Repeated positions with noisy outcomes may otherwise dominate fixed replay. This is a distinct prospective hypothesis, not a conclusion about its effectiveness; no follow-up is launched, and this experiment is not expanded.

Full raw rows/visits/games, per-seed calibration/target strata and independent intervals remain in the new research package. No best milestone is selected retrospectively.
