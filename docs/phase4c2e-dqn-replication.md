# Phase 4C.2e — Final controlled DQN replication

**Exactly four fresh runs completed, each at 90,000 successful optimizer updates.** Horizontal augmentation produced repeatable aggregate tactical and mirrored-value improvements in the two new seed pairs, but gains varied substantially and did not resolve weak play. All 400 scheduled evaluation games and all required integrity checks passed.

1. **Repeatable tactical gains: yes, within this bounded replication.** Accuracy increased on all three suites for both seeds. On the new final holdout, seed 73 improved **25/96 → 28/96**, while seed 314 improved **15/96 → 37/96**. These are gains of 3 versus 22 rows; repeatable direction does not imply a stable effect size or improvement in every category.
2. **Repeatable mirrored-value improvements: yes.** Aligned legal-Q MAE decreased in all six seed/suite comparisons. On the final holdout it fell **43.79%** for seed 73 and **40.74%** for seed 314. Mirror action agreement remains poor and is not uniformly improved.
3. **Actual Negamax strength: limited repeatability at depth 1, not established at depth 2.** Depth-1 wins increased **4/40 → 7/40** and **7/40 → 12/40**. Depth-2 wins changed **2/40 → 2/40** and **2/40 → 4/40**. Every model lost all 20 O-side depth-2 games. This supports a small shallow-opponent benefit on the frozen schedule, not broadly reliable playing strength.
4. **Significant learning problems remain: yes.** Augmented final accuracy is only **29.17% and 38.54%**, most immediate wins/blocks are missed, category reversals and large prediction changes persist, and winning-action Q-values are often negative. Numerical stability does not repair these failures.

**Recommendation: conclude the current Phase 4C DQN training investigation and proceed to a separately scoped Phase 4D.1 correctness milestone. Keep all DQN checkpoints as offline research artifacts; do not promote one into public gameplay or extend this training campaign.** No significance claim or checkpoint selection is made from two new seed pairs.

## Predeclaration, scope and verification

Fetched `origin/main`; it matched the expected `022efa80cac3e77991f410b711d7448c4b75d0ba`. The initial tree was clean. Created local branch `phase4c2e-dqn-replication` from that commit and read the completed Phase 4C.2c and Phase 4C.2d handoffs. No historical checkpoint initialized a new run. Exactly four fresh runs were authorized and executed, in order: 73 baseline, 73 augmented, 314 baseline, 314 augmented. No repeats, extra training seeds, sweeps, learning-method changes, tactical guards, inference-time augmentation, production dependency changes, API/UI integration, hosting changes, MCTS-NN work, paid requests, commits, pushes, merges or deployments occurred.

The existing `DQNTrainer`, `run_training`, experiment CLI, safe checkpoint contract, tactical summaries and Negamax/Random `play` implementation were reused. The only shared-code change parameterizes the existing model-blind holdout constructor with a construction seed and excluded fixture paths; its default behavior is preserved. The new replication module freezes the protocol and orchestrates evaluation. No additional training framework was created.

All required gates passed **before the first run**. The optional DQN environment and backend environment were used separately, preserving the API isolation check exactly.

| Gate | Environment / command | Result |
| --- | --- | --- |
| Complete DQN/validation tests | `/tmp/board-game-phase4c1-venv/bin/python -m pytest -q tests/test_connect4_dqn.py tests/test_dqn_experiment.py tests/test_dqn_diagnostics.py tests/test_dqn_symmetry.py tests/test_dqn_validation.py tests/test_dqn_replication.py` | 466 passed in 5.15 s |
| Backend regression | `.venv/bin/python -m pytest -q tests` | 353 passed, 6 optional DQN modules skipped in 6.06 s |
| Explicit API import isolation | `.venv/bin/python -m pytest -q tests/test_dqn_import_isolation.py` | 1 passed in 0.16 s |
| Whitespace gate | `git diff --check` | Passed |

The 98 new checks independently validate every final-holdout row against engine successors and contiguous-piece directions, verify literal and reflected disjointness and category/column coverage, and check paired CLI settings. Existing synthetic tests cover Bellman signs, terminal exclusion, legal masks, augmentation correctness, independent RNG ownership, successful-update epsilon/target schedules, stop boundaries and checkpoint loading. No test starts an experimental training run or scores a candidate.

## Frozen final holdout and evaluation protocol

The [final fixture](../games/connect4/dqn/replication_positions.json) was frozen at `2026-10-04T14:13:38.463774+00:00` (October 4, 2026, 10:13:38 EDT), before training started at 10:14:51 EDT. SHA-256: `8e63fd8be0419ddb4daf83ae226362f26999dced321617900938d61a60efb7da`. Configuration SHA-256: `e9ec041241064d0276705dd16663eaac6741979d983b1c528f4c587f807c1dfe`. Independent engine-label validation passed before training/inference. Labels and bytes were unchanged after evaluation.

The original 60-position suite and the Phase 4C.2d 96-row holdout are **previously inspected validation sets**. Only the additional final holdout was model-blind for this milestone. Construction reused the prior quota-based legal-rollout method with data-construction seed `42020501` (not a training seed), excluding both earlier suites under board-plus-player reflection canonicalization. Each of its 48 distinct original families comes from a separately sampled rollout and has one reflected row. There are 96 distinct literal boards, 48 wins/48 uniquely safe immediate blocks, 48 X/48 O rows, 32 rows per direction, eight rows per tactic × direction × player cell, and correct-column counts `[14,14,14,12,14,14,14]`. Neither labels nor construction were changed after predictions. Disjointness from every self-play training board is not established.

A win label requires exactly one engine-winning action. A block label requires no own immediate win, exactly one opposing immediate threat, and exactly one move that prevents an immediate winning reply. Direction is independently checked through the landing cell. These are one-reply tasks, not solved-game values.

All checkpoints receive raw greedy, legal-masked CPU/float32 inference with no learning or guards. The exact Phase 4C.2d opening schedule is frozen in `configuration.json`: 20 prefixes (ten original/mirror families), each with DQN assigned to X and O, for **40 games per model per Negamax depth**. Existing depth-1 and depth-2 agents retain their move ordering, heuristic, ties and within-game caches; each game creates a fresh opponent. Random uses the first ten prefixes × both sides for **20 games per model**, with seeds `420206 + 2 × opening_index + side`. Thus schedules match within each pair and across both training seeds. No head-to-head games were added. Bounds: 600 seconds per seed-pair evaluation, including tactical inference/loading; Random has a separate 30-second cap per pair. Deadlines are cooperative at move boundaries; partial outcomes are recorded separately.

## Training configuration and realized exposure

| Setting | All four runs |
| --- | --- |
| Architecture | 42 → 128 → 128 → 7 ReLU MLP; raw Q outputs |
| Optimization | Adam 0.001; selected-action Smooth L1; gamma 0.99; signed adversarial legal Bellman target |
| Replay / batch | 10,000 / 64; uniform sampling |
| Target sync | Every 100 successful optimizer updates |
| Epsilon | 1.0 → floor 0.10; multiply by 0.99997 per successful update |
| First reached limit | 90,000 successful updates; 100,000 plies; 7,000 completed games; 300 training seconds |
| Execution | CPU float32; one intra-op / one inter-op thread; deterministic algorithms enabled |
| Only method difference | Horizontal symmetry probability 0.0 versus 0.5 |

Separate existing `random.Random` objects serve exploration and replay; augmentation uses the domain-separated `connect4-horizontal-symmetry-v1:<seed>` stream. Matched seeds and configurations use the same deterministic initialization; complete initial diagnostic predictions were independently verified to match exactly within each seed pair. Augmentation consumes no exploration/replay RNG draws, and the disabled path consumes no augmentation draws. Policies can still induce different self-play trajectories and replay contents.

| Seed | Method | Completed games | Started games | Plies | Updates | Seconds | Stop | Partial-game plies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 73 | baseline | 4913 | 4913 | 90063 | 90000 | 113.464 | max_updates | 0 |
| 73 | augmented | 4787 | 4788 | 90063 | 90000 | 202.850 | max_updates | 17 |
| 314 | baseline | 4807 | 4808 | 90063 | 90000 | 112.825 | max_updates | 19 |
| 314 | augmented | 4806 | 4807 | 90063 | 90000 | 203.128 | max_updates | 19 |

**Exposure was exactly equal in successful updates and collected plies across all four runs.** Each stopped at `max_updates` with 90,000 updates and 90,063 plies; none hit the 300-second, 100,000-ply or 7,000-game cap first. Different completed games and replay distributions are policy consequences. The augmented runs took about 203 seconds versus about 113 seconds for baseline, so this is an equal-update comparison, not equal wall-time efficiency.

Each run has 90,000 finite recorded losses, 5,760,000 sampled optimization items, 900 target synchronizations through update 90,000, and final epsilon 0.10. Collection, optimization and per-ply event writing are included in training time; initialization, evaluation and checkpoint serialization are outside that clock. The first 63 plies warm replay. Partial games remain in replay/event accounting and are excluded from completed-game outcome counts. No run was restarted or extended.

## Paired tactical accuracy and margins

B = baseline; A = augmented. Δ is A minus B in percentage points. The original suite has 48 tactical rows plus 12 unlabeled probes; both later suites have 96 tactical rows. Margins are `Q(expected) − max Q(other legal action)` and depend on Q scale. All position-level Q-vectors, choices, changes and margins are preserved in `evaluation.json`.

### Seed 73

**Original validation suite**

| Group | B accuracy | A accuracy | Δ pp | B mean margin | A mean margin |
| --- | --- | --- | --- | --- | --- |
| Overall | 16/48 (33.33%) | 20/48 (41.67%) | +8.33 | -0.220390 | -0.050052 |
| Win | 7/24 (29.17%) | 10/24 (41.67%) | +12.50 | -0.254508 | -0.080899 |
| Block | 9/24 (37.50%) | 10/24 (41.67%) | +4.17 | -0.186271 | -0.019205 |
| Horizontal | 1/16 (6.25%) | 3/16 (18.75%) | +12.50 | -0.450272 | -0.156736 |
| Vertical | 9/16 (56.25%) | 11/16 (68.75%) | +12.50 | 0.046927 | 0.138037 |
| Diagonal | 6/16 (37.50%) | 6/16 (37.50%) | +0.00 | -0.257824 | -0.131457 |
| X | 6/24 (25.00%) | 7/24 (29.17%) | +4.17 | -0.302039 | -0.091178 |
| O | 10/24 (41.67%) | 13/24 (54.17%) | +12.50 | -0.138741 | -0.008927 |

**Phase 4C.2d validation holdout**

| Group | B accuracy | A accuracy | Δ pp | B mean margin | A mean margin |
| --- | --- | --- | --- | --- | --- |
| Overall | 21/96 (21.88%) | 27/96 (28.12%) | +6.25 | -0.370421 | -0.159287 |
| Win | 9/48 (18.75%) | 13/48 (27.08%) | +8.33 | -0.416748 | -0.206094 |
| Block | 12/48 (25.00%) | 14/48 (29.17%) | +4.17 | -0.324095 | -0.112481 |
| Horizontal | 5/32 (15.62%) | 5/32 (15.62%) | +0.00 | -0.400317 | -0.168900 |
| Vertical | 11/32 (34.38%) | 15/32 (46.88%) | +12.50 | -0.261194 | -0.006260 |
| Diagonal | 5/32 (15.62%) | 7/32 (21.88%) | +6.25 | -0.449752 | -0.302702 |
| X | 14/48 (29.17%) | 11/48 (22.92%) | -6.25 | -0.320920 | -0.176330 |
| O | 7/48 (14.58%) | 16/48 (33.33%) | +18.75 | -0.419922 | -0.142245 |

**Final holdout**

| Group | B accuracy | A accuracy | Δ pp | B mean margin | A mean margin |
| --- | --- | --- | --- | --- | --- |
| Overall | 25/96 (26.04%) | 28/96 (29.17%) | +3.12 | -0.316132 | -0.133817 |
| Win | 10/48 (20.83%) | 13/48 (27.08%) | +6.25 | -0.377248 | -0.160682 |
| Block | 15/48 (31.25%) | 15/48 (31.25%) | +0.00 | -0.255015 | -0.106953 |
| Horizontal | 6/32 (18.75%) | 7/32 (21.88%) | +3.12 | -0.191727 | -0.207649 |
| Vertical | 15/32 (46.88%) | 12/32 (37.50%) | -9.38 | -0.200217 | -0.054070 |
| Diagonal | 4/32 (12.50%) | 9/32 (28.12%) | +15.62 | -0.556450 | -0.139733 |
| X | 11/48 (22.92%) | 12/48 (25.00%) | +2.08 | -0.320447 | -0.166246 |
| O | 14/48 (29.17%) | 16/48 (33.33%) | +4.17 | -0.311816 | -0.101389 |

### Seed 314

**Original validation suite**

| Group | B accuracy | A accuracy | Δ pp | B mean margin | A mean margin |
| --- | --- | --- | --- | --- | --- |
| Overall | 9/48 (18.75%) | 23/48 (47.92%) | +29.17 | -0.287743 | 0.008932 |
| Win | 3/24 (12.50%) | 12/24 (50.00%) | +37.50 | -0.413879 | 0.074725 |
| Block | 6/24 (25.00%) | 11/24 (45.83%) | +20.83 | -0.161606 | -0.056861 |
| Horizontal | 2/16 (12.50%) | 4/16 (25.00%) | +12.50 | -0.231250 | -0.247684 |
| Vertical | 6/16 (37.50%) | 12/16 (75.00%) | +37.50 | -0.151675 | 0.323062 |
| Diagonal | 1/16 (6.25%) | 7/16 (43.75%) | +37.50 | -0.480303 | -0.048581 |
| X | 5/24 (20.83%) | 10/24 (41.67%) | +20.83 | -0.261881 | -0.095190 |
| O | 4/24 (16.67%) | 13/24 (54.17%) | +37.50 | -0.313604 | 0.113054 |

**Phase 4C.2d validation holdout**

| Group | B accuracy | A accuracy | Δ pp | B mean margin | A mean margin |
| --- | --- | --- | --- | --- | --- |
| Overall | 20/96 (20.83%) | 32/96 (33.33%) | +12.50 | -0.312050 | -0.215007 |
| Win | 9/48 (18.75%) | 16/48 (33.33%) | +14.58 | -0.407580 | -0.281641 |
| Block | 11/48 (22.92%) | 16/48 (33.33%) | +10.42 | -0.216521 | -0.148373 |
| Horizontal | 8/32 (25.00%) | 9/32 (28.12%) | +3.12 | -0.262266 | -0.230012 |
| Vertical | 8/32 (25.00%) | 16/32 (50.00%) | +25.00 | -0.214632 | -0.100209 |
| Diagonal | 4/32 (12.50%) | 7/32 (21.88%) | +9.38 | -0.459253 | -0.314799 |
| X | 9/48 (18.75%) | 17/48 (35.42%) | +16.67 | -0.293826 | -0.212623 |
| O | 11/48 (22.92%) | 15/48 (31.25%) | +8.33 | -0.330275 | -0.217391 |

**Final holdout**

| Group | B accuracy | A accuracy | Δ pp | B mean margin | A mean margin |
| --- | --- | --- | --- | --- | --- |
| Overall | 15/96 (15.62%) | 37/96 (38.54%) | +22.92 | -0.328807 | -0.096536 |
| Win | 5/48 (10.42%) | 18/48 (37.50%) | +27.08 | -0.389541 | -0.119117 |
| Block | 10/48 (20.83%) | 19/48 (39.58%) | +18.75 | -0.268073 | -0.073955 |
| Horizontal | 4/32 (12.50%) | 7/32 (21.88%) | +9.38 | -0.385465 | -0.210750 |
| Vertical | 6/32 (18.75%) | 21/32 (65.62%) | +46.88 | -0.233383 | 0.117929 |
| Diagonal | 5/32 (15.62%) | 9/32 (28.12%) | +12.50 | -0.367572 | -0.196786 |
| X | 6/48 (12.50%) | 17/48 (35.42%) | +22.92 | -0.355289 | -0.124136 |
| O | 9/48 (18.75%) | 20/48 (41.67%) | +22.92 | -0.302324 | -0.068936 |

## Symmetry, greedy concentration and correlated families

| Seed | Suite | Method | Mirror actions | Aligned Q MAE | Max Q difference | Columns 0–6 | Entropy bits | Dominant share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 73 | existing | baseline | 9/30 | 0.778712 | 3.276357 | [3, 11, 11, 9, 15, 4, 7] | 2.6461 | 25.00% |
| 73 | existing | augmented | 8/30 | 0.429776 | 2.345622 | [7, 6, 8, 12, 11, 11, 5] | 2.7419 | 20.00% |
| 73 | validation | baseline | 8/48 | 0.890258 | 4.064293 | [15, 18, 15, 11, 15, 10, 12] | 2.7812 | 18.75% |
| 73 | validation | augmented | 16/48 | 0.557133 | 2.160671 | [14, 7, 15, 19, 16, 12, 13] | 2.7579 | 19.79% |
| 73 | final | baseline | 7/48 | 0.968802 | 5.076219 | [12, 18, 18, 23, 7, 6, 12] | 2.6750 | 23.96% |
| 73 | final | augmented | 13/48 | 0.544576 | 2.671064 | [8, 18, 19, 6, 21, 13, 11] | 2.6925 | 21.88% |
| 314 | existing | baseline | 5/30 | 0.855214 | 3.708934 | [4, 10, 10, 15, 7, 1, 13] | 2.5602 | 25.00% |
| 314 | existing | augmented | 15/30 | 0.593124 | 2.345896 | [6, 4, 11, 6, 20, 4, 9] | 2.5729 | 33.33% |
| 314 | validation | baseline | 6/48 | 0.953313 | 4.438666 | [16, 10, 17, 9, 9, 8, 27] | 2.6668 | 28.12% |
| 314 | validation | augmented | 18/48 | 0.585019 | 2.808987 | [11, 16, 17, 9, 23, 7, 13] | 2.7113 | 23.96% |
| 314 | final | baseline | 11/48 | 0.884128 | 4.752809 | [20, 10, 13, 12, 8, 9, 24] | 2.6959 | 25.00% |
| 314 | final | augmented | 17/48 | 0.523890 | 2.844064 | [6, 19, 19, 6, 21, 15, 10] | 2.6631 | 21.88% |

Mirror action agreement and mirrored-Q consistency are distinct: matching argmax actions do not imply matching values. Q MAE averages each family’s mean over aligned legal actions, giving each family equal weight. Maximum discrepancy is the worst aligned legal action. Lowest-column tie breaking can itself break action symmetry. Histograms include unlabeled original-suite probes; entropy and dominant share describe concentration on these suites, not the entire policy state distribution.

| Seed | Suite | Families | B both/one/neither | A both/one/neither | Rows retained/improved/regressed | Family Δ −2/−1/0/+1/+2 |
| --- | --- | --- | --- | --- | --- | --- |
| 73 | existing | 24 | 5/6/13 | 5/10/9 | 8/12/8 | 1/3/13/5/2 |
| 73 | validation | 48 | 4/13/31 | 7/13/28 | 9/18/12 | 2/5/30/7/4 |
| 73 | final | 48 | 4/17/27 | 5/18/25 | 6/22/19 | 1/11/23/10/3 |
| 314 | existing | 24 | 1/7/16 | 9/5/10 | 5/18/4 | 0/3/9/7/5 |
| 314 | validation | 48 | 3/14/31 | 10/12/26 | 8/24/12 | 1/9/20/13/5 |
| 314 | final | 48 | 1/13/34 | 12/13/23 | 7/30/8 | 0/6/22/12/8 |

The original tactical rows count as **24 correlated families**, and each 96-row suite as **48 correlated families**. They are not 48 or 96 independent trials. The full original suite has 30 mirror families including probes. Family deltas count changes in the number of correct orientations; zero can include an orientation exchange. No independent-row confidence interval or significance claim is made. Two new seed pairs cannot establish statistical significance.

## Negamax and bounded Random results

| Seed | Method | Opponent | Games | W/L/D | As X W/L/D | As O W/L/D | Win rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 73 | baseline | negamax_1 | 40 | 4/36/0 | 1/19/0 | 3/17/0 | 10.0% |
| 73 | baseline | negamax_2 | 40 | 2/38/0 | 2/18/0 | 0/20/0 | 5.0% |
| 73 | baseline | random | 20 | 14/6/0 | 8/2/0 | 6/4/0 | 70.0% |
| 73 | augmented | negamax_1 | 40 | 7/33/0 | 3/17/0 | 4/16/0 | 17.5% |
| 73 | augmented | negamax_2 | 40 | 2/38/0 | 2/18/0 | 0/20/0 | 5.0% |
| 73 | augmented | random | 20 | 14/6/0 | 8/2/0 | 6/4/0 | 70.0% |
| 314 | baseline | negamax_1 | 40 | 7/33/0 | 3/17/0 | 4/16/0 | 17.5% |
| 314 | baseline | negamax_2 | 40 | 2/38/0 | 2/18/0 | 0/20/0 | 5.0% |
| 314 | baseline | random | 20 | 16/4/0 | 10/0/0 | 6/4/0 | 80.0% |
| 314 | augmented | negamax_1 | 40 | 12/28/0 | 8/12/0 | 4/16/0 | 30.0% |
| 314 | augmented | negamax_2 | 40 | 4/36/0 | 4/16/0 | 0/20/0 | 10.0% |
| 314 | augmented | random | 20 | 17/3/0 | 8/2/0 | 9/1/0 | 85.0% |

| Seed | Opponent | Δ wins | Δ win-rate pp |
| --- | --- | --- | --- |
| 73 | negamax_1 | 3 | +7.5 |
| 73 | negamax_2 | 0 | +0.0 |
| 73 | random | 0 | +0.0 |
| 314 | negamax_1 | 5 | +12.5 |
| 314 | negamax_2 | 2 | +5.0 |
| 314 | random | 1 | +5.0 |

All 400 scheduled games completed: 320 Negamax and 80 Random, with zero partial games. Measured evaluation driver times were 1.955 seconds for seed 73 and 1.839 seconds for seed 314, well below their caps. Paired-opening side assignments and reflected prefixes are correlated, and the same openings are reused across seeds. This is a bounded comparison against unchanged shallow heuristics, not a competitive-strength rating. Random uses varied openings, matching Phase 4C.2d; it differs from the older 100 empty-board games.

| Seed | Model/opponent | Greedy columns 0–6 | Entropy bits | Dominant share |
| --- | --- | --- | --- | --- |
| 73 | baseline/negamax_1 | [11, 43, 34, 42, 37, 20, 25] | 2.6991 | 20.28% |
| 73 | baseline/negamax_2 | [16, 45, 43, 54, 42, 36, 35] | 2.7414 | 19.93% |
| 73 | baseline/random | [17, 32, 18, 32, 24, 20, 20] | 2.7631 | 19.63% |
| 73 | augmented/negamax_1 | [20, 29, 41, 54, 50, 34, 6] | 2.6207 | 23.08% |
| 73 | augmented/negamax_2 | [14, 36, 40, 55, 46, 43, 11] | 2.6479 | 22.45% |
| 73 | augmented/random | [7, 5, 23, 27, 30, 38, 6] | 2.4858 | 27.94% |
| 314 | baseline/negamax_1 | [25, 23, 31, 30, 17, 17, 31] | 2.7678 | 17.82% |
| 314 | baseline/negamax_2 | [34, 36, 39, 37, 22, 16, 37] | 2.7525 | 17.65% |
| 314 | baseline/random | [18, 8, 29, 30, 13, 11, 31] | 2.6515 | 22.14% |
| 314 | augmented/negamax_1 | [3, 29, 36, 46, 43, 25, 13] | 2.5644 | 23.59% |
| 314 | augmented/negamax_2 | [12, 45, 52, 51, 55, 40, 21] | 2.6775 | 19.93% |
| 314 | augmented/random | [1, 38, 27, 33, 22, 12, 14] | 2.5147 | 25.85% |

## Numerical health, replay and calibration

| Run | Mean loss | Min loss | Max loss | Collected-state Q range | Max abs gradient | Reflected samples / total |
| --- | --- | --- | --- | --- | --- | --- |
| 73/baseline | 0.015766 | 0.000247 | 0.099388 | -4.4779 / 4.6738 | 0.171416 | 0 / 5,760,000 |
| 73/augmented | 0.020265 | 0.000673 | 0.084124 | -3.8754 / 4.2774 | 0.140397 | 2,879,640 / 5,760,000 |
| 314/baseline | 0.014822 | 0.000310 | 0.080821 | -4.2306 / 4.0125 | 0.149697 | 0 / 5,760,000 |
| 314/augmented | 0.020530 | 0.000711 | 0.088809 | -3.9439 / 3.9626 | 0.139714 | 2,882,051 / 5,760,000 |

All losses, per-ply Q outputs, gradients, saved model weights and diagnostic values were finite; zero mathematical/numerical invariant failures or artifact errors occurred. Exact safe reload equality passed for every model tensor and original-suite prediction at save time; the independent audit reloaded all four checkpoints with `weights_only=True` and exactly reproduced all three suites. Evaluation left loaded weights and checkpoint bytes unchanged. Finite values and small training losses do not establish tactical reliability or calibrated values.

| Run | Replay terminal/nonterminal | Replay reward 0/1 | Replay X/O | Collected rewards | Collected X/O |
| --- | --- | --- | --- | --- | --- |
| 73/baseline | 580/9420 | 9420/580 | 5163/4837 | {'0': 85158, '1': 4905} | {'0': 46421, '1': 43642} |
| 73/augmented | 514/9486 | 9487/513 | 5145/4855 | {'0': 85282, '1': 4781} | {'0': 46388, '1': 43675} |
| 314/baseline | 534/9466 | 9466/534 | 5138/4862 | {'0': 85263, '1': 4800} | {'0': 46404, '1': 43659} |
| 314/augmented | 546/9454 | 9454/546 | 5152/4848 | {'0': 85260, '1': 4803} | {'0': 46359, '1': 43704} |

| Seed | Method | Final-holdout legal Q range | Mean winning-action Q | Winning-action Q min/max | Negative winning-action Q |
| --- | --- | --- | --- | --- | --- |
| 73 | baseline | -2.9717 / 2.3772 | -0.132162 | -2.4865 / 1.6618 | 29/48 |
| 73 | augmented | -1.6533 / 1.4578 | -0.024692 | -1.2813 / 1.2723 | 25/48 |
| 314 | baseline | -3.3562 / 2.3815 | -0.006683 | -1.6401 / 2.0187 | 25/48 |
| 314 | augmented | -2.0617 / 1.9394 | 0.108067 | -1.2594 / 1.6451 | 20/48 |

An immediate winning action has exact terminal target +1. Negative winning-action values and values beyond the reward scale expose calibration problems even when the chosen action happens to be correct. Losses use changing replay and targets; they are not a fixed validation objective. Event-level health, all losses, episode moves, replay composition and augmentation counts are retained in each run’s report/events.

## Between-seed variability and historical seed 42

| Suite | Seed 73 Δ pp | Seed 314 Δ pp | Descriptive mean Δ pp | Observed Δ range |
| --- | --- | --- | --- | --- |
| existing | +8.33 | +29.17 | +18.75 | +8.33 to +29.17 |
| validation | +6.25 | +12.50 | +9.38 | +6.25 to +12.50 |
| final | +3.12 | +22.92 | +13.02 | +3.12 to +22.92 |

The final-holdout effect ranges from **+3/96 to +22/96**. Baseline accuracy itself varies from 15/96 to 25/96, and augmented accuracy from 28/96 to 37/96. Mean tactical margins improve on all six comparisons, but absolute augmented accuracy stays low. The small seed-73 aggregate gain is accompanied by **22 improved and 19 regressed rows**: only six of its 25 baseline-correct rows remain correct. At family level, 13 improve, 12 regress and 23 have no net change. Seed 314 has 30 improved/eight regressed rows and 20 improved/six regressed/22 unchanged families. These are substantial prediction changes despite a positive aggregate direction.

Category-specific reversals remain. Seed 73 final-holdout vertical accuracy falls **15/32 → 12/32**, while seed 314 rises **6/32 → 21/32**. Seed 73 final blocks stay at **15/48**, versus seed 314 improving **10/48 → 19/48**. On the previously inspected Phase 4C.2d suite, X accuracy falls **14/48 → 11/48** for seed 73 but rises **9/48 → 17/48** for seed 314. Final horizontal mean margin worsens slightly for seed 73 even though its correct count rises by one. Aggregate accuracy and mean margin therefore do not establish uniform tactical improvement.

Mirrored-value improvement is more consistent in magnitude: final MAE drops from **0.968802 → 0.544576** and **0.884128 → 0.523890**. Across all six comparisons the reduction ranges from 30.65% to 44.81%, and worst aligned discrepancies also decrease. Final mirror action matches improve **7/48 → 13/48** and **11/48 → 17/48**, but seed 73 original-suite action agreement regresses **9/30 → 8/30**. The augmented policies still disagree across most final mirror families.

Greedy concentration has no uniform suite-wide direction. Final-holdout dominant shares decrease to 21.88% for both augmented models, but seed 314 original-suite column-4 share rises from 11.67% to 33.33%; seed 73 original-suite entropy increases while its Random-game entropy drops from 2.7631 to 2.4858 bits. Game-policy entropy falls under augmentation for every opponent in both pairs. The measured policies use every column, but concentration and distribution dependence remain; no claim of globally collapse-free behavior follows.

Depth-1 gains are +3 and +5 wins; depth-2 gains are zero and +2 wins. The augmented depth-2 win rates are still only 5% and 10%, entirely from X. Random is unchanged at 14/20 for seed 73 and rises 16/20 → 17/20 for seed 314, while seed 314 Random X wins regress 10/10 → 8/10 and O wins rise 6/10 → 9/10. Those small aggregate outcomes do not support a consistent broad strength gain across opponent difficulty and playing side.

Historical seed 42 is context only: its baseline used **94,783 updates** and augmented model **96,398 updates** (+1,615, +1.70%), both stopped at 5,000 games; neither was retrained or evaluated on the new final holdout. Its original accuracy improved 15/48 → 26/48, Phase 4C.2d holdout 24/96 → 31/96, aligned mirrored-Q MAE 0.576036 → 0.433437 (original) and 0.783288 → 0.525572 (holdout). Negamax outcomes were unchanged: both 6/40 wins at depth 1 and 2/40 at depth 2. The comparable Phase 4C.2d Random schedule favored baseline 17/20 over augmented 12/20, whereas the earlier empty-board schedule favored augmentation 75/100 → 92/100. Different exposure and protocols prevent treating seed 42 as a third equal-budget replicate.

The final holdout and opening schedules were frozen before training, with no tuning or regeneration after inference. Quota selection, the small number of families, shared reflected openings, shallow opponents and only two new training seeds limit generalization. No checkpoint is selected for deployment or declared a winner solely from these diagnostics.

## Integrity, provenance and retained artifacts

All **80** pre-existing files in `preservation-manifest.json` retain their exact SHA-256 values, including previous checkpoints, training events/reports, handoffs and both earlier suites. The audit also checked all 360,252 collected training plies and 400 evaluation transcripts for legal transitions, actor attribution and outcomes, reproduced every Random/Negamax opponent choice, and verified paired schedules, summaries and tactical predictions. The audit performs no learning or additional game rollouts.

| Run | Separate Git-ignored directory | Candidate SHA-256 |
| --- | --- | --- |
| 73 baseline | [`experiment-output/phase4c2e-seed73-baseline/`](../experiment-output/phase4c2e-seed73-baseline/) | 1dbd56c214f0c500390dea280493aaeb4500cedcc2bc7a7c02e1a10c2772ecf7 |
| 73 augmented | [`experiment-output/phase4c2e-seed73-augmented/`](../experiment-output/phase4c2e-seed73-augmented/) | bc52c8fc96849ca4487547d92da6c64b0c85d7fb91f275d24f1a91b6e37a8554 |
| 314 baseline | [`experiment-output/phase4c2e-seed314-baseline/`](../experiment-output/phase4c2e-seed314-baseline/) | ab7d9ff44462285bcbf9a97049d728ae2c84437f6b5a6507c30c6dd5a2f77a6e |
| 314 augmented | [`experiment-output/phase4c2e-seed314-augmented/`](../experiment-output/phase4c2e-seed314-augmented/) | bbcce0423218eed3dbc5b3fe7f1cff4a0e639e6b990102dab9aa816580692cec |

Each run directory contains its version-1 inference-only `candidate.pt`, `configuration.json`, full `report.json`, per-ply/episode `events.jsonl`, tracked source patch, archived untracked source/fixture/test bytes, and SHA-256 artifact manifest. These checkpoints are not resumable optimizer/replay snapshots. The shared [replication artifact directory](../experiment-output/phase4c2e-replication/) contains frozen configuration and timestamp, preservation manifest, all gate logs, launch/completion records, exact command arguments, console logs, complete evaluation JSON and JSONL game transcripts, audit source/results, reporting source and final artifact manifest. Artifacts are local and Git-ignored; the tracked final fixture, evaluation protocol, tests and handoff are reviewable source changes.

The source identity records HEAD, branch, dirty tracked diff and untracked source hashes; the commit alone does not identify the launch source. Each run captures the same implementation/fixtures, with no source edits during training. Runtime was Python 3.11.17, PyTorch 2.10.0 and macOS 26.6.2 ARM64. Versions and actual deterministic/thread settings are in every run’s provenance. Runtime values are local measurements, not deployment throughput guarantees.

## Recommendation and stopping point

**Close Phase 4C’s current DQN training investigation here.** The equal-update replication supports horizontal augmentation as a useful contributor to aggregate tactical accuracy and mirrored-value consistency in this configuration. It does not make the learned policy tactically reliable or strong enough for standalone public gameplay. The seed-73 augmented model misses **35/48 immediate wins and 33/48 blocks** on the final holdout; seed 314 misses **30/48 wins and 29/48 blocks**. Numerical checks passed, but these functional failures remain material.

Retain all four candidates and the historical artifacts for reproducible offline comparisons. Do not select a deployment checkpoint from the highest diagnostic score, start another DQN sweep, or keep extending the investigation to obtain a more favorable result. Phase 4C.3 public integration/readiness is deferred rather than implicitly passed.

Proceed, under a separate assignment, to the already proposed **Phase 4D.1 correctness work** in [the Phase 4B audit](phase4b/audit.md): establish and test neural-search/value and supervision contracts with deterministic fake/oracle networks before any fresh training. That recommendation does not authorize MCTS-NN implementation, training, integration or hosting changes in this milestone. No particular additional DQN learning-method change is established as the solution by these results.

Stopped after the four authorized runs, frozen evaluation, integrity audit and this handoff. No additional training or integration was performed.

