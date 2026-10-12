# Phase 4D.4H-C — independent results review

This is an independent review carried out on the same host, from the same
worktree and the same local checkpoints. It is not an external replication.
The reviewer took no part in the design, the run or the frozen analysis.

Verdict of the review in three lines:

- Every number and every declared verdict in `results/analysis.json` was
  reproduced. No integrity or protocol violation was found. There are no
  Critical findings.
- The strength FAIL is the correct outcome of the frozen rule and is stable
  under resampling of openings.
- As evidence that ResNet 3x64 is the weaker architecture at equal
  computation, the FAIL is weak. It comes from one seed of two, the same rule
  is already met by the untrained networks, and a single arm's score moves
  between adjacent checkpoints by as much as the difference between
  architectures. What is supported is "no evidence of a ResNet advantage at
  this budget", not "the ResNet is worse".

## Scope and method

Reviewed: `DESIGN.md`, `results/design.json`, `analyze.py` (read, not
imported), `results/analysis.json`, `results/tables.md`, the raw evidence in
`results/run-evidence/`, `fixtures.json`, the development opening package, the
28 local checkpoints, and the draft write-up
`docs/phase4d4h-c-resnet-selfplay.md` as it stood during the review.

Method: `independent_review.py` imports nothing from
`research.alphazero_resnet_selfplay`. It contains its own Connect 4 rules,
its own tactical labeller, its own metric definitions, its own cluster
bootstrap (reviewer seed 20261012, streams unrelated to the frozen 4404903),
its own t and exact sign-flip tests, and its own implementation of the
declared rules. It writes `results/independent-recomputed.json`. Project code
under `games.connect4` is used in three places only: the engine as a second
opinion on game replays, the alpha-beta solver for a time-capped re-solve of
exact labels, and the frozen loader, search and arena to re-run the
generation-6 endpoints from the hashed checkpoints.

Run: one thread, 329 s for the full script, of which 260 s is the
re-execution from checkpoints. The published JSON was regenerated once more
with `--reuse-reexecution` after two descriptive fields were added (gradient
clipping, equal-games time ratio); its re-execution section is the one from
the full run and is marked as reused.

## What was confirmed

Point estimates were compared with `analysis.json` to full precision.
Interval ends come from a different bootstrap stream, so they are expected to
differ slightly.

| Item | Compared | Largest absolute discrepancy |
|---|---|---|
| Match summaries per arm and checkpoint (score, W/D/L, by color, plies) | 128 | 0 |
| Match contrasts, per seed and pooled (generations 0, 2, 4, 6 and equal games) | 15 | 0 |
| Opening-family differences at generation 6 | 80 | 0 |
| Raw-network means, every arm, generations 0 to 6, 13 measures | 364 | 0 |
| Raw-PUCT means, every arm, generations 0, 2, 4, 6, 8 measures | 128 | 0 |
| Position contrasts, per seed and pooled, all pairings | 471 | 0 |
| Computation totals and ratios | 37 | 0 |
| Verdict inputs (mean final score, gate and value contrasts) | 6 | 0 |
| Match interval ends, other stream | 10 pooled, 20 per seed | 0.0031 pooled, 0.0063 per seed (one grid step) |
| Position interval ends, other stream | 314 | 0.0090 |
| Supporting-rule verdicts that change with the reviewer's stream | 162 contrasts | none |

**1. Matches.** All 1,280 games (16 arm-checkpoints x 80) were replayed with
the reviewer's rules and again with the project engine. Every move is legal,
no move follows a finished game, every record ends at a terminal position,
and the recorded winner, result and score equal the replay in every game.
Every arm and checkpoint played exactly the 40 declared `-base` families of
the development package, each twice, with `agent_color` 0 and 1. Agent
settings (256 simulations, no root noise, no tactical guard, temperature 0),
opponent (reference negamax depth 3) and weight hashes match the design and
the ledger.

| Arm | g0 | g2 | g4 | g6 | W/D/L at g6 |
|---|---|---|---|---|---|
| Seed 42 CNN | 0.131 | 0.431 | 0.381 | 0.575 | 44/4/32 |
| Seed 42 ResNet | 0.069 | 0.244 | 0.313 | 0.306 | 22/5/53 |
| Seed 314159 CNN | 0.188 | 0.406 | 0.450 | 0.463 | 35/4/41 |
| Seed 314159 ResNet | 0.069 | 0.344 | 0.519 | 0.413 | 32/2/46 |

**2. Primary contrast**, ResNet minus CNN, reviewer's bootstrap (10,000
replicates, families resampled within seed, seeds resampled independently):

| Pairing | Seed 42 | Seed 314159 | Pooled | Frozen pooled interval |
|---|---|---|---|---|
| Generation 0 | -0.063 [-0.125, +0.000] | -0.119 [-0.213, -0.025] | -0.091 [-0.150, -0.034] | [-0.147, -0.034] |
| Generation 2 | -0.188 [-0.319, -0.050] | -0.063 [-0.188, +0.050] | -0.125 [-0.216, -0.038] | [-0.216, -0.034] |
| Generation 4 | -0.069 [-0.194, +0.056] | +0.069 [-0.063, +0.206] | +0.000 [-0.094, +0.094] | [-0.091, +0.094] |
| **Generation 6** | **-0.269 [-0.400, -0.138]** | **-0.050 [-0.200, +0.100]** | **-0.159 [-0.259, -0.056]** | [-0.263, -0.059] |
| CNN g4 vs ResNet g6 (equal games) | -0.075 [-0.194, +0.044] | -0.038 [-0.181, +0.113] | -0.056 [-0.150, +0.038] | [-0.150, +0.041] |

Mean generation-6 score of the four arms is 0.439, inside [0.10, 0.90].

**Reviewer's verdicts under the declared rules:**

| Rule | Reviewer | Frozen analysis |
|---|---|---|
| Strength | FAIL | FAIL |
| Tactical gate, raw PUCT (binding) | PASS | PASS |
| Tactical gate, raw policy (not binding) | PASS | PASS |
| Value on solved300 | WORSE | WORSE |
| Equal-compute claim | APPROXIMATELY_EQUAL (0.967) | APPROXIMATELY_EQUAL (0.967) |
| Recommendation rule | KEEP_CNN | KEEP_CNN |

**4. Tactical gate.** Correctness was re-derived from `fixtures.json` and
never read from the stored flags. Separately, all 1,200 tactical label fields
(legal moves, immediate wins, opponent threats, safe moves, category,
expected action) were re-derived from the move lists with the reviewer's
rules: no mismatch. Across 27,632 stored rows, every stored flag and value
equals the re-derived one, every `policy_action` is the unique arg-max of its
stored policy, every policy is legal-masked, every PUCT visit vector sums to
256 and every `puct_action` is a visit arg-max (179 visit ties, seeded).

| Generation 6, raw PUCT | Seed 42 | Seed 314159 | Pooled |
|---|---|---|---|
| Immediate win (100 rows, 39 clusters) | +0.000 | -0.030 | -0.015 [-0.044, +0.007] |
| Safe response (100 rows, 40 clusters) | +0.000 | +0.060 | +0.030 [-0.033, +0.096] |

PASS on 50 of 50 further bootstrap streams. Raw policy: +0.040
[-0.037, +0.122] and +0.110 [+0.039, +0.182], PASS.

**5. Value and policy.** Exact-value MSE was recomputed as (stored value minus
fixture label) squared. Constant-zero baseline: 0.6667 (100 wins, 100 draws,
100 losses).

| Generation 6, solved300 | Seed 42 CNN | Seed 42 ResNet | Seed 314159 CNN | Seed 314159 ResNet |
|---|---|---|---|---|
| Exact-value MSE | 0.647 | 0.803 | 0.689 | 0.862 |
| MSE over constant-zero | 0.97 | 1.20 | 1.03 | 1.29 |

Contrast: seed 42 +0.156 [+0.010, +0.297], seed 314159 +0.173
[-0.005, +0.364], pooled +0.165 [+0.050, +0.283]. WORSE on 50 of 50 further
streams. Optimal-action-set cross-entropy and accuracy were recomputed from
the stored policy vectors and fixture labels at every generation and match
`analysis.json` exactly. Solved and deep labels are internally consistent in
all 428 rows (legal set, value equals the best action value, optimal set,
immediate wins valued +1). Full action values of 217 positions were re-solved
with the project's alpha-beta solver with no disagreement; 19 exceeded the
node budget and 192 were not reached in the 60 s cap.

**6. Computation and scheduling.**

| | Seed 42 CNN | Seed 42 ResNet | Seed 314159 CNN | Seed 314159 ResNet |
|---|---|---|---|---|
| Games / updates | 864 / 864 | 576 / 576 | 864 / 864 | 576 / 576 |
| Per generation | 144 / 144 | 96 / 96 | 144 / 144 | 96 / 96 |
| Self-play + update seconds | 764.87 | 740.06 | 767.42 | 741.28 |
| ResNet / CNN | | 0.9676 | | 0.9659 |

Pooled ratio 0.9667, inside [0.90, 1.10]. Budgets, counters and event counts
equal the frozen budgets in all 24 training tasks. Processor over wall time
in collection is 0.992 to 1.000. Cost per neural position varies within an
arm by at most 4.9% across generations (CNN 146.6 to 149.8 microseconds,
ResNet 219.2 to 230.0), so there is no sign of a disturbed interval. The 24
training tasks ran in the declared ABBA order with no overlap (smallest gap
0.004 s); the first evaluation task started 0.012 s after the last training
task ended; at most three evaluation tasks ran together.

**7. Integrity and protocol.**

| Check | Result |
|---|---|
| 28 local checkpoints re-hashed against `status.json` | all equal |
| Model chain, 24 links: collection weights = previous learner weights; loaded state = previous saved state; parent hash; ledger hashes | no break |
| Generation 0 against `freeze.json` and `launch.json`; empty replay, zero steps | equal |
| `run-evidence/design.json` against `results/design.json` | byte-identical |
| 48 files listed in `freeze.json` | all hashes unchanged |
| Design commit 18ba3b5 against run start | commit 10.5 s before start; local reflog shows the push 9.5 s before start; `git ls-remote`, run by hand during the review, shows origin at 18ba3b5; HEAD is 18ba3b5 |
| Tasks | 60 scheduled, 60 COMPLETE, return code 0, no retry |
| Evaluation identities (32 `identity.json`, 28 dynamics rows) | all equal the published boundaries |
| 276 published evidence files against the local originals and the manifest | all equal |
| Fixtures SHA-256 `80f68518…ca2c` | equal |
| Sealed packages | no phase source references one; the only frozen package path is `openings-development.json`, with a split check |
| `analyze.py` read for pairing, sign, category, cluster and generation errors | none found |

**Re-execution from the checkpoints.** Each generation-6 boundary was loaded
strictly (file hash and declared configuration) and the endpoints were run
again without this phase's evaluation harness.

| Generation 6 | Raw outputs (500 positions) | Tactical PUCT (200 rows, action and visits) | Match games (80) | Replayed score |
|---|---|---|---|---|
| Seed 42 CNN | max difference 0 | 0 different | 0 different | 0.575 |
| Seed 42 ResNet | max difference 0 | 0 different | 0 different | 0.306 |
| Seed 314159 CNN | max difference 0 | 0 different | 0 different | 0.463 |
| Seed 314159 ResNet | max difference 0 | 0 different | 0 different | 0.413 |

The stored evidence for the primary endpoint, the tactical gate and the value
endpoint is therefore what these hashed checkpoints produce, bit for bit, on
this host.

## Findings

No Critical finding. Nothing found changes a number or a declared verdict.
The findings concern what the verdicts can carry.

| # | Finding | Severity | Evidence | Recommended wording or action |
|---|---|---|---|---|
| 1 | The strength FAIL rests on one seed of two. | Moderate | Seed 42: -0.269, t interval [-0.406, -0.132], exact sign-flip p = 0.0005, 21 families behind, 5 ahead. Seed 314159: -0.050, t interval [-0.211, +0.111], exact sign-flip p = 0.58, 15 behind, 14 ahead, 11 level. Seed 42 supplies 84% of the pooled difference. If both seeds looked like 314159 the pooled interval would be [-0.159, +0.059]. | Say "FAIL under the declared rule; the deficit is detectable on seed 42 only". Do not say the ResNet was shown weaker on both seeds. On seed 314159 only the sign is negative, by 4 points out of 80. |
| 2 | The "both seeds negative" condition is fragile on seed 314159. | Moderate | 72% of bootstrap replicates are negative, 25% positive. Removing the 2 opening families most favourable to the CNN (of 40) makes the seed difference non-negative; it takes 15 on seed 42. No single leave-one-out flips the sign. | State that the sign on seed 314159 is within opening noise. |
| 3 | The declared FAIL conditions are already met by the untrained networks. | Moderate | Generation 0: both seeds negative, pooled -0.091 [-0.150, -0.034], mean score 0.114 (inside the saturation band). The same conditions hold at generation 2 and fail at generation 4. Measured as gain from generation 0, the seeds disagree in sign: seed 42 -0.206 [-0.356, -0.056], seed 314159 +0.069 [-0.100, +0.244], pooled -0.069. | An opening-bootstrap interval that excludes zero does not by itself show a learned-strength difference: it already separates two random initializations. Report the generation-0 contrast beside the decision contrast, as the draft does, and draw this consequence from it. |
| 4 | Checkpoint-to-checkpoint movement of one arm is as large as the architecture difference. | Moderate | Absolute change between adjacent evaluated checkpoints (g2 to g4, g4 to g6), eight values: 0.006 to 0.194, mean 0.082, root mean square 0.105. Seed 42 CNN g4 to g6: +0.194 [+0.100, +0.288]. The seed-42 architecture difference itself moved by -0.200 [-0.369, -0.031] from g4 to g6. Pooled difference: 0.000 at g4, -0.159 at g6. With the CNN held at its g4 level the pooled difference is -0.056 and inconclusive (this is the equal-games pairing). | Generation 6 was fixed in advance, so the verdict stands. But the FAIL depends on whether the seed-42 CNN's last jump is durable learning or checkpoint noise, and the data cannot tell. Say so. |
| 5 | The two seeds do not estimate the same difference. | Moderate | Difference between the seed differences: -0.219, standard error 0.105, z = -2.09, p = 0.037 under opening sampling alone. | Run-to-run variation in the contrast is real and about as large as the contrast. The pooled interval excludes it by construction. With seeds as the unit (n = 2) the t interval is [-1.55, +1.23], p = 0.38. |
| 6 | The value WORSE verdict depends on the label mix and compares two value heads that are both no better than a constant. | Moderate | Both CNN heads sit at 0.97 and 1.03 of the constant-zero MSE; both ResNet heads at 1.20 and 1.29. The ResNet's mean raw value on the balanced suite is +0.18 and +0.33 (CNN -0.08 and +0.02): a bias toward the side to move. Its MSE is 1.30 and 1.71 on lost positions and 0.69 and 0.50 on won positions (CNN 0.70, 0.92 and 0.98, 0.94). On deep128, which has no lost positions, the sign reverses: -0.100 and -0.108, pooled -0.104 [-0.286, +0.053]. Seed 314159 alone: [-0.005, +0.364]. | Say "the ResNet's value is more confident and biased toward the mover; on a balanced solved suite that costs MSE; neither head beats a constant". Do not say the CNN learned a better value function. |
| 7 | The tactical gate PASS is weakly informative. | Minor | Untrained networks under search score 1.00 and 1.00 on all four arms. This is real in the rows: untrained policies are within 0.066 of uniform, untrained values have standard deviation at most 0.019, and the expected move takes 89% of visits on average (minimum 48%). Trained networks score 0.96 to 0.99 (immediate win) and 0.87 to 0.93 (safe response), with 11 to 14 wrong rows of 200 per arm. The gate's margin (lower bound at least -0.10) is wide against error rates of 0.01 to 0.13. | Say "the ResNet is not shown less reliable than the CNN under search, within the declared margin; both are below an uninformed search". Do not say "tactically equivalent". |
| 8 | Supporting contrasts fire between untrained networks as often as between trained ones. | Minor | 162 supporting contrasts, 36 flagged (16 BETTER, 20 WORSE), no multiplicity control. At generation 0, 4 of 22 are flagged (match score and deep128 value WORSE, two solved300 PUCT measures BETTER). At generation 6, 4 of 22. | Give isolated BETTER or WORSE verdicts no weight. The repeated ones (raw-policy safe response BETTER at g3, g4, g6 and equal games; solved300 value WORSE at g2, g4, g5, g6 and equal games) are the only supporting patterns worth stating. |
| 9 | The ResNet deficit at generation 6 is concentrated in one color. | Minor | Agent owns O: -0.244 [-0.369, -0.119] (seeds -0.338, -0.150). Agent owns X: -0.075 [-0.219, +0.069] (seeds -0.200, +0.050). | Descriptive only. The draft's "moving first / moving second" is inexact: 20 of the 40 openings have an odd prefix, so X is not the side to move at hand-over in half the families. Write "as X / as O". |
| 10 | The equal-games pairing supports nothing beyond the absence of a per-game advantage. | Minor | -0.056 [-0.150, +0.038], seeds -0.075 and -0.038. It pairs the ResNet's last checkpoint with the seed-42 CNN's lowest trained checkpoint (0.381), and the arms differ in refreshes (6 against 4) and in time (ResNet 1.43 and 1.47 times the CNN's, pooled 1.45). | Keep it descriptive. It does not show the ResNet learns as well per game, and it does not show it learns worse. |
| 11 | Realized computation favoured the CNN slightly; evaluation favoured the ResNet. | Note | The ResNet had 3.3% less training computation (about 25 s per seed, roughly 20 ResNet games). At equal simulations a ResNet evaluation costs about 1.5 times a CNN one (measured per-position cost ratio 1.505; per-update 1.438). | Both are disclosed. Neither is large against the effects. |
| 12 | Identical hyperparameters were not neutral. | Note | Recomputed from the per-update metrics: 0 of 1,728 CNN updates were clipped at norm 5; 62% and 65% of ResNet updates were (43% to 97% from generation 2). Mean gradient norm 0.8 to 1.7 for the CNN, 3.7 to 5.9 for the ResNet. | The comparison is "ResNet 3x64 under CNN-tuned optimizer settings". This is a further reason not to read the FAIL as a statement about the architecture. |
| 13 | Pooling the seeds by independent resampling is justified. | Note | Correlation of the 40 family differences between seeds: 0.04. Resampling families jointly gives [-0.263, -0.056]; a t interval with the family as unit across both seeds gives [-0.267, -0.052]. | None. |
| 14 | The freeze-before-training claim rests on local clocks. | Note | Commit time, the local reflog of the push and the supervisor's own launch check all agree, 10 s before the run. The remote holds the commit now. The push time cannot be confirmed from the server by this review. | State it as "committed and pushed about 10 s before launch by local record". |
| 15 | The 60 per-task logs are empty files. | Note | All 60 have the SHA-256 of the empty string. | They show that no task printed a warning. They carry no other evidence; the JSON outputs do. |
| 16 | Two tracked files changed in the working tree after the design commit. | Note | `docs/phase4d4h-c-resnet-selfplay.md` and `README.md`, edited by the authoring session during this review. Neither is in `freeze.json`. | None. Frozen sources are unchanged. |
| 17 | Wording in the draft write-up. | Minor | (a) Question 2: "ahead or level at every checkpoint" is true of the pooled difference; on seed 314159 the ResNet was ahead at generation 4 (+0.069). (b) "The CNN ends within 0.02 of a constant" is 0.020 below on seed 42 and 0.022 above on seed 314159: one head is slightly worse than the constant. (c) "Do not launch a longer ResNet campaign from this result" must not be read as evidence that a longer run would fail; the pilot is silent on that. | Tighten the three sentences. |

## Robustness of the primary verdict

**(a) Bootstrap stream.** 200 independent streams of 10,000 replicates: FAIL
200 times. Pooled upper bound between -0.0625 and -0.0563, lower bound
between -0.2656 and -0.2563. The verdict does not depend on the random
stream.

**(b) Leave one opening family out** (the same family removed from both
seeds): pooled difference between -0.176 and -0.138; every leave-one-out
upper bound is below zero (-0.077 to -0.042). Seed 42 stays between -0.288
and -0.250. Seed 314159 stays between -0.077 and -0.026: negative every
time, never far from zero.

**(c) Intervals that do not use the percentile bootstrap.**

| Method | Estimate | 95% interval | p |
|---|---|---|---|
| Welch t on the two seeds' 40 family differences | -0.159 | [-0.264, -0.055] | 0.003 |
| t with the family as unit across both seeds (n = 40) | -0.159 | [-0.267, -0.052] | 0.005 |
| Joint-family bootstrap | -0.159 | [-0.263, -0.056] | |
| Seed 42 alone, t | -0.269 | [-0.406, -0.132] | 0.0003 |
| Seed 314159 alone, t | -0.050 | [-0.211, +0.111] | 0.53 |
| Seeds as the unit of replication (n = 2, 1 degree of freedom) | -0.159 | [-1.549, +1.230] | 0.38 |

The first three agree with the frozen interval. The last row is the only one
that includes training-run variation, and with two runs it says nothing.

**(d) By color.** As X: -0.075 [-0.219, +0.069]. As O: -0.244
[-0.369, -0.119]. The interval excludes zero for one color only.

**(e) Exact tests on paired family differences** (every sign assignment
counted, differences in units of 0.25):

| | Families behind / level / ahead | Exact sign-flip p (two-sided) | Sign test p |
|---|---|---|---|
| Seed 42 | 21 / 14 / 5 | 0.0005 | 0.002 |
| Seed 314159 | 15 / 11 / 14 | 0.58 | 1.00 |
| Pooled, 80 seed-family pairs | | 0.0047 | |
| Pooled, 40 families flipped jointly | | 0.0057 | |

Game slots (opening and color) where the ResNet scored lower / higher than
the CNN: seed 42, 31 / 8; seed 314159, 20 / 18.

**(f) Checkpoint swings.**

| Arm | g2 to g4 | g4 to g6 |
|---|---|---|
| Seed 42 CNN | -0.050 [-0.163, +0.069] | +0.194 [+0.100, +0.288] |
| Seed 42 ResNet | +0.069 [-0.063, +0.194] | -0.006 [-0.131, +0.125] |
| Seed 314159 CNN | +0.044 [-0.094, +0.181] | +0.013 [-0.088, +0.113] |
| Seed 314159 ResNet | +0.175 [+0.031, +0.319] | -0.106 [-0.244, +0.031] |

Two of eight swings exclude zero by the same bootstrap that supports the
FAIL, and both are as large as the pooled architecture difference (0.159).
The architecture difference changed sign on seed 314159 between generations
4 and 6 (+0.069 to -0.050) and roughly quadrupled on seed 42 (-0.069 to
-0.269). The
mean of the generation-4 and generation-6 differences, which nobody
prespecified, is -0.169 on seed 42, +0.009 on seed 314159 and -0.080 pooled.
A single checkpoint is one draw from a trajectory whose noise is of the size
of the effect. Treating generation 6 as decisive is legitimate because it was
declared, but it is not strong.

**(g) One seed.** Yes. See findings 1, 2 and 5. The pooled interval excludes
zero because seed 42 is far from zero. Seed 314159 contributes a negative
sign and nothing more.

**How strong the evidence is.** Conditional on these four networks, this
opponent and these 40 openings, the pooled deficit is robust: every
resampling scheme, every leave-one-out and the exact test agree (p about
0.005). As evidence about the architectures at this budget it is weak: two
training runs, one of which shows no detectable difference; an
initialization-only contrast that already meets the FAIL conditions; and
within-arm checkpoint noise equal to the effect. A fair one-line summary is:
the ResNet was clearly not better, and was worse in one run of two.

## What may and may not be claimed

**Supported.**

- The declared rules give strength FAIL, tactical gate PASS, value WORSE,
  approximately equal computation (0.967) and KEEP_CNN. The run conformed to
  the frozen protocol.
- At approximately equal training computation in this pilot, ResNet 3x64
  showed no strength advantage. PASS needed a pooled difference of at least
  +0.05 with both seeds positive; the pooled difference was at or below zero
  at all four evaluated checkpoints and its generation-6 upper bound is
  -0.056.
- These two ResNet networks scored lower than these two CNN networks against
  negamax depth 3 at generation 6, clearly on seed 42.
- At generation 6 the ResNet's raw value error on solved300 is higher on both
  seeds, and neither architecture's value head beats a constant there.
- The ResNet's raw policy finds the unique safe response more often (+0.110
  [+0.039, +0.182]), on both seeds and at generations 3, 4 and 6. This did not
  appear under search.
- Keeping the CNN as the default is a sound decision on cost alone: the
  ResNet costs 1.5 times as much per position and showed no gain.

**Not supported.**

- "ResNet 3x64 is worse than the CNN at equal computation" as a statement
  about the architecture. Supported is only "no evidence it is better, and
  worse in one run of two".
- "The ResNet was weaker on both seeds" in any sense beyond the sign.
- Any statement about longer training, 512 simulations, equal thinking time,
  other opponents, a direct CNN-against-ResNet match, tuned optimizer settings
  or ResNet 5x64.
- "The ResNet learns as much per game" or "less per game" from the
  equal-games pairing. It is inconclusive and confounded.
- "The CNN has the better value function." It has the less biased of two
  uninformative ones.
- "The two are tactically equivalent." The gate is a non-inferiority check
  with a wide margin on a suite that an uninformed search solves completely.

**What should not be said at all.** That the result is statistically
significant evidence against the ResNet architecture; that a longer ResNet
campaign has been shown not to be worthwhile; that the generation-6 interval
reflects run-to-run uncertainty.

### Design limitations that bear on interpretation

- **Two seeds, one initialization each.** No estimate of run-to-run variance
  is possible; finding 5 shows it is not small.
- **Unequal games by construction** (576 against 864). Equal computation is
  the declared primary frame, so the ResNet is tested on whether cheaper
  learning per game compensates for fewer games. It did not here.
- **One decision checkpoint** on a trajectory with swings of 0.1 to 0.2.
- **An asymmetric rule.** PASS requires a margin of 0.05 and a non-negative
  generation-4 difference; FAIL requires only two negative signs and an
  interval below zero. A symmetric FAIL would still have been met here
  (-0.159, and 0.000 at generation 4), so this did not decide the outcome.
- **Generation-0 offset.** The untrained ResNets score 0.069 and the
  untrained CNNs 0.131 and 0.188. Untrained raw values are near-constant with
  different offsets (CNN +0.09 and +0.01, ResNet -0.09 and +0.01 on the
  tactical suite), which is enough to change deterministic search. The
  measure is sensitive to properties that are not learned strength.
- **One opponent, weak, deterministic, chosen with CNN checkpoints.** After
  training the arms score 0.31 to 0.58 against a depth-3 search. These are
  early, weak agents, and the endpoint mostly reflects short-horizon errors.
  An odd-depth negamax has its own horizon behaviour, and there is no second
  opponent and no direct match between the arms to check transitivity.
- **256 evaluation simulations, equal simulations rather than equal time.**
  Favours the ResNet by about 1.5 times in thinking time; not the deployed
  512.
- **Evaluation is deterministic.** The 80 games per arm are 40 openings times
  two colors, fully determined by the weights. The bootstrap treats the 40
  families as the sample; nothing else is random.
- **Shared optimizer settings bind differently** (finding 12).

## Not covered by this review

- No training was re-run. Self-play determinism, the trainer and the
  per-generation training metrics are taken from the evidence; only hashes,
  counters and the model chain were checked.
- Re-execution from checkpoints covered generation 6 only: raw outputs on
  solved300 and tactical200, raw PUCT on tactical200 and the matches. Stored
  outputs at generations 0 to 5, raw PUCT on solved300 and deep128, and
  deep128 raw outputs were checked for internal consistency and against
  labels, not re-run. The re-execution used the project's own frozen loader,
  search, arena and negamax, so it tests reproducibility of the evidence and
  not the correctness of that code.
- Exact labels of 211 of the 428 solved and deep positions were not re-solved
  (time cap and node budget), and the solver used is the project's, not an
  independent one. Tactical labels were fully re-derived independently.
- The dynamics measures (replay composition, self-play diversity, prequential
  held-out losses, batch-norm probes, calibration bins) were not recomputed.
  Numbers quoted from them in the write-up are the stored ones. Only gradient
  clipping and gradient norms were recomputed.
- `results/tables.md`, the figures and `report.py` were not audited line by
  line; the strength, tactical, value and computation figures quoted in the
  draft write-up were checked against the recomputation and agree.
- The calibration, the opponent-depth probe, the smoke test, the validation
  record and `results/regression-tests.txt` were not re-run.
- The official campaign files and the Phase 4D.4G and 4D.4H-B trees were not
  re-hashed; the supervisor's preservation results are quoted, not confirmed.
- Sealed packages were not opened. That no sealed data was used is supported
  by a source scan and by the fixtures' declared sources, not by comparing
  positions.
- The time of the push to origin could not be confirmed from the server.
- Same host, same Python and Torch build, same checkpoints. Nothing here
  speaks to reproducibility on another machine.
