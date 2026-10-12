# Phase 4D.4H-C — prospective protocol

Controlled from-scratch self-play comparison of the existing CNN against
ResNet 3x64. This file is frozen with the execution sources before any
training; its hash is in `results/freeze.json`. Exact numbers are in
`results/design.json` (budgets, seeds, thresholds, projection) and
`results/calibration.json` (measured costs).

## Question

Can ResNet 3x64 produce a stronger AlphaZero agent than the CNN when both get
approximately equal computation for self-play and optimization?

ResNet 5x64 is not included.

## Arms

Four arms: {CNN, ResNet 3x64} x seeds {42, 314159}. Each starts from its
seed-determined fresh initialization with an empty replay and zero optimizer
steps. No historical weight, replay or optimizer state is imported.

| | CNN | ResNet 3x64 |
|---|---|---|
| Identity | `connect4-conv64-128-128-policy7-scalar-tanh-v2` | `connect4-resnet3x64-bn-policy32-value32x64-scalar-tanh-v1` |
| Parameters | 325,896 | 322,440 |
| Generations | 6 | 6 |
| Games per generation | 144 | 96 |
| Optimizer updates per generation | 144 | 96 |
| Total games / updates per arm | 864 / 864 | 576 / 576 |

Identical for every arm, all unchanged project defaults: original eight-ply
exploration schedule (temperature 1 at plies 0-7, then 0); 256 PUCT simulations
per move; root noise epsilon 0.25, alpha 1; pool-eight batched self-play on the
bitboard backend; eight-generation replay window; uniform replay sampling;
reflection augmentation with probability 0.5; policy cross-entropy plus value
MSE; AdamW, learning rate 3e-4, betas (0.9, 0.999), eps 1e-8, weight decay
1e-4, gradient clip 5, batch 128; the Phase 4D.4F deterministic experimental
runner on one thread.

Per-game random streams depend on (seed, generation, game index), so the two
architectures of a seed share game identities where their indices overlap.
Initial weights are separated by architecture identity.

## Equal computation

The primary comparison is approximately equal total training computation:
self-play collection seconds plus optimizer-update seconds, on one thread of
the same machine.

Calibration (`calibrate.py`, before this freeze) used trained weights:
Phase 4D.4H-B's saved 3,000-update CNN and ResNet 3x64 states for both seeds,
opened read-only and hash-checked against the tracked 4D.4H-B record. It ran
pool-eight, 256-simulation collection in an ABBA order of half-generation
blocks adding up to one proposed generation per architecture (192 CNN games,
128 ResNet games), then batch-128 updates on disposable copies.

| Measured | CNN | ResNet 3x64 | Ratio |
|---|---|---|---|
| Self-play, seconds per game | 0.942 | 1.434 | 1.52 |
| Self-play, ms per ply | 29.2 | 43.2 | 1.48 |
| Self-play, ms per neural position | 0.150 | 0.223 | 1.49 |
| Mean game length, plies | 32.2 | 33.2 | |
| Update, ms | 45.0 | 65.1 | 1.45 |

These agree with Phase 4D.4H-B's untrained-network ratios (1.46 and 1.45).
The proposed 3:2 game and update ratio was kept.

**The proposed size was reduced before the freeze.** Six generations of 192
and 128 games project to about 75 minutes of training alone, which with
evaluation does not fit the 120-minute limit once calibration is counted.
Both budgets were scaled by 0.75 to 144 and 96. Updates stay equal to games,
which keeps roughly four training samples per new position.

Expected computation per seed, from the calibration: CNN 814 s self-play plus
39 s updates (853 s); ResNet 826 s plus 38 s (864 s); expected ResNet/CNN
ratio 1.01. The realized ratio is reported separately and classified:

| Realized ResNet/CNN ratio | Statement allowed |
|---|---|
| 0.90 to 1.10 | Approximately equal computation |
| 0.80 to 1.25 otherwise | Roughly comparable computation, stated with the ratio |
| Outside | No equal-compute claim of any kind |

Game length changes during training and differs between arms, so the realized
ratio can differ from the expected one. Budgets are never adapted to correct
it.

Trajectories are also reported by cumulative games and updates. CNN generation
4 and ResNet generation 6 both have 576 games and 576 updates, and CNN
generation 2 and ResNet generation 3 both have 288. These pairings are
descriptive. The checkpoints differ in generation count, number of network
refreshes and wall time, and they are not a matched experiment.

## Execution

`manage.py launch` verifies the freeze, the pushed design commit, the initial
boundaries and the preservation inventories, then starts a supervisor in its
own session and returns. No interactive session waits on it.

- **Training** is strictly sequential, one generation of one arm per fresh
  interpreter, each strictly reloading the previous boundary with its expected
  hash and declared configuration. Order within a generation is seed 42 CNN,
  seed 42 ResNet, seed 314159 ResNet, seed 314159 CNN, reversed on even
  generations.
- **Evaluation** starts only after the last training generation, in a pool of
  three single-thread workers. Evaluation is deterministic and its timing is
  not part of any comparison. Final-checkpoint tasks run first.
- One absolute deadline for the whole run: 7,200 s minus the seconds already
  measured by calibration, the smoke test and validation. Workers check it and
  a cancel file cooperatively; `manage.py cancel` requests a stop.
- No retry, no extra generation, no workload reduction. A failed or late
  training task ends the run incomplete. A failed evaluation task is recorded
  and not retried; independent evaluation tasks still run.
- Durable evidence: `status.json` ledger rewritten at every transition,
  `supervisor.log`, one log and one output directory per task, and a coarse
  `progress.json` inside a running generation.

Recovery: published boundaries are complete generations and can be inspected
with `experimental_cli inspect`. A generation that did not finish leaves no
boundary. Any continuation needs a new prospective declaration.

## Evaluation

All on the nonsealed development fixtures. Sealed packages are never opened.

**Fixtures.** `fixtures.json` is Phase 4D.4G's frozen file, byte-identical
(SHA-256 `80f68518…ca2c`): solved300 and deep128 with complete exact
legal-action values, and tactical200 with independent immediate-win and
safe-response labels. Opening families are all 40 base prefix families of the
development opening package.

**Checkpoints.** Raw-network measures at every generation 0 to 6. Raw PUCT on
the fixtures and matches at generations 0, 2, 4 and 6.

**Search.** Raw PUCT, 256 simulations for every model, root noise off, no
tactical guard, temperature 0 with seeded ties. One fixed search seed per
fixture position, shared by every arm, seed and checkpoint (search seed
4404901). 256 was chosen over the project's 512 default so that evaluation
fits the time limit. Equal simulations is not equal thinking time: a ResNet
simulation costs about 1.5 to 1.8 times a CNN one.

**Opponent.** Reference negamax (`connect4-reference-negamax-v1`) at depth 3,
seeded uniform ties, the same for every model. The depth was chosen by a rule
written into `calibrate.py` before the probe ran: among depths 1 to 3, the one
whose mean score against official CNN generation 2 to 4 checkpoints is closest
to 0.50, with every probed checkpoint inside [0.15, 0.85].

| Depth | Scores of official seed 42 g2, seed 42 g4, seed 314159 g3 (40 games each, 256 simulations) | Mean |
|---|---|---|
| 1 | 0.738, 0.675, 0.613 | 0.675 |
| 2 | 0.313, 0.225, 0.375 | 0.304 |
| 3 | 0.525, 0.413, 0.475 | 0.471 |

Those official checkpoints had 512 to 1,024 games, which brackets this
experiment's 576 and 864.

**Matches.** 40 opening families x both colors = 80 games per arm and
checkpoint, match seed 4404902, identical random-stream identities for every
arm. Score is (wins + half of draws) / games.

**Measures.**

- Policy: optimal-action-set cross-entropy, exact optimal-action accuracy and
  probability on optimal actions (solved300, deep128).
- Value: exact-value MSE against solved outcomes, ten-bin calibration, the
  constant-zero baseline, and a prequential held-out diagnostic (the network
  of generation g scored on the games of generation g+1, which it generated
  and was not trained on) against a fixed sample of its training replay.
- Tactics: immediate-win and safe-response accuracy, avoidable one-ply errors,
  missed wins (tactical200) and deeper exact-value mistakes (solved300,
  deep128), each for the raw policy and for raw PUCT.
- Strength: match score against the fixed opponent, by color, by opening
  family and by seed.
- Dynamics, per generation: self-play diversity and one-ply errors, replay
  composition, sampling by source generation, policy and value losses,
  gradient norms and clipping, exact update and position counts, inference
  throughput, games and updates per second, batch-norm running statistics and
  train-mode against eval-mode predictions, failed or interrupted attempts.

Self-play win rates and training loss are diagnostics only and never stand in
for strength.

## Statistics

Every contrast is ResNet minus CNN. Within a seed the two arms are paired on
identical rows (fixture positions, or opening family and color). Uncertainty is
a cluster bootstrap, 10,000 replicates, seed 4404903 with a separate stream per
contrast: opening families for matches (both colors stay together) and the
Phase 4D.4G mirror/prefix clusters for fixtures (102, 71 and 28 clusters). The
pooled contrast is the equal-weight mean of the two seeds, resampled
independently. Intervals are two-sided 95% percentile intervals.

The intervals describe these four trained networks on these fixtures. They do
not include run-to-run training variation, which two seeds cannot estimate. No
multiplicity correction: there is one primary endpoint.

With 80 games per arm the pooled match difference has a 95% half-width of
roughly 0.10 to 0.12, so only a difference of that size can pass.

## Decision rules

The decision checkpoint is generation 6 of each arm. Nothing is selected on
the best checkpoint, and no position, opponent, budget or threshold changes
after training starts.

**Primary: playing strength at approximately equal computation.** D is the
ResNet minus CNN match score.

- PASS: D > 0 on both seeds, pooled D >= +0.05, pooled 95% lower bound > 0,
  and pooled D at generation 4 >= 0.
- FAIL: D < 0 on both seeds and pooled 95% upper bound < 0.
- INCONCLUSIVE: anything else, and always if the mean generation-6 score of
  the four arms is outside [0.10, 0.90] (saturated endpoint).

A small positive difference with a wide interval is INCONCLUSIVE.

**Tactical reliability, an independent safety gate.** Raw-PUCT accuracy on
tactical200 at generation 6, immediate-win and safe-response categories
separately.

- PASS: both categories have pooled difference >= -0.03 and pooled 95% lower
  bound >= -0.10.
- FAIL: any category has pooled difference <= -0.05 and pooled 95% upper
  bound < 0.
- INCONCLUSIVE: anything else.

The same rule on the raw policy is reported and is not binding.

**Value learning, supporting.** Raw exact-value MSE on solved300 at generation
6: BETTER if the ResNet is lower on both seeds and the pooled interval excludes
zero, WORSE in the mirror case, otherwise INCONCLUSIVE.

**Every other contrast** uses the same BETTER / WORSE / INCONCLUSIVE rule and
is supporting evidence only.

**Recommendation.**

- Replace the CNN with ResNet 3x64: strength PASS, the equal-compute claim not
  invalid, and the tactical gate not FAIL.
- Keep the CNN: strength FAIL or tactical gate FAIL.
- Otherwise: no demonstrated advantage. The CNN stays the default because it
  is cheaper per position, and the ResNet is not rejected.

A negative or inconclusive result from this short pilot does not show that
ResNet 3x64 cannot outperform the CNN with more training.

If the ResNet does not replace the CNN, the next investment is a judgment
informed by these indicators, declared now:

- Value learning: generation-6 exact-value MSE at or above 0.9 times the
  constant-zero MSE in both architectures, or prequential held-out value MSE
  above the constant-zero MSE.
- Tactical search: generation-6 raw-PUCT immediate-win or safe-response
  accuracy below 0.95 in the better arm.
- Replay optimization: generation-6 replay mirror-duplicate fraction above
  0.40, or post-ply-8 self-play duplicate fraction above 0.30, in the better
  arm.

## Scientific questions and where they are answered

| Question | Evidence |
|---|---|
| 1. Does the ResNet learn useful tactics sooner? | Tactical contrasts at generations 2 and 4 (equal computation), raw-network tactical trajectory at every generation, and the equal-games pairings |
| 2. Does the CNN catch up when given more updates? | The change of each equal-computation contrast from generation 2 to 6, and the equal-games pairings |
| 3. Does either develop a more reliable value network? | Exact-value MSE, calibration and the prequential held-out gap |
| 4. Do better raw predictions translate into stronger PUCT? | Raw against PUCT contrasts on the same fixtures, beside the match contrasts |
| 5. Does the ResNet improve strength at equal computation? | The primary rule |
| 6. Is any advantage consistent across both seeds? | Per-seed signs, required by every rule |

## Integrity

Enforced at launch, in every worker and at the end, never overridden:

- Frozen execution-source bytes: the package execution closure plus this
  phase's sources, this protocol, the fixtures, the design and the
  calibration record (`results/freeze.json`).
- Runtime identity: Apple M4, CPython 3.11.17, Torch 2.10.0, one thread,
  deterministic algorithms.
- Experiment configuration: every strict load names the arm's declared
  configuration.
- Checkpoint hashes: every boundary is loaded with its expected SHA-256 and
  all are re-hashed at the end.
- The 150 official campaign files, byte for byte.
- Protected experiment bytes: every file present at launch under the Phase
  4D.4G and 4D.4H-B output and results trees and the historical checkout's
  `games/` tree must still exist with the same hash.

Phase 4D.4H-B's supervisor flagged a failure because another session
committed in a sibling worktree. Here another worktree's HEAD, cleanliness and
newly added files are recorded as observations in a separate
`external_commits` record and are never, alone, an integrity failure.

## Disclosed before the freeze

- The calibration's cost numbers and the opponent probe's scores. The probe
  used official CNN checkpoints, not an arm of this experiment.
- Eight match games per architecture of the Phase 4D.4H-B states against the
  chosen opponent, timed for cost. Their outcomes were not recorded.
- A pipeline smoke test (two generations of 12 and 8 games, 16 simulations,
  three openings) in a separate directory. Its networks and numbers are
  disposable and informed nothing but that the pipeline runs.
- No experimental arm was trained and no evaluation of one exists.

## Deviations from the proposed design

| Proposed | Frozen | Reason |
|---|---|---|
| 192 / 128 games and updates per generation | 144 / 96 | Does not fit the time limit at the measured cost |
| Evaluation budget not specified | 256 simulations | Time limit; equals the self-play budget |
| Sequential execution implied | Training sequential; evaluation in three parallel workers | Evaluation timing is not compared, and its results do not depend on scheduling |
