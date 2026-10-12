# Phase 4D.4H-B prospective protocol

Status: prospective. Frozen before any measured benchmark stage or fixed-data
training run. `results/design.json` holds the same numbers in machine-readable
form and `results/freeze.json` binds this file, every tool that runs, the
dataset membership and the runtime. A measured stage refuses to start if any
of them differs. Changing anything after the freeze requires a new documented
fork, not an edit.

Starting revision: `7f5182daa10156e04a8b0f7727d9bec9cad39cf1` (Phase 4D.4H-A).
No file under `games/` is changed, so the enforced execution-source identity
stays `3523ee04a4b523a915941379ed66871687e6163baa8c363fedddbce2c84da934` and all
three architectures run from this one revision.

The Phase 4D.4G run has finished executing (`COMPLETE_PENDING_FINAL_ANALYSIS`)
but its final analysis has not been published. Nothing here depends on it; it
bears only on the self-play schedule a later Phase 4D.4H-C would use.

## Question

Do ResNet 3x64 (322,440 parameters) and ResNet 5x64 (470,408) cost and learn
well enough, compared with the existing CNN (325,896), to justify a controlled
self-play experiment? The architectures are those of Phase 4D.4H-A, unchanged.

## What was already seen when this was written

Disclosed so the reader can judge what was and was not prospective.

- **Replay structure**, with no model involved: family sizes at each linking
  ply (`results/dataset-audit.json`). The split rule below was chosen after
  seeing that table.
- **Cost pilot** (`results/pilot.json`, 35 s): batch-128 update time 45 / 65 /
  102 ms and pool-eight search time 0.146 / 0.215 / 0.318 ms per neural
  position for CNN / ResNet 3x64 / ResNet 5x64, on synthetic positions. The
  update budgets and the throughput classes below were written knowing these.
- **Pipeline smoke test** (`results/smoke.json`, 139 s): 60 / 40 / 20 updates
  on seed 42's frozen split through every stage and the analysis, to catch
  programming errors. Its loss values were printed and seen. They cover less
  than 1% of any budget and were not used to set a budget, endpoint or
  threshold. One analysis defect (a dictionary lookup in the verdict rule) was
  found and fixed before the freeze.

No full-budget training curve, benchmark stage or held-out comparison exists
at the time of the freeze.

## Data

Read-only generation 13-20 replay from the official campaign's generation-20
resume boundaries, seeds 42 and 314159, obtained only through the pinned
historical checkout's strict reader (`experimental_import.read_historical`).
The rebuilt replay must reproduce the boundary's replay digest. 2,048 complete
games per seed: 68,856 and 64,514 positions. Each seed is trained and evaluated
only on its own replay. Sealed evaluation packages are never opened.

**Split by complete games, in whole opening families.** Two games belong to one
family when they share any mirror-canonical pre-move position at ply 8 or
later. Ply 8 is the first greedy ply of the historical action schedule.

A fully family-disjoint split with small families is infeasible. Every game
shares the empty board, and even at ply 8 one family holds 827 of 2,048 games
(seed 42) and 712 (seed 314159); at ply 16 the largest still holds 122 and 52.
The declared alternative:

1. Families larger than 5% of a seed's games are never held out (two per seed,
   47% and 42% of games). Holding one out would remove the main line from
   training; splitting one would leak.
2. The remaining families are ordered by `sha256("phase4d4h-b-heldout-family-order:seed:first game id")`
   and added whole until at least 15% of games are held out.
3. The **primary held-out set is the novel positions**: held-out positions
   whose mirror-canonical key does not occur anywhere in training. By
   construction every held-out position at ply 8 or later is novel; the
   non-novel ones are shared opening positions before ply 8. Metrics on all
   held-out positions and on the seen ones are reported beside it.

| Seed | Train games / positions | Held-out games / positions | Novel held-out | Held-out families |
|---|---|---|---|---|
| 42 | 1,739 / 59,094 | 309 / 9,762 | 7,616 | 180 |
| 314159 | 1,730 / 55,248 | 318 / 9,266 | 7,018 | 139 |

Limitations, accepted in advance: the held-out games come only from smaller
families, so they under-represent the most-played lines and are shorter on
average; 65-73% of held-out games still share a six-move prefix with a training
game; the training set repeats positions heavily (about 25,000 distinct
positions among 55,000-59,000). Membership and content hashes are in
`results/dataset-freeze.json`; the audit, including overlap with the
development suites (2 and 4 of solved300, 0 of tactical200, in training), is in
`results/dataset-audit.json`.

**The targets are biased towards the CNN.** Policy targets are root visit
counts of CNN-guided search and value targets are outcomes of CNN self-play.
Held-out loss on these targets measures how well a network imitates that
teacher on this distribution. It is not evidence of playing strength, and a
ResNet that learns something different from the CNN teacher is penalized.

## Training protocol

Identical for all three architectures within a seed:

- From scratch: `model_factory(identity).construct(seed)`, each architecture's
  own deterministic, domain-separated initialization.
- The unmodified factory trainer: policy cross-entropy on seven raw logits
  against normalized visits plus value MSE against the game outcome, equal
  weight; AdamW, learning rate 3e-4, betas (0.9, 0.999), eps 1e-8, weight decay
  1e-4 on convolution and linear weights only, gradient-norm clip 5.
- Minibatch 128, uniform over training positions without replacement within a
  batch, from a stream seeded only by `phase4d4h-b-fixed-data-sampling:seed`.
  Reflection with probability 0.5 from the standard augmenter stream. Both
  streams ignore the architecture, so update *t* uses the same 128 positions and
  the same reflections for every architecture; the analysis verifies this from
  a running hash and refuses to proceed otherwise.
- Same training positions, same held-out positions, same encoding.

**Fixed update budgets**, set from the pilot so that each architecture gets
about 305 s of optimizer time, and not adapted afterwards:

| | CNN | ResNet 3x64 | ResNet 5x64 |
|---|---|---|---|
| Updates | 6,800 | 4,700 | 3,000 |
| Pilot seconds per update | 0.0451 | 0.0647 | 0.1019 |
| Nominal optimizer seconds | 307 | 304 | 306 |

A first draft used about 360 s (8,000 / 5,500 / 3,500 updates). Its projection
came to 60.3 minutes against the 60-minute target, so the budgets were reduced
before the freeze and before any training result existed.

Evaluation milestones (updates): 0, 50, 100, 200, 400, 800, 1200, 1600, 2000,
2500, 3000, 3500, 4000, 4700, 5400, 6100, 6800, up to each architecture's
budget (the ResNet 5x64 stops at 3,000 and the ResNet 3x64 at 4,700). Training order
is CNN, ResNet 3x64, ResNet 5x64 for seed 42 and the reverse for seed 314159.
The clock for "optimizer seconds" covers sampling, augmentation and the trainer
step; evaluation time is recorded separately and excluded from it.

## Comparisons

Every contrast is ResNet minus CNN on identical positions.

1. **Equal updates.** All three at 3,000 updates (primary), and at 400, 800 and
   1,600. Equal updates are not equal computation and are not presented as such.
2. **Equal computation.** The final models, each at its own budget (primary).
   Measured optimizer seconds are reported, so the reader can see how equal the
   budgets turned out. Also, by linear interpolation in measured optimizer
   seconds between adjacent milestones: at 60, 120 and 240 s, and at the largest
   time all three reached within a seed.

Full curves against updates and against optimizer seconds are published.

## Evaluation procedures

- **Held-out replay** (novel positions; also all and seen): policy
  cross-entropy and value MSE exactly as in the training loss; KL to the visit
  target; agreement of the legal-masked argmax with the most-visited move; the
  same by ply range.
- **Training probe**: 4,096 fixed training positions scored in eval mode. The
  generalization gap is novel held-out minus probe.
- **Development solved300** (exact labels): raw-policy optimal-action accuracy
  (argmax of the legal-masked policy, lowest column on an exact tie, ties
  counted), optimal-set cross-entropy, value MSE against the exact value, and
  the rate of choosing an action worse than the position's value. Accuracy is
  also given excluding rows found in training.
- **Development tactical200**: raw-policy accuracy overall and for unique-win
  and unique-safe rows, and the avoidable-error rate. No search anywhere.
- **Gradient stability**: unclipped gradient norm (mean, median, 95th
  percentile, maximum), clipped share and loss spread per milestone interval.
  Any nonfinite loss, gradient, parameter or statistic aborts the run.
- **Batch norm**: running-statistic summaries per layer; distance of the
  running statistics from statistics accumulated over the training probe; the
  probe scored with batch statistics (train mode, on a disposable copy, 128 per
  batch) against eval mode: loss gap, mean absolute value difference, policy KL
  and choice disagreement.
- **Calibration**: ten equal-width bins. Policy confidence against agreement
  with the most-visited move (held-out) and against optimal-set accuracy
  (solved300); predicted value against mean outcome (held-out) and exact value
  (solved300).

## Uncertainty

10,000 bootstrap replicates, seed 4404801, two-sided 95% percentile intervals.
Held-out contrasts resample whole opening families and recompute
position-weighted means, paired by position. Suite contrasts resample the
development clusters of Phase 4D.4G (mirror families unioned with
reflection-equivalent prefixes through ply 8), paired by row. Pooled results
give the two seeds equal weight: held-out families are resampled independently
within each seed; suite rows are averaged across seeds before clusters are
resampled. Per-seed results are always shown. Intervals are conditional on
these six trained networks and this data; with one initialization per
architecture and seed they do not cover run-to-run training variation. No
multiplicity correction is applied.

**Verdict per metric.** BETTER: both seeds favor the ResNet and the pooled
interval excludes zero in its favor. WORSE: both seeds favor the CNN and the
pooled interval excludes zero against the ResNet. Otherwise INCONCLUSIVE.
The primary learning endpoint is the novel held-out combined loss (policy
cross-entropy plus value MSE, the training objective).

## Cost benchmarks

Apple M4, one thread, deterministic algorithms, the five library thread
variables set to 1, the same interpreter and libraries as the campaign. Weights
are seed-determined initializations.

- **Inference.** Batches 1, 8, 16 and 32 on seeded random-play positions. Both
  the bare forward pass and the complete `predict_batch` path. 20 warm-up calls,
  then 400 / 250 / 160 / 100 timed calls; seven repetitions with the
  architecture order rotated. Peak resident memory from one fresh process per
  architecture and batch size.
- **Collection.** Pool-eight `collect_batched`, 256 simulations per move, 32
  matched game identities per architecture (seed 20261012, generation 1, games
  0-31) in two blocks of 16 with the architecture order reversed between them,
  each block in a fresh process. Reported: complete games per second, neural
  positions per second, seconds per neural position and per ply, batch
  occupancy, total simulations, game lengths. Because untrained networks play
  games of different lengths, cost per neural position and per ply are the
  comparable quantities; games per second is also given normalized to a
  30-ply game.
- **Training.** Batch 128, 60 timed updates after 5 warm-up, two passes with the
  order reversed, fresh process each: time per `trainer.step`; a stage-by-stage
  replica (tensor build, forward, loss, backward, gradient checks and clip,
  optimizer step); the same with every batch norm replaced by the identity,
  which gives the measured cost of normalization; forward time in eval and
  train mode; peak resident memory.

Memory is acceptable if peak resident memory is at most 1.5x the CNN's and at
most 2,048 MiB in every benchmark. Throughput is classified by cost per neural
position against the CNN: at most 1.25x comparable; at most 2.0x acceptable with
a proportionally smaller equal-wall budget; above 2.0x costly, needing a
compensating equal-update advantage.

## Recommendation rule

Offline imitation of CNN-generated targets cannot settle the question, so a
small or inconclusive result does not disqualify an architecture.

- **Strong candidate**: BETTER on the primary endpoint at equal computation.
- **Candidate**: memory acceptable; not WORSE on the primary endpoint at equal
  updates, or WORSE there but not WORSE on solved300 and tactical200 accuracy,
  whose labels do not come from the CNN teacher; and not in the costly class
  unless BETTER at equal updates.
- **Not recommended now**: memory unacceptable, or WORSE at equal updates on
  the primary endpoint and on an exact-label accuracy.

The report states which rule fired and where judgment went beyond the rule.

## Budget, deadline and failure policy

Target about 60 minutes of measured work, hard maximum 75 minutes, counting the
pilot (35 s) and the smoke test (139 s). `results/design.json` records the
projection from the pilot, a 1.25x contingency, and the run ceiling
(75 min minus the time already used and a 60 s reserve). The design is frozen
only if the projection with contingency fits under the ceiling and the plain
projection plus time already used is at most 60 minutes.

One detached supervisor runs the stages in order: inference, training,
collection, learning seed 42, learning seed 314159. One absolute deadline. A
stage that fails or overruns ends the run as incomplete; there is no retry, no
second launch, and no reduction of budgets, games, repetitions or milestones.
An incomplete run is reported as incomplete and is not analyzed as a result.

## Preservation

All 150 official campaign files are compared with the Phase 4D.4F inventory
before and after every stage. The Phase 4D.4G worktree's commit, cleanliness
and output inventory are recorded at launch and compared at completion. Output
goes only to the ignored `experiment-output/phase4d4h-b` tree of this worktree.
Nothing is merged into main, promoted or deployed, and no self-play training is
launched.
