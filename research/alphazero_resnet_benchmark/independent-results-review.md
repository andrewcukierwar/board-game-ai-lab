# Phase 4D.4H-B — independent results review

Reviewer: a separate session that did not design, run or analyze the
experiment. Same host, same raw files; this is a review, not a replication.
Reviewed: the run under `experiment-output/phase4d4h-b/run/`, the tracked
results, the frozen tools at `bc59dc1`, and the report draft
`docs/phase4d4h-b-resnet-benchmark.md` as it stood at about 13:20 local on
2026-10-12 (with `REVIEW_SUMMARY_PLACEHOLDER` still in it).

## Outcome

No declared verdict changes under independent recomputation. The split has no
leakage. The run conformed to the frozen protocol. The supervisor's integrity
flag is explained by the Phase 4D.4G commit alone.

Three claims in the draft are stronger than the evidence and need rewording
(findings 1 to 3). Three figures need a range or a correction (findings 4 to
6). The rest are notes. None of them changes the recommendation.

| # | Severity | Finding |
|---|---|---|
| 1 | moderate | "ResNet 5x64 better on solved300 accuracy at equal updates" holds at the 3,000-update checkpoint only |
| 2 | moderate | For ResNet 3x64 at equal updates, 3,000 is the one favorable checkpoint for the primary loss between 2,000 and 4,700; the policy lead is gone from 3,500 |
| 3 | minor | The equal-time primary verdict for ResNet 3x64 sits on the inconclusive/worse boundary and leans against it in every neighbouring pairing |
| 4 | minor | Headline per-update cost uses a post hoc figure and the lowest of the available estimates; the batch-norm cost of the 5x64 differs 1.7x between passes |
| 5 | minor | "Self-play is 82-93% inference, where it pays 1.46x" mixes two quantities |
| 6 | minor | "As its equal arithmetic predicts" is not supported by anything measured |
| 7 | note | On seed 42 no model's held-out value error beats a constant prediction of zero |
| 8 | note | Checkpoint robustness of the combined-loss verdicts, quantified |
| 9 | note | Bootstrap units: 178 and 136 families, effective number 59 and 23; a jackknife interval agrees |
| 10 | note | Timing: no measurable disturbance of the ratios; seed 314159 is not an undisturbed control |
| 11 | note | Integrity adjudication verified; the bypass of the frozen refusal is a disclosed deviation |
| 12 | note | Memory: the learning-stage figure cannot separate architectures; the benchmark cells can |
| 13 | note | Small items in `analyze.py`, `metrics.py` and the report's requirement table |

## Scope and method

`independent_review.py` (run with
`$PY -B -m research.alphazero_resnet_benchmark.independent_review`, 52 s)
writes `results/independent-recomputed.json`. It imports neither `analyze.py`
nor `metrics.py`. It uses `common` for paths, the runtime, the source hashes
and the read-only official inventory, and `dataset.frozen` for the strict
historical read. Everything else is its own code:

- a mirror-canonical key, boards rebuilt from each game's move list, opening
  families by label propagation, and the novelty mask;
- a regeneration of the minibatch and reflection streams from their declared
  domains, hashed as `learn.py` hashes them;
- per-family sums from the per-position tensors, a paired family bootstrap
  (index resampling, 10,000 replicates, NumPy default generator seeded
  9071733), 20 further streams to measure Monte Carlo spread, a
  delete-one-family jackknife, and the verdict rule;
- development-suite clusters rebuilt from the development packages, the
  `correct` flags rechecked against the packages, and a cluster bootstrap;
- the same contrasts at every shared milestone, at neighbouring checkpoint
  pairings and averaged over three checkpoints, from the recorded per-family
  sums (which match mine to 4e-6 where per-position tensors exist);
- protocol checks, timing summaries, and a read-only re-derivation of the
  Phase 4D.4G inventory hash and commit history.

Separately, I ran the frozen `analyze.analyze()` into a scratch directory.
`benchmarks.json`, `learning.json`, `comparisons.json` and
`endpoint-detail.json` came out byte-identical to the tracked files.

Total computation was about 3.5 minutes. Nothing was trained, no stage was
rerun, no sealed package was opened.

## What was recomputed and how it compared

**Alignment.** For all 16 per-position files, `se` equals
`(value - outcome)^2` with outcomes taken from my own read of the held-out
list, to 1.2e-7 (float32 storage). That fixes the row order, so my novelty
mask and family labels apply to the right rows. Novel means match the
recorded milestone values to 1.3e-9.

**Held-out contrasts, ResNet minus CNN, novel positions.** Mine, pooled with
equal seed weight, 95% interval from my stream. Negative favors the ResNet for
losses.

| Comparison | Metric | Seed 42 | Seed 314159 | Pooled [95%] | Verdict (mine = analysis) |
|---|---|---|---|---|---|
| 3x64, 3,000 updates | CE+MSE | -0.0264 | -0.0062 | -0.0163 [-0.0776, +0.0378] | Inconclusive |
| | policy CE | -0.0064 | -0.0207 | -0.0135 [-0.0202, -0.0061] | Better |
| | value MSE | -0.0201 | +0.0145 | -0.0028 [-0.0623, +0.0499] | Inconclusive |
| | top-1 | +0.0075 | +0.0157 | +0.0116 [+0.0017, +0.0200] | Better |
| 5x64, 3,000 updates | CE+MSE | +0.0495 | +0.0980 | +0.0738 [+0.0336, +0.1140] | Worse |
| | policy CE | +0.0039 | -0.0094 | -0.0028 [-0.0117, +0.0066] | Inconclusive |
| | value MSE | +0.0456 | +0.1074 | +0.0765 [+0.0355, +0.1167] | Worse |
| | top-1 | -0.0059 | -0.0006 | -0.0032 [-0.0131, +0.0051] | Inconclusive |
| 3x64, final (4,700 v 6,800) | CE+MSE | -0.0102 | +0.1220 | +0.0559 [+0.0004, +0.1133] | Inconclusive |
| | policy CE | +0.0156 | +0.0053 | +0.0104 [+0.0020, +0.0188] | Worse |
| | value MSE | -0.0258 | +0.1168 | +0.0455 [-0.0114, +0.1052] | Inconclusive |
| | top-1 | -0.0018 | +0.0050 | +0.0016 [-0.0108, +0.0162] | Inconclusive |
| 5x64, final (3,000 v 6,800) | CE+MSE | +0.0369 | +0.1483 | +0.0926 [+0.0271, +0.1749] | Worse |
| | policy CE | +0.0291 | +0.0245 | +0.0268 [+0.0194, +0.0345] | Worse |
| | value MSE | +0.0078 | +0.1239 | +0.0658 [-0.0012, +0.1520] | Inconclusive |
| | top-1 | -0.0181 | -0.0184 | -0.0183 [-0.0284, -0.0085] | Worse |

- Point estimates differ from `comparisons.json` by at most 1.4e-9.
- Interval endpoints differ by at most 0.0042 (largest: 5x64 final). Across
  my 20 further streams an endpoint moves by at most 0.006.
- All 16 verdicts are the same in my stream and in each of the 20 further
  streams.
- The seven secondary comparisons (400, 800, 1,600 updates; 60, 120, 240 s
  and the largest common time, with my own interpolation) match in point
  estimate to 2e-15, in endpoints to 0.0048, and in all 56 verdicts.

**Development suites, per-row records at 3,000 updates and at the final
models.** My clusters equal the recorded ones (102 for solved300, 71 for
tactical200). Row order and every `correct` flag agree with the development
packages. All point differences equal the analysis exactly; interval
endpoints differ by at most 0.006; all 32 suite verdicts agree. Accuracy:

| Comparison | Suite | Seed 42 | Seed 314159 | Pooled [95%] | Verdict |
|---|---|---|---|---|---|
| 3x64, 3,000 | solved300 | -0.010 | +0.020 | +0.005 [-0.019, +0.026] | Inconclusive |
| 3x64, 3,000 | tactical200 | +0.060 | +0.050 | +0.055 [+0.016, +0.096] | Better |
| 5x64, 3,000 | solved300 | +0.043 | +0.017 | +0.030 [+0.006, +0.052] | Better |
| 5x64, 3,000 | tactical200 | +0.080 | +0.105 | +0.093 [+0.041, +0.151] | Better |
| 3x64, final | solved300 | -0.013 | -0.010 | -0.012 [-0.038, +0.019] | Inconclusive |
| 3x64, final | tactical200 | -0.010 | +0.010 | +0.000 [-0.045, +0.043] | Inconclusive |
| 5x64, final | solved300 | +0.017 | -0.023 | -0.003 [-0.028, +0.025] | Inconclusive |
| 5x64, final | tactical200 | -0.020 | +0.035 | +0.008 [-0.039, +0.056] | Inconclusive |

**Split and leakage.** Both seeds:

- Counts equal the DESIGN.md table (1,739 / 59,094 and 309 / 9,762 with
  7,616 novel; 1,730 / 55,248 and 318 / 9,266 with 7,018 novel; 180 and 139
  held-out families).
- All 133,370 stored observations equal the board rebuilt from the game's
  moves.
- My components over all 2,048 games equal the frozen families (586 and 584).
  No family and no game is in both splits.
- No held-out position at ply 8 or later shares a key with any training
  position (0 of 7,296 and 0 of 6,730), so all of them are novel.
- No novel position's key occurs in training. My mask equals the frozen mask.
- The seen held-out positions (2,146 and 2,248, all before ply 8) split into
  exact and mirror-only duplicates as the report states (2,053 / 93 and
  2,176 / 72).
- Suite overlap as reported: 2 and 4 solved300 rows in training, none in
  held-out, no tactical200 row in either.

Other routes checked and found closed:

- The 4,096 probe positions are training positions, used only for scoring.
- The sampling hash at every milestone of every run equals my regeneration,
  and the reflection counts match. So batches were drawn only from the
  training list, uniformly, 128 without replacement, identically for the
  three architectures, and evaluation consumed nothing from either stream.
- `learn.py` aborts if evaluation changes the weight hash or leaves the model
  in train mode, and requires every batch-norm counter to equal the budget.
- The trainers use no dropout, so no other random state enters training.

**Protocol conformance.** 36 checks, none failed:

- `status.json` records `bc59dc1`; that commit is HEAD, is on the pushed
  branch, and was made 34 s before the run started.
- No frozen file differs from it; `common.source_map()` equals `freeze.json`;
  every stage ran under the frozen runtime.
- Five stages in the declared order, all complete, no overlap, 44.9 minutes.
- Budgets 6,800 / 4,700 / 3,000; the declared milestones; the declared order
  per seed (also by file times); batch 128; AdamW 3e-4, betas (0.9, 0.999),
  eps 1e-8, decay 1e-4, clip 5, reflection 0.5.
- The ten saved model files match their recorded SHA-256.
- Inference: 7 repetitions of all 24 cells with 400 / 250 / 160 / 100 calls
  and the rotated order; 12 fresh-process memory cells.
- Training benchmark: two reversed passes of 60 updates.
- Collection: 2 x 16 games per architecture, reversed in the second block,
  256 simulations per move, simulation totals equal to 256 x plies.

## Findings

### 1. The 5x64's solved300 advantage at equal updates is a one-checkpoint result (moderate)

The draft says in the summary, the summary table and the rule application
that the 5x64 is "better on tactical and solved accuracy" at equal updates.
The verdict at 3,000 updates is computed correctly. The neighbouring
checkpoints do not support the solved300 half.

ResNet 5x64 minus CNN, solved300 accuracy, same update count:

| Updates | 800 | 1,200 | 1,600 | 2,000 | 2,500 | 3,000 |
|---|---|---|---|---|---|---|
| Seed 42 | +0.047 | +0.040 | -0.023 | +0.013 | +0.013 | +0.043 |
| Seed 314159 | -0.023 | -0.013 | -0.023 | -0.037 | -0.030 | +0.017 |

Both seeds favor the 5x64 at one of six checkpoints. In seed 314159 it is
below the CNN at each of the five earlier ones. Over the five checkpoints
from 1,200 to 3,000 the mean difference is +0.017 on seed 42 and -0.017 on
seed 314159 (`tables.md`, post hoc section), which pools to zero. The
interval at 3,000 is also near zero at its lower end (+0.006), in a table of
104 uncorrected verdicts.

The tactical200 advantage is different: both ResNets are ahead on both seeds
at every shared checkpoint from 800 updates on (5x64: +0.035 to +0.105).

**The report must** drop "and solved" from the summary bullet, the summary
table and the rule paragraph, or qualify it as holding at the 3,000-update
checkpoint only and not at the five before it. The rule outcome for the 5x64
is unchanged, because it turns on not being worse on an exact-label accuracy.

### 2. For the 3x64 at equal updates, 3,000 is the favorable checkpoint (moderate)

The equal-update table for the 3x64 stops at 3,000, and the text gives the
primary-loss difference there as -0.016. The run also holds 3,500, 4,000 and
4,700 for both networks. The same contrast and rule at each shared
checkpoint:

| Updates | CE+MSE seed 42 / 314159 | Pooled [95%] | Verdict | Policy CE pooled [95%] | Verdict |
|---|---|---|---|---|---|
| 1,200 | +0.023 / -0.041 | -0.009 [-0.057, +0.040] | Inconclusive | -0.030 [-0.039, -0.019] | Better |
| 1,600 | +0.021 / -0.001 | +0.010 [-0.027, +0.045] | Inconclusive | -0.019 [-0.026, -0.012] | Better |
| 2,000 | +0.034 / +0.052 | +0.043 [-0.005, +0.087] | Inconclusive | -0.019 [-0.028, -0.009] | Better |
| 2,500 | +0.034 / +0.034 | +0.034 [-0.014, +0.079] | Inconclusive | -0.006 [-0.014, +0.002] | Inconclusive |
| 3,000 | -0.026 / -0.006 | -0.016 [-0.078, +0.038] | Inconclusive | -0.014 [-0.020, -0.006] | Better |
| 3,500 | +0.014 / +0.041 | +0.028 [-0.024, +0.075] | Inconclusive | -0.006 [-0.014, +0.003] | Inconclusive |
| 4,000 | +0.020 / +0.013 | +0.016 [-0.052, +0.072] | Inconclusive | +0.004 [-0.005, +0.013] | Inconclusive |
| 4,700 | +0.030 / +0.069 | +0.050 [+0.004, +0.095] | Worse | -0.001 [-0.011, +0.009] | Inconclusive |

- From 2,000 to 4,700 the combined loss favors the CNN on both seeds at five
  of six checkpoints. The declared one is the exception.
- At 4,700 the declared rule gives Worse, marginally and through value error.
- Averaged over 2,000, 2,500 and 3,000 the difference is +0.020
  [-0.025, +0.061]: inconclusive, with the sign against the 3x64.
- The per-update policy lead holds through 2,000 and at 3,000 (averaged over
  2,000 to 3,000: -0.013 [-0.020, -0.005]). From 3,500 it is indistinguishable
  from zero.

**The report must** show the 3,500 to 4,700 rows, say that the -0.016 at
3,000 is not typical of its neighbours, and limit "learns the policy faster
per update" to the first 3,000 updates. The 3x64 remains a candidate under
the rule either way: if it were worse on the primary loss at equal updates,
the second branch applies, and it is not worse on either exact-label
accuracy.

### 3. The 3x64's equal-time primary verdict is on the boundary (minor)

The draft prints the pooled interval as [-0.000, +0.112].

- The lower end is -0.0002 in the analysis stream and +0.0004 in mine. It is
  above zero in 13 of my 20 further streams (-0.0007 to +0.0016). A jackknife
  interval gives -0.003 (normal) or -0.007 (t, 22 degrees of freedom).
- The verdict is Inconclusive in every stream only because seed 42's point
  estimate is -0.010. That sign changes when one family is removed
  (leave-one-family-out range -0.018 to +0.003).
- All nine pairings of the 3x64 at 3,500 / 4,000 / 4,700 with the CNN at
  5,400 / 6,100 / 6,800 give a positive pooled difference (+0.004 to +0.056).
  One of them is Worse (4,000 against 6,800). Averaged over the three
  checkpoints each, the difference is +0.031 [-0.009, +0.069].
- Policy loss is Worse in seven of the nine pairings and in the average
  (+0.014 [+0.007, +0.021]).

**The report should** print that interval with four decimals, say its lower
end is zero within Monte Carlo error, and describe the result as inconclusive
with every point estimate against the 3x64. "Level at best" in the
recommendation is consistent with this. The equal-time policy-loss verdict
(Worse, about 0.01 nats) is robust.

### 4. Per-update cost: quote the declared statistic and a range (minor)

The summary gives 1.44x and 2.28x. Those are the post hoc ratios of the
learning runs' optimizer clock, which includes about 0.6 ms per update of
sampling and augmentation that is the same for every architecture. The
available estimates:

| Estimate | ResNet 3x64 | ResNet 5x64 |
|---|---|---|
| Declared benchmark, mean of two pass means | 1.448x | 2.291x |
| Benchmark, pass medians, pass 1 / pass 2 | 1.485x / 1.441x | 2.384x / 2.271x |
| Learning runs, `trainer.step` only, seed 42 / 314159 | 1.434x / 1.450x | 2.279x / 2.309x |
| Learning runs, optimizer clock, seed 42 / 314159 | 1.429x / 1.444x | 2.262x / 2.292x |

The benchmark's first pass was slow for the ResNets only: medians 3.7% and
5.6% above the second pass, against 0.6% for the CNN. The learning runs agree
with the second pass and with the pilot.

The same pass difference affects the batch-norm cost. For the 3x64 it is 6.4
ms in both passes. For the 5x64 it is 16.1 ms in pass 1 and 9.7 ms in pass 2;
the reported 12.9 ms (12.1%) is the mean of two discordant values.

**The report should** give "about 1.45x (1.43 to 1.49)" and "about 2.3x (2.26
to 2.38)", name which statistic is the declared one, and give the 5x64's
batch-norm cost as 10 to 16 ms.

### 5. "82-93% inference, where it pays 1.46x" (minor)

1.46x is the cost per neural position of the whole search, tree work
included. On the inference part alone the collection stage measured 0.127,
0.203 and 0.310 ms per neural position: 1.60x and 2.44x. That is close to the
`predict_batch` ratios at batch 8 (1.62x and 2.50x).

The whole-search ratios also come from blocks whose mean batch occupancy
differed (5.50 for the CNN, 6.61 and 6.27 for the ResNets), which favors the
ResNets slightly. The pilot at full batches gave 1.48x and 2.18x. The
block-to-block ratios are stable (1.477 / 1.446 and 2.151 / 2.144), so the
classes hold: the 3x64 is under 2.0x and the 5x64 is above it in both blocks
and in the pilot.

**The report should** correct the sentence. The Phase 4D.4H-C budget should
be set from a ratio measured with trained networks at generation-sized pools.
The 1.46x here comes from untrained networks playing 16-game blocks.

### 6. "As its equal arithmetic predicts" (minor)

The draft explains the parity of the 3x64 and the CNN at batches 16 and 32 by
equal arithmetic. No operation count was measured or reported, and the
protocol says nothing is derived from such counts. The same table shows the
CNN's cost per position rising from 0.117 ms at batch 8 to 0.300 ms at batch
16 and 0.240 ms at batch 32, a 2.0x to 2.6x loss that the draft says was not
investigated. The 3x64 loses less there (0.191 to 0.297 and 0.243 ms, 1.3x
to 1.6x). Parity appears because the CNN gets slower, not because the 3x64
catches up.

**The report should** remove the clause, and keep the draft conclusion about
parity next to the fact that batch 8 is the cheapest operating point per
position for all three.

### 7. Seed 42: no model's held-out value error beats a constant (note)

A prediction of zero scores 0.7816 on seed 42's novel positions and 0.7957 on
seed 314159's.

| Novel value MSE | CNN 3,000 | CNN 6,800 | 3x64 3,000 | 3x64 4,700 | 5x64 3,000 | Zero |
|---|---|---|---|---|---|---|
| Seed 42 | 0.7789 | 0.8167 | 0.7588 | 0.7909 | 0.8245 | 0.7816 |
| Seed 314159 | 0.5507 | 0.5343 | 0.5652 | 0.6510 | 0.6581 | 0.7957 |

On seed 42 the best model is 0.023 below the constant and three of five are
above it. Seed 42's share of every value and combined-loss contrast therefore
compares degrees of overconfidence on single-game outcomes. This agrees with
the draft's calibration table (21 to 24% of ResNet values beyond 0.95 at
3,000 updates, against 9 to 12% for the CNN). The draft already says the
value head overfits and that the combined-loss verdicts are weak. Adding the
constant baseline would make the point exact: the 5x64's Worse verdict on the
primary loss is a statement about value overconfidence, and only seed 314159
shows any value generalization to lose.

### 8. Checkpoint robustness of the combined-loss verdicts (note)

This quantifies the draft's "single checkpoints" limitation.

- Movement of novel value MSE between adjacent checkpoints from 1,200 on:
  mean 0.017 to 0.034 per run for five runs and 0.107 for the 5x64 on seed
  314159; largest single step 0.165. Policy CE: mean 0.009 to 0.035.
- **5x64, equal updates.** Inconclusive at 800, 1,200 and 1,600. Worse at
  2,000 (+0.048), 2,500 (+0.120) and 3,000 (+0.074). Averaged over the last
  three: +0.080 [+0.041, +0.117]. The verdict is stable from 2,000; the
  magnitude varies 2.5x.
- **5x64, equal time.** Worse in all nine pairings of its 2,000 / 2,500 /
  3,000 checkpoints with the CNN's 5,400 / 6,100 / 6,800 (+0.068 to +0.173).
- **3x64.** Findings 2 and 3: Inconclusive at eight of the nine shared
  checkpoints from 800 updates on, and in eight of nine equal-time pairings.

So the Worse and Inconclusive categories do not depend on which late
checkpoint is taken. The size of a difference does, by a factor of two or
more, and the sign of the 3x64's equal-update difference does.

### 9. Bootstrap units (note)

- DESIGN.md lists 180 and 139 held-out families. The bootstrap resamples 178
  and 136. The other 2 and 3 are single seven-move games, which have no
  position at ply 8 or later and no novel position. This is immaterial.
- Families are unequal. In seed 314159 four families hold 966, 587, 538 and
  537 of 7,018 novel positions, and the five largest hold 40%. The effective
  number of families is 59 for seed 42 and 23 for seed 314159.
- With that few effective units a percentile interval can be too narrow. I
  checked with a delete-one-family jackknife standard error and a t
  multiplier for 22 degrees of freedom. All 16 primary held-out verdicts are
  unchanged; interval ends move by at most 0.022.
- 56 and 28 novel positions share a key with another held-out family. That
  is too few to matter.

### 10. Timing under the concurrent session (note)

The files in the Phase 4D.4G commit were written between 12:11:47 and
12:35:14, so the draft's window is right.

- CPU time over wall time is 0.986 to 0.998 for every stage and at least
  0.9985 for every collection block. The processes were not kept off a core.
- Inference: 8 of 168 repetition cells are more than 5% above their cell
  median; the worst is +29%. The median over seven repetitions absorbs them.
  Ratios from per-call minima agree with the reported medians within 1.5%.
- Training benchmark: finding 4.
- Collection: block ratios within 2% (finding 5).
- Learning: `trainer.step` cost in seed 42, which ran inside the window, is
  +0.7%, -0.3% and -0.6% against seed 314159, which ran after it.
- The largest excursion inside any learning run is in seed 314159: the 3x64
  ran 11% slow between updates 100 and 200. The machine was not idle after
  12:36 either, so seed 314159 is not an undisturbed control. The agreement
  of the two seeds is still good evidence that neither was materially
  affected.
- The learning contrasts are functions of update counts and cannot be
  affected by timing. Only the interpolated equal-seconds comparisons use the
  clock, and the measured optimizer times are equal within 1.3% per seed.

The cost ratios are supported to about ±3%. The draft's statement that the
timings show little disturbance is correct.

### 11. The integrity flag (note)

Verified read-only:

- The Phase 4D.4G worktree is at `01526b8`, clean. That commit is the only
  one after `b4b1d9f` and descends from it, at 12:36:27.
- It changes 138 paths (137 added, 1 modified) under `docs/`, `research/`
  and `tests/`; none under `games/` or `experiment-output/`, which is ignored
  and has no tracked file.
- My own hash over the 151 files of `experiment-output/phase4d4g-primary`
  is `7db4f97f…5f4aa2`, equal to the supervisor's before and after values.
  The newest of those files was written at 11:32:03, before this run started.
- The only field that differs between the supervisor's two records is the
  commit. The 150 official files are identical now.

The adjudication's claim holds. Two things remain as stated in the draft and
should stay visible: the declared check included the commit, so by the frozen
protocol the final status is `INTEGRITY_FAILURE`; and the frozen `analyze`
entry point refuses that status, so the analysis ran through `adjudicate.py`.
The analysis code itself was unmodified (my rerun reproduces the four result
files byte for byte).

### 12. Memory (note)

"Equivalent" is supported by the fresh-process cells: inference 255 to 275
MiB (largest ratio to the CNN 1.046, at batch 1), collection 298 to 300 MiB,
training benchmark 279 to 324 MiB against the CNN's 315. Two cautions:

- The learning-stage peak (403.4 and 392.5 MiB) is one process's high-water
  mark and is identical for the three architectures within a seed. It says
  nothing about any one architecture.
- The 5x64's training-benchmark peak was 324 MiB in one pass and 279 MiB in
  the other for the same work. Differences below about 45 MiB are noise.

### 13. Smaller items (note)

- `analyze.LOWER_IS_BETTER` has no entry for `avoidable_rate`; the default
  happens to be correct.
- The suite bootstrap seeds its stream with the cluster count and the length
  of the metric's name. No two metrics collide within a suite. Harmless.
- Interpolated equal-seconds comparisons bootstrap a blend of two checkpoints
  and exclude checkpoint variation. The draft says so.
- `metrics.py` scores ties in the visit target as the lowest column: 17 novel
  positions per seed, the same for every architecture. Suite ties are 0 at
  every milestone. Policy loss uses unmasked logits for all three, as in
  training. I found nothing that scores one architecture unfairly.
- ResNets are scored in eval mode. On the probe the eval-minus-train value
  error is at most +0.012 at the primary checkpoints, small against the
  contrasts. It is +0.087 for the 3x64 on seed 42 at 800 updates, where that
  network's held-out value error spikes to 0.889. The 800-update row for the
  3x64 is partly a running-statistics transient.
- The gradient clip binds for the ResNets early: in the first 400 updates up
  to 35% of an interval's updates are clipped for the 5x64 and up to 12% for
  the 3x64, against at most 1% for the CNN. The draft's limitation on shared
  hyperparameters covers it.
- The CNN's final model has passed over the training set 14.7 and 15.8 times,
  the 3x64 10.2 and 10.9, the 5x64 6.5 and 7.0. Equal time is unequal
  exposure by design; worth one line beside the equal-time table.
- The requirement table already marks "Independent results review" as PASS.
  It should be filled in from this document.
- README.md says SHA-256s of the per-position and series files are in the
  tracked results. The run's own records hold hashes for the model files
  only. I validated the per-position files by content instead (above).

## The draft conclusions, one by one

| Draft conclusion | Result of this review |
|---|---|
| 3x64 costs about 1.44x per update and 1.46x per search position; 5x64 about 2.28x and 2.15x | Supported to about ±3%. Per update is better stated as 1.45x and 2.3x (finding 4) |
| At batch 16 and 32 the 3x64 costs the same as the CNN; pool-eight runs at mean batch 5.5 to 6.6 where it does not | Measured as stated. Parity is in a regime where the CNN is 2.0x to 2.6x slower per position than at batch 8 (finding 6) |
| Equal updates: 3x64 inconclusive on primary, better on policy CE and tactical200 | Verdicts reproduce. 3,000 is the favorable checkpoint for the primary loss; the policy lead ends by 3,500; the tactical lead is robust (finding 2) |
| Equal updates: 5x64 worse on primary, better on tactical200 and solved300 | Primary and tactical reproduce and are robust from 2,000 updates. solved300 is not (finding 1) |
| Both ResNets better at 400 updates | Reproduces; also Better at 50, 100 and 200. The 5x64's interval at 400 ends at -0.004 |
| Equal time: 3x64 inconclusive on primary, slightly worse on policy CE | Reproduces. The primary verdict is on the boundary and leans against the 3x64 (finding 3) |
| Equal time: 5x64 worse | Reproduces in all nine neighbouring pairings |
| No tactical or solved accuracy difference at equal time | Reproduces (all eight accuracy verdicts Inconclusive) |
| Memory equivalent | Supported by the benchmark cells (finding 12) |
| Value MSE too noisy for the combined endpoint; policy CE and exact-label accuracy are informative | Supported, and stronger on seed 42 than stated (finding 7). Exact-label accuracy is informative for tactical200; solved300 moves by ±0.03 between checkpoints, as much as its contrasts |

## Not checked

- What the other session computed or how much CPU it used. I verified only
  when its files were written and that this run's processes kept their core.
- The architecture review, `validation.json` (79 checks) and the test suite.
- The figures, and the tables in `tables.md` and the report beyond the
  numbers cited here. In particular the batch-norm, calibration and gradient
  tables, the forward / backward split of the update, and the Phase 4D.4G
  reference accuracies quoted "for scale".
- Per-position tensors exist only at 0, 3,000 and the final update. At other
  milestones I relied on the recorded per-family sums. Those match mine to
  4e-6 where both exist, but are otherwise the run's own numbers. Suite
  intervals away from 3,000 and the final models cannot be computed at all.
- That 20 warm-up calls preceded each timed inference cell: the stage JSON
  does not record them. I rely on the frozen source.
- The logits themselves. Top-1, confidence and policy loss per position were
  taken from the tensors; only the value error could be re-derived from a
  stored model output.
- Run-to-run variation. One initialization per architecture and seed; nothing
  here can measure it.
- The Phase 4D.4H-C proposal, beyond the cost ratio it starts from.
