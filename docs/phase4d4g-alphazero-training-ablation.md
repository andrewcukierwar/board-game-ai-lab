# Phase 4D.4G — controlled AlphaZero exploration training ablation

## Executive summary

**Extending exploration from eight to sixteen plies did not demonstrate a better
trained AlphaZero agent in this pilot.** It reduced self-play repetition in all
four seed/generation comparisons, but failed the prospective neural-learning and
tactical gates. Playing-strength noninferiority is inconclusive overall. This is
not proof that longer exploration generally harms learning or strength.

The controlled experiment completed correctly: both seeds, both arms, generations
21/22, 512 complete self-play games, 512 AdamW updates, all 6,280 position searches
and 256 strength matches. No retries, abandoned tasks, additional generations,
model promotion or deployment occurred. All 150 official campaign files remain
byte-identical. Primary wall time was 2,016.159 seconds (33.60 minutes).

| Independent conclusion | Decision | Interpretation |
|---|---|---|
| Execution | PASS | Full budgets, strict lineages/evidence and preservation verified |
| Neural learning | FAIL | Practical improvement criteria not met; seed 314159 value-MSE regression versus control exceeds margin |
| Tactical reliability | FAIL | Four prospective 2 pp point gates fail; two additional deep gates remain inconclusive |
| Playing strength | INCONCLUSIVE | Seed314159 noninferiority passes; seed 42 cannot exclude the declared 5 pp regression |

Keep eight exploratory plies as the working schedule. Prioritize Phase 4D.4H-B:
a controlled CNN/ResNet cost, throughput and fixed-data learning comparison with
a leakage-aware held-out split. Do not adopt sixteen plies or automatically extend
this training pilot. These are four separate conclusions, without a composite
score or selected champion.

## Design, source freeze and prior research

Dedicated branch `phase4d4g-alphazero-training-ablation`, isolated worktree
`experiment-output/phase4d4g-worktree`, started at
`9325f0a19e6b0106ea1dcd265316b57323f8eb4f`. The pushed prospective design/execution
commit was **b4b1d9f5a42db0889fb2eed2acea6252da8526c9**, recorded before collection.
The complete [DESIGN.md](../research/alphazero_training_ablation/DESIGN.md) remains unchanged. Preparation, detached
execution and this final-analysis session were separate.

Four strict generation 20 forks: seeds 42/314159, control and treatment. Both arms
within seed had identical model, ordered AdamW state, replay, counters and persistent
sampling/reflection/process RNGs. Control uses τ=1 at zero-based plies 0–7 then 0;
treatment uses τ=1 at 0–15 then 0. Each new generation collects 64 complete games at 256
simulations/move, pool 8, then 64 fixed updates, batch 128. The 325,896-parameter
three-layer CNN, root noise epsilon 0.25/alpha 1, LR 3e-4, AdamW configuration, clipping 5,
reflection 0.5, losses, PUCT and eight-generation replay window remain matched.
Positions, length and time are measured, not artificially equalized.

[4A](phase4d4a-alphazero-baseline.md) identified weak raw tactics and training
bottlenecks; [4B](phase4d4b-alphazero-bitboard-puct.md) and
[4C](phase4d4c-alphazero-batched-inference.md) validated bitboard/pool-eight
optimizations. [4D](phase4d4d-alphazero-exploration-diversity.md) and
[4E](phase4d4e-alphazero-exploration-confirmation.md) replicated reduced repetition
while leaving safety and learning unresolved. [4F](phase4d4f-alphazero-training-infrastructure.md)
provided strict experimental forks and complete-boundary recovery. This phase
tests actual learning rather than treating diversity as strength.

Frozen source/runtime are in [freeze.json](../research/alphazero_training_ablation/results/freeze.json), and parent
conversion/state equality in [readiness.json](../research/alphazero_training_ablation/results/readiness.json).
All 44 enforced files still match, including the original source digest
`b8aae0d6e14292dad2d2bcdb86f6048fad93a5eeaa5d342d7b5c340e9fc2bde5`.
Fixture SHA-256 is `80f6851838b18172d6f818ebd306ce0cd36e4796cac1e3a71df29694feb3ca2c`. Runtime: Apple M4 CPU,
CPython 3.11.17, Torch 2.10.0, NumPy 1.26.4, deterministic algorithms and all intra/inter-op
and five library thread controls fixed to one. Final analysis adds only read-only
tools in the sibling `research/alphazero_training_ablation_postexecution/`, outside
the immutable execution closure; no frozen source/configuration/fixture/rule changes.

## Completion, integrity and provenance

The manager finished `COMPLETE_PENDING_FINAL_ANALYSIS` normally; all **22** scheduled
tasks completed (8 training, 10 position-evaluation, 4 match tasks). A read-only
[process check](../research/alphazero_training_ablation/results/process-verification.json) found no remaining supervisor
or worker. Exact task order/directories/success markers and empty failure logs
show no unreported retry. No temporary boundary remains. The final
[integrity audit](../research/alphazero_training_ablation/results/integrity-audit.json) independently reloads all 12
experimental generation 20/21/22 files with trusted SHA and strict source/runtime
validation, verifies full deterministic state chains, normalized visits/actor
outcomes, legal complete histories, optimizer persistence, counters and replay
expiration. Exported games and recorded per-game RNG seeds agree.

Execution order (C=control, T=treatment):42C21,42T21,314159T21,314159C21,42T22,42C22,314159C22,314159T22,
with complete development evaluation after every new boundary. Frozen collection
checks verify immutable learner/snapshot weights before replay/updates; no update
occurs during collection. Generations 21/22 evict historical 13/14 respectively,
retaining 14–21/15–22 with mixed schedule metadata. Atomic publication follows the
reviewed fsync/strict-validation/no-replace-hardlink/directory-fsync path and all
successful reloads; no independent syscall trace is claimed.

Evaluation checkpoint assignment is reconstructed from the frozen manager's
current-boundary transitions and strict evaluation loader. Raw rows lack their
own model-hash attestation; they were not re-evaluated to add one. File/evidence
hashes detect changes, not malicious jointly rewritten artifacts.

Historical parent boundary SHA-256:42=`2866722b5e55726b75ae27d4bdb70e98d9b01b9ca503b0925fe9aa8b17d210b1`;
314159=`328022e5f4c45c11650b7fd2b76fbafbbe4734f1161a75d91b5d0b4cb274f9d6`.
Matched initial-state SHA:42=`2dd15035fd4e10654a80a78cf5dec16425d4ad6e23f21323adcc219ca8bee5e4`;
314159=`5aa9916bc689710aa8ed4e6a5d268cb94d913cb8375e5d14542280587be29c77`.
These are new experimental forks. They do not reproduce the historical official
campaign's sequential stochastic trajectories.

### Local checkpoint recovery identities

All paths below are relative to this worktree’s ignored `experiment-output/phase4d4g-primary/`.

| Seed | Arm | Gen | Relative checkpoint path | File SHA-256 |
|---|---|---|---|---|
| 42 | control | 20 | `seed-42/control/fork-0020.pt` | `33411a7b81e33e24f15fb72e81a78432ae2451b1fe8120d67d13b6d61e6883e3` |
| 42 | control | 21 | `execution/42-control-g21/generation-0021.pt` | `952b5bb4a1cc5b0767e8a5c875e7169511a899f4012d3da87753b1204f556106` |
| 42 | control | 22 | `execution/42-control-g22/generation-0022.pt` | `dd42b1527b29de89620411d7bb8189b811f55ef28e301a83f7072c3b4b4cc5df` |
| 42 | treatment | 20 | `seed-42/treatment/fork-0020.pt` | `c5b676c1f67269004eb819f86652d85c07a33d57daa39339d8212134b55b4239` |
| 42 | treatment | 21 | `execution/42-treatment-g21/generation-0021.pt` | `09e54ce1770a1ba4384320cf18be829de53b2831e62ae40f822bfce4d3b7eb5f` |
| 42 | treatment | 22 | `execution/42-treatment-g22/generation-0022.pt` | `59214468b2d08c9590af8a21aebdebc1ad76f8806d1b8a15563893218ea08206` |
| 314159 | control | 20 | `seed-314159/control/fork-0020.pt` | `b321b415c7065c0080f3d866f650e566e1f2abf8f82a9fbab371e4b2c07a31c3` |
| 314159 | control | 21 | `execution/314159-control-g21/generation-0021.pt` | `3dc87ec24cb90260999b6818a4b6e6c1f28b5dc19fe47206dda3b5cd87d99e65` |
| 314159 | control | 22 | `execution/314159-control-g22/generation-0022.pt` | `0e8cb2c7395519615ac437331d1876127bc111a0f2a4da3f6e974e3975b50e70` |
| 314159 | treatment | 20 | `seed-314159/treatment/fork-0020.pt` | `e5a814915d813523f25cd63ca55a28ba92f985ce26e4749f24f14e59a823fdd1` |
| 314159 | treatment | 21 | `execution/314159-treatment-g21/generation-0021.pt` | `41fc326b6eeadf8556824afb5c5667efb946cc1fecb5dbd283c843d4bcb3c15b` |
| 314159 | treatment | 22 | `execution/314159-treatment-g22/generation-0022.pt` | `2ced466df262427574e2045b29ef3754add4a263a5ea359c9d80543e5eaf0a8f` |

Every model, optimizer, replay, RNG, counter, archive-integrity and deterministic computational-state hash is recorded in [integrity-audit.json](../research/alphazero_training_ablation/results/integrity-audit.json). Git reporting is excluded from computational equality; enforced content/runtime are not. Binary checkpoints remain local and unchanged.

## Frozen evaluation and uncertainty

The [fixtures](../research/alphazero_training_ablation/fixtures.json) contain nonsealed solved300, tactical200,
deep128 (64 complete-vector 4E families plus mirrors), and 32 opening base families.
solved/deep rows have 100% complete legal-action WDL coverage. Tactical rows have
independent immediate-win/safe labels, not complete deeper values; no arbitrary
tactical action receives an invented exact label. No new solving was performed.

Raw policy and raw PUCT 512 (noise/guards/proof propagation off) use fixed independent
evaluation identities, matched across checkpoints/arms. Each seed's identical
parent model was evaluated once and reused for its two arms. Generation 22 matches
use both colors on each opening, 64 games/arm, against the own seed's frozen parent.
These are paired parent comparisons, not direct treatment-control games.

The unchanged 10,000-replicate seed 4404703 bootstrap resamples entire mirror/shared
opening-ancestry clusters and recomputes row-weighted means:102 solved, 71 tactical,
28 deep clusters. Mirrors are not independent. Learning pools equal seed contrasts
before shared-family resampling; strength retains color pairs and independently
resamples 32 opening families within each seed. Two-sided 95% intervals use.025/.975;
one-sided 95% bounds use .05/.95. All intervals are conditional on these fixed checkpoints from two training seeds
and development families, not uncertainty over arbitrary training seeds.
No multiplicity correction was declared. Full unrounded estimates and per-seed
intervals are in [analysis.json](../research/alphazero_training_ablation/results/analysis.json).

## Neural learning: parents and both new generations

CE is `−log(sum policy on exact-optimal actions)`; accuracy uses raw legal-policy argmax; MSE uses actor-relative exactWDL.

| Seed | Checkpoint | Optimal-set CE (nats) | Raw optimal accuracy | Value MSE | Optimal probability |
|---|---|---|---|---|---|
| 42 | parent | 0.666753 | 73.33% | 0.611784 | 0.662098 |
| 42 | control-g21 | 0.664286 | 75.00% | 0.628376 | 0.663904 |
| 42 | treatment-g21 | 0.668663 | 73.33% | 0.626107 | 0.659835 |
| 42 | control-g22 | 0.664467 | 74.33% | 0.601619 | 0.666467 |
| 42 | treatment-g22 | 0.666446 | 72.67% | 0.587323 | 0.665531 |
| 314159 | parent | 0.641950 | 69.00% | 0.594897 | 0.662406 |
| 314159 | control-g21 | 0.630298 | 67.67% | 0.556936 | 0.666134 |
| 314159 | treatment-g21 | 0.633019 | 69.00% | 0.547453 | 0.664782 |
| 314159 | control-g22 | 0.635658 | 69.67% | 0.517307 | 0.665684 |
| 314159 | treatment-g22 | 0.636520 | 69.67% | 0.558629 | 0.664640 |

### Change from own generation 20 parent

Point change [two-sided 95% family interval]; accuracy changes are percentage points.

| Seed | Checkpoint | ΔCE | Δaccuracy(pp) | ΔMSE |
|---|---|---|---|---|
| 42 | control-g21 | -0.0025 [-0.0210, +0.0184] | +1.67 [-0.70, +4.00] | +0.0166 [-0.0162, +0.0534] |
| 42 | treatment-g21 | +0.0019 [-0.0151, +0.0206] | +0.00 [-2.14, +2.16] | +0.0143 [-0.0206, +0.0483] |
| 42 | control-g22 | -0.0023 [-0.0201, +0.0179] | +1.00 [-1.99, +3.80] | -0.0102 [-0.0479, +0.0230] |
| 42 | treatment-g22 | -0.0003 [-0.0175, +0.0201] | -0.67 [-3.57, +1.98] | -0.0245 [-0.0607, +0.0158] |
| 314159 | control-g21 | -0.0117 [-0.0329, +0.0067] | -1.33 [-4.05, +1.40] | -0.0380 [-0.1014, +0.0273] |
| 314159 | treatment-g21 | -0.0089 [-0.0273, +0.0078] | +0.00 [-2.87, +2.60] | -0.0474 [-0.1116, +0.0126] |
| 314159 | control-g22 | -0.0063 [-0.0274, +0.0120] | +0.67 [-2.13, +3.79] | -0.0776 [-0.1429, -0.0124] |
| 314159 | treatment-g22 | -0.0054 [-0.0237, +0.0111] | +0.67 [-1.61, +3.05] | -0.0363 [-0.1079, +0.0443] |

Both arms have lower generation 22 value-MSE point estimates relative to their own parent. Seed 314159
treatment improves less than control: the treatment-control value regression is
a relative comparison, not an assertion that treatment worsened its own parent.
Most parent-change value intervals overlap zero; only 314159 control at 22 excludes zero. Small policy changes and their uncertainty differ by checkpoint/seed.

### Primary treatment−control contrasts

| Seed | Gen | ΔCE [95%] | Δaccuracy(pp) [95%] | ΔMSE [95%] | Δoptimal probability [95%] |
|---|---|---|---|---|---|
| 42 | 21 | +0.004377 [-0.004699, +0.012900] | -1.67 [-3.45, +0.00] | -0.002269 [-0.024254, +0.017546] | -0.004069 [-0.006422, -0.001481] |
| 42 | 22 | +0.001978 [-0.009160, +0.014963] | -1.67 [-4.14, +0.64] | -0.014296 [-0.048926, +0.024434] | -0.000937 [-0.004064, +0.002145] |
| 314159 | 21 | +0.002722 [-0.004844, +0.011128] | +1.33 [-1.38, +3.74] | -0.009483 [-0.042005, +0.018678] | -0.001352 [-0.003773, +0.000668] |
| 314159 | 22 | +0.000863 [-0.008753, +0.010674] | +0.00 [-2.63, +2.59] | +0.041322 [+0.006618, +0.080432] | -0.001043 [-0.004080, +0.001584] |
| Equal-seed pooled | 22 | +0.001420 [-0.006656, +0.010774] | -0.83 [-2.87, +1.02] | +0.013513 [-0.012584, +0.042185] | -0.000990 [-0.003261, +0.001023] |

**Neural learning FAIL under the frozen rules.** Pooled CE+0.001420 fails≤−0.02;
its lower 95% bound−0.006656 is above−0.02. Pooled accuracy−0.833 pp fails≥+2 pp;
its upper 95% bound+1.025 pp is below+2 pp. Both seed CE point differences are
positive; seed 42 accuracy is−1.667 pp. Seed 314159 MSE+0.041322 exceeds+0.02
(95% interval[+0.006618,+0.080432]). Each stated frozen FAIL condition is retained.
The pooled CE/accuracy improvement intervals overlap zero and their required
confidence conditions also fail. Policy harm is not conclusively established;
the practical learning improvement sought here was not demonstrated.

### Calibration and generalization

| Seed | Checkpoint | Policy ECE proxy | Δfrom parent | Value bin error | Δfrom parent |
|---|---|---|---|---|---|
| 42 | parent | 0.2634 | +0.0000 | 0.2874 | +0.0000 |
| 42 | control-g21 | 0.2696 | +0.0062 | 0.3293 | +0.0420 |
| 42 | treatment-g21 | 0.2596 | -0.0038 | 0.2956 | +0.0083 |
| 42 | control-g22 | 0.2452 | -0.0181 | 0.3025 | +0.0152 |
| 42 | treatment-g22 | 0.2292 | -0.0342 | 0.3008 | +0.0135 |
| 314159 | parent | 0.2041 | +0.0000 | 0.3138 | +0.0000 |
| 314159 | control-g21 | 0.1900 | -0.0141 | 0.2835 | -0.0304 |
| 314159 | treatment-g21 | 0.2072 | +0.0030 | 0.2667 | -0.0472 |
| 314159 | control-g22 | 0.2114 | +0.0072 | 0.2260 | -0.0878 |
| 314159 | treatment-g22 | 0.2132 | +0.0091 | 0.2788 | -0.0350 |

Calibration uses the frozen ten bins: max-action probability versus optimal-set
correctness, and mean raw value versus exact meanWDL. PolicyECE is a diagnostic
proxy when several actions are optimal, not a validated probability-of-correctness
model. Detailed bins are in [analysis.json](../research/alphazero_training_ablation/results/analysis.json). Lower
ECE does not override worse accuracy/value endpoints; calibration effects are mixed.

There is **no prospective unseen test split or valid generalization-gap estimate**.
Training policy CE targets visit distributions over seven logits; evaluated CE
targets exact optimal sets with legal masking. Training value loss uses sampled
self-play outcomes, whereas solvedMSE uses exactWDL on different positions.
Subtracting these losses would misstate a generalization gap. The descriptive
[observed replay overlap](../research/alphazero_training_ablation/results/observed-replay-overlap.json) finds solved
rows already present in observed parent replay:2/300 for 42,4/300 for 314159;
tactical0/200 for both; deep28/128 for 42 and 6/128 for 314159. Generation 22 overlap
is 2/300 solved and 0/200 tactical in all arms, and deep28/34 for 42C/T, 8/8 for 314159C/T.
These reflection/actor matches are material for interpreting the reused 4E table.
Historical generations 1–12 are outside available retained replay, so absence
from observed replay does not establish unseen generalization. No fixture was
removed or regrouped after this check.

### Training losses and gradient stability

Each row has 64 updates; loss trajectory shows means over first/last 16 sampled minibatches, not a fixed validation set.

| Seed | Arm | Gen | Mean policy loss | Mean value loss | Combined first16→last16 | Grad mean/max |
|---|---|---|---|---|---|---|
| 42 | control | 21 | 1.0013 | 0.3923 | 1.3911→1.4108 | 1.729/2.490 |
| 42 | control | 22 | 0.9802 | 0.3996 | 1.3827→1.3712 | 1.761/2.591 |
| 42 | treatment | 21 | 1.0061 | 0.4040 | 1.4139→1.4284 | 1.781/2.380 |
| 42 | treatment | 22 | 0.9792 | 0.4183 | 1.3992→1.3806 | 1.803/2.567 |
| 314159 | control | 21 | 1.0449 | 0.3599 | 1.4066→1.4260 | 1.678/2.324 |
| 314159 | control | 22 | 1.0319 | 0.3499 | 1.4040→1.3642 | 1.625/2.103 |
| 314159 | treatment | 21 | 1.0443 | 0.3608 | 1.4129→1.4179 | 1.672/2.150 |
| 314159 | treatment | 22 | 1.0366 | 0.3542 | 1.4176→1.3611 | 1.680/2.010 |

All 512 updates are finite, with zero clipped updates and maximum gradient norm 2.591 below clip 5. Generation 22 mean losses often decline modestly; sampling noise and heterogeneous historical targets prevent monotonic-loss claims. Complete 64-step trajectories, gradient quantiles and diagnostics are in each [generation sidecar](../research/alphazero_training_ablation/results/execution-evidence/) and [mechanism-resources.json](../research/alphazero_training_ablation/results/mechanism-resources.json). Training stability is not evidence of improved playing strength.

## Tactical reliability

Win and safe categories each contain 100 rows. Unique-safe is the established mandatory-block/safe-response endpoint; no separate multiple-safe stratum is available. Last columns are avoidable one-ply/missed-win counts over 200 rows.

| Seed | Checkpoint | Policy win | Policy safe/block | PUCT win | PUCT safe/block | Policy errors/missed | PUCT errors/missed |
|---|---|---|---|---|---|---|---|
| 42 | parent-g20 | 44.00% | 63.00% | 88.00% | 91.00% | 46/56 | 9/12 |
| 42 | control-g21 | 39.00% | 62.00% | 88.00% | 91.00% | 49/61 | 9/12 |
| 42 | treatment-g21 | 44.00% | 63.00% | 88.00% | 90.00% | 47/56 | 10/12 |
| 42 | control-g22 | 40.00% | 68.00% | 88.00% | 92.00% | 42/60 | 8/12 |
| 42 | treatment-g22 | 42.00% | 65.00% | 88.00% | 92.00% | 46/58 | 8/12 |
| 314159 | parent-g20 | 36.00% | 69.00% | 92.00% | 96.00% | 43/64 | 4/8 |
| 314159 | control-g21 | 38.00% | 64.00% | 89.00% | 95.00% | 46/62 | 5/11 |
| 314159 | treatment-g21 | 37.00% | 63.00% | 89.00% | 96.00% | 49/63 | 4/11 |
| 314159 | control-g22 | 35.00% | 70.00% | 89.00% | 96.00% | 41/65 | 4/11 |
| 314159 | treatment-g22 | 37.00% | 67.00% | 90.00% | 96.00% | 44/63 | 4/10 |

Search greatly raises absolute win/safe accuracy relative to raw policy, while retaining mistakes. Exact per-checkpoint/parent intervals and paired search−policy contrasts are in [checkpoint-behavior.json](../research/alphazero_training_ablation/results/checkpoint-behavior.json); raw measurements remain separate from gates.

### Frozen 2 pp point-regression gates

Treatment differences in percentage points [two-sided 95% interval]. A point drop greater than 2 pp versus either parent or control fails, regardless of interval width.

| Seed | Gen | Method | Category | T−C(pp) [95%] | T−parent(pp) [95%] | Point gate |
|---|---|---|---|---|---|---|
| 42 | 21 | policy | win | +5.00 [+0.00, +10.71] | +0.00 [-4.08, +3.92] | PASS |
| 42 | 21 | policy | safe/block | +1.00 [-5.00, +6.78] | +0.00 [-6.25, +6.25] | PASS |
| 42 | 21 | puct | win | +0.00 [+0.00, +0.00] | +0.00 [-2.78, +3.06] | PASS |
| 42 | 21 | puct | safe/block | -1.00 [-3.33, +0.00] | -1.00 [-5.56, +3.12] | PASS |
| 42 | 22 | policy | win | +2.00 [-1.89, +5.77] | -2.00 [-6.38, +1.82] | PASS |
| 42 | 22 | policy | safe/block | -3.00 [-8.82, +2.88] | +2.00 [-2.73, +7.14] | FAIL |
| 42 | 22 | puct | win | +0.00 [-2.94, +3.00] | +0.00 [-2.78, +3.12] | PASS |
| 42 | 22 | puct | safe/block | +0.00 [-4.81, +4.90] | +1.00 [-5.32, +7.84] | PASS |
| 314159 | 21 | policy | win | -1.00 [-3.97, +2.33] | +1.00 [-3.00, +5.81] | PASS |
| 314159 | 21 | policy | safe/block | -1.00 [-7.29, +5.56] | -6.00 [-14.00, +0.93] | FAIL |
| 314159 | 21 | puct | win | +0.00 [-3.00, +2.94] | -3.00 [-6.25, +0.00] | FAIL |
| 314159 | 21 | puct | safe/block | +1.00 [+0.00, +3.33] | +0.00 [+0.00, +0.00] | PASS |
| 314159 | 22 | policy | win | +2.00 [-2.94, +6.82] | +1.00 [-2.33, +4.44] | PASS |
| 314159 | 22 | policy | safe/block | -3.00 [-8.16, +2.04] | -2.00 [-9.18, +4.46] | FAIL |
| 314159 | 22 | puct | win | +1.00 [+0.00, +3.33] | -2.00 [-6.25, +0.00] | PASS |
| 314159 | 22 | puct | safe/block | +0.00 [+0.00, +0.00] | +0.00 [-3.00, +3.00] | PASS |

Four FAIL cells:42/g22 policy-safe−3 pp versus control;314159/g21 policy-safe−6 pp
versus parent;314159/g21 PUCT-win−3 pp versus parent;314159/g22 policy-safe−3 pp
versus control. These are prospective point-margin failures, not confidence-bound
proofs of broad tactical harm. The remaining point gates pass; no margin changed.

## Deeper game-theoretic mistakes

A mistake decreases exact achievable actorWDL (win→draw/win→loss/draw→loss), not merely an eventual loss. Complete legal vectors cover every included action. Bounds shown are distinct one-sided 95% lower/upper bounds, compared with+5 pp.

| Seed | Gen | Method | Suite | Control error | Treatment error | Δ(pp) | L05,U95(pp) | T−parent(pp) | NI |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 21 | policy | solved300 | 25.00% | 26.67% | +1.67 | +0.33, +3.12 | +0.00 | PASS |
| 42 | 21 | policy | 4E128 | 22.66% | 25.78% | +3.12 | +0.83, +5.83 | +2.34 | INCONCLUSIVE |
| 42 | 21 | puct | solved300 | 20.33% | 18.67% | -1.67 | -2.78, -0.66 | -0.33 | PASS |
| 42 | 21 | puct | 4E128 | 19.53% | 20.31% | +0.78 | -3.12, +4.62 | +0.00 | PASS |
| 42 | 22 | policy | solved300 | 25.67% | 27.33% | +1.67 | -0.31, +3.74 | +0.67 | PASS |
| 42 | 22 | policy | 4E128 | 24.22% | 25.78% | +1.56 | -1.71, +5.56 | +2.34 | INCONCLUSIVE |
| 42 | 22 | puct | solved300 | 21.00% | 20.33% | -0.67 | -2.33, +1.08 | +1.33 | PASS |
| 42 | 22 | puct | 4E128 | 20.31% | 18.75% | -1.56 | -4.73, +1.54 | -1.56 | PASS |
| 314159 | 21 | policy | solved300 | 32.33% | 31.00% | -1.33 | -3.33, +0.96 | +0.00 | PASS |
| 314159 | 21 | policy | 4E128 | 18.75% | 18.75% | +0.00 | -1.96, +1.54 | +0.78 | PASS |
| 314159 | 21 | puct | solved300 | 17.00% | 15.33% | -1.67 | -3.02, -0.34 | +0.33 | PASS |
| 314159 | 21 | puct | 4E128 | 10.16% | 8.59% | -1.56 | -4.48, +0.81 | -2.34 | PASS |
| 314159 | 22 | policy | solved300 | 30.33% | 30.33% | +0.00 | -2.16, +2.22 | -0.67 | PASS |
| 314159 | 22 | policy | 4E128 | 20.31% | 19.53% | -0.78 | -2.94, +1.56 | +1.56 | PASS |
| 314159 | 22 | puct | solved300 | 16.00% | 17.00% | +1.00 | -0.51, +2.83 | +2.00 | PASS |
| 314159 | 22 | puct | 4E128 | 10.16% | 10.16% | +0.00 | +0.00, +0.00 | -0.78 | PASS |

All deep gates PASS except 42 raw-policy 4E128 at generation 21/g22: upper bounds+5.83/+5.56 pp
exceed+5 pp, yielding INCONCLUSIVE. Neither lower bound exceeds the margin, so
there is no deep-margin FAIL. These bounds do not certify arbitrary positions;
4E has only 28 clusters and substantial observed-replay overlap. All parent/arm
rates, changes and intervals are in [checkpoint-behavior.json](../research/alphazero_training_ablation/results/checkpoint-behavior.json).
The separate overall tactical **FAIL** is driven by the four immediate point gates.

## Playing strength against frozen parents

Every row has 64 matches, 32 per color. Scores=(wins+0.5 draws)/64.

| Seed | Arm | W/D/L | Score | Two-sided95% | As X score | As O score |
|---|---|---|---|---|---|---|
| 42 | control | 30/7/27 | 52.34% | 42.19–62.50% | 54.69% | 50.00% |
| 42 | treatment | 31/7/26 | 53.91% | 44.53–63.28% | 57.81% | 50.00% |
| 314159 | control | 28/5/31 | 47.66% | 38.28–57.03% | 50.00% | 45.31% |
| 314159 | treatment | 31/6/27 | 53.12% | 45.31–61.72% | 54.69% | 51.56% |

| Seed | T−C(pp) [two-sided95%] | One-sided lower(pp) | One-sided upper(pp) | −5 pp NI |
|---|---|---|---|---|
| 42 | +1.56 [-9.38, +10.94] | -7.03 | +9.38 | INCONCLUSIVE |
| 314159 | +5.47 [-5.47, +17.19] | -3.91 | +14.84 | PASS |

Both treatment point scores exceed 50%, but equal-seed pooled score 53.52% has
95% interval[47.27%,59.77%]; lower bound<50%, so the supporting positive signal
does not pass. Seed 314159 demonstrates the declared noninferiority margin on
these openings. Seed 42's lower bound−7.03 pp crosses−5 pp; its upper bound+9.38 pp
does not establish a material regression. Overall **INCONCLUSIVE**. Small positive
matched point differences do not establish a superior agent. No additional
tournament or checkpoint promotion was performed.

## Self-play diversity and outcome diagnostics

Each row is 64 complete games. Duplicate fractions count repeat occurrences after the first unique state; mirror fractions canonicalize reflection/actor. Post 8 uses only plies≥8.

| Seed/arm/gen | Positions | Mean length | Duplicates | Mirror duplicates | Post8 duplicates | Opening families | Target entropy(nats) |
|---|---|---|---|---|---|---|---|
| 42/control/g21 | 1871 | 29.23 | 36.50% | 39.23% | 23.29% | 43 | 0.700 |
| 42/control/g22 | 2254 | 35.22 | 41.88% | 45.39% | 33.08% | 37 | 0.687 |
| 42/treatment/g21 | 1617 | 25.27 | 27.03% | 30.80% | 6.41% | 43 | 0.735 |
| 42/treatment/g22 | 2032 | 31.75 | 27.21% | 29.28% | 11.05% | 33 | 0.661 |
| 314159/control/g21 | 2179 | 34.05 | 44.10% | 47.91% | 34.13% | 31 | 0.734 |
| 314159/control/g22 | 2230 | 34.84 | 53.54% | 55.20% | 46.27% | 31 | 0.682 |
| 314159/treatment/g21 | 2081 | 32.52 | 26.38% | 28.21% | 10.01% | 31 | 0.730 |
| 314159/treatment/g22 | 2133 | 33.33 | 27.00% | 29.21% | 11.10% | 33 | 0.743 |

| Seed/arm/gen | X wins/O wins/draws | Actor z counts −1/0/+1 | Avoidable errors | Missed wins | Avoidable/position |
|---|---|---|---|---|---|
| 42/control/g21 | 39/13/12 | 664/504/703 | 30 | 20 | 1.60% |
| 42/control/g22 | 40/13/11 | 876/462/916 | 29 | 9 | 1.29% |
| 42/treatment/g21 | 43/16/5 | 682/210/725 | 30 | 16 | 1.86% |
| 42/treatment/g22 | 36/18/10 | 788/420/824 | 23 | 2 | 1.13% |
| 314159/control/g21 | 33/9/22 | 611/924/644 | 14 | 9 | 0.64% |
| 314159/control/g22 | 42/5/17 | 737/714/779 | 15 | 5 | 0.67% |
| 314159/treatment/g21 | 41/12/11 | 789/462/830 | 25 | 7 | 1.20% |
| 314159/treatment/g22 | 43/11/10 | 835/420/878 | 30 | 14 | 1.41% |

Treatment reduces duplicate and mirror-duplicate rates in every matched cell
(duplicate reductions 9.48,14.67,17.72,26.54 pp). It also produces fewer positions
and shorter games in every cell. Opening-family counts and target entropy do not
uniformly increase. Self-play outcome-label distributions shift; they are not
head-to-head strength scores. One-ply errors are mixed:314159 treatment has more
avoidable errors in both generations, whereas 42/g22 has fewer. No complete deep
oracle scan of all self-play positions was prospectively budgeted; existing
immediate-win/all-unsafe exact-proof diagnostics remain in the boundaries.
Unresolved deeper self-play states are not labeled safe or erroneous.

## Replay dilution and actual optimizer exposure

Parents initialize with 68,856/64, 514 examples and 4,957/4, 890 AdamW steps for 42/314159, 2,048 games across 13–20. Generation 21 retains 14–21; generation 22 retains 15–22, replacing entire old generations despite smaller 64-game additions.

| Seed/arm/gen | Replay positions | Experimental positions | Experimental share | New sample appearances | New sampling share | Current-gen sampling share |
|---|---|---|---|---|---|---|
| 42/control/g21 | 62527 | 1871 | 2.99% | 233/8192 | 2.84% | 2.84% |
| 42/control/g22 | 56414 | 4125 | 7.31% | 567/8192 | 6.92% | 3.89% |
| 42/treatment/g21 | 62273 | 1617 | 2.60% | 198/8192 | 2.42% | 2.42% |
| 42/treatment/g22 | 55938 | 3649 | 6.52% | 509/8192 | 6.21% | 3.47% |
| 314159/control/g21 | 58966 | 2179 | 3.70% | 323/8192 | 3.94% | 3.94% |
| 314159/control/g22 | 53756 | 4409 | 8.20% | 639/8192 | 7.80% | 3.66% |
| 314159/treatment/g21 | 58868 | 2081 | 3.54% | 309/8192 | 3.77% | 3.77% |
| 314159/treatment/g22 | 53561 | 4214 | 7.87% | 619/8192 | 7.56% | 3.59% |

Sampling appearances by producing generation (generation: count), each row totals 8,192=64×128:

| Seed/arm/gen | Sampling by source generation |
|---|---|
| 42/control/g21 | 14:1088, 15:1114, 16:1239, 17:1181, 18:1122, 19:1098, 20:1117, 21:233 |
| 42/control/g22 | 15:1240, 16:1336, 17:1330, 18:1197, 19:1252, 20:1270, 21:248, 22:319 |
| 42/treatment/g21 | 14:1095, 15:1118, 16:1247, 17:1185, 18:1124, 19:1102, 20:1123, 21:198 |
| 42/treatment/g22 | 15:1245, 16:1345, 17:1336, 18:1206, 19:1266, 20:1285, 21:225, 22:284 |
| 314159/control/g21 | 14:1059, 15:1138, 16:1142, 17:1090, 18:1158, 19:1099, 20:1183, 21:323 |
| 314159/control/g22 | 15:1222, 16:1264, 17:1243, 18:1247, 19:1244, 20:1333, 21:339, 22:300 |
| 314159/treatment/g21 | 14:1060, 15:1142, 16:1147, 17:1090, 18:1158, 19:1099, 20:1187, 21:309 |
| 314159/treatment/g22 | 15:1229, 16:1264, 17:1242, 18:1251, 19:1252, 20:1335, 21:325, 22:294 |

Only 2.42–3.94% of generation 21 samples and 6.21–7.80% of generation 22 samples
come from experimental data. Across all 65,536 sample appearances, 3,397 (5.18%)
are experimental:62,139 (94.82%) remain historical. Treatment's shorter collections
lower new replay/sampling shares versus its control in every cell. Each lineage
gets only 128 new updates, approximately 2.6% of its parent's accumulated steps.

**Observed:** data repetition falls, historical replay dominates actual updates,
and practical policy gains do not emerge. **Hypotheses:** replay dilution/small
update dose may limit sensitivity; changed outcomes, harder sampled actions or
checkpoint-specific value effects may offset diversity gains. This experiment
does not identify these mechanisms causally or isolate architecture versus dose.
It does not prove exploration diversity unimportant, nor justify automatically
adding data or updates after the failed gates.

## Computational cost and budget reconciliation

Execution began**2026-10-12T10:58:27.029648-04:00**, finishing approximately
**2026-10-12T11:32:03.188250-04:00**. Finish is reconstructed from saved epoch
start plus monotonic elapsed, not a separately saved process-exit timestamp.
Primary wall**2016.159s /33.60 min**
is below 180 min; every generation attempt is below 20 min (maximum
96.001s). Child walls sum
1986.274s, with 29.884s
supervisor/process/I/O overhead. Child CPU is 1978.636s;
supervisor/startup CPU before worker timers is not fully recorded. Highest child
peakRSS is 483.8 MiB, not total machine memory.

Training-task walls total 721.044s; position evaluations 608.190s; strength 657.040s.
Direct collection totals 489.552s. Optimization stage totals 28.210s: first
before_update→before_checkpoint includes 64 updates plus summary/RNG bookkeeping,
excluding persistence; it is **not pure optimizer time**. The rest of training
task time includes strict loads, reconstruction, hashing, persistence and reloads.
Actual primary counts are 512 games/updates, 16,397 positions, 256 strength games,
7,274 strength decisions. PUCT simulations:4,197,632 self-play, 3,215,360 development
positions, 3,724,288 strength; **11,137,280 total**. Collection executed 426,755 batch
forwards and 3,211,665 neural positions (mean occupancy 7.53). Full evaluation
neural-forward throughput counters were not recorded.

Generation resource detail (all eight have 64 updates):

| Seed/arm/gen | Collection(s) | Optimizer stage(s) | Full task(s) | Batch forwards | Mean batch | Neural pos/s | PeakRSS MiB |
|---|---|---|---|---|---|---|---|
| 42/control/g21 | 54.40 | 3.46 | 84.81 | 48667 | 7.38 | 6605 | 424.9 |
| 42/control/g22 | 65.85 | 3.62 | 95.01 | 56437 | 7.50 | 6427 | 472.0 |
| 42/treatment/g21 | 48.88 | 3.40 | 78.72 | 43621 | 7.42 | 6617 | 483.8 |
| 42/treatment/g22 | 59.18 | 3.48 | 87.36 | 52165 | 7.54 | 6645 | 404.3 |
| 314159/control/g21 | 66.54 | 3.51 | 96.00 | 57834 | 7.53 | 6546 | 475.7 |
| 314159/control/g22 | 67.56 | 3.66 | 95.35 | 57704 | 7.61 | 6503 | 386.4 |
| 314159/treatment/g21 | 62.37 | 3.59 | 91.72 | 53977 | 7.63 | 6603 | 408.8 |
| 314159/treatment/g22 | 64.77 | 3.49 | 92.07 | 56350 | 7.55 | 6571 | 396.3 |

All 10 per-checkpoint evaluation walls, all 4 match task walls/CPU/RSS and exact
resource reconciliation are in [mechanism-resources.json](../research/alphazero_training_ablation/results/mechanism-resources.json)
and the [retained status ledger](../research/alphazero_training_ablation/results/execution-evidence/status.json).
No aborted/failed/retried primary work is present. Readiness 132.656s and 8 disposable
updates/72 bounded search probes were separate prelaunch work, not primary data.
Final evidence-only audits/analysis/tests are separate from the 33.60 min primary
clock; no unused training budget was filled.

## Independent critical review and retained corrections

A new reviewer agent, which authored neither the frozen implementation nor main
results analysis, independently reconstructed 6,280 evaluation metrics, 512 legal
self-play histories/targets/diversity/tactics, 256 strength outcomes/colors,
cluster components and 10,000-replicate multiplicity-weighted bootstraps.
It compared 1,316 main-analysis numerical scalars; maximum discrepancy 5.55e-16.
Labels, confidence directions, thresholds, replay exposure/resources and support
signal agree. See [independent-results-review.md](../research/alphazero_training_ablation/results/independent-results-review.md)
and [independent-recomputed.json](../research/alphazero_training_ablation/results/independent-recomputed.json).
This is independent arithmetic/source review on the same host, not laboratory replication.

Final evidence-only analysis/integrity tests:11 passed in 2.21s, without model queries or optimizer work. The unchanged frozen analysis was reproduced to a fresh file with identical parsed JSON and byte SHA; see [frozen-analysis-reproduction.json](../research/alphazero_training_ablation/results/frozen-analysis-reproduction.json) and [final-analysis-tests.txt](../research/alphazero_training_ablation/results/final-analysis-tests.txt). Prelaunch 6 ablation/56 boundary cases and 341 relevant regressions remain documented in the frozen [validation summary](../research/alphazero_training_ablation/results/validation-summary.json).

Frozen analysis sources remain untouched. Limitations: its execution label alone
does not audit the full lineage; this report requires the separate strict audit.
Pooled learning uses positional zip without a fresh ID check; all actual row
orders match frozen membership, and independent ID-based pairing reproduces it.
There is no material arithmetic defect requiring a changed scientific analysis.

The new auxiliary audit retained and corrected three reporting issues: JSON
integer-key/tuple normalization when comparing binary histories with sidecars;
an unrelated H-B worktree's untracked review/report status drift while tracked
bytes/HEAD remained equal during the protection audit; and reconstructed payload
integrity reflecting current Git reporting rather than the stored archive's
integrity field. Original failure logs, before/current/after inventories and
[integrity metadata correction](../research/alphazero_training_ablation/results/archive-integrity-reporting-correction.json)
are preserved. The final audit reads the original verified archive integrity.
No runtime/source validation or decision rule was weakened, no original result
was discarded, and no training/evaluation was repeated. The prelaunch ImportError,
wrong test-filename attempt and prior Git-reporting comparison failure remain
documented in the prospective report/review evidence and validation summary.

## Historical preservation, publication and recovery

All**150** official campaign files match the 4F byte inventory before/after final
audits. Declaration SHA remains
`c8ab2227d4b2c707b3185da8695a991f0914faede88ecd5f1c7fc84fb017b716`.
Historical checkout remains`1894bc4d8f035cf738df8e3fc0ce234f1ce2c8be`;
the designated ResNet branch/worktree remains`7f5182daa10156e04a8b0f7727d9bec9cad39cf1`.
Tracked prior source/research/benchmark files and these protected worktrees are
unchanged. The separate H-B worktree was observed active outside this task;
its untracked status drift is reported without touching it. This task issued no
write to either ResNet worktree, production/API/deployment, main or canonical benchmark.

The unchanged `publish_evidence.py` packages completed JSON/log evidence under
[execution-evidence](../research/alphazero_training_ablation/results/execution-evidence/), with
[evidence-manifest.json](../research/alphazero_training_ablation/results/evidence-manifest.json). Large checkpoint
binaries stay ignored/local. Only reviewed research logs are force-added where
the repository's generic*.log ignore rule applies. No checkpoints are committed.
[integrity-audit.json](../research/alphazero_training_ablation/results/integrity-audit.json) records every absolute
recovery path and checkpoint/model/optimizer/deterministic-state hash.

From this worktree, set all five thread variables to 1 and use the existing M4
Python with`-B`; follow [README.md](../research/alphazero_training_ablation/README.md): run evidence-only
`python -m research.alphazero_training_ablation.analyze`. Publication uses the
existing `publish_evidence` command into a fresh destination; never overwrite the
retained first package. Supplementary checks use the sibling postexecution
`audit`, `summarize` and `overlap` modules. The packaged independent scripts
reproduce metrics, diagnostics and complete-game checks without original analysis
helpers. See [postexecution instructions](../research/alphazero_training_ablation_postexecution/README.md).
Strict recovery uses original `experimental_cli inspect` on a visible boundary
and its trusted SHA; neither launcher nor training command is needed here.

## Limitations and recommended next phase

The pilot estimates these fixed CNN checkpoints, two seeds and reused development
families after two 64-game/64-update generations. It has low new-data exposure,
few optimizer steps, small strength samples, only 28 deep clusters, some directly
overlapping replay positions and unknown early historical replay provenance.
Bootstrap intervals do not include between-training-seed uncertainty; one fixed
evaluation seed per position and no multiplicity correction were declared.
Tactical point FAIL does not equal statistical proof of broad harm. Strength
does not demonstrate superiority and does not rule out the 5 pp regression in 42.
Checkpoint 21/22 effects differ, including 314159 treatment's smaller value gain
at 22. There is no evidence here to adopt sixteen exploratory plies.

Recommend **Phase 4D.4H-B controlled CNN versus ResNet inference-cost, training-
throughput and fixed-data learning comparison**, using read-only historical
generations 13–20 replay and a prospectively frozen split that groups reflected
boards, transpositions and related trajectories/openings across partitions.
Prioritize ResNet 3×64 (322,440 parameters) against CNN (325,896), with equal
optimizer/data budgets and measured wall/throughput. ResNet 5×64 (470,408) can
be a separately declared capacity/cost comparator, not presumed equally sized.
The existing ResNet implementation at 7f5182d is unchanged and no comparison is
launched here. This separates learning/architecture sensitivity from the
exploration trajectory and dilution confound before allocating more self-play.

Retain the eight-ply schedule. Longer exploration may remain a future hypothesis
if later learning studies justify it, but reduced repetition alone, this pilot's
positive match point scores or unused compute budget do not justify further
training, treatment adoption, a champion or deployment.
