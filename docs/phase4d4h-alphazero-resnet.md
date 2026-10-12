# Phase 4D.4H-A — AlphaZero ResNet architecture implementation

Two compact residual networks for Connect 4 AlphaZero are implemented behind the
Phase 4D.4F model factory: **3 blocks × 64 channels (322,440 parameters)** and
**5 blocks × 64 channels (470,408 parameters)**. Both keep the existing input,
output, loss, replay, PUCT, batched-inference and experimental-lifecycle
contracts. The historical CNN is unchanged and remains the default.

This phase delivers an implementation and correctness evidence only. Nothing
here shows that either ResNet plays or learns better than the CNN: no benchmark,
training experiment or tournament was run.

## Scope and isolation

Work was done in the new worktree `experiment-output/phase4d4h-worktree` on
branch `phase4d4h-alphazero-resnet`, created from the Phase 4D.4F commit
`9325f0a19e6b0106ea1dcd265316b57323f8eb4f`.

| File | Change |
|---|---|
| `games/connect4/alphazero_v2/resnet.py` | New: models, identities, validation, inference, trainer, factory, inference artifact |
| `games/connect4/alphazero_v2/experimental_models.py` | Registers the two ResNet factories; CNN branch unchanged |
| `tests/alphazero_resnet_cases.py`, `tests/test_alphazero_resnet.py` | New: 73 focused cases and their isolated-interpreter wrapper |
| `research/alphazero_resnet/` | New: specification record, external source pins, test logs |

Every other module in `games/connect4/alphazero_v2/` is byte-identical to the base
commit, and a test asserts it. The Phase 4D.4G worktree, its branch, the frozen
historical checkout, the shared virtual environment and all campaign artifacts
were not modified.

While 4D.4G was running, validation was limited to the focused cases and the
directly affected existing tests: one thread at `nice 19`, about four minutes
of single-core time. The 4D.4G run then completed on its own (its status file
reports `COMPLETE_PENDING_FINAL_ANALYSIS`, 512 games, 512 updates, about 34
minutes elapsed) and no 4D.4G process remained. Only after that was the full
regression sweep run, also at `nice 19`.

## Architecture

```
input (N,1,6,7) float32, current-player-relative {-1,0,+1}
  stem:    conv3x3(1→64, no bias) → BN → ReLU
  tower:   B × [ conv3x3(64→64) → BN → ReLU → conv3x3(64→64) → BN → (+ skip) → ReLU ]
  policy:  conv1x1(64→32) → BN → ReLU → flatten → linear(1344→7)            raw logits (N,7)
  value:   conv1x1(64→32) → BN → ReLU → flatten → linear(1344→64) → ReLU
           → linear(64→1) → tanh                                             value (N,1) in [-1,1]
```

| Identity | Blocks | Parameters |
|---|---|---|
| `connect4-conv64-128-128-policy7-scalar-tanh-v2` (CNN, default) | — | 325,896 |
| `connect4-resnet3x64-bn-policy32-value32x64-scalar-tanh-v1` | 3 | 322,440 |
| `connect4-resnet5x64-bn-policy32-value32x64-scalar-tanh-v1` | 5 | 470,408 |

All convolutions feeding a batch norm are bias-free. Batch norm uses PyTorch
defaults (eps 1e-5, momentum 0.1, tracked running statistics). Weights use
PyTorch default initialization, the same scheme the CNN gets, from a seed that
is domain-separated by architecture identity; the global RNG is not advanced.

## Design decisions and external influence

Network definitions were read from the five implementations audited in
Phase 4D.4A, at that audit's pinned commits, plus AlphaZero.jl's ResNet source.
Paths, commits and SHA-256s are in `research/alphazero_resnet/external_sources.json`.
The original AlphaGo Zero/AlphaZero architecture was reviewed from the papers.
No external code or weights were copied.

| Decision | Choice | What the references do | Why |
|---|---|---|---|
| Residual block | Post-activation: conv-BN-ReLU-conv-BN, identity skip, ReLU | Identical in AlphaGo Zero/AlphaZero and all five projects | Unanimous and well understood at this depth; pre-activation variants add a variable without evidence here |
| Stem | One conv3x3-BN-ReLU to 64 channels | Same in all references | Standard; no reason to differ |
| Normalization | BatchNorm2d, momentum 0.1 | All use batch norm. AlphaZero.jl sets momentum 0.1 for Connect Four (its library default is 0.6) | Batch-128 updates suit batch norm; eval-mode inference is per-row, so batched and single predictions agree |
| Convolution bias | None before batch norm | eugeneyp, suharshkumar, kuotunyu omit it; AlphaZero.jl and plkmo's stem keep it | The bias is redundant under batch norm |
| Width / depth | 64 channels; 3 and 5 blocks | suharshkumar 5×64; kuotunyu 6×96 with a 3×64 smoke preset; AlphaZero.jl and eugeneyp 5×128; plkmo 19×128; original 19–39×256 | Specified for this phase. 3×64 is parameter-matched to the CNN, 5×64 tests added depth |
| Head filters | 32 policy, 32 value | AlphaZero.jl, eugeneyp, kuotunyu use 32/32. suharshkumar and the original use 2/1 | Identical to the existing CNN heads, so the comparison isolates the trunk |
| Value hidden width | 64 | kuotunyu and suharshkumar 64; AlphaZero.jl 128; eugeneyp 256; original 256 | Identical to the CNN head; a wider value head would confound the trunk comparison |
| Weight decay | Conv/linear weights only; never biases or batch-norm affines | AlphaZero.jl regularizes only Dense/Conv weights. eugeneyp and kuotunyu decay every parameter. The original applies L2 to all | Extends the CNN rule (weights decay, biases do not); decaying batch-norm scales changes the effective learning rate rather than regularizing |
| Optimizer | The CNN's AdamW settings from `V2Config`, unchanged | Adam (AlphaZero.jl, eugeneyp), AdamW (kuotunyu), SGD-momentum (original) | A fair comparison changes the architecture only |
| Initialization | PyTorch defaults; batch-norm scale 1 | None of the reviewed network definitions customizes it | Same as the CNN; zero-initialized residual scales are unnecessary at 3–5 blocks |
| Input | Unchanged single signed plane | Every reference uses 2 or 3 planes | The input contract is preserved by requirement |

Three choices deliberately depart from the largest references. First, the heads
copy our CNN rather than the original's 2/1-filter heads, because the question
for this project is whether a residual trunk helps, and matching heads keeps
that the only difference. Second, the networks are far smaller than 5×128:
suharshkumar's source argues that on a laptop self-play throughput, not
capacity, is the bottleneck, and Phase 4D.4A found no evidence that capacity is
the primary constraint here. Third, the single signed input plane is kept. With zero padding it
makes an off-board cell look like an empty one at the first layer; the CNN shares
this property, so it does not bias the comparison, but an input-plane ablation is
a separate question.

## Parameter counts and expected computational tradeoffs

Static values from `research/alphazero_resnet/results/architectures.json`.
Multiply-accumulates (MACs) count convolution and linear weights applied per
position. They exclude batch norm, activations and framework overhead and are
**not measured speed**.

| | CNN | ResNet 3×64 | ResNet 5×64 |
|---|---|---|---|
| Parameters | 325,896 | 322,440 (0.989×) | 470,408 (1.443×) |
| Trunk / policy head / value head | 222,080 / 13,543 / 90,273 | 222,656 / 11,527 / 88,257 | 370,624 / 11,527 / 88,257 |
| MACs per position | 9,753,472 | 9,581,440 (0.982×) | 15,774,592 (1.617×) |
| Trunk MACs | 9,313,920 | 9,313,920 (1.000×) | 15,507,072 (1.665×) |
| Sequential trunk convolutions | 3 | 7 | 11 |
| Trunk receptive field | 7×7 | 15×15 | 23×23 |
| Batch-norm layers | 0 | 9 | 13 |
| State tensors (of which buffers) | 16 (0) | 60 (27) | 84 (39) |

Expectations to test, not results:

- **ResNet 3×64 is a near-exact budget match for the CNN.** Its trunk performs
  exactly the CNN's trunk arithmetic (6 × 36,864 weights equals the CNN's
  73,728 + 147,456) spread over more, narrower layers. It should cost about the
  same per position, plus overhead from nine unfused batch norms and four extra
  layer dispatches. At batch 8 on one CPU thread that overhead may be visible.
- **ResNet 5×64 costs about 1.6× the arithmetic.** Under an equal wall budget it
  will collect fewer self-play games or simulations, so it must learn more per
  game to break even.
- **Depth buys receptive field.** Each CNN trunk feature sees a 7×7 window, so
  a corner cell sees only a 4×4 region of the board and whole-board integration
  happens only in the linear heads. Both ResNets see the whole 6×7 board from
  every cell inside the trunk.
- **Training steps are somewhat dearer** than the MAC ratio suggests, because
  train-mode batch norm computes and backpropagates batch statistics.
- **Batch-norm folding** into the preceding convolutions would remove the
  inference overhead but changes floating-point results, so it would need its own
  identity and validation. It is not implemented.

## Compatibility with the current AlphaZero infrastructure

Unchanged and reused as-is: encoding, `validate_input`, `validate_outputs`,
legal-logit masking, `V2Prediction`, `training_loss`, `batch_tensors`,
`V2Example` replay, reflection augmentation, `collect_batched`, bitboard and
engine PUCT, `ExperimentalRunner`, the boundary schema, `validate_optimizer`,
`weights_sha256` and atomic artifact publication.

- **Factory.** `model_factory(identity)` returns a `ResNetFactory` bound to one
  identity, with the same surface as `CNNFactory`. `ExperimentalConfig` still
  defaults to the CNN, and `SUPPORTED_ARCHITECTURES` lists all three.
- **Inference, PUCT and batching.** `ResNetInference` subclasses `V2Inference`
  and inherits `predict` and `predict_batch` verbatim, so `PUCTSearch`,
  `EvaluationAgent` and pool-eight collection accept it without modification.
- **Training.** `ResNetTrainer` subclasses `V2Trainer` and inherits the loss,
  gradient checks, clipping and metrics. It supplies batch-norm-aware optimizer
  groups, because the CNN's name-based grouping rejects batch-norm parameters.
- **Lifecycle.** A scratch ResNet runner collects, validates, trains, persists,
  reloads and continues bit-identically through the unmodified lifecycle.

Batch norm introduces four behaviours the CNN does not have:

1. **Model state includes running statistics.** They are saved, restored,
   validated and covered by the weight hash and the boundary's model chain.
2. **A failed training step is never recoverable in place.** The train-mode
   forward moves running statistics before the optimizer runs, so
   `ResNetTrainer` invalidates itself on any exception in `step`, not only on
   nonfinite parameters. The runner already reloads the last boundary after a
   failure.
3. **Statistics are bound to the training counters.** On restore, every
   batch-norm update counter must equal the trainer step count.
4. **Inference uses running statistics, training uses batch statistics.** With
   momentum 0.1 the running statistics lag by roughly ten updates. Generation 1
   self-play uses the initial statistics (mean 0, variance 1), which gives
   near-uniform priors and unsaturated values, like the untrained CNN.

### Architecture identity and checkpoint safety

- The model carries its identity in `alphazero_v2_contract`. Because it is never
  the CNN identity, `V2Inference`, `V2Trainer`, `frozen_copy` and `model_state`
  reject a ResNet; the ResNet equivalents reject the CNN, the other variant, a
  subclass, and a model whose label disagrees with its tensors.
- Restoring state requires the exact tensor names, shapes and dtypes of the
  named architecture, finite values, nonnegative variances and counters.
- Experimental boundaries record the identity in their integrity-digested
  config. Relabelling a boundary as another architecture fails restoration.
- The standalone inference artifact stores a per-architecture contract including
  the specification. The loader requires the caller to name the architecture it
  expects; the CNN loader rejects ResNet artifacts and vice versa.

### Consequences to plan around

- **ResNets are scratch-only.** They cannot be forked from the generation-20
  CNN parents, so a comparison needs from-scratch arms, including a from-scratch
  CNN control.
- **This branch has a new execution-source identity**,
  `3523ee04a4b523a915941379ed66871687e6163baa8c363fedddbce2c84da934`, because the
  package gained a module. Strict loading therefore refuses boundaries written
  by the 4D.4F/4D.4G code, and the reverse. All arms of an architecture
  comparison must be produced from one commit.
- The official campaign path still constructs and trains only the CNN. It
  imports the new module solely as part of its source-closure check.

## Validation

| Run | Result |
|---|---|
| Focused ResNet cases (`tests/alphazero_resnet_cases.py`) | 73 passed, 6.0 s |
| Existing experimental contract cases (CNN lifecycle, unchanged file) | 56 passed, 31.6 s |
| ResNet wrapper plus the four existing source-closure checks | 5 passed, 30.4 s |
| Full AlphaZero sweep, `tests/test_alphazero*.py` (after 4D.4G completed) | 610 passed, 2 skipped, 309.7 s |
| Dependency suite: neural MCTS, import isolation, MCTS, MCTS API, engine | 308 passed, 3.5 s |

The sweep equals the Phase 4D.4F result (609 passed, 2 skipped) plus the new
wrapper. The two skips are the existing optional retained-v1-checkpoint checks.

Logs are in `research/alphazero_resnet/results/`. The focused cases cover every
requested item:

| Requirement | Cases |
|---|---|
| Input/output shapes | Batches 1/8/128; seven malformed inputs rejected |
| Legal policy masking | Full column gets exactly zero; a +50 illegal logit leaves legal probabilities unchanged; terminal and train-mode inputs rejected |
| Value range | Bounded and finite with value-head weights scaled by 1e6, in eval and train mode |
| Forward determinism | Bit-identical repeats and restored copies; per-row batch independence; inference leaves the hash unchanged |
| Backpropagation | Finite, nonzero gradient on every parameter; eight steps on one synthetic batch reduce the loss |
| Initialization | Seed determinism, global RNG untouched, batch-norm defaults, weight bounds, architecture-separated seeds |
| Serialization | State and inference-artifact round trips; detached copies; no overwrite |
| Identity and checkpoints | All six ordered cross-architecture pairs rejected for state, freeze, inference and trainer; forged, relabelled and eleven kinds of corrupt state rejected; six boundary tampers rejected |
| Model factory | Registration, default, unknown identities, config round trip |
| Batched inference | `predict_batch` equals single predictions; pool-eight collection in the lifecycle |
| PUCT | Engine and bitboard search, visit accounting, determinism |
| Lifecycle | One generation, strict reload, bit-identical continuation, evaluation hook |
| CNN compatibility | Byte-identical modules, same factory bindings, 325,896 parameters, round trip |

As a check that the tests can fail, eleven small defects were injected
into the implementation one at a time (removing the skip connection, identity
checks, variance validation, trainer invalidation, counter binding, and
others). Every one was caught, and the files were restored byte-for-byte.

**Not done.** No benchmark, training experiment or tournament was run, by
instruction. No independent review was commissioned for this phase.

## Remaining validation requirements

1. Measure inference and training cost; every tradeoff above is an expectation.
2. Measure the batch-norm train/eval gap: loss on the same replay batch in both
   modes at each generation boundary, especially in early generations and with
   few updates per generation.
3. Confirm numerical stability over a realistic number of updates. Only eight
   synthetic steps were run.
4. Confirm exact resume in a fresh process for a ResNet boundary. In-process
   continuation was tested; the fresh-process case was covered only for the CNN.
5. Decide whether a ResNet inference artifact needs the campaign-grade evidence
   contract before any use outside experiments.
6. Obtain an independent review before results from these models are relied on.

## Proposed benchmark phases

**4D.4H-B — cost and fixed-data benchmark (recommended next).** A bounded run
of about one hour, with a prospective declaration, started once the 4D.4G
final analysis is finished.

1. *Fresh-process resume.* Close item 4 above for both ResNets first.
2. *Inference cost.* For all three architectures with seed-initialized weights,
   on one frozen position set: single-thread forward latency at batch 1, 8, 16
   and 32, and `predict_batch` throughput. Interleave architectures and repeat at
   least five times. Report positions per second and the ratio to the CNN.
3. *Collection cost.* Pool-eight collection at 256 simulations on matched game
   identities, 16–32 games per architecture. Report time per neural position, not
   only games per hour, because untrained networks produce different game lengths.
4. *Training cost.* Time per batch-128 update, each architecture.
5. *Fixed-data learning.* Train all three from scratch on one frozen replay
   set, split by game into train and held-out, with identical sampling and
   update counts. Report held-out policy cross-entropy and value MSE against
   updates and against wall time, plus the batch-norm train/eval gap. Evaluate
   the final models on the development solved300 and tactical200 suites declared
   for 4D.4G. This isolates what each architecture can fit from how it
   behaves in self-play. The data source needs an explicit decision: read-only
   use of the existing generation 13–20 replay, or freshly generated games.

Decision rule for continuing: proceed to self-play only for a variant whose
fixed-data held-out metrics beat the CNN's at equal wall time on both seeds.
If neither does, the architecture is not the current bottleneck and the result
is reported as negative.

**4D.4H-C — from-scratch self-play pilot (separate approval).** CNN, ResNet 3×64
and any surviving variant from scratch, seeds 42 and 314159, using whichever
temperature schedule and update dose 4D.4G supports. The primary comparison is
**equal wall-clock budget**, with the cost ratios from 4D.4H-B fixing each
arm's game budget in advance; equal games and updates is the secondary
analysis. Endpoints follow 4D.4G: optimal-action-set policy cross-entropy and
value MSE on solved300, tactical200 gates, and paired opening-family matches
against a common fixed opponent, with thresholds declared before collection.

## Recommendation

Once the Phase 4D.4G final analysis is complete, run **Phase 4D.4H-B**: the
inference, collection and training cost benchmark, then the fixed-data learning
comparison. It is cheap, it converts every expectation in
this report into a measurement, and it yields the cost ratios without which an
equal-wall self-play comparison cannot be designed. Carry **ResNet 3×64 as the
primary candidate**: it matches the CNN's parameters and trunk arithmetic, so
any advantage is attributable to the residual, normalized, deeper trunk rather
than to a larger compute budget. Treat 5×64 as the depth test that must earn
its 1.6× cost. Do not start self-play training with either ResNet until
4D.4H-B reports.

## Requirement status

| Requirement | Status | Evidence |
|---|---|---|
| New worktree and branch from 9325f0a; 4D.4G untouched | PASS | Worktree list; 4D.4G HEAD unchanged; its run reported completion |
| Research of external and original architectures | PASS | Design table; `external_sources.json` |
| ResNet 3×64 and 5×64 implemented | PASS | `resnet.py`; parameter-count tests |
| Input, policy, value contracts preserved | PASS | Shape, masking and range cases |
| PUCT, batching, replay, loss, lifecycle compatibility | PASS | Search, batch, trainer and lifecycle cases |
| Integrated through the 4D.4F model factory | PASS | `experimental_models.py`; factory cases |
| CNN functional, unchanged and default | PASS | Byte-identity case; 56 existing cases |
| Explicit architecture identities | PASS | Identity constants and contracts |
| Incompatible checkpoints cannot load | PASS | Cross-architecture, forged and tampered cases |
| Lightweight synthetic validation while 4D.4G ran | PASS | About four single-core minutes at low priority |
| Parameter counts and specifications recorded | PASS | `results/architectures.json` |
| Full regression sweep | PASS | 610 passed, 2 skipped; run after 4D.4G completed |
| Benchmarks, training, tournaments | NOT RUN | Out of scope by instruction |
