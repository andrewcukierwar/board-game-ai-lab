# Phase 4D.4A — AlphaZero v2 baseline audit

The trained networks learned useful policies and beat their initial networks,
but their raw neural PUCT is not tactically reliable. The next change should
be a behavior-preserving bitboard state backend with direct current-player
encoding, retaining the existing network and search mathematics. It targets
measured copying and terminal-check costs before introducing learning changes.

This report distinguishes archived official development evidence, new
exploratory measurements, engineering inferences and experimental hypotheses.
Detailed machine-readable records and the complete result tables accompany
the report in `research/alphazero_baseline/results/phase4d4a-20261011/`.

## Provenance and preservation

The actual host is an Apple M4 Mac mini: 10 CPU cores, 16 GiB memory,
macOS 26.6.2 arm64, CPython 3.11.17, PyTorch 2.10.0, NumPy 1.26.4. The existing
environment is `/Users/andrewcukierwar/.venvs/board-game-alphazero-m4`.
The default shell Python has no Torch; no packages were installed.

The historical checkout was clean on `phase4d3-m4-runtime`, commit
`1894bc4d8f035cf738df8e3fc0ce234f1ce2c8be`. Its campaign source manifest
declares that same clean commit. The observed local `origin/main` is
`8704142702da273e6571f0ca9df3fd4a793be285`; that exact revision supplies the
optimized classical opponents and bitboard reference. These are pinned local
references, not a claim that a remote branch cannot subsequently advance.

Changes live in a new `phase4d4a-alphazero-baseline` worktree, created from
the research commit inside its ignored `experiment-output` directory.
Main was inspected through a separate detached reference worktree. Nothing
was merged into the historical checkout. No AGENTS.md files were found.
The canonical 112-game benchmark, deployments and other services were untouched.

The official campaign path remains
`experiment-output/phase4d3c-alphazero-v2-official-m4-20261008-110950/`.
Its declaration SHA-256 exactly matches
`c8ab2227d4b2c707b3185da8695a991f0914faede88ecd5f1c7fc84fb017b716`.
All eight requested inference checkpoints exist and their byte hashes match
the original selection/summary records. Checkpoint paths and hashes are in
each run's provenance. All 150 official files were hashed before and after
diagnostic runs; preservation records identify any mismatch. Sealed artifacts
were hashed as bytes only, never used as new evaluation inputs. No research training run,
campaign relaunch or official resume occurred.

Principal profiles use the original campaign settings: CPU float32,
one intra-op and one inter-op thread, deterministic algorithms enabled,
and all five BLAS/vector thread environment settings pinned to one. A
four-thread profile and position assessment are supplemental. Position
metrics under that supplemental runtime reproduced the archived tactical
accuracies and solved value MSEs; it does not establish bitwise equivalence
between different thread settings.

## Audit of the original implementation

`network.py` implements the 325,896-parameter basic CNN, three spatial
convolutions with channels 64/128/128, separate policy and scalar tanh heads,
and no residual blocks, batch normalization or accelerator inference path.
Inference requires CPU float32 and eval mode, converts an engine board through
NumPy into a new tensor, validates input, validates outputs inside the forward
call and again in `predict`, iterates modules to check eval mode, extracts
seven Python scalar logits, then applies a stable float64 legal-only softmax.
These checks are useful contracts but recur for every leaf. Their redundancy
is a later optimization opportunity, not evidence that they should all be removed.

`search.py` uses the historical Python Node and full mutable engine states.
Every expansion predicts once, deep-copies the board for **every** legal
successor, calls the guarded engine transition, and allocates child objects
and dictionaries. It neither reuses the previous tree nor shares transpositions.
Node/parent links form cycles. Each selection computes a score dictionary,
finds its maximum, builds a tie list, and consumes one seeded tie choice.
This work is real but much smaller than board copying and terminal scanning.

The semantics are internally consistent:

- Q belongs to the node's player to move; the parent maximizes
  `-Q(child) + 1.41 * P(child) * sqrt(max(1, N(parent))) / (1 + N(child))`.
- The engine flips the turn even after a winning move. Terminal value is
  therefore normally -1 for that winning move's child. Backup changes sign
  once per edge, increments the leaf and root, and is undiscounted.
- Root expansion is uncounted and its raw value is diagnostic only. Exactly
  256 or 512 simulations traverse root edges. Root-edge visits sum to the budget.
- Self-play noise mixes 25% Dirichlet(alpha=1) over legal columns using an
  independent PCG64 stream. Evaluation has no root noise or tactical guard.
- Action temperature is 1 before zero-based ply eight, then 0 with uniform
  maximum-visit tie sampling. Root noise persists after ply eight.
- Training policies remain normalized raw visits regardless of action
  temperature. Final outcome targets belong to the pre-move actor.

`selfplay.py` collects complete empty-board games serially using one frozen
network for both colors. `generation.py` inserts complete generations into an
eight-generation, 2,048-game replay; performs `ceil(4 * new_positions / 128)`
updates; then uses the updated model for the next collection. `data.py` samples
positions uniformly without replacement within a batch, with replacement
across updates, and reflects board/action/visit targets consistently.
`training.py` uses persistent AdamW, LR 3e-4, weight decay 1e-4 on weights,
equal-weight policy soft cross-entropy plus actor-relative outcome MSE, and
gradient clipping at five. Modes and numerical checks bracket each update.
There is no observed evidence of a value-sign inversion or visit-accounting
defect. Archived trajectories were replayed legally and their winners checked.
This audit does not prove all possible training defects absent.

Existing generation boundaries already support source/runtime-checked exact
exploratory continuation. The official campaign intentionally forbids resume.
The missing improvement is a convenient separately declared exploratory runner
and evaluation workflow, not inventing resume serialization from scratch.

## Measured search performance

The principal profile contains 560 uninstrumented searches: eight checkpoints,
seven fixed development-designated positions, two budgets and five RNG trials.
Framework warm-up, checkpoint loading and explicit pre-trial garbage collection
are outside latency. Search initialization, root prediction/expansion, all
simulations and result construction are inside latency. Action selection is
outside this search benchmark. The frozen engine/search sources were verified
against the campaign manifest before execution.

| Position | 256 simulations, ms | 512 simulations, ms |
|---|---:|---:|
| Empty | 113.01 | 227.24 |
| Early opening | 112.96 | 225.44 |
| Quiet midgame | 99.81 | 196.50 |
| Tactical midgame | 100.24 | 199.91 |
| Late game | 31.14 | 48.27 |
| Immediate win | 18.10 | 31.23 |
| Mandatory block | 85.28 | 179.19 |

These are medians of the eight checkpoint medians, not independent population
estimates. All per-checkpoint trials, ranges, CPU times, visit vectors, terminal
leaves and simulations/second are retained. Empty-board throughput is about
2,250 simulations/second. Late/tactical searches often revisit terminal leaves
and require fewer neural calls, so simulation throughput alone is misleading.
The median process CPU/wall ratio is approximately 1.000.

112 additional instrumented searches reproduced the entire original tree's
boards, values, priors, visits, depth diagnostics and RNG state exactly. The
median instrumented/uninstrumented time ratio was 1.043. Exclusive time shares,
weighted by instrumented wall time over this fixed workload, are:

| Exclusive category | Share |
|---|---:|
| Neural forward, including its contract validation | 35.45% |
| Terminal/winner checks, including transition guards | 25.52% |
| State copying | 15.90% |
| Inference wrapper/softmax/scalar extraction | 8.48% |
| Board encoding and tensor creation | 5.66% |
| PUCT child selection | 3.05% |
| Expansion bookkeeping | 1.61% |
| Legal-move generation | 1.56% |
| Transition work excluding nested terminal checks | 1.50% |
| Backup | 0.22% |
| Other | 1.05% |

Inclusive `predict` time is 58.42% and inclusive expansion time is 83.91%.
Neither is additive with the table: both contain neural/encoding/checking
work. Calling all of `predict` "GPU/NN computation" would misattribute Python
overhead. Conversely, forward itself includes framework dispatch and tensor
validation; 35.45% is not a pure convolution-only number. Separate cProfile
data corroborate deepcopy and engine winner scans as major Python hotspots.

One quiet-midgame 512-simulation allocation probe retained approximately
2.84 MiB of Python-traced allocations. Process lifetime peak RSS was 282.64 MiB,
including earlier searches, loading and later batching allocations; it is not
a search-only incremental memory measurement. Native tensor memory is largely
outside tracemalloc. The four-thread process peak was 224.78 MiB; these values
are not evidence of a controlled memory improvement from threading.

The pinned main bitboard reference matched **2,704** legal successor transitions
from all tactical/solved development positions: boards, turn, winner, terminal
status and legal-column order. Engine copy+drop cost 15.5–17.6 microseconds per
successor versus 0.97–1.08 for immutable bitboard drop. Cached terminal lookup
cost about 0.03 microseconds versus 35.7–41.5 for the engine scan; initial engine
conversion cost 3.35–5.96 microseconds. These are representation microbenchmarks,
not an implemented neural-PUCT speedup. The next phase must measure the full port.

Batch-forward probes show why batching needs its own experiment. With one CPU
thread, batch sizes 1/8/32/64 yielded approximately 6,202/10,094/4,449/5,081
positions/second. With four threads the corresponding figures were
7,348/19,391/10,363/12,921. Batch eight is promising; larger batches were not
monotonically better. These probes repeat identical inputs, exclude encoding,
and measure no game collection. They are not end-to-end self-play speedups.

## Checkpoint strength and neural behavior

Archived official development tests already established learning at 512 raw
simulations, without guards/noise. Each candidate's head-to-head comprised
200 paired-color games, 41 opening families. The Negamax-2 comparison used
40 games and nine families, with the **historical reference** Negamax, not
the new optimized main implementation. The family-bootstrap intervals below
describe those archived, reused development results.

| Seed/gen | Score vs g0, 200 games | Archived 95% interval | Score vs reference N2, 40 games | Archived 95% interval | Win / safe accuracy |
|---|---:|---|---:|---|---|
| 42/10 | 90.75% | 84.76–95.04% | 61.25% | 41.25–78.13% | 98% / 93% |
| 42/15 | 87.00% | 81.99–93.29% | 62.50% | 51.14–75.00% | 90% / 90% |
| 42/20 | 79.25% | 68.93–90.55% | 68.75% | 47.50–84.72% | 88% / 91% |
| 314159/10 | 75.00% | 68.29–81.10% | 58.75% | 36.11–79.17% | 96% / 99% |
| 314159/15 | 86.00% | 78.05–91.74% | 67.50% | 44.44–86.36% | 94% / 94% |
| 314159/20 | 80.50% | 75.00–87.80% | 66.25% | 51.39–77.89% | 92% / 96% |

Every candidate passed the declared arena score and lower-confidence-bound
requirements; all failed the two-percentage-point tactical regression rule.
Generation 0 stayed official champion. Those declarations remain correct,
immutable campaign outcomes; calling a different checkpoint promising does
not promote it retrospectively.

New exploratory position assessments used identical 200 tactical rows and
300 class-balanced solved rows across all eight checkpoints, one declared
search/tie RNG per tactical row, 512 simulations and no noise or guard.
Tactical categories each contain 100 rows representing 50 mirrored families,
balanced by actor. Solved classes each contain 100 rows. These are development
rechecks, not fresh test sets or 200 independent tactical families.

| Seed/gen | PUCT win / safe | NN-only win / safe | Solved value MSE | NN-only solved optimal |
|---|---|---|---:|---:|
| 42/0 | 100% / 100% | 26% / 15% | 0.6742 | 57.0% |
| 42/10 | 98% / 93% | 28% / 42% | 0.5903 | 66.7% |
| 42/15 | 90% / 90% | 35% / 53% | 0.5861 | 72.0% |
| 42/20 | 88% / 91% | 44% / 63% | 0.6118 | 73.3% |
| 314159/0 | 100% / 100% | 25% / 17% | 0.6666 | 59.0% |
| 314159/10 | 96% / 99% | 30% / 45% | 0.5972 | 67.3% |
| 314159/15 | 94% / 94% | 30% / 58% | 0.5633 | 67.3% |
| 314159/20 | 92% / 96% | 36% / 69% | 0.5949 | 69.0% |

Policy entropy on these tactical states fell from about 1.516 nats initially
to 1.126/1.142 at generation 20. Neural policy gains are real descriptive
evidence of learning, while immediate-win prediction remains weak. More peaked
priors can leave winning actions unvisited: the official diagnostics recorded
five such winning actions for 42/g20 and two for 314159/g20. This is a finite-
budget raw-PUCT weakness; no minimum edge-visit or root proof rule guarantees
these moves receive search. Exact tactical proof can improve inference without
improving the network's predictions.

Solved MSE compares a learned **self-play expected outcome** against the
perfect-play W/D/L oracle. It is a useful diagnostic, not on-policy calibration
or a proof that the value head represents perfect play. Both seeds' MSE improve
then worsen by generation 20, despite falling training MSE. No monotonic
playing-strength improvement or statistically established best seed follows.

The neural move distributions also change substantially. On the paired
tactical rows, 42/g0's argmax selects column6 on 164/200 rows and 314159/g0
selects column0 on 129/200. Trained models' choices spread across physical
columns: at g20, 42 chooses columns0–6 with counts20/31/38/19/42/28/22;
314159 chooses25/24/46/18/43/22/22. These are state-conditioned diagnostic
distributions, not self-play opening frequencies. Initial values have the
correct positive sign on every solved win and the wrong sign on every solved
loss; 314159/g15 reaches 72% win-sign and 84% loss-sign accuracy. This separates
learning useful value discrimination from merely a biased initial output.

The new main-opponent ladder, its paired X/O scores, game counts, full move logs,
configuration hashes and descriptive uncertainty appear in `TABLES.md` and
`summary.json`. It uses the empty board plus one development prefix, four
games per cell, 512 raw neural simulations; Classic and Tactical F1 use a
maximum of 1,000 simulations with their existing root guards (F1 may stop on
a proof); Negamax depths are 2/4/6/8 with no time cutoff and deterministic
center-order ties. The raw ladder's initial metadata mislabeled F1's
`root_guard` field as false; the pinned executed code has the guard enabled.
`metadata_errata.json` records this correction; raw results are retained.
These are smoke
measurements with two families, insufficient for stable strength rankings.
Degenerate bootstrap intervals at all wins/losses are not certainty. They
must not be pooled with the historical opponent or sealed final evaluations.

The designated ladder completed **216 games in 704.91 seconds**. Each entry
below is a score from four games, two as X and two as O:

| Seed/gen | Main N2 | Main N4 | Main N6 | Main N8 | Classic 1000 | Tactical F1 1000 |
|---|---:|---:|---:|---:|---:|---:|
| 42/0 | 25% | 25% | 0% | 0% | 0% | 0% |
| 42/10 | 100% | 50% | 0% | 37.5% | 50% | 0% |
| 42/15 | 75% | 100% | 50% | 50% | 75% | 25% |
| 42/20 | 75% | 100% | 62.5% | 37.5% | 87.5% | 25% |
| 314159/0 | 0% | 0% | 0% | 0% | 0% | 0% |
| 314159/10 | 100% | 75% | 50% | 75% | 50% | 62.5% |
| 314159/15 | 87.5% | 100% | 50% | 50% | 50% | 75% |
| 314159/20 | 75% | 100% | 50% | 100% | 62.5% | 75% |

Trained checkpoints perform better than g0 on this selected small ladder,
but non-monotonic depth results and seed differences cannot establish a
reliable ranking. Against F1, 42/g15 made one unsafe response in 22 one-reply
response opportunities and 314159/g20 missed one immediate win in four
opportunities. Actual-game errors complement the diagnostic suite; low counts
do not establish safety. Full per-color W/D/L, family intervals and action
opportunity denominators are retained, rather than treating 100% cells as proof.

An initial wider exploratory ladder was stopped after two complete ten-game
cells to reduce workload; its full position assessment is retained. Interrupted
games are not scored. The subsequent bounded ladder is the designated new
comparison. No performance significance or cross-project Elo claim is made.

## Self-play and training bottlenecks

All 10,240 archived games, 314,518 positions, were reconstructed independently.
Generation summaries identify collection as **98.43%** of collection+optimization
time for both seeds: about 29,825 seconds collecting versus 475 optimizing.
This is the training-stage split, not 98% of the entire 14.96-hour campaign,
which also included extensive development and final evaluation.

| Seed/gen | Mean length | Unique positions | Position duplicates | Duplicates after ply 8 | Unique 8-ply openings / 256 | Unique complete trajectories / 256 |
|---|---:|---:|---:|---:|---:|---:|
| 42/1 | 24.32 | 5,344 | 14.2% | 0.1% | 256 | 256 |
| 42/10 | 32.01 | 5,793 | 29.3% | 16.4% | 214 | 247 |
| 42/15 | 33.21 | 4,868 | 42.7% | 32.5% | 169 | 223 |
| 42/20 | 33.57 | 3,718 | 56.7% | 48.2% | 133 | 187 |
| 314159/1 | 25.74 | 5,698 | 13.5% | 0.0% | 256 | 256 |
| 314159/10 | 31.26 | 5,761 | 28.0% | 15.5% | 224 | 246 |
| 314159/15 | 30.91 | 4,473 | 43.5% | 31.0% | 167 | 230 |
| 314159/20 | 34.49 | 4,189 | 52.6% | 43.7% | 151 | 207 |

At generation 20 the mirror-family counts are only 3,530/4,004. The most common
eight-ply opening occurs 20/26 times; the most repeated full trajectory 7/8
times. This supports narrowing diversity, not complete collapse into one game.
Stage-conditioned repeats remain substantial after excluding the first eight
plies, so the trend is not explained just by the common empty board.

The retained generation-20 boundaries store visits/temperatures for generations
13–20. Reading them verified the 256-visit budget, exact temperature schedule,
and maximum-visit actions after ply eight. At g20, after-eight visit-target
entropy is 0.653/0.720 nats and mean maximum-action mass 0.766/0.735. Targets
still contain exploration, but execution concentrates them into greedy moves.
Older game JSONL contains moves/winners only; it cannot establish older
per-ply policy entropies independently of the generation summaries.

Tactical states exist in self-play. At 42/g20 there are 211 immediate-win
opportunities and 1,765 states with a safe response and an unsafe alternative;
22 immediate wins are missed and 93 unsafe moves are played. Corresponding
314159/g20 counts are 211/1,610 opportunities and 15/53 mistakes. The win-state
fraction is only about 2.4%; tactical curricula/stratification could help, but
absence of tactics is not the explanation. Across the campaign, official
one-reply-proof/outcome contradictions total 26; correct completed-game targets
can still disagree with perfect-play values because the self-play policy errs.

Training policy loss falls 1.933→1.021 and 1.934→1.082; value training MSE
falls 0.807→0.407 and 0.770→0.370. Those losses alone cannot justify a better
checkpoint, larger model or unchanged exploration. At g20 the replay contains
68,856/64,514 positions, while 34,432/35,328 samples are drawn per generation.
Four appearances per new position means only roughly half an appearance per
stored position per steady-state generation, not four epochs of the replay.
An eight-generation lifetime gives about four appearances in steady state;
the earliest games get more and the final generation gets less before stopping.
Actual age-sampling counts are retained for inspection.

A training-frequency ablation is cheap in wall time and justified by sparse
replay exposure. A temperature/replay ablation is justified by repeated states
and increasing tactical errors. A residual model is plausible external practice,
but current data do not identify capacity as the primary cause. None of these
hypotheses was tested by starting another training run in this phase.

## External methodology review

Public performance numbers below are author reports, not independently
replicated measurements, and opponents/depth definitions do not transfer.
Pinned public source commits and fetched-file SHA-256s accompany the comparison.
No external code was incorporated into the project.

| Project | Verified or explicitly documented methodology | Transferable lesson / caveat |
|---|---|---|
| [AlphaZero.jl tutorial](https://jonathan-laurent.github.io/AlphaZero.jl/stable/tutorial/connect_four/) | Documented 5×128 ResNet, ~1.6M parameters; 600 simulations, c=2; root eps=.25/alpha=1; temperature 1 through move 20 then .3 at 30; 5,000 games/iteration; replay 400k→1M; batched GPU workers; position averaging with logarithmic weights; Adam; arena/NN-only/solver benchmarks; saved iteration resume. | Treat search-only and network-only strength separately; track redundancy and exact move mistakes. GPU throughput and published solver curves are not M4 measurements. Tutorial config uses low nonzero evaluation temperature/noise, unlike our raw deterministic-action evaluation. |
| [eugeneyp](https://github.com/eugeneyp/connect4-alphazero) | Pinned full config: 5×128 residual model, 400 simulations, c=2, noise .25/1, temperature 1→.3 after 20; 2,000 games, replay 500k, ten training epochs, batch 1,024, arena 64 games at 200 sims, threshold .55. Code batches independent games, config batch32; coach saves weights/optimizer. | Smooth later exploration and independent-tree batching are candidates. README depth-11/Kaggle claims are not reproduced; 20-game scores cannot certify superiority. We have not verified exact RNG/replay resume equivalence. |
| [steven0226 model card](https://huggingface.co/steven0226/alphazero-connect4), [source](https://github.com/kuotunyu/alphazero-connect4) | Card describes 6×96 ResNet, 3 input planes. Pinned source config: 160 sims, c=1.5, noise .25/1, ten temperature plies, 128 parallel games, replay150k, 64 updates of batch256, AdamW; 40-game gate >.55. Code saves model/best/optimizer/replay/history with previous-checkpoint fallback. | Separate raw-policy and searched ladders, automate bounded experiments. Card's 19,200 games/7.9 h and Elo1,626 are author claims with custom anchors, not cross-project Elo. Saved state does not itself prove exact stochastic resume. |
| [suharshkumar](https://github.com/suharshkumar/connect4-selfplay-rl) | Source implements bitboards, 5×64 residual network and lockstep independent-tree leaf batches; c=1.5, noise .25/1, ten temperature plies; ten-iteration replay and two replay epochs. Defaults are 48 games/60 sims, while README describes a 64-game/80-sim run. Saves weights/history, without a complete resume payload in reviewed train code. | Architectural comparison can stay near our parameter count. The published 6.5× batching and 78.8% ladder score were not replicated. Its documented past sign bug reinforces absolute outcome-sign tests, which our v2 already has. |
| [plkmo](https://github.com/plkmo/AlphaZero_Connect4) | README describes 19×128 residual blocks. Reviewed source uses 777 searches, deepcopy/lazy children, multiprocessing self-play, root noise eps=.25/alpha=192, temperature about1.1 for first11 plies then .1, and optimizer/scheduler checkpoints. Arena randomizes seats. | Useful historical structure, not a safe parameter template: terminal leaves back up network estimates and root noise is applied by the same expansion path used in arena. README's 83/100 internal wins do not establish external strength. |

The strongest shared engineering evidence is efficient states plus batching
across independent games. The most consequential learning differences are
continued action exploration, larger/diverse replay and training reuse, and
measuring raw-policy learning separately. Residual blocks are common but their
causal benefit here remains untested. Noise .25/alpha1 already matches several
references; blindly changing it is not the first priority. Larger simulation
budgets are expensive and can even exacerbate confidence effects; tune at fixed
weights before paying for new training.

## Prioritized roadmap and focused experiments

Ranks combine impact, implementation effort, correctness risk and the evidence
available here. "High" expected impact is a hypothesis unless a result above
directly measures it. The three objectives are explicitly separated.

| Rank | Candidate | Objective | Expected impact | Difficulty / correctness risk | Evidence and first test |
|---:|---|---|---|---|---|
| 1 | Bitboard PUCT states + direct encoding | Faster computation | High | Medium / medium | 41% copying+checking; successor parity and microbenchmarks. Full identical-search port; target ≥1.5×. |
| 2 | Revised development checkpoint selection | Preserve/select learned strength | High selection impact | Low–medium / low code, high research-design risk | All learned candidates rejected despite arena gains. Future declared Pareto report of strength, NN learning and tactics; keep historical outcomes immutable. |
| 3 | Improved exploration and replay | Stronger learning | Medium–high | Medium / medium | After-eight duplicate rates and sparse reuse. A/B one change: temperature1 through ply16 versus8, then independent 4→8 sample-dose ablation. |
| 4 | Optional root tactical safety | Stronger inference decisions | High on one-reply tactics | Low–medium / medium | Raw-PUCT missed wins/blocks. Fixed-checkpoint guarded/raw paired tests; guarantee immediate wins and avoid one-reply loss when a safe action exists; log interventions. No neural-learning claim. |
| 5 | Batched neural inference | Faster computation | Medium–high after state port | Medium / medium | Batch8 forward throughput improves; larger batches can worsen CPU throughput. Lockstep eight independent trees, per-game RNG streams; target ≥1.3× complete collection throughput with matched budgets. |
| 6 | Compact neural nodes | Faster computation / lower memory | Medium | Low–medium / low–medium | Dicts, parent cycles and traced tree memory. Slots first, then consider arrays; exact tree statistics/RNG; target ≥20% traced-tree memory reduction with no latency regression. |
| 7 | Resumable exploratory runner + automated evaluations | Research efficiency | High workflow value | Medium / low–medium | Existing boundary support already exists. New output/declaration/lineage, cooperative stopping and complete/abandoned records; forced-interruption smoke tests. Never resume the official campaign. |
| 8 | Parallel self-play | Faster collection | Medium, hardware dependent | Medium–high / medium | Serial collection dominates. Benchmark 1/2/4 workers after batching, avoid nested CPU oversubscription; target ≥1.3× games/hour and complete deterministic per-game provenance. |
| 9 | Tactical proof propagation | Stronger inference / faster search | Potentially high | High / high | F1 offers reusable proof ideas, not interchangeable backup rewards. Verify W/D/L recurrence, draws and losing-root handling against exact late-game oracle; fixed-time strength tests. |
| 10 | Residual architecture | Stronger learning | Unknown–medium | Medium / low implementation, high experiment cost | External practice, no causal diagnosis here. Compare similar parameter count, fixed data then equal-wall mini self-play pilots; require improved policy/solved-value/ladder metrics across both seeds. |

Follow-up selection work should retain at least 42/g10, 314159/g10 and
314159/g15 for broader comparison. Current four-game cells cannot identify a
single best deployment model. A useful larger **proposed, not run** evaluation
would declare 20 development opening families, paired colors and mirrors,
compare selected checkpoints against N4/N6/F1 at both fixed simulations and
fixed wall time, and bootstrap by family. Fresh held-out data should be created
and sealed only after the development protocol is decided. The existing sealed
campaign data and canonical benchmark are not a fresh test set.

Exploration acceptance should require lower stage-conditioned repeat rates,
better held-out policy learning/tactical accuracy and no consistent paired-
ladder regression, under equal wall budget. Training-dose acceptance should
require better development neural metrics after equal new-data collection,
with explicit increased optimization cost; training loss alone is insufficient.
Residual acceptance must compare equal parameters and equal wall budget, not
reward a larger network for spending more compute. Root safety is evaluated
as a separate inference variant with the learned model frozen.

## Exact Phase 4D.4B scope

Implement one optional **bitboard state/encoding backend for the existing raw
v2 PUCT**, adapting main's immutable `BitboardState` and cached outcome. Convert
the engine board once at the root. Encode bits directly into the exact existing
CPU float32 `(1,1,6,7)` own=+1/opponent=-1/top-row-first tensor. Preserve center-
ordered legal actions, scalar arithmetic order, priors, c=1.41, root noise,
temperature, full child expansion, counted visits, terminal values and tie RNG.
Keep the old backend for differential testing. Retain existing Node objects in
this first experiment; compact nodes, guards, batching, tree reuse and new
networks belong in separate changes.

Acceptance: (1) exact encoded tensors and engine/bitboard state parity over
at least 10,000 legal transitions including wins, draws and full columns;
(2) exact visit vectors, raw outputs, values, selected actions and RNG states
for all eight checkpoints, both budgets, seven fixed positions and three RNG
trials, plus noisy self-play probes and the development tactical suite;
(3) at least **1.5× median full-search speedup** on empty/opening/quiet positions
at 256/512, on the same M4 deterministic one-thread runtime, with no position's
p95 latency worsening more than 5% and no increased peak traced allocation;
(4) frozen campaign hashes still match, relevant tests pass, and raw tactical
accuracy/paired games do not change when outputs are equivalent.

Copying+terminal work alone is about 41% of measured time: completely eliminating
that fraction would imply only about 1.71× by Amdahl's law, before encoding/
transition gains and residual costs. This is an idealized ceiling for that
fraction, not a predicted measured speedup. A 1.5× acceptance target is meaningful
but must be tested. If it fails, report the remaining hotspots before adding
another optimization. No training, deployment, promotion or official-evidence
change is part of Phase 4D.4B's proposed first experiment.

## Validation and remaining questions

The relevant v2/search/evaluation/campaign/launch-control/import-isolation suite
passed 374 tests, with two optional retained-v1-checkpoint tests skipped in the
isolated worktree. After final metadata and duration-validation changes, all
16 standalone-tool tests passed. The source tree under `games/connect4` remains
byte-identical to the research commit; both original and main-reference
worktrees remain clean. Tool snapshots, runtime identities, source and checkpoint
hashes, complete and interrupted statuses, and per-run preservation inventories
are packaged with the machine-readable results. The two failed archive-extractor
attempts were corrected and covered by a realistic schema regression test;
their fresh output directories remain recorded as incomplete in the run registry.

Unresolved questions are the best trained checkpoint under a broader matched
opening distribution; reliable strength against deep Negamax and F1; trained
on-policy value calibration beyond the oracle diagnostic; full-search gains
from the bitboard port and collection gains from batching; and the causal effects
of exploration, replay dose and residual architecture. Answer these through the
short, separately declared experiments above. No additional training campaign,
promotion, merge or deployment was performed for this audit.
